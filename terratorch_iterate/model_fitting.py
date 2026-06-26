"""
This module contains all the logic for fitting models
"""

import abc
import copy
import dataclasses
import importlib
import os
import shutil
import types
import uuid
import warnings
from abc import abstractmethod
from functools import wraps
from typing import Callable
import pandas as pd
import lightning.pytorch as pl
import mlflow
import optuna
from lightning import Callback, Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    Timer,
)
from lightning.pytorch.loggers.mlflow import MLFlowLogger

# from ray.air.integrations.mlflow import
from optuna.integration import PyTorchLightningPruningCallback
from ray import tune
from ray.air import CheckpointConfig, RunConfig
from ray.train._internal.storage import StorageContext
from ray.tune.experiment import Trial


from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import FIFOScheduler, TrialScheduler
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search import SearchAlgorithm, Searcher
from ray.tune.search.bohb import TuneBOHB
from terratorch.tasks import PixelwiseRegressionTask, SemanticSegmentationTask
from torchgeo.datamodules import BaseDataModule
from torchgeo.trainers import BaseTask

from terratorch_iterate.iterate_types import (
    ParameterBounds,
    ParameterTypeEnum,
    TrainingSpec,
    optimization_space_type,
    recursive_merge,
    valid_task_types,
)


from terratorch_iterate.utils import get_logger

LOGGER = get_logger()


os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = (
    "1"  # disable tune loggers, will add csv and json manually. If this is not here, it will log to tensorboard automatically
)

SEED = 42


class ParameterPicker(abc.ABC):
    @abstractmethod
    def pick_categorical(self, variable, choices):
        pass

    @abstractmethod
    def pick_int(self, variable, low, high):
        pass

    @abstractmethod
    def pick_float(self, variable, low, high, log=False):
        pass


class OptunaParameterPicker(ParameterPicker):
    def __init__(self, trial: optuna.Trial):
        super().__init__()
        self.trial = trial

    def pick_categorical(self, variable, choices):
        return self.trial.suggest_categorical(variable, choices)

    def pick_int(self, variable, low, high):
        return self.trial.suggest_int(variable, low, high)

    def pick_float(self, variable, low, high, log=False):
        return self.trial.suggest_float(variable, low, high, log=log)


class RayTuneParameterPicker(ParameterPicker):
    def __init__(self):
        super().__init__()

    def pick_categorical(self, variable, choices):
        return tune.choice(choices)

    def pick_int(self, variable, low, high):
        return tune.quniform(low, high, 1)

    def pick_float(self, variable, low, high, log=False):
        if log:
            return tune.loguniform(low, high)
        return tune.uniform(low, high)


class _TuneReportCallback(TuneReportCheckpointCallback, pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def inject_hparams(training_spec: TrainingSpec, config: dict):
    # treat batch size specially
    config_without_batch_size = copy.deepcopy(config)
    assert isinstance(config_without_batch_size, dict), (
        f"Error! Unexpected config type: {config_without_batch_size}"
    )
    batch_size: int | None = config_without_batch_size.pop("batch_size", None)  # type: ignore
    datamodule_with_generated_hparams = copy.deepcopy(training_spec.task.datamodule)
    if batch_size:
        datamodule_with_generated_hparams.batch_size = batch_size

    terratorch_task_with_generated_hparams = copy.deepcopy(
        training_spec.task.terratorch_task
    )
    if terratorch_task_with_generated_hparams is None:
        terratorch_task_with_generated_hparams = {}

    recursive_merge(terratorch_task_with_generated_hparams, config_without_batch_size)
    task_with_generated_hparams = dataclasses.replace(
        training_spec.task,
        terratorch_task=terratorch_task_with_generated_hparams,
        datamodule=datamodule_with_generated_hparams,
    )
    training_spec_with_generated_hparams = dataclasses.replace(
        training_spec, task=task_with_generated_hparams
    )
    return training_spec_with_generated_hparams


def get_default_callbacks(
    early_stop_patience: int | None, max_run_duration: str | None
) -> list[Callback]:
    default_callbacks: list[Callback] = [
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if early_stop_patience is not None:
        default_callbacks.append(
            EarlyStopping("val/loss", patience=early_stop_patience)
        )
    if max_run_duration is not None:
        default_callbacks.append(Timer(duration=max_run_duration))
    return default_callbacks


def generate_parameters(
    parameter_picker: ParameterPicker,
    current_hparams: dict,
    hparam_space: dict,
    ignore_keys: set[str] | None = None,
    dictionary_position: list[str] | None = None,
):
    if ignore_keys is None:
        ignore_keys = set()
    if dictionary_position is None:
        dictionary_position = []
    _generate_parameters(
        parameter_picker,
        current_hparams,
        hparam_space,
        ignore_keys,
        dictionary_position,
    )


def _generate_parameters(
    parameter_picker: ParameterPicker,
    current_hparams: dict,
    hparam_space: dict,
    ignore_keys: set[str],
    dictionary_position: list[str],
):
    for parameter, space in hparam_space.items():
        if parameter in ignore_keys:
            continue
        # if its a dictionary, continue to recurse
        if isinstance(space, dict):
            if parameter not in current_hparams:
                current_hparams[parameter] = {}
            dictionary_position.append(parameter)
            _generate_parameters(
                parameter_picker,
                current_hparams[parameter],
                hparam_space[parameter],
                ignore_keys,
                dictionary_position,
            )
            dictionary_position.pop()
        # if not, get a value from the parameter_picker and insert it with the name prepended by the dictionary position
        # this is important so that the full path of the parameter is used
        # this will avoid confusion between parameters with the same name but from different components
        else:
            full_parameter_name = ".".join(dictionary_position + [parameter])
            if isinstance(space, list):
                suggestion = parameter_picker.pick_categorical(
                    full_parameter_name, space
                )
                current_hparams[parameter] = suggestion
            elif isinstance(space, ParameterBounds):
                match space.type:
                    case ParameterTypeEnum.integer:
                        current_hparams[parameter] = parameter_picker.pick_int(
                            full_parameter_name,
                            int(space.min),
                            int(space.max),
                        )
                    case ParameterTypeEnum.real:
                        current_hparams[parameter] = parameter_picker.pick_float(
                            full_parameter_name, space.min, space.max, log=space.log
                        )
                    case _:
                        raise Exception(
                            f"Type {space.type} not recognized. Suggest one of {[e.value for e in ParameterTypeEnum]}"
                        )
            else:
                raise Exception(
                    "Leaves of optimization space must be lists or ParameterBounds"
                )


"""
single node - optuna
"""


def launch_training(
    trainer: Trainer,
    task: BaseTask,
    datamodule: BaseDataModule,
    run_name: str,
    experiment_name: str,
    metric: str,
    storage_uri: str,
    parent_run_id: str,
    direction: str,
    test_models: bool,
    delete_models_after_testing: bool,
) -> float:
    LOGGER.info(
        f"launch_training {trainer=} {task=} {datamodule=} {run_name=} {experiment_name=} {metric=} {storage_uri=} {direction=}"
    )
    with mlflow.start_run(run_name=run_name, nested=True) as run:
        mlflow.set_tag("mlflow.parentRunId", parent_run_id)
        # explicitly log batch_size. Since it is not a model param, it will not be logged
        mlflow.log_param("batch_size", datamodule.batch_size)

        trainer.logger = MLFlowLogger(
            experiment_name=experiment_name,
            run_id=run.info.run_id,
            save_dir=storage_uri,
            log_model=not delete_models_after_testing,
        )
        trainer.fit(task, datamodule=datamodule)
        if test_models:
            trainer.test(ckpt_path="best", datamodule=datamodule)
        if delete_models_after_testing:
            # delete the checkpoints folder in the run
            ckpts_folder = os.path.join(
                trainer.logger.save_dir,
                str(trainer.logger.name),
                trainer.logger.version,
                "checkpoints",
            )
            shutil.rmtree(ckpts_folder)

        client = mlflow.tracking.MlflowClient(
            tracking_uri=storage_uri,
        )

        if not metric.startswith("val"):
            raise Exception(
                f"Metric {metric} does not start with `val`. Please choose a validation metric"
            )
        for_pd_collect = []
        val_metrics_names = []
        for metric_name in client.get_run(run.info.run_id).data.metrics:
            if metric_name.startswith("val"):
                val_metrics_names.append(metric_name)
                val_metric_history = client.get_metric_history(
                    run.info.run_id, metric_name
                )
                pd_convertible_metric_history = [
                    {
                        "metric_name": mm.key,
                        "step": mm.step,
                        "value": mm.value,
                    }
                    for mm in val_metric_history
                ]
                for_pd_collect += pd_convertible_metric_history
        df_val_metrics = pd.DataFrame.from_records(for_pd_collect)
        df_val_metrics = df_val_metrics.set_index(
            ["metric_name", "step"], verify_integrity=True
        )
        series_val_metrics = df_val_metrics["value"]
        assert metric in series_val_metrics, (
            f"Error! {metric} is not in {series_val_metrics}"
        )
        if direction == "max":
            best_step = series_val_metrics[metric].idxmax()
        elif direction == "min":
            best_step = series_val_metrics[metric].idxmin()
        else:
            raise Exception(
                f"Error! Direction must be either `max` or `min` but got {direction}"
            )

        for val_metric_name in val_metrics_names:
            mlflow.log_metric(
                f"best_step_{val_metric_name}",
                series_val_metrics[(val_metric_name, best_step)],
            )

        return series_val_metrics[(metric, best_step)]


def fit_model(
    training_spec: TrainingSpec,
    lightning_task_class: valid_task_types,
    run_name: str,
    experiment_name: str,
    storage_uri: str,
    parent_run_id: str,
    trial: optuna.Trial | None = None,
    save_models: bool = False,
    test_models: bool = False,
) -> tuple[float, str]:
    pl.seed_everything(SEED, workers=True)
    training_spec_copy = copy.deepcopy(training_spec)
    task = training_spec_copy.task

    if lightning_task_class in [
        SemanticSegmentationTask,
        PixelwiseRegressionTask,
    ]:
        task.terratorch_task["plot_on_val"] = False
    assert isinstance(task.terratorch_task, dict), (
        f"Error! Invalid type: {task.terratorch_task}"
    )

    lightning_task = lightning_task_class(**task.terratorch_task)

    if len(training_spec.trainer_args.get("callbacks", [])) > 0:
        warnings.warn(
            "Callbacks passed to trainer. Make sure these are stateless, as they will not be reinitialized for each task!"
        )
    default_callbacks: list[Callback] = get_default_callbacks(
        task.early_stop_patience, task.max_run_duration
    )

    if task.early_prune and trial is not None:
        default_callbacks.append(
            PyTorchLightningPruningCallback(trial, monitor="val/loss")
        )

    delete_models_after_testing = False
    if test_models and not save_models:
        # we need to save the models during training to be able to test but can be deleted afterwards
        save_models = True
        delete_models_after_testing = True

    if save_models:
        default_callbacks.append(
            ModelCheckpoint(monitor=task.metric, mode=task.direction)
        )
    if "enable_checkpointing" in training_spec_copy.trainer_args:
        warnings.warn(
            f"enable_checkpointing found. Will be overwritten to the value of save_models {save_models}"
        )
    training_spec_copy.trainer_args["enable_checkpointing"] = save_models
    training_spec_copy.trainer_args["enable_progress_bar"] = (
        training_spec_copy.trainer_args.get("enable_progress_bar", True)
    )
    # get callbacks (set to empty list if none defined) and extend with default ones
    training_spec_copy.trainer_args.setdefault("callbacks", []).extend(
        default_callbacks
    )  # type: ignore

    trainer = Trainer(**training_spec_copy.trainer_args)

    return (
        launch_training(
            trainer,
            lightning_task,
            task.datamodule,
            run_name,
            experiment_name,
            task.metric,
            storage_uri,
            parent_run_id,
            task.direction,
            test_models=test_models,
            delete_models_after_testing=delete_models_after_testing,
        ),
        task.metric,
    )


def fit_model_with_hparams(
    training_spec: TrainingSpec,
    lightning_task_class: valid_task_types,
    run_name: str,
    experiment_name: str,
    hparam_space: optimization_space_type,
    storage_uri: str,
    parent_run_id: str,
    save_models: bool,
    test_models: bool,
    trial: optuna.Trial,
) -> float:
    """
    Generate parameters using the optuna trial from the given parameters.
    Then inject these into the given task.
    It is important to make sure to not overwrite the task passed in the arguments, or these updates may affect
    subsequent trials.
    """
    current_hparams: dict[str, int | float | str | bool] = {}
    task = training_spec.task
    generate_parameters(
        OptunaParameterPicker(trial),
        current_hparams,
        hparam_space,
        ignore_keys=task.optimization_except,
    )

    training_spec_with_generated_hparams = inject_hparams(
        training_spec, current_hparams
    )
    run_name = f"{run_name}_{trial.number}"
    return fit_model(
        training_spec=training_spec_with_generated_hparams,
        lightning_task_class=lightning_task_class,
        run_name=run_name,
        experiment_name=experiment_name,
        storage_uri=storage_uri,
        parent_run_id=parent_run_id,
        trial=trial,
        save_models=save_models,
        test_models=test_models,
    )[0]  # return only the metric value for optuna


"""
multi node - ray
"""


def ray_tune_model(
    training_spec: TrainingSpec,
    lightning_task_class: valid_task_types,
    hparam_space: optimization_space_type,
    storage_uri: str,
    ray_storage_path: str,
    experiment_name: str,
    save_models: bool,
    num_trials: int,
    backbone_import: str | None = None,
    searcher: Searcher | SearchAlgorithm | None = None,
) -> tune.ResultGrid:
    if not searcher:
        raise ValueError("searcher must be specified")
    trainable = tune.with_parameters(
        ray_fit_model,
        training_spec=training_spec,
        lightning_task_class=lightning_task_class,
        storage_uri=storage_uri,
        experiment_name=experiment_name,
        parent_run_id=mlflow.active_run().info.run_id,
        save_models=save_models,
        backbone_import=backbone_import,
    )

    current_hparams: dict[str, int | float | str | bool] = {}
    task = training_spec.task
    generate_parameters(
        RayTuneParameterPicker(),
        current_hparams,
        hparam_space,
        ignore_keys=task.optimization_except,
    )

    # Early stopping
    # It is unclear if this is working properly when checkpoints are disabled
    if task.early_prune:
        search_alg: Searcher | SearchAlgorithm = TuneBOHB()
        scheduler: TrialScheduler = HyperBandForBOHB(
            time_attr="training_iteration",
            max_t=training_spec.trainer_args["max_epochs"],
            reduction_factor=2,
            stop_last_trials=False,
        )
        if not save_models:
            raise RuntimeWarning(
                "It is unclear if using `early_prune=True` with `save_models=False` produces correct results."
            )
    else:
        scheduler = FIFOScheduler()
        search_alg = searcher

    # monkey patch scheduler to add trial storage dir
    def decorate_to_add_trial_info(fn: Callable):
        old_fn = fn

        @wraps(fn)
        def new_func(self, tune_controller, trial: Trial):
            trial.config["trial_storage"] = trial.storage
            return old_fn(tune_controller, trial)

        return new_func

    scheduler.on_trial_add = types.MethodType(
        decorate_to_add_trial_info(scheduler.on_trial_add), scheduler
    )

    # for ddp if required in the future
    # scaling_config = ScalingConfig(
    #     use_gpu=True,
    #     num_workers=1,
    #     resources_per_worker={"CPU": 4, "GPU": 1},
    #     trainer_resources={"CPU": 1, "GPU": 0},
    # )
    # ray_trainer = TorchTrainer(
    #     trainable,
    #     scaling_config=scaling_config,
    # )

    trainable_with_resources = tune.with_resources(
        trainable, resources={"cpu": 8, "gpu": 1}
    )

    storage_path = os.path.join(ray_storage_path, experiment_name)
    tuner = tune.Tuner(
        trainable_with_resources,
        tune_config=tune.TuneConfig(
            metric=task.metric,
            mode=task.direction,
            num_samples=num_trials,
            search_alg=search_alg,
            scheduler=scheduler,
            reuse_actors=False,
        ),
        run_config=RunConfig(
            name=mlflow.active_run().info.run_name,
            storage_path=storage_path,
            callbacks=[
                tune.logger.CSVLoggerCallback(),
                tune.logger.JsonLoggerCallback(),
                # RayLogArtifactsMlFlowCallback(),
            ],
            checkpoint_config=(
                CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_score_attribute=task.metric,
                    checkpoint_score_order=task.direction,
                )
                if save_models
                else None
            ),
            # stop={"training_iteration": training_spec.trainer_args["max_epochs"]},
        ),
        param_space=current_hparams,
    )
    results = tuner.fit()
    return results


def _generate_random_name(task_name: str):
    # needed since the random names from mlflow are affected by the seed
    # so they are always the same
    return f"{task_name}_{uuid.uuid4().hex[:8]}"


def ray_fit_model(
    config: dict,
    training_spec: TrainingSpec,
    lightning_task_class: valid_task_types,
    storage_uri: str,
    experiment_name: str,
    parent_run_id: str,
    save_models: bool = True,
    backbone_import: str | None = None,
) -> None:
    if backbone_import:
        importlib.import_module(backbone_import)
    print(config)
    pl.seed_everything(SEED, workers=True)
    tune.utils.wait_for_gpu(
        target_util=0.07, delay_s=10, retry=50
    )  # sometimes process needs some time to release GPU

    trial_storage: StorageContext = config.pop("trial_storage", None)

    training_spec_with_generated_hparams = inject_hparams(training_spec, config)
    task = training_spec_with_generated_hparams.task

    if lightning_task_class in [
        SemanticSegmentationTask,
        PixelwiseRegressionTask,
    ]:
        task.terratorch_task["plot_on_val"] = False
    lightning_task = lightning_task_class(**task.terratorch_task)

    if len(training_spec.trainer_args.get("callbacks", [])) > 0:
        warnings.warn(
            "Callbacks passed to trainer. Make sure these are stateless, as they will not be reinitialized for each task!"
        )

    default_callbacks: list[Callback] = get_default_callbacks(
        task.early_stop_patience, task.max_run_duration
    )
    default_callbacks.append(
        _TuneReportCallback(metrics=[task.metric], save_checkpoints=save_models)
    )

    if "enable_checkpointing" in training_spec_with_generated_hparams.trainer_args:
        warnings.warn(
            "enable_checkpointing found. Will be overwritten to False as ray will be responsible for saving models."
        )
    training_spec_with_generated_hparams.trainer_args["enable_checkpointing"] = False
    if "enable_progress_bar" in training_spec_with_generated_hparams.trainer_args:
        warnings.warn("enable_progress_bar found. Will be overwritten to False")
    training_spec_with_generated_hparams.trainer_args["enable_progress_bar"] = False

    # get callbacks (set to empty list if none defined) and extend with default ones
    training_spec_with_generated_hparams.trainer_args.setdefault(
        "callbacks", []
    ).extend(default_callbacks)

    trainer = Trainer(**training_spec_with_generated_hparams.trainer_args)

    # trainer = prepare_trainer(trainer)

    mlflow.set_tracking_uri(storage_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(
        run_name=_generate_random_name(training_spec.task.name),
        parent_run_id=parent_run_id,
    ) as run:
        trainer.logger = MLFlowLogger(
            experiment_name=experiment_name,
            run_id=run.info.run_id,
            run_name=run.info.run_name,
            save_dir=storage_uri,
            log_model=save_models,
        )

        # explicitly log batch_size. Since it is not a model param, it will not be logged
        mlflow.log_param("batch_size", task.datamodule.batch_size)
        trainer.fit(lightning_task, datamodule=task.datamodule)
        print("Trial Storage: ", trial_storage.trial_fs_path)
        if trial_storage is not None:
            mlflow.log_artifacts(trial_storage.trial_fs_path)
