import copy
from typing import Any

import functools
import mlflow
import optuna
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import MLFlowLogger
from ray import train, tune
from ray.air.integrations.mlflow import MLflowLoggerCallback as RayMLFlowLoggerCallback
from ray.air.integrations.mlflow import setup_mlflow
from ray.tune.schedulers import ASHAScheduler
from lightning.pytorch.strategies import SingleDeviceStrategy
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchgeo.datamodules import BaseDataModule
from torchgeo.trainers import BaseTask
import os

from benchmark.types import (
    Backbone,
    ParameterBounds,
    ParameterTypeEnum,
    Task,
    optimization_space_type,
    valid_task_types,
)


def launch_training(
    trainer: Trainer,
    task: BaseTask,
    datamodule: BaseDataModule,
    run_name: str,
    metric: str,
    storage_uri: str,
    experiment_name: str,
) -> float:
    with mlflow.start_run(run_name=run_name, nested=True) as run:
        # explicitly log batch_size. Since it is not a model param, it will not be logged
        mlflow.log_param("batch_size", datamodule.batch_size)
        trainer.logger = MLFlowLogger(
            experiment_name=experiment_name,
            run_id=run.info.run_id,
            save_dir=storage_uri,
            log_model=True,
        )
        trainer.fit(task, datamodule=datamodule)
        client = mlflow.tracking.MlflowClient(
            tracking_uri=storage_uri,
        )

        metric_history = client.get_metric_history(run.info.run_id, metric)
        if len(metric_history) == 0:
            raise Exception(
                f"No values for metric {metric}. Choose a valid metric for this task"
            )
        return metric_history[-1].value  # or best idk
        # trainer.test(task, datamodule=datamodule)


def inject_hparams(model_setup: dict[str, Any], model_hparams: dict[str, Any]):
    model_setup_with_injected_hparams = copy.deepcopy(model_setup)
    # assume maximum nesting value is 2
    for k, v in model_hparams.items():
        if k in model_setup_with_injected_hparams and isinstance(
            model_setup_with_injected_hparams[k], dict
        ):
            # overwrite / merge keys
            model_setup_with_injected_hparams[k] |= v
        else:
            # either add key or overwrite existing key
            model_setup_with_injected_hparams[k] = v
    return model_setup_with_injected_hparams


def fit_model(
    backbone: Backbone,
    model_args: dict,
    task: Task,
    lightning_task_class: valid_task_types,
    run_name: str,
    storage_uri: str,
    experiment_name: str,
    lr: float | None = None,
    batch_size: int | None = None,
    freeze_backbone: bool = False,
    save_models: bool = True,
) -> tuple[float, str]:
    if batch_size:
        task.datamodule.batch_size = (
            batch_size  # TODO: not sure if this will work, check
        )
    if lr is None:
        lr = task.lr

    lightning_task = lightning_task_class(
        model_args,
        backbone.model_factory,
        loss=task.loss,
        lr=lr,
        optimizer=torch.optim.AdamW,
        optimizer_hparams={"weight_decay": 0.05},
        freeze_backbone=freeze_backbone,
        ignore_index=task.ignore_index,
        scheduler=ReduceLROnPlateau,
        scheduler_hparams={"patience": 5},
    )
    callbacks = [
        RichProgressBar(),
        EarlyStopping(monitor="val/loss", patience=10),  # let user configure this
    ]
    if save_models:
        callbacks.append(ModelCheckpoint(monitor="val/loss"))
    trainer = Trainer(
        callbacks=callbacks,
        max_epochs=task.max_epochs,
        enable_checkpointing=save_models,
    )
    return launch_training(
        trainer,
        lightning_task,
        task.datamodule,
        run_name,
        task.metric,
        storage_uri,
        experiment_name,
    ), task.metric


def ray_fit_model(
    config: dict,
    backbone: Backbone,
    base_args: dict,
    task: Task,
    lightning_task_class: valid_task_types,
    run_name: str,
    storage_uri: str,
    experiment_name: str,
    parent_run_id: str,
    save_models: bool = True,
) -> None:
    lr = float(config.pop("lr", task.lr))
    batch_size = config.pop("batch_size", None)
    if batch_size is not None:
        batch_size = int(batch_size)
    freeze_backbone = bool(config.pop("freeze_backbone", False))
    model_args = inject_hparams(base_args, config)
    if batch_size:
        task.datamodule.batch_size = (
            batch_size  # TODO: not sure if this will work, check
        )
    if lr is None:
        lr = task.lr

    lightning_task = lightning_task_class(
        model_args,
        backbone.model_factory,
        loss=task.loss,
        lr=lr,
        optimizer=torch.optim.AdamW,
        optimizer_hparams={"weight_decay": 0.05},
        freeze_backbone=freeze_backbone,
        ignore_index=task.ignore_index,
        scheduler=ReduceLROnPlateau,
        scheduler_hparams={"patience": 5},
    )
    callbacks = [
        RayTrainReportCallback()
    ]

    if save_models:
        callbacks.append(ModelCheckpoint(monitor="val/loss"))

    trainer = Trainer(
        strategy=RayDDPStrategy(find_unused_parameters=True),
        callbacks=callbacks,
        plugins=[RayLightningEnvironment()],
        enable_checkpointing=save_models,
        accelerator="auto",
        devices="auto",
        enable_progress_bar=False,
        max_epochs=task.max_epochs
        # strategy=SingleDeviceStrategy()
    )
    trainer = prepare_trainer(trainer)

    # not sure why these are necessary
    mlflow.set_tracking_uri(storage_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(nested=True) as run:
        # hack for nestedness
        mlflow.set_tag("mlflow.parentRunId", parent_run_id)

        # mlflow.pytorch.autolog(log_datasets=False, log_models=False)
        # explicitly log batch_size. Since it is not a model param, it will not be logged
        mlflow.log_param("batch_size", task.datamodule.batch_size)
        trainer.logger = MLFlowLogger(
            experiment_name=experiment_name,
            run_id=run.info.run_id,
            save_dir=storage_uri,
            log_model=False,
        )
        trainer.fit(lightning_task, datamodule=task.datamodule)


def ray_tune_model(
    backbone: Backbone,
    task: Task,
    lightning_task_class: valid_task_types,
    base_args: dict[str, Any],
    run_name: str,
    hparam_space: optimization_space_type,
    storage_uri: str,
    experiment_name: str,
    save_models: bool,
    num_trials: int,
) -> tune.ResultGrid:
    trainable = tune.with_parameters(
        ray_fit_model,
        backbone=backbone,
        base_args=base_args,
        task=task,
        lightning_task_class=lightning_task_class,
        run_name=run_name,
        storage_uri=storage_uri,
        experiment_name=experiment_name,
        parent_run_id=mlflow.active_run().info.run_id,
        save_models=save_models,
    )

    current_hparams: dict[str, Any] = {}

    for parameter, space in hparam_space.items():
        if parameter in task.optimization_except:
            continue
        if isinstance(space, list):
            suggestion = tune.choice(space)
            if suggestion is None:
                raise Exception(f"Optuna suggested None for parameter {parameter}")
            current_hparams[parameter] = suggestion
        elif isinstance(space, ParameterBounds):
            match space.type:
                case ParameterTypeEnum.integer:
                    current_hparams[parameter] = tune.quniform(space.min, space.max, 1)
                case ParameterTypeEnum.real:
                    if space.log:
                        current_hparams[parameter] = tune.loguniform(
                            space.min, space.max
                        )
                    else:
                        current_hparams[parameter] = tune.uniform(space.min, space.max)
                case _:
                    raise Exception(
                        f"Type {space.type} not recognized. Suggest one of {[e.value for e in ParameterTypeEnum]}"
                    )
    
    scheduler = ASHAScheduler(max_t=task.max_epochs, grace_period=min(task.max_epochs, 5), reduction_factor=2)

    scaling_config = ScalingConfig(
        num_workers=1, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": 1}
    )
    run_config=train.RunConfig(
            name=run_name,
            storage_path=storage_uri
    )
    ray_trainer = TorchTrainer(
        trainable,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    tuner = tune.Tuner(
        ray_trainer,
        tune_config=tune.TuneConfig(
            metric=task.metric,
            mode="min", # let user choose this
            num_samples=num_trials,
            scheduler=scheduler,
        ),
        param_space={"train_loop_config": current_hparams},
    )

    results = tuner.fit()
    return results


def fit_model_with_hparams(
    backbone: Backbone,
    task: Task,
    lightning_task_class: valid_task_types,
    base_args: dict[str, Any],
    run_name: str,
    hparam_space: optimization_space_type,
    storage_uri: str,
    experiment_name: str,
    save_models: bool,
    trial: optuna.Trial,
) -> float:
    # treat lr and batch_size specially

    current_hparams: dict[str, int | float | str | bool] = {}

    for parameter, space in hparam_space.items():
        if parameter in task.optimization_except:
            continue
        if isinstance(space, list):
            suggestion = trial.suggest_categorical(parameter, space)
            if suggestion is None:
                raise Exception(f"Optuna suggested None for parameter {parameter}")
            current_hparams[parameter] = suggestion
        elif isinstance(space, ParameterBounds):
            match space.type:
                case ParameterTypeEnum.integer:
                    current_hparams[parameter] = trial.suggest_int(
                        parameter,
                        int(space.min),
                        int(space.max),
                        step=int(space.step) if space.step else 1,
                    )
                case ParameterTypeEnum.real:
                    current_hparams[parameter] = trial.suggest_float(
                        parameter, space.min, space.max, step=space.step, log=space.log
                    )
                case _:
                    raise Exception(
                        f"Type {space.type} not recognized. Suggest one of {[e.value for e in ParameterTypeEnum]}"
                    )
    lr = float(current_hparams.pop("lr", task.lr))
    batch_size = current_hparams.pop("batch_size", None)
    if batch_size is not None:
        batch_size = int(batch_size)
    freeze_backbone = bool(current_hparams.pop("freeze_backbone", False))
    model_args = inject_hparams(base_args, current_hparams)
    run_name = f"{run_name}_{trial.number}"
    return fit_model(
        backbone,
        model_args,
        task,
        lightning_task_class,
        run_name,
        storage_uri,
        experiment_name,
        lr=lr,
        batch_size=batch_size,
        freeze_backbone=freeze_backbone,
        save_models=save_models,
    )[0]  # return only the metric value for optuna
