"""
This module contains functions to re-run a best backbone with different seeds
"""

import copy
import importlib
import os
import glob
import warnings
import logging
from ast import literal_eval
from random import randint

import mlflow
import mlflow.entities
import pandas as pd
import ray
from jsonargparse import CLI
from lightning import Callback, Trainer
from lightning.pytorch import seed_everything
from tabulate import tabulate
from terratorch.tasks import PixelwiseRegressionTask, SemanticSegmentationTask

from lightning.pytorch.loggers.mlflow import MLFlowLogger
import time

from benchmark.benchmark_types import (
    Defaults,
    Task,
    TrainingSpec,
    combine_with_defaults,
)
from benchmark.model_fitting import (
    get_default_callbacks,
    inject_hparams,
    valid_task_types,
)


@ray.remote(num_cpus=8, num_gpus=1)
def remote_fit(
    training_spec: TrainingSpec,
    lightning_task_class: valid_task_types,
    best_params: dict,
    seed: int,
    backbone_import: str | None = None,
) -> float | None:
    seed_everything(seed, workers=True)
    if backbone_import:
        importlib.import_module(backbone_import)

    with mlflow.start_run(
        run_name=f"{lightning_task_class.name}_{seed}",
        nested=True,
    ) as run:

        training_spec_copy = copy.deepcopy(training_spec)
        training_spec_with_generated_hparams = inject_hparams(
            training_spec_copy, best_params
        )
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
        # get callbacks (set to empty list if none defined) and extend with default ones
        training_spec_with_generated_hparams.trainer_args.setdefault(
            "callbacks", []
        ).extend(
            default_callbacks
        )  # type: ignore
        if "enable_checkpointing" in training_spec_with_generated_hparams.trainer_args:
            warnings.warn(
                "enable_checkpointing found. Will be overwritten to False as ray will be responsible for saving models."
            )
        training_spec_with_generated_hparams.trainer_args["enable_checkpointing"] = (
            False
        )
        if "enable_progress_bar" in training_spec_with_generated_hparams.trainer_args:
            warnings.warn("enable_progress_bar found. Will be overwritten to False")
        training_spec_with_generated_hparams.trainer_args["enable_progress_bar"] = False
        trainer = Trainer(**training_spec_with_generated_hparams.trainer_args)
        try:
            trainer.fit(lightning_task, datamodule=task.datamodule)
            metrics = trainer.test(
                lightning_task, datamodule=task.datamodule, verbose=False
            )
            metrics = metrics[0]
        except Exception as e:
            raise Exception(str(e))
        #        warnings.warn(str(e))
        #        return None
        test_metric = "test/" + task.metric.split("/")[1]
        mlflow.log_metric(f"test_{test_metric}", metrics[test_metric])
        return metrics[test_metric]


def non_remote_fit(
    experiment_name: str,
    parent_run_id: str,
    storage_uri: str,
    task: Task,
    training_spec: TrainingSpec,
    lightning_task_class: valid_task_types,
    best_params: dict,
    seed: int,
    backbone_import: str | None = None,
) -> float | None:
    seed_everything(seed, workers=True)
    if backbone_import:
        importlib.import_module(backbone_import)
    with mlflow.start_run(
        run_name=f"{task.name}_{seed}",
        nested=True,
    ) as run:
        mlflow.set_tag("mlflow.parentRunId", parent_run_id)
        training_spec_copy = copy.deepcopy(training_spec)
        training_spec_with_generated_hparams = inject_hparams(
            training_spec_copy, best_params
        )
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
        # get callbacks (set to empty list if none defined) and extend with default ones
        training_spec_with_generated_hparams.trainer_args.setdefault(
            "callbacks", []
        ).extend(
            default_callbacks
        )  # type: ignore
        if "enable_checkpointing" in training_spec_with_generated_hparams.trainer_args:
            warnings.warn(
                "enable_checkpointing found. Will be overwritten to False as ray will be responsible for saving models."
            )
        training_spec_with_generated_hparams.trainer_args["enable_checkpointing"] = (
            False
        )
        if "enable_progress_bar" in training_spec_with_generated_hparams.trainer_args:
            warnings.warn("enable_progress_bar found. Will be overwritten to False")
        training_spec_with_generated_hparams.trainer_args["enable_progress_bar"] = False
        trainer = Trainer(**training_spec_with_generated_hparams.trainer_args)

        trainer.logger = MLFlowLogger(
            experiment_name=experiment_name,
            run_id=run.info.run_id,
            save_dir=storage_uri,
            log_model=True,
        )
        try:
            trainer.fit(lightning_task, datamodule=task.datamodule)
            metrics = trainer.test(
                lightning_task, datamodule=task.datamodule, verbose=False
            )
            metrics = metrics[0]
        except Exception as e:
            raise Exception(str(e))
        #        warnings.warn(str(e))
        #        return None
        test_metric = "test/" + task.metric.split("/")[1]
        mlflow.log_metric(f"test_{test_metric}", metrics[test_metric])
        return metrics[test_metric]


def rerun_best_from_backbone(
    logger: logging.RootLogger,
    parent_run_id: str,
    output_path: str,
    defaults: Defaults,
    tasks: list[Task],
    experiment_name: str,
    storage_uri: str,
    *args,
    tmp_dir: str | None = None,
    run_repetitions: int = 10,
    backbone_import: str | None = None,
    run_name: str | None = None,
    n_trials: int = 1,
    ray_storage_path: str | None = None,
    save_models: bool = False,
    run_id: str | None = None,
    optimization_space: dict | None = None,
    description: str | None = None,
    use_ray=False,
    **kwargs,
):
    """Repeat best experiments from a benchmark run. Only works with a ray cluster.

    Args:
        parent_run_id (str): mlflow id of parent run
        output_path (str): path to store the results of the run
        tmp_dir (str): Path to temporary directory to be used for ray
        run_repetitions (int): How many runs (each with a different seed) to run per task.

    """
    if not os.path.isabs(output_path):
        raise Exception(
            f"output_path must be absolute. Consider using $(pwd)/{output_path}."
        )
    if tmp_dir is None:
        raise Exception("tmp_dir must be specified for runs with ray.")

    if use_ray:
        os.environ["RAY_TMPDIR"] = tmp_dir
        ray.init(_temp_dir=tmp_dir)
    if backbone_import:
        importlib.import_module(backbone_import)
    mlflow.set_tracking_uri(storage_uri)
    mlflow.set_experiment(experiment_name)

    runs: list[mlflow.entities.Run] = mlflow.search_runs(
        filter_string=f"tags.mlflow.parentRunId='{parent_run_id}'", output_format="list"
    )  # type: ignore
    logger.info(f"\nFound runs: {[run.info.run_name for run in runs]}")

    task_names = [task.name for task in tasks]
    logger.info(f"Will only run the following: {task_names}")

    table_columns = [
        "Task",
        "Metric",
        "Score",
        "mlflow_run_name",
        "mlflow_run_id",
        "mlflow_run_status",
    ]
    table_entries = []
    ray_tasks = []

    repeated_storage_uri = f"{storage_uri}_repeated_exp"
    if not os.path.exists(repeated_storage_uri):
        os.makedirs(repeated_storage_uri)

    repeated_experiment_name = f"{experiment_name}_repeated_exp"
    mlflow.set_tracking_uri(repeated_storage_uri)
    mlflow.set_experiment(repeated_experiment_name)

    backbone_name = defaults.terratorch_task["model_args"]["backbone"]
    with mlflow.start_run(run_name=backbone_name, run_id=None) as run:
        for task in tasks:
            logger.info(f"\n\ntask: {task.name}")
            matching_runs = [run for run in runs if run.info.run_name.endswith(task.name)]  # type: ignore
            if len(matching_runs) == 0:
                msg = f"No runs found for task {task.name}. Skipping."
                warnings.warn(msg)
                continue
            if len(matching_runs) > 1:
                msg = f"More than 1 run found for task {task.name}"
                raise Exception(msg)

            # check if there are already results for this task and exp in the folder
            past_output_path = (
                f"{output_path.split(experiment_name)[0]}{experiment_name}_*"
            )
            past_output_path = glob.glob(past_output_path)
            if len(sorted(past_output_path)) > 0:
                output_path = sorted(past_output_path)[0]
            logger.info(f"output path: {output_path}")
            if os.path.exists(output_path):
                logger.info("there are previous results from repeated experiments")
                existing_output = pd.read_csv(output_path)
                existing_output = existing_output[table_columns]
                existing_task_output = existing_output.loc[
                    existing_output["Task"] == task.name
                ].copy()
                rows, cols = existing_task_output.shape
                logger.info(f"rows: {rows} \t cols: {cols}")
                if rows > run_repetitions:
                    logger.info("task has valid results, will not re-run")
                    continue
                past_seeds = [
                    int(item.split("_")[-1])
                    for item in existing_task_output["mlflow_run_name"].tolist()
                ]
            else:
                past_seeds = []
            logger.info(f"past_seeds for task: {past_seeds}")

            best_params = matching_runs[0].data.params
            best_params = {k: literal_eval(v) for k, v in best_params.items()}
            training_spec = combine_with_defaults(task, defaults)
            lightning_task_class = training_spec.task.type.get_class_from_enum()

            if use_ray:  # experimental
                successful_seeds = [randint(1, 5000) for i in range(run_repetitions)]
                for seed in successful_seeds:
                    ray_tasks.append(
                        remote_fit.remote(
                            training_spec,
                            lightning_task_class,
                            best_params,
                            seed,
                            backbone_import=backbone_import,
                        )
                    )
            else:
                experiment_info = mlflow.get_experiment_by_name(
                    repeated_experiment_name
                )
                seeds = [randint(1, 5000) for i in range(run_repetitions * 3)]
                seeds = [seed for seed in seeds if seed not in past_seeds]

                for seed in seeds:
                    if len(past_seeds) >= run_repetitions:
                        break

                    seed_run_name = f"{task.name}_{seed}"
                    logger.info(f"now trying: {seed_run_name}")
                    seed_run_data = mlflow.search_runs(
                        experiment_ids=[experiment_info.experiment_id],
                        filter_string=f'tags."mlflow.runName" LIKE "{seed_run_name}"',
                        output_format="list",
                    )  # type: ignore
                    if len(seed_run_data) > 0:
                        for item in seed_run_data:
                            logger.info(f"deleting existing run: {item}")
                            mlflow.delete_run(item.info.run_id)

                    score = non_remote_fit(
                        experiment_name=repeated_experiment_name,
                        parent_run_id=run.info.run_id,
                        storage_uri=repeated_storage_uri,
                        task=task,
                        training_spec=training_spec,
                        lightning_task_class=lightning_task_class,
                        best_params=best_params,
                        seed=seed,
                        backbone_import=backbone_import,
                    )
                    # check if run with name finished successfully
                    logger.info(f"score: {score}")
                    time.sleep(3600 * 2)
                    seed_run_data = mlflow.search_runs(
                        experiment_ids=[experiment_info.experiment_id],
                        filter_string=f'tags."mlflow.runName" LIKE "{seed_run_name}"',
                        output_format="list",
                    )  # type: ignore

                    logger.info(
                        f"run for task {task.name} seed {seed} is :{seed_run_data}"
                    )
                    if len(seed_run_data) > 0:
                        if seed_run_data[0].info.status != "FINISHED":
                            mlflow.delete_run(seed_run_data[0].info.run_id)
                            continue
                        past_seeds.append(seed)
                        new_data = pd.DataFrame(
                            {
                                "Task": [task.name],
                                "Metric": [task.metric.split("/")[-1]],
                                "Score": [score],
                                "mlflow_run_name": [seed_run_name],
                                "mlflow_run_id": [seed_run_data[0].info.run_id],
                                "mlflow_run_status": [seed_run_data[0].info.status],
                            }
                        )
                        logger.info(
                            f"completed seeds so far for this task: {len(past_seeds)}"
                        )
                        if os.path.exists(output_path):
                            logger.info(
                                "there are previous results from repeated experiments"
                            )
                            existing_output = pd.read_csv(output_path)
                            existing_output = existing_output[table_columns]
                            existing_output.reset_index(inplace=True)
                            existing_task_output = existing_output.loc[
                                existing_output["Task"] == task.name
                            ].copy()
                            rows, cols = existing_task_output.shape
                            logger.info(f"rows: {rows} \t cols: {cols}")
                            if rows == 0:
                                logger.info("no past results for this task")
                            existing_output = pd.concat(
                                [existing_output, new_data], axis=0
                            )
                            existing_output.reset_index(inplace=True)
                            existing_output.to_csv(output_path, index=False)
                        else:
                            new_data.to_csv(output_path, index=False)

    if use_ray:  # experimental
        results = ray.get(ray_tasks)
        table_entries = [
            [
                task.name,
                task.metric.split("/")[-1],
                result,
                matching_runs[0].info.run_id,
            ]
            for task, result in zip(
                [task for task in tasks for _ in seeds], results
            )  # expand tasks
        ]

        table = tabulate(table_entries, headers=table_columns)
        logger.info(table)
        df = pd.DataFrame(data=table_entries, columns=table_columns)
        df.to_csv(output_path, index=False)
        ray.shutdown()


def main():
    CLI(rerun_best_from_backbone, fail_untyped=False)


if __name__ == "__main__":
    main()
