"""
This module contains functions to re-run a best backbone with different seeds
"""

import copy
import importlib
import os
import warnings
from ast import literal_eval
from random import randint
from typing import Any

import mlflow
import mlflow.entities
import pandas as pd
import ray
import torch
from jsonargparse import CLI
from lightning import Callback, Trainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from tabulate import tabulate
from terratorch.tasks import PixelwiseRegressionTask, SemanticSegmentationTask

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
    backbone_import: str | None = None
) -> float | None:
    seed_everything(seed, workers=True)
    if backbone_import:
        importlib.import_module(backbone_import)

    training_spec_copy = copy.deepcopy(training_spec)
    training_spec_with_generated_hparams = inject_hparams(training_spec_copy, best_params)
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

    default_callbacks: list[Callback] = get_default_callbacks(task.early_stop_patience)
    # get callbacks (set to empty list if none defined) and extend with default ones
    training_spec_with_generated_hparams.trainer_args.setdefault("callbacks", []).extend(
        default_callbacks
    )  # type: ignore
    training_spec_copy.trainer_args["enable_checkpointing"] = False
    # remove model saving callbacks
    if "callbacks" in training_spec_copy.trainer_args:
        training_spec_copy.trainer_args["callbacks"] = [  # type: ignore
            c
            for c in training_spec_copy.trainer_args["callbacks"]  # type: ignore
            if not isinstance(c, ModelCheckpoint)
        ]  # type: ignore

    trainer = Trainer(**training_spec_with_generated_hparams.trainer_args, enable_progress_bar=False, enable_checkpointing=False)
    try:
        trainer.fit(lightning_task, datamodule=task.datamodule)
        metrics = trainer.test(
            lightning_task, datamodule=task.datamodule, verbose=False
        )
        metrics = metrics[0]
    except Exception as e:
        warnings.warn(str(e))
        return None
    test_metric = "test/" + task.metric.split("/")[1]
    return metrics[test_metric]


def rerun_best_from_backbone(
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
        raise Exception(f"output_path must be absolute. Consider using $(pwd)/{output_path}.")
    if tmp_dir is None:
        raise Exception("tmp_dir must be specified for runs with ray.")
    os.environ["RAY_TMPDIR"] = tmp_dir
    ray.init(_temp_dir=tmp_dir)
    if backbone_import:
        importlib.import_module(backbone_import)
    mlflow.set_tracking_uri(storage_uri)
    mlflow.set_experiment(experiment_name)

    runs: list[mlflow.entities.Run] = mlflow.search_runs(
        filter_string=f"tags.mlflow.parentRunId='{parent_run_id}'", output_format="list"
    )  # type: ignore
    print(f"Found runs: {[run.info.run_name for run in runs]}")

    table_columns = ["Task", "Metric", "Score", "MLFlow run id"]
    table_entries = []

    ray_tasks = []
    seeds = [randint(1, 5000) for i in range(run_repetitions)]
    for task in tasks:
        matching_runs = [run for run in runs if run.info.run_name.endswith(task.name)]  # type: ignore
        if len(matching_runs) == 0:
            msg = f"No runs found for task {task.name}. Skipping."
            warnings.warn(msg)
            continue
        if len(matching_runs) > 1:
            msg = f"More than 1 run found for task {task.name}"
            raise Exception(msg)

        best_params = matching_runs[0].data.params
        # eval them
        best_params = {k: literal_eval(v) for k, v in best_params.items()}
        training_spec = combine_with_defaults(task, defaults)
        lightning_task_class = training_spec.task.type.get_class_from_enum()
        for seed in seeds:
            ray_tasks.append(
                remote_fit.remote(
                    training_spec,
                    lightning_task_class,
                    best_params,
                    seed,
                    backbone_import=backbone_import
                )
            )
    results = ray.get(ray_tasks)
    table_entries = [
        [
            task.name,
            task.metric.split("/")[-1],
            result,
            matching_runs[0].info.run_id
        ]
        for task, result in zip(
            [task for task in tasks for _ in seeds], results
        )  # expand tasks
    ]

    table = tabulate(table_entries, headers=table_columns)
    print(table)
    df = pd.DataFrame(data=table_entries, columns=table_columns)
    df.to_csv(output_path, index=False)
    ray.shutdown()


def main():
    CLI(rerun_best_from_backbone, fail_untyped=False)


if __name__ == "__main__":
    main()
