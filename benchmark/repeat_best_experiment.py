"""
This module contains functions to re-run a best backbone with different seeds
"""

import warnings
from ast import literal_eval
from random import randint

import mlflow
import mlflow.entities
import pandas as pd
import ray
import torch
from jsonargparse import CLI
from lightning import Callback, Trainer
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from tabulate import tabulate

from benchmark.benchmark_types import (
    Backbone,
    Task,
    build_model_args,
    optimization_space_type,
)
from benchmark.model_fitting import (
    inject_hparams,
    valid_task_types,
)


@ray.remote(num_cpus=8, num_gpus=1)
def remote_fit(
    backbone: Backbone,
    model_args: dict,
    task: Task,
    lightning_task_class: valid_task_types,
    seed: int,
    precision: _PRECISION_INPUT = "16-mixed",
) -> float | None:
    torch.set_float32_matmul_precision("high")
    seed_everything(seed, workers=True)

    lr = float(model_args.pop("lr", task.lr))
    batch_size = model_args.pop("batch_size", None)
    if batch_size is not None:
        batch_size = int(batch_size)
    freeze_backbone = bool(model_args.pop("freeze_backbone", False))

    if batch_size:
        task.datamodule.batch_size = batch_size
    if lr is None:
        lr = task.lr

    lightning_task = lightning_task_class(
        model_args,
        task.model_factory,
        loss=task.loss,
        lr=lr,
        optimizer="AdamW",
        optimizer_hparams={"weight_decay": 0.05},
        freeze_backbone=freeze_backbone,
        ignore_index=task.ignore_index,
        scheduler="ReduceLROnPlateau",
    )
    callbacks: list[Callback] = []

    if task.early_stop_patience is not None:
        callbacks.append(
            EarlyStopping(
                task.metric, mode=task.direction, patience=task.early_stop_patience
            )
        )
        # callbacks.append(EarlyStopping("val/loss", patience=task.early_stop_patience))

    trainer = Trainer(
        callbacks=callbacks,
        logger=False,
        max_epochs=task.max_epochs,
        # max_epochs=1,
        enable_checkpointing=False,
        enable_progress_bar=False,
        log_every_n_steps=10,
        precision=precision,
    )
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
    backbone: Backbone,
    tasks: list[Task],
    storage_uri: str,
    experiment_name: str,
    *args,
    benchmark_suffix: str | None = None,
    n_trials: int = 1,
    ray_storage_path: str | None = None,
    optimization_space: optimization_space_type | None = None,
    save_models: bool = False,
    precision: _PRECISION_INPUT = "16-mixed",
    **kwargs,
):
    ray.init()
    mlflow.set_tracking_uri(storage_uri)
    mlflow.set_experiment(experiment_name)

    runs: list[mlflow.entities.Run] = mlflow.search_runs(
        filter_string=f"tags.mlflow.parentRunId='{parent_run_id}'", output_format="list"
    )  # type: ignore
    print(f"Found runs: {[run.info.run_name for run in runs]}")
    print(
        f"Will match with task names: {['_'.join(run.info.run_name.split('_')[1:]) for run in runs]}"
    )
    table_columns = ["Task", "Metric", "Score"]
    table_entries = []

    ray_tasks = []
    seeds = [42] + [randint(1, 5000) for i in range(9)]
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
        lightning_task_class = task.type.get_class_from_enum()
        # print(f"Task {task.name}")
        # print("Best params:")
        # print(best_params)
        # print("============")
        model_args = build_model_args(backbone, task)
        # print("Built model args:")
        # print(model_args)
        # print("============")
        model_args = inject_hparams(model_args, best_params)
        # print("Final model args")
        # print(model_args)
        # print("-------------")
        for seed in seeds:
            ray_tasks.append(
                remote_fit.remote(
                    backbone,
                    model_args,
                    task,
                    lightning_task_class,
                    seed,
                    precision=precision,
                )
            )
    results = ray.get(ray_tasks)
    table_entries = [
        [
            task.name,
            task.metric.split("/")[-1],
            result,
        ]
        for task, result in zip(
            [task for task in tasks for _ in seeds], results
        )  # expand tasks
    ]

    table = tabulate(table_entries, headers=table_columns)
    print(table)
    df = pd.DataFrame(data=table_entries, columns=table_columns)
    df.to_csv(output_path)
    ray.shutdown()


def main():
    CLI(rerun_best_from_backbone, fail_untyped=False)


if __name__ == "__main__":
    main()
