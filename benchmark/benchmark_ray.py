"""
This module contains the high level functions for benchmarking on a single node.
"""

import importlib

import mlflow
import pandas as pd
import ray
from jsonargparse import CLI
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from tabulate import tabulate

from benchmark.benchmark_types import (
    Backbone,
    Task,
    build_model_args,
    optimization_space_type,
)
from benchmark.model_fitting import fit_model, ray_tune_model, valid_task_types


def benchmark_backbone_on_task(
    backbone: Backbone,
    task: Task,
    storage_uri: str,
    experiment_name: str,
    ray_storage_path: str,
    optimization_space: optimization_space_type | None = None,
    n_trials: int = 1,
    save_models: bool = False,
    precision: _PRECISION_INPUT = "16-mixed",
    backbone_import: str|None = None
) -> dict:
    with mlflow.start_run(
        run_name=f"{mlflow.active_run().info.run_name}_{task.name}",
        nested=True,
    ) as run:
        lightning_task_class = task.type.get_class_from_enum()
        model_args = build_model_args(backbone, task)

        # if no optimization params, just run it
        if optimization_space is None:
            raise Exception("For no optimiation space, run benchmark.py")

        results = ray_tune_model(
            backbone,
            task,
            lightning_task_class,
            model_args,
            optimization_space,
            storage_uri,
            ray_storage_path,
            experiment_name,
            save_models,
            n_trials,
            precision,
            backbone_import=backbone_import
        )

        mlflow.log_table(
            results.get_dataframe(), f"results_{task.name}.json", run.info.run_id
        )
        if results.get_best_result().metrics is None:
            raise Exception("Best result metrics were none")
        if results.get_best_result().config is None:
            raise Exception("Best result config was none")

        mlflow.log_params(results.get_best_result().config)
        mlflow.log_metric(
            f"best_{task.metric}", results.get_best_result().metrics[task.metric]
        )
        return {
            "best_result": results.get_best_result().metrics[task.metric],
            "metric": task.metric,
            "best_config": results.get_best_result().config,
        }


@ray.remote(num_cpus=8, num_gpus=1)
def remote_fit(
    backbone: Backbone,
    model_args: dict,
    task: Task,
    lightning_task_class: valid_task_types,
    run_name: str,
    storage_uri: str,
    experiment_name: str,
    parent_run_id: str,
    save_models: bool,
    precision: _PRECISION_INPUT,
    backbone_import: str | None
) -> float:
    mlflow.set_tracking_uri(storage_uri)
    mlflow.set_experiment(experiment_name)
    if backbone_import:
        importlib.import_module(backbone_import)
    lr = float(model_args.pop("lr", task.lr))
    batch_size = model_args.pop("batch_size", None)
    if batch_size is not None:
        batch_size = int(batch_size)
    freeze_backbone = bool(model_args.pop("freeze_backbone", False))
    return fit_model(
        backbone,
        model_args,
        task,
        lightning_task_class,
        run_name,
        experiment_name,
        storage_uri,
        parent_run_id,
        save_models=save_models,
        lr=lr,
        batch_size=batch_size,
        freeze_backbone=freeze_backbone,
        precision=precision,
    )[0]


def benchmark_backbone(
    backbone: Backbone,
    tasks: list[Task],
    storage_uri: str,
    experiment_name: str,
    backbone_import: str | None = None,
    benchmark_suffix: str | None = None,
    n_trials: int = 1,
    ray_storage_path: str | None = None,
    optimization_space: optimization_space_type | None = None,
    save_models: bool = False,
    run_id: str | None = None,
    precision: _PRECISION_INPUT = "16-mixed",
):
    """Highest level function to benchmark a backbone using a ray cluster

    Args:
        backbone (Backbone): Backbone to be used for the benchmark
        experiment_name (str): Name of the MLFlow experiment to be used.
        tasks (list[Task]): List of Tasks to benchmark over.
        storage_uri (str): Path to storage location.
        ray_storage_path (str | None): Path to storage of ray outputs, including saved models, when using ray tune. Required if optimization_space is specified
        backbone_import (str | None): Path to module that will be imported to register a potential new backbone. Defaults to None.
        benchmark_suffix (str | None, optional): Suffix to be added to benchmark run name. Defaults to None.
        n_trials (int, optional): Number of hyperparameter optimization trials to run. Defaults to 1.
        optimization_space (optimization_space_type | None, optional): Parameters to optimize over. Should be a dictionary
            of strings (parameter name) to list (discrete set of possibilities) or ParameterBounds, defining a range to optimize over.
            Arguments belonging passed to the backbone, decoder or head should be given in the form `backbone_{argument}`, `decoder_{argument}` or `head_{argument}` Defaults to None.
        save_models (bool, optional): Whether to save the model. Defaults to False.
        run_id (str | None): id of existing mlflow run to use as top-level run. Useful to add more experiments to a previous benchmark run. Defaults to None.
        precision (str): precision to use for training. Defaults to 16-mixed.
    """
    ray.init()
    if backbone_import:
        importlib.import_module(backbone_import)
    mlflow.set_tracking_uri(storage_uri)
    mlflow.set_experiment(experiment_name)
    # mlflow.pytorch.autolog(log_datasets=False)
    run_name = (
        backbone.backbone
        if isinstance(backbone.backbone, str)
        else str(type(backbone.backbone).__name__)
    )
    if benchmark_suffix:
        run_name += f"_{benchmark_suffix}"

    table_columns = ["Task", "Metric", "Best Score", "Hyperparameters"]
    table_entries = []

    with mlflow.start_run(run_name=run_name, run_id=run_id) as run:
        mlflow.set_tag("purpose", "backbone_benchmarking")

        if optimization_space is None:
            # no hparams, parallelize over tasks
            ray_tasks = []
            for task in tasks:
                run_name = f"{backbone.backbone if isinstance(backbone.backbone, str) else str(type(backbone.backbone).__name__)}_{task.name}"
                lightning_task_class = task.type.get_class_from_enum()
                model_args = build_model_args(backbone, task)
                ray_tasks.append(
                    remote_fit.remote(
                        backbone,
                        model_args,
                        task,
                        lightning_task_class,
                        run_name,
                        storage_uri,
                        experiment_name,
                        run.info.run_id,
                        save_models,
                        precision,
                        backbone_import
                    )
                )
            results = ray.get(ray_tasks)
            table_entries = [
                [
                    task.name,
                    task.metric,
                    result,
                    None,
                ]
                for task, result in zip(tasks, results)
            ]
        else:
            if ray_storage_path is None:
                raise Exception(
                    "`ray_storage_path` must be specified if `optimization_space` is specified."
                )
            # hparams, parallelize within tasks, run one task at a time.
            results = []
            for task in tasks:
                results.append(
                    benchmark_backbone_on_task(
                        backbone,
                        task,
                        storage_uri,
                        experiment_name,
                        ray_storage_path,
                        optimization_space=optimization_space,
                        n_trials=n_trials,
                        save_models=save_models,
                        precision=precision,
                        backbone_import=backbone_import
                    )
                )

            table_entries = [
                [
                    task.name,
                    result["metric"],
                    result["best_result"],
                    str(result["best_config"]),
                ]
                for task, result in zip(tasks, results)
            ]

        table = tabulate(table_entries, headers=table_columns)
        print(table)
        df = pd.DataFrame(data=table_entries, columns=table_columns)
        df.set_index("Task")
        mlflow.log_table(
            df,
            "results_table.json",
            run.info.run_id,
        )
        ray.shutdown()


def main():
    CLI(benchmark_backbone, fail_untyped=False)


if __name__ == "__main__":
    main()
