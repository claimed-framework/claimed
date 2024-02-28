"""
This module contains the high level functions for benchmarking on a single node.
"""
from typing import Any

import mlflow
import pandas as pd
import ray
from jsonargparse import CLI
from tabulate import tabulate

from benchmark.model_fitting import ray_train_model, ray_tune_model
from benchmark.types import (
    Backbone,
    Task,
    build_model_args,
    optimization_space_type,
)


def benchmark_backbone_on_task(
    backbone: Backbone,
    task: Task,
    storage_uri: str,
    experiment_name: str,
    optimization_space: optimization_space_type | None = None,
    n_trials: int = 1,
    save_models: bool = False,
    pruning_grace_period: int | None = 10,
) -> dict:
    with mlflow.start_run(
        run_name=f"{backbone.backbone}_{task.name}", nested=True
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
            experiment_name,
            save_models,
            n_trials,
            pruning_grace_period=pruning_grace_period,
        )

        # mlflow.log_table(
        #     results.get_dataframe(), f"results_{task.name}.json", run.info.run_id
        # )

        if results.get_best_result().metrics is None:
            raise Exception("Best result metrics were none")
        return {
            "best_result": results.get_best_result().metrics[task.metric],
            "metric": task.metric,
            "best_config": results.get_best_result().config,
        }


def benchmark_backbone(
    backbone: Backbone,
    tasks: list[Task],
    storage_uri: str,
    experiment_name: str,
    benchmark_suffix: str | None = None,
    n_trials: int = 1,
    optimization_space: optimization_space_type | None = None,
    save_models: bool = False,
    pruning_grace_period: int | None = 10,
):
    """Highest level function to benchmark a backbone using a ray cluster

    Args:
        backbone (Backbone): Backbone to be used for the benchmark
        experiment_name (str): Name of the MLFlow experiment to be used.
        tasks (list[Task]): List of Tasks to benchmark over.
        storage_uri (str): Path to storage location.
        benchmark_suffix (str | None, optional): Suffix to be added to benchmark run name. Defaults to None.
        n_trials (int, optional): Number of hyperparameter optimization trials to run. Defaults to 1.
        optimization_space (optimization_space_type | None, optional): Parameters to optimize over. Should be a dictionary
            of strings (parameter name) to list (discrete set of possibilities) or ParameterBounds, defining a range to optimize over.
            Arguments belonging passed to the backbone, decoder or head should be given in the form `backbone_{argument}`, `decoder_{argument}` or `head_{argument}` Defaults to None.
        save_models (bool, optional): Whether to save the model. Defaults to False.
        pruning_grace_period (int, optional): Minimum number of epochs after which to consider pruning run. Pass None to not do pruning. Defaults to 10.
    """
    ray.init()
    mlflow.set_tracking_uri(storage_uri)
    mlflow.set_experiment(experiment_name)
    # mlflow.pytorch.autolog(log_datasets=False)
    run_name = backbone.backbone
    if benchmark_suffix:
        run_name += f"_{benchmark_suffix}"

    table_columns = ["Task", "Metric", "Best Score", "Hyperparameters"]
    table_entries = []

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("purpose", "backbone_benchmarking")

    if optimization_space is None:
        ray_tasks = []
        for task in tasks:
            lightning_task_class = task.type.get_class_from_enum()
            model_args = build_model_args(backbone, task)
            ray_tasks.append(
                ray_train_model.remote(
                    backbone,
                    task,
                    lightning_task_class,
                    model_args,
                    run_name,
                    run.info.run_id,
                    storage_uri,
                    experiment_name,
                    save_models,
                    pruning_grace_period=pruning_grace_period,
                )
            )
        results = ray.get(ray_tasks)
        table_entries = [
            [
                task.name,
                task.metric,
                result.metrics[task.metric],
                None,
            ]
            for task, result in zip(tasks, results)
        ]
    else:
        results = []
        for task in tasks:
            results.append(
                benchmark_backbone_on_task(
                    backbone,
                    task,
                    storage_uri,
                    experiment_name,
                    optimization_space=optimization_space,
                    n_trials=n_trials,
                    save_models=save_models,
                    pruning_grace_period=pruning_grace_period,
                )
            )

        table_entries = [
            [
                task.name,
                result["metric"],
                result["best_result"],
                result["best_config"],
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
