"""
This module contains the high level functions for benchmarking on a single node.
"""
from functools import partial
from typing import Any

import mlflow
import optuna
import pandas as pd
import torch
from jsonargparse import CLI
from tabulate import tabulate

from benchmark.model_fitting import fit_model, fit_model_with_hparams
from benchmark.types import (
    Backbone,
    Task,
    build_model_args,
    optimization_space_type,
)

# override Optuna's default logging to ERROR only
optuna.logging.set_verbosity(optuna.logging.ERROR)


def benchmark_backbone_on_task(
    backbone: Backbone,
    task: Task,
    storage_uri: str,
    experiment_name: str,
    optimization_space: optimization_space_type | None = None,
    n_trials: int = 1,
    save_models: bool = False,
    pruning: bool = True,
) -> tuple[float, str | list[str] | None, dict[str, Any]]:
    with mlflow.start_run(
        run_name=f"{backbone.backbone}_{task.name}", nested=True
    ) as run:
        lightning_task_class = task.type.get_class_from_enum()
        model_args = build_model_args(backbone, task)

        # if no optimization params, just run it
        if optimization_space is None:
            return (
                *fit_model(
                    backbone,
                    model_args,
                    task,
                    lightning_task_class,
                    f"{run.info.run_name}",
                    storage_uri,
                    run.info.run_id,
                    save_models=save_models,
                    pruning=pruning,
                ),
                {},
            )

        # if optimization parameters specified, do hyperparameter tuning
        study = optuna.create_study(
            direction="minimize"  # in the future may want to allow user to specify this
        )
        objective = partial(
            fit_model_with_hparams,
            backbone,
            task,
            lightning_task_class,
            model_args,
            f"{backbone.backbone}_{task.name}",
            optimization_space,
            storage_uri,
            run.info.run_id,
            save_models,
            pruning,
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            # callbacks=[champion_callback],
            catch=[torch.cuda.OutOfMemoryError],  # add a few more here?
        )

        return study.best_value, task.metric, study.best_trial.params


def benchmark_backbone(
    backbone: Backbone,
    experiment_name: str,
    tasks: list[Task],
    storage_uri: str,
    benchmark_suffix: str | None = None,
    n_trials: int = 1,
    optimization_space: optimization_space_type | None = None,
    save_models: bool = False,
    pruning: bool = True,
):
    """Highest level function to benchmark a backbone using a single node

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
        pruning (bool, optional): Whether to prune epochs if they dont improve after 10 epochs. Defaults to True.
    """
    mlflow.set_tracking_uri(storage_uri)
    mlflow.set_experiment(experiment_name)
    run_name = backbone.backbone
    if benchmark_suffix:
        run_name += f"_{benchmark_suffix}"

    table_columns = ["Task", "Metric", "Best Score", "Hyperparameters"]
    table_entries = []
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("purpose", "backbone_benchmarking")
        for task in tasks:
            best_value, metric_name, hparams = benchmark_backbone_on_task(
                backbone,
                task,
                storage_uri,
                experiment_name,
                optimization_space=optimization_space,
                n_trials=n_trials,
                save_models=save_models,
                pruning=pruning,
            )
            table_entries.append([task.name, metric_name, best_value, hparams])

        table = tabulate(table_entries, headers=table_columns)
        print(table)
        df = pd.DataFrame(data=table_entries, columns=table_columns)
        df.set_index("Task")
        mlflow.log_table(
            df,
            "results_table.json",
            run.info.run_id,
        )


def main():
    CLI(benchmark_backbone, fail_untyped=False)


if __name__ == "__main__":
    main()
