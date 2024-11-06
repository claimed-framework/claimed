"""
This module contains the high level functions for benchmarking on a single node.
"""
import importlib
from functools import partial
from typing import Any

import mlflow
import optuna
import pandas as pd
import torch
from jsonargparse import CLI
from optuna.pruners import HyperbandPruner
from optuna.samplers import BaseSampler, RandomSampler
from tabulate import tabulate

from benchmark.benchmark_types import (
    Defaults,
    ParameterBounds,
    Task,
    combine_with_defaults,
    optimization_space_type,
)
from benchmark.model_fitting import fit_model, fit_model_with_hparams

direction_type_to_optuna = {"min": "minimize", "max": "maximize"}

def unflatten(dictionary):
    resultDict = {}
    for key, value in dictionary.items():
        parts = key.split(".")
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return resultDict

def benchmark_backbone_on_task(
    defaults: Defaults,
    task: Task,
    storage_uri: str,
    experiment_name: str,
    optimization_space: optimization_space_type | None = None,
    n_trials: int = 1,
    save_models: bool = False,
    sampler: BaseSampler | None = None,
    test_models: bool = False,
) -> tuple[float, str | list[str] | None, dict[str, Any]]:
    with mlflow.start_run(
        run_name=task.name,
        nested=True,
    ) as run:
        training_spec = combine_with_defaults(task, defaults)
        if "max_epochs" not in training_spec.trainer_args:
            raise Exception("Must specify max_epochs for the trainer")
        task = training_spec.task
        lightning_task_class = training_spec.task.type.get_class_from_enum()

        # if no optimization params, just run it
        if optimization_space is None:
            return (
                *fit_model(
                    training_spec,
                    lightning_task_class,
                    run.info.run_name,
                    experiment_name,
                    storage_uri,
                    run.info.run_id,
                    save_models=save_models,
                    test_models=test_models,
                ),
                {},
            )

        # if optimization parameters specified, do hyperparameter tuning
        study = optuna.create_study(
            sampler=sampler,
            direction=direction_type_to_optuna[
                training_spec.task.direction
            ],  # in the future may want to allow user to specify this
            pruner=HyperbandPruner(),
        )
        objective = partial(
            fit_model_with_hparams,
            training_spec,
            lightning_task_class,
            task.name,
            experiment_name,
            optimization_space,
            storage_uri,
            run.info.run_id,
            save_models,
            test_models,
        )
        study.optimize(
            objective,
            n_trials=n_trials,
            # callbacks=[champion_callback],
            catch=[torch.cuda.OutOfMemoryError],  # add a few more here?
        )
        best_params = unflatten(study.best_trial.params)
        mlflow.log_params(best_params) # unflatten
        mlflow.log_metric(f"best_{task.metric}", study.best_value)
        return study.best_value, task.metric, best_params


# Custom function to parse the optimization space argument
def parse_optimization_space(space: dict | None) -> optimization_space_type | None:
    if space is None:
        return None
    parsed_space: optimization_space_type = {}
    for key, value in space.items():
        if isinstance(value, dict):
            try:
                bounds = ParameterBounds(**value)
                parsed_space[key] = bounds
            except TypeError:
                # Recursively parse nested optimization spaces
                parsed_space[key] = parse_optimization_space(value)
        elif isinstance(value, list):
            # If it's a list, leave it as is
            parsed_space[key] = value
        else:
            raise ValueError(f"Invalid type for {key}: {value}")
    return parsed_space

def benchmark_backbone(
    defaults: Defaults,
    tasks: list[Task],
    experiment_name: str,
    storage_uri: str,
    ray_storage_path: str | None = None,
    backbone_import: str | None = None,
    run_name: str | None = None,
    n_trials: int = 1,
    optimization_space: dict | None = None,
    save_models: bool = False,
    run_id: str | None = None,
    description: str = "No description provided",
    bayesian_search: bool = True,
    test_models: bool = False,
):
    """Highest level function to benchmark a backbone using a single node

    Args:
        defaults (Defaults): Defaults that are set for all tasks
        tasks (list[Task]): List of Tasks to benchmark over. Will be combined with defaults to get the final parameters of the task.
        experiment_name (str): Name of the MLFlow experiment to be used.
        storage_uri (str): Path to MLFLow storage location.
        ray_storage_path (str | None): Ignored. Exists for compatibility with ray configs.
        backbone_import (str | None): Path to module that will be imported to register a potential new backbone. Defaults to None.
        run_name (str | None, optional): Name of highest level mlflow run. Defaults to None.
        n_trials (int, optional): Number of hyperparameter optimization trials to run. Defaults to 1.
        optimization_space (dict | None): Parameters to optimize over. Should be a dictionary (may be nested)
            of strings (parameter name) to list (discrete set of possibilities) or ParameterBounds, defining a range to optimize over. The structure should be the same as would be passed under tasks.terratorch_task. Defaults to None.
        save_models (bool, optional): Whether to save the model. Defaults to False.
        run_id (str | None): id of existing mlflow run to use as top-level run. Useful to add more experiments to a previous benchmark run. Defaults to None.
        description (str): Optional description for mlflow parent run.
        bayesian_search (bool): Whether to use bayesian optimization for the hyperparameter search. False uses random sampling. Defaults to True.
    """
    if backbone_import:
        importlib.import_module(backbone_import)
    mlflow.set_tracking_uri(storage_uri)
    mlflow.set_experiment(experiment_name)

    if bayesian_search:
        sampler: BaseSampler | None = None # take the default
    else:
        sampler = RandomSampler()

    optimization_space = parse_optimization_space(optimization_space)
    table_columns = ["Task", "Metric", "Best Score", "Hyperparameters"]
    table_entries = []
    with mlflow.start_run(run_name=run_name, run_id=run_id, description=description) as run:
        for task in tasks:
            best_value, metric_name, hparams = benchmark_backbone_on_task(
                defaults,
                task,
                storage_uri,
                experiment_name,
                optimization_space=optimization_space,
                n_trials=n_trials,
                save_models=save_models,
                sampler=sampler,
                test_models=test_models,
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
