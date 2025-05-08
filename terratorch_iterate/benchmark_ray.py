"""
This module contains the high level functions for benchmarking on a single node.
"""

import importlib
import os

import mlflow
import pandas as pd
import ray
from jsonargparse import CLI
from ray.tune.search import SearchAlgorithm, Searcher
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.search.optuna import OptunaSearch
from tabulate import tabulate

from benchmark.backbone_benchmark import parse_optimization_space
from benchmark.benchmark_types import (
    Defaults,
    Task,
    TrainingSpec,
    combine_with_defaults,
    optimization_space_type,
)
from benchmark.model_fitting import fit_model, ray_tune_model, valid_task_types


def benchmark_backbone_on_task(
    training_spec: TrainingSpec,
    lightning_task_class: valid_task_types,
    storage_uri: str,
    experiment_name: str,
    ray_storage_path: str,
    optimization_space: optimization_space_type | None = None,
    n_trials: int = 1,
    save_models: bool = False,
    backbone_import: str | None = None,
    searcher: SearchAlgorithm | None = None,
) -> dict:
    if not searcher:
        raise ValueError("Searcher must not be None")
    with mlflow.start_run(
        run_name=training_spec.task.name,
        nested=True,
    ) as run:
        # if no optimization params, just run it
        if optimization_space is None:
            raise Exception("For no optimization space, run benchmark.py")

        results = ray_tune_model(
            training_spec,
            lightning_task_class,
            optimization_space,
            storage_uri,
            ray_storage_path,
            experiment_name,
            save_models,
            n_trials,
            backbone_import=backbone_import,
            searcher=searcher,
        )

        mlflow.log_table(
            results.get_dataframe(),
            f"results_{run.info.run_name}.json",
            run.info.run_id,
        )
        if results.get_best_result().metrics is None:
            raise Exception("Best result metrics were none")
        if results.get_best_result().config is None:
            raise Exception("Best result config was none")

        mlflow.log_params(results.get_best_result().config)
        mlflow.log_metric(
            f"best_{training_spec.task.metric}",
            results.get_best_result().metrics[training_spec.task.metric],
        )
        return {
            "best_result": results.get_best_result().metrics[training_spec.task.metric],
            "metric": training_spec.task.metric,
            "best_config": results.get_best_result().config,
        }


@ray.remote(num_cpus=8, num_gpus=1)
def remote_fit(
    training_spec: TrainingSpec,
    lightning_task_class: valid_task_types,
    run_name: str,
    storage_uri: str,
    experiment_name: str,
    parent_run_id: str,
    save_models: bool,
    backbone_import: str | None,
) -> float:
    mlflow.set_tracking_uri(storage_uri)
    mlflow.set_experiment(experiment_name)
    if backbone_import:
        importlib.import_module(backbone_import)
    return fit_model(
        training_spec,
        lightning_task_class,
        run_name,
        experiment_name,
        storage_uri,
        parent_run_id,
        save_models=save_models,
    )[0]


def benchmark_backbone(
    defaults: Defaults,
    tasks: list[Task],
    experiment_name: str,
    storage_uri: str,
    tmp_dir: str | None = None,
    backbone_import: str | None = None,
    run_name: str | None = None,
    n_trials: int = 1,
    ray_storage_path: str | None = None,
    optimization_space: dict | None = None,
    save_models: bool = False,
    run_id: str | None = None,
    description: str = "No description provided",
    bayesian_search: bool = True,
):
    """Highest level function to benchmark a backbone using a ray cluster

    Args:
        tmp_dir (str): Path to temporary directory to be used for ray
        defaults (Defaults): Defaults that are set for all tasks
        tasks (list[Task]): List of Tasks to benchmark over. Will be combined with defaults to get the final parameters of the task.
        experiment_name (str): Name of the MLFlow experiment to be used.
        storage_uri (str): Path to MLFlow storage location.
        ray_storage_path (str | None): Path to storage of ray outputs, including saved models, when using ray tune. Required if optimization_space is specified
        backbone_import (str | None): Path to module that will be imported to register a potential new backbone. Defaults to None.
        run_name (str | None, optional): Name of highest level mlflow run. Defaults to None.
        n_trials (int, optional): Number of hyperparameter optimization trials to run. Defaults to 1.
        optimization_space (dict | None): Parameters to optimize over. Should be a dictionary (may be nested)
            of strings (parameter name) to list (discrete set of possibilities) or ParameterBounds, defining a range to optimize over. The structure should be the same as would be passed under tasks.terratorch_task. Defaults to None.
        save_models (bool, optional): Whether to save the models. Defaults to False.
        run_id (str | None): id of existing mlflow run to use as top-level run. Useful to add more experiments to a previous benchmark run. Defaults to None.
        description (str): Optional description for mlflow parent run.
        bayesian_search (bool): Whether to use bayesian optimization for the hyperparameter search. False uses random sampling. Defaults to True.
    """
    if tmp_dir is None:
        raise Exception("tmp_dir must be specified for runs with ray.")
    os.environ["RAY_TMPDIR"] = tmp_dir
    ray.init(_temp_dir=tmp_dir)
    if backbone_import:
        importlib.import_module(backbone_import)
    mlflow.set_tracking_uri(storage_uri)
    mlflow.set_experiment(experiment_name)
    # mlflow.pytorch.autolog(log_datasets=False)

    if bayesian_search:
        searcher: Searcher | SearchAlgorithm = OptunaSearch()
    else:
        searcher = BasicVariantGenerator()

    optimization_space = parse_optimization_space(optimization_space)

    table_columns = ["Task", "Metric", "Best Score", "Hyperparameters"]
    table_entries = []

    with mlflow.start_run(
        run_name=run_name, run_id=run_id, description=description
    ) as run:

        if optimization_space is None:
            # no hparams, parallelize over tasks
            ray_tasks = []
            for task in tasks:
                training_spec = combine_with_defaults(task, defaults)
                if "max_epochs" not in training_spec.trainer_args:
                    raise Exception("Must specify max_epochs for the trainer")
                task = training_spec.task
                lightning_task_class = training_spec.task.type.get_class_from_enum()
                ray_tasks.append(
                    remote_fit.remote(
                        training_spec,
                        lightning_task_class,
                        run.info.run_name,
                        storage_uri,
                        experiment_name,
                        run.info.run_id,
                        save_models,
                        backbone_import,
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
                training_spec = combine_with_defaults(task, defaults)
                if "max_epochs" not in training_spec.trainer_args:
                    raise Exception("Must specify max_epochs for the trainer")
                task = training_spec.task
                lightning_task_class = training_spec.task.type.get_class_from_enum()
                results.append(
                    benchmark_backbone_on_task(
                        training_spec,
                        lightning_task_class,
                        storage_uri,
                        experiment_name,
                        ray_storage_path,
                        optimization_space=optimization_space,
                        n_trials=n_trials,
                        save_models=save_models,
                        backbone_import=backbone_import,
                        searcher=searcher,
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
