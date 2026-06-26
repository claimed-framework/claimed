"""
This module contains the high level functions for benchmarking on a single node.
"""

# import argparse
import os
import importlib
from functools import partial
from typing import Any, Dict
from pathlib import Path
import mlflow
import optuna
import pandas as pd
import torch
import logging
from optuna.pruners import HyperbandPruner
from optuna.samplers import BaseSampler, RandomSampler
from tabulate import tabulate
import pickle
from terratorch_iterate.iterate_types import (
    Defaults,
    ParameterBounds,
    Task,
    combine_with_defaults,
    optimization_space_type,
)
from terratorch_iterate.model_fitting import fit_model, fit_model_with_hparams
from terratorch_iterate.repeat_best_experiment import rerun_best_from_backbone
from terratorch_iterate.utils import (
    check_existing_task_parent_runs,
    check_existing_experiments,
    unflatten,
    get_logger,
    sync_mlflow_optuna,
    REPEATED_SEEDS_DEFAULT,
)

direction_type_to_optuna = {"min": "minimize", "max": "maximize"}


def benchmark_backbone_on_task(
    logger: logging.RootLogger,
    defaults: Defaults,
    task: Task,
    storage_uri: str,
    experiment_name: str,
    experiment_run_id: str,
    task_run_id: str | None = None,
    optimization_space: optimization_space_type | None = None,
    n_trials: int = 1,
    save_models: bool = False,
    sampler: BaseSampler | None = None,
    test_models: bool = False,
) -> tuple[float, str | list[str] | None, dict[str, Any]]:
    logger.info(
        f"starting backbone benchmark on task {task.name} {task_run_id=} {experiment_name=}"
    )
    if storage_uri.startswith("http"):
        optuna_db_path = Path(".") / "optuna_db"
    else:
        optuna_db_path = Path(storage_uri).parents[0] / "optuna_db"

    if not os.path.exists(optuna_db_path):
        os.makedirs(optuna_db_path)
    optuna_db_path = optuna_db_path / f"{experiment_name}_{experiment_run_id}"
    optuna_db_path = str(optuna_db_path)

    task_run_id = sync_mlflow_optuna(
        optuna_db_path=optuna_db_path,
        storage_uri=storage_uri,
        experiment_name=experiment_name,
        task_run_id=task_run_id,
        task=task,
        n_trials=n_trials,
        logger=logger,
    )
    if task_run_id is not None:
        # run_name is used only when run_id is unspecified.
        run_name = None
    else:
        run_name = task.name
    logger.info(f"start run: {run_name=} {task_run_id=}")
    with mlflow.start_run(run_name=run_name, nested=True, run_id=task_run_id) as run:
        logger.info(f"starting task run with id: {run.info.run_id}")
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
            study_name=task.name,
            storage="sqlite:///{}.db".format(optuna_db_path),
            load_if_exists=True,
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

        n_trials = n_trials - len(study.trials)
        for trial in study.trials:
            if (trial.state == optuna.trial.TrialState.FAIL) | (
                trial.state == optuna.trial.TrialState.RUNNING
            ):
                n_trials = n_trials + 1

        study.optimize(
            objective,
            n_trials=n_trials,
            # callbacks=[champion_callback],
            catch=[torch.cuda.OutOfMemoryError],
        )

        tags = {
            "early_stop_patience": str(training_spec.task.early_stop_patience),
            "partition_name": (
                str(training_spec.task.datamodule.partition)
                if hasattr(training_spec.task.datamodule, "partition")
                else "default"
            ),
            "decoder": (
                str(training_spec.task.terratorch_task["model_args"]["decoder"])
                if "decoder" in training_spec.task.terratorch_task["model_args"]
                else training_spec.task.terratorch_task["model_args"]["framework"]
            ),
            "task": str(training_spec.task.type).split(".")[-1],
            "backbone": str(
                training_spec.task.terratorch_task["model_args"]["backbone"]
            ),
            "n_trials": str(n_trials),
        }
        mlflow.set_tags(tags)

        best_params = unflatten(study.best_trial.params)
        mlflow.log_params(best_params)  # unflatten
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


def _run_hpo(
    run_name: str | None,
    run_id: str | None,
    description: str,
    tasks: list,
    completed_task_run_names: list,
    task_run_to_id_match: dict,
    defaults,
    storage_uri: str,
    experiment_name: str,
    optimization_space,
    n_trials,
    save_models,
    sampler,
    test_models,
    table_entries,
    table_columns,
    backbone,
    task_names,
    PATH_TO_JOB_TRACKING,
    logger,
) -> tuple[str, str]:
    logger.info(
        f"Running hyperparameter optimization: {run_name=} {run_id=} {description=}"
    )
    if run_id is not None:
        run_name = None

    with mlflow.start_run(
        run_name=run_name, run_id=run_id, description=description
    ) as run:
        for task in tasks:
            # only run task if it was not completed before
            task_run_name = task.name
            if task_run_name in completed_task_run_names:
                logger.info(f"{task_run_name} already completed")
                continue
            else:
                logger.info(f"{task_run_name} not completed. starting now")

            task_run_id = (
                task_run_to_id_match[task_run_name]
                if task_run_name in task_run_to_id_match
                else None
            )
            best_value, metric_name, hparams = benchmark_backbone_on_task(
                logger,
                defaults,
                task,
                storage_uri,
                experiment_name,
                experiment_run_id=run.info.run_id,
                task_run_id=task_run_id,
                optimization_space=optimization_space,
                n_trials=n_trials,
                save_models=save_models,
                sampler=sampler,
                test_models=test_models,
            )
            table_entries.append([task.name, metric_name, best_value, hparams])
            table_entries_filename = str(
                PATH_TO_JOB_TRACKING
                / f"{experiment_name}-{run.info.run_id}_table_entries.pkl"
            )
            with open(table_entries_filename, "wb") as handle:
                pickle.dump(table_entries, handle, protocol=pickle.HIGHEST_PROTOCOL)

        table = tabulate(table_entries, headers=table_columns)
        logger.info(table)
        df = pd.DataFrame(data=table_entries, columns=table_columns)
        df.set_index("Task")
        logger.info("Starting to save results")
        mlflow.log_table(
            df,
            "results_table.json",
            run.info.run_id,
        )
        experiment_id = run.info.experiment_id

        # check completion of HPO for all tasks before proceeding to next stage
        existing_experiments = check_existing_experiments(
            logger=logger,
            storage_uri=storage_uri,
            experiment_name=experiment_name,
            exp_parent_run_name=run_name,
            task_names=task_names,
            n_trials=n_trials,
            backbone=backbone,
        )
        if existing_experiments["finished_run"] is not None:
            finished_run_id = existing_experiments["finished_run"]
        else:
            logger.info("HPO is not complete. Please re-run this experiment")
            raise RuntimeError

        return experiment_id, finished_run_id


def benchmark_backbone(
    defaults: Defaults,
    tasks: list[Task],
    experiment_name: str,
    storage_uri: str,
    logger: logging.RootLogger | None,
    ray_storage_path: str | None = None,
    backbone_import: str | None = None,
    run_name: str | None = None,
    n_trials: int = 1,
    optimization_space: dict | None = None,
    save_models: bool = False,
    run_id: str | None = None,
    description: str = "No description provided",
    bayesian_search: bool = True,
    continue_existing_experiment: bool = True,
    test_models: bool = False,
    run_repetitions: int = REPEATED_SEEDS_DEFAULT,
    report_on_best_val: bool = True,
) -> Dict[str, str]:
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
        run_repetitions (int): Number of times that the experiment will be repeated. Defaults to 1.
    """
    base = Path(storage_uri).parents[0]
    PATH_TO_JOB_TRACKING = base / "job_progress_tracking"
    REPEATED_EXP_FOLDER = base / "repeated_exp_output_csv"

    # https://mlflow.org/docs/latest/ml/tracking/system-metrics/#using-the-environment-variable-to-control-system-metrics-logging
    if os.getenv("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING") is None:
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

    if logger is None:
        logger = get_logger(log_folder=str(base / "job_logs"))

    if not os.path.exists(REPEATED_EXP_FOLDER):
        os.makedirs(REPEATED_EXP_FOLDER)
    if not os.path.exists(PATH_TO_JOB_TRACKING):
        os.makedirs(PATH_TO_JOB_TRACKING)

    if backbone_import:
        importlib.import_module(backbone_import)
    logger.info(f"Setting tracking URI: {storage_uri}")
    mlflow.set_tracking_uri(storage_uri)
    logger.info(f"Setting experiment name: {experiment_name}")
    mlflow.set_experiment(experiment_name)
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    optimization_space = parse_optimization_space(optimization_space)

    backbone: str = defaults.terratorch_task["model_args"]["backbone"]
    task_names = [task.name for task in tasks]
    run_name = f"top_run_{experiment_name}" if run_name is None else run_name

    completed_task_run_names = []
    optimize_hyperparams = True
    task_run_to_id_match = {}
    if continue_existing_experiment:
        # find status of existing runs, and delete incomplete runs except one with the most complete tasks
        existing_experiments = check_existing_experiments(
            logger=logger,
            storage_uri=storage_uri,
            experiment_name=experiment_name,
            exp_parent_run_name=run_name,
            task_names=task_names,
            n_trials=n_trials,
            backbone=backbone,
        )
        if existing_experiments["no_existing_runs"]:
            logger.info("\nStarting new experiment from scratch")
        else:
            if (existing_experiments["incomplete_run_to_finish"] is not None) and (
                run_id is None
            ):
                logger.info("Continuing previous experiment parent run")
                run_id = existing_experiments["incomplete_run_to_finish"]
                logger.debug(f"incomplete_run_to_finish: {run_id=}")
                experiment_id = existing_experiments["experiment_id"]
                optimize_hyperparams = True

            if existing_experiments["finished_run"] is not None:
                optimize_hyperparams = False
                finished_run_id = existing_experiments["finished_run"]
                logger.debug(f"finished_run: {run_id=}")
                run_id = existing_experiments["finished_run"]

            # get previously completed tasks
            completed_task_run_names, _, task_run_to_id_match = (
                check_existing_task_parent_runs(
                    logger, run_id, storage_uri, experiment_name, n_trials
                )
            )

            table_entries_filename = str(
                PATH_TO_JOB_TRACKING / f"{experiment_name}-{run_id}_table_entries.pkl"
            )
            if os.path.exists(table_entries_filename):
                with open(table_entries_filename, "rb") as handle:
                    table_entries = pickle.load(handle)
    else:
        logger.info("Starting new experiment from scratch")

    # only run hyperparameter optimization (HPO) if there are no experiments with finished HPO
    if optimize_hyperparams:
        if bayesian_search:
            sampler: BaseSampler | None = None  # take the default
        else:
            sampler = RandomSampler()
        table_columns = ["Task", "Metric", "Best Score", "Hyperparameters"]
        table_entries = []
        experiment_id, finished_run_id = _run_hpo(
            run_name=run_name,
            run_id=run_id,
            description=description,
            tasks=tasks,
            task_names=task_names,
            completed_task_run_names=completed_task_run_names,
            task_run_to_id_match=task_run_to_id_match,
            defaults=defaults,
            storage_uri=storage_uri,
            experiment_name=experiment_name,
            n_trials=n_trials,
            save_models=save_models,
            sampler=sampler,
            test_models=test_models,
            table_entries=table_entries,
            table_columns=table_columns,
            backbone=backbone,
            PATH_TO_JOB_TRACKING=PATH_TO_JOB_TRACKING,
            optimization_space=optimization_space,
            logger=logger,
        )
        logger.info("HPO complete")

    if run_repetitions >= 1:
        logger.info(f"run_repetitions: {run_repetitions}")
        # run repeated experiments
        logger.info(
            f"Now running {run_repetitions} repeats per experiment \n\
                    Parent run: {finished_run_id} \n\
                    Experiment name: {experiment_name} \n\
                    "
        )
        path_to_final_results = str(
            REPEATED_EXP_FOLDER / f"{experiment_name}_repeated_exp_mlflow.csv"
        )

        rerun_best_from_backbone(
            logger=logger,
            parent_run_id=finished_run_id,
            output_path=path_to_final_results,
            defaults=defaults,
            tasks=tasks,
            experiment_name=experiment_name,
            storage_uri=storage_uri,
            tmp_dir=ray_storage_path,
            backbone_import=backbone_import,
            run_name=run_name,
            n_trials=n_trials,
            ray_storage_path=ray_storage_path,
            optimization_space=optimization_space,
            save_models=save_models,
            description=description,
            use_ray=False,
            run_repetitions=run_repetitions,
            report_on_best_val=report_on_best_val,
        )

    return {"experiment_id": experiment_id, "finished_run_id": finished_run_id}
