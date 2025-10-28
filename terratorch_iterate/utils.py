import os
from typing import Any, Dict
import mlflow
import datetime
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from ast import literal_eval
import optuna
from terratorch_iterate.iterate_types import Task
from terratorch_iterate import plot_tools
import sys
from mlflow.entities.experiment import Experiment
import importlib
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from collections import defaultdict

N_TRIALS_DEFAULT = 16
REPEATED_SEEDS_DEFAULT = 10
DATA_PARTITIONS = {
    "default": 100,
    "1.00x_train": 100,
    "0.50x_train": 50,
    "0.20x_train": 20,
    "0.10x_train": 10,
    "0.01x_train": 1,
}


def unflatten(dictionary: Dict[str, Any]):
    resultDict: Dict = {}
    for key, value in dictionary.items():
        parts = key.split(".")
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return resultDict


def sync_mlflow_optuna(
    optuna_db_path: str,
    storage_uri: str,
    experiment_name: str,
    task_run_id: str | None,
    task: Task,
    n_trials: int,
    logger: logging.RootLogger,
) -> str | None:
    """
        syncs the number of completed trials in mflow and optuna
    Args:
        optuna_db_path: path to optuna database
        storage_uri: path to mlflow storage folder
        experiment_name: name on experiment in mlflow
        task_run_id: run_id of the task
        task: name of the task
        logger: logging.RootLogger to save logs to file
    Returns:
        task_run_id: run id of the task to be continued (if one exists) or None
    """
    logger.info(
        f"sync_mlflow_optuna - {optuna_db_path=} {storage_uri=} {task_run_id=} {experiment_name=} {task_run_id=}"
    )
    # check number of successful mlflow runs in task
    client = mlflow.tracking.MlflowClient(tracking_uri=storage_uri)
    completed_in_mlflow_for_task = []
    all_mlflow_runs_for_task = []
    if task_run_id is not None:
        all_mlflow_runs_for_task.append(task_run_id)
        logger.info(f"sync_mlflow_optuna - {task_run_id=}")
        experiment_info = client.get_experiment_by_name(experiment_name)
        assert isinstance(experiment_info, Experiment), (
            f"Error! Unexpected type of {experiment_info=}"
        )
        individual_run_data = client.search_runs(
            experiment_ids=[experiment_info.experiment_id],
            filter_string=f'tags."mlflow.parentRunId" LIKE "{task_run_id}"',
        )
        for individual_run in individual_run_data:
            if individual_run.info.status == "FINISHED":
                completed_in_mlflow_for_task.append(individual_run.info.run_id)
            all_mlflow_runs_for_task.append(individual_run.info.run_id)

    # check number of successful optuna trials in the database
    study_names = optuna.study.get_all_study_names(
        storage="sqlite:///{}.db".format(optuna_db_path)
    )
    if task.name in study_names:
        loaded_study = optuna.load_study(
            study_name=task.name, storage="sqlite:///{}.db".format(optuna_db_path)
        )
        logger.info(f"loaded_study has : {len(loaded_study.trials)} trials")
        incomplete = 0
        for trial in loaded_study.trials:
            if (trial.state == optuna.trial.TrialState.FAIL) | (
                trial.state == optuna.trial.TrialState.RUNNING
            ):
                incomplete += 1
        logger.info(f"{incomplete} trials are incomplete")
        successful_optuna_trials = len(loaded_study.trials) - incomplete
        too_many_trials = successful_optuna_trials > n_trials
        no_existing_task = task_run_id is None
        optuna_mlflow_mismatch = (
            len(completed_in_mlflow_for_task) != successful_optuna_trials
        )
        logger.info(
            f"successful optuna trials {successful_optuna_trials} . mlflow runs {len(completed_in_mlflow_for_task)}"
        )

        if too_many_trials or no_existing_task or optuna_mlflow_mismatch:
            logger.info(f"deleting study with name {task.name}")
            logger.info(f"too_many_trials {too_many_trials}")
            logger.info(f"no_existing_task {no_existing_task}")

            # delete optuna study in database
            optuna.delete_study(
                study_name=task.name, storage="sqlite:///{}.db".format(optuna_db_path)
            )

            # delete any existing mlflow runs
            if len(all_mlflow_runs_for_task) > 0:
                for item in all_mlflow_runs_for_task:
                    logger.info(f"deleting {item}")
                    client.delete_run(item)
                    assert isinstance(experiment_info, Experiment), (
                        f"Error! Unexpected type of {experiment_info=}"
                    )
                    os.system(f"rm -r {experiment_info.artifact_location}/{item}")
                    task_run_id = None
    else:
        # delete any existing mlflow runs
        if len(all_mlflow_runs_for_task) > 0:
            for item in all_mlflow_runs_for_task:
                logger.info(f"deleting {item}")
                client.delete_run(item)
                assert isinstance(experiment_info, Experiment), (
                    f"Error! Unexpected type of {experiment_info=}"
                )
                os.system(f"rm -r {experiment_info.artifact_location}/{item}")
            task_run_id = None
    logging.info(f"sync_mlflow_optuna returns {task_run_id=}")
    return task_run_id


def extract_repeated_experiment_results(
    storage_uri: str,
    logger: logging.RootLogger,
    experiments: list,
    task_metrics: list,
    task_names: list,
    num_repetitions: int = REPEATED_SEEDS_DEFAULT,
) -> (pd.DataFrame, list):
    """
    extracts results of repeated experiments from mlflow logs and saves them in csv
    save list of incomplete experiments to a txt file
    Args:
        storage_uri: path to mlflow storage folder
        logger: logging.RootLogger to save logs to file
        experiments: list of experiment names
        num_repetitions: number of repeated seeds per task
        task_names: list of tasks
    """
    if Path(storage_uri).exists() and Path(storage_uri).is_dir():
        storage_uri = Path(storage_uri)
        repeated_exp_storage_uri = storage_uri.with_name(
            f"{storage_uri.name}_repeated_exp"
        )
    else:
        print("Please use a valid directory for storage_uri")
        raise ValueError
    logger.info(
        f"\n Extracting results of repeated experiments from: {str(repeated_exp_storage_uri)}"
    )
    client = mlflow.tracking.MlflowClient(tracking_uri=str(repeated_exp_storage_uri))
    experiments = list(set(experiments))
    incomplete_experiments = []
    num_tasks = len(task_names)
    combine_exp_results = []

    task_info = dict(zip(task_names, task_metrics))

    for original_experiment_name in experiments:
        experiment_name = f"{original_experiment_name}_repeated_exp"
        logger.info(f"\nexperiment_name: {experiment_name}")
        experiment_info = client.get_experiment_by_name(experiment_name)
        if experiment_info is None:
            logger.info(
                f"EXPERIMENT {experiment_name} DOES NOT EXIST IN THIS FOLDER: {str(repeated_exp_storage_uri)}"
            )
            incomplete_experiments.append(experiment_name)
            continue
        experiment_id = experiment_info.experiment_id
        logger.info(f"experiment_id: {experiment_id}")
        logger.info(f"experiment_info: {experiment_info}")
        experiment_parent_run_data = client.search_runs(experiment_ids=[experiment_id])
        run_names = []
        run_ids = []
        run_seed = []
        run_task = []
        run_score = []
        run_metric = []
        run_status = []
        exp_ids = []
        exp_names = []
        logger.info(f"experiment_parent_run_data: {len(experiment_parent_run_data)}")
        for run in experiment_parent_run_data:
            run_name = run.info.run_name
            task = "_".join(run_name.split("_")[:-1])
            if (task in task_names) and (run.info.status == "FINISHED"):
                seed = int(run.info.run_name.split("_")[-1])
                if task in task_info:
                    metric_name = task_info[task]
                    metric_name = (
                        "test_test/" + metric_name.split("/")[-1]
                        if "/" in metric_name
                        else "test_test_"
                        + metric_name.replace(metric_name.split("_")[0] + "_", "")
                    )
                else:
                    continue

                if metric_name not in run.data.metrics:
                    logger.info(f"{metric_name} not found in task {task}. Skipping")
                    continue
                score = run.data.metrics[metric_name]
                run_names.append(run.info.run_name)
                exp_ids.append(experiment_id)
                exp_names.append(original_experiment_name)
                run_ids.append(run.info.run_id)
                run_status.append(run.info.status)
                run_seed.append(seed)
                run_metric.append(metric_name.split("/")[-1])
                run_task.append(task)
                run_score.append(score)

        df = pd.DataFrame(
            {
                "dataset": run_task,
                "Metric": run_metric,
                "test metric": run_score,
                "mlflow_run_name": run_names,
                "mlflow_run_id": run_ids,
                "mlflow_run_status": run_status,
                "Seed": run_seed,
                "experiment_id": exp_ids,
                "experiment_name": exp_names,
            }
        )
        if len(run_task) == 0:
            logger.info(
                f"EXPERIMENT INCOMPLETE: {experiment_name} has no complete tasks."
            )
            incomplete_experiments.append(experiment_name)
            continue
        print(f"\n\n\ndf: {df}")

        # get successful results per task
        combine_task_results = []
        for task in task_names:
            task_df = df.loc[
                (df["dataset"] == task) & (df["mlflow_run_status"] == "FINISHED")
            ].copy()
            task_df = task_df.loc[(task_df["test metric"] != 0.0)].copy()
            rows, _ = task_df.shape
            if (rows >= num_repetitions) and (
                sum(np.isnan(task_df["test metric"])) == 0
            ):
                task_df = task_df.iloc[list(range(num_repetitions))].copy()
                combine_task_results.append(task_df)
            elif rows < num_repetitions:
                logger.info(f"TASK INCOMPLETE: {task} only has {rows} seeds")
                incomplete_experiments.append(experiment_name)
        if len(combine_task_results) > 0:
            combine_task_results = pd.concat(combine_task_results, axis=0)
            combine_exp_results.append(combine_task_results)
        if len(combine_task_results) < num_tasks:
            logger.info(
                f"EXPERIMENT INCOMPLETE: {experiment_name} has {len(combine_task_results)} complete tasks only"
            )
            incomplete_experiments.append(experiment_name)
    if len(combine_exp_results) > 0:
        combine_exp_results = pd.concat(combine_exp_results, axis=0)
    else:
        combine_exp_results = pd.DataFrame()
    print(f"\n\n\ncombine_exp_results: {combine_exp_results}")
    return (combine_exp_results, incomplete_experiments)


def extract_parameters(
    storage_uri: str,
    logger: logging.RootLogger,
    experiments: list,
    task_names: list,
) -> pd.DataFrame:
    """
    extracts hyper-parameter information for each experiment from the mlflow logs
    saves this information to a csv file

    Args:
        storage_uri: path to mlflow storage folder used in configs
        logger: logging.RootLogger to save logs to file
        experiment_data: list of experiment names
        task_names: list of tasks
    """
    logger.info(f"\n Extracting parameters of experiments from: {storage_uri}")
    experiments = list(set(experiments))
    all_params = []
    client = mlflow.tracking.MlflowClient(tracking_uri=storage_uri)
    for experiment_name in experiments:
        # get experiment id
        experiment_info = client.get_experiment_by_name(experiment_name)
        if experiment_info is None:
            continue
        experiment_id = experiment_info.experiment_id
        logger.info(f"\nexperiment_name: {experiment_name} ")
        logger.info(f"experiment_id: {experiment_info.experiment_id}")
        exp_parent_run_name = f"top_run_{experiment_name}"
        experiment_parent_run_data = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f'tags."mlflow.runName" LIKE "{exp_parent_run_name}"',
        )
        if (len(experiment_parent_run_data) > 1) or (
            len(experiment_parent_run_data) == 0
        ):
            logger.debug(
                f"The number of parent runs for each experiment should be 1. \
                         It is currently {len(experiment_parent_run_data)}"
            )
            raise RuntimeError
        for run in experiment_parent_run_data:
            exp_parent_run_id = run.info.run_id

        mlflow.set_tracking_uri(storage_uri)
        mlflow.set_experiment(experiment_name)
        runs: list[mlflow.entities.Run] = mlflow.search_runs(
            filter_string=f"tags.mlflow.parentRunId='{exp_parent_run_id}'",
            output_format="list",
        )  # type: ignore
        logger.info(f"Found runs: {[run.info.run_name for run in runs]}")

        for task in task_names:
            logger.info(f"task: {task}")
            matching_runs = [run for run in runs if run.info.run_name.endswith(task)]  # type: ignore
            best_params = matching_runs[0].data.params

            # eval them
            best_params = {k: literal_eval(v) for k, v in best_params.items()}
            best_params["experiment_name"] = experiment_name
            best_params["dataset"] = task
            best_params["decoder"] = matching_runs[0].data.tags["decoder"]
            best_params["backbone"] = matching_runs[0].data.tags["backbone"]
            best_params["early_stop_patience"] = matching_runs[0].data.tags[
                "early_stop_patience"
            ]
            best_params["n_trials"] = matching_runs[0].data.tags["n_trials"]
            best_params["partition_name"] = matching_runs[0].data.tags["partition_name"]
            best_params["data_percentages"] = DATA_PARTITIONS[
                best_params["partition_name"]
            ]
            if "optimizer_hparams" in best_params:
                logger.info(
                    f"optimizer_hparams: {best_params['optimizer_hparams'].items()}"
                )
                optimizer_hparams = {
                    k: v for k, v in best_params["optimizer_hparams"].items()
                }
                best_params.update(optimizer_hparams)
                del best_params["optimizer_hparams"]
            if "model_args" in best_params:
                model_args = {k: v for k, v in best_params["model_args"].items()}
                best_params.update(model_args)
                del best_params["model_args"]

            best_params = pd.DataFrame(best_params, index=[0])
            all_params.append(best_params)
    all_params = pd.concat(all_params, axis=0)
    all_params = all_params.reset_index()
    return all_params


def get_results_and_parameters(
    benchmark_name: str,
    storage_uri: str,
    logger: logging.RootLogger,
    experiments: list,
    task_metrics: list,
    task_names: list,
    num_repetitions: int = REPEATED_SEEDS_DEFAULT,
    visualise: bool = True,
) -> pd.DataFrame:
    """
    extracts results and parameters for experiments from mlflow logs

    Args:
        benchmark_name: string to used to name resulting csv file
        storage_uri: path to mlflow storage folder used in configs
        logger: logging.RootLogger to save logs to file
        experiment_data: list of experiment names
        task_metrics: metrics used to evaluate each task
        task_names: list of tasks
        num_repetitions: number of repeated seeds per task
        visualise: whether to visualise the summarised results or not
    Returns:
        pd.DataFrame with results and parameters
    """
    if Path(storage_uri).exists() and Path(storage_uri).is_dir():
        results_dir = (
            Path(storage_uri).parents[0] / "summarized_results" / benchmark_name
        )
    else:
        print("Please use a valid directory for storage_uri")
        raise ValueError
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    parameters = extract_parameters(
        storage_uri=storage_uri,
        logger=logger,
        experiments=experiments,
        task_names=task_names,
    )

    # extract repeated experiment results from mlflow logs
    (results, incomplete_experiments) = extract_repeated_experiment_results(
        storage_uri=storage_uri,
        logger=logger,
        experiments=experiments,
        num_repetitions=num_repetitions,
        task_names=task_names,
        task_metrics=task_metrics,
    )

    with open(f"{results_dir}/incomplete_experiments.txt", "w") as f:
        for line in incomplete_experiments:
            f.write(f"{line}\n")
    results_and_parameters = results.merge(
        parameters, on=["experiment_name", "dataset"]
    )
    results_and_parameters.to_csv(
        f"{str(results_dir)}/results_and_parameters.csv", index=False
    )

    if visualise:
        visualize_combined_results(
            combined_results=results_and_parameters,
            storage_uri=storage_uri,
            logger=logger,
            plot_file_base_name="summary_plot",
        )

    return results_and_parameters


def delete_nested_experiment_parent_runs(
    logger: logging.RootLogger,
    delete_runs: list,
    experiment_info: mlflow.entities.experiment.Experiment,
    client: mlflow.tracking.client.MlflowClient,
    leave_one: bool = True,
) -> str | None:
    """
    if there are multiple runs for a single experiment,
    will delete all runs except the one with the most nested runs (most complete)
    Args:
        logger: logging.RootLogger to save logs to file
        delete_runs: list of runs to delete
        experiment_info: info of experiment
        client: mlflow client pointing to correct storage uri
        leave_one: if True, will not delete the most complete experiment. If False, will delete all experiments
    Returns:
        run id of the experiment run that was not deleted or None
    """
    experiment_id = experiment_info.experiment_id
    exp_parent_run_ids = []
    counts = []
    runs_in_experiment = []
    logger.info(f"Deleting from experiment_id:{experiment_id} ")
    logger.info(f"delete_runs:{delete_runs} ")

    for exp_parent_run_id in delete_runs:
        runs = []
        runs.append(exp_parent_run_id)
        task_parent_run_data = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f'tags."mlflow.parentRunId" LIKE "{exp_parent_run_id}"',
        )
        for task_parent_run in task_parent_run_data:
            task_parent_run_id = task_parent_run.info.run_id
            runs.append(task_parent_run_id)
            individual_run_data = client.search_runs(
                experiment_ids=[experiment_id],
                filter_string=f'tags."mlflow.parentRunId" LIKE "{task_parent_run_id}"',
            )
            for individual_run in individual_run_data:
                runs.append(individual_run.info.run_id)
        exp_parent_run_ids.append(exp_parent_run_id)
        counts.append(len(runs))
        runs_in_experiment.append(runs)

    if leave_one and (len(counts) > 0):
        index_to_keep = counts.index(max(counts))
        incomplete_run_to_finish = exp_parent_run_ids[index_to_keep]
        runs_in_experiment.pop(index_to_keep)
    else:
        incomplete_run_to_finish = None

    logger.info(f"Deleting runs:{runs_in_experiment} ")
    logger.info(
        f"experiment_info.artifact_location:{experiment_info.artifact_location}"
    )
    for runs in runs_in_experiment:
        for run_id in runs:
            client.delete_run(run_id)
            os.system(f"rm -r {experiment_info.artifact_location}/{run_id}")
    return incomplete_run_to_finish


def check_existing_task_parent_runs(
    logger: logging.RootLogger,
    exp_parent_run_id: str,
    storage_uri: str,
    experiment_name: str,
    n_trials: int = N_TRIALS_DEFAULT,
):
    """
    checks if tasks have been completed (both task run and nested individual runs are complete)
    Args:
        logger: logging.RootLogger to save logs to file
        exp_parent_run_id: run id of the experiment run being used (top level run id)
        storage_uri: folder containing mlflow log data
        experiment_name: name of experiment
        n_trials: number of trials (runs) expected in HPO of each task
    Returns:
        complete_task_run_names: list of task names that have been completed
        all_tasks_finished: bool showing if all tasks have been completed
        task_run_to_id_match: dict matching task names to the task run id

    """
    client = mlflow.tracking.MlflowClient(tracking_uri=storage_uri)
    experiment_info = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment_info.experiment_id
    task_parent_run_data = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f'tags."mlflow.parentRunId" LIKE "{exp_parent_run_id}"',
    )
    complete_task_run_names = []
    all_tasks_finished = []
    #   TO DO: make sure we only have one task_parent_run for each name (needed for repeated exps)
    task_run_to_id_match = {}
    for task_parent_run in task_parent_run_data:
        task_run_statuses = []
        task_run_ids = []
        task_run_statuses.append(task_parent_run.info.status)
        task_run_ids.append(task_parent_run.info.run_id)

        individual_run_data = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f'tags."mlflow.parentRunId" LIKE "{task_parent_run.info.run_id}"',
        )
        for individual_run in individual_run_data:
            if (individual_run.info.status == "RUNNING") or (
                individual_run.info.status == "FAILED"
            ):
                continue
            task_run_statuses.append(individual_run.info.status)
            task_run_ids.append(individual_run.info.run_id)

        task_run_to_id_match[task_parent_run.info.run_name] = (
            task_parent_run.info.run_id
        )
        task_run_statuses = list(set(task_run_statuses))

        condition_1 = len(task_run_statuses) == 1
        condition_2 = task_run_statuses[0] == "FINISHED"
        # condition_3 = len(task_run_ids) == (n_trials+1)
        if condition_1 and condition_2:  # and condition_3:
            complete_task_run_names.append(task_parent_run.info.run_name)
            task_parent_status = True
        else:
            task_parent_status = False
        all_tasks_finished.append(task_parent_status)

    if all(all_tasks_finished) and (len(all_tasks_finished) > 0):
        all_tasks_finished = True
    else:
        all_tasks_finished = False
    complete_task_run_names = list(set(complete_task_run_names))
    return complete_task_run_names, all_tasks_finished, task_run_to_id_match


def check_existing_experiments(
    logger: logging.RootLogger,
    storage_uri: str,
    experiment_name: str,
    exp_parent_run_name: str,
    task_names: list,
    n_trials: int,
    backbone: str,
) -> Dict[str, Any]:
    """
    checks if experiment has been completed (i.e. both task run and nested individual runs are complete)
    Args:
        logger: logging.RootLogger to save logs to file
        storage_uri: folder containing mlflow log data
        experiment_name: name of experiment
        exp_parent_run_name: run name of the top level experiment run
        task_names: list of task names that should be completed
        n_trials: number of trials (runs) expected in HPO of each task
    Returns:
        output: dict with:
            no_existing_runs: bool, if True, there are no existing runs
            incomplete_run_to_finish: str | None, run id of the experiment run to finish
            finished_run: str | None, run id of the finished experiment run
            experiment_id: str | None, experiment id it experiment already exists

    """
    client = mlflow.tracking.MlflowClient(tracking_uri=storage_uri)
    experiment_info = client.get_experiment_by_name(experiment_name)

    output = {
        "no_existing_runs": True,
        "incomplete_run_to_finish": None,
        "finished_run": None,
        "experiment_id": None,
    }
    if experiment_info is None:
        return output

    experiment_id = experiment_info.experiment_id
    logger.info(f"\nexperiment_id: {experiment_id}")
    logger.info(f"experiment_name: {experiment_name}")
    output["experiment_id"] = experiment_id
    experiment_parent_run_data = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f'tags."mlflow.runName" LIKE "{exp_parent_run_name}"',
    )
    if len(experiment_parent_run_data) >= 1:
        logger.info("there is at least one experiment parent run")
        finished_run_id = None
        incomplete_runs = []

        # check if one of the runs is complete
        for run in experiment_parent_run_data:
            (
                completed_task_run_names,
                all_tasks_in_experiment_finished,
                _,
            ) = check_existing_task_parent_runs(
                logger=logger,
                exp_parent_run_id=run.info.run_id,
                storage_uri=storage_uri,
                experiment_name=experiment_name,
                n_trials=n_trials,
            )
            logger.info(f"tasks that should be completed: {task_names}")
            logger.info(f"completed_task_run_names: {completed_task_run_names}")
            logger.info(
                f"all_tasks_in_experiment_finished: {all_tasks_in_experiment_finished}"
            )
            all_expected_tasks_completed = [
                item for item in task_names if item in completed_task_run_names
            ]
            all_expected_tasks_completed = len(task_names) == len(
                all_expected_tasks_completed
            )
            if all_expected_tasks_completed:
                finished_run_id = run.info.run_id
                logger.info(
                    f"The following run FINISHED and will be used for repeated experiments: {finished_run_id}"
                )
            else:
                incomplete_tasks = [
                    item for item in task_names if item not in completed_task_run_names
                ]
                logger.info(
                    f"The following run {run.info.run_id} is incomplete, with status {run.info.status} and missing tasks: {incomplete_tasks}"
                )
                incomplete_runs.append(run.info.run_id)

        if finished_run_id is not None:
            # delete all incomplete runs
            delete_nested_experiment_parent_runs(
                logger=logger,
                delete_runs=incomplete_runs,
                experiment_info=experiment_info,
                client=client,
                leave_one=False,
            )
            output["finished_run"] = finished_run_id
            output["no_existing_runs"] = False
        else:
            # delete all incomplete runs, leave one
            logger.info(f"incomplete_runs: {incomplete_runs}")
            output["incomplete_run_to_finish"] = delete_nested_experiment_parent_runs(
                logger=logger,
                delete_runs=incomplete_runs,
                experiment_info=experiment_info,
                client=client,
                leave_one=True,
            )
            output["no_existing_runs"] = False
    return output


def visualize_combined_results(
    combined_results: pd.DataFrame,
    storage_uri: str,
    logger: logging.RootLogger,
    plot_file_base_name: str,
):
    """
    compiles and visualizes results from experiment
    Args:
        combined_results: table containing results and parameters for all experiments
        storage_uri: storage_uri from config
        logger: logging.RootLogger to save logs to file
        plot_file_base_name: unique string to be added to all file names
    """
    logger.info("Starting to visualize")
    save_folder = Path(storage_uri).parents[0] / "visualizations"
    tables_folder = save_folder / "tables"
    plots_folder = save_folder / "plots"
    if not os.path.exists(tables_folder):
        os.makedirs(tables_folder)
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    model_order = []
    experiments = list(set(combined_results["experiment_name"]))
    combined_results = combined_results.rename(columns={"experiment_name": "model"})
    num_experiments = len(experiments)
    fig_size = (num_experiments * 5, 6) if num_experiments >= 3 else (15, 6)
    n_legend_rows = num_experiments // 3 if num_experiments >= 3 else 1
    model_order = sorted(experiments)
    model_colors = dict(
        zip(model_order, sns.color_palette("tab20", n_colors=len(model_order)))
    )

    plot_tools.plot_per_dataset(
        combined_results,
        model_order=model_order,
        aggregated_name=plot_file_base_name,
        model_colors=model_colors,
        metric="test metric",
        sharey=False,
        inner="points",
        fig_size=fig_size,
        n_legend_rows=n_legend_rows,
    )
    plt.savefig(
        str(f"{plots_folder}/violin_{plot_file_base_name}_raw.png"),
        bbox_inches="tight",
    )
    plt.close()

    # plot normalized, bootstrapped values values
    plot_tools.make_normalizer(
        combined_results,
        metrics=("test metric",),
        benchmark_name=plots_folder,
    )

    plot_tools.normalize_bootstrap_and_plot(
        combined_results,
        # plot_file_base_name=plot_file_base_name,
        metric="test metric",
        benchmark_name=plots_folder,
        model_order=model_order,
        model_colors=model_colors,
        fig_size=fig_size,
        n_legend_rows=n_legend_rows,
    )

    plt.savefig(
        str(f"{plots_folder}/violin_{plot_file_base_name}_normalized_bootstrapped.png"),
        bbox_inches="tight",
    )
    plt.close()

    combined_results.to_csv(
        str(f"{tables_folder}/{plot_file_base_name}_normalized_combined_results.csv")
    )


def get_logger(log_level="INFO", log_folder="./experiment_logs") -> logging.RootLogger:
    # set up logging file
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    current_time = datetime.datetime.now()
    current_time = (
        str(current_time).replace(" ", "_").replace(":", "-").replace(".", "-")
    )
    log_file = f"{log_folder}/{current_time}"
    logger = logging.getLogger()
    logger.setLevel(log_level)
    handler = logging.FileHandler(log_file)
    handler.setLevel(log_level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logging.basicConfig(level=logging.CRITICAL)
    return logger


def import_custom_modules(
    logger: logging.RootLogger,
    custom_modules_path: str | Path | None = None,
) -> None:
    if custom_modules_path:
        custom_modules_path = Path(custom_modules_path)

        if custom_modules_path.is_dir():
            # Add 'custom_modules' folder to sys.path
            workdir = custom_modules_path.parents[0]
            module_dir = custom_modules_path.name

            sys.path.insert(0, str(workdir))

            try:
                importlib.import_module(module_dir)
                logger.info(f"Found {custom_modules_path}")
            except ImportError:
                raise ImportError(
                    f"It was not possible to import modules from {custom_modules_path}."
                )
        else:
            raise ValueError(
                f"Modules path {custom_modules_path} isn't a directory. Check if you have defined it properly."
            )
    else:
        logger.debug("No custom module is being used.")


if __name__ == "__main__":
    logger = get_logger()
    storage_uri = "results_folder/hpo"  # storage_uri from config

    list_of_experiments = [
        "early_stopping_10_prithvi_600",
        "early_stopping_10_prithvi_600_tl",
        "early_stopping_10_dofa_vit_300",
    ]
    # get results and parameters from mlflow logs
    results_and_parameters = get_results_and_parameters(
        storage_uri=storage_uri,
        logger=logger,
        experiments=list_of_experiments,
    )

    settings_per_model = [
        "detection",
    ]

    # create box plots across multiple models
    for setting in settings_per_model:
        combined_results = results_and_parameters.loc[
            results_and_parameters["experiment_name"].str.contains(setting)
        ].copy()
        model_order = visualize_combined_results(
            combined_results=results_and_parameters,
            storage_uri=storage_uri,
            logger=logger,
            plot_file_base_name=f"multiple_models_{setting}",
        )


### code written with the help of Perplexity platform
def get_nested_runs(experiment_id, filter_string=None, mlflow_uri="mlflow"):
    client = MlflowClient(mlflow_uri)

    # Get all runs for the experiment
    all_runs = client.search_runs(
        experiment_ids=[experiment_id], run_view_type=ViewType.ACTIVE_ONLY
    )

    # Create a dictionary to store the run hierarchy
    run_hierarchy = defaultdict(list)
    parent_runs = []

    # First pass: Identify parent-child relationships
    for run in all_runs:
        parent_run_id = run.data.tags.get("mlflow.parentRunId")

        if parent_run_id:
            run_hierarchy[parent_run_id].append(run)
        else:
            parent_runs.append(run)

    # Function to create a nested dictionary for a run and its children
    def create_nested_dict(run):
        run_dict = {
            "run": run,
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName", "Unnamed"),
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "children": [
                create_nested_dict(child) for child in run_hierarchy[run.info.run_id]
            ],
        }
        return run_dict

    # Create the final nested structure
    if filter_string:
        nested_runs = [
            create_nested_dict(parent_run)
            for parent_run in parent_runs
            if parent_run.data.tags.get("mlflow.runName", "Unnamed").find(filter_string)
            > -1
        ]
    else:
        nested_runs = [create_nested_dict(parent_run) for parent_run in parent_runs]

    return nested_runs
