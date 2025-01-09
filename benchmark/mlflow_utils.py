import os
import mlflow
from pathlib import Path
import datetime
import yaml
import logging
from typing import Any


def delete_nested_experiment_parent_runs(
        logger,
        delete_runs: list, 
        experiment_id, 
        client, 
        leave_one: bool):
    """
        if there are moutliple runs for a single experiment, 
        deletes all runs except the one with the most nested runs (most complete)
        Args:
            logger:
            delete_runs: list of runs to delete
            experiment_id: id of experiment to check
            client: mlflow client pointing to correct storage uri
            leave_one: if True, will not delete the most complete experiment. If False, will delete all experiments
        Returns:
            run id of the experiment run that was not deleted

    """
    experiment_ids = []
    counts = []
    runs_in_experiment = []
    for exp_parent_run_id in delete_runs:
        runs = []
        runs.append(exp_parent_run_id)
        task_parent_run_data = client.search_runs(experiment_ids=[experiment_id], 
                        filter_string=f'tags."mlflow.parentRunId" LIKE "{exp_parent_run_id}"')
        for task_parent_run in task_parent_run_data:
            task_parent_run_id = task_parent_run.info.run_id
            runs.append(task_parent_run_id)
            individual_run_data = client.search_runs(experiment_ids=[experiment_id], 
                        filter_string=f'tags."mlflow.parentRunId" LIKE "{task_parent_run_id}"')
            for individual_run in individual_run_data:
                individual_run_id = individual_run.info.run_id
                runs.append(individual_run_id)
        logger.info(f"{exp_parent_run_id}: {len(runs)}")
        experiment_ids.append(exp_parent_run_id)
        counts.append(len(runs))
        runs_in_experiment.append(runs)

    logger.info(f"experiment_ids: {experiment_ids}")
    logger.info(f"number of nested runs in each experiment run: {counts}")
    logger.info(f"runs_in_experiment: {len(runs_in_experiment)}")
    if leave_one and (len(counts) >0):
        index_to_keep = counts.index(max(counts))
        incomplete_run_to_finish = experiment_ids[index_to_keep]
        runs_in_experiment.pop(index_to_keep)
    else:
        incomplete_run_to_finish = None

    for runs in runs_in_experiment:
        for run_id in runs:
            client.delete_run(run_id)
            os.system(f"rm -r {experiment_info.artifact_location}/{run_id}")
    
    return incomplete_run_to_finish




def check_existing_task_parent_runs(
        logger,
        exp_parent_run_id, 
        storage_uri, 
        experiment_name, 
        n_trials):
    """
        checks if tasks have been completed (both task run and nested individual runs are complete)
        Args:
            logger:
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

    task_parent_run_data = client.search_runs(experiment_ids=[experiment_id], 
                        filter_string=f'tags."mlflow.parentRunId" LIKE "{exp_parent_run_id}"')
    runs_to_delete = []
    complete_task_run_names = []
    all_tasks_finished = []
    #   TO DO: make sure we only have one task_parent_run for each name (needed for repeated exps)
    task_run_to_id_match = {}
    for task_parent_run in task_parent_run_data:
        task_run_statuses = []
        task_run_ids = []
        
        task_run_statuses.append(task_parent_run.info.status)
        task_run_ids.append(task_parent_run.info.run_id)
        individual_run_data = client.search_runs(experiment_ids=[experiment_id], 
                    filter_string=f'tags."mlflow.parentRunId" LIKE "{task_parent_run.info.run_id}"')
        for individual_run in individual_run_data:
            if (individual_run.info.status == "RUNNING") or (individual_run.info.status == "FAILED"):
                continue
            task_run_statuses.append(individual_run.info.status)
            task_run_ids.append(individual_run.info.run_id)

        task_run_to_id_match[task_parent_run.info.run_name] = task_parent_run.info.run_id

        task_run_statuses = list(set(task_run_statuses))

        condition_1 = len(task_run_statuses) == 1
        condition_2 = task_run_statuses[0]=="FINISHED"
        condition_3 = len(task_run_ids) == (n_trials+1) 
        if condition_1 and condition_2 and condition_3:
            complete_task_run_names.append(task_parent_run.info.run_name)
            all_tasks_finished.append(True)
        else:
            all_tasks_finished.append(False)

    if all(all_tasks_finished) and (len(all_tasks_finished) > 0) :
        all_tasks_finished = True
    else:
        all_tasks_finished = False

    complete_task_run_names = list(set(complete_task_run_names))
    return complete_task_run_names, all_tasks_finished, task_run_to_id_match



def check_existing_experiments(logger,
                            storage_uri: str, 
                            experiment_name: str, 
                            exp_parent_run_name: str,
                            backbone: str,
                            task_names: list,
                            n_trials: int):
    """
        checks if tasks have been completed (both task run and nested individual runs are complete)
        Args:
            logger:
            storage_uri: folder containing mlflow log data
            experiment_name: name of experiment
            exp_parent_run_name: run name of the top level experiment run
            backbone: name of backbone being used in experiment
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

    output = {"no_existing_runs": True,
            "incomplete_run_to_finish":None,
            "finished_run":None,
            "experiment_id":None}
    if experiment_info is None:
        return output

    experiment_id = experiment_info.experiment_id
    logger.info(f"\n\n\nexperiment_id: {experiment_id}")
    logger.info(f"experiment_name: {experiment_name}")
    output["experiment_id"] = experiment_id
    experiment_parent_run_data = client.search_runs(experiment_ids=[experiment_id], 
                            filter_string=f'tags."mlflow.runName" LIKE "{exp_parent_run_name}"')
    #logger.info(f"experiment_parent_run_data: {experiment_parent_run_data}")
    if len(experiment_parent_run_data)>=1:
        logger.info("there is at least one experiment parent run")
        finished_run_id = None
        incomplete_runs = []

        #check if one of the runs is complete
        for run in experiment_parent_run_data:
            completed_task_run_names, all_tasks_in_experiment_finished, _ = check_existing_task_parent_runs(
                                                                                    logger,
                                                                                    run.info.run_id, 
                                                                                    storage_uri, 
                                                                                    experiment_name, 
                                                                                    n_trials)
            logger.info(f"tasks that should be completed: {task_names}")
            logger.info(f"completed_task_run_names: {completed_task_run_names}")
            logger.info(f"all_tasks_in_experiment_finished: {all_tasks_in_experiment_finished}")
            all_expected_tasks_completed = [item for item in task_names if item in completed_task_run_names]
            all_expected_tasks_completed = len(task_names) == len(all_expected_tasks_completed)
            #all_expected_tasks_completed =  all(x == y for x, y in zip(sorted(task_names), sorted(completed_task_run_names)))
            #same_num_tasks = len(task_names) == len(completed_task_run_names)
            if all_expected_tasks_completed:# and all_tasks_in_experiment_finished and same_num_tasks:
                finished_run_id = run.info.run_id
                logger.info(f"The following run FINISHED and will be used for repeated experiments: {finished_run_id}")
            else:
                incomplete_tasks = [item for item in task_names if item not in completed_task_run_names]
                logger.info(f"The following run {run.info.run_id} is incomplete, with status {run.info.status} and missing tasks: {incomplete_tasks}")
                incomplete_runs.append(run.info.run_id)

        
        if finished_run_id is not None:
            #delete all incomplete runs
            delete_nested_experiment_parent_runs(logger,
                                                incomplete_runs, 
                                                experiment_id=experiment_id, 
                                                client=client, 
                                                leave_one=False)
            output["finished_run"] = finished_run_id
            output["no_existing_runs"] = False
        else:
            #delete all incomplete runs, leave one
            logger.info(f"incomplete_runs: {incomplete_runs}")
            output["incomplete_run_to_finish"] = delete_nested_experiment_parent_runs(logger,
                                                                                    incomplete_runs, 
                                                                                    experiment_id=experiment_id, 
                                                                                    client=client, 
                                                                                    leave_one=True)
            output["no_existing_runs"] = False
    return output





def get_logger(log_level="INFO",
               log_folder="./experiment_logs"):
    #set up logging file
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    current_time = datetime.datetime.now()
    current_time = str(current_time).replace(" ", "_").replace(":", "-").replace(".", "-")
    log_file = f"{log_folder}/{current_time}"
    logger = logging.getLogger()
    logger.setLevel(log_level)
    handler = logging.FileHandler(log_file)
    #handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logging.basicConfig(level=logging.CRITICAL)
    return logger
