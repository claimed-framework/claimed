import os
import mlflow
from pathlib import Path
import datetime
import yaml
import logging
from typing import Any
import plot_tools
import pandas as pd
import numpy as np
import glob
from pathlib import Path
import datetime
import yaml
import seaborn as sns
from matplotlib import pyplot as plt
from ast import literal_eval
import optuna
from benchmark.benchmark_types import Task

SEGMENTATION_BASE_TASKS = ['chesapeake', 'sa_crop_type', 'pv4ger_seg', 'cashew', 'neontree', 'nz_cattle']
CLASSIFICATION_BASE_TASKS = ['pv4ger', 'so2sat', 'brick_kiln', 'big_earth_net', 'eurosat', 'forestnet']




def sync_mlflow_optuna(optuna_db_path: str,
                       storage_uri: str,
                       experiment_name: str,
                       task_run_id: str | None,
                       task: Task,
                       logger,
                       ) -> str | None:
    """
        syncs the number of completed trials in mflow and optuna
    Returns:
        task_run_id: run id of the task to be continued (if one exists) or None
    """
    #check number of successful mlflow runs in task
    client = mlflow.tracking.MlflowClient(tracking_uri=storage_uri)
    completed_in_mlflow_for_task = []
    all_mlflow_runs_for_task = []
    if task_run_id is not None:
        all_mlflow_runs_for_task.append(task_run_id)
        logger.info(f"task_run_id : {task_run_id}")
        experiment_info = client.get_experiment_by_name(experiment_name)
        individual_run_data = client.search_runs(experiment_ids=[experiment_info.experiment_id], 
                    filter_string=f'tags."mlflow.parentRunId" LIKE "{task_run_id}"')
        for individual_run in individual_run_data:
            if individual_run.info.status == "FINISHED":
                completed_in_mlflow_for_task.append(individual_run.info.run_id)
            all_mlflow_runs_for_task.append(individual_run.info.run_id)

    #check number of successful optuna trials in the database
    study_names = optuna.study.get_all_study_names(storage= "sqlite:///{}.db".format(optuna_db_path))
    if task.name in study_names:
        loaded_study = optuna.load_study(study_name=task.name, 
                                        storage= "sqlite:///{}.db".format(optuna_db_path))
        logger.info(f"loaded_study has : {len(loaded_study.trials)} trials")
        incomplete = 0
        for trial in loaded_study.trials:
            if (trial.state == optuna.trial.TrialState.FAIL) | (trial.state == optuna.trial.TrialState.RUNNING):
                incomplete+=1
        logger.info(f"{incomplete} trials are incomplete")
        successful_optuna_trials = len(loaded_study.trials) - incomplete
        too_many_trials = successful_optuna_trials > n_trials
        no_existing_task =  task_run_id is None
        optuna_mlflow_mismatch = len(completed_in_mlflow_for_task) != successful_optuna_trials
        logger.info(f"successful optuna trials {successful_optuna_trials} . mlflow runs {len(completed_in_mlflow_for_task)}")

        if too_many_trials or no_existing_task or optuna_mlflow_mismatch:
            logger.info(f"deleting study with name {task.name}")
            logger.info(f"too_many_trials {too_many_trials}")
            logger.info(f"no_existing_task {no_existing_task}")

            #delete optuna study in database
            optuna.delete_study(study_name=task.name, 
                                storage= "sqlite:///{}.db".format(optuna_db_path))

            #delete any existing mlflow runs
            if len(all_mlflow_runs_for_task) > 0:
                for item in all_mlflow_runs_for_task:
                    logger.info(f"deleting {item}")
                    client.delete_run(item)
                    os.system(f"rm -r {experiment_info.artifact_location}/{item}")
                    task_run_id = None
    else:
        #delete any existing mlflow runs
        if len(all_mlflow_runs_for_task) > 0:
            for item in all_mlflow_runs_for_task:
                logger.info(f"deleting {item}")
                client.delete_run(item)
                os.system(f"rm -r {experiment_info.artifact_location}/{item}")
            task_run_id = None
    return task_run_id


def extract_repeated_experiment_results(
        storage_uri: str,
        logger,
        task_type: str = "segmentation",
        experiment_data_path: str = "job_status/combined_segmentation_final.csv",
        num_repetitions: int = 10,
        num_tasks:int = 6
        ):

    results_tables_dir = "/".join(storage_uri.split("/")[:-1]) + "/" + "repeated_exp_output_mlflow_final"
    if not os.path.exists(results_tables_dir):
        os.makedirs(results_tables_dir)
    
    task_names = SEGMENTATION_BASE_TASKS if task_type=="segmentation" else CLASSIFICATION_BASE_TASKS
    
    logger.info(f"\n\n\n\n\n\nSTARTING task_type: {task_type} storage_uri:{storage_uri}") 
    experiment_data = pd.read_csv(experiment_data_path)
    client = mlflow.tracking.MlflowClient(tracking_uri=storage_uri)

    experiments = experiment_data["experiment_name"].tolist()
    experiments = [exp.split(": ")[-1] for exp in experiments]
    experiments = list(set(experiments))
    experiments_to_be_run_again = []

    for experiment_name in experiments:
        experiment_name = f"{experiment_name}_repeated_exp"
        logger.info(f"\n\n\nexperiment_name: {experiment_name}") 
        experiment_info = client.get_experiment_by_name(experiment_name)
        if experiment_info is None:
            logger.info(f"NO experiment_info IN THIS FOLDER: {experiment_name}")
            experiments_to_be_run_again.append(experiment_name)
            continue
        experiment_id = experiment_info.experiment_id
        logger.info(f"experiment_id: {experiment_id}")
        logger.info(f"experiment_name: {experiment_name}")
        logger.info(f"experiment_info: {experiment_info}")
        experiment_parent_run_data = client.search_runs(experiment_ids=[experiment_id])

        run_names = []
        run_ids = []
        run_status = []
        run_task = []
        run_score = []
        run_metric = []

        for run in experiment_parent_run_data:
            run_name = run.info.run_name
            task = "_".join(run_name.split("_")[:-1])
            
            if (task in task_names) and (run.info.status =="FINISHED"):
                seed = run.data.params["seed"] if "seed" in run.data.params else "NA"
                if task_type == "segmentation":
                    metric_name = 'test_test/Multiclass_Jaccard_Index' 
                else:
                    if task == "big_earth_net":
                        metric_name = 'test_test/Multilabel_F1_Score'  
                    else:
                        metric_name = 'test_test/Overall_Accuracy'
                
                if metric_name not in run.data.metrics:
                    continue
                score = run.data.metrics[metric_name]
                run_names.append(run.info.run_name)
                run_ids.append(run.info.run_id)
                run_status.append(run.info.status)
                run_metric.append(metric_name.split("/")[-1])
                run_task.append(task)
                run_score.append(score)
        
        df = pd.DataFrame({
            "Task": run_task,
            "Metric": run_metric,
            "Score": run_score,
            "mlflow_run_name": run_names,
            "mlflow_run_id": run_ids,
            "mlflow_run_status": run_status
        })
        if len(run_task) == 0: 
            logger.info(f"TO BE RE-RUN: {experiment_name}. \nHas NO tasks")
            experiments_to_be_run_again.append(experiment_name)
            continue

        #get ten successful results per task
        combine_task_results = []
        for task in task_names:
            task_df = df.loc[(df["Task"] == task) & (df["mlflow_run_status"] == "FINISHED") ].copy()
            task_df = task_df.loc[(task_df["Score"] != 0.0)].copy()
            rows, cols = task_df.shape
            if rows >= num_repetitions:
                task_df = task_df.iloc[list(range(num_repetitions))].copy()
            elif rows < num_repetitions:
                logger.info(f"NOT COMPLETE: task: {task} only has {rows} rows")
            combine_task_results.append(task_df)
        if len(combine_task_results) > 0:
            combine_task_results = pd.concat(combine_task_results, axis=0)
            combine_task_results.to_csv(f"{results_tables_dir}/{experiment_name}_mlflow.csv", index=False) 
        if len(combine_task_results) < num_tasks:
            logger.info(f"TO BE RE-RUN: {experiment_name}. \nHas {len(combine_task_results)} tasks only")
            experiments_to_be_run_again.append(experiment_name)
        
    with open(f"{results_tables_dir}/need_to_be_re_run{task_type}.txt", 'w') as f:
        for line in experiments_to_be_run_again:
            f.write(f"{line}\n")

        


def extract_parameters(
                    save_folder,
                    experiment_data_path,
                    logger,
                    storage_uris,
                    task_type,
                    exp_filter,
                    all_settings,
                    ):
    """
    extracts hyper-parameter information for each experiment from the mlflow logs
    saves this information to the csv file: f"{save_folder}/{task_type}_parameters.csv"

    Args:
        save_folder: folder where results will be saved
        experiment_data_path: path to csv file containing the following information about the experiments
                                columns: experiment_name, backbone, early_stop_patience, n_trials, data_percentages, decoder)
        logger: logging object
        storage_uris: folder containing MLFlow logs for the given experiments
        task_type: segmentation/classification
        filter: filter to select which experiments in the csv file should be considered
    """

    logger.info(f"\nSTARTING TO CHECK PARAMS for {task_type}")  

    #get list of experiments
    experiment_data = pd.read_csv(experiment_data_path)
    list_of_experiments = experiment_data["experiment_name"].tolist()
    list_of_experiments = [item.split("experiment_name: ")[-1] for item in list_of_experiments]
    list_of_experiments = list(set(list_of_experiments))

    save_folder = f"{save_folder}/tables"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)     

    list_of_experiments = [item for item in list_of_experiments if exp_filter in item]
    logger.info(f"list_of_experiments: {len(list_of_experiments)}") 
    
    list_of_all_params = []
    for storage_uri in storage_uris:
        client = mlflow.tracking.MlflowClient(tracking_uri=storage_uri)
        for experiment_name in list_of_experiments:
            #get backbone name
            exp_info = experiment_data.loc[experiment_data["experiment_name"] == f"experiment_name: {experiment_name}"]
            backbone = exp_info["backbones"].tolist()[0]

            #colect all relevant settings from csv file
            decoder = exp_info["decoders"].tolist()[0]
            early_stop_patience = exp_info["early_stop_patience"].tolist()[0]
            n_trials = exp_info["n_trials"].tolist()[0]
            data_percentages = exp_info["data_percentages"].tolist()[0]
            exp_parent_run_name = f"{backbone}_geobench_v2"

            #get experiment id
            experiment_info = client.get_experiment_by_name(experiment_name)
            if experiment_info is None:
                continue
            #logger.info(f"\n\nexperiment_id: {experiment_info.experiment_id}")
            logger.info(f"\nexperiment_name: {experiment_name} ")  
            logger.info(f"backbone: {backbone} decoder: {decoder} early_stop_patience:\
                         {early_stop_patience} n_trials: {n_trials} data_percentages: {data_percentages}")  
            experiment_id = experiment_info.experiment_id

            experiment_parent_run_data = client.search_runs(experiment_ids=[experiment_id], 
                                filter_string=f'tags."mlflow.runName" LIKE "{exp_parent_run_name}"')
            logger.info(f"experiment_parent_run_data: {len(experiment_parent_run_data)}")  

            for run in experiment_parent_run_data:
                exp_parent_run_id = run.info.run_id

            mlflow.set_tracking_uri(storage_uri)
            mlflow.set_experiment(experiment_name)

            runs: list[mlflow.entities.Run] = mlflow.search_runs(
                filter_string=f"tags.mlflow.parentRunId='{exp_parent_run_id}'", output_format="list"
            )  # type: ignore
            logger.info(f"Found runs: {[run.info.run_name for run in runs]}")

            if task_type == "segmentation":
                tasks = SEGMENTATION_BASE_TASKS
            elif task_type == "classification":
                tasks = CLASSIFICATION_BASE_TASKS

            for task in tasks:
                logger.info(f"task: {task}")  
                matching_runs = [run for run in runs if run.info.run_name.endswith(task)]  # type: ignore
                best_params = matching_runs[0].data.params
                # eval them
                best_params = {k: literal_eval(v) for k, v in best_params.items()}
                best_params["experiment_name"] = experiment_name
                best_params["task"] = task
                best_params["backbone"] = backbone
                best_params["decoder"] = decoder
                best_params["early_stop_patience"] = early_stop_patience
                best_params["n_trials"] = n_trials
                best_params["data_percentages"] = data_percentages
                logger.info(f"best_params: {best_params}")  
                if 'optimizer_hparams' in best_params:
                    logger.info(f"optimizer_hparams: {best_params['optimizer_hparams'].items()}")  
                    
                    optimizer_hparams = {k: v for k,v in best_params['optimizer_hparams'].items()}
                    best_params.update(optimizer_hparams) 
                    del best_params['optimizer_hparams']
                best_params = pd.DataFrame(best_params, index=[0])
                list_of_all_params.append(best_params)
    list_of_all_params = pd.concat(list_of_all_params, axis=0)
    list_of_all_params = list_of_all_params.reset_index()

    for setting in all_settings:
        exp_for_setting = [exp for exp in list_of_experiments if setting in exp]
        params_for_setting = list_of_all_params.loc[list_of_all_params["experiment_name"].isin(exp_for_setting)].copy()
        params_for_setting.to_csv(f"{save_folder}/{task_type}_{setting}_parameters.csv", index=False) 



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



def compile_and_visualize(
                            completed_experiments,
                            save_folder,
                            logger,
                            plot_file_base_name,
                            task_type = "segmentation",
                            partition_name = "0.10xtrain",
                            repeated_experiment_folder = "results/repeated_exp_output"
                            ):

    if not os.path.exists(f"{save_folder}/tables/"):
        os.makedirs(f"{save_folder}/tables/")
    if not os.path.exists(f"{save_folder}/plots/"):
        os.makedirs(f"{save_folder}/plots/")
    if not os.path.exists(f"{save_folder}/tracking/"):
        os.makedirs(f"{save_folder}/tracking/")

    logger.info(f"\nSTARTING TO COMPILE AND VISUALIZE")  

    combined_results = []
    model_order = []
    mean_value = []
    experiment_names = []
    for experiment_name in completed_experiments:
        logger.info(f"experiment_name: {experiment_name}")  
        repeated_results_files = f"{repeated_experiment_folder}/{experiment_name}_*"
        repeated_results_files = glob.glob(repeated_results_files)
        logger.info(repeated_results_files)
        if len(set(repeated_results_files)) > 1:
            logger.info(f"too many files for repeated experiments: {experiment_name}")
            logger.info(len(set(repeated_results_files)))
            logger.info(repeated_results_files)
            continue
            #raise RuntimeError
        elif len(set(repeated_results_files)) == 0:
            continue
        repeated_results_files = sorted(repeated_results_files)[-1]
        repeated_experiment_results = pd.read_csv(repeated_results_files)
        if "Unnamed: 0" in list(repeated_experiment_results.columns):
            repeated_experiment_results.drop(["Unnamed: 0"], axis=1, inplace=True)
        logger.info(f"repeated_experiment_results: {repeated_experiment_results.shape}")
        if (0.0 in list(repeated_experiment_results["Score"])) or (sum(np.isnan(repeated_experiment_results["Score"])) > 0):
            logger.info(f"Has a zero, REDO: {experiment_name}")
            logger.info(repeated_results_files)
            continue
            #raise RuntimeError

        repeated_experiment_results["Backbone"] = experiment_name
        combined_results.append(repeated_experiment_results)
        model_order.append(experiment_name)

    num_experiments = len(combined_results)
    fig_size = (num_experiments*5, 6) if num_experiments>=3 else (15,6)
    n_legend_rows = num_experiments//3 if num_experiments>=3 else 1
    logger.info(f"number of experiments to compare: {len(combined_results)}")

    if len(combined_results) == 0:
        return None
    combined_results = pd.concat(combined_results, ignore_index=True)
    combined_results = combined_results.rename(columns={"Task": "dataset", "Backbone": "model", "Score": "test metric"})
    combined_results.to_csv(f"{save_folder}/tables/{plot_file_base_name}_combined_results.csv")
    logger.info(f"combined_results :{combined_results.shape}")

    model_colors = dict( zip(model_order, sns.color_palette("tab20", n_colors=len(model_order))))
    combined_results = pd.read_csv(f"{save_folder}/tables/{plot_file_base_name}_combined_results.csv", index_col="Unnamed: 0")
    combined_results["partition name"] = partition_name 
    model_order = sorted(model_order)
    logger.info(f"model_order: {model_order}")
    logger.info(f"model_colors: {model_colors}")

    try: 
        #plot raw values
        plot_tools.box_plot_per_dataset(combined_results, 
                                    model_order=model_order, 
                                    plot_file_base_name=plot_file_base_name,
                                    model_colors=model_colors, 
                                    metric="test metric", 
                                    sharey=False, 
                                    inner="points", 
                                    fig_size=fig_size, 
                                    n_legend_rows=n_legend_rows)
        plt.savefig(f"{save_folder}/plots/box_{plot_file_base_name}_raw.png", bbox_inches="tight")
        plt.close()

        #plot normalized, bootstrapped values values
        normalizer = plot_tools.make_normalizer(combined_results, 
                                                metrics=("test metric",), 
                                                benchmark_name=plot_file_base_name)
        bootstrapped_iqm, normalized_combined_results = plot_tools.normalize_bootstrap_and_plot(combined_results, 
                                                                plot_file_base_name=plot_file_base_name,
                                                                metric="test metric",
                                                                benchmark_name=plot_file_base_name, 
                                                                model_order=model_order, 
                                                                model_colors=model_colors, 
                                                                fig_size=fig_size,
                                                                n_legend_rows=n_legend_rows)
                                                                #dataset_name_map=dataset_name_map)
        
        plt.savefig(f"{save_folder}/plots/box_{plot_file_base_name}_normalized_bootstrapped.png", bbox_inches="tight")
        plt.close()
        bootstrapped_iqm.to_csv(f"{save_folder}/tables/{plot_file_base_name}_bootstrapped_iqm.csv")
        combined_results.to_csv(f"{save_folder}/tables/{plot_file_base_name}_normalized_combined_results.csv")
    except Exception as e:
        logger.info(f"could not visualize due to error: {e}")
    return model_order




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
    handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logging.basicConfig(level=logging.CRITICAL)
    return logger



if __name__ == "__main__":
    logger = get_logger()
    save_folder = "leaderboard_ready_results"
    experiment_data_path = "segmentation_experiment_info_new.csv"
    task_type = "segmentation"
    results_root_folder = "results"
    hpo_experiment_storage_uri = f"{results_root_folder}/{task_type}/hpo_results"
    repeated_experiment_folder =  f"{results_root_folder}/{task_type}/repeated_exp_output"
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    #get hyperparameters from mlflow logs
    logger.info(f"Now compiling results for: {task_type}")
    extract_parameters(
                save_folder = save_folder,
                experiment_data_path= experiment_data_path,
                logger =logger,
                storage_uris = [hpo_storage_uri],
                task_type = task_type,
                all_settings = [
                                "early_stopping_10_data_100_perc", 
                                "early_stopping_50_data_10_perc", 
                                "early_stopping_50_data_100_perc",
                                ]
                )

    #extract repeated experiment results from mlflow logs
    extract_repeated_experiment_results(
        storage_uri = hpo_storage_uri,
        logger = logger,
        task_type = task_type,
        experiment_data_path= experiment_data_path,
        num_repetitions = 10,
        num_tasks = 6
        )

    experiment_data = pd.read_csv(experiment_data_path)
    experiment_names = experiment_data["experiment_name"].tolist()
    experiment_names = [item.replace("experiment_name: ", "") for item in experiment_names]
    experiment_names = list(set(experiment_names))


    SETTINGS_PER_MODEL= ["early_stopping_10_data_100_perc", 
                    "early_stopping_50_data_10_perc", 
                    "early_stopping_50_data_100_perc"]
    
    #create box plots across multiple models
    for setting in SETTINGS_PER_MODEL:
        #only compare across best decoder for each model for completed experiments
        if task_type == "segmentation":
            comparison_1 = "upernetdecoder" 
            comparison_2 = "unet" 
        elif task_type ==  "classification":
            comparison_1 = "_" 
            comparison_2 = "_" 
        exp_for_setting = [exp for exp in experiment_names if setting in exp]
        best_exp_for_setting = [exp for exp in exp_for_setting if ((comparison_1 in exp) or (comparison_2 in exp))]
        model_order = compile_and_visualize(
                            best_exp_for_setting,
                            save_folder,
                            logger,
                            plot_file_base_name = f"multiple_models_{setting}_{comparison_1}_{comparison_2}",
                            task_type = task_type,
                            partition_name = "1.00xtrain",
                            repeated_experiment_folder = repeated_experiment_folder
                            )
