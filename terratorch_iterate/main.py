import os
from jsonargparse import Namespace
import logging
from pathlib import Path
from jsonargparse import ArgumentParser
import pandas as pd
from terratorch_iterate.backbone_benchmark import benchmark_backbone
from terratorch_iterate.iterate_types import Defaults, Task
from terratorch_iterate.repeat_best_experiment import rerun_best_from_backbone
from terratorch_iterate.utils import (
    get_logger,
    import_custom_modules,
    get_results_and_parameters,
)
from terratorch_iterate.config_util import build_iterate_config


def _summarize(
    config_init: Namespace,
    hpo: bool,
    repeat: bool,
    storage_uri: str,
    logger: logging.RootLogger,
) -> pd.DataFrame:
    """only summarize results from multiple experiments

    Args:
        config_init (Namespace): _description_
        hpo (bool): flag that indicates whether to run hpo
        repeat (bool): flag that indicates whether to repeat best experiment
        storage_uri (str): path to directory in which results will be stored
        logger (logging.RootLogger): logger variable

    Returns:
        _type_: _description_
    """
    assert hpo is False and repeat is False, (
        f"Error! both {repeat=} and {hpo=} must be False when summarizing results from multiple experiments."
    )

    list_of_experiment_names = config_init.list_of_experiment_names
    assert isinstance(list_of_experiment_names, list), (
        f"Error! {list_of_experiment_names=} is not a list"
    )
    for exp in list_of_experiment_names:
        assert isinstance(exp, str), f"Error! {exp=} is not a str"

    task_names = config_init.task_names
    assert isinstance(task_names, list), f"Error! {task_names=} is not a list"
    for t in task_names:
        assert isinstance(t, str), f"Error! {t=} is not a str"

    task_metrics = config_init.task_metrics
    assert isinstance(task_metrics, list), f"Error! {task_metrics=} is not a list"
    for t in task_metrics:
        assert isinstance(t, str), f"Error! {t=} is not a str"

    benchmark_name = config_init.benchmark_name
    assert isinstance(benchmark_name, str), f"Error! {benchmark_name=} is not a str"

    run_repetitions = config_init.run_repetitions
    assert isinstance(run_repetitions, int) and run_repetitions > 0, (
        f"Error! {run_repetitions=} is invalid"
    )
    # get results and parameters from mlflow logs
    results_and_parameters = get_results_and_parameters(
        benchmark_name=benchmark_name,
        storage_uri=storage_uri,
        logger=logger,
        experiments=list_of_experiment_names,
        task_names=task_names,
        num_repetitions=run_repetitions,
        task_metrics=task_metrics,
    )
    return results_and_parameters


def _repeat_experiment(
    config_init: Namespace,
    storage_uri: str,
    experiment_name: str,
    parent_run_id: str,
    defaults: Defaults,
    tasks: list[Task],
    optimization_space: dict,
    run_repetitions: int,
    save_models: bool,
    report_on_best_val: bool,
    logger: logging.RootLogger,
):
    """repeat best experiments

    Args:
        config_init (Namespace): _description_
        storage_uri (str): _description_
        experiment_name (str): _description_
        parent_run_id (str): _description_
        defaults (Defaults): _description_
        tasks (list[Task]): _description_
        optimization_space (dict): _description_
        run_repetitions (int): _description_
        save_models (bool): _description_
        report_on_best_val (bool): _description_
        logger (logging.RootLogger): _description_

    Returns:
        _type_: _description_
    """
    output: str | None = config_init.output_path
    if output is None:
        storage_uri_path = Path(storage_uri)
        assert storage_uri_path.exists() and storage_uri_path.is_dir(), (
            f"Error! Unable to create new output_path based on storage_uri_path because the latter does not exist: {storage_uri_path}"
        )
        output_path = storage_uri_path.parents[0] / "repeated_exp_output_csv"
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / f"{experiment_name}_repeated_exp_mlflow.csv"
        output = str(output_path)

    logger.info("Rerun best experiments...")
    rerun_best_from_backbone(
        logger=logger,
        parent_run_id=parent_run_id,
        output_path=output_path,
        defaults=defaults,
        tasks=tasks,
        experiment_name=experiment_name,
        storage_uri=storage_uri,
        optimization_space=optimization_space,
        run_repetitions=run_repetitions,
        save_models=save_models,
        report_on_best_val=report_on_best_val,
    )


def _convert_config(args: Namespace):
    """
    This function processes command-line arguments to convert configuration files.

    Parameters:
    args (argparse.Namespace): Namespace object containing command-line arguments.

    Raises:
    AssertionError: If input or output paths are invalid or missing.

    This function performs the following steps:
    1. Asserts that the 'input' argument is a non-empty string and checks if the file exists.
    2. Asserts that the 'output' argument is a non-empty string.
    3. Calls the `generate_iterate_config` function from the `build_iterate_config` module, passing the input path, output path, prefix (if provided), and template (if provided).
    """
    input: str = args.input
    assert input is not None and isinstance(input, str), (
        f"Error! Invalid value: {input=}"
    )
    input_path = Path(input)
    assert input_path.exists()

    output: str = args.output
    assert output is not None and isinstance(output, str), (
        f"Error! Invalid value: {output=}"
    )
    output_path = Path(output)
    template: str | None = args.template

    prefix: str | None = args.prefix

    template: str | None = args.template
    build_iterate_config.generate_iterate_config(
        input=input_path, output=output_path, prefix=prefix, template=template
    )


def main():
    print("DEPRECATED: iterate-classic is deprecated. Please use iterate instead.")
    parser = ArgumentParser()

    parser.add_argument("--defaults", type=Defaults)  # to ignore model
    parser.add_argument("--optimization_space", type=dict)  # to ignore model
    parser.add_argument("--experiment_name", type=str)  # to ignore model
    parser.add_argument("--run_name", type=str)  # to ignore model
    parser.add_argument("--save_models", type=bool)  # to ignore model
    parser.add_argument("--storage_uri", type=str)  # to ignore model
    parser.add_argument("--ray_storage_path", type=str)  # to ignore model
    parser.add_argument("--n_trials", type=int)  # to ignore model
    parser.add_argument("--run_repetitions", type=int)  # to ignore model
    parser.add_argument("--tasks", type=list[Task])
    parser.add_argument("--parent_run_id", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--logger", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--custom_modules_path", type=str)
    parser.add_argument("--report_on_best_val", type=bool, default=True)
    parser.add_argument("--test_models", type=bool, default=False)
    parser.add_argument("--bayesian_search", type=bool, default=True)
    parser.add_argument("--hpo", help="optimize hyperparameters", action="store_true")
    parser.add_argument("--repeat", help="repeat best experiments", action="store_true")
    parser.add_argument(
        "--continue_existing_experiments",
        help="continue existing experiments",
        action="store_true",
    )
    parser.add_argument(
        "--summarize",
        help="summarize results from repeated experiments",
        action="store_true",
    )
    parser.add_argument("--list_of_experiment_names", type=list[str])
    parser.add_argument("--task_names", type=list[str])
    parser.add_argument("--task_metrics", type=list[str])
    parser.add_argument(
        "--benchmark_name",
        type=str,
        help="name of summarized results file",
    )
    # arguments to convert terratorch's config into iterate's config
    parser.add_argument(
        "--build_iterate_config",
        help="convert terratorch's config into terratorch-iterate's config",
        action="store_true",
    )
    parser.add_argument(
        "--input",
        help="input file or directory",
        type=str,
    )
    parser.add_argument(
        "--output",
        help="output file or directory",
        type=str,
    )
    parser.add_argument(
        "--template",
        help="template for creating config files",
        type=str,
    )
    parser.add_argument(
        "--prefix",
        help="prefix of new config files",
        type=str,
    )

    args = parser.parse_args()
    if args.build_iterate_config is not None and args.build_iterate_config is True:
        _convert_config(args)
    else:
        config_path: str | None = args.config
        if config_path is None:
            msg = """
            Error: config argument has not been passed
            Usage: iterate [-h] [--hpo] [--repeat] [--summarize] [--config CONFIG] 
            """
            print(msg)
        else:
            assert isinstance(config_path, str), (
                f"Error! Unexpected config type: {config_path}"
            )
            config = parser.parse_path(config_path)

            config_init: Namespace = parser.instantiate_classes(config)

            summarize: bool = args.summarize
            assert isinstance(summarize, bool), f"Error! {summarize=} is not a bool"
            repeat = args.repeat
            assert isinstance(repeat, bool), f"Error! {repeat=} is not a bool"
            hpo = args.hpo
            assert isinstance(hpo, bool), f"Error! {hpo=} is not a bool"

            continue_existing_experiments: bool = args.continue_existing_experiments
            assert isinstance(continue_existing_experiments, bool), (
                f"Error! {continue_existing_experiments=} is not a bool"
            )

            storage_uri = config_init.storage_uri
            assert isinstance(storage_uri, str), f"Error! {storage_uri=} is not a str"
            os.environ["MLFLOW_TRACKING_URI"] = storage_uri
            # handling relative paths
            if storage_uri.startswith(".") or storage_uri.startswith(".."):
                repo_home_dir = Path(__file__).parent.parent
                abs_path = repo_home_dir / storage_uri
                storage_uri = str(abs_path.resolve())

            logger_path = config_init.logger
            if logger_path is None:
                storage_uri_path = Path(storage_uri)
                logger = get_logger(
                    log_folder=f"{str(storage_uri_path.parents[0])}/job_logs"
                )
            else:
                logging.config.fileConfig(
                    fname=logger_path, disable_existing_loggers=False
                )
                logger = logging.getLogger("terratorch-iterate")

            # only summarize results from multiple experiments
            if summarize:
                return _summarize(
                    config_init=config_init,
                )

            # optimize hyperparameters and/or do repeated runs for single experiments
            assert hpo is True or repeat is True, (
                f"Error! either {repeat=} or {hpo=} must be True"
            )
            parent_run_id = args.parent_run_id
            if parent_run_id is not None:
                assert isinstance(parent_run_id, str), (
                    f"Error! {parent_run_id=} is not a str"
                )

            # validate the objects
            experiment_name = config_init.experiment_name
            assert isinstance(experiment_name, str), (
                f"Error! {experiment_name=} is not a str"
            )
            run_name = config_init.run_name
            if run_name is not None:
                assert isinstance(run_name, str), f"Error! {run_name=} is not a str"
            # validate defaults
            defaults = config_init.defaults
            assert isinstance(defaults, Defaults), (
                f"Error! {defaults=} is not a Defaults"
            )

            tasks = config_init.tasks
            assert isinstance(tasks, list), f"Error! {tasks=} is not a list"
            for t in tasks:
                assert isinstance(t, Task), f"Error! {t=} is not a Task"
                # if there is not specific terratorch_task specified, then use default terratorch_task
                if t.terratorch_task is None:
                    t.terratorch_task = defaults.terratorch_task
            # defaults.trainer_args["max_epochs"] = 5

            optimization_space = config_init.optimization_space
            assert isinstance(optimization_space, dict), (
                f"Error! {optimization_space=} is not a dict"
            )

            # ray_storage_path is optional
            ray_storage_path = config_init.ray_storage_path
            if ray_storage_path is not None:
                assert isinstance(ray_storage_path, str), (
                    f"Error! {ray_storage_path=} is not a str"
                )

            n_trials = config_init.n_trials
            assert isinstance(n_trials, int) and n_trials > 0, (
                f"Error! {n_trials=} is invalid"
            )
            run_repetitions = config_init.run_repetitions

            report_on_best_val = config_init.report_on_best_val
            assert isinstance(report_on_best_val, bool), (
                f"Error! {ray_storage_path=} is not a bool"
            )

            save_models = config_init.save_models
            assert isinstance(save_models, bool), f"Error! {save_models=} is not a bool"

            test_models = config_init.test_models
            assert isinstance(test_models, bool), f"Error! {test_models=} is not a bool"

            bayesian_search = config_init.bayesian_search
            assert isinstance(bayesian_search, bool), (
                f"Error! {bayesian_search=} is not a bool"
            )

            # custom_modules_path is optional
            custom_modules_path = config_init.custom_modules_path
            if custom_modules_path is not None:
                assert isinstance(custom_modules_path, str), (
                    f"Error! {custom_modules_path=} is not a str"
                )
                import_custom_modules(
                    logger=logger, custom_modules_path=custom_modules_path
                )

            if repeat and not hpo:
                _repeat_experiment(
                    config_init=config_init,
                    storage_uri=storage_uri,
                    experiment_name=experiment_name,
                    defaults=defaults,
                    tasks=tasks,
                    optimization_space=optimization_space,
                    run_repetitions=run_repetitions,
                    save_models=save_models,
                    logger=logger,
                    parent_run_id=parent_run_id,
                    report_on_best_val=report_on_best_val,
                )
            else:
                if not repeat and hpo:
                    run_repetitions = 0

                # run_repetitions is an optional parameter
                experiment_info: dict = benchmark_backbone(
                    defaults=defaults,
                    tasks=tasks,
                    experiment_name=experiment_name,
                    storage_uri=storage_uri,
                    ray_storage_path=ray_storage_path,
                    run_name=run_name,
                    run_id=None,
                    optimization_space=optimization_space,
                    n_trials=n_trials,
                    run_repetitions=run_repetitions,
                    save_models=save_models,
                    report_on_best_val=report_on_best_val,
                    test_models=test_models,
                    bayesian_search=bayesian_search,
                    continue_existing_experiment=continue_existing_experiments,
                    logger=logger,
                )
                return experiment_info


if __name__ == "__main__":
    main()
