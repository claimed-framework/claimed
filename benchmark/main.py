from typing import Any, List
from jsonargparse import ArgumentParser
from benchmark.backbone_benchmark import benchmark_backbone
from benchmark.benchmark_types import Defaults, Task
from benchmark.repeat_best_experiment import rerun_best_from_backbone


def main():
    print(f"Running terratorch-iterate...")
    parser = ArgumentParser()

    parser.add_argument('--defaults', type=Defaults)  # to ignore model
    parser.add_argument('--optimization_space', type=dict)  # to ignore model
    parser.add_argument('--experiment_name', type=str)  # to ignore model
    parser.add_argument('--run_name', type=str)  # to ignore model
    parser.add_argument('--save_models', type=bool)  # to ignore model
    parser.add_argument('--storage_uri', type=str)  # to ignore model
    parser.add_argument('--ray_storage_path', type=str)  # to ignore model
    parser.add_argument('--n_trials', type=int)  # to ignore model
    parser.add_argument('--run_repetitions', type=int)  # to ignore model
    parser.add_argument('--tasks', type=list[Task])
    parser.add_argument("--config", action="config")
    parser.add_argument("--hpo", help="optimize hyperparameters", action="store_true")
    parser.add_argument("--repeat", help="repeat best experiments", action="store_true")

    args = parser.parse_args()
    print(args)
    paths: List[Any] = args.config
    path = paths[0]
    print(f"args={args} path={path}")
    repeat = args.repeat
    assert isinstance(repeat, bool), f"Error! {repeat=} is not a bool"
    hpo = args.hpo
    assert isinstance(hpo, bool), f"Error! {hpo=} is not a bool"
    config = parser.parse_path(path)

    config_init = parser.instantiate_classes(config)
    # validate the objects
    experiment_name = config_init.experiment_name
    assert isinstance(experiment_name, str), f"Error! {experiment_name=} is not a str"
    run_name = config_init.run_name
    if run_name is not None:
        assert isinstance(run_name, str), f"Error! {run_name=} is not a str"
    tasks = config_init.tasks
    assert isinstance(tasks, list), f"Error! {tasks=} is not a list"
    for t in tasks:
        assert isinstance(t, Task), f"Error! {t=} is not a Task"
    defaults = config_init.defaults
    assert isinstance(defaults, Defaults), f"Error! {defaults=} is not a Defaults"
    # defaults.trainer_args["max_epochs"] = 5
    storage_uri = config_init.storage_uri
    assert isinstance(storage_uri, str), f"Error! {storage_uri=} is not a str"

    optimization_space = config_init.optimization_space
    assert isinstance(
        optimization_space, dict
    ), f"Error! {optimization_space=} is not a dict"
    ray_storage_path = config_init.ray_storage_path
    assert isinstance(ray_storage_path, str), f"Error! {ray_storage_path=} is not a str"

    n_trials = config_init.n_trials
    assert isinstance(n_trials, int) and n_trials > 0, f"Error! {n_trials=} is invalid"
    run_repetitions = config_init.run_repetitions
    if repeat and not hpo:
        rerun_best_from_backbone(
            defaults=defaults,
            tasks=tasks,
            experiment_name=experiment_name,
            storage_uri=storage_uri,
            optimization_space=optimization_space,
            run_repetitions=run_repetitions,
        )
    else:
        if not repeat and hpo:
            run_repetitions = 0

        # run_repetions is an optional parameter
        benchmark_backbone(
            defaults=defaults,
            tasks=tasks,
            experiment_name=experiment_name,
            storage_uri=storage_uri,
            ray_storage_path=ray_storage_path,
            run_name=run_name,
            optimization_space=optimization_space,
            n_trials=n_trials,
            run_repetitions=run_repetitions,
        )


if __name__ == "__main__":
    main()
