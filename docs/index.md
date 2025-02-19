# Benchmarking

A tool for benchmarking and hpo using [TerraTorch](https://github.ibm.com/GeoFM-Finetuning/terratorch).

Leverages MLFlow for experiment logging, optuna for hyperparameter optimization and ray for parallelization.

## Environment

Using a virtual environment for all commands in this guide is strongly recommended.
Package development was carried out with [Poetry](https://python-poetry.org/)

## Installation

Install `torch >= 2.0` before installing this package

### Package installation

```sh
# assuming you have an SSH key set up on GitHub
pip install "git+ssh://git@github.ibm.com/GeoFM-Finetuning/benchmark.git@main"
```

### Suggested setup for development

```sh
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -r dev_requirements.txt
pip install -e .
```

## Usage

This tool allows you to design a benchmark test for a backbone that exists in `TerraTorch` over:

- Several tasks

- Several hyperparameter configurations

To do this it relies on a configuration file where the benchmark is defined. This consists of:

- `experiment_name`: MLFLow experiment to run the benchmark on. This is the highest level grouping of runs in MLFLow.

- `run_name`: Name of the parent (top-level) run under the experiment.

- `description`: Description added to top level mlflow run.

- `defaults`: Defaults that are set for all tasks. Can be overriden under each task.

- `tasks`: List of tasks to perform. Tasks specify parameters for the decoder, datamodule to be used and training parameters.

- `n_trials`: Number of trials to be carried out per task, in the case of hyperparameter tuning.

- `save_models`: Whether to save models. Defaults to False. (Setting this to true can take up a lot of space). Models will be logged as artifacts for each run in MLFlow.

- `storage_uri`: Location to use for storage for mlflow.

- `optimization_space`: Hyperparameter space to search over. Bayesian optimization tends to work well with a small number of hyperparameters.

- `bayesian_search`: Whether to perform a bayesian search (future hparam configs depend on results of past runs) or a random search. Defaults to True, which does bayesian search.

See `benchmark_v2_template.yaml` in the git repo for an example.

To run a benchmark, use `benchmark --config <benchmark_file>`.

To run a benchmark over a ray cluster (which must be created before running), use `ray_benchmark --config <benchmark_file>`.

To check the experiment results, use `mlflow ui --host $(hostname -f) --port <port> --backend-store-uri <storage_uri>` and click the link.
![mlflow demo](images/mlflow.png)

## :::benchmark.backbone_benchmark.benchmark_backbone

## Default and Task specification

Under each of these, as well as for the `optimization_space`, the structure of parameters and their hierarchy for `terratorch_task` follows the same as used in terratorch. The terratorch task contains the actual parameters that will be passed to terratorch.

An exception is made for `batch_size` in `optimization_space`, which should be passed in the root level and is not passed to the `terratorch_task`.

### :::benchmark.benchmark_types.Defaults

### :::benchmark.benchmark_types.Task

## :::benchmark.benchmark_types.ParameterBounds

## :::benchmark.benchmark_types.TaskTypeEnum

## :::benchmark.benchmark_types.ParameterTypeEnum

## Credits

Work by Carlos Gomes (<carlos.gomes@ibm.com>).
This project was created using <https://github.ibm.com/innersource/python-blueprint>.
