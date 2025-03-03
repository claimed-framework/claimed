<!-- [![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
![alt text](./coverage.svg) -->
# TerraTorch-iterate

A tool for benchmarking and hyper-parameter optimization using [TerraTorch](https://github.ibm.com/GeoFM-Finetuning/terratorch).

Leverages MLFlow for experiment logging, optuna for hyperparameter optimization and ray for parallelization.

## Installation

We recommend using python 3.10, 3.11 or 3.12 and also using a virtual environment for all commands in this guide.

### Package installation

```sh
pip install "git+https://github.com/IBM/terratorch-iterate"
```

### Suggested setup for development

```sh
pip install --upgrade pip setuptools wheel
pip install -e .
```

## Usage

This tool allows you to design a benchmark test for a backbone that exists in `TerraTorch` over:

- Several tasks

- Several hyperparameter configurations

To do this it relies on a configuration file where the benchmark is defined. This consists of:

- `experiment_name`: MLFLow experiment to run the benchmark on. This is the highest level grouping of runs in MLFLow.

- `run_name`: Name of the parent (top-level) run under the experiment. NOTE: This item should not be included in the config if you wish to use the parameters extraction function in `mlfow_utils` to compile results.

- `defaults`: Defaults that are set for all tasks. Can be overriden under each task.

- `tasks`: List of tasks to perform. Tasks specify parameters for the decoder, datamodule to be used and training parameters.

- `n_trials`: Number of trials to be carried out per task, in the case of hyperparameter tuning.

- `save_models`: Whether to save models. Defaults to False. (Setting this to true can take up a lot of space). Models will be logged as artifacts for each run in MLFlow.

- `storage_uri`: Location to use for storage for mlflow.

- `optimization_space`: Hyperparameter space to search over. Bayesian optimization tends to work well with a small number of hyperparameters.

See `benchmark_v2_template.yaml` in the git repo for an example.

To run a benchmark, use `benchmark --config <benchmark_file>`.

To run a benchmark over a ray cluster (which must be created before running), use `ray_benchmark --config <benchmark_file>`.

To check the experiment results, use `mlflow ui --host $(hostname -f) --port <port> --backend-store-uri <storage_uri>` 

## Ray
You can also parallelize your runs over a ray cluster. 

Check out instructions in the [docs](./docs/ray.md)



