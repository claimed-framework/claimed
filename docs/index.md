# CLAIMED Framework

[![OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/projects/6718/badge)](https://bestpractices.coreinfrastructure.org/projects/6718)
[![PyPI](https://img.shields.io/pypi/v/claimed)](https://pypi.org/project/claimed/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/claimed-framework/claimed/blob/main/LICENSE)

**CLAIMED** is a framework for building, packaging, and executing portable AI components at scale.

---

## What is CLAIMED?

CLAIMED began life as a **library of reusable AI and data components** — ready-made,
shareable building blocks for the recurring chores of machine learning: reading and
writing data from COS/S3, NLP preprocessing, model training, and benchmarking. That
component library is still here (`claimed.components.*`) and still the foundation
everything else builds on.

What turned CLAIMED from a handy library into a genuinely powerful framework is the
**C3 Component Compiler**. C3 takes *any* notebook, Python script, or R script and
compiles it into a fully containerised, runnable, composable component — complete with
a CLI, a Docker image, and KubeFlow Pipelines / CWL / Kubernetes descriptors — with
zero boilerplate. Suddenly every piece of code you write becomes a first-class,
reusable CLAIMED component, not just the ones someone hand-packaged in advance. C3 is
the heart of the framework today.

From there, CLAIMED grew outward by **integrating two complementary projects** rather
than reinventing them:

- **MLX** (`claimed.mlx`) brings **experiment and asset tracking** — full provenance
  for every dataset, model, and job — and powers the grid-compute backend that
  distributes component runs across heterogeneous clusters.
- **terratorch-iterate** (the `iterate2` engine, shipped inside `claimed`) adds
  **hyperparameter optimisation (HPO) and neural architecture search (NAS)** on top of
  your components, with MLflow logging, Optuna-driven search, and Ray parallelisation.

Put together, CLAIMED lets you go from a single script all the way to a tracked,
optimised, cluster-scale AI pipeline — without ever leaving the framework.

| Layer | Package / module | Purpose |
|---|---|---|
| **C3** – Component Compiler | `claimed.c3` | Turns notebooks, Python scripts, and R scripts into fully containerised, executable, composable AI components |
| **Component Library** | `claimed.components.*` | The original ready-to-use components for COS/S3 I/O, benchmarking, NLP, training, and more |
| **MLX** – ML eXchange backend | `claimed.mlx` | Tracks datasets, models, jobs and other assets; powers the grid-compute backend |
| **iterate** – HPO / NAS | `claimed` (`iterate2`) | Hyperparameter optimisation and neural architecture search via MLflow, Optuna, and Ray |

---

## Key Features

- **Zero-boilerplate packaging** – point C3 at any `.ipynb`, `.py`, or `.R` file and get a Docker image plus KFP/CWL/Kubernetes descriptors
- **Grid parallelisation** – distribute work across heterogeneous clusters with a single `claimed run` call
- **MLX asset tracking** – full provenance for every dataset, model, and job
- **CLI-first** – every component is callable as `claimed run <module> --param value`
- **KubeFlow Pipelines & Kubernetes** – first-class output formats

---

## Quick Install

```bash
pip install claimed
```

---

## Quick Example

```bash
# List files in a COS/S3 bucket
claimed run claimed.components.util.cosutils \
    --cos-connection s3://KEY:SECRET@endpoint/bucket \
    --operation ls \
    --local-path .

# Show all parameters for any module
claimed run claimed.components.util.cosutils --help
```

---

## Video Introduction

<iframe width="700" height="394"
  src="https://www.youtube.com/embed/FuV2oG55C5s"
  title="CLAIMED intro" frameborder="0" allowfullscreen></iframe>

---

# terratorch-iterate


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

## :::claimed.iterate.backbone_benchmark.benchmark_backbone

## Default and Task specification

Under each of these, as well as for the `optimization_space`, the structure of parameters and their hierarchy for `terratorch_task` follows the same as used in terratorch. The terratorch task contains the actual parameters that will be passed to terratorch.

An exception is made for `batch_size` in `optimization_space`, which should be passed in the root level and is not passed to the `terratorch_task`.

### :::claimed.iterate.benchmark_types.Defaults

### :::claimed.iterate.benchmark_types.Task

## :::claimed.iterate.benchmark_types.ParameterBounds

## :::claimed.iterate.benchmark_types.TaskTypeEnum

## :::claimed.iterate.benchmark_types.ParameterTypeEnum

## Credits

Work by Carlos Gomes (<carlos.gomes@ibm.com>).
This project was created using <https://github.ibm.com/innersource/python-blueprint>.
