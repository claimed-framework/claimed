[![OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/projects/6718/badge)](https://bestpractices.coreinfrastructure.org/projects/6718)
[![GitHub](https://img.shields.io/badge/issue_tracking-github-blue.svg)](https://github.com/claimed-framework/component-library/issues)


# C3 - the CLAIMED Component Compiler

**TL;DR**
- takes arbitrary assets (Jupyter notebooks, python scripts, R scripts) as input
- automatically creates container images and pushes to container registries
- automatically installs all required dependencies into the container image
- creates KubeFlow Pipeline components (target workflow execution engines are pluggable)
- creates Kubernetes job configs for execution on Kubernetes/Openshift clusters
- can be triggered from CICD pipelines


C3 (CLAIMED Component Compiler) is the central project of the CLAIMED framework.
It automates the transformation of arbitrary code assets — such as Jupyter notebooks, Python scripts, or R scripts — into fully portable, executable AI components.

While the component library is now maintained primarily as an example repository, C3 is where active development and innovation take place.
The most utilized and powerful feature of C3 is grid compute parallelization, enabling distributed execution of AI workloads across heterogeneous compute environments.

## MLX Integration

The Machine Learning eXchange (MLX) is now fully integrated as the backend for C3’s grid computing system, responsible for tracking all assets, including:

- data

- models

- jobs

- and other related resources

This integration allows C3 to seamlessly manage asset lifecycle, provenance, and discovery within a unified infrastructure.

To learn more on how this library works in practice, please have a look at the following [video](https://www.youtube.com/watch?v=FuV2oG55C5s)

## Getting started 

### Install

```sh
pip install claimed
```

### Usage

Just run the following command with your python script or notebook: 
```sh
c3_create_operator "<your-operator-script>.py" --repository "<registry>/<namespace>"
```

Your code needs to follow certain requirements which are explained in [Getting Started](https://github.com/claimed-framework/c3/blob/main/GettingStarted.md). 


## Getting Help

```sh
c3_create_operator --help
```

We welcome your questions, ideas, and feedback. Please create an [issue](https://github.com/claimed-framework/component-library/issues) or a [discussion thread](https://github.com/claimed-framework/component-library/discussions).
Please see [VULNERABILITIES.md](VULNERABILITIES.md) for reporting vulnerabilities.

## Contributing to CLAIMED
Interested in helping make CLAIMED better? We encourage you to take a look at our 
[Contributing](CONTRIBUTING.md) page.


## Credits

**CLAIMED** is supported by the EU’s Horizon Europe program under Grant Agreement number **101131841** and also received funding from the Swiss State Secretariat for Education, Research and Innovation (**SERI**) and the UK Research and Innovation (**UKRI**).

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Flag_of_Europe.svg/960px-Flag_of_Europe.svg.png" width="33%">

**Co-Funded by the European Union**

## License
This software is released under Apache License v2.0.

---

# terratorch-iterate (bundled)

# TerraTorch-iterate

A tool for benchmarking and hyper-parameter optimization using [TerraTorch](https://github.ibm.com/GeoFM-Finetuning/terratorch).

Leverages MLFlow for experiment logging, optuna for hyperparameter optimization and ray for parallelization.

## Installation

We recommend using python 3.10, 3.11 or 3.12 and also using a virtual environment for all commands in this guide.

### Package installation

```sh
pip install terratorch-iterate
```

### New instructions for iterate v0.3
Iterate v0.3 can optimize over arbitrary code running on arbitrary workload managers. 
Slurm and LSF are supported, Kubernetes/OpenShift and PBS coming soon.

From version 0.3 on the current iterate can be used using `iterate-classig`. Here are some usage examples

#### Prerequisites
mkdir deleteme.iterate  
cd deleteme.iterate  
python -m venv .venv  
source ./venv/bin/activate  
wget https://raw.githubusercontent.com/terrastackai/iterate/refs/heads/main/examples/bumpy_function.py  
wget https://raw.githubusercontent.com/terrastackai/iterate/refs/heads/main/examples/bumpy_hpo.yaml  
pip install terratorch-iterate==0.3  

#### Run locally
```
iterate \
        --script bumpy_function.py \
        --root-dir . \
        --optuna-study-name terratorch_hpo_nas_2 \
        --optuna-db-path "sqlite:///iterate_study.db" \
        --hpo-yaml bumpy_hpo.yaml \
        --wlm none \
        --metric yval
```
#### Run on LSF
```
iterate \
        --script bumpy_function.py \
        --root-dir . \
        --optuna-study-name terratorch_hpo_nas_2 \
        --optuna-db-path "sqlite:///iterate_study.db" \
        --hpo-yaml bumpy_hpo.yaml \
        --wlm lsf \
        --metric yval \
        --gpu-count 0
```
#### Useful commands
```
pip install optuna-dashboard
optuna-dashboard --host 0.0.0.0 sqlite:///iterate_study.db
```
### Suggested setup for development

```sh
git clone https://github.com/IBM/terratorch-iterate.git
cd terratorch-iterate
pip install --upgrade pip setuptools wheel
pip install -e .
pip install -e ".[dev]"
pip install -e ".[test]"
pip install -e ".[utilities]"
```

## Usage for benchmarking

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

- `optimization_space`: Hyperparameter space to search over. Bayesian optimization tends to work well with a small number of hyperparameters.

- `run_repetitions`: number of repetitions to be run using best setting from optimization. 

- `storage_uri`: Location to use for mlflow storage for the hyperparameter optimization (hpo) stage. During optimization, additional folders will be created in parent directory of `storage_uri`. For example, if `storage_uri` is `/opt/benchmark_experiments/hpo`, additional folders will include: 
```
/opt/benchmark_experiments/
        └──hpo_results
        └──hpo_repeated_exp
        └──repeated_exp_output_csv
        └──job_logs
        └──optuna_db
```

Please see `configs/benchmark_v2_template.yaml` in the git repo for an example.

Besides the `--config` argument, terratorch-iterate also has two other arguments: 
* if users include `--hpo` argument, then terratorch-iterate will optimize hyperparameters. Otherwise, it will rerun best experiment 
* if users include `--repeat` argument, then terratorch-iterate will repeat the best experiment. Otherwise, terratorch-iterate will not rerun any experiment

If users want to optimize hyperparameters:
```shell
terratorch iterate --hpo --config <config-file>
```

Another way to run terratorch-iterate is to omit `terratorch` by running:
```shell
iterate --hpo --config <config-file>
```

For instance:
```shell
iterate --hpo --config configs/dofa_large_patch16_224_upernetdecoder_true_modified.yaml
```


If users want to rerun best experiment, please use the same config file. Additionally, the `parent_run_id`, which is the mlflow run id from optimization, should be added as shown below:
```shell
iterate --repeat --config <config-file> --parent_run_id <mlflow run_id from hpo>
```
For instance:
```shell
iterate --repeat --config configs/dofa_large_patch16_224_upernetdecoder_true_modified.yaml --parent_run_id 61bdee4a35a94f988ad30c46c87d4fbd
```

If users want to optimize hyperparameters then the rerun best experiment in a single command, please use both settings as shown below:
```shell
iterate --hpo --repeat --config <config-file>
```
For instance:
```shell
iterate --hpo --repeat --config configs/dofa_large_patch16_224_upernetdecoder_true_modified.yaml
```

To check the experiment results, use `mlflow ui --host $(hostname -f) --port <port> --backend-store-uri <storage_uri>` 


## Summarizing results
Summarizing results and hyperparameters of multiple experiments relies on a configuration file where the experiments and tasks are defined. This consists of:

- `list_of_experiment_names`: List of MLFLOW experiment names which had been completed.

- `task_names`: List of tasks found in each experiment in `list_of_experiment_names`.

- `run_repetitions`: Number of repetitions to be expected for each experiment. This should be the same as the value used in in each experiment config file during optimization. 

- `storage_uri`: Location to use for mlflow storage for the hyperparameter optimization (hpo) stage. This should be the same value used as `storage_uri` in each experiment config file during optimization (see above).

- `benchmark_name`: string to be used to name resulting csv file



See `configs/summarize_results_template.yaml` in the git repo for an example.

To summarize results and hyperparameters, please run the following: 
```shell
iterate --summarize --config <summarize-config-file>
```
For instance:
```shell
iterate --summarize --config configs/summarize_results.yaml
```

The results and hyperparameters are extracted into a csv file. For example, if `storage_uri` is `/opt/benchmark_experiments/hpo`, then sumarized results will be saved in last file as shown below:
```
/opt/benchmark_experiments/
        └──hpo_results
        └──hpo_repeated_exp
        └──repeated_exp_output_csv
        └──job_logs
        └──optuna_db
        └──summarized_results/
            └──<benchmark_name>/
                └──results_and_parameters.csv
```


## Ray
You can also parallelize your runs over a ray cluster. 

Check out instructions in the [docs](./docs/ray.md)


## terratorch integration

terratorch-iterate provides an utility to convert terratorch's config file into a single terratorch-iterate's config file. You have to follow these steps:

1. Go to `<$TERRATORCH-ITERATE-HOME>/benchmark/config_util`
2. Copy all terratorch config files that you want to be converted into a new directory. Note that all yaml files within this directory will be parsed to generate the terratorch-iterate file, so make sure that only yaml files that you want to be parsed are located in this directory. 
3. Run build_geobench_configs.py script by specifying 3 input parameters:
     1. `input_dir`  - Full path to the directory that contains all terratorch config yaml files
     2. `output_dir` - Full path to the directory that will stored the generated files
     3. `template` - Full or relative path to the template file
     4. `prefix` - prefix to the generated file names, e.g., if set prefix to `my_prefix` then a generated filename could be `my_prefix_Clay.yaml`
   
For instance, this is an example of such command:
```shell
python3 build_geobench_configs.py --input_dir /Users/john/terratorch/examples/confs/geobenchv2_detection --output_dir /Users/john/terratorch-iterate/benchmark/config_util --template geobenchv2_template.yaml --prefix my_prefix_
```
