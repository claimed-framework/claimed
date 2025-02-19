# Handover

## Benchmark Basics

Benchmark has two main purposes:

- Allowing for a model to be trained on a variety of tasks
- Optionally performing hyperparameter tuning on each of those tasks

In that sense, we can visualize its operation as two loops:

```
given: model_defaults, hparam_space

for task in tasks:
    model_setup = model_defaults + task_specific_model_settings
    for trial in range(n_trials):
        hparams = get_next_hparams(hparam_space)
        save_results(train(model_setup, task, hparams))
```

Benchmark only actually handles setting up the two loops and saving the final results. Everything else is outsourced to the following major components:

- Training Engine (train() function): **Terratorch.** We use terratorch to handle the training of all the models. In order to do this, we need to build the appropriate configuration for each combination of model, task and hparams.

- Experiment Tracking: **MLFLow.** We leverage mlflow to create a hierarchy of runs that mirrors the hierarchy of loops above. The `save_results` function then only queries mlflow to collect the final results.

- Hyperparameter search (get_next_hparams()): **Optuna / Ray Tune**. We leverage optuna to decide how to best explore the parameter space. When using ray, we also leverage Optuna, although this is abstracted as a search engine which can be easily replaced with others.

### Parallelization

For large jobs, it may be useful to parallelize over multiple GPUs. Benchmark implements this as distributing training jobs over individual GPUs (each instance of the `train()` function has access to only 1 GPU). To do this, we leverage ray.

#### No hparam search

When there is no hyperparameter search, the above structure is the same, except `train()` becomes a `ray.remote` function. This essentially means it is asynchronous, with each call being queued up for ray to run in an available worker on the cluster.

```
given: model_defaults

for task in tasks:
    model_setup = model_defaults + task_specific_model_settings
    save_results(ray_train(model_setup, task, hparams)) # ray_train is asynchronous. It will call this function on an available worker
```

#### Hparam search

When we do have hyperparameters to search over, we leverage Ray Tune, with the above loop being transformed into:

```
given: model_defaults, hparam_space

for task in tasks:
    model_setup = model_defaults + task_specific_model_settings
    ray_tune(n_trials, model, task, hparam_space)
```

We give up full control of the second loop to ray tune, which now handles the whole process for us. However, notice that we are parallelizing **within** tasks. 
This is because ray tune only works with one task at a time and, unfortunately, multiple ray tune instances cannot be launched on the same ray cluster.

## The Benchmark Spec

From the above section, we can now identify what information we require from the user in order to perform a benchmarking run:

- The model architecture, split into:
    - model defaults, applied over all tasks
    - task specific settings, which overwrite / complete the defaults

- Dataset specification for each task

- Hyperparameter space definition

- a few other admin details, like paths for storage, ...

With these details, we must iterate over the tasks' datasets, exploring the hyperparameter space, with the final requirement for each instance of the `train` function being a complete terratorch task which contains:

- the datamodule for that task
- the model architecture combining (with each element overwriting the previous):
    - the defaults
    - the task architecture
    - the hyperparameters for that trial

So, the structure of benchmark inputs will be configuration files where the above components are specified in a way as similar to terratorch configs as possible, in order to make our life easier.

For example:

``` yaml
experiment_name: geobench_v2_test # used for mlflow experiment name
run_name: test_models_saved_multiple_epochs_no_ray # used for mlflow run name

defaults:
  trainer_args:
    max_epochs: 300
    ...
  terratorch_task:
    ... # anything youd pass to a terratorch task
    
tasks:
    # parameters which are outside the scope of the terratorch task
  - name: chesapeake
    type: segmentation
    direction: max
    metric: val/Multiclass_Jaccard_Index
    early_stop_patience: 50
    terratorch_task:
      # anything youd pass to a terratorch task
    datamodule:
      # exactly what you would pass to a terratorch datamodule
  - # next task

n_trials: 16
save_models: False
# admin parameters
storage_uri: /dccstor/geofm-finetuning/carlosgomes/benchmark
ray_storage_path: /dccstor/geofm-finetuning/carlosgomes/ray_storage
#
optimization_space:
  batch_size:
      - 8
      - 32
      - 64
  lr:
    max: 1e-3
    min: 1e-6
    type: real
    log: true
  optimizer_hparams:
    weight_decay:
      min: 0
      max: 0.4
      type: real
  model_args:
    decoder_channels:
      - 64
      - 128
      - 256
```

## The nasty implementation bits

### Creating terratorch configs

Benchmark does not actually assemble a yaml config file for each `train()`. Instead, it uses the programatic interface of terratorch. 

See e.g. `model_fitting.py:fit_model` and `model_fitting.py:launch_training`

In order to do all the merging, there is a reasonable amount of dictionary merging going on, which could be a source of bugs if not done carefully.

e.g. `benchmark_types.py`

``` python

def recursive_merge(first_dict: dict[str, Any], second_dict: dict[str, Any]):
    # consider using deepmerge instead of this
    for key, val in second_dict.items():
        if key not in first_dict:
            first_dict[key] = val
        else:
            # if it is a dictionary, recurse deeper
            if isinstance(val, dict):
                recursive_merge(first_dict[key], val)
            # if it is not further nested, just replace the value
            else:
                first_dict[key] = val

def combine_with_defaults(task: Task, defaults: Defaults) -> TrainingSpec:
    terratorch_task = copy.deepcopy(defaults.terratorch_task)
    recursive_merge(terratorch_task, task.terratorch_task)
    task_with_defaults = replace(task, terratorch_task=terratorch_task)
    return TrainingSpec(task_with_defaults, defaults.trainer_args)
```

and `model_fitting.py`

```python
def inject_hparams(training_spec: TrainingSpec, config: dict):
    # treat batch size specially
    config_without_batch_size = copy.deepcopy(config)
    batch_size: int | None = config_without_batch_size.pop("batch_size", None)  # type: ignore
    datamodule_with_generated_hparams = copy.deepcopy(training_spec.task.datamodule)
    if batch_size:
        datamodule_with_generated_hparams.batch_size = batch_size

    terratorch_task_with_generated_hparams = copy.deepcopy(
        training_spec.task.terratorch_task
    )
    recursive_merge(terratorch_task_with_generated_hparams, config_without_batch_size)

    task_with_generated_hparams = dataclasses.replace(
        training_spec.task,
        terratorch_task=terratorch_task_with_generated_hparams,
        datamodule=datamodule_with_generated_hparams,
    )
    training_spec_with_generated_hparams = dataclasses.replace(
        training_spec, task=task_with_generated_hparams
    )
    return training_spec_with_generated_hparams
```

### Config parsing

To parse the config file, we use `jsonargparse`. It is in principle very nice, because it can directly instantiate objects from the config and pass them to the main function. It is the same library that `LightningCLI` uses under the hood.

We use this to instantiate the datamodule, and it works!

Ideally, we would also use this to instantiate the Trainer, any Callbacks, ...
However, we run into sneaky bugs here. This is because, for every new `train()`, we need a fresh new instance of a lightning Trainer, as well as all callbacks which store some state.

In order to deal with this as cleanly as possible, we define a few dataclasses as types in `benchmark_types.py`. These try to help the parser out by defining the keys it can expect as much as possible. For the ones we cannot know (e.g. anything that can be passed into a terratorch task, or a lightning trainer), we leave them as dicts. 

This results in some loss of type checking in return for flexibility.

This is the reason for things such as `trainer_args`, `terratorch_task` and specific arguments such as `early_stop_patience` under the tasks. With a bit more work, its possible this could be done more cleanly.


### Tight integration with MLFLow

The current state of benchmark has a very tight coupling with MLFlow. However, in theory, we could use any other experiment tracker. This would take some refactoring, but may be worth it in the long run.

## The nasty implementation bits (ray)

### Code duplication

Currently, there is an entirely different file which serves as the entrypoint for benchmarking with ray. A few of the functions in the `model_fitting.py` module, with both ray and non-ray benchmarking share, also have some slight duplication. This means that we need to remember to implement a feature / fix a big in two different places, which is not great. However, we should keep in mind that for opensourcing the current idea is to do that only for the non-ray bit! So there are some arguments for keeping them split and having duplication.

### Ray + MLFlow + Lightning integration

The potentially more troublesome bits come in this part.

There are tutorials and examples provided by Ray on the integration of [ray tune with mlflow and lightning](https://docs.ray.io/en/latest/tune/examples/includes/mlflow_ptl_example.html).

However, at the time of writing, these seemed to be experimental and somewhat lacking in features.
In particular, for the desired nesting of runs within runs, this quickly became a mess.

There are also a variety of them which at some points contradict each other...

### `ray.air.integrations.mlflow`

Thus, for development, we explicitly disregarded the use of the utilities provided by `from ray.air.integrations.mlflow` such as `setup_mlflow` and the MLFLow Callback.

This comes with some disadvantages:

- We must handle all the setup of mlflow ourselves

- Logging of models saved by ray as mlflow artifacts is not done

For example, so set the parent run for mlflow on a run started by a ray worker, we do:

``` python
with mlflow.start_run(run_name=run_name, nested=True) as run:
        mlflow.set_tag("mlflow.parentRunId", parent_run_id)
```

However, it gives us the flexibility to log our runs as desired.

### Model checkpointing

Similarly here, the lack of clear guidance in the Ray Tune docs made this a challenge.
There are several reasonable places that could take care of model logging and tracking:

- The usual lightning ModelCheckpoint

- The MLFLow logger

- The Ray Tune instance

In order to maintain compatibility with some features of Ray, such as BOHB which may interrupt runs and resume them, when using Ray, Ray Tune takes care of the checkpointing.

This comes with some disadvantages:

- The model name is worse

- We do not log the model in mlflow as a model, but only a generic artifact.

When not using ray, the lightning ModelCheckpoint is used.

## Missing features

### Tracking

Currently, the biggest clear missing feature has to do with tracking. When running a benchmark, a copy of the config used to run the benchmark should be made and stored, and associated with the benchmark.

Without this, the only parameters we track are those the terratorch logs, which are only the ones within `terratorch_task`. The defaults, the path to the model weights, the hyperparameter space amongst others are currently lost, and require the user saves the config file themselves.

Furthermore, the results csvs that are produced should point back to the mlflow run id that created it.

It should ideally be very easy to see all of these things in mlflow and always be able to find the config and pretrained weights which generated a file.

### MLFlow corrupted runs

For some reason, after some time, mlflow runs some runs start to be corrupted. This causes the entire mlflow experiment they are under to not be rendered by `mlflow ui`. In order to solve this, the current solution is to run `mlflow_corrupted.py` to find and delete the corrupted run.

This is probably a deeper issue in mlflow that would need to be investigated.