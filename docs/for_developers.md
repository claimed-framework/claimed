# Development

This section intends to give an insight into some of the design and implementation choices made in this package.

## Ray + MLFlow + Lightning integration

There are tutorials and examples provided by Ray on the integration of [ray tune with mlflow and lightning](https://docs.ray.io/en/latest/tune/examples/includes/mlflow_ptl_example.html).

However, at the time of writing, these seemed to be experimental and somewhat lacking in features.
In particular, for the desired nesting of runs within runs, this quickly became a mess.

### `ray.air.integrations.mlflow`

Thus, for development, we explicitly disregarded the use of the utilities provided by `from ray.air.integrations.mlflow` such as `setup_mlflow` and the MLFLow Callback.

This comes with some disadvantages:

- We must handle all the setup of mlflow ourselves

- Logging of models saved by ray as mlflow artifacts is not done

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

## Future work

- If ray integrations and tutorials improve, it would be great to use them instead of having to do all the setup ourselves

- MLFlow is slow, and after many runs they tend to get corrupted. This could be further investigated, or we could decouple from MLFlow and allow for other logging frameworks.
