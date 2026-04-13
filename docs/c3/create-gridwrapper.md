# create_gridwrapper

Wraps an existing component to run in parallel over a collection of inputs
using one of several storage backends.

## CLI

```bash
c3_create_gridwrapper <source_file> [options]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `source_file` | path | *required* | `.ipynb` or `.py` component to wrap |
| `--backend` | str | `local` | Storage backend (see table below) |
| `--component-inputs` | str | `''` | Comma-separated parameter names that vary per grid cell |
| `--component-dependencies` | str | `''` | Pip dependencies to inject |
| `--repository` | str | | Container registry namespace |
| `--log-level` | str | `WARNING` | Python logging level |

## Backends

| Key | Description |
|---|---|
| `local` | Local filesystem, simple parallelism |
| `cos` | IBM COS – iterate over objects in a bucket prefix |
| `s3kv` | MLX S3 key-value store backend |
| `simple_grid_wrapper` | Source-only, minimal overhead |
| `folder_grid_wrapper` | Separate source and target folder |
| `legacy_cos_grid_wrapper` | Older COS format |

## Python API

::: claimed.c3.create_gridwrapper
    options:
      members:
        - wrap_component
        - create_gridwrapper

## Example

```bash
# Wrap a training script to process every CSV in a COS bucket in parallel
c3_create_gridwrapper train_model.py \
    --backend cos \
    --component-inputs input_file \
    --repository docker.io/myuser
```

This emits `gw_train_model.py` which, when containerised, launches one worker per input file.
