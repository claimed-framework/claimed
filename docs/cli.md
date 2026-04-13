# CLI Reference

The `claimed` command is the single entry-point for the CLAIMED framework.

---

## Synopsis

```
claimed <subcommand> [options]
```

---

## Subcommands

### `claimed run`

Directly invoke the `run()` function of any CLAIMED Python module.

```
claimed run <module.path> [--param-name value ...] [--help]
```

**Arguments**

| Argument | Description |
|---|---|
| `module.path` | Fully-qualified Python module containing a `run()` function (e.g. `claimed.components.util.cosutils`) |
| `--<param-name>` | Any parameter accepted by `run()`. Hyphens are converted to underscores. |
| `--help` | Print the function signature, docstring, and parameter list, then exit. |

**Type coercion**

String values from the command line are automatically cast to the type declared in the function signature
(annotation or default-value type).  
For example, `--recursive true` is cast to `bool` if the parameter is annotated as `bool`.

**Examples**

```bash
# List objects in a COS bucket
claimed run claimed.components.util.cosutils \
    --cos-connection s3://KEY:SECRET@endpoint/bucket \
    --operation ls \
    --local-path .

# Download a file
claimed run claimed.components.util.cosutils \
    --cos-connection s3://KEY:SECRET@endpoint/bucket/file.zip \
    --operation get \
    --local-path .

# Show help for any module
claimed run claimed.components.util.cosutils --help

# CPU benchmark
claimed run claimed.components.util.gpu_performance_test \
    --mode cpu \
    --matrix-size 4096 \
    --iterations 100
```

---

### `claimed create operator`

Generate a container image + KFP/CWL/Kubernetes descriptors from a script or notebook.

```
claimed create operator <script_or_notebook> [options]
```

| Option | Description |
|---|---|
| `--repository` | Container registry namespace, e.g. `docker.io/myuser` |
| `--version` | Image tag (default: auto-detected from script) |
| `--additional-files` | Space-separated list of extra files to bundle |

Example:

```bash
claimed create operator my_script.py --repository docker.io/myuser
```

---

### `claimed create gridwrapper`

Wrap a component so it executes in parallel over a collection of inputs.

```
claimed create gridwrapper <script_or_notebook> [options]
```

| Option | Description |
|---|---|
| `--backend` | Storage backend: `local` \| `cos` \| `s3kv` \| `simple_grid_wrapper` \| `folder_grid_wrapper` |
| `--component-inputs` | Comma-separated parameter names that vary across grid cells |
| `--repository` | Container registry namespace |

Example:

```bash
claimed create gridwrapper my_script.py \
    --backend cos \
    --component-inputs input_file \
    --repository docker.io/myuser
```

---

### `claimed --component` *(legacy)*

Run a component image via Docker.

```
claimed --component <image> [--param-name value ...]
```

| Option | Description |
|---|---|
| `--component` | Docker image reference, e.g. `docker.io/claimed/my-op:latest` |
| `--<param-name>` | Environment variable to pass into the container |

Set `CLAIMED_DATA_PATH` to mount a local directory as `/opt/app-root/src/data` inside the container.

---

## Environment Variables

| Variable | Effect |
|---|---|
| `CLAIMED_DATA_PATH` | Local path mounted as `/opt/app-root/src/data` when using `--component` |
| `CLAIMED_CONTAINERLESS_OPERATOR_PATH` | Root path for containerless operator resolution |
