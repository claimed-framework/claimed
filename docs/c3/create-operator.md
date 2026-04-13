# create_operator

Builds a Docker image and generates KFP, CWL, and Kubernetes descriptors
from a Jupyter notebook, Python script, or R script.

## CLI

```bash
c3_create_operator <source_file> [options]
```

| Option | Type | Default | Description |
|---|---|---|---|
| `source_file` | path | *required* | `.ipynb`, `.py`, or `.R` file |
| `--repository` | str | *required* | Container registry namespace, e.g. `docker.io/myuser` |
| `--version` | str | auto | Image tag; auto-detected from `image_version` variable in source |
| `--additional-files` | list | `[]` | Extra files to `ADD` into the image |
| `--dockerfile` | path | auto | Custom Dockerfile template |
| `--log-level` | str | `WARNING` | Python logging level |

## Python API

::: claimed.c3.create_operator
    options:
      members:
        - create_operator
        - create_dockerfile

## Output Files

After a successful run you will find:

| File | Description |
|---|---|
| `<name>.dockerfile` | Generated Dockerfile |
| `<name>.yaml` | KubeFlow Pipelines component spec |
| `<name>.job.yaml` | Kubernetes Job spec |
| `<name>.cwl` | CWL component descriptor |
