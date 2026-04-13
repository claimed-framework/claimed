# Getting Started

## Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.7 |
| Docker / Podman | any recent version (for building images) |
| pip | ≥ 22 |

---

## Installation

```bash
pip install claimed
```

To install directly from the repository:

```bash
git clone https://github.com/claimed-framework/claimed.git
cd claimed
pip install -e .
```

---

## Your First Component

### 1. Write a Python script (or notebook)

CLAIMED reads **parameter declarations** from the top of your script – one variable per line, with an optional comment describing it:

```python title="my_operator.py"
import os

# input CSV file path
input_file = os.environ.get('input_file', 'data.csv')

# number of rows to process
num_rows = int(os.environ.get('num_rows', 100))

# --- your logic below ---
import pandas as pd
df = pd.read_csv(input_file, nrows=num_rows)
print(df.head())
```

### 2. Build a container image

```bash
c3_create_operator my_operator.py --repository myregistry/myuser
```

C3 will:

1. Parse the parameter declarations
2. Generate a `Dockerfile`
3. Build and push the image
4. Write a KubeFlow Pipelines component YAML and a Kubernetes Job YAML

### 3. Run the component

```bash
# locally via Docker
claimed --component myregistry/myuser/my-operator \
        --input-file data.csv \
        --num-rows 50

# or directly as a Python function
claimed run my_operator --input-file data.csv --num-rows 50
```

---

## Grid Wrappers

A **grid wrapper** parallelises a component over a set of inputs:

```bash
c3_create_gridwrapper my_operator.py \
    --backend cos \
    --component-inputs "input_file" \
    --repository myregistry/myuser
```

See the [C3 overview](c3/index.md) for full details.

---

## Using the Component Library

Every module under `claimed.components` exposes a `run()` function:

```python
from claimed.components.util.cosutils import run as cos

cos(
    cos_connection='s3://KEY:SECRET@endpoint/bucket/path',
    operation='ls',
    local_path='.',
)
```

Or from the CLI:

```bash
claimed run claimed.components.util.cosutils \
    --cos-connection s3://KEY:SECRET@endpoint/bucket/path \
    --operation ls \
    --local-path .
```

---

## Next Steps

| Topic | Link |
|---|---|
| Full CLI reference | [CLI Reference](cli.md) |
| C3 internals | [C3 – Component Compiler](c3/index.md) |
| MLX asset backend | [MLX Backend](mlx/index.md) |
| COS/S3 utilities | [cosutils](components/util/cosutils.md) |
| GPU benchmarking | [gpu_performance_test](components/util/gpu-benchmark.md) |
