# C3 – CLAIMED Component Compiler

C3 automates the transformation of arbitrary code assets into fully portable, executable AI components.

---

## What C3 does

```
 ┌──────────────────────┐
 │  .ipynb / .py / .R   │   ← your code
 └──────────┬───────────┘
            │  c3_create_operator
            ▼
 ┌──────────────────────────────────────────┐
 │  Dockerfile           (build + push)     │
 │  KubeFlow component YAML                 │
 │  Kubernetes Job YAML                     │
 │  CWL component descriptor                │
 └──────────────────────────────────────────┘
```

C3 reads **parameter declarations** from the top of your source file:

```python
import os

# description of my_param
my_param = os.environ.get('my_param', 'default_value')
```

Each `os.environ.get(...)` line is parsed into a typed, documented parameter
that appears in the generated YAML descriptors and KFP UI.

---

## Modules

| Module | CLI entry-point | Purpose |
|---|---|---|
| [`create_operator`](create-operator.md) | `c3_create_operator` | Build container images and component descriptors |
| [`create_gridwrapper`](create-gridwrapper.md) | `c3_create_gridwrapper` | Wrap a component for parallel grid execution |
| [`create_containerless_operator`](create-operator.md) | `c3_create_containerless_operator` | Containerless variant (runs in-process) |
| [`operator_utils`](operator-utils.md) | – | Shared helpers (connection strings, logging) |
| `parser` | – | Source-file parameter parser |
| `notebook` | – | Jupyter notebook handler |
| `pythonscript` | – | Python script handler |
| `rscript` | – | R script handler |

---

## Grid Compute Backends

| Backend key | Description |
|---|---|
| `local` | Plain local filesystem |
| `cos` / `cos_grid_wrapper` | IBM Cloud Object Storage |
| `s3kv` | S3-backed key-value store (MLX) |
| `simple_grid_wrapper` | Minimal wrapper – source folder only |
| `folder_grid_wrapper` | Source **and** target folder variant |
| `legacy_cos_grid_wrapper` | Older COS format, kept for backwards compatibility |
