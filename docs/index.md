# CLAIMED Framework

[![OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/projects/6718/badge)](https://bestpractices.coreinfrastructure.org/projects/6718)
[![PyPI](https://img.shields.io/pypi/v/claimed)](https://pypi.org/project/claimed/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/claimed-framework/claimed/blob/main/LICENSE)

**CLAIMED** is a framework for building, packaging, and executing portable AI components at scale.

---

## What is CLAIMED?

CLAIMED has three interlocking layers:

| Layer | Package / module | Purpose |
|---|---|---|
| **C3** – Component Compiler | `claimed.c3` | Turns notebooks, Python scripts, and R scripts into fully containerised, executable AI components |
| **MLX** – ML eXchange backend | `claimed.mlx` | Tracks datasets, models, jobs and other assets; powers the grid-compute backend |
| **Component Library** | `claimed.components.*` | Ready-to-use components for COS/S3 I/O, benchmarking, NLP, training, and more |

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
