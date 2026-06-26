#!/usr/bin/env bash
# =============================================================================
# Example: iterate2 with --lsf-gpu-config-string on an LSF cluster
#
# By default, iterate2 generates a simple bsub -gpu num=<N> fragment.
# For advanced GPU scheduling (exclusive process mode, MPS, specific GPU model)
# pass the full LSF -gpu option string via --lsf-gpu-config-string.
#
# The value is inserted verbatim as:
#   bsub -gpu "<lsf-gpu-config-string>" -K ...
#
# Resulting bsub command will resemble:
#   bsub -n 20 -R "span[hosts=1]" \
#        -gpu "num=1:mode=exclusive_process:mps=yes:gmodel=NVIDIAA100_SXM4_80GB" \
#        -M 512G -J hpo_trial_0 "..."
#
# Note: --gpu-count still controls the rusage reservation string; set it to
#       match the num= value in your --lsf-gpu-config-string.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

iterate \
  --script        "${SCRIPT_DIR}/bumpy_function.py"  \
  --root-dir      "${SCRIPT_DIR}"                    \
  --venv          ""                                 \
  --wlm           lsf                                \
  --gpu-count     1                                  \
  --cpu-count     20                                 \
  --mem-gb        512                                \
  --lsf-gpu-config-string \
      "num=1:mode=exclusive_process:mps=yes:gmodel=NVIDIAA100_SXM4_80GB" \
  --optuna-study-name  bumpy_lsf_gpu_study           \
  --optuna-db-path     "sqlite:///bumpy_lsf_gpu_hpo.db" \
  --optuna-n-trials    20                            \
  --hpo-yaml      "${SCRIPT_DIR}/bumpy_hpo.yaml"     \
  --metric        "yval"
