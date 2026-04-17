#!/usr/bin/env bash
# =============================================================================
# Example: iterate with --wlm lsf for gridfm-graphkit HPO on an LSF cluster
#
# Prerequisites
# -------------
#   * LSF bsub/bjobs available on PATH
#   * gridfm-graphkit installed in the venv (or via module load)
#   * configs/gridfm_graphkit_hpo.yaml present
#
# How it works
# ------------
# 1. For each Optuna trial iterate:
#    a. Samples hyperparameters from gridfm_graphkit_hpo.yaml
#    b. Builds the gridfm_graphkit CLI invocation from static + sampled params
#    c. Submits a bsub job (-K blocks until completion)
#    d. Reads stdout/stderr from trial<N>.out / trial<N>.err
#    e. Extracts metrics and reports them to Optuna
#
# Customise
# ---------
#   LSF_GPU_CONFIG   – full -gpu option string for bsub
#   GPU_COUNT        – must match num= in LSF_GPU_CONFIG
#   OC_NAMESPACE     – not used for LSF; kept as no-op for parity
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Adjust to match your cluster's GPU model and scheduling policy.
LSF_GPU_CONFIG="${LSF_GPU_CONFIG:-num=2:mode=exclusive_process:mps=no:gmodel=NVIDIAA100_SXM4_80GB}"

iterate \
  --script            "gridfm_graphkit train"                       \
  --interpreter       ""                                            \
  --wlm               lsf                                           \
  --no-underscore-to-hyphen                                         \
  --gpu-count         2                                             \
  --cpu-count         32                                            \
  --mem-gb            256                                           \
  --lsf-gpu-config-string "${LSF_GPU_CONFIG}"                       \
  --optuna-study-name gridfm_lsf_hpo                                \
  --optuna-db-path    "js:///gridfm_lsf_hpo.journal"                \
  --parallelism       4                                             \
  --optuna-n-trials   20                                            \
  --hpo-yaml          "${REPO_ROOT}/configs/gridfm_graphkit_hpo.yaml"
