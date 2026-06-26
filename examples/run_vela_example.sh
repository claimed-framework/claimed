#!/usr/bin/env bash
# =============================================================================
# Example: iterate2 with Vela/OpenShift job submission (MLBatch PyTorchJob)
#
# iterate2 is a pure Optuna orchestrator – it knows nothing about Vela/OpenShift.
# For every trial it:
#   1. Samples hyperparameters via Optuna
#   2. Calls --script with all params exposed as ITERATE_PARAM_<KEY> env vars
#   3. Reads metrics from ITERATE_OUT_FILE after the script exits
#
# The trial script (examples/wlm_plugins/vela_plugin.py) owns ALL
# cluster concerns: job template rendering, helm/oc submission, waiting, etc.
#
# Prerequisites
# -------------
#   * helm CLI installed and on PATH
#   * oc CLI logged in to the target cluster
#   * mlbatch/tools/pytorchjob-generator/chart checked out locally
#   * configs/gridfm_graphkit_hpo.yaml present
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

iterate \
  --script            "${SCRIPT_DIR}/wlm_plugins/vela_plugin.py" \
  --optuna-study-name gridfm_vela_hpo                             \
  --optuna-db-path    "js:///gridfm_vela_hpo.journal"             \
  --parallelism       16                                          \
  --optuna-n-trials   20                                          \
  --hpo-yaml          "${REPO_ROOT}/configs/gridfm_graphkit_hpo.yaml"
