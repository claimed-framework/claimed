#!/usr/bin/env bash
# =============================================================================
# Example: iterate2 with LSF job submission and PostgreSQL coordinator
#
# iterate2 is a pure Optuna orchestrator - it knows nothing about LSF.
# For every trial it:
#   1. Samples hyperparameters via Optuna
#   2. Calls --script with all params exposed as ITERATE_PARAM_<KEY> env vars
#   3. Reads metrics from ITERATE_OUT_FILE after the script exits
#
# The trial script (examples/wlm_plugins/lsf_plugin.sh) owns ALL
# cluster concerns: venv activation, CUDA setup, bsub submission, etc.
#
# Prerequisites
#   * LSF bsub available on PATH
#   * configs/gridfm_graphkit_hpo.yaml present
#   * psycopg2-binary:  pip install 'terratorch-iterate[postgresql]'
#   * POSTGRES_URL set
#
#   export POSTGRES_URL="postgresql://user:password@host:5432/optuna_studies"
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${POSTGRES_URL:?Please set POSTGRES_URL=postgresql://user:password@host:port/dbname}"

iterate \
  --script            "${SCRIPT_DIR}/wlm_plugins/lsf_plugin.sh" \
  --optuna-study-name gridfm_lsf_postgres_hpo                    \
  --optuna-db-path    "${POSTGRES_URL}"                          \
  --parallelism       4                                          \
  --optuna-n-trials   20                                         \
  --hpo-yaml          "${REPO_ROOT}/configs/gridfm_graphkit_hpo.yaml"
