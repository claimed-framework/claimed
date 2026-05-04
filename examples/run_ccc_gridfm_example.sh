#!/usr/bin/env bash
# =============================================================================
# Run gridfm-graphkit HPO on CCC (IBM Spectrum LSF cluster)
#
# iterate2 orchestrates Optuna trials.  For each trial it calls
# examples/wlm_plugins/ccc_plugin.sh, which owns all LSF concerns:
# venv activation, CUDA setup, and bsub submission.
#
# Prerequisites
#   * bsub / bjobs available on PATH
#   * gridfm-graphkit installed in GRIDFM_VENV
#   * configs/gridfm_graphkit_hpo.yaml present
#   * psycopg2-binary:  pip install 'terratorch-iterate[postgresql]'
#   * POSTGRES_URL exported
#
#   export POSTGRES_URL="postgresql://user:password@host:5432/optuna_studies"
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${POSTGRES_URL:?Please export POSTGRES_URL=postgresql://user:password@host:port/dbname}"

# Override via env vars if your paths differ from the plugin defaults
export GRIDFM_ROOT="${GRIDFM_ROOT:-/dccstor/terratorch/users/rkie/gitco/gridfm-graphkit}"
export GRIDFM_VENV="${GRIDFM_VENV:-/u/rkie/venvs/venv_gridfm-graphkit}"
export CUDA_BASE="${CUDA_BASE:-/opt/share/cuda-12.8.1}"

iterate \
  --script            "${SCRIPT_DIR}/wlm_plugins/ccc_plugin.sh" \
  --optuna-study-name gridfm_ccc_hpo                             \
  --optuna-db-path    "${POSTGRES_URL}"                          \
  --parallelism       4                                          \
  --optuna-n-trials   20                                         \
  --hpo-yaml          "${REPO_ROOT}/configs/gridfm_graphkit_hpo.yaml"
