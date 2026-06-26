#!/usr/bin/env bash
# =============================================================================
# Run gridfm-graphkit HPO on ZuVela (IBM Spectrum LSF cluster)
#
# iterate2 orchestrates Optuna trials.  For each trial it calls
# examples/wlm_plugins/zuvela_plugin.sh, which owns all LSF concerns:
# micromamba environment activation and bsub submission.
#
# Prerequisites
#   * bsub / bjobs available on PATH
#   * gridfm-graphkit installed in the "gridfm" micromamba env
#   * configs/gridfm_graphkit_hpo.yaml present
#   * psycopg2-binary:  pip install 'terratorch-iterate[postgresql]'
#   * POSTGRES_URL set:  export POSTGRES_URL="postgresql://user:password@host:5432/optuna_studies"
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Override via env vars if your paths differ from the plugin defaults
export GRIDFM_ROOT="${GRIDFM_ROOT:-${HOME}/gitco/gridfm-graphkit}"
export MICROMAMBA_ENV="${MICROMAMBA_ENV:-gridfm}"

# Require a PostgreSQL URL – set via environment before calling this script:
#   export POSTGRES_URL="postgresql://user:password@host:5432/optuna_studies"
: "${POSTGRES_URL:?Please set POSTGRES_URL=postgresql://user:password@host:port/dbname}"
STUDY_DB="${POSTGRES_URL}"

iterate \
  --script            "${SCRIPT_DIR}/wlm_plugins/zuvela_plugin.sh" \
  --optuna-study-name gridfm_zuvela_hpo                             \
  --optuna-db-path    "${STUDY_DB}"                                 \
  --parallelism       4                                             \
  --optuna-n-trials   20                                            \
  --hpo-yaml          "${REPO_ROOT}/configs/gridfm_graphkit_hpo.yaml"
