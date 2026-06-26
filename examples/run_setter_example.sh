#!/usr/bin/env bash
# =============================================================================
# Example: run a local trial script (no cluster, no WLM)
#
# iterate2 calls examples/bumpy_function.py directly for each trial.
# All hyperparameters are supplied via ITERATE_PARAM_<KEY> environment
# variables.  The script is responsible for:
#   - reading those variables
#   - running the computation
#   - writing "metric_name: value" lines to ITERATE_OUT_FILE
#
# The metrics section in the HPO YAML tells iterate2 which names to look for.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

iterate \
  --script            "${SCRIPT_DIR}/bumpy_function.py"  \
  --optuna-study-name bumpy_local_study                   \
  --optuna-db-path    "sqlite:///bumpy_local_hpo.db"      \
  --optuna-n-trials   20                                  \
  --hpo-yaml          "${SCRIPT_DIR}/bumpy_hpo.yaml"
