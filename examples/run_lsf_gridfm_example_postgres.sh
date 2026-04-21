#!/usr/bin/env bash
# =============================================================================
# Example: iterate --wlm lsf with PostgreSQL coordinator for gridfm-graphkit HPO
#
# Each Optuna trial is submitted as an LSF job that looks like:
#
#   bsub -gpu "num=1:mode=exclusive_process:mps=no:gmodel=NVIDIAA100_SXM4_80GB" \
#        -K -o trial<N>.out -e trial<N>.err \
#        -R "rusage[ngpus=1, cpu=16, mem=32GB]" \
#        -J hpo_trial_<N> \
#        "export PATH='/opt/share/cuda-12.8.1/bin:$PATH' && \
#         export CUDA_HOME='/opt/share/cuda-12.8.1/' && \
#         export LD_LIBRARY_PATH='/opt/share/cuda-12.8.1/lib64:$LD_LIBRARY_PATH' && \
#         cd /dccstor/terratorch/users/rkie/gitco/gridfm-graphkit && \
#         source /u/rkie/venvs/venv_gridfm-graphkit/bin/activate && \
#         gridfm_graphkit train <hpo_params> <static_params>"
#
# Prerequisites
# -------------
#   * LSF bsub/bjobs available on PATH
#   * gridfm-graphkit installed in the venv below
#   * configs/gridfm_graphkit_hpo.yaml present
#   * psycopg2-binary installed:  pip install 'terratorch-iterate[postgresql]'
#   * POSTGRES_URL set (or hard-code it in --optuna-db-path below)
#
# PostgreSQL coordinator
# ----------------------
# Using PostgreSQL instead of SQLite / JournalFS is the recommended backend for
# high-parallelism HPO on a cluster: multiple bsub jobs can safely write trial
# results concurrently without lock contention.
#
# Set the connection URL as an env-var to avoid embedding credentials in scripts
# that may end up in version control:
#
#   export POSTGRES_URL="postgresql://user:password@host:5432/optuna_studies"
#
# or pass it inline:
#
#   POSTGRES_URL="postgresql://..." bash run_lsf_gridfm_example_postgres.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Required: PostgreSQL connection URL
# ---------------------------------------------------------------------------
: "${POSTGRES_URL:?Please set POSTGRES_URL=postgresql://user:password@host:port/dbname}"

# ---------------------------------------------------------------------------
# Customisable paths – override via environment variables
# ---------------------------------------------------------------------------
GRIDFM_ROOT="${GRIDFM_ROOT:-/dccstor/terratorch/users/rkie/gitco/gridfm-graphkit}"
GRIDFM_VENV="${GRIDFM_VENV:-/u/rkie/venvs/venv_gridfm-graphkit}"
CUDA_BASE="${CUDA_BASE:-/opt/share/cuda-12.8.1}"
DATA_PATH="${DATA_PATH:-/u/rkie/}"
LOG_DIR="${LOG_DIR:-logs}"

# ---------------------------------------------------------------------------
# LSF GPU resource string
# Adjust gmodel to the GPU type available on your cluster.
# ---------------------------------------------------------------------------
LSF_GPU_CONFIG="${LSF_GPU_CONFIG:-num=1:mode=exclusive_process:mps=no:gmodel=NVIDIAA100_SXM4_80GB}"

# ---------------------------------------------------------------------------
# Pre-run commands executed inside every bsub job before the training script.
# Order matters:
#   1. Export CUDA paths so the GPU driver / toolkit is visible.
#   2. cd into the project root so relative config paths resolve correctly.
#   3. Activate the project venv.
# ---------------------------------------------------------------------------
PRE_RUN="\
export PATH='${CUDA_BASE}/bin:\$PATH' && \
export CUDA_HOME='${CUDA_BASE}' && \
export LD_LIBRARY_PATH='${CUDA_BASE}/lib64:\$LD_LIBRARY_PATH' && \
cd '${GRIDFM_ROOT}' && \
source '${GRIDFM_VENV}/bin/activate'"

# ---------------------------------------------------------------------------
# Static training arguments (not part of the HPO search space).
# These are appended verbatim after the sampled hyperparameters.
# ---------------------------------------------------------------------------
STATIC_ARGS_JSON='{
  "--config":           "./examples/config/HGNS_PF_datakit_case118.yaml",
  "--data_path":        "'"${DATA_PATH}"'",
  "--log_dir":          "'"${LOG_DIR}"'",
  "--report-performance": ""
}'

# ---------------------------------------------------------------------------
# Launch iterate
# ---------------------------------------------------------------------------
iterate \
  --script              "gridfm_graphkit train"                        \
  --interpreter         ""                                             \
  --root-dir            "${GRIDFM_ROOT}"                               \
  --wlm                 lsf                                            \
  --pre-run-commands    "${PRE_RUN}"                                   \
  --no-underscore-to-hyphen                                            \
  --gpu-count           1                                              \
  --cpu-count           16                                             \
  --mem-gb              32                                             \
  #--lsf-gpu-config-string "${LSF_GPU_CONFIG}"                          \
  --optuna-study-name   gridfm_lsf_postgres_hpo                        \
  --optuna-db-path      "${POSTGRES_URL}"                              \
  --parallelism         4                                              \
  --optuna-n-trials     20                                             \
  --hpo-yaml            "${REPO_ROOT}/configs/gridfm_graphkit_hpo.yaml" \
  --static-args-json    "${STATIC_ARGS_JSON}"
