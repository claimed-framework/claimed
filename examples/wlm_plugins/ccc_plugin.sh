#!/usr/bin/env bash
# =============================================================================
# Trial script for iterate2  -  CCC (IBM Spectrum LSF) backend
#
# Called once per Optuna trial.  Activates the venv, builds the training
# command from ITERATE_PARAM_* env vars, and submits it via bsub -K.
#
# Environment variables provided by iterate2
# ------------------------------------------
#   ITERATE_TRIAL_NUMBER   integer trial ID
#   ITERATE_OUT_FILE       metric lines are read from here by iterate2
#   ITERATE_ERR_FILE       path for error output
#   ITERATE_PARAM_<KEY>    one variable per HPO + static parameter
#                          (key uppercased, hyphens -> underscores)
#
# Path overrides (set in the run script or your environment)
# ----------------------------------------------------------
#   GRIDFM_ROOT   repo root          (default below)
#   GRIDFM_VENV   Python venv path   (default below)
#   CUDA_BASE     CUDA install root  (default below)
# =============================================================================

set -euo pipefail

# -- iterate2 standard vars ---------------------------------------------------
TRIAL_NUMBER="${ITERATE_TRIAL_NUMBER:?ITERATE_TRIAL_NUMBER not set}"
OUT_FILE="${ITERATE_OUT_FILE:?ITERATE_OUT_FILE not set}"
ERR_FILE="${ITERATE_ERR_FILE:?ITERATE_ERR_FILE not set}"

# -- Paths (override via env) -------------------------------------------------
GRIDFM_ROOT="${GRIDFM_ROOT:-/dccstor/terratorch/users/rkie/gitco/gridfm-graphkit}"
GRIDFM_VENV="${GRIDFM_VENV:-/u/rkie/venvs/venv_gridfm-graphkit}"
CUDA_BASE="${CUDA_BASE:-/opt/share/cuda-12.8.1}"

# -- LSF resources ------------------------------------------------------------
# GPU_COUNT comes from the HPO group param so LSF allocates the right number.
GPU_COUNT="${ITERATE_PARAM_GPU_NUM:-1}"
CPU_COUNT=16
MEM_GB=32
MEM_MB=$(( MEM_GB * 1024 ))
GPU_STRING="num=${GPU_COUNT}:mode=exclusive_process:mps=no:gmodel=NVIDIAA100_SXM4_80GB"
# QUEUE="normal"   # uncomment to target a specific queue

# -- Build training command from ITERATE_PARAM_* vars -------------------------
# --gpu_num is NOT a gridfm_graphkit flag; GPU count is controlled via bsub -gpu.
TRAIN_CMD="gridfm_graphkit train"
TRAIN_CMD+=" --batch_size ${ITERATE_PARAM_BATCH_SIZE}"
TRAIN_CMD+=" --num_workers ${ITERATE_PARAM_NUM_WORKERS}"
TRAIN_CMD+=" --config     ${ITERATE_PARAM_CONFIG}"
TRAIN_CMD+=" --data_path  ${ITERATE_PARAM_DATA_PATH}"
# [[ -n "${ITERATE_PARAM_COMPILE:-}" ]] && TRAIN_CMD+=" --compile ${ITERATE_PARAM_COMPILE}"

# -- Compose full job shell command -------------------------------------------
JOB_CMD="\
export PATH='${CUDA_BASE}/bin:\$PATH' && \
export CUDA_HOME='${CUDA_BASE}' && \
export LD_LIBRARY_PATH='${CUDA_BASE}/lib64:\$LD_LIBRARY_PATH' && \
cd '${GRIDFM_ROOT}' && \
source '${GRIDFM_VENV}/bin/activate' && \
${TRAIN_CMD}"

# -- Submit via bsub ----------------------------------------------------------
# -K  : blocks until the job completes (iterate2 runs each trial in a thread)
# -n  : CPU slots
# -o/-e: redirect job stdout/stderr to paths iterate2 will read metrics from
bsub \
  -K \
  -gpu  "${GPU_STRING}" \
  -n    "${CPU_COUNT}" \
  -R    "rusage[mem=${MEM_MB}]" \
  -o    "${OUT_FILE}" \
  -e    "${ERR_FILE}" \
  -J    "hpo_trial_${TRIAL_NUMBER}" \
  "${JOB_CMD}"

echo "[ccc_plugin] trial ${TRIAL_NUMBER} finished"
