#!/usr/bin/env bash
# =============================================================================
# Trial script for iterate2  -  IBM Spectrum LSF backend
#
# iterate2 calls this script once per trial.  It owns ALL cluster concerns:
# activating the venv, composing the training command from env vars, and
# submitting the job via bsub.
#
# Environment variables provided by iterate2
# ------------------------------------------
#   ITERATE_TRIAL_NUMBER        integer trial ID
#   ITERATE_OUT_FILE            path where metric lines must be written
#   ITERATE_ERR_FILE            path for error output
#   ITERATE_PARAM_<KEY>         one variable per HPO + static parameter
#                               (key uppercased, hyphens -> underscores)
#
# Customise the sections marked CONFIGURE below.
#
# Exit code
# ---------
# Exit 0 on success.  iterate2 marks the Optuna trial FAILED on non-zero exit.
# =============================================================================

set -euo pipefail

# -- Read iterate2 standard vars -----------------------------------------------
TRIAL_NUMBER="${ITERATE_TRIAL_NUMBER:?ITERATE_TRIAL_NUMBER not set}"
OUT_FILE="${ITERATE_OUT_FILE:?ITERATE_OUT_FILE not set}"
ERR_FILE="${ITERATE_ERR_FILE:?ITERATE_ERR_FILE not set}"

# -- CONFIGURE: paths ----------------------------------------------------------
GRIDFM_ROOT="${GRIDFM_ROOT:-/path/to/gridfm-graphkit}"
GRIDFM_VENV="${GRIDFM_VENV:-/path/to/venv}"
CUDA_BASE="${CUDA_BASE:-/opt/share/cuda-12.8.1}"

# -- CONFIGURE: LSF resources --------------------------------------------------
GPU_COUNT=1
CPU_COUNT=16
MEM_GB=32
GPU_STRING="num=${GPU_COUNT}:mode=exclusive_process:mps=no"
# QUEUE="normal"   # uncomment to target a specific queue

# -- Build the training command from ITERATE_PARAM_* vars ----------------------
# Each HPO / static parameter is available as ITERATE_PARAM_<KEY>.
# Translate them into the CLI flags your training script expects.
TRAIN_CMD="gridfm_graphkit train"
TRAIN_CMD+=" --gpu_num   ${ITERATE_PARAM_GPU_NUM}"
TRAIN_CMD+=" --batch_size ${ITERATE_PARAM_BATCH_SIZE}"
TRAIN_CMD+=" --num_workers ${ITERATE_PARAM_NUM_WORKERS}"
TRAIN_CMD+=" --config    ${ITERATE_PARAM_CONFIG}"
TRAIN_CMD+=" --data_path  ${ITERATE_PARAM_DATA_PATH}"
# Add further params as needed:
# [[ -n "${ITERATE_PARAM_COMPILE:-}" ]] && TRAIN_CMD+=" --compile ${ITERATE_PARAM_COMPILE}"

# -- Compose the full job command ----------------------------------------------
JOB_CMD="\
export PATH='${CUDA_BASE}/bin:\$PATH' && \
export CUDA_HOME='${CUDA_BASE}' && \
export LD_LIBRARY_PATH='${CUDA_BASE}/lib64:\$LD_LIBRARY_PATH' && \
cd '${GRIDFM_ROOT}' && \
source '${GRIDFM_VENV}/bin/activate' && \
${TRAIN_CMD}"

# -- Submit via bsub -----------------------------------------------------------
# -K  blocks until the job finishes (iterate2 runs each trial in a thread).
# -o/-e redirect LSF job stdout/stderr to the files iterate2 will read.
MEM_MB=$(( MEM_GB * 1024 ))

bsub \
  -K \
  -gpu  "${GPU_STRING}" \
  -n    "${CPU_COUNT}" \
  -R    "rusage[mem=${MEM_MB}]" \
  -o    "${OUT_FILE}" \
  -e    "${ERR_FILE}" \
  -J    "hpo_trial_${TRIAL_NUMBER}" \
  "${JOB_CMD}"

echo "[lsf_plugin] trial ${TRIAL_NUMBER} finished"
