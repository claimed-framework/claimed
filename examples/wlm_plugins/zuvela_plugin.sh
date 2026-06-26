#!/usr/bin/env bash
# =============================================================================
# Trial script for iterate2  -  ZuVela (IBM Spectrum LSF) backend
#
# Called once per Optuna trial.  Activates the micromamba environment, builds
# the training command from ITERATE_PARAM_* env vars, and submits via bsub -K.
#
# bsub pattern used (non-interactive, blocking):
#   bsub -gpu num=<N> -K \
#        -R "rusage[ngpus=<N>, cpu=<C>, mem=<M>GB]" \
#        -J gridfm_<trial> \
#        'cd ~/gitco/gridfm-graphkit && source ~/.bashrc && micromamba activate gridfm && <cmd>'
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
#   GRIDFM_ROOT        repo root (default: ~/gitco/gridfm-graphkit)
#   MICROMAMBA_ENV     micromamba env name (default: gridfm)
# =============================================================================

set -euo pipefail

# -- iterate2 standard vars ---------------------------------------------------
TRIAL_NUMBER="${ITERATE_TRIAL_NUMBER:?ITERATE_TRIAL_NUMBER not set}"
OUT_FILE="${ITERATE_OUT_FILE:?ITERATE_OUT_FILE not set}"
ERR_FILE="${ITERATE_ERR_FILE:?ITERATE_ERR_FILE not set}"

# -- Paths (override via env) -------------------------------------------------
GRIDFM_ROOT="${GRIDFM_ROOT:-${HOME}/gitco/gridfm-graphkit}"
MICROMAMBA_ENV="${MICROMAMBA_ENV:-gridfm}"

# -- LSF resources ------------------------------------------------------------
# GPU_COUNT comes from the HPO group param so LSF allocates the right number.
GPU_COUNT="${ITERATE_PARAM_GPU_NUM:-1}"
CPU_COUNT=16
MEM_GB=32

# -- Build training command from ITERATE_PARAM_* vars -------------------------
# --gpu_num is NOT a gridfm_graphkit flag; GPU count is controlled via bsub -gpu.
TRAIN_CMD="gridfm_graphkit train"
TRAIN_CMD+=" --batch_size  ${ITERATE_PARAM_BATCH_SIZE}"
TRAIN_CMD+=" --num_workers ${ITERATE_PARAM_NUM_WORKERS}"
TRAIN_CMD+=" --config      ${ITERATE_PARAM_CONFIG}"
TRAIN_CMD+=" --data_path   ${ITERATE_PARAM_DATA_PATH}"
# [[ -n "${ITERATE_PARAM_COMPILE:-}" ]] && TRAIN_CMD+=" --compile ${ITERATE_PARAM_COMPILE}"
[[ -n "${ITERATE_PARAM_RUN_NAME:-}"  ]] && TRAIN_CMD+=" --run_name ${ITERATE_PARAM_RUN_NAME}"
[[ -n "${ITERATE_PARAM_LOG_DIR:-}"   ]] && TRAIN_CMD+=" --log_dir ${ITERATE_PARAM_LOG_DIR}"
[[ "${ITERATE_PARAM_REPORT_PERFORMANCE:-}" == "True" ]] && TRAIN_CMD+=" --report-performance"

# -- Compose full job shell command -------------------------------------------
# source ~/.bashrc to initialise micromamba shell hooks
JOB_CMD="\
cd '${GRIDFM_ROOT}' && \
source ~/.bashrc && \
micromamba activate '${MICROMAMBA_ENV}' && \
${TRAIN_CMD}"

# -- Submit via bsub ----------------------------------------------------------
# -K  : blocks until the job completes (iterate2 runs each trial in a thread)
# -o/-e: write output to paths iterate2 will scan for metrics
bsub \
  -K \
  -gpu  "num=${GPU_COUNT}" \
  -R    "rusage[ngpus=${GPU_COUNT}, cpu=${CPU_COUNT}, mem=${MEM_GB}GB]" \
  -o    "${OUT_FILE}" \
  -e    "${ERR_FILE}" \
  -J    "gridfm_trial_${TRIAL_NUMBER}" \
  "${JOB_CMD}"

echo "[zuvela_plugin] trial ${TRIAL_NUMBER} finished"
