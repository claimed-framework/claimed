#!/usr/bin/env bash
# =============================================================================
# Example: iterate2 GPU performance benchmark HPO on Vela (OpenShift / MLBatch)
#
# What iterate2 does per trial
# ----------------------------
# 1. Samples hyperparameters from configs/gpu_perf_hpo.yaml
# 2. Builds the CLI call:
#      python /app/.local/.../gpu_performance_test.py \
#        --mode single_gpu --batch-size <N> --num-workers <N> ...
# 3. Patches examples/vela_gpu_perf_template.yaml:
#      - appends  -trial-<N>  to jobName
#      - sets     numGpusPerPod = gpu_num
#      - replaces {{HPO_COMMAND}} with the CLI call (plain single-line string)
# 4. Submits:
#      helm template -f <patched.yaml> <chart> | oc create [-n <ns>] -f-
# 5. Streams:  oc logs -f <jobName>-master-0  (blocks until container exits)
# 6. Extracts metrics using Nth-occurrence syntax (name#N, 0-based):
#      Samples/sec#0 (DataLoader), Samples/sec#1 (Training),
#      Samples/sec#2 (Inference), GFLOPS
# 7. Deletes the PyTorchJob resource.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Path to the mlbatch pytorchjob-generator helm chart.
# Clone first:  git clone https://github.com/project-codeflare/mlbatch
CHART_PATH="${MLBATCH_CHART_PATH:-${HOME}/tmp/mlbatch/tools/pytorchjob-generator/chart}"

# Only pass --vela-namespace when OC_NAMESPACE is set; an empty string causes
# argparse to receive a bare "" token it treats as an unrecognised argument.
NAMESPACE_ARG=()
[[ -n "${OC_NAMESPACE:-}" ]] && NAMESPACE_ARG=(--vela-namespace "${OC_NAMESPACE}")

iterate \
  --script                 "/app/.local/lib/python3.12/site-packages/claimed/components/util/gpu_performance_test.py" \
  --interpreter            "python"                                        \
  --no-underscore-to-hyphen                                                \
  --wlm                    vela                                            \
  --vela-job-template      "${SCRIPT_DIR}/vela_gpu_perf_template.yaml"    \
  --vela-chart-path        "${CHART_PATH}"                                 \
  "${NAMESPACE_ARG[@]}"                                                    \
  --vela-pod-ready-timeout 300                                             \
  --vela-job-timeout       7200                                            \
  --gpu-count              1                                               \
  --optuna-study-name      gpu_perf_hpo                                    \
  --optuna-db-path         "js:///gpu_perf_hpo.journal"                    \
   --optuna-n-trials        250                                             \
  --hpo-yaml               "${REPO_ROOT}/configs/gpu_perf_hpo.yaml"        \
  --parallelism          5                                               \
