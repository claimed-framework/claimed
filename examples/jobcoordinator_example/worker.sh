#!/usr/bin/env bash
#
# Example CLAIMED job-coordinator worker.
#
# The coordinator (`claimed work_jobs`) invokes this script once per task and
# passes the task name (the file path recorded by `claimed propagate_jobs`) in
# two ways, so use whichever is convenient:
#
#   $1                  -> the task name (first positional argument)
#   $CLAIMED_TASK       -> the task name (environment variable)
#   $CLAIMED_JOB        -> the job (batch) this task belongs to
#   $CLAIMED_WORKER_ID  -> id of the worker session running this task
#
# Contract:
#   exit 0        -> the coordinator marks the task "succeeded"
#   exit non-zero -> the coordinator marks the task "failed" (error recorded)
#
# Replace the body with your real work: load the file at "$TASK", process it,
# and write results wherever you need.
set -euo pipefail

TASK="${1:-${CLAIMED_TASK:-}}"

if [[ -z "$TASK" ]]; then
  echo "worker.sh: no task name provided" >&2
  exit 2
fi

echo "worker ${CLAIMED_WORKER_ID:-?} handling task: $TASK"

# --- do the real work here -------------------------------------------------
# e.g. gdalinfo "$TASK"; python process.py --input "$TASK" --output out/
sleep 1
# ---------------------------------------------------------------------------

# Demo of the failure path: any task whose name contains FAIL exits non-zero.
if [[ "$TASK" == *FAIL* ]]; then
  echo "worker.sh: simulated failure for $TASK" >&2
  exit 1
fi

echo "worker done: $TASK"
