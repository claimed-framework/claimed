"""Command-line entry points for the CLAIMED database-backed job coordinator.

A **job** (``--job``) is a whole batch/run; a **task** is one unit of work
within it (typically a file path).  Two verbs, dispatched from
``claimed.claimed.main``:

``claimed propagate_jobs --db <url> --job <name> <pattern>``
    Expand *pattern* (a glob or directory) and insert each matching path as a
    pending task of the job.  Idempotent -- re-running skips tasks that already
    exist.

``claimed work_jobs --db <url> --job <name> --worker <script> [options]``
    Pull the job's pending tasks one-by-one, running *script* per task.  The
    task name (the file path) is passed to the worker both as the first
    positional argument (``$1``) and as the ``CLAIMED_TASK`` environment
    variable.  Exit code 0 marks the task ``succeeded``; anything else marks it
    ``failed``.

Any executable can act as a worker -- see ``examples/jobcoordinator_example/``.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import socket
import subprocess
import sys
import time

from claimed.jobcoordinator import db

logger = logging.getLogger("claimed.jobcoordinator.cli")


# ---------------------------------------------------------------------------
# propagate_jobs
# ---------------------------------------------------------------------------

def _expand_pattern(pattern: str) -> list:
    """Return the list of task names for *pattern* (glob or directory)."""
    if os.path.isdir(pattern):
        return sorted(
            os.path.join(pattern, name) for name in os.listdir(pattern)
        )
    # recursive=True enables the ** wildcard, e.g. dir/**/*.tif
    return sorted(glob.glob(pattern, recursive=True))


def propagate_jobs(argv) -> int:
    parser = argparse.ArgumentParser(
        prog="claimed propagate_jobs",
        description="Create pending tasks for a job in the coordinator database "
                    "from a glob or directory.",
    )
    parser.add_argument("--db", required=True, help="Database URL or SQLite file path.")
    parser.add_argument(
        "--job", required=True,
        help="Job name. Namespaces this batch of tasks so multiple independent "
             "jobs can share one database. Workers only process their own job.",
    )
    parser.add_argument(
        "pattern",
        help="Glob pattern (e.g. 'data/**/*.tif') or a directory. Quote globs so the "
             "shell does not expand them.",
    )
    args = parser.parse_args(argv)

    names = _expand_pattern(args.pattern)
    if not names:
        print(f"No files matched pattern: {args.pattern!r}", file=sys.stderr)
        return 1

    engine = db.get_engine(args.db)
    inserted = db.insert_tasks(engine, args.job, names)
    already = len(names) - inserted
    pending = db.count_pending(engine, args.job)
    print(
        f"Inserted {inserted} tasks into job '{args.job}' "
        f"({already} already existed). Total pending: {pending}"
    )
    return 0


# ---------------------------------------------------------------------------
# work_jobs
# ---------------------------------------------------------------------------

def _run_worker(worker_script: str, task_name: str, job: str,
                worker_id: str) -> subprocess.CompletedProcess:
    """Invoke the worker with the task name as $1 and via CLAIMED_TASK."""
    env = dict(os.environ)
    env["CLAIMED_TASK"] = task_name
    env["CLAIMED_JOB"] = job
    env["CLAIMED_WORKER_ID"] = worker_id
    return subprocess.run([worker_script, task_name], env=env)


def work_jobs(argv) -> int:
    parser = argparse.ArgumentParser(
        prog="claimed work_jobs",
        description="Pull a job's pending tasks one-by-one and run a worker script "
                    "for each.",
    )
    parser.add_argument("--db", required=True, help="Database URL or SQLite file path.")
    parser.add_argument(
        "--job", required=True,
        help="Job to process. Must match the job used in propagate_jobs. "
             "Workers only claim tasks from this job.",
    )
    parser.add_argument(
        "--worker", required=True,
        help="Path to the worker script/executable. Receives the task name as $1 "
             "and as $CLAIMED_TASK.",
    )
    parser.add_argument(
        "--worker-id", default=None,
        help="Identifier recorded on claimed tasks. Default: <hostname>-<pid>.",
    )
    parser.add_argument(
        "--max-tasks", type=int, default=None,
        help="Stop after processing this many tasks (default: drain the queue).",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=0.0,
        help="When the queue is empty, wait this many seconds and retry instead of "
             "exiting. 0 (default) exits as soon as the queue is empty.",
    )
    args = parser.parse_args(argv)

    worker_id = args.worker_id or f"{socket.gethostname()}-{os.getpid()}"
    engine = db.get_engine(args.db)

    processed = 0
    while args.max_tasks is None or processed < args.max_tasks:
        task_name = db.claim_next(engine, args.job, worker_id)
        if task_name is None:
            if args.poll_interval > 0:
                logger.info("Queue empty; sleeping %.1fs before retry.", args.poll_interval)
                time.sleep(args.poll_interval)
                continue
            break

        print(f"[{worker_id}] processing: {task_name}")
        try:
            result = _run_worker(args.worker, task_name, args.job, worker_id)
        except Exception as exc:  # worker could not be launched at all
            db.mark(engine, args.job, task_name, db.FAILED, error=str(exc))
            print(f"[{worker_id}] FAILED (launch error): {task_name}: {exc}", file=sys.stderr)
        else:
            if result.returncode == 0:
                db.mark(engine, args.job, task_name, db.SUCCEEDED)
                print(f"[{worker_id}] succeeded: {task_name}")
            else:
                db.mark(
                    engine, args.job, task_name, db.FAILED,
                    error=f"worker exited with code {result.returncode}",
                )
                print(
                    f"[{worker_id}] FAILED (exit {result.returncode}): {task_name}",
                    file=sys.stderr,
                )
        processed += 1

    summary = db.stats(engine, args.job)
    summary_str = ", ".join(f"{k}={v}" for k, v in sorted(summary.items())) or "(empty)"
    print(
        f"[{worker_id}] done. Processed {processed} task(s) this run. "
        f"Job '{args.job}': {summary_str}"
    )
    return 0


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_VERBS = {
    "propagate_jobs": propagate_jobs,
    "work_jobs": work_jobs,
}


def main(verb: str, argv) -> None:
    """Entry point called from ``claimed.claimed.main`` for the two job verbs."""
    logging.basicConfig(
        level=os.environ.get("CLAIMED_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    handler = _VERBS.get(verb)
    if handler is None:  # pragma: no cover - guarded by caller
        print(f"Unknown job coordinator verb: {verb}", file=sys.stderr)
        sys.exit(2)
    sys.exit(handler(argv))
