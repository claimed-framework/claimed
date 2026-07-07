# Job Coordinator example

A database-backed coordinator that turns any shell script into a distributed
worker. One command fills a **job** (a batch of tasks) from a set of files;
another pulls the job's **tasks** one-by-one and runs your worker for each. The
database (SQLite or PostgreSQL) holds the queue and provides atomic claiming, so
many worker sessions — on the same machine or across many nodes — can cooperate
without a shared filesystem and without ever processing the same task twice.

See [`../../docs/jobcoordinator.md`](../../docs/jobcoordinator.md) for the full
reference.

## 1. Create some input files

```bash
mkdir -p /tmp/jc/data
touch /tmp/jc/data/a.tif /tmp/jc/data/b.tif /tmp/jc/data/c.tif
```

## 2. Propagate tasks (status: pending)

Every batch is namespaced by a mandatory `--job`, so many independent jobs can
share one database without colliding.

Local SQLite (no server needed — a bare path becomes a SQLite database):

```bash
claimed propagate_jobs --db /tmp/jc/jobs.db --job scenes-2026 '/tmp/jc/data/**/*.tif'
# Inserted 3 tasks into job 'scenes-2026' (0 already existed). Total pending: 3
```

PostgreSQL (real multi-node coordination):

```bash
claimed propagate_jobs \
  --db postgresql://user:pass@localhost:5432/claimed \
  --job scenes-2026 \
  '/data/scenes/**/*.tif'
```

Quote the glob so your shell does not expand it before CLAIMED sees it.
`propagate_jobs` is idempotent: re-running skips `(job, task_name)` pairs that
already exist.

## 3. Work the queue

```bash
claimed work_jobs --db /tmp/jc/jobs.db --job scenes-2026 --worker ./worker.sh
```

Each task is claimed atomically and marked `processing`, then `succeeded` /
`failed` based on the worker's exit code. The task name (the file path) is passed
to the worker as `$1` **and** as `$CLAIMED_TASK` (with `$CLAIMED_JOB` and
`$CLAIMED_WORKER_ID` also exported). Workers only claim tasks from their own
`--job`.

Run several workers at once for parallelism — just launch the command multiple
times (or on multiple nodes) against the same `--db` and `--job`:

```bash
claimed work_jobs --db /tmp/jc/jobs.db --job scenes-2026 --worker ./worker.sh &
claimed work_jobs --db /tmp/jc/jobs.db --job scenes-2026 --worker ./worker.sh &
wait
```

### Useful options

- `--worker-id NAME`   label recorded on claimed tasks (default `<host>-<pid>`)
- `--max-tasks N`      stop after N tasks (default: drain the queue)
- `--poll-interval S`  when the queue is empty, wait S seconds and retry instead
                       of exiting — handy when producers keep adding tasks

## Writing your own worker

`worker.sh` here is a template. Read the task name from `$1` or
`$CLAIMED_TASK`, do your work, and exit `0` on success or non-zero on
failure. Anything executable works — a Python script, a `bsub`/`sbatch`
submission wrapper, a container invocation, etc.
