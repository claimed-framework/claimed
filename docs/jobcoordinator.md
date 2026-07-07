# Job Coordinator (database-backed)

The job coordinator turns **any executable** into a distributed worker. It is a
database-backed sibling of the file-based grid wrapper: instead of
`.LOCKED/.PROCESSED/.FAILED` marker files on a shared filesystem, the queue and
its coordination state live in a database (SQLite or PostgreSQL).

A **job** is a whole batch/run; a **task** is one unit of work within it
(typically a file path). Two commands make up the workflow:

| Command | Purpose |
| --- | --- |
| `claimed propagate_jobs` | Populate a job with tasks from a glob/directory (status `pending`). |
| `claimed work_jobs`      | Claim the job's tasks one-by-one and run a worker script per task. |

Because the database is the single source of truth and claiming is atomic, many
workers — on one machine or across many nodes — cooperate safely and never
process the same task twice.

## Concepts

Each task is one row in the `claimed_jobs` table, identified by its **job** (the
batch namespace) plus its **task_name** (the file path emitted by
`propagate_jobs`). The job is a mandatory namespace: it lets many independent
batches share one database, and workers only ever claim tasks from the job they
were pointed at. Uniqueness is per `(job, task_name)`, so the same file path may
appear in more than one job. A task moves through:

```
pending ── claim ──▶ processing ── worker exit 0 ──▶ succeeded
                               └── worker exit ≠0 ──▶ failed
```

The table also records `worker_id`, `attempts`, timestamps, and (on failure) an
`error` message.

## `propagate_jobs`

```bash
claimed propagate_jobs --db <url> --job <name> <pattern>
```

- `--db <url>` — a SQLite file path (e.g. `jobs.db`, `/tmp/jobs.db`) or a full
  database URL. Bare paths and `*.db` / `*.sqlite` become SQLite; `sqlite:///…`
  and `postgresql://…` (also the legacy `postgres://…`) are used as given.
- `--job <name>` — **mandatory** namespace for this batch. Use a distinct job
  name per independent run so several runs can share one database.
- `<pattern>` — a glob (with `**` recursion) or a directory. **Quote it** so the
  shell does not expand the glob before CLAIMED sees it.

The command is **idempotent**: re-running only inserts `(job, task_name)` pairs
that do not already exist (`INSERT … ON CONFLICT DO NOTHING`), so you can grow
the queue incrementally.

```bash
claimed propagate_jobs --db /tmp/jobs.db --job run-2026 'data/**/*.tif'
# Inserted 42 tasks into job 'run-2026' (0 already existed). Total pending: 42
```

## `work_jobs`

```bash
claimed work_jobs --db <url> --job <name> --worker <script> [options]
```

The worker loop claims the oldest pending task **in `--job`**, marks it
`processing`, runs the worker, then marks it `succeeded` (exit 0) or `failed`
(any other exit code or a launch error). It exits when the job's queue is
empty. The job must match the one used in `propagate_jobs`.

The task name is passed to the worker **both** ways:

- as the first positional argument (`$1`)
- as the `CLAIMED_TASK` environment variable

`CLAIMED_JOB` and `CLAIMED_WORKER_ID` are also exported.

Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--worker-id NAME`   | `<hostname>-<pid>` | Identifier recorded on claimed tasks. |
| `--max-tasks N`      | drain queue        | Stop after N tasks this run. |
| `--poll-interval S`  | `0`                | When empty, wait S seconds and retry instead of exiting. |

### Running workers in parallel

Just launch `work_jobs` more than once against the same `--db` and `--job`:

```bash
for i in 1 2 3 4; do
  claimed work_jobs --db postgresql://user:pass@host/claimed --job run-2026 --worker ./worker.sh &
done
wait
```

Two runs that use **different** `--job` values against the same database run
fully independently — neither sees or claims the other's tasks.

## Atomic claiming

Claiming is always scoped to the worker's `--job`.

- **PostgreSQL** uses `SELECT … FOR UPDATE SKIP LOCKED` so concurrent workers
  skip rows another worker is already claiming — no contention, no duplicates.
- **SQLite** has no `SKIP LOCKED`; writers are serialized with `BEGIN IMMEDIATE`
  plus a guarded `UPDATE … WHERE status='pending'` and a short retry loop, with
  WAL mode and a busy timeout enabled for cross-process concurrency.

## PostgreSQL support

PostgreSQL needs the `psycopg2` driver, available via the project extra:

```bash
pip install 'claimed[postgresql]'
```

Remote hosts should use SSL — pass it in the URL, e.g.
`postgresql://user:pass@host:5432/claimed?sslmode=require`.

## Writing a worker

Any executable works. Read the task name, do the work, and use the exit code to
report success or failure:

```bash
#!/usr/bin/env bash
set -euo pipefail
TASK="${1:-$CLAIMED_TASK}"
python process.py --input "$TASK" --output "out/$(basename "$TASK")"
```

A complete, runnable example lives in
[`examples/jobcoordinator_example/`](https://github.com/claimed-framework/claimed/tree/main/examples/jobcoordinator_example).

## Relationship to the grid wrapper

The [grid wrapper](c3/create-gridwrapper.md) coordinates work through marker
files on a shared filesystem (or object store) and wraps a CLAIMED *component*.
The job coordinator coordinates through a database and wraps an *arbitrary
script*. Use the grid wrapper when you already have a containerised component and
a shared filesystem; use the job coordinator when you want a database as the
source of truth, robust multi-node claiming, or a plain shell/Python worker.
