"""Storage engine, schema and atomic claim logic for the CLAIMED job coordinator.

A single ``jobs`` table holds the work queue.  A **job** is a whole batch/run
(the namespace); a **task** is one unit of work within it (typically a file
path).  Each row is one task and moves through the lifecycle::

    pending -> processing -> succeeded
                          \\-> failed

Two backends are supported through one URL-based abstraction (SQLAlchemy Core):

* **PostgreSQL** -- real multi-node coordination.  Atomic claiming uses
  ``SELECT ... FOR UPDATE SKIP LOCKED`` so N workers never grab the same job.
* **SQLite** -- local / testing.  ``SELECT ... FOR UPDATE`` does not exist, so
  claiming serializes writers with ``BEGIN IMMEDIATE`` plus a guarded
  ``UPDATE ... WHERE status='pending'`` and a small retry loop.

The URL-normalisation helpers mirror the iterate2 coordinator plugins
(``terratorch_iterate/iterate2/plugin/coordinator/{sqlite,postgresql}.py``) so the
accepted ``--db`` syntax is consistent across CLAIMED.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    func,
    select,
    text,
    update,
)
from sqlalchemy.engine import Engine

logger = logging.getLogger("claimed.jobcoordinator")

# Job lifecycle states.
PENDING = "pending"
PROCESSING = "processing"
SUCCEEDED = "succeeded"
FAILED = "failed"

_SQLITE_BUSY_TIMEOUT = 30  # seconds

_metadata = MetaData()

JOBS = Table(
    "claimed_jobs",
    _metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    # job namespaces independent runs so many parallel jobs can share one
    # database. Uniqueness is per (job, task_name), not task_name alone.
    Column("job", Text, nullable=False),
    Column("task_name", Text, nullable=False),
    Column("status", String(16), nullable=False, default=PENDING),
    Column("worker_id", Text, nullable=True),
    Column("attempts", Integer, nullable=False, default=0),
    Column("created_at", Text, nullable=True),
    Column("updated_at", Text, nullable=True),
    Column("error", Text, nullable=True),
    UniqueConstraint("job", "task_name", name="uq_claimed_jobs_job_task"),
)


# ---------------------------------------------------------------------------
# URL handling
# ---------------------------------------------------------------------------

def _redact(url: str) -> str:
    """Replace the password in a DB URL with '***' for safe logging."""
    return re.sub(r"(://[^:/]+:)[^@]+(@)", r"\1***\2", url)


def normalize_url(db_url: str) -> str:
    """Turn a user-supplied ``--db`` value into a SQLAlchemy URL.

    * ``postgres://`` is rewritten to ``postgresql://`` (SQLAlchemy 1.4+ dropped
      the short form).
    * An explicit scheme (``postgresql://``, ``sqlite:///``) is passed through.
    * Anything else is treated as a filesystem path to a SQLite database.
    """
    if db_url.startswith("postgres://") and not db_url.startswith("postgresql://"):
        db_url = "postgresql://" + db_url[len("postgres://"):]
    if db_url.startswith("postgresql://") or db_url.startswith("sqlite:///"):
        return db_url
    # Bare path (foo.db, /abs/path, ./rel/path.sqlite, ...) -> SQLite.
    return f"sqlite:///{db_url}"


def is_sqlite(engine: Engine) -> bool:
    return engine.dialect.name == "sqlite"


def get_engine(db_url: str) -> Engine:
    """Create a SQLAlchemy engine and ensure the ``claimed_jobs`` table exists."""
    url = normalize_url(db_url)
    logger.info("Job coordinator database: %s", _redact(url))

    if url.startswith("sqlite:///"):
        # future=True keeps 2.0 semantics; timeout lets SQLite wait on the busy
        # write lock instead of raising immediately under contention.
        engine = create_engine(
            url,
            future=True,
            connect_args={"timeout": _SQLITE_BUSY_TIMEOUT},
        )

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, _record):  # pragma: no cover - trivial
            cur = dbapi_conn.cursor()
            # WAL improves read/write concurrency between worker processes.
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute(f"PRAGMA busy_timeout={_SQLITE_BUSY_TIMEOUT * 1000}")
            cur.close()
    else:
        if url.startswith("postgresql://"):
            _require_psycopg2()
        engine = create_engine(url, future=True, pool_pre_ping=True)

    _metadata.create_all(engine, checkfirst=True)
    return engine


def _require_psycopg2() -> None:
    try:
        import psycopg2  # noqa: F401
    except ImportError:
        raise ImportError(
            "psycopg2 is not installed but is required for PostgreSQL support.\n\n"
            "Install options:\n"
            "  pip install psycopg2-binary        # pre-built wheel, no compiler\n"
            "  pip install 'claimed[postgresql]'  # via the project extra\n"
        ) from None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Queue operations
# ---------------------------------------------------------------------------

def insert_tasks(engine: Engine, job: str, names) -> int:
    """Insert *names* as pending tasks of *job*, ignoring duplicates.

    Uniqueness is per ``(job, task_name)``, so the same file path may exist
    independently in different jobs.  Returns the number of rows inserted.
    """
    names = list(dict.fromkeys(n for n in names if n))  # de-dupe, keep order
    if not names:
        return 0

    now = _now()
    rows = [{"job": job, "task_name": n, "status": PENDING,
             "attempts": 0, "created_at": now, "updated_at": now} for n in names]

    before = count_all(engine, job)
    with engine.begin() as conn:
        # ON CONFLICT DO NOTHING is supported by both PostgreSQL and SQLite
        # (>= 3.24).  Use the dialect-specific insert to express it.
        if is_sqlite(engine):
            from sqlalchemy.dialects.sqlite import insert as _insert
        else:
            from sqlalchemy.dialects.postgresql import insert as _insert
        stmt = _insert(JOBS).values(rows)
        stmt = stmt.on_conflict_do_nothing(
            index_elements=[JOBS.c.job, JOBS.c.task_name]
        )
        conn.execute(stmt)
    after = count_all(engine, job)
    return after - before


def claim_next(engine: Engine, job: str, worker_id: str):
    """Atomically claim the oldest pending task of *job* for *worker_id*.

    Returns the claimed ``task_name`` or ``None`` when the job's queue is empty.
    Claiming is scoped to *job*, so parallel jobs in the same database never
    contend for or steal each other's tasks.
    """
    if is_sqlite(engine):
        return _claim_next_sqlite(engine, job, worker_id)
    return _claim_next_postgres(engine, job, worker_id)


def _claim_next_postgres(engine: Engine, job: str, worker_id: str):
    stmt = text(
        """
        UPDATE claimed_jobs
           SET status='processing',
               worker_id=:w,
               attempts=attempts + 1,
               updated_at=:ts
         WHERE id = (
               SELECT id FROM claimed_jobs
                WHERE status='pending' AND job=:j
                ORDER BY id
                   FOR UPDATE SKIP LOCKED
                LIMIT 1
         )
        RETURNING task_name
        """
    )
    with engine.begin() as conn:
        row = conn.execute(
            stmt, {"w": worker_id, "ts": _now(), "j": job}
        ).fetchone()
    return row[0] if row else None


def _claim_next_sqlite(engine: Engine, job: str, worker_id: str):
    # SKIP LOCKED is unavailable; serialize writers with BEGIN IMMEDIATE and a
    # guarded UPDATE.  Retry only on a lost race (0 rows updated) where another
    # writer grabbed the row we just selected.
    for _ in range(50):
        with engine.begin() as conn:
            conn.execute(text("BEGIN IMMEDIATE"))
            row = conn.execute(
                select(JOBS.c.id, JOBS.c.task_name, JOBS.c.attempts)
                .where(JOBS.c.status == PENDING, JOBS.c.job == job)
                .order_by(JOBS.c.id)
                .limit(1)
            ).fetchone()
            if row is None:
                return None
            task_id, task_name, attempts = row
            result = conn.execute(
                update(JOBS)
                .where(JOBS.c.id == task_id, JOBS.c.status == PENDING)
                .values(
                    status=PROCESSING,
                    worker_id=worker_id,
                    attempts=attempts + 1,
                    updated_at=_now(),
                )
            )
            if result.rowcount == 1:
                return task_name
        # Lost the race for this row; try again.
    logger.warning("claim_next gave up after repeated lost races (SQLite)")
    return None


def mark(engine: Engine, job: str, task_name: str, status: str,
         error: str | None = None) -> None:
    """Record a terminal *status* for *task_name* within *job*."""
    with engine.begin() as conn:
        conn.execute(
            update(JOBS)
            .where(JOBS.c.job == job, JOBS.c.task_name == task_name)
            .values(status=status, error=error, updated_at=_now())
        )


def count_all(engine: Engine, job: str) -> int:
    with engine.connect() as conn:
        return int(
            conn.execute(
                select(func.count()).select_from(JOBS)
                .where(JOBS.c.job == job)
            ).scalar_one()
        )


def count_pending(engine: Engine, job: str) -> int:
    with engine.connect() as conn:
        return int(
            conn.execute(
                select(func.count()).select_from(JOBS)
                .where(JOBS.c.status == PENDING, JOBS.c.job == job)
            ).scalar_one()
        )


def stats(engine: Engine, job: str) -> dict:
    """Return a ``{status: count}`` mapping for *job*."""
    with engine.connect() as conn:
        rows = conn.execute(
            select(JOBS.c.status, func.count())
            .where(JOBS.c.job == job)
            .group_by(JOBS.c.status)
        ).fetchall()
    return {status: int(n) for status, n in rows}
