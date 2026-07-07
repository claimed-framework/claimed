"""
Integration tests for the database-backed CLAIMED job coordinator.

SQLite      – always runs (uses a temp file).
PostgreSQL  – skipped unless POSTGRES_URL is set, e.g.:

    export POSTGRES_URL="postgresql://user:password@localhost:5432/claimed_test"
    pytest tests/integration/test_jobcoordinator.py -v
"""

from __future__ import annotations

import os
import tempfile
import threading

import pytest

from claimed.jobcoordinator import db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def sqlite_engine():
    with tempfile.TemporaryDirectory() as tmp:
        engine = db.get_engine(os.path.join(tmp, "jobs.db"))
        yield engine
        engine.dispose()


# ---------------------------------------------------------------------------
# URL normalisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "given,expected",
    [
        ("jobs.db", "sqlite:///jobs.db"),
        ("/tmp/x.sqlite", "sqlite:////tmp/x.sqlite"),
        ("sqlite:///already.db", "sqlite:///already.db"),
        ("postgres://u:p@h/d", "postgresql://u:p@h/d"),
        ("postgresql://u:p@h/d", "postgresql://u:p@h/d"),
    ],
)
def test_normalize_url(given, expected):
    assert db.normalize_url(given) == expected


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

JOB = "job1"


def test_propagate_and_lifecycle(sqlite_engine):
    engine = sqlite_engine
    inserted = db.insert_tasks(engine, JOB, ["a.tif", "b.tif", "c.tif"])
    assert inserted == 3
    assert db.count_pending(engine, JOB) == 3

    # Claim, then mark terminal states.
    n1 = db.claim_next(engine, JOB, "worker-1")
    n2 = db.claim_next(engine, JOB, "worker-1")
    assert {n1, n2} == {"a.tif", "b.tif"}  # oldest first
    assert db.count_pending(engine, JOB) == 1

    db.mark(engine, JOB, n1, db.SUCCEEDED)
    db.mark(engine, JOB, n2, db.FAILED, error="boom")

    n3 = db.claim_next(engine, JOB, "worker-1")
    assert n3 == "c.tif"
    db.mark(engine, JOB, n3, db.SUCCEEDED)

    assert db.claim_next(engine, JOB, "worker-1") is None  # queue drained
    assert db.stats(engine, JOB) == {db.SUCCEEDED: 2, db.FAILED: 1}


def test_propagate_is_idempotent(sqlite_engine):
    engine = sqlite_engine
    assert db.insert_tasks(engine, JOB, ["a", "b", "c"]) == 3
    # Re-propagating the same names inserts nothing.
    assert db.insert_tasks(engine, JOB, ["a", "b", "c"]) == 0
    # A partial overlap inserts only the new ones.
    assert db.insert_tasks(engine, JOB, ["c", "d", "e"]) == 2
    assert db.count_pending(engine, JOB) == 5


def test_jobs_are_isolated(sqlite_engine):
    """The same task name lives independently in different jobs, and claiming
    in one job never consumes another job's tasks."""
    engine = sqlite_engine
    assert db.insert_tasks(engine, "alpha", ["shared", "a-only"]) == 2
    assert db.insert_tasks(engine, "beta", ["shared", "b-only"]) == 2

    # Drain alpha entirely.
    alpha_claimed = set()
    while (t := db.claim_next(engine, "alpha", "w")) is not None:
        alpha_claimed.add(t)
    assert alpha_claimed == {"shared", "a-only"}

    # beta is untouched.
    assert db.count_pending(engine, "beta") == 2
    beta_claimed = set()
    while (t := db.claim_next(engine, "beta", "w")) is not None:
        beta_claimed.add(t)
    assert beta_claimed == {"shared", "b-only"}


def test_concurrent_claims_are_disjoint(sqlite_engine):
    """Two threads draining one queue must claim disjoint task sets."""
    engine = sqlite_engine
    names = [f"task-{i:03d}" for i in range(60)]
    db.insert_tasks(engine, JOB, names)

    claimed: dict[str, list] = {"w1": [], "w2": []}

    def drain(worker_id):
        while True:
            task = db.claim_next(engine, JOB, worker_id)
            if task is None:
                break
            claimed[worker_id].append(task)

    t1 = threading.Thread(target=drain, args=("w1",))
    t2 = threading.Thread(target=drain, args=("w2",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    all_claimed = claimed["w1"] + claimed["w2"]
    assert sorted(all_claimed) == sorted(names)   # every job claimed once
    assert len(all_claimed) == len(set(all_claimed))  # no duplicates


# ---------------------------------------------------------------------------
# PostgreSQL (optional)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("POSTGRES_URL"),
    reason="POSTGRES_URL not set; skipping PostgreSQL job coordinator test",
)
def test_postgres_lifecycle():
    url = os.environ["POSTGRES_URL"]
    engine = db.get_engine(url)
    try:
        # Isolate from any prior run.
        with engine.begin() as conn:
            from sqlalchemy import text
            conn.execute(text("DELETE FROM claimed_jobs"))

        assert db.insert_tasks(engine, "pg", ["p1", "p2", "p3"]) == 3
        got = {db.claim_next(engine, "pg", "pw") for _ in range(3)}
        assert got == {"p1", "p2", "p3"}
        assert db.claim_next(engine, "pg", "pw") is None
        for name in ["p1", "p2", "p3"]:
            db.mark(engine, "pg", name, db.SUCCEEDED)
        assert db.stats(engine, "pg") == {db.SUCCEEDED: 3}
    finally:
        with engine.begin() as conn:
            from sqlalchemy import text
            conn.execute(text("DELETE FROM claimed_jobs"))
        engine.dispose()
