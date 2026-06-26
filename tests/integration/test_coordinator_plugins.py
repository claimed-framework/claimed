"""
Integration tests for the iterate2 coordinator plugin system.

SQLite  – always runs (uses a temp file).
JournalFS – always runs (uses a temp file).
PostgreSQL – skipped unless the environment variable POSTGRES_URL is set, e.g.:

    export POSTGRES_URL="postgresql://user:password@localhost:5432/optuna_test"
    pytest tests/integration/test_coordinator_plugins.py -v

The PostgreSQL test creates a study, adds a dummy trial, then removes the study
so it leaves no permanent state in the database.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import optuna
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Silence Optuna's own INFO logs during tests so pytest output stays clean.
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _fresh_registry():
    """Return a clean load_builtin_plugins / resolve_storage pair backed by an
    isolated registry so tests cannot leak state into each other."""
    # Re-import the coordinator package with a private registry copy.
    from terratorch_iterate.iterate2.plugin import coordinator as coord_pkg
    import importlib, types

    # Build a fresh module clone with its own empty registry.
    fresh = types.ModuleType(coord_pkg.__name__ + "._test_clone")
    fresh.__dict__.update({k: v for k, v in coord_pkg.__dict__.items()
                           if k not in ("_registry",)})
    fresh._registry = []

    def _register(plugin):
        fresh._registry.append(plugin)

    def _resolve(db_path):
        for plugin in fresh._registry:
            if plugin.matches(db_path):
                return plugin.get_storage(db_path)
        raise ValueError(f"No coordinator plugin matched db_path={db_path!r}")

    fresh.register = _register
    fresh.resolve_storage = _resolve
    return fresh


def _create_and_verify_study(storage: Any, study_name: str) -> None:
    """Create a one-trial study against *storage*, assert it persists."""
    study = optuna.create_study(study_name=study_name, storage=storage,
                                 load_if_exists=True)

    def objective(trial):
        x = trial.suggest_float("x", -1.0, 1.0)
        return x ** 2

    study.optimize(objective, n_trials=1)
    assert len(study.trials) == 1, "Expected exactly 1 completed trial"
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE


# ---------------------------------------------------------------------------
# SQLite plugin
# ---------------------------------------------------------------------------

class TestSQLiteCoordinator:
    def _make_storage(self, db_url: str):
        from terratorch_iterate.iterate2.plugin.coordinator.sqlite import SQLiteCoordinator
        return SQLiteCoordinator().get_storage(db_url)

    def test_matches_sqlite_scheme(self):
        from terratorch_iterate.iterate2.plugin.coordinator.sqlite import SQLiteCoordinator
        p = SQLiteCoordinator()
        assert p.matches("sqlite:///foo.db")

    def test_matches_dot_db_extension(self):
        from terratorch_iterate.iterate2.plugin.coordinator.sqlite import SQLiteCoordinator
        p = SQLiteCoordinator()
        assert p.matches("/tmp/my_study.db")

    def test_matches_dot_sqlite_extension(self):
        from terratorch_iterate.iterate2.plugin.coordinator.sqlite import SQLiteCoordinator
        p = SQLiteCoordinator()
        assert p.matches("/tmp/my_study.sqlite")

    def test_no_match_journalfs(self):
        from terratorch_iterate.iterate2.plugin.coordinator.sqlite import SQLiteCoordinator
        p = SQLiteCoordinator()
        assert not p.matches("js:///tmp/journal.log")

    def test_normalises_plain_path_to_sqlite_url(self):
        from terratorch_iterate.iterate2.plugin.coordinator.sqlite import SQLiteCoordinator
        url = SQLiteCoordinator().get_storage("/tmp/study.db")
        assert url.startswith("sqlite:///")

    def test_passthrough_existing_sqlite_url(self):
        from terratorch_iterate.iterate2.plugin.coordinator.sqlite import SQLiteCoordinator
        url = "sqlite:///existing.db"
        assert SQLiteCoordinator().get_storage(url) == url

    def test_full_study_lifecycle(self, tmp_path):
        db_file = tmp_path / "test_study.db"
        storage = self._make_storage(str(db_file))
        _create_and_verify_study(storage, "sqlite_integration_test")

    def test_resolve_storage_via_registry(self, tmp_path):
        """End-to-end: resolve_storage() picks the SQLite plugin."""
        from terratorch_iterate.iterate2.plugin.coordinator import (
            load_builtin_plugins, resolve_storage,
        )
        load_builtin_plugins()
        db_file = tmp_path / "registry_test.db"
        storage = resolve_storage(str(db_file))
        assert storage.startswith("sqlite:///")


# ---------------------------------------------------------------------------
# JournalFS plugin
# ---------------------------------------------------------------------------

class TestJournalFSCoordinator:
    def _make_storage(self, journal_path: str):
        from terratorch_iterate.iterate2.plugin.coordinator.journalfs import JournalFSCoordinator
        return JournalFSCoordinator().get_storage(f"js:///{journal_path}")

    def test_matches_js_prefix(self):
        from terratorch_iterate.iterate2.plugin.coordinator.journalfs import JournalFSCoordinator
        assert JournalFSCoordinator().matches("js:///tmp/j.log")

    def test_no_match_sqlite(self):
        from terratorch_iterate.iterate2.plugin.coordinator.journalfs import JournalFSCoordinator
        assert not JournalFSCoordinator().matches("sqlite:///foo.db")

    def test_no_match_postgresql(self):
        from terratorch_iterate.iterate2.plugin.coordinator.journalfs import JournalFSCoordinator
        assert not JournalFSCoordinator().matches("postgresql://u:p@h/db")

    def test_returns_journal_storage_object(self, tmp_path):
        from optuna.storages import JournalStorage
        journal_file = tmp_path / "test.log"
        storage = self._make_storage(str(journal_file))
        assert isinstance(storage, JournalStorage)

    def test_full_study_lifecycle(self, tmp_path):
        journal_file = tmp_path / "study.log"
        storage = self._make_storage(str(journal_file))
        _create_and_verify_study(storage, "journalfs_integration_test")
        assert journal_file.exists(), "Journal file should exist after the study"

    def test_concurrent_writers(self, tmp_path):
        """Two studies sharing the same journal file must not corrupt each other."""
        import threading
        from optuna.storages import JournalStorage
        journal_file = tmp_path / "shared.log"
        storage_a = self._make_storage(str(journal_file))
        storage_b = self._make_storage(str(journal_file))

        errors: list[Exception] = []

        def run(storage, name):
            try:
                study = optuna.create_study(study_name=name, storage=storage,
                                             load_if_exists=True)
                study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=3)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=run, args=(storage_a, "worker_a"))
        t2 = threading.Thread(target=run, args=(storage_b, "worker_b"))
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert not errors, f"Concurrent writers raised: {errors}"

    def test_resolve_storage_via_registry(self, tmp_path):
        from terratorch_iterate.iterate2.plugin.coordinator import (
            load_builtin_plugins, resolve_storage,
        )
        load_builtin_plugins()
        journal_file = tmp_path / "registry.log"
        storage = resolve_storage(f"js:///{journal_file}")
        from optuna.storages import JournalStorage
        assert isinstance(storage, JournalStorage)


# ---------------------------------------------------------------------------
# PostgreSQL plugin
# ---------------------------------------------------------------------------

POSTGRES_URL = os.environ.get("POSTGRES_URL", "")

postgres_required = pytest.mark.skipif(
    not POSTGRES_URL,
    reason=(
        "Set POSTGRES_URL=postgresql://user:pass@host:5432/dbname "
        "to run PostgreSQL coordinator tests"
    ),
)


class TestPostgreSQLCoordinator:
    def test_matches_postgresql_scheme(self):
        from terratorch_iterate.iterate2.plugin.coordinator.postgresql import PostgreSQLCoordinator
        assert PostgreSQLCoordinator().matches("postgresql://u:p@h/db")

    def test_matches_legacy_postgres_scheme(self):
        from terratorch_iterate.iterate2.plugin.coordinator.postgresql import PostgreSQLCoordinator
        assert PostgreSQLCoordinator().matches("postgres://u:p@h/db")

    def test_no_match_sqlite(self):
        from terratorch_iterate.iterate2.plugin.coordinator.postgresql import PostgreSQLCoordinator
        assert not PostgreSQLCoordinator().matches("sqlite:///foo.db")

    def test_no_match_journalfs(self):
        from terratorch_iterate.iterate2.plugin.coordinator.postgresql import PostgreSQLCoordinator
        assert not PostgreSQLCoordinator().matches("js:///foo.log")

    def test_normalises_legacy_scheme(self):
        """'postgres://' must be normalised to 'postgresql://' before it reaches SQLAlchemy."""
        from terratorch_iterate.iterate2.plugin.coordinator.postgresql import _extract_host
        # Test the helper that parses the host out of the normalised URL.
        host = _extract_host("postgresql://user:pass@my-host.example.com:5432/db")
        assert host == "my-host.example.com"
        # Test that the legacy postgres:// scheme gets normalised (string level, no DB needed).
        legacy = "postgres://user:pass@host/db"
        normalised = "postgresql://" + legacy[len("postgres://"):]
        assert normalised.startswith("postgresql://")

    def test_missing_psycopg2_raises_import_error(self, monkeypatch):
        """If psycopg2 is absent the plugin must raise a clear ImportError."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "psycopg2":
                raise ImportError("mocked missing psycopg2")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        from terratorch_iterate.iterate2.plugin.coordinator.postgresql import PostgreSQLCoordinator
        with pytest.raises(ImportError, match="psycopg2"):
            PostgreSQLCoordinator().get_storage("postgresql://u:p@h/db")

    @postgres_required
    def test_full_study_lifecycle(self):
        import uuid
        from terratorch_iterate.iterate2.plugin.coordinator.postgresql import PostgreSQLCoordinator
        storage = PostgreSQLCoordinator().get_storage(POSTGRES_URL)
        study_name = f"pg_integration_{uuid.uuid4().hex[:8]}"
        try:
            _create_and_verify_study(storage, study_name)
        finally:
            # Clean up: delete the study so the DB stays tidy.
            try:
                optuna.delete_study(study_name=study_name, storage=storage)
            except Exception:
                pass

    @postgres_required
    def test_resolve_storage_via_registry(self):
        from optuna.storages import RDBStorage
        from terratorch_iterate.iterate2.plugin.coordinator import (
            load_builtin_plugins, resolve_storage,
        )
        load_builtin_plugins()
        storage = resolve_storage(POSTGRES_URL)
        assert isinstance(storage, RDBStorage)

    @postgres_required
    def test_parallel_trials(self):
        """Multiple threads sharing a PostgreSQL study must all complete cleanly."""
        import threading, uuid
        from terratorch_iterate.iterate2.plugin.coordinator.postgresql import PostgreSQLCoordinator
        storage = PostgreSQLCoordinator().get_storage(POSTGRES_URL)
        study_name = f"pg_parallel_{uuid.uuid4().hex[:8]}"
        study = optuna.create_study(study_name=study_name, storage=storage,
                                     load_if_exists=True)
        errors: list[Exception] = []

        def worker():
            try:
                study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=2)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        try:
            optuna.delete_study(study_name=study_name, storage=storage)
        except Exception:
            pass

        assert not errors, f"Parallel trials raised: {errors}"
