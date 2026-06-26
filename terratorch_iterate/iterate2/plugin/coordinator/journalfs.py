"""
JournalFS coordinator plugin for iterate2.

Matches any db_path that starts with the ``js:///`` prefix.

The prefix is stripped and the remainder is treated as a local filesystem
path.  The storage object returned is an Optuna
``JournalStorage(JournalFileBackend(...))`` instance which is safe for
concurrent, multi-process access and does not require a database server.

Self-registers at import time.
"""

from __future__ import annotations

import logging

from optuna.storages import JournalStorage

# Prefer the non-deprecated JournalFileBackend (Optuna ≥4.0); fall back to the
# legacy JournalFileStorage for older installations.
try:
    from optuna.storages.journal import JournalFileBackend as _JournalFileBackend  # type: ignore
    _USE_BACKEND = True
except ImportError:
    from optuna.storages import JournalFileStorage as _JournalFileBackend  # type: ignore
    _USE_BACKEND = False

from terratorch_iterate.iterate2.plugin.coordinator import CoordinatorPlugin, register

logger = logging.getLogger("iterate2.coordinator.journalfs")

_PREFIX = "js:///"

# ---------------------------------------------------------------------------
# Plugin implementation
# ---------------------------------------------------------------------------

class JournalFSCoordinator(CoordinatorPlugin):
    name = "journalfs"

    def matches(self, db_path: str) -> bool:
        return db_path.startswith(_PREFIX)

    def get_storage(self, db_path: str) -> JournalStorage:
        journal_path = db_path[len(_PREFIX):]
        backend_cls = "JournalFileBackend" if _USE_BACKEND else "JournalFileStorage"
        logger.info("JournalStorage backend=%s path=%s", backend_cls, journal_path)
        return JournalStorage(_JournalFileBackend(journal_path))


# ---------------------------------------------------------------------------
# Auto-register
# ---------------------------------------------------------------------------

register(JournalFSCoordinator())
