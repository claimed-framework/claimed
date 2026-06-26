"""
SQLite coordinator plugin for iterate2.

Matches any db_path that:
  - already contains the ``sqlite:///`` scheme, OR
  - ends with ``.db`` or ``.sqlite``, OR
  - contains the substring ``sqlite``

The storage value returned to Optuna is always a fully-qualified
``sqlite:///...`` URL so that SQLAlchemy can open it correctly.

Self-registers at import time.
"""

from __future__ import annotations

import logging

from terratorch_iterate.iterate2.plugin.coordinator import CoordinatorPlugin, register

logger = logging.getLogger("iterate2.coordinator.sqlite")

# ---------------------------------------------------------------------------
# Plugin implementation
# ---------------------------------------------------------------------------

class SQLiteCoordinator(CoordinatorPlugin):
    name = "sqlite"

    def matches(self, db_path: str) -> bool:
        return (
            db_path.startswith("sqlite:///")
            or db_path.endswith(".db")
            or db_path.endswith(".sqlite")
            or "sqlite" in db_path
        )

    def get_storage(self, db_path: str) -> str:
        if db_path.startswith("sqlite:///"):
            storage_url = db_path
        else:
            storage_url = f"sqlite:///{db_path}"
        logger.info("SQLite storage URL: %s", storage_url)
        return storage_url


# ---------------------------------------------------------------------------
# Auto-register
# ---------------------------------------------------------------------------

register(SQLiteCoordinator())
