"""
PostgreSQL coordinator plugin for iterate2.

Matches any db_path that starts with ``postgresql://`` or ``postgres://``.

The raw URL is passed directly to Optuna's RDB storage layer (SQLAlchemy
under the hood).  Requires the ``psycopg2`` (or ``psycopg2-binary``) package
to be installed in the active environment::

    pip install psycopg2-binary

Example db_path values
-----------------------
``postgresql://user:password@localhost:5432/optuna_studies``
``postgres://user:password@db-host/mydb``

Self-registers at import time.
"""

from __future__ import annotations

import logging
import re

from terratorch_iterate.iterate2.plugin.coordinator import CoordinatorPlugin, register

logger = logging.getLogger("iterate2.coordinator.postgresql")

_SCHEMES = ("postgresql://", "postgres://")

# Default connect_timeout (seconds) injected into every connection so that
# cloud databases with firewalled ports don't cause silent hangs.
_DEFAULT_CONNECT_TIMEOUT = 30

# ---------------------------------------------------------------------------
# Plugin implementation
# ---------------------------------------------------------------------------

class PostgreSQLCoordinator(CoordinatorPlugin):
    name = "postgresql"

    def matches(self, db_path: str) -> bool:
        return any(db_path.startswith(scheme) for scheme in _SCHEMES)

    def get_storage(self, db_path: str):
        """Return an ``optuna.storages.RDBStorage`` configured for *db_path*.

        The storage object is used (rather than a bare URL string) so that we
        can inject ``connect_args`` (e.g. ``connect_timeout``, ``sslmode``)
        without requiring the caller to embed those options in the URL.

        SSL
        ---
        If ``sslmode`` is not already present in the URL query string *and* the
        host is not ``localhost`` / ``127.0.0.1``, ``sslmode=require`` is added
        automatically.  Pass ``?sslmode=disable`` in the URL to suppress this.
        """
        # Normalise legacy "postgres://" → "postgresql://" because SQLAlchemy
        # 1.4+ dropped support for the short-form scheme.
        if db_path.startswith("postgres://") and not db_path.startswith("postgresql://"):
            db_path = "postgresql://" + db_path[len("postgres://"):]
            logger.debug("Normalised scheme to: %s", db_path)

        # Verify psycopg2 is available early so the error is clear.
        try:
            import psycopg2  # noqa: F401
        except ImportError:
            raise ImportError(
                "psycopg2 is not installed but is required for PostgreSQL storage.\n\n"
                "Install options:\n"
                "  # recommended – pre-built wheel, no compiler needed:\n"
                "  pip install psycopg2-binary\n\n"
                "  # or via the project's postgresql extra:\n"
                "  pip install 'terratorch-iterate[postgresql]'\n\n"
                "  # production deployments that compile against a system libpq:\n"
                "  pip install psycopg2\n"
            ) from None

        connect_args: dict = {"connect_timeout": _DEFAULT_CONNECT_TIMEOUT}

        # Auto-enable SSL for non-local hosts when sslmode not already set.
        if "sslmode" not in db_path:
            host = _extract_host(db_path)
            if host not in ("localhost", "127.0.0.1", "::1", ""):
                connect_args["sslmode"] = "require"
                logger.debug("Auto-enabled sslmode=require for host '%s'", host)

        logger.info("PostgreSQL storage URL: %s", _redact(db_path))

        from optuna.storages import RDBStorage
        return RDBStorage(
            url=db_path,
            engine_kwargs={"connect_args": connect_args},
        )


def _extract_host(url: str) -> str:
    """Return the hostname portion of a postgresql:// URL."""
    try:
        # url looks like postgresql://user:pass@host:port/db
        after_at = url.split("@", 1)[1]
        host_port = after_at.split("/")[0]
        return host_port.split(":")[0]
    except (IndexError, AttributeError):
        return ""


def _redact(url: str) -> str:
    """Replace the password in a DB URL with '***' for safe logging."""
    return re.sub(r"(://[^:]+:)[^@]+(@)", r"\1***\2", url)


# ---------------------------------------------------------------------------
# Auto-register
# ---------------------------------------------------------------------------

register(PostgreSQLCoordinator())
