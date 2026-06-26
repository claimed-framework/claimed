"""
Lightweight coordinator plugin registry for Optuna storage backends.

Each coordinator plugin lives in its own module inside this package:
  - sqlite.py
  - journalfs.py
  - postgresql.py

Plugins register themselves by calling ``register()`` at import time.
``resolve_storage()`` walks the registry in insertion order and returns the
first matching plugin's storage object.

Usage
-----
>>> from terratorch_iterate.iterate2.plugin.coordinator import resolve_storage
>>> storage = resolve_storage("sqlite:///my_study.db")
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger("iterate2.coordinator")

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class CoordinatorPlugin(ABC):
    """Abstract base for Optuna storage coordinator plugins."""

    #: Human-readable name shown in log messages.
    name: str = "base"

    @abstractmethod
    def matches(self, db_path: str) -> bool:
        """Return ``True`` when this plugin should handle *db_path*."""

    @abstractmethod
    def get_storage(self, db_path: str) -> Any:
        """Return an Optuna-compatible storage object (or URL string) for *db_path*."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_registry: list[CoordinatorPlugin] = []


def register(plugin: CoordinatorPlugin) -> None:
    """Register a coordinator plugin.  Later registrations take lower priority."""
    _registry.append(plugin)
    logger.debug("Registered coordinator plugin: %s", plugin.name)


def resolve_storage(db_path: str) -> Any:
    """Walk the registry and return the storage for *db_path*.

    Raises
    ------
    ValueError
        When no registered plugin matches *db_path*.
    """
    for plugin in _registry:
        if plugin.matches(db_path):
            logger.info("Coordinator plugin '%s' handling db_path '%s'", plugin.name, db_path)
            return plugin.get_storage(db_path)
    raise ValueError(
        f"No coordinator plugin matched db_path={db_path!r}. "
        "Make sure the appropriate plugin module is imported before calling resolve_storage()."
    )


# ---------------------------------------------------------------------------
# Auto-load built-in plugins
# ---------------------------------------------------------------------------

def load_builtin_plugins() -> None:
    """Import all built-in coordinator plugins so they self-register."""
    import importlib
    _builtins = [
        "terratorch_iterate.iterate2.plugin.coordinator.sqlite",
        "terratorch_iterate.iterate2.plugin.coordinator.journalfs",
        "terratorch_iterate.iterate2.plugin.coordinator.postgresql",
    ]
    for mod in _builtins:
        try:
            importlib.import_module(mod)
        except ImportError as exc:
            logger.warning("Could not load coordinator plugin '%s': %s", mod, exc)
