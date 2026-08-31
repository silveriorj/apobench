"""Optimizers — reimplementations of published APO methods, plus a
no-search baseline. See each module's docstring for its citation."""
from __future__ import annotations

from pof.optimizers.base import BaseOptimizer

_REGISTRY: dict = {}
_LOADED = False


def register_optimizer(name: str):
    """Decorator to register an optimizer class."""
    def decorator(cls):
        _REGISTRY[name.lower()] = cls
        return cls
    return decorator


def _discover_third_party() -> dict:
    """Optimizers contributed by installed packages or APOBENCH_PLUGINS.

    `_load_all()` below imports the built-ins by name, which can never reach a
    class living in someone else's package — the decorator fires only if the
    module is imported, and nothing imported it. That is why an external
    method registered correctly and still came back as "Unknown optimizer".
    """
    from pof.plugins import OPTIMIZER_GROUP, discover

    found = discover(OPTIMIZER_GROUP)
    for key, cls in found.items():
        # Built-ins win a name collision: a third-party package must not be
        # able to silently replace a built-in method and change what a
        # published number means.
        if key in _REGISTRY:
            import logging
            logging.getLogger(__name__).warning(
                f"[plugins] ignoring third-party optimizer '{key}' — a built-in "
                "already claims that name"
            )
            continue
        _REGISTRY[key] = cls
    return _REGISTRY


def get_optimizer(name: str) -> type:
    """Get optimizer class by name."""
    # Lazy imports to avoid circular dependencies
    _load_all()
    _discover_third_party()
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown optimizer: '{name}'. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[key]


def list_optimizers() -> list:
    """List all registered optimizer names, built-in and third-party."""
    _load_all()
    _discover_third_party()
    return sorted(_REGISTRY.keys())


def _load_all():
    """Import all optimizer modules to trigger registration.

    Guarded by an explicit flag rather than `if not _REGISTRY`. Emptiness is the
    wrong test: importing any optimizer module directly registers that one as a
    side effect, leaving the registry non-empty but INCOMPLETE, after which the
    old guard skipped loading and every other optimizer stayed invisible —
    surfacing later as a spurious "Unknown optimizer" for a method that exists.
    """
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    from pof.optimizers import see  # noqa: F401
    from pof.optimizers import gaapo  # noqa: F401
    from pof.optimizers import capo  # noqa: F401
    from pof.optimizers import gepa  # noqa: F401
    from pof.optimizers import baseline  # noqa: F401


__all__ = ["BaseOptimizer", "get_optimizer", "list_optimizers", "register_optimizer"]