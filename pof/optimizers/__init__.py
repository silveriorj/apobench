"""Optimizers — all APO algorithm implementations.

Priority methods: SEE, SWIFT, APEX, GAAPO, CAPO, GEPA.
"""
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


def get_optimizer(name: str) -> type:
    """Get optimizer class by name."""
    # Lazy imports to avoid circular dependencies
    _load_all()
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown optimizer: '{name}'. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[key]


def list_optimizers() -> list:
    """List all registered optimizer names."""
    _load_all()
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
    from pof.optimizers import swift  # noqa: F401
    from pof.optimizers import swift_v2  # noqa: F401
    from pof.optimizers import apex  # noqa: F401
    from pof.optimizers import apex_v2  # noqa: F401
    from pof.optimizers import gaapo  # noqa: F401
    from pof.optimizers import capo  # noqa: F401
    from pof.optimizers import gepa  # noqa: F401
    from pof.optimizers import gspe  # noqa: F401
    from pof.optimizers import funnel  # noqa: F401
    from pof.optimizers import funnel_v2  # noqa: F401
    from pof.optimizers import funnel_v3  # noqa: F401
    from pof.optimizers import funnel_v4a  # noqa: F401
    from pof.optimizers import funnel_v4b  # noqa: F401
    from pof.optimizers import funnel_v4c  # noqa: F401
    from pof.optimizers import funnel_v4d  # noqa: F401
    from pof.optimizers import funnel_v5  # noqa: F401
    from pof.optimizers import funnel_v6  # noqa: F401


__all__ = ["BaseOptimizer", "get_optimizer", "list_optimizers", "register_optimizer"]