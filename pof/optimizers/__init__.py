"""Optimizers — all APO algorithm implementations.

Priority methods: SEE, SWIFT, APEX, GAAPO, CAPO, GEPA.
"""
from __future__ import annotations

from pof.optimizers.base import BaseOptimizer

_REGISTRY: dict = {}


def register_optimizer(name: str):
    """Decorator to register an optimizer class."""
    def decorator(cls):
        _REGISTRY[name.lower()] = cls
        return cls
    return decorator


def get_optimizer(name: str) -> type:
    """Get optimizer class by name."""
    # Lazy imports to avoid circular dependencies
    if not _REGISTRY:
        _load_all()
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown optimizer: '{name}'. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[key]


def list_optimizers() -> list:
    """List all registered optimizer names."""
    if not _REGISTRY:
        _load_all()
    return sorted(_REGISTRY.keys())


def _load_all():
    """Import all optimizer modules to trigger registration."""
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


__all__ = ["BaseOptimizer", "get_optimizer", "list_optimizers", "register_optimizer"]