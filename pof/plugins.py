"""Third-party extension discovery for APOBench.

The problem this solves
-----------------------
APOBench documents four extension points — optimizers, datasets, score
functions, and LLM backends. All four were dispatched from hardcoded tables
inside the package: `_load_all()` imported 27 optimizer modules by name,
`load_dataset_by_name`, `create_score_function` and `create_llm` were if/elif
chains. The `@register_optimizer` decorator worked, but nothing ever imported a
third party's module, so their optimizer never registered and the CLI reported
"Unknown optimizer" for a method that existed.

The evidence that this was not hypothetical is in the repository:
`pof/llm/sap_aicore_backend.register()` monkey-patches `factory.create_llm` at
runtime and rebinds the name inside `pof.orchestration.runner`, because a
`from ... import` there had already captured the original. Adding one backend
required patching the framework from the inside. A benchmark whose own author
needs that is not extensible by anyone else.

Two mechanisms, both standard
-----------------------------
1. **Entry points.** A third-party package declares, in its own pyproject.toml:

       [project.entry-points."apobench.optimizers"]
       my_method = "my_pkg.my_method:MyOptimizer"

   Installing that package is all it takes. Groups: `apobench.optimizers`,
   `apobench.datasets`, `apobench.scorers`, `apobench.backends`.

2. **APOBENCH_PLUGINS.** A path-separated list of modules to import, for code
   that is not installed as a package:

       APOBENCH_PLUGINS=my_experiment.methods,other.thing

   Useful during development, where publishing a package per experiment is
   friction that stops people from running the benchmark at all.

Both are additive. Every built-in dispatch stays exactly as it was, so no
existing config changes meaning. Discovery failures are logged and skipped
rather than raised: one broken third-party plugin must not stop a benchmark run
that does not use it.
"""
from __future__ import annotations

import importlib
import logging
import os
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

OPTIMIZER_GROUP = "apobench.optimizers"
DATASET_GROUP = "apobench.datasets"
SCORER_GROUP = "apobench.scorers"
BACKEND_GROUP = "apobench.backends"

_ALL_GROUPS = (OPTIMIZER_GROUP, DATASET_GROUP, SCORER_GROUP, BACKEND_GROUP)

# group -> {name: loaded object}. Populated once, on first discovery.
_DISCOVERED: Dict[str, Dict[str, Any]] = {}
_MODULES_IMPORTED = False


def _entry_points(group: str) -> List[Any]:
    """Entry points for one group, across the importlib.metadata versions.

    Python 3.10+ takes `select(group=...)`; 3.9 returns a dict. The project
    supports >=3.9, so both shapes have to work.
    """
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - Python < 3.8 is unsupported
        return []

    try:
        eps = entry_points()
        if hasattr(eps, "select"):
            return list(eps.select(group=group))
        return list(eps.get(group, []))  # type: ignore[union-attr]
    except Exception as e:  # pragma: no cover - metadata backends vary
        logger.debug(f"entry point lookup failed for {group}: {e}")
        return []


def _import_plugin_modules() -> None:
    """Import whatever APOBENCH_PLUGINS names, for registration side effects."""
    global _MODULES_IMPORTED
    if _MODULES_IMPORTED:
        return
    _MODULES_IMPORTED = True

    raw = os.environ.get("APOBENCH_PLUGINS", "").strip()
    if not raw:
        return
    for mod in (m.strip() for m in raw.replace(os.pathsep, ",").split(",")):
        if not mod:
            continue
        try:
            importlib.import_module(mod)
            logger.info(f"[plugins] imported {mod} from APOBENCH_PLUGINS")
        except Exception as e:
            # Skip, do not raise. A typo in one plugin should not take down a
            # run that does not use it.
            logger.warning(f"[plugins] could not import '{mod}': {e}")


def discover(group: str) -> Dict[str, Any]:
    """Load and return every third-party object registered under `group`.

    Results are cached: entry point loading imports modules, and repeating that
    on every dataset lookup would be a silent per-call cost.
    """
    _import_plugin_modules()
    if group in _DISCOVERED:
        return _DISCOVERED[group]

    found: Dict[str, Any] = {}
    for ep in _entry_points(group):
        try:
            found[ep.name.lower()] = ep.load()
            logger.info(f"[plugins] loaded {group}:{ep.name}")
        except Exception as e:
            logger.warning(f"[plugins] failed to load {group}:{ep.name}: {e}")
    _DISCOVERED[group] = found
    return found


def load_all() -> None:
    """Trigger discovery for every group.

    Called where a complete listing is needed — `pof list`, for instance, which
    would otherwise show only the built-ins and quietly omit exactly the
    third-party method its user came to check.
    """
    for group in _ALL_GROUPS:
        discover(group)
