"""APEX-Holdout — deprecated alias.

Held-out final selection (the mechanism this class introduced) was ported
directly into `APEXOptimizer` and is now ON BY DEFAULT there (see
`APEXOptimizer.__init__`'s `use_holdout_selection` flag and the module-level
docstring in `pof/optimizers/apex.py`). This class is kept only so existing
configs/scripts that reference `--methods apex_holdout` by name keep working
identically to plain `apex` -- it no longer carries its own copy of the
holdout logic (which had an independent bug: finalist ranking mixed
minibatch-scale and dev-scale scores; fixed once, in the shared
implementation, rather than in two places).

Use `apex` going forward; this name is a compatibility shim.
"""
from __future__ import annotations

from pof.optimizers import register_optimizer
from pof.optimizers.apex import APEXOptimizer


@register_optimizer("apex_holdout")
class APEXHoldoutOptimizer(APEXOptimizer):
    """Deprecated: identical to `APEXOptimizer` (holdout selection is now
    the shared default). Kept as a name-compatible alias only."""

    name = "apex_holdout"
