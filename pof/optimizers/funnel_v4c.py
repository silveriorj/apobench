"""FUNNELv4c — FUNNELv4a with the guaranteed family trimmed to zero-cost ops.

v4a inherits v3's full guaranteed family every phase: the three zero-LLM-call
demonstration-split operators (`instruction_only`, `few_shot_fixed`,
`few_shot_aug`) plus two paid decomposition operators (`facet_edit`, 2 calls;
`facet_enrich`, 1 call), applied to the top-3 elites — up to 9 forced LLM
calls/phase for the two paid ones alone.

v3's own docstring records `facet_edit`'s hit rate as the best of any operator
measured (1 win in 12 generations across six runs) — but that is also the
best hit rate BECAUSE it was already a bandit arm in `funnel_v2.py`'s pool
(`_HEAVY`) before v3 additionally guaranteed it; UCB1 would keep pulling a
good arm on its own once it has a few observations. `facet_enrich` has no
comparable measurement — it replaced `decompose_recompose` (which fired zero
times in six runs) but its own hit rate as a GUARANTEED operator was never
isolated from the guarantee itself.

**The change.** Drop both from `STATIC_ARMS`. `facet_edit` stays reachable —
it is still in `FUNNEL_V2_POOL`, so UCB1 can still select it, just no longer
forced on 3 elites every phase regardless of whether it is earning that cost.
`facet_enrich` is not in the base bandit pool (it is v3-only, resolved only
through `EXTRA_TECHNIQUES`, which `_step`'s bandit loop does not consult), so
dropping it here retires it rather than relocating it. Only the three
zero-cost demonstration-split operators stay guaranteed — the machinery v3's
own diagnosis actually pointed at (the boolean_expressions win came from a
few-shot example surviving into the final prompt, not from decomposition).

Cheaper by construction (up to 9 fewer forced LLM calls/phase); freed budget
goes to ordinary UCB1 exploration instead, which may recover facet_edit's
value where it is real and skip it where the guarantee was just paying for a
weak operator on a given task — the same logic v4a's adaptive gate already
applies to the family as a whole, applied here at the level of the two
individually-costed guaranteed operators instead.
"""
from __future__ import annotations

from typing import List

from pof.optimizers import register_optimizer
from pof.optimizers.funnel_v4a import FUNNELv4aOptimizer

V4C_STATIC: List[str] = ["instruction_only", "few_shot_fixed", "few_shot_aug"]


@register_optimizer("funnel_v4c")
class FUNNELv4cOptimizer(FUNNELv4aOptimizer):
    """FUNNELv4a with facet_edit/facet_enrich demoted out of the guarantee."""

    name = "funnel_v4c"

    STATIC_ARMS: List[str] = V4C_STATIC
