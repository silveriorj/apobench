"""FUNNELv7 ("FUNNEL-Prime") — the synthesis of every proven mechanism this
project built, in one legible class.

FUNNEL went through eight iterations (v1 -> v2 -> v3 -> v4a/b/c/d -> v5 ->
v6), each adding one measured, evidence-driven mechanism. Every one of those
mechanisms is still a separate class; this is that synthesis, made explicit.

**Inherited unchanged (see each origin file for the underlying
measurements):**

- Accumulating-fresh validation (`funnel_v2.py`): a growing, nested-prefix
  dev pool, one deterministic shuffle — fixes dev/test overfitting from
  reusing the same fixed sample every phase.
- Held-out selection (`_finalize`/`_select_on_holdout`, `funnel_v2.py`):
  re-verifies top contenders at the run's final N, then picks the winner on
  a slice the search never touched — corrects the winner's-curse bias of
  picking the dev-pool argmax directly.
- Demonstration-split family — `instruction_only`, `few_shot_fixed`,
  `few_shot_aug` (v3/v4c): guaranteed on every phase's top elites because
  demonstrations are ABSORBING once a candidate carries them, so without a
  forced bare-instruction candidate the with-demonstrations form takes over
  the population within 1-2 phases.
- Static-baseline floor (v4b): Phase 0 is seeded with the canonical
  `json_3shot` prompt, so the search can never finish worse than a baseline
  already known to beat it on some tasks.
- Trimmed guarantee + adaptive family-disable gate (v4c + v4a): once the
  guaranteed family's best score stops beating the static baseline (checked
  from phase 2 onward), it's disabled for the rest of the run — one-way,
  never re-enabled.
- Batch-level Hoeffding racing, perfect-EM early stop, dev-pool complexity
  ordering (v4d, "FUNNEL-Lean" — the best macro-accuracy variant measured to
  date): measured cost reductions with no accuracy cost.
- Mode family + adaptive mode gating (v5 "FUNNEL-Wide" + v6
  "FUNNEL-Indexed"): eval mode (answer-only / brief-CoT / full-thinking) as
  a heritable per-candidate property rather than one fixed run-level
  setting, gated by `REASONING_TASK_INDEX` once real evidence agrees with
  the prediction — the single largest measured effect of anything built
  this session.

**What v7 adds on top:**

1. Widens `STATIC_ARMS` to also guarantee `multi_aspect_critique` (CRISPO-
   style: critiques failures along 4 named independent aspects rather than
   one blended paragraph, so a fixable issue on one axis can't get buried).
   Fits the existing per-elite guarantee mechanism unchanged.

2. Guarantees `trajectory_momentum` (OPRO's ranked history plus TextGrad-
   style momentum: condition the edit on the DIRECTION of recent
   improvement, not just the destination) — but NOT via the per-elite loop,
   since `t_trajectory_momentum` reads `self.population` directly and has no
   per-elite target; looping it `static_top_k` times would bill correlated
   draws against an identical input as distinct operations. `_apply_static_core`
   is overridden to split `STATIC_ARMS` into per-elite operators and
   `POPULATION_SCOPED_ARMS` (invoked exactly once per phase).

3. Prunes `decompose_recompose` from `BANDIT_ARMS`: its trigger (a missing
   required grammar facet) essentially never occurs, so it never fires; a
   dead arm still costs UCB1 exploration budget whenever it's pulled.

Neither new operator is removed from `BANDIT_ARMS` by being guaranteed —
same precedent as v3's `few_shot`. Neither has an isolated in-project
hit-rate measurement the way `facet_edit` or `few_shot` do; they're
guaranteed on literature grounding (CRISPO, TextGrad) and mechanism-level
argument rather than a measured win rate, which is a real difference in
evidentiary standing worth remembering when reading this method's results.
"""
from __future__ import annotations

import logging
from typing import List

from pof.core.types import PromptRecord
from pof.optimizers import register_optimizer
from pof.optimizers.funnel_v2 import _ALL_TECHNIQUE_FNS
from pof.optimizers.funnel_v4c import V4C_STATIC
from pof.optimizers.funnel_v6 import FUNNELv6Optimizer

logger = logging.getLogger(__name__)

# v4c's TRIMMED demonstration-split family (not v3's full 5-op family --
# facet_edit/facet_enrich were dropped there for cost/evidence reasons, see
# funnel_v4c.py) plus the newly-guaranteed critique operator.
V7_STATIC: List[str] = V4C_STATIC + ["multi_aspect_critique"]

# Guaranteed, but population-scoped rather than per-elite -- see point 2
# above. Invoked once per phase regardless of static_top_k.
V7_POPULATION_SCOPED: List[str] = ["trajectory_momentum"]

# decompose_recompose removed: confirmed dead (0/6 fire rate), freeing its
# UCB1 exploration budget for arms that actually produce candidates.
V7_BANDIT: List[str] = [
    t for t in FUNNELv6Optimizer.BANDIT_ARMS if t != "decompose_recompose"
]


@register_optimizer("funnel_v7")
class FUNNELv7Optimizer(FUNNELv6Optimizer):
    """FUNNEL-Prime: the synthesis of every proven mechanism from v1-v6."""

    name = "funnel_v7"

    STATIC_ARMS: List[str] = V7_STATIC
    POPULATION_SCOPED_ARMS: List[str] = V7_POPULATION_SCOPED
    BANDIT_ARMS: List[str] = V7_BANDIT

    def _apply_static_core(self, candidates: List[PromptRecord]) -> int:
        """Per-elite guarantee (base behavior) plus population-scoped arms.

        Population-scoped arms (currently just `trajectory_momentum`) are
        invoked exactly once per phase -- not once per elite in
        `static_top_k` like the base class's loop does for everything in
        `STATIC_ARMS`, which would be correct only for operators that
        actually target a specific elite.
        """
        made = super()._apply_static_core(candidates)
        made += self._apply_population_scoped(candidates)
        return made

    def _apply_population_scoped(self, candidates: List[PromptRecord]) -> int:
        if not self.POPULATION_SCOPED_ARMS or not self.population:
            return 0
        made = 0
        for name in self.POPULATION_SCOPED_ARMS:
            fn = self.EXTRA_TECHNIQUES.get(name) or _ALL_TECHNIQUE_FNS[name]
            text = fn(self)
            if text and not self._is_duplicate(text):
                candidates.append(self._create_record(
                    text, operator=name,
                    parent_ids=[r.id for r in self.population[:2]],
                ))
                made += 1
        if made:
            logger.info(
                f"[{self.name} Phase {self._phase_idx}] population-scoped "
                f"arms produced {made} candidate(s)"
            )
        return made


# Alias: paper-facing name, same class.
register_optimizer("funnel_prime")(FUNNELv7Optimizer)
