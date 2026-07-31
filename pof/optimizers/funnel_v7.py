"""FUNNELv7 ("FUNNEL-Prime") — the synthesis of every proven mechanism this
project built, in one legible class.

FUNNEL went through eight iterations this session (v1 -> v2 -> v3 -> v4a/b/c/d
-> v5 -> v6), each adding one measured, evidence-driven mechanism. Every one
of those mechanisms is still a separate class; "the best method" was whichever
alias you remembered to type. This is that synthesis, made explicit.

**Inherited unchanged (already proven -- see each origin file for the actual
measurements this project made; only cited here):**

- Accumulating-fresh validation (`funnel_v2.py`): a growing, nested-prefix dev
  pool, one deterministic shuffle. Fixed a measured dev/test overfitting
  correlation of -0.695 (dev gains that didn't transfer to test) -> +0.918
  once every phase saw genuinely fresh instances instead of the same fixed
  sample four times over.
- Held-out selection (`_finalize`/`_select_on_holdout`, `funnel_v2.py`):
  re-verifies the top 6 contenders at the run's final N, then picks the
  winner on a slice the search never touched. Fixed a -0.46 dev/test
  correlation (winner's curse: the seed scoring highest on dev scored lowest
  on test) down to a handful of finalists compared on fresh evidence.
- Demonstration-split family -- `instruction_only`, `few_shot_fixed`,
  `few_shot_aug` (v3/v4c): guaranteed on every phase's top elites because
  demonstrations are ABSORBING once a candidate carries them -- without a
  forced bare-instruction candidate, the with-demonstrations form silently
  takes over the whole population within 1-2 phases and the comparison this
  family exists to make (does adding demonstrations actually help?) collapses.
- Static-baseline floor (v4b): Phase 0 is seeded with the canonical
  `json_3shot` prompt built from the official chain-of-thought-hub file, so
  the search can never finish worse than a baseline already known to beat it
  on some tasks.
- Trimmed guarantee + adaptive family-disable gate (v4c + v4a): once the
  guaranteed family's best score stops beating the static baseline (checked
  from phase 2 onward), it's disabled for the rest of the run -- one-way,
  self-correcting only in the direction of NOT paying for a family that isn't
  earning its keep. Measured to shrink formal_fallacies' gap against the
  static baseline from a significant loss (t=-4.9) to statistically
  indistinguishable.
- Batch-level Hoeffding racing, perfect-EM early stop, dev-pool complexity
  ordering (v4d, "FUNNEL-Lean" -- the best macro-accuracy variant measured to
  date: 0.715 vs. v4a's 0.701/v3's 0.699/v2's 0.689/v1's 0.677 on Qwen3-4B,
  7 common BBH tasks): all measured cost reductions with no accuracy cost.
- Mode family + adaptive mode gating (v5 "FUNNEL-Wide" + v6
  "FUNNEL-Indexed"): eval mode (answer-only / brief-CoT / full-thinking) as a
  heritable per-candidate property rather than one fixed run-level setting,
  gated by `REASONING_TASK_INDEX` once real evidence agrees with the
  prediction. This is the single largest measured effect of anything built
  this session: boolean_expressions reaches 0.997 under full CoT vs. 0.887-
  0.896 best answer-only; causal_judgement is WORSE under full CoT (0.583 vs.
  0.649 answer-only) -- a reversal that independently reproduces Suzgun et
  al. (2023) Table 3's own CoT-hurts-causal_judgement finding on much larger
  models. Guessing one mode per run means guessing wrong on a real fraction
  of tasks; this doesn't guess.

**What v7 adds on top:**

1. Widens `STATIC_ARMS` to also guarantee `multi_aspect_critique` (CRISPO-
   style: critiques failures along 4 named independent aspects -- answer
   format, reasoning approach, edge-case handling, instruction clarity --
   rather than one blended strategy paragraph, so a precise fixable issue on
   one axis can't get buried under commentary about another). It fits the
   existing per-elite guarantee mechanism unchanged: like every other
   guaranteed operator, it targets a specific elite via `_pick_elite`.

2. Guarantees `trajectory_momentum` (OPRO's ranked history + an explicit
   trend-identification step before extrapolating -- TextGrad's "momentum":
   condition the edit on the DIRECTION of recent improvement, not just the
   destination) -- but NOT via the same per-elite loop. `t_trajectory_momentum`
   reads `self.population` directly; it has no per-elite target. Guaranteeing
   it through `_apply_static_core`'s normal per-elite loop would invoke it
   `static_top_k` times a phase against the IDENTICAL population-level input
   -- three correlated stochastic draws billed as three distinct operations.
   `_apply_static_core` is overridden here to split `STATIC_ARMS` into
   per-elite operators (looped as usual) and `POPULATION_SCOPED_ARMS`
   (invoked exactly once per phase, matching what the operator actually is).

3. Prunes `decompose_recompose` from `BANDIT_ARMS`: confirmed 0/6 fire rate
   across every run that measured it (its trigger -- a missing REQUIRED
   grammar facet -- essentially never occurs, since the decomposer always
   emits both required fields). A dead arm still costs UCB1 exploration
   budget every time it's pulled; removing it frees that budget for arms that
   actually produce candidates.

Neither `multi_aspect_critique` nor `trajectory_momentum` is removed from
`BANDIT_ARMS` by being guaranteed -- precedent already exists for this in
v3's own design (`few_shot` is guaranteed via the demonstration family AND
still an ordinary bandit arm, reaching candidates the guarantee's
`static_top_k` cutoff doesn't cover).

Neither newly-guaranteed operator has an isolated, in-project measurement of
its own hit rate the way `facet_edit` (1/12) or `few_shot` (1/18) do --
they're guaranteed here on the strength of their literature grounding
(CRISPO, TextGrad) and mechanism-level argument, not a measured win rate.
That's a real difference in evidentiary standing from everything else this
class guarantees, and it's worth remembering when reading results from this
method specifically.
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
