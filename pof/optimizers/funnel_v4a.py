"""FUNNELv4a — FUNNELv4b with the guaranteed few-shot/decomposition family made
adaptive instead of unconditional.

Same diagnosis as v4b (see its module docstring): v3's guaranteed family
(`instruction_only`, `few_shot_fixed`, `few_shot_aug`, `facet_edit`,
`facet_enrich`, applied to every phase's top elites) wins big on some tasks
and actively hurts on others — it is significantly beaten by the plain static
`json_3shot` baseline on formal_fallacies, and undershoots even
`instruction_only` on logical_deduction_five_objects. v4b puts a floor under
that damage by making the static baseline itself a candidate; v4a goes
further and asks WHETHER to keep paying for the guaranteed family at all on a
given task.

**The mechanism.** After each phase (once the static-baseline candidate and
the guaranteed family have both been evaluated at that phase's sample size —
`_baseline_record` only enters the comparison once its evaluation is current,
so an early or stale score never triggers this), compare the best score any
guaranteed-family candidate has reached against the static baseline's score.
If the family isn't beating the baseline, disable `STATIC_ARMS` for the rest
of the run — the search falls back to pure UCB1 bandit selection, where
`few_shot` remains available as an ordinary arm (it isn't removed from the
pool, just demoted from guaranteed), so the family is not gone, only no
longer subsidized.

This is a one-way switch: once disabled, the family stays off. Re-enabling on
a later phase's noise would just re-pay the cost the switch exists to avoid.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from pof.core.types import PromptRecord
from pof.optimizers import register_optimizer
from pof.optimizers.funnel_v4b import FUNNELv4bOptimizer

logger = logging.getLogger(__name__)


@register_optimizer("funnel_v4a")
class FUNNELv4aOptimizer(FUNNELv4bOptimizer):
    """FUNNELv4b with the guaranteed family disabled once it stops earning its keep."""

    name = "funnel_v4a"

    def _init_population(self) -> List[PromptRecord]:
        self._family_enabled = True
        self._baseline_record: Optional[PromptRecord] = None
        pop = super()._init_population()
        for record in self.tracker.history.records.values():
            if record.operator == "static_json3shot_baseline":
                self._baseline_record = record
                break
        return pop

    def _apply_static_core(self, candidates: List[PromptRecord]) -> int:
        if not self._family_enabled:
            return 0
        return super()._apply_static_core(candidates)

    def _step(self) -> List[PromptRecord]:
        result = super()._step()
        self._maybe_disable_family()
        return result

    def _maybe_disable_family(self) -> None:
        if not self._family_enabled or self._baseline_record is None:
            return
        n_now = self._phase_sizes[self._phase_idx]
        baseline_n = len(self._baseline_record.per_sample_details or [])
        if baseline_n < n_now:
            return  # baseline's score is stale for this phase -- wait for a fresher one

        family_names = set(self.STATIC_ARMS)
        family_scores = [
            r.score for r in self.tracker.history.records.values()
            if r.operator in family_names and len(r.per_sample_details or []) >= n_now
        ]
        if not family_scores:
            return

        family_best = max(family_scores)
        baseline_score = self._baseline_record.score
        if family_best <= baseline_score:
            self._family_enabled = False
            note = (
                f"guaranteed family disabled after phase {self._phase_idx}: "
                f"best={family_best:.4f} <= static baseline={baseline_score:.4f} "
                f"(N={n_now}); falling back to bandit-only for remaining phases"
            )
            logger.info(f"[{self.name}] {note}")
            self.tracker.add_note(note)
