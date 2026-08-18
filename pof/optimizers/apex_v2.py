"""APEXv2 — APEX with a historically-informed, variance-aware bandit.

Two changes from v1, both about the bandit rather than the operator set
(all 7 operator implementations are unchanged):

1. **`format_constraint` is free** (`record.text.rstrip() + constraint`: a
   deterministic string append, no LLM call), so it shouldn't have to
   compete for bandit slots against 6 arms that cost a full generation.
   v2 removes it from the bandit and always applies it to the top-2 elites
   every generation instead.
2. **UCB1-Tuned** (Auer et al. 2002, the same paper APEX cites for UCB1's
   regret bound): the exploration bonus is scaled by each arm's own
   empirical variance instead of a flat term, so noisy arms keep exploring
   longer and consistent arms converge faster. The remaining 6 arms are
   also warm-started with `prior_pulls` pseudo-observations at a historical
   mean, so one unlucky first pull can't bury a normally-strong arm.

Also mixes in `LengthAwareDedupeMixin` (`_v2_common.py`, shared with
SWIFTv2): tournament selection nudges toward shorter prompts when
candidates are within noise of each other (CAPO-style length-penalized
fitness), and dedup catches near-identical text via `difflib`.

Registered as a separate optimizer name ("apex_v2") — v1 is untouched.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Tuple

from pof.core.types import PromptRecord
from pof.optimizers import register_optimizer
from pof.optimizers._v2_common import LengthAwareDedupeMixin
from pof.optimizers.apex import APEXOptimizer

logger = logging.getLogger(__name__)

# Historical per-arm (mean, std) for the 6 arms that remain in the bandit
# (format_constraint excluded -- see module docstring). Used only as a
# warm-start prior; the bandit adapts to the live task/model from there.
# Means are pinned at 0.0: the base class credits fitness IMPROVEMENT
# (candidate score minus a shared elite baseline), a quantity centered near
# 0, and reusing the old absolute-score means here would miscalibrate the
# first `prior_pulls` rounds. std values are an approximate relative-
# noisiness ranking, used by the UCB1-Tuned variance term.
HISTORICAL_ARM_STATS: Dict[str, Tuple[float, float]] = {
    "few_shot":          (0.0, 0.121),
    "crossover":         (0.0, 0.221),
    "semantic_var":      (0.0, 0.236),
    "failure_guided":    (0.0, 0.227),
    "expert_refine":     (0.0, 0.239),
    "trajectory":        (0.0, 0.270),
}


@register_optimizer("apex_v2")
class APEXv2Optimizer(LengthAwareDedupeMixin, APEXOptimizer):
    """APEX v2 — free always-on format_constraint, warm-started variance-aware
    (UCB1-Tuned) bandit over the remaining 6 arms.

    Every operator implementation is inherited unchanged from v1; only which
    arms compete in the bandit, how they're scored, and the addition of the
    free format-constraint pass change.
    """

    name = "apex_v2"

    def __init__(self, *args, prior_pulls: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior_pulls = prior_pulls
        # Warm start: seed each of the 6 remaining arms with `prior_pulls`
        # pseudo-observations at its historical mean, so one unlucky first
        # pull can't bury a normally-strong arm. The parent's _step() appends
        # real scores to these same lists, so the prior's influence dilutes
        # naturally as live evidence accumulates.
        self._operator_scores = {
            name: [mean] * prior_pulls
            for name, (mean, _std) in HISTORICAL_ARM_STATS.items()
        }

    def _step(self) -> List[PromptRecord]:
        """Adaptive step, plus a free always-on format_constraint pass.

        Generating a format_constraint variant costs nothing (no LLM call —
        see module docstring), so it's applied directly to this generation's
        top-2 survivors rather than competing for a bandit slot. Scoring it
        still costs a real evaluation, so — same as every other new candidate
        this step — it goes through the minibatch gate, not a full eval.
        """
        selected = super()._step()
        elites = sorted(selected, key=lambda r: r.score, reverse=True)[:2]
        new_records = []
        for record in elites:
            for text in self._op_format_constraint_from(record):
                if text and not self._is_duplicate(text):
                    new_records.append(self._create_record(
                        text, operator="format_constraint_free", parent_ids=[record.id]
                    ))
        if not new_records:
            return selected
        baseline = self.best_record.score if self.best_record else 0.0
        self._evaluate_with_minibatch_gate(new_records, baseline, slack=self.gate_slack)
        # Must re-sort independently: this method assembles its own final
        # population from selected+new_records, so v1's re-sort (inside
        # super()._step()) doesn't cover it.
        final = self._tournament_select(selected + new_records)
        final.sort(key=lambda r: r.score, reverse=True)
        return final

    def _op_format_constraint_from(self, record: "PromptRecord") -> List[str]:
        """Same rule as v1's _op_format_constraint, applied to a specific record
        instead of a random elite (v2 calls it directly, not through the bandit)."""
        targets = [s["target"] for s in self.dataset.get_few_shot_examples(n=4)]
        sample_answers = ", ".join(repr(t)[:30] for t in targets[:3])
        constraint = (
            f"\n\nAnswer with ONLY the final answer, exactly in the same format "
            f"as these examples: {sample_answers}. No explanation."
        )
        return [record.text.rstrip() + constraint]

    def _select_operators(self) -> List[tuple]:
        """UCB1-Tuned bandit over the 6 remaining operators (Auer et al. 2002,
        Section 4). format_constraint is excluded -- see module docstring.

        value = mean + decay(gen) * sqrt( (ln(N)/n) * min(1/4, V_i) )
        where V_i = sample_variance_i + sqrt(2*ln(N)/n) is the paper's own
        variance-correction term. Arms with more actual observed spread keep a
        larger exploration bonus; consistent arms converge on exploitation faster.

        The whole bonus is additionally scaled by `1/sqrt(generation+1)`, a
        standard bandit decay schedule: explore broadly in early generations
        when every arm's estimate is still mostly prior, exploit more as real,
        task-specific evidence accumulates. Pure formula change, zero extra cost.
        """
        decay = 1.0 / math.sqrt(self.generation + 1)
        all_operators = [
            ("expert_refine", self._op_expert_refine),
            ("failure_guided", self._op_failure_guided),
            ("crossover", self._op_crossover),
            ("trajectory", self._op_trajectory),
            ("semantic_var", self._op_semantic_variation),
            ("few_shot", self._op_few_shot),
        ]

        total_pulls = sum(len(v) for v in self._operator_scores.values())

        scored = []
        for name, fn in all_operators:
            pulls = self._operator_scores.get(name, [])
            n = len(pulls)
            if n == 0:
                ucb = float("inf")
            else:
                mean = sum(pulls) / n
                if n > 1:
                    var = sum((x - mean) ** 2 for x in pulls) / n
                else:
                    var = 0.25  # max possible variance for a [0,1]-bounded reward
                log_term = math.log(max(total_pulls, 2)) / n
                v_i = var + math.sqrt(2 * log_term)
                ucb = mean + decay * self.ucb_c * math.sqrt(log_term * min(0.25, v_i))
            scored.append((ucb, name, fn))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [(name, fn) for _, name, fn in scored[:4]]
