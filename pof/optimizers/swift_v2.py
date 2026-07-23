"""SWIFTv2 — SWIFT with Phase 2 rebuilt from operator-effectiveness audit data.

Candidate-level audit data across 181 SWIFT v1 runs (see analysis session) ranked
all 13 operators by mean dev score and by how often each one actually produced the
final winning prompt. Two findings drove this redesign:

- `trajectory_climb` (OPRO-style, Phase 2) was the weakest operator in the library
  by BOTH measures: last place on mean dev score (0.476) and second-to-last on
  win-rate (4.4% of runs). It rarely helps and almost never wins outright.
- `crossover` (also Phase 2) has a middling average (0.535) but is the
  third-most-common source of the final winning prompt (13.3%) — a genuine
  high-variance, high-reward operator, kept unchanged.

Phase 2 v2 replaces `trajectory_climb` with two operators expected to outperform it:

1. **A second failure-guided pass** — the single most reliable operator in the
   library by both measures (#1 mean dev score at 0.611, #2 win-rate at 14.9%).
   Phase 1 only gives each elite one shot at failure-guided refinement; this
   gives it a second pass against the population as it stands after Phase 1.
2. **A mid-search local edit** — `local_edit_polish` (Phase 3 only, in v1) is
   empirically the single most common source of the final winning prompt across
   all 13 operators (18.2% of runs), despite an unremarkable average dev score.
   This tests whether that reliability holds earlier in the search, not just at
   the polish step.

Crossover itself is also re-targeted, not just kept: v1 pairs elites
sequentially — (0,1), (1,2), (2,3) — which wastes one of its three crossovers
on two mediocre parents (elites 2 and 3). v2 anchors every crossover on the
current best elite instead (best×1, best×2, best×3), the standard elitist-
mating strategy in GA literature, so every crossover attempt carries the best
specimen's material rather than sometimes combining two weaker ones.

Also mixes in `LengthAwareDedupeMixin` (see `_v2_common.py`): selection now
nudges toward shorter prompts when candidates are within noise of each other
on raw dev score (CAPO-style length-penalized fitness), and duplicate
detection catches near-identical near-duplicates via `difflib`, not just
byte-identical text — both free, CPU-only additions shared with APEXv2.

This is a v2 for controlled comparison, not a replacement — registered as a
separate optimizer name so existing "swift" results and their comparability with
prior runs are untouched. Phase 0 (diverse seeding), Phase 1 (failure-guided
improvement), and Phase 3 (polish) are inherited from SWIFT unchanged.
"""
from __future__ import annotations

import logging
from typing import List

from pof.core.types import PromptRecord
from pof.optimizers import register_optimizer
from pof.optimizers._v2_common import LengthAwareDedupeMixin
from pof.optimizers.swift import SWIFTOptimizer

logger = logging.getLogger(__name__)


@register_optimizer("swift_v2")
class SWIFTv2Optimizer(LengthAwareDedupeMixin, SWIFTOptimizer):
    """SWIFT v2 — Phase 2 rebuilt from operator-effectiveness audit data.

    Reference for the operator-effectiveness data driving this design: audit
    trail analysis of 181 SWIFT v1 runs (candidate-level dev scores and
    final-winner attribution per operator).
    """

    name = "swift_v2"

    def _phase_trajectory_crossover(self) -> List[PromptRecord]:
        """Phase 2 v2: failure-guided (2nd pass) + mid-search local edit + crossover.

        Drops trajectory_climb (weakest operator by both mean score and win-rate).
        """
        logger.info("[SWIFTv2 Phase 2] Failure-guided (2nd pass) + Local edit + Crossover")
        candidates = list(self.population)

        # Second failure-guided pass on the top-2 elites: re-diagnose against
        # the population as it stands after Phase 1, giving the single
        # best-performing operator in the library a second chance.
        for record in self.population[:2]:
            details = record.per_sample_details
            if not details and record.text:
                samples = self.dataset.get_eval_samples("dev", n=self.eval_sample_size)
                result = self.evaluator.evaluate(record.text, samples)
                details = result.per_sample_details
                record.per_sample_details = details
                record.score = result.score
                record.performance_vector = result.performance_vector
            failures = [d for d in details if not d["correct"]]
            if failures:
                improved = self._structured_improve(record.text, failures)
                if improved and not self._is_duplicate(improved):
                    new_record = self._create_record(
                        improved, operator="failure_guided_phase2", parent_ids=[record.id]
                    )
                    candidates.append(new_record)
                    # Same paraphrase-expansion Phase 1 gives failure_guided ->
                    # failure_guided_var, empirically the single best operator
                    # in the library (0.611 avg vs. raw failure_guided's 0.504)
                    # -- give the second pass the same treatment.
                    variants = self._semantic_variation(improved, n=1)
                    if variants and not self._is_duplicate(variants[0]):
                        candidates.append(self._create_record(
                            variants[0], operator="failure_guided_phase2_var",
                            parent_ids=[record.id],
                        ))

        # Mid-search local edit on the top-2 elites: local_edit_polish is the
        # single most common source of the final winning prompt in v1 (18.2%
        # of runs), currently only tried in Phase 3 — test it here too.
        for record in self.population[:2]:
            edited = self._local_edit(record.text)
            if edited and not self._is_duplicate(edited):
                candidates.append(self._create_record(
                    edited, operator="local_edit_mid", parent_ids=[record.id]
                ))

        # Crossover: elite-anchored pairing instead of v1's sequential adjacent
        # pairs (0,1),(1,2),(2,3). v1 wastes one of its three crossovers on two
        # mediocre parents (elites 2 and 3); pairing the current best against
        # each other elite instead (standard elitist-mating strategy in GA
        # literature) means every crossover attempt carries the best
        # specimen's material, maximizing the chance of inheriting it.
        best = self.population[0]
        for other in self.population[1:4]:
            cross = self._crossover(best.text, other.text)
            if cross.strip() and not self._is_duplicate(cross):
                candidates.append(self._create_record(
                    cross, operator="crossover", parent_ids=[best.id, other.id]
                ))

        new_candidates = [c for c in candidates if c.score == 0.0]
        baseline = self.best_record.score if self.best_record else 0.0
        self._evaluate_with_minibatch_gate(new_candidates, baseline)
        return self._select_top_k(candidates)
