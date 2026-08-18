"""FUNNELv4d — FUNNELv4c plus batch-level Hoeffding racing for brand-new
candidates.

Racing lets a candidate's evaluation stop early once its running score plus
a Hoeffding bound can no longer reach a threshold (see
`Evaluator.evaluate_with_batch_racing`). Unlike the codebase's existing
`evaluate_with_racing` (used by CAPO/GAAPO/GSPE/SEE), which checks the bound
after every sample and so cannot batch generation calls, this checks between
BATCHES, keeping full batching throughput within a batch.

Applied only to brand-new candidates (zero prior evaluation), never to a
survivor's incremental re-evaluation — cutting a survivor's evidence short
would reintroduce the winner's-curse bias `_evaluate_phase` guards against.
The elimination bar is the population's floor score at the start of the
phase (before new candidates are added); only active from phase 1 onward,
since phase 0 has no stable floor yet.
"""
from __future__ import annotations

import logging
from typing import Dict, List

from pof.core.types import PromptRecord
from pof.optimizers import register_optimizer
from pof.optimizers.funnel_v4c import FUNNELv4cOptimizer

# v4d is the best-performing FUNNEL variant measured to date, achieved with
# less guaranteed work per phase than v3/v4a. "FUNNEL-Lean" names that result
# for use outside this codebase (papers, reports).

logger = logging.getLogger(__name__)

# Looser than the evaluator's 0.05 default: elimination only needs to be
# right on average across many raced candidates, not any single call, and a
# tighter bound (higher confidence) would rarely trigger early enough to save
# anything.
RACING_CONFIDENCE = 0.10
RACING_MIN_BATCHES = 1


@register_optimizer("funnel_v4d")
class FUNNELv4dOptimizer(FUNNELv4cOptimizer):
    """FUNNELv4c with batch-level racing for brand-new candidates."""

    name = "funnel_v4d"

    def _order_dev_pool(self, pool: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Front-load instances that look more complex into early phases.

        A true discriminative-instance ordering would need historical
        per-dev-instance correctness data, which this project doesn't
        persist (only test-split details are saved, and dev/test are
        disjoint). Input length is used as a cheap, data-available proxy:
        longer inputs tend to be more discriminative, so the accumulating-
        fresh schedule's early, cheap phases see more separating signal.
        Stable sort preserves the seeded shuffle within equal lengths.
        """
        return sorted(pool, key=lambda s: len(str(s.get("input", ""))), reverse=True)

    def _eval_overrides(self, record: PromptRecord):
        """(system_prompt_override, max_new_tokens_override) for one record.

        (None, None) here -- use the evaluator's own per-run defaults.
        Overridden by funnel_v5 to route each record through the eval mode
        stored in its own metadata rather than one fixed per-run setting.
        """
        return None, None

    def _evaluate_phase(self, candidates: List[PromptRecord], phase: int) -> None:
        n_target = self._phase_sizes[min(phase, len(self._phase_sizes) - 1)]
        threshold = None
        if phase >= 1 and self.population:
            threshold = min(r.score for r in self.population)

        n_full, n_incr, n_cached, n_raced = 0, 0, 0, 0

        for record in candidates:
            if not record.text:
                continue
            cached = list(record.per_sample_details or [])
            have = len(cached)
            sp_override, tok_override = self._eval_overrides(record)

            if have >= n_target:
                details = cached[:n_target]
                n_cached += 1
            else:
                increment = self._dev_pool[have:n_target]
                if not increment:
                    continue
                if have == 0 and threshold is not None:
                    res = self.evaluator.evaluate_with_batch_racing(
                        record.text, increment, threshold=threshold,
                        confidence=RACING_CONFIDENCE, min_batches=RACING_MIN_BATCHES,
                        system_prompt_override=sp_override, max_new_tokens_override=tok_override,
                    )
                    if res.metadata.get("racing_terminated"):
                        n_raced += 1
                    else:
                        n_full += 1
                else:
                    res = self.evaluator.evaluate(
                        record.text, increment,
                        system_prompt_override=sp_override, max_new_tokens_override=tok_override,
                    )
                    if have == 0:
                        n_full += 1
                    else:
                        n_incr += 1
                details = cached + list(res.per_sample_details)

            em = sum(1.0 for d in details if d.get("correct")) / max(len(details), 1)
            mean_len = self._mean_output_tokens(details)
            penalized = self._barrier_score(em, mean_len)

            record.per_sample_details = details
            record.performance_vector = [1.0 if d.get("correct") else 0.0 for d in details]
            record.scores["dev"] = em
            record.scores["gate_score"] = penalized
            record.scores["out_len"] = mean_len
            record.scores["eval_n"] = float(len(details))
            record.score = max(0.0, penalized)

        logger.info(
            f"[{self.name} Phase {phase}] all candidates at N={n_target} "
            f"({n_full} full, {n_incr} incremental, {n_cached} cached, "
            f"{n_raced} raced-out)"
        )

    def _step(self) -> List[PromptRecord]:
        result = super()._step()
        self._maybe_stop_if_perfect()
        return result

    def _maybe_stop_if_perfect(self) -> None:
        """Stop early once the best candidate is a perfect match on dev.

        Remaining phases only grow the sample the best candidate is scored
        on (accumulating-fresh), so EM=1.0 means no higher score is left to
        find. Checks raw EM (`scores["dev"]`), not the barrier-penalized
        selection score — a length penalty pulling `score` below 1.0 doesn't
        mean a higher-accuracy candidate remains to be found.
        """
        if not self.best_record:
            return
        em = self.best_record.scores.get("dev")
        if em is not None and em >= 1.0:
            note = (
                f"perfect dev score (EM=1.0) reached after phase {self._phase_idx} "
                f"-- stopping early, remaining phases would only re-confirm it on more data"
            )
            logger.info(f"[{self.name}] {note}")
            self.tracker.add_note(note)
            self._finalize()
            raise StopIteration


# Alias: same class, registered under the paper-facing name too. Both
# "funnel_v4d" and "funnel_lean" resolve to FUNNELv4dOptimizer; results land
# under whichever name is passed to run_swift_apex.py's --methods.
register_optimizer("funnel_lean")(FUNNELv4dOptimizer)
