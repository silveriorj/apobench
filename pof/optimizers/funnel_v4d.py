"""FUNNELv4d — FUNNELv4c plus batch-level Hoeffding racing for brand-new
candidates.

**The idea.** Every phase, `_evaluate_phase` pays for a full increment of
evaluation on every candidate, including ones that are obviously going to
lose the tournament. Racing lets a candidate's evaluation stop early once its
running score plus a Hoeffding confidence bound can no longer reach a
threshold — see `Evaluator.evaluate_with_batch_racing`.

**Why batch-level, not the existing `evaluate_with_racing`.** That method
already exists in the codebase (used by CAPO/GAAPO/GSPE/SEE) but checks the
bound after every SAMPLE, which means it cannot batch generation calls at
all. At FUNNEL's per-phase N (22-88, batch_size ~4-8) the lost batching
throughput for candidates that do NOT get eliminated likely costs more than
early elimination saves on the ones that do. `evaluate_with_batch_racing`
checks the bound between BATCHES instead, keeping full batching throughput
within a batch.

**Scope: brand-new candidates only, never survivors.** Racing is applied only
to candidates with zero prior evaluation (`have == 0` in `_evaluate_phase`'s
terms) -- never to a survivor's incremental re-evaluation. Survivors are the
ones equal-N comparability exists to protect; cutting a survivor's evidence
short would reintroduce exactly the winner's-curse bias `_evaluate_phase`'s
own docstring warns against. New candidates racing out just means the search
stops paying to confirm what the population's current floor already implies.

**Threshold and timing.** The elimination bar is the CURRENT population's
floor score (`min(r.score for r in self.population)`) at the start of the
phase, before this phase's new candidates are added -- i.e. what a new
candidate needs to beat to have any chance of displacing an existing
survivor. Only active from phase 1 onward: phase 0 (`_init_population`)
builds `self.population` incrementally with no stable floor yet to race
against.
"""
from __future__ import annotations

import logging
from typing import Dict, List

from pof.core.types import PromptRecord
from pof.optimizers import register_optimizer
from pof.optimizers.funnel_v4c import FUNNELv4cOptimizer

# v4d is the best-performing FUNNEL variant measured to date (Qwen3-4B, 7
# tasks common across the whole family): macro 0.715 vs. v4a's 0.701, v3's
# 0.699, v2's 0.689, v1's 0.677 -- the best or tied-best on 5 of 7 tasks,
# achieved with LESS guaranteed work per phase than v3/v4a (trimmed
# decomposition family + batch-level racing). "FUNNEL-Lean" names that result
# for use outside this codebase (papers, reports): best measured accuracy,
# achieved cheaper, not despite being cheaper.

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

        A true discriminative-instance ordering (prioritize instances where
        past candidates' scores disagreed most, per the "reward variance"
        literature on prompt-overfitting) needs historical per-DEV-instance
        correctness data. This project only persists TEST-split per-sample
        details in result files, and dev/test are disjoint splits by
        construction -- test-instance difficulty tells you nothing about a
        dev instance's difficulty, so that data doesn't actually support the
        real thing. This is a substitute that IS supportable from data
        available up front, with no circularity: input length as a cheap
        complexity proxy. Longer/more complex-looking inputs tend to be more
        discriminative (more ways to get partially or fully wrong), so
        putting them early means the accumulating-fresh schedule's early,
        cheap phases see more of the signal that actually separates
        candidates, rather than however a random shuffle happened to land.

        Stable sort: within a length, the seeded random order set before
        this hook runs is preserved, so this doesn't reduce the value of the
        RNG shuffle for tasks/inputs where length carries little signal.
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

        Remaining phases only grow the sample the best candidate is scored on
        (accumulating-fresh) -- if it is already at EM=1.0, there is no higher
        score left to find; further phases would only re-confirm the same
        result on more data at real GPU cost, which matters a lot more once
        a "phase" is a batch of full CoT generations on an Ollama-served
        model with no request parallelism (single-digit minutes per batch vs.
        seconds on the HF/answer-only path this project mostly runs).
        Checks raw EM (`scores["dev"]`), not the barrier-penalized selection
        score, since a length penalty reducing `score` below 1.0 doesn't mean
        there's a HIGHER-accuracy candidate still to find -- it means a
        shorter answer might score marginally higher under the barrier, which
        isn't what this check is for.
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
