"""Held-out final-selection mixin — shared winner's-curse correction.

Bare argmax selection over a fixed, reused dev pool is upward-biased once
enough candidates are compared. Mechanism: reserve ~30% of dev (floor
`MIN_HOLDOUT_N`) that search never touches; at `_finalize()`, re-rank the
top `FINALIZE_TOP_K` dev-argmax finalists on that untouched slice exactly
once, and report the holdout winner instead of the dev-pool argmax.
Factored out here so any `BaseOptimizer` subclass can opt in with one
`_init_holdout()` call.
"""
from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional

from pof.core.types import OptimizationResult, PromptRecord

logger = logging.getLogger(__name__)


class HoldoutSelectionMixin:
    """Mix in before `BaseOptimizer` in the MRO, e.g.
    `class Foo(HoldoutSelectionMixin, BaseOptimizer)`. Call
    `self._init_holdout(use_holdout_selection=...)` from `__init__` after
    `super().__init__(...)` (needs `self.dataset` to already be set)."""

    MIN_HOLDOUT_N = 12
    FINALIZE_TOP_K = 6

    def _init_holdout(self, use_holdout_selection: bool = True) -> None:
        self.use_holdout_selection = use_holdout_selection
        self._opt_pool: Optional[List[Dict[str, str]]] = None
        self._holdout: Optional[List[Dict[str, str]]] = None
        self._holdout_winner: Optional[PromptRecord] = None
        if self.use_holdout_selection:
            full_dev = self.dataset.get_eval_samples("dev", n=None)
            pool = list(full_dev)
            random.Random(42).shuffle(pool)
            n_dev = len(pool)
            n_opt = max(1, n_dev - max(self.MIN_HOLDOUT_N, round(0.30 * n_dev)))
            n_opt = min(n_opt, n_dev)
            self._opt_pool = pool[:n_opt]
            self._holdout = pool[n_opt:]
            logger.info(
                f"[{self.name}] holdout selection enabled: dev={n_dev} -> "
                f"optimization pool={len(self._opt_pool)}, "
                f"held-out selection slice={len(self._holdout)}"
            )

    def _sample_dev(self, n: Optional[int], seed: int = 42) -> List[Dict[str, str]]:
        """Sample only from the search-visible pool when holdout selection
        is enabled -- never the holdout slice."""
        if not self.use_holdout_selection or self._opt_pool is None:
            return super()._sample_dev(n, seed=seed)  # type: ignore[misc]
        pool = self._opt_pool
        if n is None or n >= len(pool):
            return list(pool)
        return random.Random(seed).sample(pool, n)

    def _finalize(self) -> None:
        """Re-rank the top contenders on the never-searched holdout slice."""
        if self._finalized:  # type: ignore[attr-defined]
            return
        if self.use_holdout_selection:
            self._select_on_holdout()
        self._finalized = True  # type: ignore[attr-defined]

    def _select_on_holdout(self) -> None:
        if not self._holdout:
            return
        # Only rank candidates that received a genuine full dev evaluation
        # -- a gate-rejected candidate's `.score` is a 16-sample minibatch
        # score while a gate-passed candidate's `.score` is a 50-sample
        # full-dev score; those are on incomparable variance scales, so a
        # lucky minibatch-only score must not be able to outrank a true
        # full-dev score into a finalist slot.
        dev_scored = [
            r for r in self.tracker.history.records.values()  # type: ignore[attr-defined]
            if r.text and "dev" in r.scores
        ]
        contenders = sorted(
            dev_scored, key=lambda r: r.scores["dev"], reverse=True
        )[: self.FINALIZE_TOP_K]
        if not contenders:
            return

        for record in contenders:
            res = self.evaluator.evaluate(record.text, self._holdout)  # type: ignore[attr-defined]
            record.scores["holdout"] = res.score
            record.scores["holdout_n"] = float(len(self._holdout))

        winner = max(contenders, key=lambda r: r.scores.get("holdout", 0.0))
        self._holdout_winner = winner
        opt_best = contenders[0]
        note = (
            f"holdout selection over {len(contenders)} finalists on "
            f"{len(self._holdout)} held-out instances: winner holdout="
            f"{winner.scores.get('holdout', 0.0):.3f} (opt-pool={winner.scores.get('dev', 0.0):.3f}); "
            f"opt-pool argmax would have picked holdout="
            f"{opt_best.scores.get('holdout', 0.0):.3f} (opt-pool={opt_best.scores.get('dev', 0.0):.3f})"
            + ("" if winner is opt_best else " -- DIFFERENT prompt chosen")
        )
        logger.info(f"[{self.name}] {note}")  # type: ignore[attr-defined]
        self.tracker.add_note(note)  # type: ignore[attr-defined]

    def optimize(self) -> OptimizationResult:
        """Run the search, then report the held-out winner if one was chosen."""
        result = super().optimize()  # type: ignore[misc]
        winner = self._holdout_winner
        if winner is not None and winner.text:
            result.best_prompt = winner.text
            # Report the holdout score, not the still-upward-biased
            # opt-pool dev score -- the holdout score is what actually
            # decided the winner.
            result.best_score = winner.scores.get("holdout", winner.score)
        return result
