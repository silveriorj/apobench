"""Shared helpers for v2 optimizer variants (SWIFTv2, APEXv2).

Two additive, zero-extra-LLM-call improvements, applicable to any
generate-then-select evolutionary optimizer, so they live here once rather
than being duplicated in each v2 file:

1. **Length-aware selection** (CAPO-style length-penalized fitness, cf.
   Zehle et al. 2025, already used elsewhere in this study). Current
   prompt-optimization literature calls unchecked prompt growth across
   iterations "prompt distributional overfitting" — failure-guided fixes,
   few-shot appends, and local edits tend to make prompts longer and more
   narrowly rule-laden generation over generation, which hurts
   generalization. `LENGTH_PENALTY_PER_WORD` is deliberately small: a 50-word
   length gap costs about 0.01, well under the real score differences
   observed between operators in the v1 audit data (0.03-0.15) — so it only
   acts as a tie-breaker toward shorter prompts when accuracy is already a
   wash, never overrides a genuine accuracy difference.

2. **Near-duplicate detection.** BaseOptimizer's `_is_duplicate` only catches
   byte-identical text. Paraphrase, crossover, and local-edit operators
   frequently produce output that's textually close but not identical to
   something already in the population — each one still burns a full
   evaluation under v1. This adds a cheap (stdlib `difflib`, no new
   dependency, no LLM call) similarity check against the current population
   before falling through to a real evaluation.
"""
from __future__ import annotations

import random
from difflib import SequenceMatcher
from typing import List, Optional

from pof.core.types import PromptRecord

LENGTH_PENALTY_PER_WORD = 0.0002
NEAR_DUP_SIMILARITY_THRESHOLD = 0.92  # difflib ratio, 1.0 = identical


class LengthAwareDedupeMixin:
    """Mix in before the v1 optimizer class, e.g.

        class SWIFTv2Optimizer(LengthAwareDedupeMixin, SWIFTOptimizer): ...

    so these overrides take priority in the MRO over the v1 base's plain
    exact-hash `_is_duplicate` and raw-score `_select_top_k`.
    """

    def _length_adjusted_score(self, record: PromptRecord) -> float:
        n_words = len(record.text.split()) if record.text else 0
        return record.score - LENGTH_PENALTY_PER_WORD * n_words

    def _select_top_k(
        self, candidates: List[PromptRecord], k: Optional[int] = None
    ) -> List[PromptRecord]:
        k = k or self.population_size
        ranked = sorted(candidates, key=self._length_adjusted_score, reverse=True)
        return ranked[:k]

    def _tournament_select(
        self, candidates: List[PromptRecord], tournament_size: int = 3
    ) -> List[PromptRecord]:
        """Same tournament structure as APEX v1's, ranked by length-adjusted
        score instead of raw score. Unused by SWIFTv2 (no tournament there),
        overrides APEXOptimizer's version for APEXv2."""
        sorted_candidates = sorted(candidates, key=self._length_adjusted_score, reverse=True)
        selected = [sorted_candidates[0]]
        remaining = sorted_candidates[1:]
        while len(selected) < self.population_size and remaining:
            tournament = random.sample(remaining, min(tournament_size, len(remaining)))
            winner = max(tournament, key=self._length_adjusted_score)
            selected.append(winner)
            remaining.remove(winner)
        return selected

    def _is_duplicate(self, text: str) -> bool:
        if super()._is_duplicate(text):  # exact-hash check first (cheap, existing)
            return True
        if not text:
            return True
        candidate = text.strip()
        for record in self.population:
            if not record.text:
                continue
            ratio = SequenceMatcher(None, candidate, record.text.strip()).ratio()
            if ratio >= NEAR_DUP_SIMILARITY_THRESHOLD:
                return True
        return False
