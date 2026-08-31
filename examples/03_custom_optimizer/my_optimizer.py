"""Example custom optimizer for APOBench.

Minimal but complete: generates a small population of paraphrases each
generation and keeps the best-scoring ones. Copy this file as a starting
point for your own method.

To wire it in, see the two steps at the bottom of this file's docstring
and in README.md — writing the class is NOT enough on its own.
"""
from __future__ import annotations

import logging
from typing import List

from pof.core.types import PromptRecord
from pof.optimizers import register_optimizer
from pof.optimizers.base import BaseOptimizer

logger = logging.getLogger(__name__)


@register_optimizer("my_method")
class MyOptimizer(BaseOptimizer):
    """Toy optimizer: paraphrase-and-select each generation.

    Every BaseOptimizer subclass must implement exactly two methods:
    - _init_population(): build and evaluate the starting candidate pool
    - _step(): produce ONE generation's new candidate pool (return the
      updated population; base.py's optimize() loop applies it and calls
      _step() again until num_iterations, a budget cap, or StopIteration)

    Everything else (audit trail, budget tracking, evaluation-with-racing,
    dedup, shared generation techniques like _semantic_variation/_crossover)
    is inherited from BaseOptimizer — see pof/optimizers/base.py.
    """

    name = "my_method"
    # Left at the default deliberately -- this is a toy example, not a claim
    # about beating anything. See README.md for what "contribution" and
    # "baseline" mean and when to use them.
    tier = "in_house"

    def _init_population(self) -> List[PromptRecord]:
        candidates = [self._create_record(self.seed_prompt, operator="seed")]

        # _semantic_variation() is a shared BaseOptimizer helper: asks the
        # LLM to paraphrase a prompt while preserving its meaning.
        variations = self._semantic_variation(self.seed_prompt, n=self.population_size - 1)
        for text in variations:
            candidates.append(self._create_record(text, operator="paraphrase_init"))

        # _evaluate_population() runs the full dev-set eval (with racing,
        # if enabled in your config) and sets each record's .score in place.
        self._evaluate_population(candidates)
        return candidates

    def _step(self) -> List[PromptRecord]:
        candidates = list(self.population)

        for record in self.population:
            variants = self._semantic_variation(record.text, n=1)
            for text in variants:
                if not self._is_duplicate(text):
                    candidates.append(self._create_record(
                        text, operator="paraphrase", parent_ids=[record.id]
                    ))

        new_candidates = [c for c in candidates if c.score == 0.0]
        self._evaluate_population(new_candidates)

        # Keep the top population_size by score.
        candidates.sort(key=lambda r: r.score, reverse=True)
        return candidates[: self.population_size]


# -----------------------------------------------------------------------
# Wiring this in (both steps required — the decorator alone is not enough):
#
# 1. Registration happens at IMPORT time, not at decoration time. This file
#    only registers "my_method" once something actually imports it. For a
#    real (non-example) optimizer, add it to _load_all() in
#    pof/optimizers/__init__.py:
#
#        from pof.optimizers import my_optimizer  # noqa: F401
#
#    Without that line, `pof list` / `get_optimizer("my_method")` will not
#    see it even though the class exists and is decorated.
#
# 2. For THIS example specifically, run.py imports my_optimizer.py directly
#    (in the same Python process) before building the orchestrator, so step
#    1 isn't needed just to try it out — see run.py. `pof run` on the CLI is
#    a separate process and would NOT see this registration without step 1.
#
# Optional: HoldoutSelectionMixin (pof/optimizers/holdout.py) is used by
# several optimizers in this project use `HoldoutSelectionMixin` to correct winner's-curse bias in final-prompt
# selection (it reserves a slice of dev for selection, separate from the
# slice search optimizes against). It is NOT required by BaseOptimizer —
# mix it in only if your method's final selection step also argmaxes over
# a dev set your search already optimized against.
# -----------------------------------------------------------------------
