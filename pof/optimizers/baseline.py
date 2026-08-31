"""BaselineSeed — zero-search reference point: evaluate the seed prompt once.

Exists to answer one question honestly: how much did optimization actually
buy over the unoptimized starting prompt, under the *exact* protocol the
optimized runs used (same split, same CoT/task_type config, same eval
harness)? `experiments/bbh_reference_baseline.py` answers a related but
different question -- it reproduces Suzgun et al.'s answer-only 3-shot setup,
which runs under a 32-token, no-CoT cap, not this project's CoT protocol. A
baseline measured under a different eval config than the optimized runs is
not a fair "before" number for them.

This class reuses the standard orchestration path (`_init_population`
evaluates the seed once, `_step` immediately raises `StopIteration`) so it
inherits identical dataset splits, CoT config, and held-out test evaluation
to every other method launched through `run_swift_apex.py` -- the only
degree of freedom removed is search itself.
"""
from __future__ import annotations

from typing import List

from pof.core.types import PromptRecord
from pof.optimizers import register_optimizer
from pof.optimizers.base import BaseOptimizer


@register_optimizer("baseline_seed")
class BaselineSeedOptimizer(BaseOptimizer):
    """Evaluates only the seed prompt -- no candidate generation, no search."""

    name = "baseline_seed"
    tier = "baseline"

    def _init_population(self) -> List[PromptRecord]:
        if not self.seed_prompt:
            raise ValueError(f"[{self.name}] requires a seed_prompt; none provided")
        return [self._create_record(self.seed_prompt, operator="seed")]

    def _evaluate_population(self, population: List[PromptRecord]) -> None:
        """No-op: skip the dev-set pass entirely.

        `BaseOptimizer.optimize()` unconditionally evaluates the population
        on the dev split right after `_init_population()` -- overriding
        `_init_population` alone doesn't skip it, since that call lives in
        the template method, not delegated to a hook. There is exactly one
        candidate and no search here, so a dev score only picks a winner
        among candidates that don't exist; it costs a full dev-set eval pass
        (the dominant cost of a baseline_seed run) for a number nothing
        downstream reads. The runner's own held-out test evaluation, run on
        `best_record` regardless of dev score, is the only number this
        method reports.
        """
        return

    def _step(self) -> List[PromptRecord]:
        raise StopIteration
