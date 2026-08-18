"""FUNNEL — UCB1-scheduled search over a pooled top-20 operator library that
shrinks empirically across runs.

Pools the best-performing operators across every method evaluated in this
study (see `_funnel_techniques.py` for selection/dedup), then prunes that
library over time from validated cross-run evidence rather than a curated
order fixed up front (SWIFT) or narrowing within a single run (APEX).

Selection within a run is UCB1 over the active pool (see `_select_operators`;
`ucb_c` is tuned for this budget, not derived from the regret bound, and
only the top-M arms are pulled per round). Pruning is cross-run: a
technique is dropped only once it has accumulated enough independent
observations (default n >= 40) to be statistically below the pool average
(`_funnel_stats.prune_techniques`) — never judged on a single run's noise.
Techniques with enough history also warm-start this run's UCB1 priors
(empirical-Bayes, computed live from the audit trail).
"""
from __future__ import annotations

import logging
import math
import random
from typing import Dict, List, Optional, Tuple

from pof.core.types import PromptRecord
from pof.optimizers import register_optimizer
from pof.optimizers.base import BaseOptimizer
from pof.optimizers._funnel_stats import prune_techniques, scan_technique_stats
from pof.optimizers._funnel_techniques import (
    ALL_TECHNIQUES,
    BOOTSTRAP_TECHNIQUES,
    SINGLE_RECORD_TECHNIQUES,
)

logger = logging.getLogger(__name__)


@register_optimizer("funnel")
class FUNNELOptimizer(BaseOptimizer):
    """FUNNEL — UCB1-scheduled search over a top-20 pooled library, pruned
    across runs by accumulated cross-run evidence.

    Reference for the operator pool: audit of 889 runs across GAAPO, SEE,
    CAPO, SWIFT, APEX, and GEPA (see `_funnel_techniques.py`). Reference
    for the pruning/warm-start mechanism: `_funnel_stats.py`.
    """

    name = "funnel"

    def __init__(
        self,
        llm,
        dataset,
        evaluator,
        population_size: int = 5,
        num_iterations: int = 5,
        candidates_per_operator: int = 2,
        top_m_operators: int = 8,
        ucb_c: float = 0.5,
        prior_pulls: int = 3,
        min_n_for_pruning: int = 40,
        prune_z_threshold: float = 1.64,
        output_dir: str = "outputs",
        **kwargs,
    ):
        super().__init__(
            llm=llm,
            dataset=dataset,
            evaluator=evaluator,
            population_size=population_size,
            num_iterations=num_iterations,
            output_dir=output_dir,
            **kwargs,
        )
        self.candidates_per_operator = candidates_per_operator
        self.top_m_operators = top_m_operators
        self.ucb_c = ucb_c

        historical = scan_technique_stats(
            output_dir, method_name=self.name, technique_names=set(ALL_TECHNIQUES.keys())
        )
        self._active_techniques, self._dropped_techniques = prune_techniques(
            historical,
            list(ALL_TECHNIQUES.keys()),
            min_n=min_n_for_pruning,
            z_threshold=prune_z_threshold,
        )
        if self._dropped_techniques:
            logger.info(
                f"[FUNNEL] pruned {len(self._dropped_techniques)} technique(s) on cross-run "
                f"evidence (n>={min_n_for_pruning}, z<=-{prune_z_threshold}): "
                f"{', '.join(self._dropped_techniques)}"
            )
        logger.info(f"[FUNNEL] active library: {len(self._active_techniques)}/{len(ALL_TECHNIQUES)} techniques")

        # Warm-start: `prior_pulls` pseudo-observations at each technique's
        # historical mean. Techniques with no history start cold (empty
        # list -> infinite UCB value -> tried once before exploitation).
        self._operator_scores: Dict[str, List[float]] = {}
        for op in self._active_techniques:
            if op in historical:
                self._operator_scores[op] = [historical[op].mean] * prior_pulls

    def _init_population(self) -> List[PromptRecord]:
        """Diverse bootstrap: Lamarckian seeds + the raw seed prompt + a
        couple of single-record techniques applied to it, so the pool isn't
        empty before the first UCB1-scheduled iteration."""
        candidates: List[PromptRecord] = []
        train_samples = self.dataset.get_few_shot_examples(n=5)

        for name, fn in BOOTSTRAP_TECHNIQUES.items():
            for _ in range(2):
                text = fn(self)
                if text and not self._is_duplicate(text):
                    candidates.append(self._create_record(text, operator=name))

        if self.seed_prompt:
            candidates.append(self._create_record(self.seed_prompt, operator="seed"))

        # A small, fixed diversity sample from the active pool's
        # single-record techniques, so init doesn't depend on which arms
        # UCB1 would pick with zero data.
        bootstrap_sample = [
            t for t in self._active_techniques if t in SINGLE_RECORD_TECHNIQUES
        ][:4]
        self.population = candidates or [self._create_record("Solve the task.", operator="fallback_seed")]
        for name in bootstrap_sample:
            text = SINGLE_RECORD_TECHNIQUES[name](self)
            if text and not self._is_duplicate(text):
                candidates.append(self._create_record(text, operator=name))

        self._evaluate_population(candidates)
        return self._select_top_k(candidates)

    def _step(self) -> List[PromptRecord]:
        """UCB1-select operators from the active (cross-run-pruned) pool
        and apply them, mirroring APEX's adaptive step exactly."""
        logger.info(f"[FUNNEL Gen {self.generation}] UCB1-scheduled step")
        candidates = list(self.population)

        operators = self._select_operators()
        for name in operators:
            fn = ALL_TECHNIQUES[name]
            for _ in range(self.candidates_per_operator):
                text = fn(self)
                if text and not self._is_duplicate(text):
                    candidates.append(self._create_record(
                        text, operator=name, parent_ids=[r.id for r in self.population[:2]]
                    ))

        new_candidates = [c for c in candidates if c.score == 0.0]
        baseline = self.best_record.score if self.best_record else 0.0
        self._evaluate_with_minibatch_gate(new_candidates, baseline)

        for record in new_candidates:
            self._operator_scores.setdefault(record.operator, []).append(record.score)

        return self._tournament_select(candidates)

    def _select_operators(self) -> List[str]:
        """UCB1 over the active pool: value = mean + c*sqrt(ln(total)/pulls).

        Same formula as APEX's `_select_operators` — see the SWIFT/APEX
        paper's Section 3.2 for why `ucb_c` and the top-M truncation are
        empirical adaptations, not the canonical algorithm.
        """
        total_pulls = sum(len(v) for v in self._operator_scores.values())
        if total_pulls == 0:
            return self._active_techniques[: self.top_m_operators]

        scored: List[Tuple[float, str]] = []
        for name in self._active_techniques:
            pulls = self._operator_scores.get(name, [])
            if not pulls:
                ucb = float("inf")
            else:
                mean = sum(pulls) / len(pulls)
                ucb = mean + self.ucb_c * math.sqrt(math.log(max(total_pulls, 2)) / len(pulls))
            scored.append((ucb, name))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [name for _, name in scored[: self.top_m_operators]]

    def _tournament_select(
        self, candidates: List[PromptRecord], tournament_size: int = 3
    ) -> List[PromptRecord]:
        """Tournament selection with elitism (same as APEX)."""
        sorted_candidates = sorted(candidates, key=lambda r: r.score, reverse=True)
        selected = [sorted_candidates[0]]
        remaining = sorted_candidates[1:]
        while len(selected) < self.population_size and remaining:
            tournament = random.sample(remaining, min(tournament_size, len(remaining)))
            winner = max(tournament, key=lambda r: r.score)
            selected.append(winner)
            remaining.remove(winner)
        return selected
