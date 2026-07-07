"""CAPO — Confidence-Aware Prompt Optimization.

Based on Zehle et al., arXiv:2504.16005. Uses statistical confidence bounds
to make optimization decisions:

1. Racing evaluation with Hoeffding bounds for candidate comparison
2. Confidence-aware selection (only promote if statistically significant)
3. Hill-climbing with statistical validation
4. Budget-aware early stopping

Key features:
- Statistically rigorous candidate comparison
- Avoids promoting candidates that are only marginally better (noise)
- Budget-efficient through racing
- Conservative but reliable improvements
"""
from __future__ import annotations

import logging
import math
import random
from typing import Any, Dict, List, Optional

from pof.core.types import EvalResult, PromptRecord
from pof.optimizers import register_optimizer
from pof.optimizers.base import BaseOptimizer

logger = logging.getLogger(__name__)


@register_optimizer("capo")
class CAPOOptimizer(BaseOptimizer):
    """CAPO — Confidence-Aware Prompt Optimization.

    Reference: Zehle et al., arXiv:2504.16005.
    """

    name = "capo"

    def __init__(
        self,
        llm,
        dataset,
        evaluator,
        population_size: int = 5,
        num_iterations: int = 5,
        confidence_level: float = 0.05,
        min_improvement: float = 0.02,
        hill_climb_attempts: int = 3,
        **kwargs,
    ):
        super().__init__(
            llm=llm,
            dataset=dataset,
            evaluator=evaluator,
            population_size=population_size,
            num_iterations=num_iterations,
            **kwargs,
        )
        self.confidence_level = confidence_level
        self.min_improvement = min_improvement
        self.hill_climb_attempts = hill_climb_attempts

    def _init_population(self) -> List[PromptRecord]:
        """Initialize with diverse candidates, evaluated with full confidence."""
        candidates: List[PromptRecord] = []
        train_samples = self.dataset.get_few_shot_examples(n=5)

        # Diverse initialization
        lamarckian = self._lamarckian_generate(train_samples, n=3)
        for text in lamarckian:
            candidates.append(self._create_record(text, operator="lamarckian_init"))

        semantic = self._semantic_variation(
            self.seed_prompt or "Solve the task.", n=3
        )
        for text in semantic:
            candidates.append(self._create_record(text, operator="semantic_init"))

        if self.seed_prompt:
            candidates.append(self._create_record(self.seed_prompt, operator="seed"))

        # Full evaluation for initial confidence
        self._evaluate_population(candidates)
        return self._select_top_k(candidates)

    def _step(self) -> List[PromptRecord]:
        """Confidence-aware optimization step.

        For each candidate in population:
        1. Generate improvement attempts (hill climbing)
        2. Evaluate with racing
        3. Only accept if statistically significantly better
        """
        logger.info(f"[CAPO Gen {self.generation}] Confidence-aware step")
        improved_population = list(self.population)

        for i, record in enumerate(self.population[:self.population_size]):
            # Try to improve this candidate
            improved = self._hill_climb_with_confidence(record)
            if improved and improved.score > record.score:
                # Check if improvement is statistically significant
                if self._is_significant_improvement(record, improved):
                    improved_population.append(improved)
                    logger.info(
                        f"  Candidate {i}: improved {record.score:.3f} → {improved.score:.3f} "
                        f"(significant)"
                    )
                else:
                    logger.debug(
                        f"  Candidate {i}: improvement not significant "
                        f"({record.score:.3f} → {improved.score:.3f})"
                    )

        return self._select_top_k(improved_population)

    def _hill_climb_with_confidence(self, record: PromptRecord) -> Optional[PromptRecord]:
        """Attempt hill climbing with multiple strategies."""
        best_candidate = None
        best_score = record.score

        for attempt in range(self.hill_climb_attempts):
            # Choose improvement strategy
            strategy = random.choice(["feedback", "semantic", "trajectory"])

            if strategy == "feedback":
                samples = self.dataset.get_eval_samples("dev", n=20)
                result = self.evaluator.evaluate(record.text, samples)
                failures = [d for d in result.per_sample_details if not d["correct"]]
                if failures:
                    new_text = self._feedback_improve(record.text, failures)
                else:
                    continue
            elif strategy == "semantic":
                variations = self._semantic_variation(record.text, n=1)
                new_text = variations[0] if variations else None
            else:  # trajectory
                context = "\n".join(
                    f"Score: {r.score:.3f} | {r.text[:60]}"
                    for r in sorted(self.population, key=lambda r: r.score)[-3:]
                )
                meta_prompt = (
                    f"Improve this instruction based on what works well:\n\n"
                    f"Context (best performers):\n{context}\n\n"
                    f"Current instruction:\n{record.text}\n\n"
                    f"Improved instruction:"
                )
                new_text = self._generate_prompt(meta_prompt, temperature=0.7)

            if not new_text or not new_text.strip():
                continue

            # Evaluate with racing against current best
            candidate = self._create_record(
                new_text.strip(), operator=f"hill_climb_{strategy}",
                parent_ids=[record.id]
            )
            samples = self.dataset.get_eval_samples("dev", n=self.eval_sample_size)
            result = self.evaluator.evaluate_with_racing(
                candidate.text, samples, best_score,
                confidence=self.confidence_level,
            )
            candidate.score = result.score
            candidate.performance_vector = result.performance_vector
            candidate.scores["dev"] = result.score

            if candidate.score > best_score:
                best_candidate = candidate
                best_score = candidate.score

        return best_candidate

    def _is_significant_improvement(
        self, baseline: PromptRecord, candidate: PromptRecord
    ) -> bool:
        """Check if improvement is statistically significant using Hoeffding bound."""
        n = len(candidate.performance_vector)
        if n == 0:
            return False

        improvement = candidate.score - baseline.score

        # Hoeffding bound for the difference
        bound = math.sqrt(math.log(2.0 / self.confidence_level) / (2 * n))

        # Significant if improvement exceeds bound AND minimum threshold
        return improvement > bound and improvement >= self.min_improvement