"""SEE — Strategic Exploration & Exploitation optimizer.

Ported from Projeto's best-performing implementation (Algorithm 1 from the paper).
4 sequential phases with tolerance-based convergence:

Phase 0 (Initialization): Generate diverse initial pool via Lamarckian + Semantic operators
Phase 1 (Feedback): Improve top candidates using failure analysis
Phase 2 (Fusion): EDA + Crossover to combine best elements
Phase 3 (Semantic): Fine-grained semantic variations of the best

Key features from Projeto:
- Performance vectors for fine-grained analysis
- Tolerance-based convergence (skip phases if no improvement)
- Efficient LLM usage (~30-35 calls total)
- Proven results with small models (qwen3-4B)

Enhanced with:
- Full PromptRecord lineage tracking
- Audit trail with generation snapshots
- Hoeffding racing for evaluation efficiency
"""
from __future__ import annotations

import logging
import random
from enum import Enum
from typing import Any, Dict, List, Optional

from pof.core.types import PromptRecord
from pof.optimizers import register_optimizer
from pof.optimizers.base import BaseOptimizer

logger = logging.getLogger(__name__)


class SEEPhase(Enum):
    """Phases aligned with paper Section 3.3."""
    INITIALIZATION = 0
    FEEDBACK = 1
    FUSION = 2
    SEMANTIC = 3


class SEEOperator(Enum):
    """Operators from paper Section 3.2."""
    LAMARCKIAN = "O_L"
    FEEDBACK = "O_F"
    EDA = "O_E"
    CROSSOVER = "O_C"
    SEMANTIC = "O_S"


@register_optimizer("see")
class SEEOptimizer(BaseOptimizer):
    """Strategic Exploration & Exploitation optimizer.

    Paper: Cui et al., arXiv:2402.11347
    Implementation: Ported from Projeto (best-performing version).
    """

    name = "see"

    def __init__(
        self,
        llm,
        dataset,
        evaluator,
        population_size: int = 5,
        pool_size_init: int = 15,
        eval_sample_size: int = 50,
        tolerance_threshold: float = 0.02,
        max_stagnation: int = 2,
        seed_prompt: str = "",
        **kwargs,
    ):
        super().__init__(
            llm=llm,
            dataset=dataset,
            evaluator=evaluator,
            population_size=population_size,
            num_iterations=3,  # SEE always has 3 phases after init
            seed_prompt=seed_prompt,
            eval_sample_size=eval_sample_size,
            **kwargs,
        )
        self.pool_size_init = pool_size_init
        self.tolerance_threshold = tolerance_threshold
        self.max_stagnation = max_stagnation
        self._current_phase = SEEPhase.INITIALIZATION
        self._stagnation_count = 0
        self._phase_scores: List[float] = []

    def _init_population(self) -> List[PromptRecord]:
        """Phase 0: Generate diverse initial pool.

        Uses Lamarckian (from I/O pairs) and Semantic (paraphrasing) operators
        to create a diverse initial pool, then selects top-K.
        """
        self._current_phase = SEEPhase.INITIALIZATION
        candidates: List[PromptRecord] = []

        # Get training samples for Lamarckian generation
        train_samples = self.dataset.get_few_shot_examples(n=5)

        # Lamarckian generation: reverse-engineer from I/O pairs
        n_lamarckian = self.pool_size_init // 2
        logger.info(f"[SEE Phase 0] Generating {n_lamarckian} Lamarckian candidates")
        lamarckian_prompts = self._lamarckian_generate(train_samples, n=n_lamarckian)
        for text in lamarckian_prompts:
            record = self._create_record(
                text=text,
                operator=SEEOperator.LAMARCKIAN.value,
            )
            candidates.append(record)

        # Semantic variations of seed prompt (or Lamarckian results)
        n_semantic = self.pool_size_init - len(candidates)
        base_prompts = [self.seed_prompt] if self.seed_prompt else [
            c.text for c in candidates[:3]
        ]
        logger.info(f"[SEE Phase 0] Generating {n_semantic} Semantic candidates")
        for base in base_prompts:
            variations = self._semantic_variation(base, n=n_semantic // max(len(base_prompts), 1))
            for text in variations:
                record = self._create_record(
                    text=text,
                    operator=SEEOperator.SEMANTIC.value,
                    parent_ids=[c.id for c in candidates[:1]] if candidates else [],
                )
                candidates.append(record)

        # Add seed prompt if provided
        if self.seed_prompt:
            seed_record = self._create_record(
                text=self.seed_prompt,
                operator="seed",
            )
            candidates.append(seed_record)

        # Evaluate all and select top-K
        self._evaluate_population(candidates)
        selected = self._select_top_k(candidates, self.population_size)

        self._phase_scores.append(selected[0].score if selected else 0.0)
        return selected

    def _step(self) -> List[PromptRecord]:
        """Execute one SEE phase (1, 2, or 3)."""
        phase_idx = self.generation  # 1, 2, 3
        if phase_idx == 1:
            return self._phase_feedback()
        elif phase_idx == 2:
            return self._phase_fusion()
        elif phase_idx == 3:
            result = self._phase_semantic()
            raise StopIteration  # Signal completion after phase 3
        else:
            raise StopIteration

    def _phase_feedback(self) -> List[PromptRecord]:
        """Phase 1: Feedback-based improvement.

        For each top candidate, analyze failures and generate improved versions.
        """
        self._current_phase = SEEPhase.FEEDBACK
        logger.info("[SEE Phase 1] Feedback-based improvement")

        candidates = list(self.population)
        samples = self.dataset.get_eval_samples("dev", n=self.eval_sample_size)

        for record in self.population[:self.population_size]:
            # Get failures for this candidate
            result = self.evaluator.evaluate(record.text, samples)
            failures = [
                d for d in result.per_sample_details if not d["correct"]
            ]

            if failures:
                # Generate improved version from failures
                improved_text = self._feedback_improve(record.text, failures)
                if improved_text.strip():
                    new_record = self._create_record(
                        text=improved_text,
                        operator=SEEOperator.FEEDBACK.value,
                        parent_ids=[record.id],
                    )
                    candidates.append(new_record)

        # Evaluate new candidates
        new_candidates = [c for c in candidates if c.score == 0.0]
        baseline = self.best_record.score if self.best_record else 0.0
        self._evaluate_with_racing(new_candidates, baseline)

        # Select top-K
        selected = self._select_top_k(candidates, self.population_size)

        # Check convergence
        self._check_convergence(selected[0].score if selected else 0.0)
        return selected

    def _phase_fusion(self) -> List[PromptRecord]:
        """Phase 2: Fusion via EDA + Crossover.

        Combines best elements from the population using:
        - EDA: Generate from distribution of top prompts
        - Crossover: Pairwise combination of top candidates
        """
        self._current_phase = SEEPhase.FUSION
        logger.info("[SEE Phase 2] Fusion (EDA + Crossover)")

        candidates = list(self.population)
        top_prompts = [r.text for r in self.population[:self.population_size]]

        # EDA: Generate from distribution
        n_eda = max(2, self.population_size // 2)
        for _ in range(n_eda):
            eda_text = self._eda_generate(top_prompts)
            if eda_text.strip():
                record = self._create_record(
                    text=eda_text,
                    operator=SEEOperator.EDA.value,
                    parent_ids=[r.id for r in self.population[:3]],
                )
                candidates.append(record)

        # Crossover: Pairwise combination
        n_crossover = max(2, self.population_size // 2)
        for i in range(min(n_crossover, len(self.population) - 1)):
            parent_a = self.population[i]
            parent_b = self.population[i + 1]
            cross_text = self._crossover(parent_a.text, parent_b.text)
            if cross_text.strip():
                record = self._create_record(
                    text=cross_text,
                    operator=SEEOperator.CROSSOVER.value,
                    parent_ids=[parent_a.id, parent_b.id],
                )
                candidates.append(record)

        # Evaluate and select
        new_candidates = [c for c in candidates if c.score == 0.0]
        baseline = self.best_record.score if self.best_record else 0.0
        self._evaluate_with_racing(new_candidates, baseline)

        selected = self._select_top_k(candidates, self.population_size)
        self._check_convergence(selected[0].score if selected else 0.0)
        return selected

    def _phase_semantic(self) -> List[PromptRecord]:
        """Phase 3: Semantic refinement.

        Generate fine-grained semantic variations of the best candidates
        for final polishing.
        """
        self._current_phase = SEEPhase.SEMANTIC
        logger.info("[SEE Phase 3] Semantic refinement")

        candidates = list(self.population)

        # Generate semantic variations of top candidates
        for record in self.population[:3]:
            variations = self._semantic_variation(record.text, n=2)
            for text in variations:
                new_record = self._create_record(
                    text=text,
                    operator=SEEOperator.SEMANTIC.value,
                    parent_ids=[record.id],
                )
                candidates.append(new_record)

        # Full evaluation (no racing) for final selection
        new_candidates = [c for c in candidates if c.score == 0.0]
        self._evaluate_population(new_candidates)

        selected = self._select_top_k(candidates, self.population_size)
        return selected

    def _check_convergence(self, current_best: float) -> None:
        """Check if optimization has converged (tolerance-based)."""
        if self._phase_scores:
            prev_best = self._phase_scores[-1]
            improvement = current_best - prev_best
            if improvement < self.tolerance_threshold:
                self._stagnation_count += 1
                logger.info(
                    f"[SEE] Stagnation {self._stagnation_count}/{self.max_stagnation} "
                    f"(improvement={improvement:.4f} < threshold={self.tolerance_threshold})"
                )
            else:
                self._stagnation_count = 0

        self._phase_scores.append(current_best)