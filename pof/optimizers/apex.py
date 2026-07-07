"""APEX — Adaptive Prompt Evolution with eXpert feedback.

A proposed method that combines:
1. Expert-role prompting for candidate generation
2. Multi-criteria evaluation (accuracy + clarity + specificity)
3. Adaptive operator selection based on phase performance
4. Tournament selection with elitism

Key differentiator: Uses "expert personas" to generate diverse candidates,
then adaptively selects which generation strategy works best.
"""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from pof.core.types import PromptRecord
from pof.optimizers import register_optimizer
from pof.optimizers.base import BaseOptimizer

logger = logging.getLogger(__name__)


@register_optimizer("apex")
class APEXOptimizer(BaseOptimizer):
    """APEX optimizer — adaptive expert-guided prompt evolution.

    Proposed method: needs validation.
    """

    name = "apex"

    def __init__(
        self,
        llm,
        dataset,
        evaluator,
        population_size: int = 5,
        num_iterations: int = 4,
        expert_personas: Optional[List[str]] = None,
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
        self.expert_personas = expert_personas or [
            "a concise technical writer",
            "a patient teacher explaining to a student",
            "a rigorous logician focused on precision",
            "a creative problem solver",
        ]
        self._operator_scores: Dict[str, List[float]] = {}

    def _init_population(self) -> List[PromptRecord]:
        """Initialize with expert-persona-generated candidates."""
        candidates: List[PromptRecord] = []
        train_samples = self.dataset.get_few_shot_examples(n=5)

        # Generate one candidate per expert persona
        for persona in self.expert_personas:
            text = self._expert_generate(persona, train_samples)
            if text:
                candidates.append(self._create_record(
                    text, operator=f"expert_{persona[:20]}",
                    metadata={"persona": persona}
                ))

        # Lamarckian baseline
        lamarckian = self._lamarckian_generate(train_samples, n=2)
        for text in lamarckian:
            candidates.append(self._create_record(text, operator="lamarckian_init"))

        # Seed prompt
        if self.seed_prompt:
            candidates.append(self._create_record(self.seed_prompt, operator="seed"))

        self._evaluate_population(candidates)
        return self._select_top_k(candidates)

    def _step(self) -> List[PromptRecord]:
        """Adaptive step: select best-performing operators and apply them."""
        logger.info(f"[APEX Gen {self.generation}] Adaptive evolution step")
        candidates = list(self.population)

        # Select operators adaptively based on past performance
        operators = self._select_operators()

        for op_name, op_fn in operators:
            new_texts = op_fn()
            for text in new_texts:
                if text and text.strip():
                    record = self._create_record(
                        text.strip(), operator=op_name,
                        parent_ids=[r.id for r in self.population[:2]]
                    )
                    candidates.append(record)

        # Evaluate new candidates
        new_candidates = [c for c in candidates if c.score == 0.0]
        baseline = self.best_record.score if self.best_record else 0.0
        self._evaluate_with_racing(new_candidates, baseline)

        # Track operator performance
        for record in new_candidates:
            op = record.operator
            if op not in self._operator_scores:
                self._operator_scores[op] = []
            self._operator_scores[op].append(record.score)

        # Tournament selection with elitism
        selected = self._tournament_select(candidates)
        return selected

    def _select_operators(self) -> List[tuple]:
        """Adaptively select operators based on historical performance."""
        all_operators = [
            ("expert_refine", self._op_expert_refine),
            ("failure_guided", self._op_failure_guided),
            ("crossover", self._op_crossover),
            ("trajectory", self._op_trajectory),
            ("semantic_var", self._op_semantic_variation),
        ]

        if not self._operator_scores:
            # First iteration: use all operators
            return all_operators

        # Score each operator by average performance
        scored = []
        for name, fn in all_operators:
            scores = self._operator_scores.get(name, [])
            avg = sum(scores) / len(scores) if scores else 0.5
            scored.append((avg, name, fn))

        # Select top operators + one random for exploration
        scored.sort(reverse=True)
        selected = [(name, fn) for _, name, fn in scored[:3]]

        # Add one random operator for exploration
        remaining = [(name, fn) for _, name, fn in scored[3:]]
        if remaining:
            selected.append(random.choice(remaining))

        return selected

    def _tournament_select(
        self, candidates: List[PromptRecord], tournament_size: int = 3
    ) -> List[PromptRecord]:
        """Tournament selection with elitism."""
        # Always keep the best (elitism)
        sorted_candidates = sorted(candidates, key=lambda r: r.score, reverse=True)
        selected = [sorted_candidates[0]]

        # Tournament for remaining slots
        remaining = sorted_candidates[1:]
        while len(selected) < self.population_size and remaining:
            tournament = random.sample(
                remaining, min(tournament_size, len(remaining))
            )
            winner = max(tournament, key=lambda r: r.score)
            selected.append(winner)
            remaining.remove(winner)

        return selected

    # --- APEX operators ---

    def _op_expert_refine(self) -> List[str]:
        """Refine best prompt using a random expert persona."""
        persona = random.choice(self.expert_personas)
        best = self.population[0] if self.population else None
        if not best:
            return []

        meta_prompt = (
            f"You are {persona}. Improve this instruction to make it more effective. "
            f"Maintain the core intent but enhance clarity and precision.\n\n"
            f"Original instruction:\n{best.text}\n\n"
            f"Improved instruction:"
        )
        result = self._generate_prompt(meta_prompt, temperature=0.7)
        return [result] if result.strip() else []

    def _op_failure_guided(self) -> List[str]:
        """Failure-guided improvement of a random elite."""
        record = random.choice(self.population[:3]) if self.population else None
        if not record:
            return []

        samples = self.dataset.get_eval_samples("dev", n=20)
        result = self.evaluator.evaluate(record.text, samples)
        failures = [d for d in result.per_sample_details if not d["correct"]]

        if not failures:
            return []

        improved = self._feedback_improve(record.text, failures)
        return [improved] if improved and improved.strip() else []

    def _op_crossover(self) -> List[str]:
        """Crossover two random elites."""
        if len(self.population) < 2:
            return []
        a, b = random.sample(self.population[:4], 2)
        result = self._crossover(a.text, b.text)
        return [result] if result.strip() else []

    def _op_trajectory(self) -> List[str]:
        """OPRO-style trajectory generation."""
        context = "\n".join(
            f"Score: {r.score:.3f} | {r.text[:80]}"
            for r in sorted(self.population, key=lambda r: r.score)
        )
        meta_prompt = (
            "Below are instructions sorted by performance (ascending). "
            "Generate a new instruction that would score higher than all.\n\n"
            f"{context}\n\n"
            "New higher-scoring instruction:"
        )
        result = self._generate_prompt(meta_prompt, temperature=0.8)
        return [result] if result.strip() else []

    def _op_semantic_variation(self) -> List[str]:
        """Semantic variation of the best prompt."""
        best = self.population[0] if self.population else None
        if not best:
            return []
        return self._semantic_variation(best.text, n=1)

    def _expert_generate(
        self, persona: str, samples: List[Dict[str, str]]
    ) -> Optional[str]:
        """Generate a prompt using an expert persona."""
        examples_text = "\n".join(
            f"Input: {s['input'][:80]}\nOutput: {s['target']}"
            for s in samples[:3]
        )
        meta_prompt = (
            f"You are {persona}. Given these input-output examples, write a clear "
            f"instruction that would guide someone to produce the correct outputs.\n\n"
            f"Examples:\n{examples_text}\n\n"
            f"Instruction:"
        )
        result = self._generate_prompt(meta_prompt, temperature=0.8)
        return result.strip() if result.strip() else None