"""GAAPO — Genetic Algorithm for Automatic Prompt Optimization.

Based on Sécheresse et al., 2025. A genetic algorithm approach that uses:
1. Standard GA operators (selection, crossover, mutation)
2. LLM-based crossover and mutation (not random string ops)
3. Fitness-proportionate selection
4. Generational replacement with elitism

Key features:
- Population-based evolutionary approach
- Multiple crossover strategies (uniform, single-point via LLM)
- Mutation via paraphrasing and local edits
- Elitism to preserve best candidates
"""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from pof.core.types import PromptRecord
from pof.optimizers import register_optimizer
from pof.optimizers.base import BaseOptimizer

logger = logging.getLogger(__name__)


@register_optimizer("gaapo")
class GAAPOOptimizer(BaseOptimizer):
    """GAAPO — Genetic Algorithm for Automatic Prompt Optimization.

    Reference: Sécheresse et al., 2025.
    """

    name = "gaapo"

    def __init__(
        self,
        llm,
        dataset,
        evaluator,
        population_size: int = 8,
        num_iterations: int = 5,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.3,
        elitism_count: int = 2,
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
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count

    def _init_population(self) -> List[PromptRecord]:
        """Initialize population with diverse generation strategies."""
        candidates: List[PromptRecord] = []
        train_samples = self.dataset.get_few_shot_examples(n=5)

        # Lamarckian seeds
        lamarckian = self._lamarckian_generate(train_samples, n=self.population_size // 2)
        for text in lamarckian:
            candidates.append(self._create_record(text, operator="lamarckian_init"))

        # Semantic variations
        base = self.seed_prompt or (candidates[0].text if candidates else "Solve the task.")
        n_semantic = self.population_size - len(candidates)
        variations = self._semantic_variation(base, n=n_semantic)
        for text in variations:
            candidates.append(self._create_record(text, operator="semantic_init"))

        # Add seed if provided
        if self.seed_prompt:
            candidates.append(self._create_record(self.seed_prompt, operator="seed"))

        self._evaluate_population(candidates)
        return self._select_top_k(candidates, self.population_size)

    def _step(self) -> List[PromptRecord]:
        """One GA generation: selection → crossover → mutation → evaluation."""
        logger.info(f"[GAAPO Gen {self.generation}] GA evolution step")

        # Elitism: preserve top candidates
        sorted_pop = sorted(self.population, key=lambda r: r.score, reverse=True)
        elites = sorted_pop[:self.elitism_count]
        offspring: List[PromptRecord] = list(elites)

        # Generate offspring until population is full
        while len(offspring) < self.population_size:
            # Selection (tournament)
            parent_a = self._tournament_select()
            parent_b = self._tournament_select()

            # Crossover
            if random.random() < self.crossover_rate:
                child_text = self._crossover(parent_a.text, parent_b.text)
                child = self._create_record(
                    child_text, operator="ga_crossover",
                    parent_ids=[parent_a.id, parent_b.id]
                )
            else:
                # Copy better parent
                better = parent_a if parent_a.score >= parent_b.score else parent_b
                child = self._create_record(
                    better.text, operator="ga_copy",
                    parent_ids=[better.id]
                )

            # Mutation
            if random.random() < self.mutation_rate:
                mutated_text = self._mutate(child.text)
                if mutated_text and mutated_text.strip():
                    child = self._create_record(
                        mutated_text, operator="ga_mutation",
                        parent_ids=[child.id]
                    )

            offspring.append(child)

        # Evaluate new offspring (skip elites already evaluated)
        new_candidates = [c for c in offspring if c.score == 0.0]
        baseline = self.best_record.score if self.best_record else 0.0
        self._evaluate_with_racing(new_candidates, baseline)

        return self._select_top_k(offspring, self.population_size)

    def _tournament_select(self, tournament_size: int = 3) -> PromptRecord:
        """Tournament selection from current population."""
        tournament = random.sample(
            self.population, min(tournament_size, len(self.population))
        )
        return max(tournament, key=lambda r: r.score)

    def _mutate(self, prompt: str) -> Optional[str]:
        """LLM-based mutation: paraphrase or local edit."""
        mutation_type = random.choice(["paraphrase", "local_edit", "expand", "compress"])

        if mutation_type == "paraphrase":
            results = self._semantic_variation(prompt, n=1)
            return results[0] if results else None
        elif mutation_type == "local_edit":
            meta_prompt = (
                "Make a small random change to this instruction. "
                "Modify one aspect: add a constraint, change wording, or adjust emphasis.\n\n"
                f"Instruction:\n{prompt}\n\n"
                "Modified instruction:"
            )
            return self._generate_prompt(meta_prompt, temperature=0.9)
        elif mutation_type == "expand":
            meta_prompt = (
                "Expand this instruction by adding one helpful detail or clarification.\n\n"
                f"Instruction:\n{prompt}\n\n"
                "Expanded instruction:"
            )
            return self._generate_prompt(meta_prompt, temperature=0.7)
        else:  # compress
            meta_prompt = (
                "Make this instruction more concise without losing important information.\n\n"
                f"Instruction:\n{prompt}\n\n"
                "Concise instruction:"
            )
            return self._generate_prompt(meta_prompt, temperature=0.5)