"""GAAPO — Genetic Algorithm Applied to Prompt Optimization.

Based on Sécheresse et al., arXiv:2504.07157. Unlike classic GAs that rely
only on mutation and crossover, GAAPO integrates multiple specialized prompt
generation strategies within the evolutionary framework. Each generation
runs three phases:

1. Generation — new candidates from five strategies, each operating on
   high-performing prompts from previous generations. Paper weights:
     - Random mutator (40%): one of eight mutation techniques
     - APO / ProTeGi (20%): error analysis → textual gradient → new prompt
     - OPRO (20%): score-ranked trajectory (with dropout) → better prompt
     - Few-shot (10%): append 1-3 labeled examples to a parent
     - Crossover (10%): midpoint split-and-merge of two parents
2. Evaluation — racing (statistical elimination) instead of the paper's
   successive halving; both cut cost by dropping weak candidates early
3. Selection — top-k with elitism

Random-mutator techniques (Sec. generation phase): instruction expansion,
expert persona injection, structural variation, constraint addition,
creative backstory, task decomposition, concise optimization, role assignment.
"""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from pof.core.types import PromptRecord
from pof.optimizers import register_optimizer
from pof.optimizers.base import (
    format_exemplar,
    BaseOptimizer,
    _GENERATE_SYSTEM_PROMPT,
    _IMPROVE_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

# The eight mutation techniques of GAAPO's random mutator
_MUTATION_TECHNIQUES: Dict[str, str] = {
    "instruction_expansion": (
        "Expand this instruction with additional helpful detail or clarification."
    ),
    "expert_persona": (
        "Rewrite this instruction as if written by a domain expert, injecting "
        "expert framing (e.g. 'As an expert in ...')."
    ),
    "structural_variation": (
        "Restructure this instruction: reorder its parts, or convert prose "
        "into steps / steps into prose, keeping the same meaning."
    ),
    "constraint_addition": (
        "Add one useful constraint or rule to this instruction."
    ),
    "creative_backstory": (
        "Add a brief motivating context or scenario to this instruction."
    ),
    "task_decomposition": (
        "Rewrite this instruction as a short sequence of sub-steps to follow."
    ),
    "concise_optimization": (
        "Make this instruction more concise without losing important information."
    ),
    "role_assignment": (
        "Prepend an appropriate role assignment (e.g. 'You are a ...') to this "
        "instruction."
    ),
}


@register_optimizer("gaapo")
class GAAPOOptimizer(BaseOptimizer):
    """GAAPO — Genetic Algorithm Applied to Prompt Optimization.

    Reference: Sécheresse et al., arXiv:2504.07157.
    """

    name = "gaapo"

    def __init__(
        self,
        llm,
        dataset,
        evaluator,
        population_size: int = 8,
        num_iterations: int = 5,
        elitism_count: int = 2,
        offspring_per_gen: int = 10,
        strategy_weights: Optional[Dict[str, float]] = None,
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
        self.elitism_count = elitism_count
        self.offspring_per_gen = offspring_per_gen
        # Paper defaults: mutations 40%, APO 20%, OPRO 20%, few-shot 10%, crossover 10%
        self.strategy_weights = strategy_weights or {
            "mutation": 0.4,
            "apo": 0.2,
            "opro": 0.2,
            "few_shot": 0.1,
            "crossover": 0.1,
        }

    def _init_population(self) -> List[PromptRecord]:
        """Initialize population with diverse generation strategies."""
        candidates: List[PromptRecord] = []
        train_samples = self.dataset.get_few_shot_examples(n=5)

        lamarckian = self._lamarckian_generate(train_samples, n=self.population_size // 2)
        for text in lamarckian:
            candidates.append(self._create_record(text, operator="lamarckian_init"))

        base = self.seed_prompt or (candidates[0].text if candidates else "Solve the task.")
        n_semantic = max(1, self.population_size - len(candidates))
        for text in self._semantic_variation(base, n=n_semantic):
            candidates.append(self._create_record(text, operator="semantic_init"))

        if self.seed_prompt:
            candidates.append(self._create_record(self.seed_prompt, operator="seed"))

        self._evaluate_population(candidates)
        return self._select_top_k(candidates, self.population_size)

    def _step(self) -> List[PromptRecord]:
        """One GAAPO generation: strategy-mix generation → racing → selection."""
        logger.info(f"[GAAPO Gen {self.generation}] Strategy-mix generation")

        sorted_pop = sorted(self.population, key=lambda r: r.score, reverse=True)
        elites = sorted_pop[: self.elitism_count]

        # Allocate offspring counts per strategy from the paper's weights
        allocation = self._allocate_offspring()
        offspring: List[PromptRecord] = []
        strategies = {
            "mutation": self._gen_mutation,
            "apo": self._gen_apo,
            "opro": self._gen_opro,
            "few_shot": self._gen_few_shot,
            "crossover": self._gen_crossover,
        }
        for name, count in allocation.items():
            for _ in range(count):
                child = strategies[name]()
                if child is not None:
                    offspring.append(child)

        # Racing evaluation (statistical early elimination, cf. successive halving)
        baseline = self.best_record.score if self.best_record else 0.0
        self._evaluate_with_racing(offspring, baseline)

        return self._select_top_k(elites + offspring + sorted_pop, self.population_size)

    # --- Generation strategies ---

    def _allocate_offspring(self) -> Dict[str, int]:
        """Distribute offspring_per_gen across strategies by weight."""
        allocation = {
            name: int(self.offspring_per_gen * w)
            for name, w in self.strategy_weights.items()
        }
        # Remaining slots go to the highest-weighted strategy (paper behavior)
        remaining = self.offspring_per_gen - sum(allocation.values())
        if remaining > 0:
            top = max(self.strategy_weights, key=self.strategy_weights.get)
            allocation[top] += remaining
        return allocation

    def _pick_parent(self, top_k: int = 4) -> Optional[PromptRecord]:
        pool = sorted(self.population, key=lambda r: r.score, reverse=True)[:top_k]
        return random.choice(pool) if pool else None

    def _gen_mutation(self) -> Optional[PromptRecord]:
        """Random mutator: one of the eight techniques on a top parent."""
        parent = self._pick_parent()
        if parent is None:
            return None
        technique, instruction = random.choice(list(_MUTATION_TECHNIQUES.items()))
        meta_prompt = (
            f"{instruction}\n\n"
            f"Instruction:\n{parent.text}\n\n"
            "Modified instruction:"
        )
        text = self._generate_prompt(
            meta_prompt, temperature=0.9, system_prompt=_GENERATE_SYSTEM_PROMPT
        )
        if not text.strip():
            return None
        return self._create_record(
            text.strip(), operator=f"mutator_{technique}", parent_ids=[parent.id]
        )

    def _gen_apo(self) -> Optional[PromptRecord]:
        """APO/ProTeGi: error analysis → textual gradient → improved prompt."""
        parent = self._pick_parent(top_k=3)
        if parent is None:
            return None

        details = parent.per_sample_details
        if not details:
            samples = self.dataset.get_eval_samples("dev", n=self.eval_sample_size)
            result = self.evaluator.evaluate(parent.text, samples)
            details = result.per_sample_details
            parent.per_sample_details = details

        failures = [d for d in details if not d["correct"]]
        if not failures:
            return None

        text = self._feedback_improve(parent.text, failures)
        if not text.strip():
            return None
        return self._create_record(
            text.strip(), operator="apo_gradient", parent_ids=[parent.id]
        )

    def _gen_opro(self) -> Optional[PromptRecord]:
        """OPRO: score-ranked trajectory with stochastic dropout → new prompt."""
        ranked = sorted(self.population, key=lambda r: r.score)
        # Stochastic dropout on the trajectory (keep each entry with p=0.8)
        trajectory = [r for r in ranked if random.random() < 0.8] or ranked
        context = "\n".join(
            f"Score: {r.score:.3f} | {r.text[:100]}" for r in trajectory
        )
        meta_prompt = (
            "Below are instructions with their performance scores, sorted "
            "ascending. Write a NEW instruction that would achieve a higher "
            "score than all of them.\n\n"
            f"{context}\n\n"
            "New instruction:"
        )
        text = self._generate_prompt(
            meta_prompt, temperature=0.8, system_prompt=_GENERATE_SYSTEM_PROMPT
        )
        if not text.strip():
            return None
        return self._create_record(
            text.strip(), operator="opro_trajectory",
            parent_ids=[r.id for r in trajectory[-2:]],
        )

    def _gen_few_shot(self) -> Optional[PromptRecord]:
        """Few-shot: append 1-3 random labeled examples to a top parent."""
        parent = self._pick_parent()
        if parent is None:
            return None
        train = self.dataset.get_few_shot_examples(n=8, seed=random.randint(0, 10**6))
        if not train:
            return None
        k = random.randint(1, min(3, len(train)))
        shots = random.sample(train, k)
        examples = "\n\n".join(
            format_exemplar(self.evaluator, s) for s in shots
        )
        text = f"{parent.text}\n\nExamples:\n{examples}"
        return self._create_record(
            text, operator="few_shot_augment", parent_ids=[parent.id],
            num_few_shots=k,
        )

    def _gen_crossover(self) -> Optional[PromptRecord]:
        """Crossover: midpoint split-and-merge of two top parents (paper Sec. 3)."""
        pool = sorted(self.population, key=lambda r: r.score, reverse=True)[:4]
        if len(pool) < 2:
            return None
        parent_a, parent_b = random.sample(pool, 2)

        # Split each prompt near its midpoint at a sentence boundary
        half_a = self._first_half(parent_a.text)
        half_b = self._second_half(parent_b.text)
        text = f"{half_a} {half_b}".strip()
        if not text:
            return None
        return self._create_record(
            text, operator="midpoint_crossover",
            parent_ids=[parent_a.id, parent_b.id],
        )

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        import re
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p for p in parts if p]

    def _first_half(self, text: str) -> str:
        sents = self._split_sentences(text)
        return " ".join(sents[: max(1, len(sents) // 2)])

    def _second_half(self, text: str) -> str:
        sents = self._split_sentences(text)
        return " ".join(sents[len(sents) // 2:]) if len(sents) > 1 else text
