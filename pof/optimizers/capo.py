"""CAPO — Cost-Aware Prompt Optimization.

Based on Zehle et al., arXiv:2504.16005. An evolutionary algorithm with
LLM operators that jointly optimizes instructions AND few-shot examples:

1. Genome = (instruction, few-shot example set) — both evolve together
2. Crossover: LLM combines two parent instructions; child inherits a mix
   of the parents' few-shot examples
3. Mutation: LLM rewrites the instruction; few-shot set is mutated by
   adding/removing/swapping examples
4. Racing: candidates are eliminated early when statistically inferior
   (Hoeffding bound), saving evaluation budget
5. Length penalty: fitness = accuracy − gamma × (prompt length ratio),
   pushing toward shorter prompts at equal accuracy

Reference implementation: github.com/finitearth/capo
"""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from pof.core.types import PromptRecord
from pof.optimizers import register_optimizer
from pof.optimizers.base import format_exemplar, BaseOptimizer

logger = logging.getLogger(__name__)


@register_optimizer("capo")
class CAPOOptimizer(BaseOptimizer):
    """CAPO — Cost-Aware Prompt Optimization.

    Reference: Zehle et al., arXiv:2504.16005.
    """

    name = "capo"
    tier = "baseline"

    def __init__(
        self,
        llm,
        dataset,
        evaluator,
        population_size: int = 5,
        num_iterations: int = 5,
        max_few_shots: int = 3,
        length_penalty: float = 0.05,
        crossovers_per_iter: int = 4,
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
        self.max_few_shots = max_few_shots
        self.length_penalty = length_penalty
        self.crossovers_per_iter = crossovers_per_iter
        # record_id -> (instruction, few_shot_examples)
        self._genomes: Dict[str, Tuple[str, List[Dict[str, str]]]] = {}
        # Reference length for the penalty term (longest prompt seen so far)
        self._max_length: int = 1

    def _init_population(self) -> List[PromptRecord]:
        """Initialize with instruction variants × random few-shot sets."""
        train_samples = self.dataset.get_few_shot_examples(n=10)

        instructions: List[str] = []
        if self.seed_prompt:
            instructions.append(self.seed_prompt)
        instructions.extend(
            self._semantic_variation(
                self.seed_prompt or "Solve the task.", n=self.population_size - 1
            )
        )
        # Lamarckian backup if paraphrasing under-delivered
        while len(instructions) < self.population_size:
            extra = self._lamarckian_generate(train_samples, n=1)
            if not extra:
                break
            instructions.extend(extra)

        candidates: List[PromptRecord] = []
        for instr in instructions[: self.population_size + 2]:
            few_shots = self._sample_few_shots(train_samples)
            candidates.append(self._make_candidate(instr, few_shots, "init"))

        self._evaluate_penalized(candidates)
        return self._select_top_k(candidates)

    def _step(self) -> List[PromptRecord]:
        """One CAPO generation: crossover → mutation → racing → selection."""
        logger.info(f"[CAPO Gen {self.generation}] Evolutionary step")
        train_samples = self.dataset.get_few_shot_examples(n=10)
        offspring: List[PromptRecord] = []

        # Crossover phase
        for _ in range(self.crossovers_per_iter):
            if len(self.population) < 2:
                break
            parent_a, parent_b = random.sample(
                self.population[: self.population_size], 2
            )
            instr_a, shots_a = self._genome_of(parent_a)
            instr_b, shots_b = self._genome_of(parent_b)

            child_instr = self._crossover(instr_a, instr_b)
            if not child_instr.strip():
                continue
            # Few-shots: sample from the union of both parents' example sets
            pool = shots_a + shots_b
            k = min(len(pool), random.randint(0, self.max_few_shots))
            child_shots = random.sample(pool, k) if k else []
            child = self._make_candidate(
                child_instr.strip(), child_shots, "capo_crossover",
                parent_ids=[parent_a.id, parent_b.id],
            )
            offspring.append(child)

        # Mutation phase: each crossover child is mutated
        mutated: List[PromptRecord] = []
        for child in offspring:
            instr, shots = self._genome_of(child)
            new_instr = self._semantic_variation(instr, n=1)
            new_shots = self._mutate_few_shots(shots, train_samples)
            if new_instr:
                mutant = self._make_candidate(
                    new_instr[0], new_shots, "capo_mutation",
                    parent_ids=[child.id],
                )
                mutated.append(mutant)

        # Racing evaluation against current best (penalized) fitness
        candidates = offspring + mutated
        baseline = self.best_record.score if self.best_record else 0.0
        self._evaluate_with_racing(candidates, baseline)
        self._apply_length_penalty(candidates)

        survivors = self._select_top_k(list(self.population) + candidates)
        return survivors

    # --- CAPO internals ---

    def _make_candidate(
        self,
        instruction: str,
        few_shots: List[Dict[str, str]],
        operator: str,
        parent_ids: Optional[List[str]] = None,
    ) -> PromptRecord:
        """Build a PromptRecord whose text = instruction + few-shot block."""
        text = self._render_prompt(instruction, few_shots)
        record = self._create_record(
            text, operator=operator, parent_ids=parent_ids,
            num_few_shots=len(few_shots),
        )
        self._genomes[record.id] = (instruction, few_shots)
        self._max_length = max(self._max_length, len(text))
        return record

    def _genome_of(self, record: PromptRecord) -> Tuple[str, List[Dict[str, str]]]:
        """Get (instruction, few_shots) for a record, falling back to raw text."""
        return self._genomes.get(record.id, (record.text, []))

    def _render_prompt(self, instruction: str, few_shots: List[Dict[str, str]]) -> str:
        # No longer a staticmethod: exemplars are now rendered in the output
        # format the evaluator expects, which requires access to the evaluator.
        if not few_shots:
            return instruction
        examples = "\n\n".join(
            format_exemplar(self.evaluator, s) for s in few_shots
        )
        return f"{instruction}\n\nExamples:\n{examples}"

    def _sample_few_shots(
        self, train_samples: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        k = random.randint(0, min(self.max_few_shots, len(train_samples)))
        return random.sample(train_samples, k) if k else []

    def _mutate_few_shots(
        self,
        shots: List[Dict[str, str]],
        train_samples: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Add, remove, or swap one example."""
        shots = list(shots)
        op = random.choice(["add", "remove", "swap"])
        unused = [s for s in train_samples if s not in shots]

        if op == "add" and unused and len(shots) < self.max_few_shots:
            shots.append(random.choice(unused))
        elif op == "remove" and shots:
            shots.remove(random.choice(shots))
        elif op == "swap" and shots and unused:
            shots.remove(random.choice(shots))
            shots.append(random.choice(unused))
        return shots

    def _evaluate_penalized(self, candidates: List[PromptRecord]) -> None:
        """Full evaluation followed by the length penalty."""
        self._evaluate_population(candidates)
        self._apply_length_penalty(candidates)

    def _apply_length_penalty(self, candidates: List[PromptRecord]) -> None:
        """fitness = accuracy − gamma × (len / max_len). Raw accuracy kept in scores."""
        for record in candidates:
            if not record.text:
                continue
            record.scores["accuracy"] = record.scores.get("dev", record.score)
            penalty = self.length_penalty * (len(record.text) / self._max_length)
            record.score = max(0.0, record.scores["accuracy"] - penalty)
