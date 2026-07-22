"""APEX — Adaptive Prompt Evolution with eXpert feedback.

A proposed method that combines:
1. Expert-role prompting for candidate generation
2. Adaptive operator selection via UCB1 bandit (cf. ProTeGi's bandit
   selection, Pryzant et al. 2023) — balances exploiting operators that
   produced good candidates with exploring under-sampled ones
3. Joint instruction + few-shot search (cf. CAPO/SEE: exemplars in the
   search space are the strongest lever on BBH-style tasks)
4. Format-constraint operator: appends an explicit output-format rule
   derived from the targets (format drift, not reasoning, costs strict
   extraction points on small models — Sclar et al. 2023)
5. GEPA-style minibatch gate (Agrawal et al. 2025): fresh random dev
   minibatch screens candidates; only survivors get the full dev eval,
   so selection never overfits one fixed subset
6. Tournament selection with elitism; hash dedup so paraphrase/crossover
   never burn evaluations on identical text

Key differentiator: uses "expert personas" for diverse initialization,
then adaptively reallocates generation budget to whichever operator is
actually producing improvements on this task.
"""
from __future__ import annotations

import logging
import math
import random
from typing import Any, Dict, List, Optional

from pof.core.types import PromptRecord
from pof.optimizers import register_optimizer
from pof.optimizers.base import BaseOptimizer, _GENERATE_SYSTEM_PROMPT, _IMPROVE_SYSTEM_PROMPT

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
        population_size: int = 8,
        num_iterations: int = 5,
        expert_personas: Optional[List[str]] = None,
        candidates_per_operator: int = 2,
        ucb_c: float = 0.5,
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
        self.candidates_per_operator = candidates_per_operator
        self.ucb_c = ucb_c
        self._operator_scores: Dict[str, List[float]] = {}

    def _init_population(self) -> List[PromptRecord]:
        """Initialize with expert-persona-generated candidates."""
        candidates: List[PromptRecord] = []
        train_samples = self.dataset.get_few_shot_examples(n=5)

        # Generate one candidate per expert persona
        for persona in self.expert_personas:
            text = self._expert_generate(persona, train_samples)
            if text and not self._is_duplicate(text):
                candidates.append(self._create_record(
                    text, operator=f"expert_{persona[:20]}",
                    metadata={"persona": persona}
                ))

        # Lamarckian baseline
        lamarckian = self._lamarckian_generate(train_samples, n=2)
        for text in lamarckian:
            if not self._is_duplicate(text):
                candidates.append(self._create_record(text, operator="lamarckian_init"))

        # Seed prompt
        if self.seed_prompt:
            candidates.append(self._create_record(self.seed_prompt, operator="seed"))

        self._evaluate_population(candidates)
        return self._select_top_k(candidates)

    def _step(self) -> List[PromptRecord]:
        """Adaptive step: UCB1-select operators and apply them."""
        logger.info(f"[APEX Gen {self.generation}] Adaptive evolution step")
        candidates = list(self.population)

        operators = self._select_operators()

        for op_name, op_fn in operators:
            for _ in range(self.candidates_per_operator):
                new_texts = op_fn()
                for text in new_texts:
                    text = (text or "").strip()
                    if text and not self._is_duplicate(text):
                        record = self._create_record(
                            text, operator=op_name,
                            parent_ids=[r.id for r in self.population[:2]]
                        )
                        candidates.append(record)

        # Evaluate new candidates: GEPA-style minibatch gate — fresh random
        # minibatch filters, survivors get the full dev evaluation.
        new_candidates = [c for c in candidates if c.score == 0.0]
        baseline = self.best_record.score if self.best_record else 0.0
        self._evaluate_with_minibatch_gate(new_candidates, baseline)

        # Track operator performance for the bandit (minibatch score for
        # rejected candidates still informs credit assignment).
        for record in new_candidates:
            self._operator_scores.setdefault(record.operator, []).append(record.score)

        # Tournament selection with elitism
        selected = self._tournament_select(candidates)
        return selected

    def _select_operators(self) -> List[tuple]:
        """UCB1 bandit over operators (cf. ProTeGi's bandit selection).

        value = mean(scores) + c * sqrt(ln(total_pulls) / pulls_i).
        Unpulled operators have infinite value, so every operator is tried
        before any is repeated — no premature greedy lock-in.
        """
        all_operators = [
            ("expert_refine", self._op_expert_refine),
            ("failure_guided", self._op_failure_guided),
            ("crossover", self._op_crossover),
            ("trajectory", self._op_trajectory),
            ("semantic_var", self._op_semantic_variation),
            ("few_shot", self._op_few_shot),
            ("format_constraint", self._op_format_constraint),
        ]

        total_pulls = sum(len(v) for v in self._operator_scores.values())
        if total_pulls == 0:
            return all_operators

        scored = []
        for name, fn in all_operators:
            pulls = self._operator_scores.get(name, [])
            if not pulls:
                ucb = float("inf")
            else:
                mean = sum(pulls) / len(pulls)
                ucb = mean + self.ucb_c * math.sqrt(
                    math.log(max(total_pulls, 2)) / len(pulls)
                )
            scored.append((ucb, name, fn))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [(name, fn) for _, name, fn in scored[:4]]

    def _tournament_select(
        self, candidates: List[PromptRecord], tournament_size: int = 3
    ) -> List[PromptRecord]:
        """Tournament selection with elitism."""
        sorted_candidates = sorted(candidates, key=lambda r: r.score, reverse=True)
        selected = [sorted_candidates[0]]

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
        """Refine a random elite using a random expert persona."""
        persona = random.choice(self.expert_personas)
        record = random.choice(self.population[:3]) if self.population else None
        if not record:
            return []

        meta_prompt = (
            f"You are {persona}. Improve this instruction to make it more effective. "
            f"Maintain the core intent but enhance clarity and precision.\n\n"
            f"Original instruction:\n{record.text}\n\n"
            f"Improved instruction:"
        )
        result = self._generate_prompt(meta_prompt, temperature=0.7, system_prompt=_IMPROVE_SYSTEM_PROMPT)
        return [result] if result.strip() else []

    def _op_failure_guided(self) -> List[str]:
        """Failure-guided improvement of a random elite."""
        record = random.choice(self.population[:3]) if self.population else None
        if not record:
            return []

        details = record.per_sample_details
        if not details and record.text:
            samples = self.dataset.get_eval_samples("dev", n=self.eval_sample_size)
            result = self.evaluator.evaluate(record.text, samples)
            details = result.per_sample_details
            record.per_sample_details = details
            record.score = result.score
            record.performance_vector = result.performance_vector

        failures = [d for d in details if not d["correct"]]
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
        """OPRO-style trajectory generation.

        Full instructions (not truncated fragments) plus task exemplars in
        the meta-prompt, per Yang et al. 2023's ablations.
        """
        context = "\n".join(
            f"Score: {r.score:.3f}\nInstruction: {r.text}\n"
            for r in sorted(self.population, key=lambda r: r.score)
        )
        exemplars = self.dataset.get_few_shot_examples(n=2)
        exemplar_text = "\n".join(
            f"Input: {s['input'][:150]}\nOutput: {s['target']}" for s in exemplars
        )
        meta_prompt = (
            "Below are instructions for a task, sorted by performance score "
            "(ascending). Task examples:\n"
            f"{exemplar_text}\n\n"
            f"Instructions and scores:\n{context}\n"
            "Write a NEW instruction that would score higher than all of the above.\n\n"
            "New higher-scoring instruction:"
        )
        result = self._generate_prompt(meta_prompt, temperature=0.8, system_prompt=_GENERATE_SYSTEM_PROMPT)
        return [result] if result.strip() else []

    def _op_semantic_variation(self) -> List[str]:
        """Semantic variation of the best prompt."""
        best = self.population[0] if self.population else None
        if not best:
            return []
        return self._semantic_variation(best.text, n=1)

    def _op_few_shot(self) -> List[str]:
        """Append 1-3 labeled exemplars to an elite (joint instruction+ICL
        search, cf. CAPO/SEE)."""
        record = random.choice(self.population[:3]) if self.population else None
        if not record:
            return []
        train = self.dataset.get_few_shot_examples(n=8, seed=random.randint(0, 10**6))
        if not train:
            return []
        k = random.randint(1, min(3, len(train)))
        shots = random.sample(train, k)
        examples = "\n\n".join(
            f"Input: {s['input']}\nOutput: {s['target']}" for s in shots
        )
        return [f"{record.text}\n\nExamples:\n{examples}"]

    def _op_format_constraint(self) -> List[str]:
        """Append an explicit output-format rule derived from the targets.

        Strict extraction penalizes format drift more than reasoning errors
        on small models (format sensitivity, Sclar et al. 2023).
        """
        record = random.choice(self.population[:3]) if self.population else None
        if not record:
            return []
        targets = [s["target"] for s in self.dataset.get_few_shot_examples(n=4)]
        sample_answers = ", ".join(repr(t)[:30] for t in targets[:3])
        constraint = (
            f"\n\nAnswer with ONLY the final answer, exactly in the same format "
            f"as these examples: {sample_answers}. No explanation."
        )
        return [record.text.rstrip() + constraint]

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
        result = self._generate_prompt(meta_prompt, temperature=0.8, system_prompt=_GENERATE_SYSTEM_PROMPT)
        return result.strip() if result.strip() else None
