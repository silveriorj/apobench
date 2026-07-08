"""GEPA — Genetic-Pareto prompt optimizer.

Based on Agrawal et al., arXiv:2507.19457 ("GEPA: Reflective Prompt Evolution
Can Outperform Reinforcement Learning"). Core algorithm:

1. Maintain a candidate pool with per-instance scores on a fixed dev set
2. Select the next candidate to evolve via Pareto-based sampling:
   candidates that achieve the best score on at least one instance form the
   Pareto front; sample proportional to how many instances each one wins
3. Reflective mutation: run the candidate on a minibatch, collect failure
   traces, and have the LLM reflect on them in natural language to diagnose
   problems and propose an improved prompt
4. Accept the child only if it improves over its parent on the minibatch;
   accepted children are evaluated on the full dev set and join the pool

Single-prompt adaptation: the paper targets compound AI systems with multiple
modules; here the system has one instruction module, so system-aware merge
(crossover across modules) is omitted and reflective mutation is the sole
variation operator, per the paper's single-module ablation.
"""
from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional

from pof.core.types import PromptRecord
from pof.optimizers import register_optimizer
from pof.optimizers.base import BaseOptimizer, _IMPROVE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@register_optimizer("gepa")
class GEPAOptimizer(BaseOptimizer):
    """GEPA — Genetic-Pareto reflective prompt evolution.

    Reference: Agrawal et al., arXiv:2507.19457.
    """

    name = "gepa"

    def __init__(
        self,
        llm,
        dataset,
        evaluator,
        population_size: int = 5,
        num_iterations: int = 6,
        minibatch_size: int = 8,
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
        self.minibatch_size = minibatch_size
        # Fixed dev set so performance vectors are comparable across candidates
        self._pareto_samples: List[Dict[str, str]] = []
        # Candidate pool (grows over time; population is a top-k view of it)
        self._pool: List[PromptRecord] = []

    def _init_population(self) -> List[PromptRecord]:
        """Initialize the pool with the seed prompt (plus Lamarckian backup)."""
        self._pareto_samples = self.dataset.get_eval_samples(
            "dev", n=self.eval_sample_size
        )

        candidates: List[PromptRecord] = []
        if self.seed_prompt:
            candidates.append(self._create_record(self.seed_prompt, operator="seed"))
        else:
            # No seed: bootstrap one candidate from I/O examples
            train_samples = self.dataset.get_few_shot_examples(n=5)
            for text in self._lamarckian_generate(train_samples, n=1):
                candidates.append(self._create_record(text, operator="lamarckian_init"))

        for record in candidates:
            self._eval_on_pareto_set(record)

        self._pool = list(candidates)
        return self._select_top_k(self._pool)

    def _step(self) -> List[PromptRecord]:
        """One GEPA iteration: Pareto-select → reflect → mutate → validate."""
        logger.info(f"[GEPA Gen {self.generation}] Reflective mutation round")

        parent = self._pareto_select()
        if parent is None:
            raise StopIteration

        # Sample a minibatch and collect traces for reflection
        minibatch = random.sample(
            self._pareto_samples, min(self.minibatch_size, len(self._pareto_samples))
        )
        parent_result = self.evaluator.evaluate(parent.text, minibatch)
        traces = parent_result.per_sample_details

        # Reflective mutation from natural-language diagnosis of the traces
        child_text = self._reflective_mutate(parent.text, traces)
        if not child_text or not child_text.strip():
            logger.debug("[GEPA] mutation produced empty prompt, skipping round")
            return self._select_top_k(self._pool)

        # Gate: child must beat parent on the same minibatch
        child_result = self.evaluator.evaluate(child_text.strip(), minibatch)
        if child_result.score <= parent_result.score:
            logger.info(
                f"[GEPA] child rejected on minibatch "
                f"({child_result.score:.3f} <= {parent_result.score:.3f})"
            )
            return self._select_top_k(self._pool)

        # Accepted: evaluate on the full Pareto set and add to pool
        child = self._create_record(
            child_text.strip(), operator="reflective_mutation",
            parent_ids=[parent.id],
        )
        self._eval_on_pareto_set(child)
        self._pool.append(child)
        logger.info(
            f"[GEPA] child accepted: minibatch {parent_result.score:.3f} → "
            f"{child_result.score:.3f}, dev={child.score:.3f}"
        )

        return self._select_top_k(self._pool)

    # --- GEPA internals ---

    def _eval_on_pareto_set(self, record: PromptRecord) -> None:
        """Evaluate a candidate on the fixed dev set (fills performance_vector)."""
        result = self.evaluator.evaluate(record.text, self._pareto_samples)
        record.score = result.score
        record.performance_vector = result.performance_vector
        record.per_sample_details = result.per_sample_details
        record.scores["dev"] = result.score

    def _pareto_select(self) -> Optional[PromptRecord]:
        """Sample a candidate from the Pareto front.

        For each dev instance, find the best per-instance score across the
        pool. Candidates that attain that maximum on at least one instance
        are on the front; sampling weight = number of instances they win.
        """
        pool = [r for r in self._pool if r.performance_vector]
        if not pool:
            return self._pool[0] if self._pool else None

        n_instances = min(len(r.performance_vector) for r in pool)
        wins: Dict[str, int] = {r.id: 0 for r in pool}

        for i in range(n_instances):
            best = max(r.performance_vector[i] for r in pool)
            for r in pool:
                if r.performance_vector[i] >= best:
                    wins[r.id] += 1

        front = [r for r in pool if wins[r.id] > 0]
        if not front:
            return max(pool, key=lambda r: r.score)

        weights = [wins[r.id] for r in front]
        return random.choices(front, weights=weights, k=1)[0]

    def _reflective_mutate(
        self, prompt: str, traces: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Two-step reflective mutation: diagnose in natural language, then rewrite."""
        failures = [t for t in traces if not t.get("correct")]
        shown = failures[:4] if failures else traces[:4]
        trace_text = "\n".join(
            f"- Input: {t.get('input', '')[:120]}\n"
            f"  Expected: {t.get('target', '')}\n"
            f"  Model output: {t.get('prediction', '')[:120]}\n"
            f"  Correct: {t.get('correct')}"
            for t in shown
        )

        # Step 1: natural-language reflection on the traces
        reflection_prompt = (
            "An AI assistant used the instruction below and produced these "
            "results. Reflect on what the instruction fails to convey: what "
            "task rules, edge cases, or output format details is it missing? "
            "Answer in 2-4 sentences.\n\n"
            f"Instruction:\n{prompt}\n\n"
            f"Execution traces:\n{trace_text}\n\n"
            "Reflection:"
        )
        reflection = self._generate_prompt(reflection_prompt, temperature=0.7)
        if not reflection.strip():
            return None

        # Step 2: rewrite the instruction using the diagnosis
        rewrite_prompt = (
            "Rewrite the instruction to fix the issues identified in the "
            "reflection. Keep what works; add the missing task rules or "
            "format details. Output only the new instruction.\n\n"
            f"Current instruction:\n{prompt}\n\n"
            f"Reflection:\n{reflection.strip()}\n\n"
            "Improved instruction:"
        )
        return self._generate_prompt(
            rewrite_prompt, temperature=0.7, system_prompt=_IMPROVE_SYSTEM_PROMPT
        )
