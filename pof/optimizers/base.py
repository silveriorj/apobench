"""Base optimizer — template method pattern with shared techniques.

All optimizers inherit from BaseOptimizer and implement:
- _init_population(): Generate initial candidate pool
- _step(): One optimization iteration/phase

The base class provides:
- Audit integration (PromptRecord creation, history tracking)
- Evaluation helpers (with racing support)
- Shared generation techniques (Lamarckian, crossover, mutation, etc.)
"""
from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pof.audit.tracker import AuditTracker
from pof.core.types import EvalResult, GenerationConfig, OptimizationResult, PromptRecord
from pof.datasets.loader import TaskDataset
from pof.evaluation.evaluator import Evaluator
from pof.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class BaseOptimizer(ABC):
    """Abstract base for all prompt optimizers.

    Template method: optimize() calls _init_population() then _step() in a loop.
    Subclasses implement the specific optimization logic.
    """

    # Class-level name for registry
    name: str = "base"

    def __init__(
        self,
        llm: BaseLLM,
        dataset: TaskDataset,
        evaluator: Evaluator,
        population_size: int = 5,
        num_iterations: int = 3,
        seed_prompt: str = "",
        eval_sample_size: int = 50,
        output_dir: str = "outputs",
        **kwargs: Any,
    ):
        self.llm = llm
        self.dataset = dataset
        self.evaluator = evaluator
        self.population_size = population_size
        self.num_iterations = num_iterations
        self.seed_prompt = seed_prompt
        self.eval_sample_size = eval_sample_size
        self.kwargs = kwargs

        # Audit tracker
        self.tracker = AuditTracker(
            method_name=self.name,
            dataset_name=dataset.name,
            config=self._get_config_dict(),
            output_dir=output_dir,
        )

        # Population state
        self.population: List[PromptRecord] = []
        self.generation: int = 0
        self.best_record: Optional[PromptRecord] = None

    def optimize(self) -> OptimizationResult:
        """Run the full optimization loop.

        Returns:
            OptimizationResult with best prompt, scores, and audit trail.
        """
        self.tracker.start()
        logger.info(f"Starting {self.name} optimization on {self.dataset.name}")

        try:
            # Phase 0: Initialize population
            self.population = self._init_population()
            self._evaluate_population(self.population)
            self._update_best()
            self.tracker.record_generation(self.generation, self.population)
            logger.info(
                f"[Gen {self.generation}] Population initialized: "
                f"{len(self.population)} candidates, best={self.best_record.score:.4f}"
            )

            # Optimization loop
            for i in range(self.num_iterations):
                self.generation += 1
                try:
                    new_population = self._step()
                except StopIteration:
                    logger.info(f"Optimizer signaled early stop at generation {self.generation}")
                    break

                if new_population:
                    self.population = new_population
                self._update_best()
                self.tracker.record_generation(self.generation, self.population)
                logger.info(
                    f"[Gen {self.generation}] best={self.best_record.score:.4f}, "
                    f"pop_size={len(self.population)}"
                )

        except Exception as e:
            self.tracker.add_note(f"ERROR: {e}")
            logger.error(f"Optimization failed: {e}", exc_info=True)
            raise
        finally:
            self.tracker.end()
            self.tracker.usage = self.llm.get_usage()

        result = self.tracker.to_optimization_result()
        logger.info(
            f"Optimization complete: best_score={result.best_score:.4f}, "
            f"time={result.total_time:.1f}s, "
            f"llm_calls={self.llm.usage.total_calls}"
        )
        return result

    @abstractmethod
    def _init_population(self) -> List[PromptRecord]:
        """Generate initial population of candidates.

        Returns:
            List of PromptRecord candidates (unevaluated).
        """
        ...

    @abstractmethod
    def _step(self) -> List[PromptRecord]:
        """Execute one optimization step/phase.

        Returns:
            Updated population. Raise StopIteration to end early.
        """
        ...

    # --- Shared helpers ---

    def _evaluate_population(self, population: List[PromptRecord]) -> None:
        """Evaluate all candidates in the population."""
        samples = self.dataset.get_eval_samples("dev", n=self.eval_sample_size)
        for record in population:
            if record.score == 0.0 and record.text:
                result = self.evaluator.evaluate(record.text, samples)
                record.score = result.score
                record.performance_vector = result.performance_vector
                record.scores["dev"] = result.score

    def _evaluate_with_racing(
        self, candidates: List[PromptRecord], baseline_score: float
    ) -> List[PromptRecord]:
        """Evaluate candidates with racing against baseline."""
        samples = self.dataset.get_eval_samples("dev", n=self.eval_sample_size)
        for record in candidates:
            if record.text:
                result = self.evaluator.evaluate_with_racing(
                    record.text, samples, baseline_score
                )
                record.score = result.score
                record.performance_vector = result.performance_vector
                record.scores["dev"] = result.score
        return candidates

    def _select_top_k(
        self, candidates: List[PromptRecord], k: Optional[int] = None
    ) -> List[PromptRecord]:
        """Select top-k candidates by score."""
        k = k or self.population_size
        sorted_candidates = sorted(candidates, key=lambda r: r.score, reverse=True)
        return sorted_candidates[:k]

    def _update_best(self) -> None:
        """Update best_record from current population."""
        if self.population:
            current_best = max(self.population, key=lambda r: r.score)
            if self.best_record is None or current_best.score > self.best_record.score:
                self.best_record = current_best

    def _create_record(
        self,
        text: str,
        operator: str,
        parent_ids: Optional[List[str]] = None,
        **metadata: Any,
    ) -> PromptRecord:
        """Create a new PromptRecord with lineage info."""
        record = PromptRecord(
            text=text,
            operator=operator,
            parent_ids=parent_ids or [],
            generation_created=self.generation,
            metadata=metadata,
        )
        self.tracker.add_record(record)
        return record

    def _generate_prompt(
        self,
        instruction: str,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> str:
        """Generate text using the LLM."""
        config = GenerationConfig(
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
        )
        return self.llm.generate(instruction, config)

    def _get_config_dict(self) -> Dict[str, Any]:
        """Get optimizer configuration as dict."""
        return {
            "method": self.name,
            "population_size": self.population_size,
            "num_iterations": self.num_iterations,
            "eval_sample_size": self.eval_sample_size,
            "seed_prompt": self.seed_prompt[:100],
            **{k: v for k, v in self.kwargs.items() if isinstance(v, (str, int, float, bool))},
        }

    # --- Shared generation techniques ---

    def _lamarckian_generate(
        self, samples: List[Dict[str, str]], n: int = 1
    ) -> List[str]:
        """Lamarckian operator: reverse-engineer instruction from I/O pairs.

        Given input-output examples, ask the LLM to infer what instruction
        would produce those outputs.
        """
        few_shot = samples[:5]
        examples_text = "\n".join(
            f"Input: {s['input']}\nOutput: {s['target']}" for s in few_shot
        )

        meta_prompt = (
            "Given the following input-output examples, reverse-engineer the instruction "
            "that would produce these outputs from the inputs. Write ONLY the instruction, "
            "nothing else.\n\n"
            f"Examples:\n{examples_text}\n\n"
            "Instruction:"
        )

        results = []
        for _ in range(n):
            result = self._generate_prompt(meta_prompt, temperature=0.8)
            if result.strip():
                results.append(result.strip())
        return results

    def _semantic_variation(self, prompt: str, n: int = 1) -> List[str]:
        """Generate semantic variations of a prompt (paraphrasing)."""
        meta_prompt = (
            "Rewrite the following instruction in a different way while preserving "
            "its exact meaning and intent. Use different words and structure.\n\n"
            f"Original instruction:\n{prompt}\n\n"
            "Rewritten instruction:"
        )

        results = []
        for _ in range(n):
            result = self._generate_prompt(meta_prompt, temperature=0.9)
            if result.strip():
                results.append(result.strip())
        return results

    def _crossover(self, prompt_a: str, prompt_b: str) -> str:
        """LLM-based crossover of two prompts."""
        meta_prompt = (
            "Combine the best elements of these two instructions into a single, "
            "improved instruction that captures the strengths of both.\n\n"
            f"Instruction A:\n{prompt_a}\n\n"
            f"Instruction B:\n{prompt_b}\n\n"
            "Combined instruction:"
        )
        return self._generate_prompt(meta_prompt, temperature=0.7)

    def _feedback_improve(
        self, prompt: str, failures: List[Dict[str, Any]]
    ) -> str:
        """Improve a prompt based on failure analysis."""
        failure_text = "\n".join(
            f"- Input: {f.get('input', '')[:80]}\n"
            f"  Expected: {f.get('target', '')}\n"
            f"  Got: {f.get('prediction', '')[:80]}"
            for f in failures[:5]
        )

        meta_prompt = (
            "The following instruction was used but produced incorrect outputs "
            "for some inputs. Analyze the failures and rewrite the instruction "
            "to fix these issues.\n\n"
            f"Current instruction:\n{prompt}\n\n"
            f"Failures:\n{failure_text}\n\n"
            "Improved instruction:"
        )
        return self._generate_prompt(meta_prompt, temperature=0.7)

    def _eda_generate(self, prompts: List[str]) -> str:
        """EDA operator: generate from distribution of existing prompts."""
        prompts_text = "\n---\n".join(prompts[:5])
        meta_prompt = (
            "Here are several instructions that all aim to accomplish the same task. "
            "Generate a NEW instruction that captures the common patterns and best "
            "elements from all of them, but is distinct from each.\n\n"
            f"Existing instructions:\n{prompts_text}\n\n"
            "New instruction:"
        )
        return self._generate_prompt(meta_prompt, temperature=0.8)