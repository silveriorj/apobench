"""Evaluator — prompt evaluation with batching and Hoeffding racing.

Combines Projeto's efficient batch evaluation with the framework's racing
for early termination of clearly inferior candidates.
"""
from __future__ import annotations

import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple

from pof.core.types import EvalResult, GenerationConfig
from pof.evaluation.scoring import ScoreFunction, create_score_function
from pof.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate prompts against task samples with optional racing.

    Features:
    - Batch evaluation for efficiency
    - Hoeffding racing for early termination
    - Per-sample performance vectors
    - Configurable score functions
    """

    def __init__(
        self,
        llm: BaseLLM,
        score_fn: Optional[ScoreFunction] = None,
        task_type: str = "auto",
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        batch_size: int = 8,
    ):
        self.llm = llm
        self.score_fn = score_fn or create_score_function(task_type)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.batch_size = batch_size

    def evaluate(
        self,
        prompt: str,
        samples: List[Dict[str, str]],
        num_samples: Optional[int] = None,
        shuffle: bool = True,
    ) -> EvalResult:
        """Evaluate a prompt on a set of samples.

        Args:
            prompt: The instruction/system prompt to evaluate.
            samples: List of dicts with 'input' and 'target' keys.
            num_samples: Max samples to evaluate (None = all).
            shuffle: Whether to shuffle samples before evaluation.

        Returns:
            EvalResult with score, performance vector, and details.
        """
        if num_samples and num_samples < len(samples):
            if shuffle:
                samples = random.sample(samples, num_samples)
            else:
                samples = samples[:num_samples]

        config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
        )

        # Build evaluation prompts
        eval_prompts = [
            self._format_eval_prompt(prompt, sample["input"])
            for sample in samples
        ]

        # Batch generate
        predictions = self._batch_generate(eval_prompts, config)

        # Score
        performance_vector = []
        per_sample_details = []
        num_correct = 0

        for i, (pred, sample) in enumerate(zip(predictions, samples)):
            target = sample["target"]
            score = self.score_fn(pred, target)
            performance_vector.append(score)
            num_correct += score
            per_sample_details.append({
                "input": sample["input"][:100],
                "target": target,
                "prediction": pred[:200],
                "correct": bool(score),
            })

        total = len(samples)
        accuracy = num_correct / total if total > 0 else 0.0

        return EvalResult(
            score=accuracy,
            num_correct=num_correct,
            num_total=total,
            performance_vector=performance_vector,
            per_sample_details=per_sample_details,
        )

    def evaluate_with_racing(
        self,
        prompt: str,
        samples: List[Dict[str, str]],
        baseline_score: float,
        confidence: float = 0.05,
        min_samples: int = 10,
        max_samples: Optional[int] = None,
    ) -> EvalResult:
        """Evaluate with Hoeffding racing — early stop if clearly worse than baseline.

        Uses Hoeffding's inequality to determine if a candidate is statistically
        worse than the baseline, allowing early termination to save compute.

        Args:
            prompt: The prompt to evaluate.
            samples: Full sample set.
            baseline_score: Score to beat (current best).
            confidence: Significance level (alpha).
            min_samples: Minimum samples before racing kicks in.
            max_samples: Maximum samples to evaluate.

        Returns:
            EvalResult (may be partial if racing terminated early).
        """
        max_samples = max_samples or len(samples)
        samples_to_use = samples[:max_samples]

        config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
        )

        performance_vector = []
        per_sample_details = []
        num_correct = 0

        for i, sample in enumerate(samples_to_use):
            # Generate prediction
            eval_prompt = self._format_eval_prompt(prompt, sample["input"])
            pred = self.llm.generate(eval_prompt, config)

            # Score
            target = sample["target"]
            score = self.score_fn(pred, target)
            performance_vector.append(score)
            num_correct += score
            per_sample_details.append({
                "input": sample["input"][:100],
                "target": target,
                "prediction": pred[:200],
                "correct": bool(score),
            })

            # Racing check after min_samples
            n = i + 1
            if n >= min_samples:
                current_score = num_correct / n
                # Hoeffding bound: P(|X - E[X]| >= t) <= 2*exp(-2*n*t^2)
                # Solving for t: t = sqrt(ln(2/alpha) / (2*n))
                bound = math.sqrt(math.log(2.0 / confidence) / (2 * n))

                # If upper bound of current score is below baseline, terminate
                if current_score + bound < baseline_score:
                    logger.debug(
                        f"Racing: terminated at {n}/{len(samples_to_use)} samples "
                        f"(score={current_score:.3f}, bound={bound:.3f}, "
                        f"baseline={baseline_score:.3f})"
                    )
                    break

        total = len(performance_vector)
        accuracy = num_correct / total if total > 0 else 0.0

        return EvalResult(
            score=accuracy,
            num_correct=num_correct,
            num_total=total,
            performance_vector=performance_vector,
            per_sample_details=per_sample_details,
            metadata={"racing_terminated": total < len(samples_to_use)},
        )

    def _batch_generate(
        self, prompts: List[str], config: GenerationConfig
    ) -> List[str]:
        """Generate predictions in batches for efficiency."""
        all_predictions = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i: i + self.batch_size]
            predictions = self.llm.generate_batch(batch, config)
            all_predictions.extend(predictions)
        return all_predictions

    def _format_eval_prompt(self, instruction: str, input_text: str) -> str:
        """Format evaluation prompt combining instruction and input."""
        if not input_text:
            return instruction
        return f"{instruction}\n\n{input_text}"