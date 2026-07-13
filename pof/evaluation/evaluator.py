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

# Injected on every evaluation call so the model skips CoT and gives a direct
# answer.  This lets max_new_tokens stay short (64) while scoring correctly.
_EVAL_SYSTEM_PROMPT = (
    'Output your answer as JSON with a single field, e.g. {"answer": "C"} or {"answer": "Yes"}. '
    "Use only the answer letter or word — no reasoning, no explanation, no preamble."
)

# Math eval: brief CoT allowed; scorer extracts "The answer is X" from the end.
_MATH_EVAL_SYSTEM_PROMPT = (
    "Solve the problem step by step. "
    "At the end, write 'The answer is ' followed by the final answer."
)

# Code tasks (HumanEval): output executable code only — the scorer runs it.
_CODE_EVAL_SYSTEM_PROMPT = (
    "You are a Python coding assistant. Output only the code completion — "
    "no explanations, no usage examples, no markdown commentary."
)

# Dyck-n: bracket completion requires stack simulation — allow brief CoT.
# Scorer extracts the final bracket sequence from the last line or "So the answer is X".
_DYCK_EVAL_SYSTEM_PROMPT = (
    "Simulate the bracket stack step by step, then output the closing sequence. "
    "End with 'So the answer is ' followed by the closing brackets separated by spaces "
    "(e.g. ] } ])."
)


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
        self.task_type = task_type
        self.system_prompt = {
            "math": _MATH_EVAL_SYSTEM_PROMPT,
            "code": _CODE_EVAL_SYSTEM_PROMPT,
            "dyck": _DYCK_EVAL_SYSTEM_PROMPT,
        }.get(task_type, _EVAL_SYSTEM_PROMPT)
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

        n_batches = math.ceil(len(samples) / self.batch_size)
        logger.debug(
            f"[Eval] {len(samples)} samples | batch_size={self.batch_size}"
            f" → {n_batches} batch(es) | max_tokens={self.max_new_tokens}"
        )

        eval_prompts = [
            self._format_eval_prompt(prompt, sample["input"])
            for sample in samples
        ]

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
                "input": sample["input"],
                "target": target,
                "prediction": pred,
                "correct": bool(score),
            })

        total = len(samples)
        accuracy = num_correct / total if total > 0 else 0.0
        logger.info(f"[Eval] score={accuracy:.3f} ({num_correct}/{total} correct)")

        if logger.isEnabledFor(logging.DEBUG):
            for d in per_sample_details:
                mark = "✓" if d["correct"] else "✗"
                logger.debug(
                    f"  [{mark}] target={d['target']!r} pred={d['prediction']!r}"
                )

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

        logger.debug(
            f"[Racing] baseline={baseline_score:.3f} | up to {len(samples_to_use)} samples"
            f" | max_tokens={self.max_new_tokens}"
        )

        config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
        )

        performance_vector = []
        per_sample_details = []
        num_correct = 0

        for i, sample in enumerate(samples_to_use):
            eval_prompt = self._format_eval_prompt(prompt, sample["input"])
            pred = self.llm.generate(
                eval_prompt, config, system_prompt=self.system_prompt
            )

            target = sample["target"]
            score = self.score_fn(pred, target)
            performance_vector.append(score)
            num_correct += score
            per_sample_details.append({
                "input": sample["input"],
                "target": target,
                "prediction": pred,
                "correct": bool(score),
            })

            n = i + 1
            if n >= min_samples:
                current_score = num_correct / n
                bound = math.sqrt(math.log(2.0 / confidence) / (2 * n))

                if current_score + bound < baseline_score:
                    logger.info(
                        f"[Racing] eliminated at {n}/{len(samples_to_use)} samples"
                        f" (score={current_score:.3f}+{bound:.3f} < baseline={baseline_score:.3f})"
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
        total = len(prompts)
        n_batches = math.ceil(total / self.batch_size)
        for batch_idx, i in enumerate(range(0, total, self.batch_size), start=1):
            batch = prompts[i: i + self.batch_size]
            logger.debug(f"[Eval] batch {batch_idx}/{n_batches} (samples {i+1}–{i+len(batch)}/{total})")
            predictions = self.llm.generate_batch(
                batch, config, system_prompt=self.system_prompt
            )
            all_predictions.extend(predictions)
        return all_predictions

    def _format_eval_prompt(self, instruction: str, input_text: str) -> str:
        """Format evaluation prompt combining instruction and input."""
        if not input_text:
            return instruction
        return f"{instruction}\n\n{input_text}"