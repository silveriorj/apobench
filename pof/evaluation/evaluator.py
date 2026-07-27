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

# General CoT for BBH tasks: brief one-line reasoning steps, end with "So the answer is X."
# Mirrors the GSM8K style ("brief one-line calculations") to cap output length.
# _score_auto extracts the final answer via _extract_cot_answer.
_COT_EVAL_SYSTEM_PROMPT = (
    "Reason through the problem using brief one-line steps — one thought per line, no prose. "
    "Do not state the answer directly without reasoning first. "
    "End with 'So the answer is ' followed by your final answer "
    "(a single word, letter, or short phrase — nothing after it)."
)

# "Thinking" mode: full step-by-step reasoning, ending in \boxed{}. Unlike
# _COT_EVAL_SYSTEM_PROMPT (brief one-line steps, "So the answer is X"), this
# allows free-form reasoning and uses the \boxed{} convention shared with math
# tasks. _extract_cot_answer tries \boxed{} first, so this and _score_auto's
# MCQ/AO/text fallbacks compose without a dedicated score function.
_THINKING_EVAL_SYSTEM_PROMPT = (
    "Think through the problem step by step, then give your final answer. "
    "End your response with \\boxed{X} where X is your final answer — "
    "the option letter, word, or short phrase, nothing else inside the box."
)

# Dyck-n: bracket completion requires stack simulation — allow brief CoT.
# Force the compact chain-of-thought-hub format (one line per symbol, no markdown)
# so the simulation fits within the 512-token budget even for 50-symbol inputs.
_DYCK_EVAL_SYSTEM_PROMPT = (
    "Use compact stack notation, one line per symbol, exactly like the examples "
    "(e.g. '1: [ ; stack: [ {'). No markdown, no headers, no bullet points. "
    "After the last symbol write the conclusion and end with "
    "'So the answer is X' where X is the closing brackets separated by spaces."
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
            "cot": _COT_EVAL_SYSTEM_PROMPT,
            "thinking": _THINKING_EVAL_SYSTEM_PROMPT,
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

    def evaluate_with_batch_racing(
        self,
        prompt: str,
        samples: List[Dict[str, str]],
        threshold: float,
        confidence: float = 0.05,
        min_batches: int = 1,
    ) -> EvalResult:
        """Batched evaluation with a Hoeffding-bound early stop between batches.

        `evaluate_with_racing` checks the bound after every SAMPLE, which means
        it cannot use `_batch_generate`'s batching — one generate call per
        sample instead of one per `batch_size` samples. At FUNNEL's per-phase N
        (22-88) that lost batching throughput usually costs more than early
        elimination saves. This checks the bound between BATCHES instead,
        keeping full batching throughput within each batch while still cutting
        off a candidate that is already statistically unlikely to reach
        `threshold` before it pays for the remaining batches.

        Args:
            threshold: score a candidate must plausibly reach to be worth
                continuing to evaluate (typically the current population's
                floor score, not a fixed baseline).
            min_batches: batches to run before the bound check kicks in, so a
                single unlucky first batch cannot eliminate a candidate.
        """
        config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
        )
        performance_vector: List[int] = []
        per_sample_details: List[Dict[str, Any]] = []
        num_correct = 0
        terminated = False

        for batch_idx, i in enumerate(range(0, len(samples), self.batch_size), start=1):
            batch = samples[i:i + self.batch_size]
            eval_prompts = [
                self._format_eval_prompt(prompt, s["input"]) for s in batch
            ]
            predictions = self.llm.generate_batch(
                eval_prompts, config, system_prompt=self.system_prompt
            )
            for pred, sample in zip(predictions, batch):
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

            n = len(performance_vector)
            if batch_idx >= min_batches and n < len(samples):
                current_score = num_correct / n
                bound = math.sqrt(math.log(2.0 / confidence) / (2 * n))
                if current_score + bound < threshold:
                    logger.info(
                        f"[BatchRacing] eliminated at {n}/{len(samples)} samples "
                        f"(score={current_score:.3f}+{bound:.3f} < "
                        f"threshold={threshold:.3f})"
                    )
                    terminated = True
                    break

        total = len(performance_vector)
        accuracy = num_correct / total if total > 0 else 0.0
        return EvalResult(
            score=accuracy,
            num_correct=num_correct,
            num_total=total,
            performance_vector=performance_vector,
            per_sample_details=per_sample_details,
            metadata={"racing_terminated": terminated},
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