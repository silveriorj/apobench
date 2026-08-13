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


# task_type -> system prompt, shared between Evaluator's own default and any
# caller that needs to route a SPECIFIC call to a different mode than the
# instance default (see evaluate()'s system_prompt_override) -- e.g. an
# optimizer searching across AO/CoT/thinking as a candidate-level property
# rather than a fixed per-run setting.
SYSTEM_PROMPT_BY_TASK_TYPE: Dict[str, str] = {
    "math": _MATH_EVAL_SYSTEM_PROMPT,
    "code": _CODE_EVAL_SYSTEM_PROMPT,
    "cot": _COT_EVAL_SYSTEM_PROMPT,
    "thinking": _THINKING_EVAL_SYSTEM_PROMPT,
    "dyck": _DYCK_EVAL_SYSTEM_PROMPT,
}


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
        system_prompt: Optional[str] = None,
    ):
        self.llm = llm
        self.score_fn = score_fn or create_score_function(task_type)
        self.task_type = task_type
        # None (default) -> auto-select by task_type, same as before. Any
        # string (including "") -> use it verbatim, e.g. to strip the eval
        # harness's format-enforcing system prompt entirely and isolate how
        # much of a CoT run's score comes from that scaffolding vs. the
        # seed prompt / model's own reasoning tendencies.
        self.system_prompt = (
            system_prompt if system_prompt is not None
            else SYSTEM_PROMPT_BY_TASK_TYPE.get(task_type, _EVAL_SYSTEM_PROMPT)
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.batch_size = batch_size

    def evaluate(
        self,
        prompt: str,
        samples: List[Dict[str, str]],
        num_samples: Optional[int] = None,
        shuffle: bool = True,
        system_prompt_override: Optional[str] = None,
        max_new_tokens_override: Optional[int] = None,
    ) -> EvalResult:
        """Evaluate a prompt on a set of samples.

        Args:
            prompt: The instruction/system prompt to evaluate.
            samples: List of dicts with 'input' and 'target' keys.
            num_samples: Max samples to evaluate (None = all).
            shuffle: Whether to shuffle samples before evaluation.
            system_prompt_override: Use this system prompt instead of the
                instance default for this call only. For an optimizer
                searching across eval modes (AO/CoT/thinking) as a
                candidate-level property rather than one fixed per-run
                setting -- see SYSTEM_PROMPT_BY_TASK_TYPE.
            max_new_tokens_override: Use this token budget instead of the
                instance default for this call only (pair with
                system_prompt_override -- a CoT-style prompt needs far more
                room than the instance's answer-only default).

        Returns:
            EvalResult with score, performance vector, and details.
        """
        if num_samples and num_samples < len(samples):
            if shuffle:
                samples = random.sample(samples, num_samples)
            else:
                samples = samples[:num_samples]

        system_prompt = system_prompt_override if system_prompt_override is not None else self.system_prompt
        max_new_tokens = max_new_tokens_override if max_new_tokens_override is not None else self.max_new_tokens

        config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
        )

        n_batches = math.ceil(len(samples) / self.batch_size)
        logger.debug(
            f"[Eval] {len(samples)} samples | batch_size={self.batch_size}"
            f" → {n_batches} batch(es) | max_tokens={max_new_tokens}"
        )

        eval_prompts = [
            self._format_eval_prompt(prompt, sample["input"])
            for sample in samples
        ]

        predictions = self._batch_generate(eval_prompts, config, system_prompt=system_prompt)
        predictions = self._retry_empty(eval_prompts, predictions, config, system_prompt)

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
        system_prompt_override: Optional[str] = None,
        max_new_tokens_override: Optional[int] = None,
        shuffle: bool = True,
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
            system_prompt_override: Use this system prompt instead of the
                instance default for this call only (parity with
                `evaluate()`/`evaluate_with_batch_racing()` — previously
                missing here, so any caller searching across AO/CoT/thinking
                modes silently fell back to the instance default whenever it
                used this method instead of the other two).
            max_new_tokens_override: Use this token budget instead of the
                instance default for this call only. Previously missing
                here too: a CoT/thinking/math-mode Evaluator constructed
                with the AO-mode default of 32 tokens would silently
                truncate every generation mid-reasoning when evaluated via
                this method, with no way to override per call.
            shuffle: Whether to shuffle `samples` before racing. The
                Hoeffding bound assumes i.i.d. sampling order; if `samples`
                arrives pre-sorted or grouped (e.g. by sub-task/difficulty),
                the running score used for early elimination is computed
                over a non-representative prefix. Defaults to True, matching
                `evaluate()`'s own default.

        Returns:
            EvalResult (may be partial if racing terminated early).
        """
        max_samples = max_samples or len(samples)
        samples_to_use = list(samples)
        if shuffle:
            random.shuffle(samples_to_use)
        samples_to_use = samples_to_use[:max_samples]

        system_prompt = system_prompt_override if system_prompt_override is not None else self.system_prompt
        max_new_tokens = max_new_tokens_override if max_new_tokens_override is not None else self.max_new_tokens

        logger.debug(
            f"[Racing] baseline={baseline_score:.3f} | up to {len(samples_to_use)} samples"
            f" | max_tokens={max_new_tokens}"
        )

        config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
        )

        performance_vector = []
        per_sample_details = []
        num_correct = 0

        for i, sample in enumerate(samples_to_use):
            eval_prompt = self._format_eval_prompt(prompt, sample["input"])
            pred = self.llm.generate(
                eval_prompt, config, system_prompt=system_prompt
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
        system_prompt_override: Optional[str] = None,
        max_new_tokens_override: Optional[int] = None,
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
        # Bug fix: the between-batch bound assumes i.i.d. batch order, same
        # as evaluate_with_racing's per-sample bound. `samples` previously
        # went through unshuffled, so if the caller passed a pre-sorted or
        # grouped list, the running score at the first bound check could be
        # computed over a non-representative prefix.
        samples = list(samples)
        random.shuffle(samples)
        system_prompt = system_prompt_override if system_prompt_override is not None else self.system_prompt
        max_new_tokens = max_new_tokens_override if max_new_tokens_override is not None else self.max_new_tokens
        config = GenerationConfig(
            max_new_tokens=max_new_tokens,
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
                eval_prompts, config, system_prompt=system_prompt
            )
            predictions = self._retry_empty(eval_prompts, predictions, config, system_prompt)
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
        self, prompts: List[str], config: GenerationConfig,
        system_prompt: Optional[str] = None,
    ) -> List[str]:
        """Generate predictions in batches for efficiency."""
        all_predictions = []
        total = len(prompts)
        n_batches = math.ceil(total / self.batch_size)
        for batch_idx, i in enumerate(range(0, total, self.batch_size), start=1):
            batch = prompts[i: i + self.batch_size]
            logger.debug(f"[Eval] batch {batch_idx}/{n_batches} (samples {i+1}–{i+len(batch)}/{total})")
            predictions = self.llm.generate_batch(
                batch, config, system_prompt=system_prompt if system_prompt is not None else self.system_prompt
            )
            all_predictions.extend(predictions)
        return all_predictions

    def _retry_empty(
        self,
        eval_prompts: List[str],
        predictions: List[str],
        config: GenerationConfig,
        system_prompt: Optional[str],
    ) -> List[str]:
        """LM-Assertion-style hard constraint: an eval call must produce SOME
        answer. Retry just the empty ones once, with the violation appended.

        Motivated by a real failure observed this session: a reasoning model
        under a tight token budget spent its whole budget "thinking" and
        returned an empty answer field -- not wrong, just absent, which
        `score_fn` scores identically to any other wrong answer even though
        it's a distinct failure mode (format/budget violation, not a
        reasoning error) that a one-line correction can often fix.
        """
        empty_idx = [i for i, p in enumerate(predictions) if not (p or "").strip()]
        if not empty_idx:
            return predictions

        retry_prompts = [
            f"{eval_prompts[i]}\n\n"
            "(Your previous response was empty. You MUST provide an answer "
            "in the required format -- do not leave it blank.)"
            for i in empty_idx
        ]
        system_prompt = system_prompt if system_prompt is not None else self.system_prompt
        retried = self.llm.generate_batch(retry_prompts, config, system_prompt=system_prompt)
        n_recovered = sum(1 for r in retried if (r or "").strip())
        if n_recovered:
            logger.info(
                f"[Eval] retried {len(empty_idx)} empty response(s), "
                f"recovered {n_recovered}"
            )
        out = list(predictions)
        for i, r in zip(empty_idx, retried):
            out[i] = r
        return out

    def _format_eval_prompt(self, instruction: str, input_text: str) -> str:
        """Format evaluation prompt combining instruction and input."""
        if not input_text:
            return instruction
        return f"{instruction}\n\n{input_text}"