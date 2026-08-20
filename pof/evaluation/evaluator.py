"""Evaluator — prompt evaluation with batching and Hoeffding racing.

Combines Projeto's efficient batch evaluation with the framework's racing
for early termination of clearly inferior candidates.
"""
from __future__ import annotations

import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import re

from pof.core.types import EvalResult, GenerationConfig
from pof.evaluation.scoring import ScoreFunction, create_score_function
from pof.llm.base import BaseLLM


def _has_final_answer(text: str) -> bool:
    """True if text contains a recognizable final-answer marker.

    Used by _retry_truncated to distinguish 'hit the token limit mid-computation'
    from 'finished but answered incorrectly'.
    """
    if not text:
        return False
    if re.search(r"\\boxed\{", text):
        return True
    if re.search(r"<solution>", text, re.IGNORECASE):
        return True
    if re.search(r"[Tt]he answer is\s+\S", text):
        return True
    if re.search(r"(?:^|\n)Answer:\s+\S", text, re.MULTILINE):
        return True
    if re.search(r"####\s+\S", text):
        return True
    # Trailing digits (math_comp AIME: model ends response with "025")
    if re.search(r"\d{1,4}\s*$", text.strip()):
        return True
    return False

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

# "Thinking" mode: full free-form reasoning, ending in \boxed{} (shared
# convention with math tasks; _extract_cot_answer tries \boxed{} first).
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


# task_type -> system prompt. Also used by callers routing a specific call to
# a different mode than the instance default (see evaluate()'s
# system_prompt_override).
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
        racing_enabled: bool = True,
        racing_confidence: float = 0.05,
        racing_min_samples: int = 10,
    ):
        self.llm = llm
        self.score_fn = score_fn or create_score_function(task_type)
        self.task_type = task_type
        # None -> auto-select by task_type. Any string (including "") is
        # used verbatim, e.g. to strip the format-enforcing system prompt
        # entirely.
        self.system_prompt = (
            system_prompt if system_prompt is not None
            else SYSTEM_PROMPT_BY_TASK_TYPE.get(task_type, _EVAL_SYSTEM_PROMPT)
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.batch_size = batch_size
        # Racing policy lives here rather than at each call site: these keys
        # were declared in EvalConfig and set in YAML for many runs but never
        # reached the evaluator, so `racing_enabled: false` was a no-op and
        # the hardcoded 0.05/10 were always in force. Anything the config
        # declares must actually bind.
        self.racing_enabled = racing_enabled
        self.racing_confidence = racing_confidence
        self.racing_min_samples = racing_min_samples

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
                instance default for this call only (see SYSTEM_PROMPT_BY_TASK_TYPE).
            max_new_tokens_override: Use this token budget instead of the
                instance default for this call only (pair with
                system_prompt_override — CoT-style prompts need more room).

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
        predictions = self._retry_truncated(eval_prompts, predictions, config, system_prompt)

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
        confidence: Optional[float] = None,
        min_samples: Optional[int] = None,
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
            confidence: Significance level (alpha). None uses the
                instance value from config.
            min_samples: Minimum samples before racing kicks in. None uses
                the instance value from config.
            max_samples: Maximum samples to evaluate.
            system_prompt_override: Use this system prompt instead of the
                instance default for this call only.
            max_new_tokens_override: Use this token budget instead of the
                instance default for this call only.
            shuffle: Whether to shuffle `samples` before racing. The
                Hoeffding bound assumes i.i.d. sampling order, so a
                pre-sorted or grouped `samples` list would otherwise bias
                the running score used for early elimination.

        Returns:
            EvalResult (may be partial if racing terminated early).
        """
        if not self.racing_enabled:
            return self.evaluate(
                prompt, samples, num_samples=max_samples, shuffle=shuffle,
                system_prompt_override=system_prompt_override,
                max_new_tokens_override=max_new_tokens_override,
            )
        confidence = self.racing_confidence if confidence is None else confidence
        min_samples = self.racing_min_samples if min_samples is None else min_samples
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
        confidence: Optional[float] = None,
        min_batches: int = 1,
        system_prompt_override: Optional[str] = None,
        max_new_tokens_override: Optional[int] = None,
    ) -> EvalResult:
        """Batched evaluation with a Hoeffding-bound early stop between batches.

        Unlike `evaluate_with_racing` (which checks the bound per sample and
        so cannot batch generation calls), this checks between BATCHES,
        keeping batching throughput within each batch while still cutting
        off a candidate unlikely to reach `threshold`.

        Args:
            threshold: score a candidate must plausibly reach to be worth
                continuing to evaluate (typically the current population's
                floor score, not a fixed baseline).
            min_batches: batches to run before the bound check kicks in, so a
                single unlucky first batch cannot eliminate a candidate.
        """
        if not self.racing_enabled:
            return self.evaluate(
                prompt, samples,
                system_prompt_override=system_prompt_override,
                max_new_tokens_override=max_new_tokens_override,
            )
        confidence = self.racing_confidence if confidence is None else confidence
        # The between-batch bound assumes i.i.d. batch order, so shuffle to
        # avoid biasing the running score if samples arrive pre-sorted.
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
            predictions = self._retry_truncated(eval_prompts, predictions, config, system_prompt)
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
        """Retry empty predictions once with a note about the violation.

        A reasoning model under a tight token budget can spend it all
        "thinking" and return an empty answer — a distinct failure mode
        from a wrong-but-present answer, and one a retry can often fix.
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

    def _retry_truncated(
        self,
        eval_prompts: List[str],
        predictions: List[str],
        config: GenerationConfig,
        system_prompt: Optional[str],
        hard_cap: int = 4096,
    ) -> List[str]:
        """Extend generation for responses that hit the token limit mid-computation.

        A response is considered truncated when it has no final-answer marker
        (\\boxed{}, 'the answer is', 'Answer:', '####', '<solution>', trailing
        digits) AND its length exceeds ~1.5× the base token budget in chars
        (math text averages ~2 chars/token, so max_new_tokens × 3 chars ≈ the
        full budget; 1.5× catches responses that consumed most of it).

        Re-generates only the truncated responses with 2× the base token budget,
        capped at hard_cap. Calls that already have max_new_tokens ≥ hard_cap
        are returned unchanged.
        """
        if config.max_new_tokens >= hard_cap:
            return predictions

        extended_max = min(config.max_new_tokens * 2, hard_cap)
        # ~2 chars/token for math-heavy text; 1.5× leaves a margin so short
        # responses that answered quickly are not needlessly retried.
        truncation_threshold = config.max_new_tokens * 1.5

        needs_ext = [
            i for i, pred in enumerate(predictions)
            if not _has_final_answer(pred) and len(pred or "") > truncation_threshold
        ]
        if not needs_ext:
            return predictions

        ext_config = GenerationConfig(
            max_new_tokens=extended_max,
            temperature=config.temperature,
            do_sample=config.do_sample,
        )
        retry_prompts = [eval_prompts[i] for i in needs_ext]
        extended = self.llm.generate_batch(
            retry_prompts, ext_config, system_prompt=system_prompt
        )
        logger.info(
            f"[Eval] extended {len(needs_ext)} truncated response(s) "
            f"{config.max_new_tokens} → {extended_max} tokens"
        )
        out = list(predictions)
        for i, ext_pred in zip(needs_ext, extended):
            out[i] = ext_pred
        return out

    def _format_eval_prompt(self, instruction: str, input_text: str) -> str:
        """Format evaluation prompt combining instruction and input."""
        if not input_text:
            return instruction
        return f"{instruction}\n\n{input_text}"