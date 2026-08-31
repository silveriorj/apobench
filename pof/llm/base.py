"""Base LLM interface — abstract contract for all backends."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pof.core.types import GenerationConfig, LLMUsageStats


class BaseLLM(ABC):
    """Abstract base class for LLM backends.

    All backends must implement `generate()` and `generate_batch()`.
    Usage statistics are tracked automatically via the `_track_call` helper.
    """

    def __init__(
        self, model_name: str, default_max_new_tokens: int = 512, **kwargs: Any
    ):
        self.model_name = model_name
        # The budget non-eval callers get when they don't specify one
        # themselves -- operator/generation meta-prompts (rewrite, critique,
        # crossover, ...) via BaseOptimizer._generate_prompt. Eval calls go
        # through Evaluator, which always states its own max_new_tokens
        # explicitly and never falls back to this. Previously hardcoded to
        # 512 in _generate_prompt's own signature, with LLMConfig.max_new_tokens
        # accepted by every config schema and never actually read by any
        # backend -- a run could set llm.max_new_tokens: 4096 in its YAML and
        # every operator call would silently still generate at 512.
        self.default_max_new_tokens = default_max_new_tokens
        self.usage = LLMUsageStats()
        self._budget = None
        # Generations that used their entire token budget, i.e. were cut off
        # rather than stopping at EOS. Backends that can detect this increment
        # it and the OpenAI-family backends (finish_reason == "length"); ollama
        # leaves it at zero. See `truncated_generations`.
        self._truncated_generations = 0

    @property
    def truncated_generations(self) -> int:
        """How many generations were cut off by the token cap so far.

        A truncated generation is scored like any other, but its answer never
        finished, so the score measures the decode budget rather than the
        prompt. Left unmeasured this is indistinguishable from genuinely tied
        methods: on a sibling archive, four different optimizers on GSM8K at a
        32-token cap all scored exactly 34.8% with 80% of generations truncated.

        Snapshot this around an evaluation and divide by the sample count to get
        that evaluation's truncation rate.
        """
        return getattr(self, "_truncated_generations", 0)

    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a single response.

        Args:
            prompt: The user prompt/instruction.
            config: Generation configuration (temperature, max_tokens, etc.).
            system_prompt: Optional system prompt.

        Returns:
            Generated text response.
        """
        ...

    @abstractmethod
    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> List[str]:
        """Generate responses for multiple prompts efficiently.

        Args:
            prompts: List of user prompts.
            config: Generation configuration.
            system_prompt: Optional system prompt (applied to all).

        Returns:
            List of generated text responses.
        """
        ...

    def _track_call(
        self,
        input_tokens: int,
        output_tokens: int,
        elapsed: float,
        is_eval: bool = False,
    ) -> None:
        """Track a single LLM call for usage statistics."""
        self.usage.total_calls += 1
        self.usage.total_input_tokens += input_tokens
        self.usage.total_output_tokens += output_tokens
        self.usage.total_time_seconds += elapsed
        if is_eval:
            self.usage.evaluation_calls += 1
        else:
            self.usage.generation_calls += 1

    def reset_usage(self) -> None:
        """Reset usage statistics."""
        self.usage = LLMUsageStats()

    def get_usage(self) -> LLMUsageStats:
        """Get current usage statistics."""
        return self.usage

    # ---- Budget integration ----
    def attach_budget(self, budget: Any) -> None:
        """Attach a BudgetManager-like object providing caps and usage checks."""
        self._budget = budget

    def get_budget(self) -> Any:
        """Return attached budget manager (if any)."""
        return self._budget
