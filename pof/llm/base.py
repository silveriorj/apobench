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

    def __init__(self, model_name: str, **kwargs: Any):
        self.model_name = model_name
        self.usage = LLMUsageStats()
        self._budget = None

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
