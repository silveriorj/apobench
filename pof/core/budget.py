"""Budget management — hard-cap enforcement for time, calls, and tokens.

Centralized BudgetManager used by Orchestrator/LLM/Optimizers to ensure:
- Hard caps on wall-clock time, API calls, and token usage
- Early feasibility checks before generation
- Per-call planning to clamp max_new_tokens so limits are not exceeded
"""
from __future__ import annotations

import time
from typing import Any, Optional

from pof.config.schemas import BudgetConfig
from pof.core.exceptions import BudgetExceeded
from pof.core.types import LLMUsageStats


class BudgetManager:
    """Hard-cap budget manager.

    Caps (all optional; None means no cap):
        - time_seconds: wall-clock seconds
        - max_calls: total LLM calls (generate or generate_batch)
        - max_total_tokens: total tokens (input + output)
        - max_input_tokens: total input tokens
        - max_output_tokens: total output tokens
        - max_generations: enforced in optimizer loop
        - early_stop_patience: enforced in optimizer loop

    Typical usage:
        bm = BudgetManager(config.budget)
        llm.attach_budget(bm)  # to read live usage stats
        # Before a generation call, the LLM asks:
        allowed = bm.plan_generation(input_tokens_sum, prompts_in_call, planned_max_new_tokens)
        # If allowed == 0 or BudgetExceeded raised, the call must not proceed.
    """

    def __init__(self, cfg: Optional[BudgetConfig] = None):
        self.config: BudgetConfig = cfg or BudgetConfig()
        self.start_time: float = time.time()
        self._llm: Optional[Any] = None

    # ---- Lifecycle ----

    def attach_llm(self, llm: Any) -> None:
        """Attach LLM instance to read live usage stats."""
        self._llm = llm

    # ---- Read current usage ----

    @property
    def usage(self) -> LLMUsageStats:
        if self._llm is None:
            return LLMUsageStats()
        return self._llm.get_usage()

    # ---- Remaining budgets ----

    def remaining_time(self) -> Optional[float]:
        if self.config.time_seconds is None:
            return None
        elapsed = time.time() - self.start_time
        return max(0.0, self.config.time_seconds - elapsed)

    def remaining_calls(self) -> Optional[int]:
        if self.config.max_calls is None:
            return None
        return max(0, self.config.max_calls - self.usage.total_calls)

    def remaining_total_tokens(self) -> Optional[int]:
        if self.config.max_total_tokens is None:
            return None
        return max(0, self.config.max_total_tokens - self.usage.total_tokens)

    def remaining_input_tokens(self) -> Optional[int]:
        if self.config.max_input_tokens is None:
            return None
        return max(0, self.config.max_input_tokens - self.usage.total_input_tokens)

    def remaining_output_tokens(self) -> Optional[int]:
        if self.config.max_output_tokens is None:
            return None
        return max(0, self.config.max_output_tokens - self.usage.total_output_tokens)

    # ---- Global stop check ----

    def should_stop(self) -> Optional[str]:
        """Return reason string if any cap is exhausted, else None."""
        # Time
        rt = self.remaining_time()
        if rt is not None and rt <= 0.0:
            return "time"
        # Calls
        rc = self.remaining_calls()
        if rc is not None and rc <= 0:
            return "calls"
        # Total tokens
        rtt = self.remaining_total_tokens()
        if rtt is not None and rtt <= 0:
            return "total_tokens"
        # Input tokens
        rit = self.remaining_input_tokens()
        if rit is not None and rit <= 0:
            return "input_tokens"
        # Output tokens
        rot = self.remaining_output_tokens()
        if rot is not None and rot <= 0:
            return "output_tokens"
        return None

    # ---- Per-call planning ----

    def _ensure_call_possible(self) -> None:
        reason = self.should_stop()
        if reason is not None:
            raise BudgetExceeded(f"Budget exhausted: {reason}", kind=reason)

    def plan_generation(
        self,
        input_tokens_sum: int,
        prompts_in_call: int,
        planned_max_new_tokens: int,
    ) -> int:
        """Compute allowed max_new_tokens per prompt for this call.

        Args:
            input_tokens_sum: Sum of input tokens for this API call (batched or single).
            prompts_in_call: Number of prompts in this call (>=1).
            planned_max_new_tokens: Desired max_new_tokens per prompt from config.

        Returns:
            Allowed max_new_tokens per prompt (may be lower than planned).

        Raises:
            BudgetExceeded if the call cannot be made without violating a hard cap.
        """
        if prompts_in_call <= 0:
            raise ValueError("prompts_in_call must be >= 1")

        # Check time and calls first
        self._ensure_call_possible()

        # Token-level feasibility checks before the call
        r_total = self.remaining_total_tokens()
        if r_total is not None and input_tokens_sum > r_total:
            raise BudgetExceeded(
                f"Budget would exceed total tokens: need_in={input_tokens_sum} > remaining_total={r_total}",
                kind="total_tokens",
            )

        r_input = self.remaining_input_tokens()
        if r_input is not None and input_tokens_sum > r_input:
            raise BudgetExceeded(
                f"Budget would exceed input tokens: need_in={input_tokens_sum} > remaining_input={r_input}",
                kind="input_tokens",
            )

        # Compute allowed new tokens per prompt under output and total caps
        allowed_per_prompt = planned_max_new_tokens

        # Output-only cap (shared across prompts)
        r_output = self.remaining_output_tokens()
        if r_output is not None:
            per_prompt_output_cap = max(0, r_output // prompts_in_call)
            allowed_per_prompt = min(allowed_per_prompt, per_prompt_output_cap)

        # Total tokens cap also constrains output: remaining_total - input_tokens_sum
        if r_total is not None:
            remaining_for_output_total = max(0, r_total - input_tokens_sum)
            per_prompt_total_cap = max(0, remaining_for_output_total // prompts_in_call)
            allowed_per_prompt = min(allowed_per_prompt, per_prompt_total_cap)

        if allowed_per_prompt <= 0:
            raise BudgetExceeded(
                "No remaining output tokens available for this call", kind="output_tokens"
            )

        return allowed_per_prompt