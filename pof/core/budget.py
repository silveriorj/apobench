"""Budget management — hard-cap enforcement for time, calls, and tokens.

Centralized BudgetManager used by Orchestrator/LLM/Optimizers to ensure:
- Hard caps on wall-clock time, API calls, and token usage
- Early feasibility checks before generation
- Per-call planning to clamp max_new_tokens so limits are not exceeded
"""
from __future__ import annotations

import time
from typing import Any, List, Optional

from pof.config.schemas import BudgetConfig
from pof.core.exceptions import BudgetExceeded
from pof.core.types import LLMUsageStats


# Rolling window of observed full-dev-evaluation wall-clock durations
# (seconds), used to size the finalize-time reserve dynamically -- see
# remaining_search_time()'s docstring for the fixed-reserve gap this closes.
_EVAL_DURATION_WINDOW = 10

# Generic estimate of how many full evaluations _finalize() needs (e.g.
# HoldoutSelectionMixin re-evaluates FINALIZE_TOP_K=6 finalists on the
# holdout slice). Not tied to any specific optimizer's exact constant --
# just a conservative shared assumption so budget.py doesn't need to know
# about holdout.py.
_FINALIZE_EVAL_COUNT_ESTIMATE = 6


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
        self._eval_durations: List[float] = []

    # ---- Lifecycle ----

    def attach_llm(self, llm: Any) -> None:
        """Attach LLM instance to read live usage stats."""
        self._llm = llm

    # ---- Adaptive finalize-reserve tracking ----

    def record_eval_duration(self, seconds: float) -> None:
        """Feed an observed full-dev-evaluation wall-clock duration into a
        rolling window, used by remaining_search_time()/has_time_for_another_eval()
        to size the finalize-time reserve to how slow evaluation actually is
        on this task/dataset, instead of a fixed guess a single slow
        generation could still blow through."""
        self._eval_durations.append(max(0.0, seconds))
        if len(self._eval_durations) > _EVAL_DURATION_WINDOW:
            self._eval_durations.pop(0)

    def avg_eval_duration(self) -> Optional[float]:
        """Mean of the most recent recorded full-eval durations, or None
        if none have been recorded yet (e.g. before generation 0's
        evaluation completes)."""
        if not self._eval_durations:
            return None
        return sum(self._eval_durations) / len(self._eval_durations)

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

    def remaining_search_time(self) -> Optional[float]:
        """Like remaining_time(), but reserves a slice for _finalize().

        The optimization loop should check this (not remaining_time()) when
        deciding whether to start another generation, so a slice of the
        time budget is always left over for post-search work like holdout
        re-ranking. Per-call hard enforcement (_ensure_call_possible) still
        uses the true remaining_time(), so this reserve is advisory for the
        loop, not a second hard cap.

        The reserve is the LARGER of the static config-driven floor/fraction
        and a dynamic estimate (avg observed full-eval duration x
        _FINALIZE_EVAL_COUNT_ESTIMATE). Found via the 2026-08-14 BBH
        validation run: a single generation with slow per-candidate
        evaluation (BBH's code-adjacent tasks run much slower than
        HumanEval's) could still consume the ENTIRE static reserve on its
        own, since should_stop_for_search() is only checked once per
        generation boundary -- a generation that starts with plenty of
        reserve left can still finish having devoured it all. The dynamic
        estimate makes the reserve track actual observed cost instead of a
        fixed guess; has_time_for_another_eval() (below) provides the
        complementary mid-generation check so a single generation's
        evaluation phase can bail early rather than only being caught at
        the next generation boundary.
        """
        rt = self.remaining_time()
        if rt is None:
            return None
        static_reserve = max(
            self.config.finalize_reserve_min_seconds,
            (self.config.time_seconds or 0) * self.config.finalize_reserve_fraction,
        )
        avg = self.avg_eval_duration()
        dynamic_reserve = avg * _FINALIZE_EVAL_COUNT_ESTIMATE if avg is not None else 0.0
        reserve = max(static_reserve, dynamic_reserve)
        return max(0.0, rt - reserve)

    def has_time_for_another_eval(self) -> bool:
        """True if there's likely enough time for one more full-dev
        evaluation without starving _finalize()'s reserve.

        Meant to be checked WITHIN a generation, before each individual
        full-eval call (e.g. inside _evaluate_with_minibatch_gate's
        per-candidate loop) -- not just once per generation boundary like
        should_stop_for_search(). This is what lets a generation bail
        partway through its evaluation phase (skip remaining candidates,
        leave them at their minibatch/unset score) instead of running every
        gate-passed candidate's full eval regardless of how much reserve
        that would consume.
        """
        rst = self.remaining_search_time()
        if rst is None:
            return True
        avg = self.avg_eval_duration()
        if avg is None:
            # No observations yet (e.g. generation 0) -- fall back to a
            # coarse "any reserve left at all" check rather than blocking
            # the very first evaluation before there's any duration data.
            return rst > 0.0
        return rst > avg

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

    def should_stop_for_search(self) -> Optional[str]:
        """Like should_stop(), but treats the finalize time reserve as the
        time cap. Use this to decide whether to start another optimization
        generation; use should_stop() for the true hard per-call cap."""
        rst = self.remaining_search_time()
        if rst is not None and rst <= 0.0:
            return "time_reserved_for_finalize"
        return self.should_stop()

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