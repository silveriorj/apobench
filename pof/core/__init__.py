"""Core layer — types, protocols, exceptions. No outward dependencies."""

from pof.core.types import (
    PromptRecord,
    EvalResult,
    GenerationConfig,
    LLMUsageStats,
    OptimizationResult,
)
from pof.core.exceptions import POFError, LLMError, EvaluationError, ConfigError

__all__ = [
    "PromptRecord",
    "EvalResult",
    "GenerationConfig",
    "LLMUsageStats",
    "OptimizationResult",
    "POFError",
    "LLMError",
    "EvaluationError",
    "ConfigError",
]