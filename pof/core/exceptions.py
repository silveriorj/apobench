"""Framework-wide exception hierarchy."""


class POFError(Exception):
    """Base exception for all POF errors."""

    pass


class LLMError(POFError):
    """Error during LLM generation or communication."""

    pass


class EvaluationError(POFError):
    """Error during prompt evaluation."""

    pass


class ConfigError(POFError):
    """Error in configuration loading or validation."""

    pass


class OptimizationError(POFError):
    """Error during optimization process."""

    pass


class DatasetError(POFError):
    """Error loading or processing datasets."""

    pass