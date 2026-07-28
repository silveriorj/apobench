"""LLM factory — create backend instances from configuration."""
from __future__ import annotations

from typing import Any, Optional

from pof.config.schemas import LLMConfig
from pof.core.exceptions import LLMError
from pof.llm.base import BaseLLM


def create_llm(config: Optional[LLMConfig] = None, **kwargs: Any) -> BaseLLM:
    """Create an LLM backend instance from configuration.

    Args:
        config: LLMConfig instance. If None, uses defaults.
        **kwargs: Override config fields.

    Returns:
        Configured BaseLLM instance.

    Raises:
        LLMError: If backend is unknown or initialization fails.
    """
    if config is None:
        config = LLMConfig(**kwargs)

    backend = config.backend.lower()

    if backend == "huggingface" or backend == "hf":
        from pof.llm.huggingface import HuggingFaceLLM

        return HuggingFaceLLM(
            model_name=config.model_name,
            device=config.device,
            dtype=config.dtype,
            thinking_mode=config.thinking_mode,
        )
    elif backend == "openai":
        from pof.llm.openai_backend import OpenAILLM

        return OpenAILLM(
            model_name=config.model_name,
            api_key=config.api_key,
            base_url=config.base_url,
        )
    elif backend == "ollama":
        from pof.llm.ollama_backend import OllamaLLM

        return OllamaLLM(
            model_name=config.model_name,
            base_url=config.base_url or "http://127.0.0.1:11434",
            thinking_mode=config.thinking_mode,
        )
    else:
        raise LLMError(
            f"Unknown LLM backend: '{backend}'. Supported: 'huggingface', 'openai', 'ollama'"
        )