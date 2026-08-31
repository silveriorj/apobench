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

    # Third-party backends first, so a package can add one without patching
    # this function. `sap_aicore_backend.register()` had to monkey-patch this
    # module at runtime and rebind the name inside pof.orchestration.runner —
    # that is what an extension point looks like when it does not exist.
    from pof.plugins import BACKEND_GROUP, discover

    external = discover(BACKEND_GROUP)
    if backend in external:
        factory_fn = external[backend]
        return factory_fn(config)

    if backend == "huggingface" or backend == "hf":
        from pof.llm.huggingface import HuggingFaceLLM

        return HuggingFaceLLM(
            model_name=config.model_name,
            device=config.device,
            dtype=config.dtype,
            thinking_mode=config.thinking_mode,
            default_max_new_tokens=config.max_new_tokens,
        )
    elif backend == "openai":
        from pof.llm.openai_backend import OpenAILLM

        kwargs_out: dict = dict(
            model_name=config.model_name,
            api_key=config.api_key,
            base_url=config.base_url,
            default_max_new_tokens=config.max_new_tokens,
        )
        if config.max_workers is not None:
            kwargs_out["max_workers"] = config.max_workers
        return OpenAILLM(**kwargs_out)
    elif backend == "ollama":
        from pof.llm.ollama_backend import OllamaLLM

        return OllamaLLM(
            model_name=config.model_name,
            base_url=config.base_url or "http://127.0.0.1:11434",
            default_max_new_tokens=config.max_new_tokens,
            thinking_mode=config.thinking_mode,
        )
    else:
        known = ["huggingface", "openai", "ollama"] + sorted(external)
        raise LLMError(
            f"Unknown LLM backend: '{backend}'. Supported: {known}. "
            "Third-party backends are added via the 'apobench.backends' entry "
            "point group or the APOBENCH_PLUGINS environment variable."
        )