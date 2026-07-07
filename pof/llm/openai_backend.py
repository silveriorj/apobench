"""OpenAI backend — API-based generation with usage tracking."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from pof.core.exceptions import LLMError
from pof.core.types import GenerationConfig
from pof.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """OpenAI API backend with batched generation support.

    Supports any OpenAI-compatible API (OpenAI, Azure, local vLLM, etc.)
    via the `base_url` parameter.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(model_name, **kwargs)
        self._api_key = api_key
        self._base_url = base_url
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize OpenAI client."""
        try:
            import openai

            client_kwargs: Dict[str, Any] = {}
            if self._api_key:
                client_kwargs["api_key"] = self._api_key
            if self._base_url:
                client_kwargs["base_url"] = self._base_url

            self._client = openai.OpenAI(**client_kwargs)
        except ImportError:
            raise LLMError("openai package not installed. Run: pip install openai")
        except Exception as e:
            raise LLMError(f"Failed to initialize OpenAI client: {e}") from e

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a single response via OpenAI API."""
        config = config or GenerationConfig()
        messages = self._build_messages(prompt, system_prompt)

        start = time.time()
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
            )
            elapsed = time.time() - start

            # Track usage from response
            usage = response.usage
            if usage:
                self._track_call(
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                    elapsed=elapsed,
                )
            else:
                self._track_call(0, 0, elapsed)

            return response.choices[0].message.content or ""

        except Exception as e:
            raise LLMError(f"OpenAI API call failed: {e}") from e

    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> List[str]:
        """Generate responses for multiple prompts (sequential API calls)."""
        results = []
        for prompt in prompts:
            result = self.generate(prompt, config, system_prompt)
            results.append(result)
        return results

    def _build_messages(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Build OpenAI-format messages."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages