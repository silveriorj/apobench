"""OpenAI backend — API-based generation with usage tracking."""
from __future__ import annotations

import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from pof.core.exceptions import LLMError
from pof.core.types import GenerationConfig
from pof.llm.base import BaseLLM

logger = logging.getLogger(__name__)

# Retry policy for rate-limit (429) and transient server errors (5xx).
# 429 can also mean exhausted prepay credits, which retries won't fix — but
# the loop can't distinguish that from the exception alone, so it always
# backs off and eventually surfaces the real error after MAX_RETRIES.
_MAX_RETRIES = 6
_BASE_DELAY = 2.0  # seconds, doubles each retry
_MAX_DELAY = 60.0


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
        max_workers: int = 16,
        **kwargs: Any,
    ):
        super().__init__(model_name, **kwargs)
        self._api_key = api_key
        self._base_url = base_url
        self._client = None
        self._max_workers = max_workers
        self._track_lock = threading.Lock()
        self._init_client()

    def _init_client(self) -> None:
        """Initialize OpenAI client."""
        try:
            import openai

            client_kwargs: Dict[str, Any] = {
                # Disable the SDK's own retry — generate()'s loop is the
                # only retry policy that should apply.
                "max_retries": 0,
            }
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
        """Generate a single response via OpenAI API.

        Retries on 429 (rate limit) and 5xx (transient server error) with
        exponential backoff + jitter, honoring the server's `Retry-After`
        header when present. Every other exception (auth, bad request,
        model-not-found) fails immediately -- those never resolve by
        waiting.
        """
        import openai

        config = config or GenerationConfig()
        messages = self._build_messages(prompt, system_prompt)

        start = time.time()
        last_err: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                )
                elapsed = time.time() - start

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

            except (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError) as e:
                last_err = e
                if attempt >= _MAX_RETRIES:
                    break
                delay = self._retry_delay(e, attempt)
                logger.warning(
                    f"[{self.model_name}] {type(e).__name__} (attempt {attempt + 1}/{_MAX_RETRIES}), "
                    f"retrying in {delay:.1f}s"
                )
                time.sleep(delay)
            except Exception as e:
                raise LLMError(f"OpenAI API call failed: {e}") from e

        raise LLMError(
            f"OpenAI API call failed after {_MAX_RETRIES} retries: {last_err}"
        ) from last_err

    def _retry_delay(self, exc: Exception, attempt: int) -> float:
        """Honor the server's Retry-After header if present, else exponential backoff + jitter."""
        response = getattr(exc, "response", None)
        if response is not None:
            retry_after = response.headers.get("retry-after")
            if retry_after:
                try:
                    return min(float(retry_after), _MAX_DELAY)
                except ValueError:
                    pass
        base = min(_BASE_DELAY * (2 ** attempt), _MAX_DELAY)
        return base + random.uniform(0, base * 0.25)

    def _track_call(self, input_tokens: int, output_tokens: int, elapsed: float, **kw) -> None:
        """Thread-safe usage tracking."""
        with self._track_lock:
            super()._track_call(input_tokens, output_tokens, elapsed, **kw)

    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> List[str]:
        """Generate responses for multiple prompts concurrently via thread pool."""
        if not prompts:
            return []
        if len(prompts) == 1:
            return [self.generate(prompts[0], config, system_prompt)]

        results: List[str] = [""] * len(prompts)
        workers = min(self._max_workers, len(prompts))

        def _call(idx: int, prompt: str) -> tuple[int, str]:
            return idx, self.generate(prompt, config, system_prompt)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_call, i, p): i for i, p in enumerate(prompts)}
            for future in as_completed(futures):
                try:
                    idx, text = future.result()
                    results[idx] = text
                except Exception as e:
                    idx = futures[future]
                    logger.error(f"Batch call {idx} failed: {e}")
                    results[idx] = ""

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