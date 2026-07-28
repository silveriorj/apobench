"""Ollama backend — native /api/chat, with control over thinking mode.

Ollama also exposes an OpenAI-compatible endpoint (`OpenAILLM` with
`base_url` pointed at it works), but that endpoint does not forward Ollama's
`think` field -- a reasoning model (e.g. Qwen3.5) keeps emitting a full
thinking trace regardless, which burns the entire `max_new_tokens` budget on
reasoning and leaves nothing for the answer under this project's short
answer-only eval budgets (32-64 tokens). This backend hits Ollama's native
API directly so `thinking_mode=False` (the project default, same field the
HuggingFace backend uses for Qwen3's `enable_thinking`) actually disables it:
confirmed by direct testing, `think: false` returns the answer alone in 2
output tokens instead of truncating mid-reasoning-trace at 32.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from pof.core.exceptions import LLMError
from pof.core.types import GenerationConfig
from pof.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class OllamaLLM(BaseLLM):
    """Ollama native-API backend with thinking-mode control.

    Args:
        model_name: Ollama model tag (e.g. "qwen3.5:9b").
        base_url: Ollama server URL, e.g. "http://127.0.0.1:11435" (no
            trailing /v1 -- that's the OpenAI-compat path, this uses /api/chat).
        thinking_mode: Passed through as Ollama's `think` field. False (the
            project default) disables the reasoning trace entirely.
        max_workers: Concurrent requests for generate_batch. Ollama serves
            one request per model instance by default (OLLAMA_NUM_PARALLEL=1
            unless raised server-side), so this mostly overlaps request
            queueing/network latency rather than achieving true parallelism —
            still worth it over strictly serial calls.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://127.0.0.1:11434",
        thinking_mode: bool = False,
        max_workers: int = 4,
        timeout: float = 300.0,
        **kwargs: Any,
    ):
        super().__init__(model_name, **kwargs)
        self.base_url = base_url.rstrip("/")
        self.thinking_mode = thinking_mode
        self._max_workers = max_workers
        self._timeout = timeout
        self._track_lock = threading.Lock()

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        config = config or GenerationConfig()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "think": self.thinking_mode,
            "options": {
                "num_predict": config.max_new_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
            },
        }

        start = time.time()
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise LLMError(f"Ollama API call failed: {e}") from e
        elapsed = time.time() - start

        text = (data.get("message") or {}).get("content", "") or ""
        input_tokens = data.get("prompt_eval_count", 0) or 0
        output_tokens = data.get("eval_count", 0) or 0
        self._track_call(input_tokens, output_tokens, elapsed)
        return text

    def _track_call(self, input_tokens: int, output_tokens: int, elapsed: float, **kw) -> None:
        with self._track_lock:
            super()._track_call(input_tokens, output_tokens, elapsed, **kw)

    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> List[str]:
        if not prompts:
            return []
        if len(prompts) == 1:
            return [self.generate(prompts[0], config, system_prompt)]

        results: List[str] = [""] * len(prompts)
        workers = min(self._max_workers, len(prompts))

        def _call(idx: int, prompt: str) -> tuple:
            return idx, self.generate(prompt, config, system_prompt)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_call, i, p): i for i, p in enumerate(prompts)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    idx, text = future.result()
                    results[idx] = text
                except Exception as e:
                    logger.error(f"Batch call {idx} failed: {e}")
                    results[idx] = ""

        return results
