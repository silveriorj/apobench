"""Google Gemini backend — LOCAL ONLY, intentionally not committed.

Gemini exposes an OpenAI-compatible `/chat/completions` endpoint, so this
subclasses `OpenAILLM` with only the base URL and auth plumbing changed.

Credentials are read from the environment or a JSON config file:

    GOOGLE_API_KEY=<your key>

or a JSON file with its path in:

    GEMINI_CONFIG=/path/to/gemini.json

with contents: {"api_key": "AIza..."}

The file is also searched at the project root and ~/Documents/gemini.json.

Usage:

    import pof.llm.gemini_backend as gemini; gemini.register()
    # then any config with `backend: gemini` works normally

`register()` patches the factory in-process (same approach as sap_aicore_backend)
so this file is the only thing that has to exist outside the tracked tree.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from pof.llm.openai_backend import OpenAILLM

logger = logging.getLogger(__name__)

_BACKEND_ALIASES = {"gemini", "google", "google_gemini"}
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

_CONFIG_CANDIDATES = [
    Path(__file__).parent.parent.parent / "gemini.json",
    Path.home() / "Documents" / "gemini.json",
    Path.home() / "gemini.json",
]


def _resolve_api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if key:
        return key

    config_path = os.environ.get("GEMINI_CONFIG")
    candidates = [Path(config_path)] + _CONFIG_CANDIDATES if config_path else _CONFIG_CANDIDATES
    for path in candidates:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                k = data.get("api_key") or data.get("GOOGLE_API_KEY") or data.get("GEMINI_API_KEY")
                if k:
                    logger.info("Gemini: loaded API key from %s", path)
                    return k
            except Exception:
                pass

    raise RuntimeError(
        "Gemini API key not found. Set GOOGLE_API_KEY env var or create "
        "gemini.json with {\"api_key\": \"AIza...\"} in the project root or "
        "~/Documents/."
    )


class GeminiLLM(OpenAILLM):
    """OpenAI-compatible Gemini backend."""

    def __init__(self, model_name: str = "gemini-2.0-flash", **kwargs: Any) -> None:
        api_key = _resolve_api_key()
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            base_url=_GEMINI_BASE_URL,
            **kwargs,
        )
        logger.info("GeminiLLM initialised: model=%s", model_name)


def register() -> None:
    """Teach `create_llm` about `backend: gemini`, in-process.

    Also rebinds the name inside `pof.orchestration.runner` (same pattern as
    sap_aicore_backend.register) so the patched factory is seen everywhere.
    """
    from pof.llm import factory

    if getattr(factory, "_gemini_registered", False):
        return
    original = factory.create_llm

    def create_llm(config=None, **kwargs):  # type: ignore[no-untyped-def]
        backend = (getattr(config, "backend", "") or "").lower()
        if config is not None and backend in _BACKEND_ALIASES:
            out: Dict[str, Any] = {"model_name": config.model_name}
            if getattr(config, "max_workers", None):
                out["max_workers"] = config.max_workers
            return GeminiLLM(**out)
        return original(config, **kwargs)

    factory.create_llm = create_llm
    factory._gemini_registered = True

    runner = sys.modules.get("pof.orchestration.runner")
    if runner is not None and hasattr(runner, "create_llm"):
        runner.create_llm = create_llm

    logger.info("[gemini] backend registered (aliases: %s)", sorted(_BACKEND_ALIASES))
