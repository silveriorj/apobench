"""Backend-appropriate structured output enforcement.

Constrained decoding guarantees the *shape* of a response — valid JSON with
the required fields — at the decoding step, so a malformed emission is
impossible rather than merely detected. JSONSchemaBench
([arXiv:2501.10868](https://arxiv.org/html/2501.10868v1)) measures real
constraint engines as accuracy-neutral-to-positive (GSM8K 80.1 → 83.8) and
*faster* than unconstrained decoding, so the usual cost objection does not
apply. The widely-cited "structure hurts reasoning" result
([arXiv:2408.02442](https://arxiv.org/abs/2408.02442)) tested JSON-mode
prompting rather than a constraint engine, and does not transfer.

Three tiers, chosen by what the backend actually supports:

1. `XGrammarDecoder`  — local HuggingFace models, via a `LogitsProcessor`
   that drops into the existing `gen_kwargs`, preserving batching.
2. `ResponseFormatDecoder` — OpenAI-compatible APIs (SAP AI Core, Gemini),
   via native `response_format={"type": "json_schema", ...}`.
3. `NullDecoder` — no enforcement; the caller still validates and retries.
   This is a real fallback, not a stub: the contract's semantic checks
   (anchor spans, protected regions) are Python-side regardless, so the
   method degrades in reliability rather than breaking.

Selecting tier 3 is also how the value of tiers 1-2 gets measured: running
the same optimizer with and without enforcement gives the compliance delta
directly.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class StructuredDecoder:
    """Base: no enforcement."""

    name = "none"

    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema

    @property
    def available(self) -> bool:
        return False

    def logits_processors(self, batch_size: int = 1) -> Optional[List[Any]]:
        """Per-call logits processors for a local model, or None."""
        return None

    def request_kwargs(self) -> Dict[str, Any]:
        """Extra kwargs for an OpenAI-compatible chat call."""
        return {}


class NullDecoder(StructuredDecoder):
    """Explicit no-op, used as the control arm."""

    name = "none"


class XGrammarDecoder(StructuredDecoder):
    """Grammar-constrained decoding for local HuggingFace models.

    The compiled grammar is cached on the instance because compilation is
    the expensive step; the per-generation `LogitsProcessor` is cheap and
    MUST be rebuilt per call, since it carries the matcher's position state
    and reusing one across calls would leave it mid-parse.
    """

    name = "xgrammar"

    def __init__(self, schema: Dict[str, Any], tokenizer: Any, vocab_size: int):
        super().__init__(schema)
        self._compiled = None
        self._xgr = None
        try:
            import xgrammar as xgr

            info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
            compiler = xgr.GrammarCompiler(info)
            self._compiled = compiler.compile_json_schema(schema)
            self._xgr = xgr
            self._probe_apply_path()
            logger.info("[structured] xgrammar enforcement active")
        except ImportError as e:
            # Either xgrammar itself, or the Triton kernel it loads lazily
            # for the CUDA bitmask path. Distinguish them: "install
            # xgrammar" is unhelpful advice when xgrammar is present and
            # Triton is what is missing (common on Windows).
            self._compiled = None
            missing = "xgrammar" if "xgrammar" in str(e).lower() else str(e)
            logger.warning(
                f"[structured] constrained decoding unavailable ({missing}); "
                f"falling back to validate-and-retry only"
            )
        except Exception as e:  # malformed schema, tokenizer mismatch, ...
            self._compiled = None
            logger.warning(f"[structured] xgrammar unavailable ({e}); "
                           f"falling back to validate-and-retry only")

    def _probe_apply_path(self) -> None:
        """Exercise the bitmask kernel now, not mid-generation.

        `apply_token_bitmask_inplace` imports a Triton kernel lazily on first
        CUDA call, so a missing Triton surfaces as an ImportError *during*
        generation -- which on a multi-hour run means losing the run rather
        than degrading. Probing here converts that into a clean fallback at
        construction time.
        """
        import torch
        import xgrammar as xgr

        device = "cuda" if torch.cuda.is_available() else "cpu"
        vocab = self._compiled.tokenizer_info.vocab_size
        matcher = xgr.GrammarMatcher(self._compiled)
        bitmask = xgr.allocate_token_bitmask(1, vocab)
        matcher.fill_next_token_bitmask(bitmask)
        scores = torch.zeros(1, vocab, dtype=torch.float32, device=device)
        xgr.apply_token_bitmask_inplace(scores, bitmask.to(scores.device))

    @property
    def available(self) -> bool:
        return self._compiled is not None

    def logits_processors(self, batch_size: int = 1) -> Optional[List[Any]]:
        if not self.available:
            return None
        try:
            from xgrammar.contrib.hf import LogitsProcessor

            return [LogitsProcessor(self._compiled)]
        except Exception as e:
            logger.warning(f"[structured] could not build logits processor ({e})")
            return None


class ResponseFormatDecoder(StructuredDecoder):
    """Schema enforcement for OpenAI-compatible endpoints."""

    name = "response_format"

    @property
    def available(self) -> bool:
        return True

    def request_kwargs(self) -> Dict[str, Any]:
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "pact_contract",
                    "strict": True,
                    "schema": self.schema,
                },
            }
        }


def make_decoder(llm: Any, schema: Dict[str, Any], enabled: bool = True) -> StructuredDecoder:
    """Pick the strongest enforcement this backend supports.

    Never raises: an optimizer that cannot constrain decoding should still
    run, because the Python-side contract validation is what actually
    guarantees correctness — constraint only reduces how often it has to
    reject.
    """
    if not enabled:
        return NullDecoder(schema)
    tokenizer = getattr(llm, "_tokenizer", None)
    model = getattr(llm, "_model", None)
    if tokenizer is not None and model is not None:
        vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
        if vocab_size:
            dec = XGrammarDecoder(schema, tokenizer, vocab_size)
            if dec.available:
                return dec
            return NullDecoder(schema)
    if getattr(llm, "_client", None) is not None:
        return ResponseFormatDecoder(schema)
    return NullDecoder(schema)
