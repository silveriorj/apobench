"""HuggingFace backend — efficient local inference with batching.

Ported from Projeto's HFWrapper with enhancements:
- Batch generation for evaluation efficiency
- Thinking mode support (/think and /no_think tags)
- Chat template formatting
- Automatic device/dtype selection
- Usage statistics tracking
"""
from __future__ import annotations

import gc
import logging
import re
import time
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pof.core.exceptions import LLMError
from pof.core.types import GenerationConfig
from pof.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class HuggingFaceLLM(BaseLLM):
    """Local HuggingFace model backend with efficient batching.

    Features:
    - Automatic device placement (CUDA/CPU)
    - Chat template formatting via tokenizer
    - Batch generation for parallel evaluation
    - Thinking mode support (strips thinking tags from output)
    - Memory-efficient with explicit cleanup
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: str = "auto",
        thinking_mode: bool = False,
        **kwargs: Any,
    ):
        super().__init__(model_name, **kwargs)
        self.thinking_mode = thinking_mode
        self._device = self._resolve_device(device)
        self._dtype = self._resolve_dtype(dtype)
        self._model = None
        self._tokenizer = None
        self._load_model()

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _resolve_dtype(self, dtype: str) -> torch.dtype:
        if dtype == "auto":
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    return torch.bfloat16
                return torch.float16
            return torch.float32
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(dtype, torch.float32)

    def _load_model(self) -> None:
        """Load model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.model_name} on {self._device} ({self._dtype})")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left",
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=self._dtype,
                device_map=self._device if self._device == "auto" else None,
                trust_remote_code=True,
            )
            if self._device != "auto":
                self._model = self._model.to(self._device)
            self._model.eval()
            logger.info(f"Model loaded successfully: {self.model_name}")
        except Exception as e:
            raise LLMError(f"Failed to load model {self.model_name}: {e}") from e

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate a single response using chat template."""
        config = config or GenerationConfig()
        messages = self._build_messages(prompt, system_prompt)
        input_text = self._apply_chat_template(messages)

        # Budget-aware max_new_tokens
        budget = self.get_budget()
        if budget is not None:
            input_tokens_sum = len(self._tokenizer.encode(input_text))
            eff_max = budget.plan_generation(input_tokens_sum, 1, config.max_new_tokens)
            config = GenerationConfig(
                max_new_tokens=eff_max,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                do_sample=config.do_sample,
                num_return_sequences=config.num_return_sequences,
                stop_sequences=list(config.stop_sequences),
                repetition_penalty=config.repetition_penalty,
            )

        start = time.time()
        output = self._generate_text(input_text, config)
        elapsed = time.time() - start

        input_tokens = len(self._tokenizer.encode(input_text))
        output_tokens = len(self._tokenizer.encode(output))
        tok_per_sec = output_tokens / elapsed if elapsed > 0 else 0
        logger.debug(f"[LLM] {elapsed:.1f}s | {output_tokens}tok out | {tok_per_sec:.0f} tok/s")
        self._track_call(input_tokens, output_tokens, elapsed)

        return self._clean_output(output)

    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None,
    ) -> List[str]:
        """Generate responses for multiple prompts with batched inference."""
        if not prompts:
            return []

        config = config or GenerationConfig()
        messages_list = [self._build_messages(p, system_prompt) for p in prompts]
        input_texts = [self._apply_chat_template(m) for m in messages_list]

        # Budget-aware max_new_tokens for batched generation
        budget = self.get_budget()
        if budget is not None:
            total_input = sum(len(self._tokenizer.encode(t)) for t in input_texts)
            prompts_in_call = len(input_texts)
            eff_max = budget.plan_generation(total_input, prompts_in_call, config.max_new_tokens)
            config = GenerationConfig(
                max_new_tokens=eff_max,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                do_sample=config.do_sample,
                num_return_sequences=config.num_return_sequences,
                stop_sequences=list(config.stop_sequences),
                repetition_penalty=config.repetition_penalty,
            )

        start = time.time()
        outputs = self._generate_batch_texts(input_texts, config)
        elapsed = time.time() - start

        total_input = sum(len(self._tokenizer.encode(t)) for t in input_texts)
        total_output = sum(len(self._tokenizer.encode(o)) for o in outputs)
        tok_per_sec = total_output / elapsed if elapsed > 0 else 0
        logger.debug(
            f"[LLM] {elapsed:.1f}s | batch={len(input_texts)}"
            f" ≈{total_output // max(len(outputs), 1)}tok/prompt out | {tok_per_sec:.0f} tok/s"
        )
        self._track_call(total_input, total_output, elapsed)

        return [self._clean_output(o) for o in outputs]

    def _build_messages(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Build chat messages list."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add thinking mode tags if enabled
        if self.thinking_mode:
            prompt = f"/think\n{prompt}"

        messages.append({"role": "user", "content": prompt})
        return messages

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Apply tokenizer's chat template.

        Passes enable_thinking so Qwen3 respects thinking_mode=False; retries
        without the kwarg for other models. Models whose templates reject the
        system role (e.g. Gemma-2 raises "System role not supported") get the
        system prompt folded into the first user message instead.
        """
        for msgs in (messages, self._fold_system_into_user(messages)):
            try:
                return self._tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.thinking_mode,
                )
            except TypeError:
                # Tokenizer doesn't support enable_thinking (non-Qwen3 model)
                try:
                    return self._tokenizer.apply_chat_template(
                        msgs,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    continue
            except Exception:
                continue
        # Plain-text fallback (no usable chat template at all)
        parts = []
        for msg in messages:
            parts.append(f"<|{msg['role']}|>\n{msg['content']}")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    @staticmethod
    def _fold_system_into_user(
        messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Merge a system message into the first user message.

        For chat templates that reject the system role (e.g. Gemma-2).
        """
        system = [m for m in messages if m["role"] == "system"]
        if not system:
            return messages
        rest = [dict(m) for m in messages if m["role"] != "system"]
        instructions = "\n".join(m["content"] for m in system)
        for m in rest:
            if m["role"] == "user":
                m["content"] = f"{instructions}\n\n{m['content']}"
                break
        return rest

    @torch.inference_mode()
    def _generate_text(self, input_text: str, config: GenerationConfig) -> str:
        """Generate text from a single input."""
        inputs = self._tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self._model.device)

        input_length = inputs["input_ids"].shape[1]

        gen_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "temperature": max(config.temperature, 0.01),
            "top_p": config.top_p,
            "top_k": config.top_k,
            "do_sample": config.do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
            "repetition_penalty": config.repetition_penalty,
        }

        # Greedy if temperature is very low
        if config.temperature < 0.01:
            gen_kwargs["do_sample"] = False
            gen_kwargs.pop("temperature", None)
            gen_kwargs.pop("top_p", None)
            gen_kwargs.pop("top_k", None)

        outputs = self._model.generate(**inputs, **gen_kwargs)
        generated = outputs[0][input_length:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    @torch.inference_mode()
    def _generate_batch_texts(
        self, input_texts: List[str], config: GenerationConfig
    ) -> List[str]:
        """Batch generation for efficiency."""
        inputs = self._tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(self._model.device)

        input_lengths = [
            (inputs["attention_mask"][i] == 1).sum().item()
            for i in range(len(input_texts))
        ]

        gen_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "temperature": max(config.temperature, 0.01),
            "top_p": config.top_p,
            "top_k": config.top_k,
            "do_sample": config.do_sample,
            "pad_token_id": self._tokenizer.pad_token_id,
            "repetition_penalty": config.repetition_penalty,
        }

        if config.temperature < 0.01:
            gen_kwargs["do_sample"] = False
            gen_kwargs.pop("temperature", None)
            gen_kwargs.pop("top_p", None)
            gen_kwargs.pop("top_k", None)

        outputs = self._model.generate(**inputs, **gen_kwargs)

        results = []
        for i, output in enumerate(outputs):
            # Skip input tokens (account for left-padding)
            generated = output[inputs["input_ids"].shape[1]:]
            text = self._tokenizer.decode(generated, skip_special_tokens=True)
            results.append(text)

        return results

    def _clean_output(self, text: str) -> str:
        """Strip <think>…</think> blocks unconditionally — they are never useful output."""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.strip()

    def cleanup(self) -> None:
        """Release GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()