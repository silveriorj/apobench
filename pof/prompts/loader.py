"""Prompt loader — fetch and cache initial prompts from GitHub repositories.

Supports:
- BBH prompts from EvoPrompt repo (per-task .txt files)
- GSM8K prompts from chain-of-thought-hub
- Local file loading
- Caching to avoid repeated downloads
"""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = Path(".cache/prompts")

# BBH prompt source (EvoPrompt repo)
BBH_PROMPT_BASE_URL = (
    "https://raw.githubusercontent.com/beeevita/EvoPrompt/main/BBH/lib_prompt"
)

# GSM8K prompt source (chain-of-thought-hub) — mid CoT: "Let's think step by step" + "The answer is N"
GSM8K_PROMPT_URL = (
    "https://raw.githubusercontent.com/FranxYao/chain-of-thought-hub/main/"
    "gsm8k/lib_prompt/prompt_mid.txt"
)

# BBH tasks available
BBH_TASKS_WITH_PROMPTS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]


def fetch_bbh_prompt(task: str, cache: bool = True) -> str:
    """Fetch the initial prompt for a BBH task from EvoPrompt repo.

    The prompt file contains:
    - Line 1: The task instruction (seed prompt)
    - Remaining: Few-shot CoT examples

    Args:
        task: BBH task name (e.g., 'dyck_languages').
        cache: Whether to cache the downloaded file locally.

    Returns:
        Full prompt text (instruction + few-shot examples).
    """
    # Check cache first
    cache_path = CACHE_DIR / "bbh" / f"{task}.txt"
    if cache and cache_path.exists():
        logger.debug(f"Loading cached BBH prompt for '{task}'")
        return cache_path.read_text(encoding="utf-8")

    # Download
    url = f"{BBH_PROMPT_BASE_URL}/{task}.txt"
    logger.info(f"Fetching BBH prompt for '{task}' from {url}")

    try:
        response = urllib.request.urlopen(url, timeout=30)
        content = response.read().decode("utf-8")
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch BBH prompt for '{task}' from {url}: {e}"
        ) from e

    # Cache
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(content, encoding="utf-8")
        logger.debug(f"Cached BBH prompt to {cache_path}")

    return content


def fetch_gsm8k_prompt(cache: bool = True) -> str:
    """Fetch the GSM8K few-shot prompt from chain-of-thought-hub.

    Returns:
        Full prompt text (mid CoT: step-by-step + "The answer is N").
    """
    cache_path = CACHE_DIR / "gsm8k" / "prompt_mid.txt"
    if cache and cache_path.exists():
        logger.debug("Loading cached GSM8K prompt")
        return cache_path.read_text(encoding="utf-8")

    logger.info(f"Fetching GSM8K prompt from {GSM8K_PROMPT_URL}")

    try:
        response = urllib.request.urlopen(GSM8K_PROMPT_URL, timeout=30)
        content = response.read().decode("utf-8")
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch GSM8K prompt from {GSM8K_PROMPT_URL}: {e}"
        ) from e

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(content, encoding="utf-8")

    return content


def extract_instruction(prompt_text: str) -> str:
    """Extract just the instruction (first line) from a full prompt file.

    BBH prompt files have the instruction as the first line,
    followed by few-shot examples.
    """
    lines = prompt_text.strip().split("\n")
    if lines:
        return lines[0].strip()
    return ""


def get_seed_prompt(
    dataset: str,
    task: str = "",
    use_full_prompt: bool = True,
    cache: bool = True,
) -> str:
    """Get the seed prompt for a given dataset/task combination.

    Args:
        dataset: Dataset name ('bbh', 'gsm8k', 'humaneval').
        task: Task name (required for BBH).
        use_full_prompt: If True, return full prompt with examples.
            If False, return only the instruction line.
        cache: Whether to cache downloads.

    Returns:
        Seed prompt string.
    """
    if dataset.lower() == "bbh":
        if not task:
            raise ValueError("Task name required for BBH dataset")
        content = fetch_bbh_prompt(task, cache=cache)
        if use_full_prompt:
            return content
        return extract_instruction(content)

    elif dataset.lower() == "gsm8k":
        content = fetch_gsm8k_prompt(cache=cache)
        instruction = (
            "Solve the following math problem step by step. "
            "Let's think step by step. "
            "Show your work clearly and provide the final numeric answer."
        )
        if use_full_prompt:
            return content.rstrip() + "\n\n" + instruction
        return instruction

    elif dataset.lower() in ("livebench_math", "livebench/math"):
        return (
            "Solve the following math problem. "
            "Write 'The answer is ' followed by the final answer only."
        )

    elif dataset.lower() == "humaneval":
        return (
            "Write a correct Python function that solves the following "
            "programming problem. Include only the function definition "
            "in your response."
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset}")