"""Prompt loader — fetch and cache initial prompts from GitHub repositories.

Supports:
- BBH prompts from Joschka/big_bench_hard (HF dataset; same source used by
  the UK AISI's inspect_evals BBH implementation, see fetch_bbh_prompt_v2)
- GSM8K prompts from chain-of-thought-hub
- Local file loading
- Caching to avoid repeated downloads
"""
from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = Path(".cache/prompts")

# BBH prompt source (chain-of-thought-hub) -- get_seed_prompt() now sources
# BBH from Joschka/big_bench_hard instead (see fetch_bbh_prompt_v2), but
# fetch_bbh_prompt/extract_instruction below are still used directly by
# apex_lean.py, funnel_v4b.py, and experiments/bbh_reference_baseline.py.
BBH_PROMPT_BASE_URL = (
    "https://raw.githubusercontent.com/FranxYao/chain-of-thought-hub/main/BBH/lib_prompt"
)

# BBH prompt source used by get_seed_prompt: Joschka/big_bench_hard, a
# curated HF dataset with pre-separated answer_only_prompt and
# chain_of_thought_prompt fields per task -- this is the exact dataset the
# UK AISI's inspect_evals BBH implementation uses (see
# https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/bbh/bbh.py),
# so adopting it lets our seed prompts cite a maintained, versioned source
# instead of scraping raw .txt files from chain-of-thought-hub and
# reimplementing our own (fragile) first-line instruction extraction.
BBH_HF_DATASET_PATH = "Joschka/big_bench_hard"
BBH_HF_DATASET_REVISION = "76eaa8c29ad448752cd44201a1246618e2454cac"

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


def fetch_bbh_prompt_v2(task: str, cache: bool = True) -> dict:
    """Fetch both the answer-only and chain-of-thought seed prompts for a
    BBH task from Joschka/big_bench_hard's `few_shot_prompts` config.

    Unlike the retired fetch_bbh_prompt() (chain-of-thought-hub .txt scrape
    + first-line heuristic for the answer-only variant), this dataset
    curates both prompt variants directly per task -- no string-slicing
    guesswork needed to get a real answer-only prompt.

    Returns:
        Dict with 'answer_only_prompt' and 'chain_of_thought_prompt' keys.
    """
    cache_path = CACHE_DIR / "bbh_v2" / f"{task}.json"
    if cache and cache_path.exists():
        logger.debug(f"Loading cached BBH v2 prompt for '{task}'")
        return json.loads(cache_path.read_text(encoding="utf-8"))

    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise RuntimeError(
            "'datasets' package required. Run: pip install datasets"
        )

    logger.info(
        f"Fetching BBH v2 prompts for '{task}' from {BBH_HF_DATASET_PATH} "
        "(few_shot_prompts config)"
    )
    try:
        prompts_dataset = hf_load_dataset(
            BBH_HF_DATASET_PATH,
            "few_shot_prompts",
            split="few_shot_prompts",
            revision=BBH_HF_DATASET_REVISION,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch BBH few_shot_prompts from {BBH_HF_DATASET_PATH}: {e}"
        ) from e

    row = next((r for r in prompts_dataset if r["dataset_name"] == task), None)
    if row is None:
        raise ValueError(
            f"No prompts found for BBH task '{task}' in {BBH_HF_DATASET_PATH}"
        )
    result = {
        "answer_only_prompt": row["answer_only_prompt"],
        "chain_of_thought_prompt": row["chain_of_thought_prompt"],
    }

    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result), encoding="utf-8")
        logger.debug(f"Cached BBH v2 prompt to {cache_path}")

    return result


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
        prompts = fetch_bbh_prompt_v2(task, cache=cache)
        if use_full_prompt:
            return prompts["chain_of_thought_prompt"]
        return prompts["answer_only_prompt"]

    elif dataset.lower() == "gsm8k":
        return (
            "Solve the following math problem step by step. "
            "Use brief one-line calculations. "
            "End with \"The answer is N\" where N is your final answer."
        )

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

    elif dataset.lower() in ("livebench_coding", "livebench/coding", "livecodebench"):
        return (
            "Write a correct Python solution to the following programming "
            "problem. If a class/method signature is given, complete it "
            "exactly as specified. Include only the code in your response."
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset}")