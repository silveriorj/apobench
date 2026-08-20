"""Experiment runner: SWIFT & APEX on BBH, SVAMP, HumanEval.

Iterates over the experiment matrix defined in the config, fetching seed prompts
from the appropriate repositories and running each method/task combination.

Usage:
    python experiments/run_swift_apex.py
    python experiments/run_swift_apex.py --methods swift
    python experiments/run_swift_apex.py --tasks dyck_languages formal_fallacies
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Windows' console defaults to a legacy codepage (cp1252/cp437), not UTF-8 --
# the checkmark/cross markers below (U+2713/U+2717) then either crash the
# StreamHandler or get silently mangled into literal "✓" escape text in
# logs (observed running this script under PowerShell). Reconfiguring stdout/
# stderr to UTF-8 before logging is configured fixes both write paths at the
# source; unaffected on Linux/macOS, where the default is already UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pof.config.loader import load_config
from pof.config.schemas import RunConfig
from pof.prompts.loader import get_seed_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# EXPERIMENT MATRIX
# =============================================================================

GENERIC_PROMPT = "Solve the following problem correctly."

# Minimal universal format cue -- unlike the per-task_type system prompts
# (_THINKING_EVAL_SYSTEM_PROMPT's \boxed{X}, _DYCK_EVAL_SYSTEM_PROMPT's stack
# notation, etc.), this is ONE short instruction applied uniformly across
# every task, deliberately not tailored per-task by this project -- but it
# borrows Qwen's own documented eval conventions (Qwen2.5 technical report /
# eval harness recommendations) rather than inventing an arbitrary format:
# math -> step-by-step reasoning ending in \boxed{}, MCQ -> answer with the
# option letter directly. Falls back to "Answer: X" for anything that's
# neither, which pof/evaluation/scoring.py's _extract_cot_answer already
# recognizes (after \boxed{} and "the answer is X"), so nothing here is a
# made-up format the scorer can't parse.
SIMPLE_SYSTEM_PROMPT = (
    "Solve the following problem correctly. "
    "If it is a math problem, reason step by step and put your final answer "
    "within \\boxed{}. "
    "If it provides multiple-choice options, answer with the option's "
    "letter directly. "
    "Otherwise, end your response with the final line: Answer: X"
)

METHODS = ["swift", "apex", "capo", "gaapo", "see", "gepa",
           "swift_v2", "apex_v2", "funnel", "funnel_v2", "funnel_v3",
           "funnel_v4a", "funnel_v4b", "funnel_v4c", "funnel_v4d",
           "funnel_lean", "funnel_v5", "funnel_wide",
           "funnel_v6", "funnel_indexed", "funnel_v7", "funnel_prime",
           "apex_lean", "apex_holdout", "baseline_seed"]  # Methods to run

# Random seeds for statistical robustness (3 runs per configuration)
SEEDS = [42, 123, 7]

# Models served through Ollama instead of the HuggingFace/transformers stack.
# base_url points at the user-local Ollama instance on t101 (port 11435,
# separate from the stale system install on 11434). thinking_mode=False routes
# through the native-API OllamaLLM backend, which is the only path that
# actually disables a reasoning model's thinking trace -- Ollama's
# OpenAI-compatible endpoint does not forward the `think` field, and at this
# project's short answer-only eval budgets (32-64 tokens) a reasoning model
# left thinking burns the whole budget and returns an empty answer.
OLLAMA_MODELS: Dict[str, Dict[str, Any]] = {
    "qwen3.5:4b": {"base_url": "http://127.0.0.1:11435", "thinking_mode": False},
    "qwen3.5:9b": {"base_url": "http://127.0.0.1:11435", "thinking_mode": False},
    "qwen3.5:27b": {"base_url": "http://127.0.0.1:11435", "thinking_mode": False},
}

# Gemini models routed through the existing OpenAILLM backend via Google's
# OpenAI-compatible endpoint -- no new backend needed, just base_url + key.
# API key is read from GEMINI_API_KEY at run time, never hardcoded or logged.
# No local GPU/eval-batch-size tuning applies here (OpenAILLM.generate_batch
# threads its own concurrency, capped by max_workers below); the per-request
# rate limit governs throughput, not GPU memory.
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# max_workers=64, 5 PARALLEL PROCESSES: verified clean 2026-08-03 (~1.19K/4K
# RPM, zero 429s over ~15min sustained). Attempts to scale further --
# max_workers=64 x 12 processes, then max_workers=16 x 12 processes -- both
# produced persistent 429 storms, not just a transient launch burst. 12
# simultaneous processes appears to be past whatever the real effective
# ceiling is regardless of per-process throttling; reverted to the last
# configuration that ran cleanly. If revisiting, scale the PROCESS COUNT
# cautiously (try 6-8) rather than assuming linear headroom from the 4K RPM
# dashboard number, which does not seem to reflect the actually-enforced
# limit for this project/key.
GEMINI_MODELS: Dict[str, Dict[str, Any]] = {
    "gemini-2.0-flash": {
        "backend": "openai",
        "base_url": _GEMINI_BASE_URL,
        "api_key": os.environ.get("GEMINI_API_KEY"),
        "max_workers": 64,
    },
    "gemini-1.5-flash": {
        "backend": "openai",
        "base_url": _GEMINI_BASE_URL,
        "api_key": os.environ.get("GEMINI_API_KEY"),
        "max_workers": 64,
    },
    "gemini-1.5-pro": {
        "backend": "openai",
        "base_url": _GEMINI_BASE_URL,
        "api_key": os.environ.get("GEMINI_API_KEY"),
        "max_workers": 64,
    },
    # A "thinking" model like the Ollama qwen3.5 entries above -- burns part
    # of max_tokens on hidden reasoning before any visible answer. Verified
    # working on a fresh no-billing project (2026-08-03): gemini-2.0-flash's
    # free tier is 0 on that project, but 2.5-flash has real free quota.
    # Only run this with --cot (1536-token budget) -- the 32-64 token
    # answer-only budgets get consumed entirely by thinking tokens, same
    # empty-answer failure mode a non-thinking-disabled reasoning model
    # hits elsewhere in this harness.
    "gemini-2.5-flash": {
        "backend": "openai",
        "base_url": _GEMINI_BASE_URL,
        "api_key": os.environ.get("GEMINI_API_KEY"),
        "max_workers": 64,
    },
    # Separate free-tier daily quota bucket from "gemini-2.5-flash" (tracked
    # by literal model-name string on Google's side, not by underlying
    # model) -- verified 2026-08-03 after 2.5-flash's 20-request/day quota
    # was exhausted, this alias still succeeded. Same thinking-model
    # constraint: --cot only.
    "gemini-flash-latest": {
        "backend": "openai",
        "base_url": _GEMINI_BASE_URL,
        "api_key": os.environ.get("GEMINI_API_KEY"),
        "max_workers": 64,
    },
    # "gemini-3.1-flash-live-preview" (a real model on this key, confirmed via
    # models.list()) is NOT usable here -- Live models only support Google's
    # separate bidiGenerateContent WebSocket protocol for realtime audio/
    # video, not the generateContent/chat-completions path OpenAILLM uses.
    # These non-Live gemini-3.x models work fine and each has its own
    # separate free-tier daily quota bucket, same as the 2.5/flash-latest
    # models above -- verified 2026-08-03.
    "gemini-3-flash-preview": {
        "backend": "openai",
        "base_url": _GEMINI_BASE_URL,
        "api_key": os.environ.get("GEMINI_API_KEY"),
        "max_workers": 64,
    },
    "gemini-3.5-flash": {
        "backend": "openai",
        "base_url": _GEMINI_BASE_URL,
        "api_key": os.environ.get("GEMINI_API_KEY"),
        "max_workers": 64,
    },
    "gemini-3.6-flash": {
        "backend": "openai",
        "base_url": _GEMINI_BASE_URL,
        "api_key": os.environ.get("GEMINI_API_KEY"),
        "max_workers": 64,
    },
    # "gemini-2.5-flash-lite" is listed by models.list() but 404s on every
    # actual call: "no longer available to new users" -- Google keeps the
    # ID around for existing callers but blocks it for new API keys.
    # gemini-flash-lite-latest is the closest live equivalent (Google's
    # rolling "latest lite" alias). Verified working 2026-08-03, each with
    # its own separate free-tier daily quota bucket.
    "gemini-flash-lite-latest": {
        "backend": "openai",
        "base_url": _GEMINI_BASE_URL,
        "api_key": os.environ.get("GEMINI_API_KEY"),
        "max_workers": 64,
    },
    "gemini-3.1-flash-lite": {
        "backend": "openai",
        "base_url": _GEMINI_BASE_URL,
        "api_key": os.environ.get("GEMINI_API_KEY"),
        "max_workers": 64,
    },
    "gemini-3.5-flash-lite": {
        "backend": "openai",
        "base_url": _GEMINI_BASE_URL,
        "api_key": os.environ.get("GEMINI_API_KEY"),
        "max_workers": 64,
    },
}

# Per-task eval max_new_tokens (eval output only; operator/LLM generation uses
# llm.max_new_tokens from the YAML, which stays at 512).
# dyck previously used _DYCK_EVAL_SYSTEM_PROMPT (CoT allowed, 1024 tokens) --
# switched to answer-only like every other BBH task: at a 2h/task budget, the
# CoT variant costs ~6h for a full 3-seed run on a 9B Ollama model (no request
# batching there, -np 1). Full CoT for dyck is deferred to the separate
# "thinking" task_type batch (see evaluator.py's _THINKING_EVAL_SYSTEM_PROMPT)
# instead of being mixed into this answer-only exploration pass.
# MCQ/boolean/color tasks: single word/letter → 16 is sufficient.
# SVAMP/HumanEval use different system prompts that allow CoT/code output.
EVAL_MAX_NEW_TOKENS: Dict[str, int] = {
    # BBH — 32-64 tokens: sufficient for JSON {"answer": "X"} with no CoT.
    "dyck_languages": 64,
    "boolean_expressions": 32,
    "causal_judgement": 32,
    "disambiguation_qa": 32,
    "formal_fallacies": 32,
    "hyperbaton": 32,
    "logical_deduction_five_objects": 32,
    "penguins_in_a_table": 32,  # kept for reference; not in the default matrix
    "reasoning_about_colored_objects": 32,
    "web_of_lies": 32,
    # word_sorting answers are the full re-ordered word list (up to 20 words /
    # ~180 chars observed), not a single letter/word -- 32 truncates it almost
    # every time (measured: 3 completed runs at 32 tokens scored 0.21-0.28,
    # consistent with truncation since this task scores by exact match).
    "word_sorting": 192,
    # Other datasets
    "svamp": 256,  # 1-2 step arithmetic CoT: much shorter than GSM8K's 512
    "gsm8k": 512,  # kept for reference/opt-in; not in the default DATASETS matrix
    "humaneval": 768,  # empirically: truncated completions are never correct (0/28 at 1024 cap);
                       # 768 loses only 5/1725 correct Qwen3-4B answers vs 1024, 0 for Llama/Gemma
    "livebench_coding": 1024,  # LiveCodeBench problems (competitive programming) run longer/
                                # more complex than HumanEval on average -- no empirical tuning yet.
    "livebench_math": 64,   # numeric/symbolic answers; same ceiling as math eval configs.
}

# Default fallback when task is not listed above
_DEFAULT_EVAL_MAX_NEW_TOKENS = 32

# CoT/"thinking" mode (--cot flag): full step-by-step reasoning ending in
# \boxed{}, via the "thinking" task_type built in evaluator.py. Per stored
# guidance: generous headroom, well above the 1024 dyck previously used --
# reasoning length before \boxed{} varies a lot and truncation silently kills
# an otherwise-correct answer.
#
# Raised 1536 -> 2048 (2026-08-18) for margin on the longest reasoning
# chains. It is a cap, not a target: a chain that reaches \boxed{} at 400
# tokens costs 400 either way, so the extra headroom is only ever paid when
# it is actually needed -- which is exactly the case it exists for.
# HuggingFaceLLM now counts generations that consume the full budget, so
# whether this bound ever binds is measured rather than assumed.
COT_MAX_NEW_TOKENS = 2048

# Brief CoT (--cot-brief flag): "cot" task_type -- one-line-per-step
# reasoning, no prose, ending in "So the answer is X" (see
# _COT_EVAL_SYSTEM_PROMPT in evaluator.py, originally built for GSM8K, never
# wired in here). The hypothesis this exists to check: does most of full
# CoT's accuracy gain survive on a much shorter, much faster generation --
# the same motivation as capping thinking to 64 tokens for word_sorting, just
# applied to reasoning instead of the final answer. 256 tokens is generous
# for one-line-per-step on BBH-scale problems (rarely more than ~15 steps).
COT_BRIEF_MAX_NEW_TOKENS = 256

# Per-task eval batch size.
# Originally calibrated for DeepSeek-Coder-7B/8B-class models on 20 GB (~14-16 GB
# weights, ~4-5 GB headroom). Qwen3-4B-Instruct-2507 weighs ~8 GB in bf16 --
# a single CoT run (batch=2) measured at ~9.5 GB used of 20 GB total, so ~10 GB
# of headroom was going unused. Bumped 3-4x for this model; still leaves a
# buffer for KV growth on the longer-context tasks (svamp/humaneval).
EVAL_BATCH_SIZE: Dict[str, int] = {
    # BBH — batch=8 for Qwen3-4B (was 2, calibrated for 8B-class models).
    "dyck_languages": 8,
    "boolean_expressions": 8,
    "causal_judgement": 8,
    "disambiguation_qa": 8,
    "formal_fallacies": 8,
    "hyperbaton": 8,
    "logical_deduction_five_objects": 8,
    "penguins_in_a_table": 8,  # kept for reference; not in the default matrix
    "reasoning_about_colored_objects": 8,
    "web_of_lies": 8,
    # SVAMP — short 1-2 step CoT, seq well under GSM8K's ~900 tok
    "svamp": 6,
    # GSM8K — seq ~900 tok, kept for reference; not in default matrix
    "gsm8k": 4,
    # HumanEval — seq ~1500 tok, longest generations of the set
    "humaneval": 3,
}

_DEFAULT_EVAL_BATCH_SIZE = 8

# Per-task wall-clock budget (seconds).
# SVAMP and HumanEval get 7200s so that slower methods (GAAPO, SEE) hit the cap
# and their scores are recorded as lower bounds, making SWIFT/APEX cuts visible.
EVAL_TIME_BUDGET: Dict[str, int] = {
    "svamp": 7200,
    "gsm8k": 7200,  # kept for reference; not in default matrix
    "humaneval": 7200,
}
_DEFAULT_EVAL_TIME_BUDGET = 7200  # BBH tasks

# Per-task evaluator task_type override.
# Empty string means auto-detect from dataset samples (default for most BBH tasks).
# dyck_languages previously mapped to "dyck" (_DYCK_EVAL_SYSTEM_PROMPT, brief
# CoT + bracket answer) -- now runs answer-only like every other task; see the
# EVAL_MAX_NEW_TOKENS note above.
EVAL_TASK_TYPE: Dict[str, str] = {}

# Dataset configurations
DATASETS = {
    "bbh": {
        "tasks": [
            "boolean_expressions",
            "causal_judgement",
            "disambiguation_qa",
            "formal_fallacies",
            "hyperbaton",
            "logical_deduction_five_objects",
            "reasoning_about_colored_objects",
            "sports_understanding",
        ],
        "task_type": "auto",
    },
    "svamp": {
        "tasks": [""],  # Single split (test set, 300 problems)
        "task_type": "math",
    },
    "humaneval": {
        "tasks": [""],  # Single task
        "task_type": "text",
    },
    "livebench_coding": {
        "tasks": [""],  # Single task (LiveCodeBench-sourced problems)
        "task_type": "text",
    },
    "livebench_math": {
        "tasks": [""],  # Single task (LiveBench math problems)
        "task_type": "math",
    },
}


def _model_slug(model_name: str) -> str:
    """Filesystem-safe short name for a model, e.g. Qwen/Qwen3-0.6B → qwen3-0.6b.

    Ollama tags use "name:tag" (e.g. "qwen3.5:9b") -- ":" is invalid in
    Windows paths, and results get synced there, so it's replaced too.
    """
    return model_name.split("/")[-1].lower().replace(":", "-")


def build_run_config(
    base_config_path: str,
    method: str,
    dataset: str,
    task: str,
    seed_prompt: str,
    seed: int = 42,
    model_name: Optional[str] = None,
    output_root: str = "outputs/swift_apex_benchmark",
    cot_mode: str = "",
    dev_test_split: float = 0.0,
    strip_system_prompt: bool = False,
    simple_system_prompt: bool = False,
) -> RunConfig:
    """Build a RunConfig for a specific method/dataset/task/model combination.

    cot_mode: "" (answer-only, per-task defaults), "full" (free-form
        reasoning ending in \\boxed{}, task_type="thinking"), or "brief"
        (one-line-per-step reasoning ending in "So the answer is X",
        task_type="cot" -- much shorter generations, checks whether most of
        full CoT's accuracy gain survives without paying for a long trace).
    """
    task_label = f"{dataset}_{task}" if task else dataset
    key = task or dataset
    eval_batch_size = EVAL_BATCH_SIZE.get(key, _DEFAULT_EVAL_BATCH_SIZE)
    eval_time_budget = EVAL_TIME_BUDGET.get(key, _DEFAULT_EVAL_TIME_BUDGET)
    if cot_mode == "full":
        # Uniform across every task/method in a --cot run: full reasoning,
        # ending in \boxed{}, regardless of what the AO harness uses for the
        # same task -- see evaluator.py's _THINKING_EVAL_SYSTEM_PROMPT.
        eval_max_tokens = COT_MAX_NEW_TOKENS
        eval_task_type = "thinking"
    elif cot_mode == "brief":
        eval_max_tokens = COT_BRIEF_MAX_NEW_TOKENS
        eval_task_type = "cot"
    else:
        eval_max_tokens = EVAL_MAX_NEW_TOKENS.get(key, _DEFAULT_EVAL_MAX_NEW_TOKENS)
        eval_task_type = EVAL_TASK_TYPE.get(key, "")
    run_dir = f"{output_root}/{method}/{task_label}/seed_{seed}"
    if model_name:
        run_dir = f"{output_root}/{_model_slug(model_name)}/{method}/{task_label}/seed_{seed}"
    dataset_overrides: Dict[str, Any] = {"name": dataset, "task": task}
    if eval_task_type:
        dataset_overrides["task_type"] = eval_task_type
    if dev_test_split > 0.0:
        dataset_overrides["dev_test_split"] = dev_test_split
    eval_overrides: Dict[str, Any] = {
        "max_new_tokens": eval_max_tokens,
        "batch_size": eval_batch_size,
    }
    if strip_system_prompt:
        eval_overrides["system_prompt_override"] = ""
    elif simple_system_prompt:
        eval_overrides["system_prompt_override"] = SIMPLE_SYSTEM_PROMPT
    overrides: Dict[str, Any] = {
        "optimizer": {
            "method": method,
            "seed_prompt": seed_prompt,
        },
        "dataset": dataset_overrides,
        "evaluation": eval_overrides,
        "budget": {
            "time_seconds": eval_time_budget,
        },
        "seed": seed,
        "output_dir": run_dir,
    }
    if model_name:
        if model_name in OLLAMA_MODELS:
            overrides["llm"] = {
                "model_name": model_name,
                "backend": "ollama",
                **OLLAMA_MODELS[model_name],
            }
        elif model_name in GEMINI_MODELS:
            gemini_cfg = GEMINI_MODELS[model_name]
            if not gemini_cfg.get("api_key"):
                raise RuntimeError(
                    f"GEMINI_API_KEY not set in the environment -- required to run "
                    f"model '{model_name}'. Export it before launching."
                )
            overrides["llm"] = {"model_name": model_name, **gemini_cfg}
        else:
            overrides["llm"] = {"model_name": model_name}
    return load_config(base_config_path, overrides=overrides)


def _next_run_dir(base: str = "outputs") -> str:
    """First non-existing outputs/run_N directory (fresh default per launch)."""
    i = 1
    while Path(f"{base}/run_{i}").exists():
        i += 1
    return f"{base}/run_{i}"


def _read_yaml_meta(config_path: str) -> Dict[str, Any]:
    """Read raw YAML for runner-level keys not in RunConfig (models, output_dir)."""
    import yaml
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def run_experiment(
    methods: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    models: Optional[List[str]] = None,
    config_path: str = "experiments/configs/swift_apex_benchmark.yaml",
    output_dir: Optional[str] = None,
    dry_run: bool = False,
    cot_mode: str = "",
    dev_test_split: float = 0.0,
    generic_prompt: bool = False,
    strip_system_prompt: bool = False,
    simple_system_prompt: bool = False,
):
    """Run the full experiment matrix with multiple seeds (and optionally models).

    Args:
        methods: Methods to run (default: all).
        datasets: Datasets to run (default: all).
        tasks: Specific tasks to run (default: all per dataset).
        seeds: Random seeds for repetition (default: [42, 123, 7]).
        models: HF model names to loop over. Defaults to the `models:` list in
            the YAML if present; otherwise the single llm.model_name is used.
        config_path: Path to base config YAML.
        cot_mode: "" (default, answer-only), "full" (free-form reasoning
            ending in \\boxed{}), or "brief" (one-line-per-step reasoning,
            much shorter generations -- see build_run_config). Applied
            uniformly across every method/task in the run and loads the full
            worked-example seed prompt for either CoT mode, which is what
            keeps it a fair comparison rather than a one-off tweak.
        output_dir: Root output directory. Overrides the YAML's output_dir.
        dry_run: If True, only print what would be run.
        dev_test_split: Fraction of the post-train BBH pool given to dev (rest
            to test), e.g. 0.5 for an even 50/50 split. 0.0 (default)
            preserves the original fixed-size split (test capped at 115).
        generic_prompt: If True, replace every task's fetched seed prompt
            with a fixed generic instruction (GENERIC_PROMPT), carrying no
            task-specific wording or worked examples. Isolates how much of
            measured performance comes from prompt engineering vs. the
            model's raw zero-shot ability, given only the eval harness's own
            system prompt for output format.
        strip_system_prompt: If True, use an empty eval system prompt instead
            of the task_type default (e.g. _THINKING_EVAL_SYSTEM_PROMPT under
            --cot). Isolates how much of a run's score comes from the
            format-enforcing scaffolding (e.g. "end with \\boxed{X}") vs. the
            seed prompt / model's own reasoning tendencies. Orthogonal to
            generic_prompt -- combine both to test raw capability with
            neither task-specific instruction nor format scaffolding.
        simple_system_prompt: If True, use one short, task-type-agnostic
            format cue (SIMPLE_SYSTEM_PROMPT, "...Answer: X") instead of the
            per-task_type default. A middle ground between the full scaffolding
            and strip_system_prompt's nothing-at-all: tests whether a minimal
            universal cue recovers what strip_system_prompt collapsed, without
            the task-tailored elaboration of the default prompts. Mutually
            exclusive with strip_system_prompt (strip wins if both are set).
    """
    methods = methods or METHODS
    datasets_to_run = datasets or list(DATASETS.keys())
    seeds = seeds or SEEDS

    yaml_meta = _read_yaml_meta(config_path)
    if models is None:
        models = yaml_meta.get("models") or [None]
    # Priority: CLI flag → YAML output_dir → fresh outputs/run_N
    output_root = output_dir or yaml_meta.get("output_dir") or _next_run_dir()
    logger.info(f"Output root: {output_root}")

    results: Dict[str, Any] = {}
    total_runs = 0
    completed_runs = 0
    failed_runs = 0

    # Count total runs (models × tasks × methods × seeds)
    for model in models:
        for dataset in datasets_to_run:
            ds_config = DATASETS[dataset]
            ds_tasks = tasks if tasks else ds_config["tasks"]
            for task in ds_tasks:
                for method in methods:
                    for seed in seeds:
                        total_runs += 1

    logger.info(f"{'='*70}")
    logger.info(f"EXPERIMENT: APO Benchmark (multi-seed, multi-model)")
    logger.info(f"Models: {[m or 'from-yaml' for m in models]}")
    logger.info(f"Methods: {methods}")
    logger.info(f"Datasets: {datasets_to_run}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Total runs: {total_runs} ({total_runs // len(seeds)} configs × {len(seeds)} seeds)")
    logger.info(f"{'='*70}")

    if dry_run:
        logger.info("\n[DRY RUN] Would execute:")
        for model in models:
            for dataset in datasets_to_run:
                ds_config = DATASETS[dataset]
                ds_tasks = tasks if tasks else ds_config["tasks"]
                for task in ds_tasks:
                    for method in methods:
                        task_label = f"{dataset}/{task}" if task else dataset
                        if cot_mode == "full":
                            eval_tok = COT_MAX_NEW_TOKENS
                        elif cot_mode == "brief":
                            eval_tok = COT_BRIEF_MAX_NEW_TOKENS
                        else:
                            eval_tok = EVAL_MAX_NEW_TOKENS.get(task or dataset, _DEFAULT_EVAL_MAX_NEW_TOKENS)
                        logger.info(
                            f"  [{model or 'default'}] {method} on {task_label} "
                            f"(eval_max_tokens={eval_tok}, seeds={seeds})"
                        )
        return

    # Execute runs. Models loop is outermost so each model's weights are only
    # loaded/unloaded once per block of runs (orchestrator reloads per run,
    # but the HF disk cache stays warm and VRAM never holds two models).
    for model in models:
        model_label = _model_slug(model) if model else "default"
        for dataset in datasets_to_run:
            ds_config = DATASETS[dataset]
            ds_tasks = tasks if tasks else ds_config["tasks"]

            for task in ds_tasks:
                # Fetch seed prompt (once per task, shared across seeds).
                # BBH answer-only: instruction line only — full CoT examples
                # conflict with the JSON system prompt, causing the model to
                # output reasoning instead of {"answer": "X"}. Either CoT mode
                # flips this: the whole run uses a CoT task_type, which wants
                # (and was validated with) the full worked-example prompt.
                if generic_prompt:
                    # No task-specific instruction or worked examples at all --
                    # isolates how much of measured performance comes from
                    # prompt engineering vs. the model's raw zero-shot ability
                    # on this task, given only the eval harness's own system
                    # prompt (which still specifies output format).
                    seed_prompt = GENERIC_PROMPT
                    logger.info(f"Using generic prompt for {dataset}/{task} (no task-specific text)")
                else:
                    use_full = bool(cot_mode)
                    try:
                        seed_prompt = get_seed_prompt(dataset, task, use_full_prompt=use_full)
                        logger.info(f"Loaded seed prompt for {dataset}/{task} ({len(seed_prompt)} chars)")
                    except Exception as e:
                        logger.error(f"Failed to load seed prompt for {dataset}/{task}: {e}")
                        seed_prompt = ""

                for method in methods:
                    task_label = f"{dataset}/{task}" if task else dataset

                    for seed in seeds:
                        run_key = f"{model_label}_{method}_{task_label}_seed{seed}"

                        # Skip if result file already exists (safe to re-launch)
                        _task_fs = f"{dataset}_{task}" if task else dataset
                        if model:
                            _run_dir = Path(output_root) / _model_slug(model) / method / _task_fs / f"seed_{seed}"
                        else:
                            _run_dir = Path(output_root) / method / _task_fs / f"seed_{seed}"
                        # Glob rather than a fixed filename: the result file
                        # is named after the OPTIMIZER CLASS's hardcoded
                        # .name (e.g. "funnel_v4d"), not necessarily the
                        # method string used to launch it -- an alias like
                        # "funnel_lean"/"funnel_wide"/"funnel_indexed"
                        # resolves to a class whose .name differs from the
                        # alias, so a fixed f"result_{method}_..." filename
                        # never matched and skip-existing silently never
                        # skipped anything for alias-launched runs. The
                        # directory itself is already correctly scoped by
                        # the alias string, so any result file in it is the
                        # right one regardless of what name is embedded.
                        if _run_dir.exists() and list(_run_dir.glob("result_*.json")):
                            logger.info(f"  ↩ SKIP {method}/{_task_fs}/seed_{seed} (result exists)")
                            completed_runs += 1
                            continue

                        logger.info(f"\n{'='*60}")
                        logger.info(f"RUN: {method} on {task_label} [model={model_label} seed={seed}]")
                        logger.info(f"{'='*60}")

                        orchestrator = None
                        try:
                            config = build_run_config(
                                base_config_path=config_path,
                                method=method,
                                dataset=dataset,
                                task=task,
                                seed_prompt=seed_prompt,
                                seed=seed,
                                model_name=model,
                                output_root=output_root,
                                cot_mode=cot_mode,
                                dev_test_split=dev_test_split,
                                strip_system_prompt=strip_system_prompt,
                                simple_system_prompt=simple_system_prompt,
                            )

                            from pof.orchestration.runner import RunOrchestrator
                            orchestrator = RunOrchestrator(config)
                            result = orchestrator.run()

                            results[run_key] = {
                                "model": model or config.llm.model_name,
                                "method": method,
                                "dataset": dataset,
                                "task": task,
                                "seed": seed,
                                "best_score": result.best_score,
                                "test_score": result.test_score,
                                "total_time": result.total_time,
                                "llm_calls": result.llm_usage.total_calls if result.llm_usage else 0,
                                "total_tokens": result.llm_usage.total_tokens if result.llm_usage else 0,
                                "best_prompt": result.best_prompt,
                            }
                            completed_runs += 1
                            logger.info(
                                f"  ✓ dev={result.best_score:.4f} | test={result.test_score:.4f} | "
                                f"Time: {result.total_time:.1f}s | Seed: {seed}"
                            )

                        except Exception as e:
                            logger.error(f"  ✗ FAILED: {e}")
                            results[run_key] = {
                                "model": model,
                                "method": method,
                                "dataset": dataset,
                                "task": task,
                                "seed": seed,
                                "error": str(e),
                            }
                            failed_runs += 1
                        finally:
                            if orchestrator is not None:
                                orchestrator.cleanup()

    # Save results summary
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "experiment_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print final summary
    logger.info(f"\n{'='*70}")
    logger.info(f"EXPERIMENT COMPLETE")
    logger.info(f"  Completed: {completed_runs}/{total_runs}")
    logger.info(f"  Failed: {failed_runs}/{total_runs}")
    logger.info(f"  Seeds used: {seeds}")
    logger.info(f"  Results saved to: {summary_path}")
    logger.info(f"{'='*70}")

    # Aggregate results across seeds (mean ± std)
    _print_aggregated_results(results, seeds)


def _print_aggregated_results(results: Dict[str, Any], seeds: List[int]) -> None:
    """Print results aggregated across seeds (mean ± std)."""
    from collections import defaultdict
    import statistics

    # Group by model+method+task (across seeds) — track both dev and test scores
    grouped_dev: Dict[str, List[float]] = defaultdict(list)
    grouped_test: Dict[str, List[float]] = defaultdict(list)
    for key, r in results.items():
        if "error" not in r:
            model = _model_slug(r["model"]) if r.get("model") else "default"
            group_key = f"{model}|{r['method']}|{r['dataset']}/{r.get('task', '')}"
            grouped_dev[group_key].append(r["best_score"])
            grouped_test[group_key].append(r.get("test_score", 0.0))

    if not grouped_test:
        return

    logger.info(
        f"\n{'Model':<24} {'Method':<8} {'Dataset/Task':<38} "
        f"{'Dev mean':<10} {'Test mean':<10} {'Test std':<10} {'Runs':<5}"
    )
    logger.info("-" * 110)
    for group_key in sorted(grouped_test.keys()):
        dev_scores = grouped_dev[group_key]
        test_scores = grouped_test[group_key]
        model, method, task_label = group_key.split("|", 2)
        logger.info(
            f"{model:<24} {method:<8} {task_label:<38} "
            f"{statistics.mean(dev_scores):<10.4f} "
            f"{statistics.mean(test_scores):<10.4f} "
            f"{(statistics.stdev(test_scores) if len(test_scores) > 1 else 0.0):<10.4f} "
            f"{len(test_scores)}/{len(seeds)}"
        )


def main():
    parser = argparse.ArgumentParser(description="Run SWIFT & APEX benchmark experiment")
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help="Methods to run (default: swift apex)",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Datasets to run (default: bbh svamp humaneval)",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="Specific BBH tasks to run (default: all 8 in DATASETS['bbh']['tasks'])",
    )
    parser.add_argument(
        "--config", type=str, default="experiments/configs/swift_apex_benchmark.yaml",
        help="Path to base config YAML",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Random seeds (default: 42 123 7)",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="HF model names to loop over (default: `models:` list in the YAML, "
             "or the single llm.model_name)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Root output directory (overrides the YAML's output_dir)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be run without executing",
    )
    parser.add_argument(
        "--cot", action="store_true",
        help="Run every task in this invocation with full step-by-step "
             "reasoning ending in \\boxed{} (task_type='thinking') instead of "
             "each task's normal answer-only settings, seeded with the full "
             "worked-example prompt. Applied uniformly to every method/task "
             "passed, not per-task. Mutually exclusive with --cot-brief.",
    )
    parser.add_argument(
        "--cot-brief", action="store_true",
        help="Like --cot, but one-line-per-step reasoning ending in 'So the "
             "answer is X' (task_type='cot') instead of free-form reasoning "
             "ending in \\boxed{} -- much shorter generations. Checks whether "
             "most of full CoT's accuracy gain survives without paying for a "
             "long trace. Mutually exclusive with --cot.",
    )
    parser.add_argument(
        "--dev-test-split", type=float, default=0.0,
        help="Fraction of the post-train BBH pool given to dev, rest to test "
             "(e.g. 0.5 for an even 50/50 split). Default 0.0 preserves the "
             "original fixed-size split (test capped at 115, dev gets the "
             "remainder). Tests whether a larger, evenly-sized dev pool lets "
             "search gains actually transfer to test.",
    )
    parser.add_argument(
        "--generic-prompt", action="store_true",
        help="Replace every task's fetched seed prompt with a fixed generic "
             "instruction carrying no task-specific wording or worked "
             "examples (GENERIC_PROMPT). Isolates how much measured "
             "performance comes from prompt engineering vs. the model's raw "
             "zero-shot ability on the task.",
    )
    parser.add_argument(
        "--strip-system-prompt", action="store_true",
        help="Use an empty eval system prompt instead of the task_type "
             "default (e.g. under --cot, drops the \\boxed{X} format "
             "instruction). Isolates how much of a run's score comes from "
             "that scaffolding vs. the seed prompt / model's own reasoning "
             "tendencies. Combine with --generic-prompt to test raw "
             "capability with neither task-specific instruction nor format "
             "scaffolding.",
    )
    parser.add_argument(
        "--simple-system-prompt", action="store_true",
        help="Use one short, task-type-agnostic format cue "
             "(SIMPLE_SYSTEM_PROMPT, ending in 'Answer: X') instead of the "
             "per-task_type default. Middle ground between the full "
             "scaffolding and --strip-system-prompt's nothing-at-all -- tests "
             "whether a minimal universal cue recovers what stripping "
             "collapsed. Mutually exclusive with --strip-system-prompt.",
    )

    args = parser.parse_args()
    if args.strip_system_prompt and args.simple_system_prompt:
        parser.error("--strip-system-prompt and --simple-system-prompt are mutually exclusive")
    if args.cot and args.cot_brief:
        parser.error("--cot and --cot-brief are mutually exclusive")
    cot_mode = "full" if args.cot else "brief" if args.cot_brief else ""

    run_experiment(
        methods=args.methods,
        datasets=args.datasets,
        tasks=args.tasks,
        seeds=args.seeds,
        models=args.models,
        config_path=args.config,
        output_dir=args.output_dir,
        dev_test_split=args.dev_test_split,
        dry_run=args.dry_run,
        cot_mode=cot_mode,
        generic_prompt=args.generic_prompt,
        strip_system_prompt=args.strip_system_prompt,
        simple_system_prompt=args.simple_system_prompt,
    )


if __name__ == "__main__":
    main()