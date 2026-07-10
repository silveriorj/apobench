"""Experiment runner: SWIFT & APEX on BBH, GSM8K, HumanEval.

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
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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

METHODS = ["swift", "apex", "gspe", "gaapo", "see", "capo"]  # Methods to run (default: all)

# Random seeds for statistical robustness (3 runs per configuration)
SEEDS = [42, 123, 7]

# Per-task eval max_new_tokens (eval output only; operator/LLM generation uses
# llm.max_new_tokens from the YAML, which stays at 512).
# BBH answers vary: dyck needs bracket sequences (~40 tok), most others are
# single-word or letter answers (Yes/No, A/B/C, color name).
EVAL_MAX_NEW_TOKENS: Dict[str, int] = {
    # BBH
    "dyck_languages": 16,
    "causal_judgement": 16,
    "disambiguation_qa": 16,
    "formal_fallacies": 16,
    "hyperbaton": 16,
    "logical_deduction_five_objects": 64,
    "penguins_in_a_table": 32,
    "reasoning_about_colored_objects": 64,
    "web_of_lies": 16,
    # Other datasets
    "gsm8k": 512,           # CoT via prompt_mid few-shot; 512 ensures full chain-of-thought fits
    "humaneval": 1024,     # 512 was truncating solutions (p90/p95/p99 all hit cap)
}

# Default fallback when task is not listed above
_DEFAULT_EVAL_MAX_NEW_TOKENS = 32

# Dataset configurations
DATASETS = {
    "bbh": {
        "tasks": [
            "causal_judgement",
            "disambiguation_qa",
            "formal_fallacies",
            "hyperbaton",
            # "logical_deduction_five_objects",
            # "penguins_in_a_table",
            # "reasoning_about_colored_objects",
            # "web_of_lies",
        ],
        "task_type": "auto",
    },
    "gsm8k": {
        "tasks": [""],  # Single split (test set, 1319 problems)
        "task_type": "math",
    },
    "humaneval": {
        "tasks": [""],  # Single task
        "task_type": "text",
    },
}


def _model_slug(model_name: str) -> str:
    """Filesystem-safe short name for a model, e.g. Qwen/Qwen3-0.6B → qwen3-0.6b."""
    return model_name.split("/")[-1].lower()


def build_run_config(
    base_config_path: str,
    method: str,
    dataset: str,
    task: str,
    seed_prompt: str,
    seed: int = 42,
    model_name: Optional[str] = None,
    output_root: str = "outputs/swift_apex_benchmark",
) -> RunConfig:
    """Build a RunConfig for a specific method/dataset/task/model combination."""
    task_label = f"{dataset}_{task}" if task else dataset
    eval_max_tokens = EVAL_MAX_NEW_TOKENS.get(task or dataset, _DEFAULT_EVAL_MAX_NEW_TOKENS)
    run_dir = f"{output_root}/{method}/{task_label}/seed_{seed}"
    if model_name:
        run_dir = f"{output_root}/{_model_slug(model_name)}/{method}/{task_label}/seed_{seed}"
    overrides: Dict[str, Any] = {
        "optimizer": {
            "method": method,
            "seed_prompt": seed_prompt,
        },
        "dataset": {
            "name": dataset,
            "task": task,
        },
        "evaluation": {
            "max_new_tokens": eval_max_tokens,
        },
        "seed": seed,
        "output_dir": run_dir,
    }
    if model_name:
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
        output_dir: Root output directory. Overrides the YAML's output_dir.
        dry_run: If True, only print what would be run.
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
                # Fetch seed prompt (once per task, shared across seeds)
                try:
                    seed_prompt = get_seed_prompt(dataset, task, use_full_prompt=True)
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
                        _result_file = _run_dir / f"result_{method}_{_task_fs}.json"
                        if _result_file.exists():
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
        help="Datasets to run (default: bbh gsm8k humaneval)",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="Specific BBH tasks to run (default: all 7)",
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

    args = parser.parse_args()

    run_experiment(
        methods=args.methods,
        datasets=args.datasets,
        tasks=args.tasks,
        seeds=args.seeds,
        models=args.models,
        config_path=args.config,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()