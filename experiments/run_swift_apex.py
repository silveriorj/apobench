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

METHODS = ["swift", "apex"]

# Random seeds for statistical robustness (3 runs per configuration)
SEEDS = [42, 123, 7]

# Per-task eval max_new_tokens (eval output only; operator/LLM generation uses
# llm.max_new_tokens from the YAML, which stays at 512).
# BBH answers vary: dyck needs bracket sequences (~40 tok), most others are
# single-word or letter answers (Yes/No, A/B/C, color name).
EVAL_MAX_NEW_TOKENS: Dict[str, int] = {
    # BBH
    "dyck_languages": 16,
    "causal_judgement": 8,
    "disambiguation_qa": 8,
    "formal_fallacies": 8,
    "hyperbaton": 8,
    "logical_deduction_five_objects": 16,
    "reasoning_about_colored_objects": 16,
    # Other datasets
    "gsm8k": 32,       # just the numeric answer
    "humaneval": 1024,  # full function body
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
            "logical_deduction_five_objects",
            "penguins_in_a_table",
            "reasoning_about_colored_objects",
            "web_of_lies",
        ],
        "task_type": "auto",
    },
    "gsm8k": {
        "tasks": [""],  # Single task
        "task_type": "math",
    },
    "humaneval": {
        "tasks": [""],  # Single task
        "task_type": "text",
    },
}


def build_run_config(
    base_config_path: str,
    method: str,
    dataset: str,
    task: str,
    seed_prompt: str,
    seed: int = 42,
) -> RunConfig:
    """Build a RunConfig for a specific method/dataset/task combination."""
    task_label = f"{dataset}_{task}" if task else dataset
    eval_max_tokens = EVAL_MAX_NEW_TOKENS.get(task or dataset, _DEFAULT_EVAL_MAX_NEW_TOKENS)
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
        "output_dir": f"outputs/swift_apex_benchmark/{method}/{task_label}/seed_{seed}",
    }
    return load_config(base_config_path, overrides=overrides)


def run_experiment(
    methods: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    config_path: str = "experiments/configs/swift_apex_benchmark.yaml",
    dry_run: bool = False,
):
    """Run the full experiment matrix with multiple seeds.

    Args:
        methods: Methods to run (default: all).
        datasets: Datasets to run (default: all).
        tasks: Specific tasks to run (default: all per dataset).
        seeds: Random seeds for repetition (default: [42, 123, 7]).
        config_path: Path to base config YAML.
        dry_run: If True, only print what would be run.
    """
    methods = methods or METHODS
    datasets_to_run = datasets or list(DATASETS.keys())
    seeds = seeds or SEEDS

    results: Dict[str, Any] = {}
    total_runs = 0
    completed_runs = 0
    failed_runs = 0

    # Count total runs (methods × tasks × seeds)
    for dataset in datasets_to_run:
        ds_config = DATASETS[dataset]
        ds_tasks = tasks if tasks else ds_config["tasks"]
        for task in ds_tasks:
            for method in methods:
                for seed in seeds:
                    total_runs += 1

    logger.info(f"{'='*70}")
    logger.info(f"EXPERIMENT: SWIFT & APEX Benchmark (multi-seed)")
    logger.info(f"Methods: {methods}")
    logger.info(f"Datasets: {datasets_to_run}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Total runs: {total_runs} ({total_runs // len(seeds)} configs × {len(seeds)} seeds)")
    logger.info(f"{'='*70}")

    if dry_run:
        logger.info("\n[DRY RUN] Would execute:")
        for dataset in datasets_to_run:
            ds_config = DATASETS[dataset]
            ds_tasks = tasks if tasks else ds_config["tasks"]
            for task in ds_tasks:
                for method in methods:
                    task_label = f"{dataset}/{task}" if task else dataset
                    eval_tok = EVAL_MAX_NEW_TOKENS.get(task or dataset, _DEFAULT_EVAL_MAX_NEW_TOKENS)
                    logger.info(
                        f"  {method} on {task_label} "
                        f"(eval_max_tokens={eval_tok}, seeds={seeds})"
                    )
        return

    # Execute runs
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
                    run_key = f"{method}_{task_label}_seed{seed}"

                    logger.info(f"\n{'='*60}")
                    logger.info(f"RUN: {method} on {task_label} [seed={seed}]")
                    logger.info(f"{'='*60}")

                    try:
                        config = build_run_config(
                            base_config_path=config_path,
                            method=method,
                            dataset=dataset,
                            task=task,
                            seed_prompt=seed_prompt,
                            seed=seed,
                        )

                        from pof.orchestration.runner import RunOrchestrator
                        orchestrator = RunOrchestrator(config)
                        result = orchestrator.run()

                        results[run_key] = {
                            "method": method,
                            "dataset": dataset,
                            "task": task,
                            "seed": seed,
                            "best_score": result.best_score,
                            "test_score": result.test_score,
                            "total_time": result.total_time,
                            "llm_calls": result.llm_usage.total_calls if result.llm_usage else 0,
                            "total_tokens": result.llm_usage.total_tokens if result.llm_usage else 0,
                            "best_prompt": result.best_prompt[:200],
                        }
                        completed_runs += 1
                        logger.info(
                            f"  ✓ dev={result.best_score:.4f} | test={result.test_score:.4f} | "
                            f"Time: {result.total_time:.1f}s | Seed: {seed}"
                        )

                    except Exception as e:
                        logger.error(f"  ✗ FAILED: {e}")
                        results[run_key] = {
                            "method": method,
                            "dataset": dataset,
                            "task": task,
                            "seed": seed,
                            "error": str(e),
                        }
                        failed_runs += 1

    # Save results summary
    output_dir = Path("outputs/swift_apex_benchmark")
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

    # Group by method+task (across seeds) — track both dev and test scores
    grouped_dev: Dict[str, List[float]] = defaultdict(list)
    grouped_test: Dict[str, List[float]] = defaultdict(list)
    for key, r in results.items():
        if "error" not in r:
            group_key = f"{r['method']}|{r['dataset']}/{r.get('task', '')}"
            grouped_dev[group_key].append(r["best_score"])
            grouped_test[group_key].append(r.get("test_score", 0.0))

    if not grouped_test:
        return

    logger.info(
        f"\n{'Method':<8} {'Dataset/Task':<38} "
        f"{'Dev mean':<10} {'Test mean':<10} {'Test std':<10} {'Runs':<5}"
    )
    logger.info("-" * 85)
    for group_key in sorted(grouped_test.keys()):
        dev_scores = grouped_dev[group_key]
        test_scores = grouped_test[group_key]
        method, task_label = group_key.split("|", 1)
        logger.info(
            f"{method:<8} {task_label:<38} "
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
        "--dry-run", action="store_true",
        help="Print what would be run without executing",
    )

    args = parser.parse_args()

    run_experiment(
        methods=args.methods,
        datasets=args.datasets,
        tasks=args.tasks,
        seeds=args.seeds,
        config_path=args.config,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()