"""Evaluate seed prompts (no optimization) on all three benchmarks.

Produces result files in the same format as run_swift_apex.py so the baseline
row in tables is directly comparable with any method re-run.

Token limits match the corrected config:
  - BBH tasks:  16 tokens  (single-word / letter answers)
  - GSM8K:     512 tokens  (was 256 — CoT was being truncated)
  - HumanEval: 1024 tokens (was 512 — solutions hit cap at p90/p95/p99)

Usage (from the pof/ repo root):
    python experiments/run_baseline_gsm8k.py                      # all benchmarks
    python experiments/run_baseline_gsm8k.py --datasets bbh
    python experiments/run_baseline_gsm8k.py --datasets gsm8k humaneval
    python experiments/run_baseline_gsm8k.py --output-dir outputs/my_dir
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pof.config.loader import load_config
from pof.datasets.loader import load_dataset_by_name
from pof.evaluation.evaluator import Evaluator
from pof.llm.factory import create_llm
from pof.prompts.loader import get_seed_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = "experiments/configs/swift_apex_benchmark.yaml"
OUTPUT_DIR  = "outputs/swift_apex_v2_qwen3-4-instruct"
SEEDS       = [42, 123, 7]
DEV_SAMPLES  = 50
TEST_SAMPLES = 115

BBH_TASKS = [
    "causal_judgement",
    "disambiguation_qa",
    "formal_fallacies",
    "hyperbaton",
]

# Per-dataset eval settings — mirrors EVAL_MAX_NEW_TOKENS in run_swift_apex.py.
DATASET_SETTINGS = {
    "bbh": {
        "task_type":          "auto",
        "max_new_tokens":     16,
        "num_dataset_samples": 300,
    },
    "gsm8k": {
        "task_type":          "math",
        "max_new_tokens":     512,
        "num_dataset_samples": 500,
    },
    "humaneval": {
        "task_type":          "text",
        "max_new_tokens":     1024,
        "num_dataset_samples": 200,
    },
}


def _run_single(
    dataset: str,
    task: str,
    seed_prompt: str,
    settings: dict,
    seeds: list,
    output_dir: str,
    llm,
    config,
) -> None:
    """Evaluate one dataset/task combination across all seeds."""
    task_label = f"{dataset}_{task}" if task else dataset

    ds = load_dataset_by_name(
        dataset, task=task,
        num_samples=settings["num_dataset_samples"],
        seed=42,
    )

    evaluator = Evaluator(
        llm=llm,
        task_type=settings["task_type"],
        max_new_tokens=settings["max_new_tokens"],
        temperature=0.0,
        batch_size=config.evaluation.batch_size,
    )

    for seed in seeds:
        out_dir    = Path(output_dir) / "baseline" / task_label / f"seed_{seed}"
        result_file = out_dir / f"result_baseline_{task_label}.json"

        if result_file.exists():
            logger.info(f"  ↩ SKIP baseline/{task_label}/seed_{seed} (result exists)")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"BASELINE  {task_label}  seed={seed}")
        logger.info(f"{'='*60}")

        t0 = time.time()

        dev_result  = evaluator.evaluate(seed_prompt, ds.dev_samples,
                                         num_samples=DEV_SAMPLES, shuffle=True)
        test_result = evaluator.evaluate(seed_prompt, ds.test_samples,
                                         num_samples=TEST_SAMPLES, shuffle=False)

        elapsed = time.time() - t0
        logger.info(
            f"  dev={dev_result.score:.4f}  test={test_result.score:.4f}  {elapsed:.1f}s"
        )

        result = {
            "method_name":         "baseline",
            "dataset_name":        dataset,
            "best_prompt":         seed_prompt,
            "best_score":          dev_result.score,
            "test_score":          test_result.score,
            "optimization_history": [],
            "final_population":    [],
            "llm_usage": {
                "total_calls":        dev_result.num_samples + test_result.num_samples,
                "total_time_seconds": elapsed,
            },
            "total_time":    elapsed,
            "num_iterations": 0,
            "config": {
                "method":             "baseline",
                "eval_max_new_tokens": settings["max_new_tokens"],
                "model":              config.llm.model_name,
            },
        }

        out_dir.mkdir(parents=True, exist_ok=True)
        result_file.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(f"  Saved → {result_file}")


def run_baselines(datasets: list, seeds: list, output_dir: str, config_path: str) -> None:
    config = load_config(config_path, overrides={"evaluation": {"max_new_tokens": 32}})
    llm    = create_llm(config.llm)

    try:
        for dataset in datasets:
            if dataset not in DATASET_SETTINGS:
                logger.error(
                    f"Unknown dataset: '{dataset}'. Choose from: {list(DATASET_SETTINGS)}"
                )
                continue

            settings = DATASET_SETTINGS[dataset]

            if dataset == "bbh":
                for task in BBH_TASKS:
                    seed_prompt = get_seed_prompt("bbh", task, use_full_prompt=True)
                    logger.info(
                        f"\nSeed prompt for bbh/{task} ({len(seed_prompt)} chars)"
                    )
                    _run_single(
                        dataset="bbh", task=task,
                        seed_prompt=seed_prompt, settings=settings,
                        seeds=seeds, output_dir=output_dir,
                        llm=llm, config=config,
                    )
            else:
                seed_prompt = get_seed_prompt(dataset, "", use_full_prompt=True)
                logger.info(
                    f"\nSeed prompt for {dataset} ({len(seed_prompt)} chars):\n"
                    f"{seed_prompt[:200]}..."
                )
                _run_single(
                    dataset=dataset, task="",
                    seed_prompt=seed_prompt, settings=settings,
                    seeds=seeds, output_dir=output_dir,
                    llm=llm, config=config,
                )
    finally:
        if hasattr(llm, "cleanup"):
            llm.cleanup()

    logger.info("\nBaseline evaluation complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Baseline seed-prompt evaluation (no optimization)"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["bbh", "gsm8k", "humaneval"],
        help="Datasets to evaluate (default: all three)",
    )
    parser.add_argument("--seeds",      nargs="+", type=int, default=SEEDS)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--config",     default=CONFIG_PATH)
    args = parser.parse_args()

    run_baselines(
        datasets=args.datasets,
        seeds=args.seeds,
        output_dir=args.output_dir,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
