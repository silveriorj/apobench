"""Fill missing seeds in proposal_comparison_3bench_qwen3-4-instruct.

Targets only the slots that are absent after discarding v2 results:
  - capo   / gsm8k      (seeds 42, 123, 7)
  - gspe   / humaneval  (seeds 42, 123, 7)
  - capo   / humaneval  (seeds 42, 123, 7)
  - apex   / humaneval  (seed 42 only — seeds 7 and 123 already exist as apex_v1)

Token settings match the existing v1 runs:
  - GSM8K:     256 tokens
  - HumanEval: 512 tokens

Usage (from the pof/ repo root):
    python experiments/run_fill_gaps.py
    python experiments/run_fill_gaps.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from pof.config.loader import load_config
from pof.prompts.loader import get_seed_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_ROOT = "outputs/proposal_comparison_3bench_qwen3-4-instruct"
CONFIG_PATH = "experiments/configs/swift_apex_benchmark.yaml"

# Token limits matching the existing v1 runs
EVAL_MAX_NEW_TOKENS = {
    "gsm8k":     256,
    "humaneval": 512,
}

# Gaps to fill: (method, dataset, task, seeds, output_dir_name)
# output_dir_name is the subdirectory under OUTPUT_ROOT — must match existing layout.
GAPS: List[Dict[str, Any]] = [
    {"method": "capo",  "dataset": "gsm8k",     "task": "",   "seeds": [42, 123, 7], "dir_name": "capo"},
    {"method": "swift", "dataset": "gsm8k",     "task": "",   "seeds": [42, 123, 7], "dir_name": "swift"},
    {"method": "apex", "dataset": "gsm8k",     "task": "",   "seeds": [42, 123, 7], "dir_name": "apex"},
    {"method": "gspe",  "dataset": "humaneval",  "task": "",   "seeds": [42, 123, 7], "dir_name": "gspe"},
    {"method": "capo",  "dataset": "humaneval",  "task": "",   "seeds": [42, 123, 7], "dir_name": "capo"},
    # apex seed_42 humaneval: seeds 7 and 123 exist under apex_v1/; place seed_42 there too.
    {"method": "apex",  "dataset": "humaneval",  "task": "",   "seeds": [42],         "dir_name": "apex_v1"},
]


def run_gap(
    gap: Dict[str, Any],
    config_path: str,
    output_root: str,
    dry_run: bool = False,
) -> None:
    method    = gap["method"]
    dataset   = gap["dataset"]
    task      = gap["task"]
    seeds     = gap["seeds"]
    dir_name  = gap["dir_name"]

    task_label = f"{dataset}_{task}" if task else dataset
    eval_max   = EVAL_MAX_NEW_TOKENS.get(dataset, 32)

    seed_prompt = get_seed_prompt(dataset, task, use_full_prompt=True)
    logger.info(f"Seed prompt for {dataset}/{task or '<all>'} ({len(seed_prompt)} chars)")

    for seed in seeds:
        run_dir     = Path(output_root) / dir_name / task_label / f"seed_{seed}"
        result_file = run_dir / f"result_{method}_{task_label}.json"

        if result_file.exists():
            logger.info(f"  SKIP {dir_name}/{task_label}/seed_{seed} (exists)")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"RUN  {dir_name}/{task_label}/seed_{seed}  (method={method})")
        logger.info(f"{'='*60}")

        if dry_run:
            logger.info("  [DRY RUN]")
            continue

        overrides: Dict[str, Any] = {
            "optimizer": {"method": method, "seed_prompt": seed_prompt},
            "dataset":   {"name": dataset, "task": task},
            "evaluation": {"max_new_tokens": eval_max},
            "seed":      seed,
            "output_dir": str(run_dir),
        }
        config = load_config(config_path, overrides=overrides)

        orchestrator = None
        try:
            from pof.orchestration.runner import RunOrchestrator
            orchestrator = RunOrchestrator(config)
            result = orchestrator.run()
            logger.info(
                f"  dev={result.best_score:.4f}  test={result.test_score:.4f}  "
                f"time={result.total_time:.0f}s"
            )
        except Exception as e:
            logger.error(f"  FAILED: {e}")
        finally:
            if orchestrator is not None:
                orchestrator.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Fill missing result slots")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--config", default=CONFIG_PATH)
    args = parser.parse_args()

    logger.info(f"Output root: {args.output_root}")
    logger.info(f"Gaps to fill: {len(GAPS)} method/dataset combos")

    for gap in GAPS:
        run_gap(gap, config_path=args.config,
                output_root=args.output_root, dry_run=args.dry_run)

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
