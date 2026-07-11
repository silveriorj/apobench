"""Retroactive test-score repair for result files missing their test evaluation.

Scans output directories for result_*.json files where:
  - best_prompt is non-empty (optimization ran and found something)
  - test_score == 0.0 AND best_score > 0  (test eval never ran; score is real)

For each such file the script loads the evaluator from a base config, runs the
held-out test evaluation, and writes the updated score back into the JSON.

The fix in runner.py (llm.attach_budget(None) before test eval) prevents new
occurrences; this script cleans up the existing ones.

Usage (from the pof/ repo root):
    python experiments/repair_test_scores.py --scan                # dry run, list affected files
    python experiments/repair_test_scores.py                       # scan + repair all
    python experiments/repair_test_scores.py --dirs outputs/swift_apex_benchmark
    python experiments/repair_test_scores.py --min-best-score 0.01  # skip 0-score files
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CONFIG_PATH   = "experiments/configs/swift_apex_benchmark.yaml"
TEST_SAMPLES  = 115

# Per-dataset evaluation settings — mirrors run_swift_apex.py
DATASET_SETTINGS = {
    "bbh":       {"task_type": "auto",  "max_new_tokens": 16},
    "gsm8k":     {"task_type": "math",  "max_new_tokens": 512},
    "humaneval": {"task_type": "text",  "max_new_tokens": 1024},
}

# BBH sub-tasks recognised from filenames
BBH_TASKS = {
    "causal_judgement", "disambiguation_qa", "formal_fallacies",
    "hyperbaton", "logical_deduction_five_objects", "penguins_in_a_table",
    "reasoning_about_colored_objects", "web_of_lies", "dyck_languages",
}


def _detect_dataset_task(file_path: Path, result: dict) -> Optional[tuple[str, str]]:
    """Infer (dataset, task) from file path and result content."""
    dataset_name = result.get("dataset_name", "")

    # dataset_name field often contains e.g. "bbh_disambiguation_qa" or "gsm8k"
    if dataset_name.startswith("bbh_"):
        return "bbh", dataset_name[len("bbh_"):]
    if dataset_name in ("gsm8k", "humaneval"):
        return dataset_name, ""

    # Fall back to parsing the path
    parts = [p.lower() for p in file_path.parts]
    for ds in ("gsm8k", "humaneval"):
        if any(ds in p for p in parts):
            return ds, ""
    for task in BBH_TASKS:
        if any(task in p for p in parts):
            return "bbh", task
    # Try filename
    fname = file_path.stem  # e.g. result_apex_bbh_reasoning_about_colored_objects
    m = re.search(r"bbh_(.+)$", fname)
    if m:
        return "bbh", m.group(1)

    return None


def find_broken(dirs: list[str], min_best_score: float = 0.001) -> list[Path]:
    """Return result files that need a test-score repair."""
    broken = []
    for root in dirs:
        for path in Path(root).rglob("result_*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Cannot read {path}: {e}")
                continue

            ts = data.get("test_score", -1)
            bs = data.get("best_score", -1)
            bp = data.get("best_prompt", "")

            if bp and (ts == 0 or ts == 0.0) and bs >= min_best_score:
                broken.append(path)

    return broken


def repair(
    path: Path,
    config_path: str = CONFIG_PATH,
    test_samples_n: int = TEST_SAMPLES,
    dry_run: bool = False,
) -> bool:
    """Run test evaluation for one result file and update it in place.

    Returns True on success, False on failure.
    """
    from pof.config.loader import load_config
    from pof.datasets.loader import load_dataset_by_name
    from pof.evaluation.evaluator import Evaluator
    from pof.llm.factory import create_llm

    data = json.loads(path.read_text(encoding="utf-8"))
    best_prompt = data["best_prompt"]

    dataset_task = _detect_dataset_task(path, data)
    if dataset_task is None:
        logger.warning(f"  Cannot detect dataset for {path} — skipping")
        return False

    dataset_name, task = dataset_task
    settings = DATASET_SETTINGS.get(dataset_name)
    if settings is None:
        logger.warning(f"  Unknown dataset '{dataset_name}' for {path} — skipping")
        return False

    logger.info(f"  Dataset: {dataset_name}/{task or '<all>'}  settings={settings}")

    if dry_run:
        logger.info("  [DRY RUN] would evaluate and update")
        return True

    # Load minimal config (budget intentionally left unlimited — repair eval only)
    config = load_config(config_path, overrides={"evaluation": {"max_new_tokens": settings["max_new_tokens"]}})

    llm = create_llm(config.llm)
    try:
        ds = load_dataset_by_name(
            dataset_name, task=task,
            num_samples=config.dataset.num_samples,
            seed=42,
        )
        test_samples_list = ds.get_eval_samples("test", n=test_samples_n)
        if not test_samples_list:
            logger.warning("  No test samples found — skipping")
            return False

        evaluator = Evaluator(
            llm=llm,
            task_type=settings["task_type"],
            max_new_tokens=settings["max_new_tokens"],
            temperature=0.0,
            batch_size=config.evaluation.batch_size,
        )

        result = evaluator.evaluate(best_prompt, test_samples_list)
        test_score = result.score
        logger.info(f"  test_score = {test_score:.4f}  (prev best_score={data['best_score']:.4f})")

        # Write updated result back
        data["test_score"] = test_score
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"  Saved → {path}")
        return True

    finally:
        if hasattr(llm, "cleanup"):
            llm.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Repair missing test_scores in result files")
    parser.add_argument(
        "--dirs", nargs="+",
        default=["outputs/ongoing", "outputs/ongoing2",
                 "outputs/swift_apex_benchmark",
                 "outputs/proposal_comparison_3bench_qwen3-4-instruct"],
        help="Output directories to scan (default: all standard dirs)",
    )
    parser.add_argument(
        "--min-best-score", type=float, default=0.001,
        help="Skip files whose best_score is below this threshold (default: 0.001)",
    )
    parser.add_argument(
        "--scan", action="store_true",
        help="Dry run: list affected files without evaluating",
    )
    parser.add_argument(
        "--config", default=CONFIG_PATH,
        help="Base config YAML for LLM / evaluator settings",
    )
    parser.add_argument(
        "--test-samples", type=int, default=TEST_SAMPLES,
        help=f"Number of test samples to evaluate (default: {TEST_SAMPLES})",
    )
    args = parser.parse_args()

    broken = find_broken(args.dirs, min_best_score=args.min_best_score)
    logger.info(f"Found {len(broken)} file(s) needing repair:")
    for p in broken:
        logger.info(f"  {p}")

    if not broken:
        logger.info("Nothing to repair.")
        return

    if args.scan:
        logger.info("\n[DRY RUN] Pass without --scan to repair.")
        return

    ok = 0
    failed = 0
    for path in broken:
        logger.info(f"\nRepairing: {path}")
        try:
            success = repair(
                path,
                config_path=args.config,
                test_samples_n=args.test_samples,
                dry_run=False,
            )
            if success:
                ok += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            failed += 1

    logger.info(f"\nDone. Repaired: {ok}  Failed: {failed}")


if __name__ == "__main__":
    main()
