"""GSM8K + HumanEval re-run with corrected evaluation token limits.

Changes vs the original proposal_comparison run:
  - EVAL_MAX_NEW_TOKENS['gsm8k']:   256 → 512   (CoT was being truncated)
  - EVAL_MAX_NEW_TOKENS['humaneval']: 512 → 1024 (solutions hit cap at p90/p95/p99)
  - GSM8K seed prompt: unchanged (pof CoT seed is the correct starting point)

Usage (from the pof/ repo root):
    python experiments/run_gsm8k_rerun.py                        # both datasets
    python experiments/run_gsm8k_rerun.py --datasets gsm8k       # GSM8K only
    python experiments/run_gsm8k_rerun.py --datasets humaneval   # HumanEval only
    python experiments/run_gsm8k_rerun.py --methods swift apex   # subset of methods
    python experiments/run_gsm8k_rerun.py --dry-run              # preview
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.run_swift_apex import run_experiment

OUTPUT_DIR = "outputs/proposal_comparison_math_code_rerun_qwen3-4-instruct"

METHODS = ["swift", "apex", "gspe", "gaapo", "see", "capo"]
DATASETS = ["gsm8k", "humaneval"]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GSM8K + HumanEval re-run (512-tok GSM8K, 1024-tok HumanEval)"
    )
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7])
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument(
        "--config",
        default="experiments/configs/swift_apex_benchmark.yaml",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_experiment(
        methods=args.methods,
        datasets=args.datasets,
        tasks=None,
        seeds=args.seeds,
        models=None,
        config_path=args.config,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )
