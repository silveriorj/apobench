"""Full re-run of SWIFT and APEX across all three benchmarks.

Incorporates all fixes applied after the original proposal_comparison run:
  - format_constraint operator gated for math/code tasks (no longer strips CoT)
  - EVAL_MAX_NEW_TOKENS['gsm8k']:    256 → 512
  - EVAL_MAX_NEW_TOKENS['humaneval']: 512 → 1024
  - Structured decomposition: field-targeted mutation and crossover
  - APEX: ProTeGi paraphrase expansion in failure_guided operator
  - APEX: field_targeted UCB arm added; op_crossover uses structured first
  - APEX: failure_guided no longer re-evaluates cached records

Results are written to a new output directory — the original
proposal_comparison_3bench_qwen3-4-instruct results are untouched and
remain valid for GAAPO, SEE, CAPO, GSPE, and baseline comparisons.

Usage (from the pof/ repo root):
    python experiments/run_swift_apex_rerun.py
    python experiments/run_swift_apex_rerun.py --datasets bbh gsm8k
    python experiments/run_swift_apex_rerun.py --methods swift
    python experiments/run_swift_apex_rerun.py --dry-run
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.run_swift_apex import run_experiment

OUTPUT_DIR = "outputs/swift_apex_v2_qwen3-4-instruct"
METHODS = ["swift", "apex"]
DATASETS = ["bbh", "gsm8k", "humaneval"]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SWIFT + APEX full re-run with structured decomposition fixes"
    )
    parser.add_argument("--methods",  nargs="+", default=METHODS)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds",    nargs="+", type=int, default=[42, 123, 7])
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
