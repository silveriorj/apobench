"""CLI entry point for the Prompt Optimization Framework."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from pof.config.loader import load_config
from pof.orchestration.runner import RunOrchestrator


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="pof",
        description="Prompt Optimization Framework — low-cost, auditable prompt evolution",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run command ---
    run_parser = subparsers.add_parser("run", help="Run a single optimization")
    run_parser.add_argument(
        "-c", "--config", type=str, default=None,
        help="Path to config file (YAML/JSON)",
    )
    run_parser.add_argument(
        "-m", "--method", type=str, default=None,
        help="Optimization method (see, swift, apex, gaapo, capo, gepa)",
    )
    run_parser.add_argument(
        "--model", type=str, default=None,
        help="Model name (e.g., Qwen/Qwen2.5-3B-Instruct)",
    )
    run_parser.add_argument(
        "--dataset", type=str, default=None,
        help="Dataset name (bbh) or path to JSON file",
    )
    run_parser.add_argument(
        "--task", type=str, default=None,
        help="Specific task within dataset",
    )
    run_parser.add_argument(
        "--seed-prompt", type=str, default=None,
        help="Initial seed prompt",
    )
    run_parser.add_argument(
        "-o", "--output", type=str, default="outputs",
        help="Output directory",
    )
    run_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose logging",
    )

    # --- benchmark command ---
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark multiple methods")
    bench_parser.add_argument(
        "-c", "--config", type=str, default=None,
        help="Path to config file",
    )
    bench_parser.add_argument(
        "--methods", type=str, nargs="+", default=None,
        help="Methods to benchmark (default: all)",
    )
    bench_parser.add_argument(
        "--model", type=str, default=None,
        help="Model name",
    )
    bench_parser.add_argument(
        "--dataset", type=str, default=None,
        help="Dataset name or path",
    )
    bench_parser.add_argument(
        "--task", type=str, default=None,
        help="Specific task",
    )
    bench_parser.add_argument(
        "-o", "--output", type=str, default="outputs",
        help="Output directory",
    )
    bench_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose logging",
    )

    # --- list command ---
    list_parser = subparsers.add_parser("list", help="List available optimizers")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    # Setup logging
    log_level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "list":
        from pof.optimizers import list_optimizers
        print("Available optimizers:")
        for name in list_optimizers():
            print(f"  - {name}")
        return 0

    # Build config overrides from CLI args
    overrides = {}
    if getattr(args, "method", None):
        overrides.setdefault("optimizer", {})["method"] = args.method
    if getattr(args, "model", None):
        overrides.setdefault("llm", {})["model_name"] = args.model
    if getattr(args, "dataset", None):
        overrides.setdefault("dataset", {})["name"] = args.dataset
    if getattr(args, "task", None):
        overrides.setdefault("dataset", {})["task"] = args.task
    if getattr(args, "seed_prompt", None):
        overrides.setdefault("optimizer", {})["seed_prompt"] = args.seed_prompt
    if getattr(args, "output", None):
        overrides["output_dir"] = args.output

    # Load config
    config_path = getattr(args, "config", None)
    config = load_config(config_path, overrides=overrides if overrides else None)

    # Create orchestrator
    orchestrator = RunOrchestrator(config)

    if args.command == "run":
        result = orchestrator.run()
        print(f"\n✓ Optimization complete!")
        print(f"  Method: {result.method_name}")
        print(f"  Best score: {result.best_score:.4f}")
        print(f"  Time: {result.total_time:.1f}s")
        print(f"  Best prompt: {result.best_prompt[:100]}...")
        return 0

    elif args.command == "benchmark":
        methods = getattr(args, "methods", None)
        results = orchestrator.benchmark(methods=methods)
        print(f"\n✓ Benchmark complete! {len(results)} methods compared.")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())