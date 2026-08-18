"""Convert reference benchmark results to proposal_comparison format.

Supports two source layouts:

  Layout A — full_benchmark:
    {root}/{dataset}/{dataset}/{method}_seed{N}/
        {method}_{dataset}_{dataset}_seed{N}_metrics.json
        {method}_{dataset}_{dataset}_seed{N}_optimization.json

  Layout B — baseline_all_*:
    {root}/{dataset}/{task}/{seed_label}/{method}/
        summary.json       <- primary_metric = test_score
        history.json       <- best_prompt, best_score (dev)
    (for BBH: dataset=bbh, task=causal_judgement, etc.)
    (for gsm8k/humaneval: dataset=gsm8k, task=gsm8k, etc.)

Target layout (proposal_comparison format):
  {output_root}/{method}/{task_label}/seed_{seed}/result_{method}_{task_label}.json

Usage:
    python experiments/convert_full_benchmark.py                    # all sources, all datasets
    python experiments/convert_full_benchmark.py --dry-run
    python experiments/convert_full_benchmark.py --sources 20260616
    python experiments/convert_full_benchmark.py --methods gaapo see
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

RESULTS_ROOT = Path(
    r"C:\Users\Admin\Documents\prompt-optmization-framework\experiments\results"
)
OUTPUT_ROOT = Path(
    r"C:\Users\Admin\Documents\pof\outputs\proposal_comparison_3bench_qwen3-4-instruct"
)

# Named sources: short key -> (directory, layout)
SOURCES = {
    "full_benchmark": (RESULTS_ROOT / "full_benchmark", "A"),
    "20260616": (RESULTS_ROOT / "baseline_all_20260616_222637", "B"),
    "20260617": (RESULTS_ROOT / "baseline_all_20260617_092639", "B"),
}

# For layout B: which datasets to look for and how they map to task labels
# BBH tasks are discovered automatically from subdirectories
LAYOUT_B_DATASETS = {
    "gsm8k": "gsm8k",
    "humaneval": "humaneval",
    "bbh": None,  # tasks discovered from subdirs; label becomes bbh_{task}
}


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_result(out_file: Path, result: dict, dry_run: bool, label: str) -> None:
    status = "exists" if out_file.exists() else "new"
    tag = "[DRY]" if dry_run else "     "
    ts = result.get("test_score", 0.0)
    bs = result.get("best_score", 0.0)
    print(f"  {tag} {label}  test={ts:.4f} best={bs:.4f} ({status})")
    if not dry_run:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)


def _build_result(
    method: str,
    task_label: str,
    test_score: float,
    best_score: float,
    best_prompt: str,
    total_time: float,
    num_iterations: int,
    total_llm_calls: int,
    total_tokens: int,
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
    source_name: str,
    source_seed: int,
) -> dict:
    return {
        "method_name": method,
        "dataset_name": task_label,
        "best_prompt": best_prompt,
        "best_score": best_score,
        "test_score": test_score,
        "optimization_history": [],
        "final_population": [],
        "llm_usage": {
            "total_calls": total_llm_calls,
            "total_input_tokens": prompt_tokens,
            "total_output_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "total_time_seconds": total_time,
        },
        "total_time": total_time,
        "num_iterations": num_iterations,
        "config": {"method": method, "model": model},
        "_source": source_name,
        "_source_seed": source_seed,
    }


# ── Layout A ─────────────────────────────────────────────────────────────────

def _convert_layout_a(
    src_root: Path,
    source_name: str,
    datasets: list[str],
    methods_filter: list[str] | None,
    dry_run: bool,
) -> tuple[int, int]:
    converted = skipped = 0
    dataset_map = {"gsm8k": "gsm8k", "humaneval": "humaneval"}

    for dataset, task_label in dataset_map.items():
        if dataset not in datasets:
            continue
        dataset_dir = src_root / dataset / dataset
        if not dataset_dir.exists():
            continue

        runs: list[tuple[str, int]] = []
        for entry in sorted(dataset_dir.iterdir()):
            if not entry.is_dir():
                continue
            m = re.fullmatch(r"(.+)_seed(\d+)", entry.name)
            if m:
                meth, seed = m.group(1), int(m.group(2))
                if methods_filter is None or meth in methods_filter:
                    runs.append((meth, seed))

        print(f"\n[{source_name}] {dataset} -> {task_label}  ({len(runs)} runs)")

        for method, seed in runs:
            src_dir = dataset_dir / f"{method}_seed{seed}"
            prefix = f"{method}_{dataset}_{dataset}_seed{seed}"
            metrics = _load_json(src_dir / f"{prefix}_metrics.json")
            optim = _load_json(src_dir / f"{prefix}_optimization.json")

            if not metrics and not optim:
                skipped += 1
                continue

            tok = metrics.get("token_usage", {})
            result = _build_result(
                method=method,
                task_label=task_label,
                test_score=metrics.get("primary_metric") or metrics.get("metrics", {}).get("accuracy", 0.0),
                best_score=optim.get("best_score", 0.0),
                best_prompt=optim.get("best_prompt", ""),
                total_time=optim.get("elapsed_seconds") or metrics.get("elapsed_seconds", 0.0),
                num_iterations=optim.get("num_generations", 0),
                total_llm_calls=optim.get("total_llm_calls", 0),
                total_tokens=tok.get("total_tokens", 0),
                prompt_tokens=tok.get("prompt_tokens", 0),
                completion_tokens=tok.get("completion_tokens", 0),
                model=metrics.get("model", ""),
                source_name=source_name,
                source_seed=seed,
            )
            out_file = OUTPUT_ROOT / method / task_label / f"seed_{seed}" / f"result_{method}_{task_label}.json"
            _write_result(out_file, result, dry_run, f"{method}/{task_label}/seed_{seed}")
            converted += 1

    return converted, skipped


# ── Layout B ─────────────────────────────────────────────────────────────────

def _convert_layout_b_task(
    task_dir: Path,
    task_label: str,
    source_name: str,
    methods_filter: list[str] | None,
    dry_run: bool,
) -> tuple[int, int]:
    converted = skipped = 0

    for seed_entry in sorted(task_dir.iterdir()):
        if not seed_entry.is_dir():
            continue
        m = re.fullmatch(r"seed(\d+)", seed_entry.name)
        if not m:
            continue
        seed = int(m.group(1))

        for method_entry in sorted(seed_entry.iterdir()):
            if not method_entry.is_dir():
                continue
            method = method_entry.name
            if methods_filter is not None and method not in methods_filter:
                continue

            summary = _load_json(method_entry / "summary.json")
            history = _load_json(method_entry / "history.json")

            if not summary and not history:
                skipped += 1
                continue

            tok = summary.get("token_usage", {})
            result = _build_result(
                method=method,
                task_label=task_label,
                test_score=summary.get("primary_metric") or summary.get("metrics", {}).get("accuracy", 0.0),
                best_score=history.get("best_score", 0.0),
                best_prompt=history.get("best_prompt", ""),
                total_time=history.get("elapsed_seconds") or summary.get("elapsed_seconds", 0.0),
                num_iterations=history.get("num_generations", 0),
                total_llm_calls=history.get("total_llm_calls", 0),
                total_tokens=tok.get("total_tokens", 0),
                prompt_tokens=tok.get("prompt_tokens", 0),
                completion_tokens=tok.get("completion_tokens", 0),
                model=summary.get("model", ""),
                source_name=source_name,
                source_seed=seed,
            )
            out_file = OUTPUT_ROOT / method / task_label / f"seed_{seed}" / f"result_{method}_{task_label}.json"
            _write_result(out_file, result, dry_run, f"{method}/{task_label}/seed_{seed}")
            converted += 1

    return converted, skipped


def _convert_layout_b(
    src_root: Path,
    source_name: str,
    datasets: list[str],
    methods_filter: list[str] | None,
    dry_run: bool,
) -> tuple[int, int]:
    total_converted = total_skipped = 0

    for dataset in sorted(src_root.iterdir()):
        if not dataset.is_dir() or dataset.name not in datasets:
            continue

        if dataset.name == "bbh":
            # Each subdirectory is a BBH task
            for task_dir in sorted(dataset.iterdir()):
                if not task_dir.is_dir() or task_dir.name == "aggregated_summary.csv":
                    continue
                task_label = f"bbh_{task_dir.name}"
                print(f"\n[{source_name}] bbh/{task_dir.name} -> {task_label}")
                c, s = _convert_layout_b_task(task_dir, task_label, source_name, methods_filter, dry_run)
                total_converted += c
                total_skipped += s
        else:
            # gsm8k, humaneval: nested as {dataset}/{dataset}/
            task_dir = dataset / dataset.name
            if not task_dir.exists():
                continue
            task_label = dataset.name
            print(f"\n[{source_name}] {dataset.name} -> {task_label}")
            c, s = _convert_layout_b_task(task_dir, task_label, source_name, methods_filter, dry_run)
            total_converted += c
            total_skipped += s

    return total_converted, total_skipped


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert reference benchmarks to proposal_comparison format")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--sources", nargs="+", default=list(SOURCES.keys()),
        help=f"Sources to convert (default: all). Options: {list(SOURCES.keys())}",
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=["gsm8k", "humaneval", "bbh"],
        help="Datasets to convert (default: gsm8k humaneval bbh)",
    )
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help="Methods to include (default: all found)",
    )
    args = parser.parse_args()

    total_converted = total_skipped = 0

    for source_key in args.sources:
        if source_key not in SOURCES:
            print(f"Unknown source: {source_key}. Options: {list(SOURCES.keys())}")
            continue
        src_path, layout = SOURCES[source_key]
        if not src_path.exists():
            print(f"Source not found: {src_path}")
            continue

        print(f"\n{'='*65}")
        print(f"Source: {source_key}  ({layout})  {src_path.name}")
        print(f"{'='*65}")

        if layout == "A":
            c, s = _convert_layout_a(src_path, source_key, args.datasets, args.methods, args.dry_run)
        else:
            c, s = _convert_layout_b(src_path, source_key, args.datasets, args.methods, args.dry_run)

        total_converted += c
        total_skipped += s

    print(f"\nDone. Converted: {total_converted}  Skipped: {total_skipped}")
    if not args.dry_run:
        print(f"Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
