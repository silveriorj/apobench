# Notebooks

Analysis notebooks over `outputs/` run results (gitignored, not published —
see [`docs/OUTPUT_SCHEMA.md`](../docs/OUTPUT_SCHEMA.md) to reproduce the
directory structure these expect).

| Notebook | What it covers |
|---|---|
| `ultimate_analysis.ipynb` | Full statistical protocol: Friedman + Nemenyi CD, Holm-corrected pairwise Wilcoxon, bootstrap CIs, Cliff's delta, Bradley-Terry, Pareto — across methods and model families. Most complete version; start here. |
| `analysis.ipynb` | Earlier iteration of the same cross-model comparative analysis; kept for provenance. |
| `multi_model_analysis.ipynb` | Cross-model generalization analysis (multiple base LLMs, same protocol). |
| `results_analysis.ipynb` | Per-run dashboard: accuracy, seed-variance stability, token/call efficiency, wall-clock speed, dev↔test generalization gap. Set `RUN_CUTOFF` to exclude stale runs from before the last clean launch. |

`aggregated_stats.csv` / `llm_usage_stats.csv` are cached intermediate outputs
some notebooks read to avoid re-scanning `outputs/` on every run — regenerate
by re-running the aggregation cells if `outputs/` has changed. `image.png` is
a figure referenced by one of the notebooks above.
