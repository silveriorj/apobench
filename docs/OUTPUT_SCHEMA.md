# Output schema

`outputs/` is gitignored — no run data is published with the repository.
This document describes the layout well enough to reproduce comparable
results, and for `notebooks/*.ipynb` to know what to aggregate.

## Directory layout

A sweep launched via `experiments/run_swift_apex.py --output-dir outputs/my_sweep`
produces:

```
outputs/my_sweep/
  <model-slug>/                    # only present when --models is used
    <method>/
      <dataset>[_<task>]/
        seed_<N>/
          result_<method>_<dataset>.json     # OptimizationResult (see below)
          audit_<method>_<dataset>_<runid>.json  # full lineage/audit trail
          audit_<method>_<dataset>_<runid>.csv   # per-generation summary
  experiment_results.json          # sweep-level summary, one entry per run
```

`<model-slug>` is only inserted when the sweep loops over multiple models
(`--models A B`); a single-model run via `pof run` or a sweep with one
implicit model omits that level. `<dataset>_<task>` collapses to just
`<dataset>` for single-task datasets (`gsm8k`, `svamp`, `humaneval`);
BBH tasks get `bbh_<task_name>` (e.g. `bbh_boolean_expressions`).

**Skip-existing behavior**: a run directory containing any `result_*.json`
is skipped on re-launch (safe to resume an interrupted sweep). Result
filenames are matched by glob, not a fixed name — the file is named after
the optimizer *class*'s `.name` attribute, which can differ from the CLI
alias used to launch it. Don't parse the filename to infer the method; read
`method_name` from inside the JSON instead.

## `result_<method>_<dataset>.json` — one run's `OptimizationResult`

```json
{
  "method_name": "gepa",
  "dataset_name": "bbh_boolean_expressions",
  "best_prompt": "...",
  "best_score": 0.91,          // dev/optimization-time score
  "test_score": 0.8841,        // held-out test score (the number to report)
  "test_per_sample_details": [...],
  "optimization_history": [...],   // per-generation population snapshots
  "final_population": [...],
  "llm_usage": {
    "total_calls": 492,
    "total_input_tokens": ...,
    "total_output_tokens": ...,
    "total_tokens": 723557,
    "total_time_seconds": ...,
    "generation_calls": ...,
    "evaluation_calls": ...
  },
  "total_time": 3065.8,
  "num_iterations": 4,
  "config": {
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "eval_max_new_tokens": 32,
    "eval_temperature": 0.0,
    "eval_sample_size": 50,
    "full_eval_size": 115,
    "dataset_num_samples": 100,
    "dataset_task": "boolean_expressions",
    "run_seed": 42
    // ... plus every field from the run's RunConfig
  }
}
```

`test_score` is the number every experiment-log table in this project
reports — `best_score` is the optimization-time dev score and is subject to
winner's-curse bias if the same dev pool was both searched and argmaxed over
(see `HoldoutSelectionMixin`, `pof/optimizers/holdout.py`).

## `experiment_results.json` — sweep-level summary

One entry per `(model, method, dataset/task, seed)` cell, keyed by
`"<model>_<method>_<dataset>_seed<N>"`:

```json
{
  "qwen3-4b-instruct-2507_apex_bbh_boolean_expressions_seed42": {
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "method": "gepa",
    "dataset": "bbh",
    "task": "boolean_expressions",
    "seed": 42,
    "best_score": 0.91,
    "test_score": 0.8841,
    "total_time": 3065.8,
    "llm_calls": 492,
    "total_tokens": 723557,
    "best_prompt": "..."
  }
}
```

A failed run's entry has an `"error"` field instead of scores — filter these
out before aggregating.

## Reproducing a fixed test split

Built-in datasets (`bbh`, `gsm8k`, `svamp`, `humaneval`) carve their test
split with a fixed `TEST_SPLIT_SEED = 42` regardless of the run seed — so
the held-out test set is identical across a seed sweep, and only
train/dev vary. Custom JSON datasets follow the same convention (see
[`examples/04_custom_dataset/`](../examples/04_custom_dataset/)). This is
what makes `test_score` comparable across seeds within one sweep.

## What the analysis notebooks expect

`notebooks/*.ipynb` (see [`notebooks/README.md`](../notebooks/README.md))
scan `outputs/` for `result_*.json` files, group by
`(model, method, dataset, task)`, and aggregate `test_score` across seeds —
mean, std, and the statistical comparisons described there. No additional
setup is needed beyond the layout above; just point a notebook's root path
at your own `outputs/<sweep_name>/`.
