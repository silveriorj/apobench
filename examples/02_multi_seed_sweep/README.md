# Multi-seed sweep: the APOBench protocol

The core evaluation pattern used throughout this project: a fixed grid of
**methods × datasets/tasks × seeds**, all run under identical LLM/evaluation/
budget settings from one shared config. This — not any single run — is what
"the APOBench protocol" refers to when cited in a paper: comparisons are only
meaningful when every method sees the same model, the same sample sizes, and
the same budget caps.

## Run

```bash
bash examples/02_multi_seed_sweep/run.sh
```

Or invoke the underlying runner directly with your own matrix:

```bash
python experiments/run_swift_apex.py \
    --methods see apex swift \
    --datasets bbh humaneval \
    --tasks boolean_expressions hyperbaton \
    --config examples/02_multi_seed_sweep/config.yaml \
    --seeds 42 123 7 \
    --output-dir outputs/my_sweep \
    --dry-run   # preview the matrix without spending any LLM calls
```

Drop `--dry-run` once the printed matrix looks right.

## Why 3 seeds

Optimizer search involves LLM sampling with real stochasticity — a single
seed can't distinguish a genuine improvement from noise. `[42, 123, 7]` is
this project's convention (see `experiments/configs/swift_apex_benchmark.yaml`);
use whatever seed set fits your own statistical design, but keep it fixed
across the methods you're comparing.

## Key flags

Run `python experiments/run_swift_apex.py --help` for the full list. The ones
that matter most for reproducing or varying the protocol:

- `--methods` / `--datasets` / `--tasks` / `--seeds` / `--models` — the matrix axes.
- `--strip-system-prompt` / `--simple-system-prompt` — isolate how much of a
  result comes from evaluation scaffolding vs. the optimized prompt itself.
- `--generic-prompt` — replace the task's fetched seed prompt with a fixed
  generic instruction, to isolate prompt-engineering gains from raw model capability.
- `--dev-test-split` — control the held-out test fraction (default preserves
  a fixed-size test split, capped at 115 samples).

## Output

Each `(method, dataset/task, seed)` cell produces one `result_*.json` under
`outputs/02_multi_seed_sweep/<model>/<method>/<dataset>/<task>/seed_<N>/`, plus
a top-level `experiment_results.json` summary. See
[`docs/OUTPUT_SCHEMA.md`](../../docs/OUTPUT_SCHEMA.md) for the full schema.

## Next steps

- [`../03_custom_optimizer/`](../03_custom_optimizer/) — add your own method to the matrix.
- [`../04_custom_dataset/`](../04_custom_dataset/) — add your own task to the matrix.
