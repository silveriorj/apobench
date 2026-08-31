# Multi-seed sweep: the APOBench protocol

The core evaluation pattern used throughout this project: a fixed grid of
**methods × seeds**, all run under identical LLM/evaluation/budget settings
from one shared config. This — not any single run — is what "the APOBench
protocol" refers to when cited in a paper: comparisons are only meaningful
when every method sees the same model, the same sample sizes, and the same
budget caps.

## Run

```bash
bash examples/02_multi_seed_sweep/run.sh
```

Or invoke `pof run` directly, one call per `(method, seed)` cell:

```bash
pof run -c examples/02_multi_seed_sweep/config.yaml -m gepa \
    -o outputs/my_sweep/gepa/seed_42
```

`pof run` doesn't take a seed flag directly — set `seed:` in the config (or a
copy of it) before each call, the way `run.sh` does with `sed`.

## Why 3 seeds

Optimizer search involves LLM sampling with real stochasticity — a single
seed can't distinguish a genuine improvement from noise. `[42, 123, 7]` is
this project's convention; use whatever seed set fits your own statistical
design, but keep it fixed across the methods you're comparing.

## Key config fields

Everything a sweep varies lives in the YAML config (`-c`), not CLI flags —
see [`config.yaml`](config.yaml):

- `optimizer.method` (or `-m` on the CLI) / `seed` — the matrix axes this
  example loops over.
- `evaluation.sample_size` / `full_eval_size` — dev/test split sizes.
- `budget.*` — hard caps on wall-clock time, LLM calls, and tokens, shared
  across every cell so no method gets a bigger budget than another.

## Output

Each `(method, seed)` cell produces one result JSON under
`outputs/02_multi_seed_sweep/<method>/seed_<N>/`. See
[`docs/OUTPUT_SCHEMA.md`](../../docs/OUTPUT_SCHEMA.md) for the full schema.

## Next steps

- [`../03_custom_optimizer/`](../03_custom_optimizer/) — add your own method to the matrix.
- [`../04_custom_dataset/`](../04_custom_dataset/) — add your own task to the matrix.
