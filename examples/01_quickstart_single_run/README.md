# Quickstart: single optimizer, single task, single seed

The smallest possible APOBench run: one method (`see`, the cheapest
registered optimizer), one BBH task, one seed, tight budget caps. Good
for confirming your install works before committing to a real sweep.

## Run

```bash
# from the repository root
pof run -c examples/01_quickstart_single_run/config.yaml
```

Or override fields from the CLI without touching the YAML:

```bash
pof run -c examples/01_quickstart_single_run/config.yaml -m gepa --model Qwen/Qwen3-4B-Instruct-2507
```

## What to expect

- Runtime: a few minutes on a single consumer GPU (small model, tiny
  sample sizes, `max_generations: 3`).
- Output: `outputs/01_quickstart/` — one `result_*.json` per run, plus an
  audit trail. See [`docs/OUTPUT_SCHEMA.md`](../../docs/OUTPUT_SCHEMA.md)
  for the exact layout.
- This config intentionally uses small `sample_size`/`full_eval_size`
  values and hard budget caps (`budget.time_seconds`, `budget.max_calls`)
  so a first run is cheap. Real experiments (see
  [`../02_multi_seed_sweep/`](../02_multi_seed_sweep/)) use larger samples
  and multiple seeds.

## Next steps

- [`../02_multi_seed_sweep/`](../02_multi_seed_sweep/) — the real
  methods × datasets × seeds protocol.
- [`../03_custom_optimizer/`](../03_custom_optimizer/) — write your own method.
- [`../04_custom_dataset/`](../04_custom_dataset/) — bring your own task.
