# Bringing your own dataset

There are two ways to add a dataset, and they are **not** symmetric with how
optimizers are registered — worth knowing up front so you don't go looking
for a decorator that isn't there.

## Path A — arbitrary local JSON (no code change)

`load_dataset_by_name` ([`pof/datasets/loader.py`](../../pof/datasets/loader.py))
already falls through to `_load_json()` for any `name` that ends in `.json`
or exists as a path on disk. Just point `dataset.name` at your file:

```yaml
dataset:
  name: examples/04_custom_dataset/sample_data.json
```

**Expected JSON schema** — either a flat list:

```json
[{"input": "...", "target": "..."}, ...]
```

(see [`sample_data.json`](sample_data.json) for a working example — a toy
prime-number yes/no task), or pre-split:

```json
{"train": [...], "dev": [...], "test": [...]}
```

For the flat-list form, the loader carves a fixed-seed `test` split first
(so it's identical across a seed sweep, matching every built-in dataset),
then splits the remainder into `train`/`dev` with your run seed. Task type
(`boolean`/`mcq`/`math`/`text`/...) is auto-detected from the `target`
values — see `_detect_task_type()` in the same file if you want to
understand or override the detection.

Run it:

```bash
pof run -c examples/04_custom_dataset/config.yaml
```

## Path B — registering a named dataset (code change)

If you want `--datasets mydata` to work from `experiments/run_swift_apex.py`
sweeps without passing a file path every time, add a branch to
`load_dataset_by_name()`:

```python
elif name.lower() == "mydata":
    return _load_mydata(num_samples, seed)
```

plus a `_load_mydata()` function alongside the existing `_load_bbh`,
`_load_gsm8k`, etc. in the same file. This is a plain if/elif dispatch, not
a decorator registry — unlike optimizers
([`../03_custom_optimizer/`](../03_custom_optimizer/)), there's no
`@register_dataset`. LLM backends follow the same if/elif pattern in
[`pof/llm/factory.py`](../../pof/llm/factory.py) if you're adding a new
backend instead.

## Next steps

- [`../02_multi_seed_sweep/`](../02_multi_seed_sweep/) — run your dataset
  across multiple methods/seeds.
- [`../03_custom_optimizer/`](../03_custom_optimizer/) — pair it with a
  custom optimizer.
