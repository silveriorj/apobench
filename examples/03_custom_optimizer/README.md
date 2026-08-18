# Writing a custom optimizer

Every optimizer in APOBench subclasses `BaseOptimizer`
([`pof/optimizers/base.py`](../../pof/optimizers/base.py)) and implements
exactly two methods:

- `_init_population()` — build and evaluate the starting candidate pool
- `_step()` — produce one generation's new candidate pool

`BaseOptimizer` handles everything else: the optimization loop, budget
tracking, audit/lineage recording, evaluation (with optional Hoeffding
racing), duplicate detection, and a library of shared generation techniques
(`_semantic_variation`, `_crossover`, `_lamarckian_generate`, `_create_record`,
`_evaluate_population`, ...) any subclass can call. See
[`my_optimizer.py`](my_optimizer.py) for a minimal complete example
(paraphrase-and-select).

## The two non-obvious steps

1. **Registration needs an import, not just the decorator.** `@register_optimizer("my_method")`
   only fires when the module is actually imported. For a real optimizer you're
   contributing, add it to `_load_all()` in
   [`pof/optimizers/__init__.py`](../../pof/optimizers/__init__.py):
   ```python
   from pof.optimizers import my_optimizer  # noqa: F401
   ```
   Without that line, `pof list` and `get_optimizer("my_method")` won't see
   it — the class exists and is decorated, but nothing ever imports the file.

2. **`HoldoutSelectionMixin` is optional, not required.** `apex`, `swift`, and
   `swift_v2` mix in [`pof/optimizers/holdout.py`](../../pof/optimizers/holdout.py)
   to correct winner's-curse bias when the final prompt is selected by argmax
   over the same dev set the search optimized against (it reserves a slice
   of dev purely for final selection). `BaseOptimizer` itself has no
   dependency on it — only use it if your method has that same
   optimize-then-argmax-on-the-same-pool selection pattern.

## Run this example

```bash
python examples/03_custom_optimizer/run.py
```

This runs in-process (not via the `pof` CLI) specifically so
`my_optimizer.py`'s registration takes effect without needing to edit
`pof/optimizers/__init__.py` just to try the example. Once you're
contributing a real optimizer, follow step 1 above so it's usable from the
CLI and `experiments/run_swift_apex.py` sweeps like everyone else's.

## Next steps

- [`../04_custom_dataset/`](../04_custom_dataset/) — bring your own task.
- [`../02_multi_seed_sweep/`](../02_multi_seed_sweep/) — once registered
  via step 1, your method works in `--methods my_method ...` sweeps too.
