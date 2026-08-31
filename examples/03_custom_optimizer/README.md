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
   only fires when the module is actually imported. Two ways to make that happen,
   depending on where your code lives:

   - **Your own package, outside this repo (the common case).** Declare an
     entry point in your `pyproject.toml`:
     ```toml
     [project.entry-points."apobench.optimizers"]
     my_method = "my_pkg.my_optimizer:MyOptimizer"
     ```
     Installing your package is all it takes — `pof list` and
     `get_optimizer("my_method")` discover it via
     [`pof/plugins.py`](../../pof/plugins.py). No edit inside `pof/` at all.
     Not packaged yet? Set `APOBENCH_PLUGINS=my_pkg.my_optimizer` in the
     environment instead — same discovery, no install needed.
   - **Contributing directly into this repository.** Add it to `_load_all()` in
     [`pof/optimizers/__init__.py`](../../pof/optimizers/__init__.py):
     ```python
     from pof.optimizers import my_optimizer  # noqa: F401
     ```
     This is a second, separate path (the built-in registry) — use it only for
     a PR against this repo, not for a method living in your own project.

   Either way, set `tier` on the class: `"contribution"` for the method your
   work is about, `"baseline"` only for a reimplementation of *published* work
   (cite it in the docstring), otherwise the default `"in_house"`. This is what
   lets `pof list` — and anyone reading your results — tell "beat the
   literature" from "beat our own prior design" at a glance.

2. **`HoldoutSelectionMixin` is optional, not required.** Some optimizers in
   this project mix in [`pof/optimizers/holdout.py`](../../pof/optimizers/holdout.py)
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

To see the entry-point/plugin path instead — the one an external package
actually uses, with zero edits inside `pof/` and no import in the calling
script at all:

```bash
python examples/03_custom_optimizer/run_via_plugin.py
```

This runs `pof list` as a subprocess with `APOBENCH_PLUGINS=my_optimizer`,
proving discovery happens through `pof/plugins.py` rather than through the
calling script importing anything.

## Next steps

- [`../04_custom_dataset/`](../04_custom_dataset/) — bring your own task.
- [`../02_multi_seed_sweep/`](../02_multi_seed_sweep/) — once registered
  via step 1, your method works in `--methods my_method ...` sweeps too.
