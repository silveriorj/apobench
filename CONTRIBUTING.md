# Contributing to APOBench

## Commit messages

This project uses Conventional-Commits-style prefixes — already the organic
convention in the git history, formalized here:

| Prefix | Use for |
|---|---|
| `feat:` | A new optimizer, dataset, CLI flag, or other user-facing capability. |
| `fix:` | A correctness bug fix (e.g. `fix: score-scale mixing was still live in the actual generation-driving path`). |
| `perf:` | A performance change with no behavior change (e.g. `perf: tighten racing threshold from baseline_score to elite-minus-slack`). |
| `docs:` | README, example, or docstring changes only. |
| `refactor:` | Code restructuring with no behavior or performance change. |
| `test:` | Adding or fixing tests. |
| `chore:` | Repo housekeeping — file moves, gitignore, dependency bumps. |

One logical change per commit. If a fix also required a behavior-changing
refactor to land cleanly, that's normal — just say so in the body, the way
existing commits document *why* a bug existed and *what specifically* broke
(see `git log` for the house style: explain the failure mode, not just the
diff).

## Adding a new optimizer

See [`examples/03_custom_optimizer/`](examples/03_custom_optimizer/) for the
full walkthrough. Short version:

1. Subclass `BaseOptimizer` (`pof/optimizers/base.py`), implement
   `_init_population()` and `_step()`.
2. Decorate the class with `@register_optimizer("your_name")`.
3. Set `tier` on the class: `"contribution"` if this is the method your work
   is about, `"baseline"` only if it reimplements a *published* method (cite it
   in the docstring), otherwise leave the default `"in_house"`. `pof list`
   groups by this — it is how a reader tells "this beat the literature" from
   "this beat our own prior work" without reading every docstring.
4. Register it: contributing this optimizer *into this repository*, add
   `from pof.optimizers import your_module  # noqa: F401` to `_load_all()` in
   `pof/optimizers/__init__.py` (registration only happens on import, the
   decorator alone isn't enough). Using APOBench from your *own* package
   instead, skip this step entirely — declare an `apobench.optimizers` entry
   point in your own `pyproject.toml`, or set `APOBENCH_PLUGINS` to your
   module; see `pof/plugins.py` and the README's Extending section.
5. `HoldoutSelectionMixin` (`pof/optimizers/holdout.py`) is optional — only
   mix it in if your method's final selection argmaxes over the same dev
   pool the search optimized against.

## Adding a new dataset

See [`examples/04_custom_dataset/`](examples/04_custom_dataset/). Arbitrary
local JSON works with zero code changes.

For a named dataset contributed *into this repository*, add an `elif` branch +
loader function to `load_dataset_by_name()` in `pof/datasets/loader.py`. From
your *own* package, register an `apobench.datasets` entry point instead — no
edit to this repo needed; see the README's Extending section. The same split
applies to score functions (`apobench.scorers`) and LLM backends
(`apobench.backends`).

## Tests

New optimizer tests should live either co-located next to the source
(`pof/optimizers/test_<name>.py` — `pyproject.toml`'s `testpaths` already
includes `pof/optimizers`) or under `tests/` for anything not
optimizer-specific.

Before opening a PR:

```bash
pytest -v
ruff check pof/
```

## What's not covered yet

`pof/optimizers/`, `pof/llm/`, `pof/datasets/loader.py`, and
`pof/orchestration/runner.py` have no test coverage today — contributions
closing these gaps are especially welcome.
