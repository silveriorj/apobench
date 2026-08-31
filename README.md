# APOBench — Automatic Prompt Optimization Benchmark

Low-cost, auditable prompt evolution with full lineage tracking. Package
name is `pof` (Prompt Optimization Framework) for historical reasons — the
project's public identity as a benchmark protocol is **APOBench**.

## What is APOBench

APOBench is a fixed evaluation protocol for comparing automatic prompt
optimization (APO) methods: a shared **methods × datasets × seeds × models**
matrix, run under identical LLM/evaluation/budget settings so results are
comparable across methods rather than confounded by different setups. Every
run produces a full per-candidate audit trail (lineage, SHA-256 hashing,
generation snapshots), so a reported result traces back to the exact search
path that produced it — not just a final number.

The framework is extensible by design: a third-party package registers an
optimizer, dataset, scorer, or LLM backend via a Python entry point (or the
`APOBENCH_PLUGINS` environment variable during development) — no fork, no edit
inside `pof/`. See [`pof/plugins.py`](pof/plugins.py) for the discovery
mechanism and [`examples/`](examples/) for guided walkthroughs of each
extension point.

## Methods

Built-in optimizers are reimplementations of published APO methods, plus a
zero-search reference. Every optimizer carries a `tier` (`pof list` groups by
it) that says what kind of evidence a comparison against it is — `baseline`
means a reimplementation of a *published* method, cited by name in the class
docstring, so beating one is a claim about the literature. Extending the
framework with your own method (see below) registers it under `in_house` or
`contribution` by default, so `pof list` always shows what kind of comparison
your own results support.

| Tier | Members | Description |
|---|---|---|
| **baseline** | `see` | Cui et al. — 4-phase Init → Feedback → Fusion → Semantic. Ported from a prior best-performing implementation. |
| **baseline** | `gaapo` | Sécheresse et al. (arXiv:2504.07157) — genetic algorithm with LLM-based operators. |
| **baseline** | `capo` | Zehle et al. (arXiv:2504.16005) — cost-aware evolutionary search with Hoeffding-bound racing. |
| **baseline** | `gepa` | Agrawal et al. (arXiv:2507.19457) — component-level evolution with free-form decomposition and Pareto-frontier selection. |
| **baseline** | `baseline_seed` | Zero-search reference: evaluates the seed prompt once, no optimization. The number every other method's gain is measured against. |

```bash
pof list   # print the live registry, grouped by tier — always the source of truth
```

## Datasets

- **`bbh`** — BigBench-Hard, 27 tasks (boolean_expressions, causal_judgement,
  dyck_languages, hyperbaton, ... — see `BBH_TASKS` in `pof/datasets/loader.py`)
- **`gsm8k`**, **`svamp`** — grade-school and elementary math word problems
- **`humaneval`**, **`livebench_coding`** — code generation. **Security note:**
  scoring these executes LLM-generated code locally
  (`subprocess.run([sys.executable, "-c", program], timeout=30)`), unsandboxed,
  as the user running the benchmark. This is standard practice for HumanEval-style
  harnesses, but it means an adversarial or buggy generation runs with your
  permissions. Run in a container or VM if the model/prompt source is untrusted.
- **`livebench_math`**, **`livebench_coding`** (alias `livecodebench`) — contamination-resistant benchmarks (coding shares the execution note above)
- **Arbitrary local JSON** — any file path ending `.json`, see [`examples/04_custom_dataset/`](examples/04_custom_dataset/)

## Installation

```bash
pip install -e .
```

## Quickstart

```bash
pof list                                    # available optimizers
pof run -c examples/01_quickstart_single_run/config.yaml
```

See [`examples/01_quickstart_single_run/`](examples/01_quickstart_single_run/)
for a walkthrough, or the Python API below to call it programmatically.

### Python API

```python
from pof.config import load_config
from pof.orchestration import RunOrchestrator

config = load_config("config.yaml")
result = RunOrchestrator(config).run()

print(f"Best score: {result.best_score:.4f}")
print(f"Test score: {result.test_score:.4f}")
```

### Lower-level API

```python
from pof.llm import create_llm
from pof.datasets import load_dataset_by_name
from pof.evaluation import Evaluator
from pof.optimizers import get_optimizer
from pof.config.schemas import LLMConfig

llm = create_llm(LLMConfig(model_name="Qwen/Qwen3-4B-Instruct-2507"))
dataset = load_dataset_by_name("bbh", task="boolean_expressions")
evaluator = Evaluator(llm, task_type=dataset.task_type)

SEE = get_optimizer("see")
optimizer = SEE(
    llm=llm, dataset=dataset, evaluator=evaluator,
    seed_prompt="Evaluate the boolean expression and output True or False.",
)
result = optimizer.optimize()

optimizer.tracker.save_json()  # full lineage
optimizer.tracker.save_csv()   # per-generation metrics
```

## Running a benchmark sweep

The methods × seeds matrix is what makes a comparison "under the APOBench
protocol" — see [`examples/02_multi_seed_sweep/`](examples/02_multi_seed_sweep/)
for a full walkthrough, including the config fields that isolate scaffolding
effects, seed-prompt effects, and dev/test split size.

## Extending APOBench

Four extension points, each discovered at runtime through `pof/plugins.py` —
either a Python entry point group (for an installed package) or the
`APOBENCH_PLUGINS` environment variable (a comma-separated list of modules to
import, for code that isn't packaged yet). Nothing inside `pof/` needs editing.

In your own package's `pyproject.toml`:

```toml
[project.entry-points."apobench.optimizers"]
my_method = "my_pkg.my_method:MyOptimizer"

[project.entry-points."apobench.datasets"]
my_data = "my_pkg.my_dataset:load_my_dataset"

[project.entry-points."apobench.scorers"]
my_task_type = "my_pkg.my_scorer:score_my_task"

[project.entry-points."apobench.backends"]
my_backend = "my_pkg.my_backend:make_my_backend"
```

Or, without packaging anything, point APOBench at a local module:

```bash
APOBENCH_PLUGINS=my_experiment.methods pof run --method my_method ...
```

- **New optimizer** → [`examples/03_custom_optimizer/`](examples/03_custom_optimizer/):
  subclass `BaseOptimizer`, decorate with `@register_optimizer(...)`. Set
  `tier` on the class (`"contribution"`, `"baseline"`, or `"in_house"` —
  see Methods above) so `pof list` reports what kind of comparison it supports.
- **New dataset** → [`examples/04_custom_dataset/`](examples/04_custom_dataset/):
  arbitrary local JSON works with zero code changes; a named dataset registers
  a loader function matching `load_dataset_by_name`'s signature.
- **New score function** → a `(prediction: str, target: str) -> int` function,
  registered against a `task_type` name.
- **New LLM backend** → a factory function taking `LLMConfig` and returning a
  `BaseLLM`.

Built-in dispatch (the optimizers bundled with this repo, the `bbh`/`gsm8k`/
etc. datasets, `math`/`mcq`/etc. scorers, `huggingface`/`openai`/`ollama`
backends) is unaffected — third-party registration is checked first, and a
name collision with a built-in is rejected with a warning rather than silently
overriding it, so an installed plugin can never change what a published result
using a built-in name means.

## Architecture

```
pof/
├── core/          # Types (PromptRecord, OptimizationResult), exceptions — no outward deps
├── audit/         # Tracker, history, lineage exporters
├── config/        # Pydantic v2 schemas, YAML/JSON loader with env-var substitution
├── llm/           # Backends (HuggingFace, OpenAI-compatible, Ollama), factory
├── evaluation/    # Scoring, answer extraction, Hoeffding racing
├── datasets/      # Dataset loading and management
├── optimizers/    # Built-in APO algorithms — reimplementations tiered `baseline`
├── orchestration/ # RunOrchestrator — run management, benchmarking
└── cli.py         # `pof list` / `pof run` / `pof benchmark`
```

## Reproducibility

Every run's output layout, result schema, and the fixed-seed test-split
convention are documented in [`docs/OUTPUT_SCHEMA.md`](docs/OUTPUT_SCHEMA.md) —
raw run data isn't published with this repo, but the schema is complete
enough to reproduce and compare against it independently.

## Audit & traceability

Every optimization run produces a full lineage trail alongside its result:

```python
lineage = tracker.history.get_lineage(best_record.id)
for record in lineage:
    print(f"  {record.operator} → score={record.score:.3f}")
```

## Citing APOBench

See [`CITATION.cff`](CITATION.cff) — GitHub's "Cite this repository" button
reads from it, or use:

```bibtex
@software{apobench2026,
  author = {Silverio, R. J.},
  title  = {APOBench: Automatic Prompt Optimization Benchmark},
  year   = {2026},
  url    = {https://github.com/silveriorj/apobench}
}
```

## Testing

```bash
pip install -e ".[dev]"
pytest -v
ruff check pof/
```

Coverage today: `pof/core/`, `pof/config/`, and `pof/evaluation/scoring.py`
are tested. Most optimizer implementations, `pof/llm/`, `pof/datasets/loader.py`,
and `pof/orchestration/runner.py` are not yet — contributions welcome (see
[`CONTRIBUTING.md`](CONTRIBUTING.md)).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for commit conventions and how to
add an optimizer or dataset.

## License

MIT — see [`LICENSE`](LICENSE).
