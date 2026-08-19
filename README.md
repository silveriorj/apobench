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

The framework is extensible by design: optimizers register via a decorator,
datasets and LLM backends are added by extending a dispatch function, and
none of it requires touching the orchestration/audit/evaluation layers. See
[`examples/`](examples/) for guided walkthroughs of each extension point.

## Methods

27 registered optimizers, grouped by family (some entries below are aliases
registered against the same class — e.g. `funnel_lean` and `funnel_v4d` are
the same implementation under two names for historical continuity):

| Family | Members | Description |
|---|---|---|
| **SEE** | `see` | Cui et al. — 4-phase Init → Feedback → Fusion → Semantic. Ported from a prior best-performing implementation. |
| **SWIFT** | `swift`, `swift_v2` | Failure-guided improvement + trajectory search + racing, budget-comparable to SEE. `v2` rebuilds Phase 2 from operator-effectiveness audit data. |
| **APEX** | `apex`, `apex_v2`, `apex_lean` (alias `swift_apex_lean`), `apex_holdout` (deprecated alias, see below) | Adaptive expert-persona generation + UCB1 bandit operator selection. `v2` uses a historically-informed, variance-aware bandit; `lean` adds the one proven-cheap mechanism (held-out final selection) on top of `v2`. |
| **GAAPO** | `gaapo` | Sécheresse et al. (arXiv:2504.07157) — genetic algorithm with LLM-based operators. |
| **CAPO** | `capo` | Zehle et al. (arXiv:2504.16005) — cost-aware evolutionary search with Hoeffding-bound racing. |
| **GEPA** | `gepa` | Agrawal et al. (arXiv:2507.19457) — component-level evolution with free-form decomposition and Pareto-frontier selection. |
| **GSPE** | `gspe` | Grammar-guided structured prompt evolution with formal field specs. |
| **FUNNEL** | `funnel`, `funnel_v2`, `funnel_v3`, `funnel_v4a`, `funnel_v4b`, `funnel_v4c`, `funnel_v4d` (alias `funnel_lean`), `funnel_v5` (alias `funnel_wide`), `funnel_v6` (alias `funnel_indexed`), `funnel_v7` (alias `funnel_prime`) | An in-house lineage: UCB1-scheduled search over a shrinking operator pool spanning 15+ published APO techniques. Each version adds one audited mechanism — guaranteed few-shot exploration, adaptive vs. unconditional operator families, batch-level racing, searchable eval mode — culminating in `v7`/`funnel_prime`, the synthesis of every proven mechanism in one class. |
| **Baseline** | `baseline_seed` | Zero-search reference: evaluates the seed prompt once, no optimization. The number every other method's gain is measured against. |

`apex_holdout` is a deprecated alias — the held-out-selection mechanism it
introduced was ported into `apex`/`apex_v2`/`swift`/`swift_v2` directly via
`HoldoutSelectionMixin` (`pof/optimizers/holdout.py`); use those instead.

```bash
pof list   # print the live registry — always the source of truth
```

## Datasets

- **`bbh`** — BigBench-Hard, 27 tasks (boolean_expressions, causal_judgement,
  dyck_languages, hyperbaton, ... — see `BBH_TASKS` in `pof/datasets/loader.py`)
- **`gsm8k`**, **`svamp`** — grade-school and elementary math word problems
- **`humaneval`** — code generation
- **`livebench_math`**, **`livebench_coding`** (alias `livecodebench`) — contamination-resistant benchmarks
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

The methods × datasets × seeds matrix is what makes a comparison
"under the APOBench protocol":

```bash
python experiments/run_swift_apex.py \
    --methods see apex swift \
    --datasets bbh humaneval \
    --seeds 42 123 7 \
    --output-dir outputs/my_sweep \
    --dry-run   # preview before spending any LLM calls
```

See [`examples/02_multi_seed_sweep/`](examples/02_multi_seed_sweep/) for a
full walkthrough and the key flags (`--strip-system-prompt`,
`--generic-prompt`, `--dev-test-split`, ...).

## Extending APOBench

- **New optimizer** → [`examples/03_custom_optimizer/`](examples/03_custom_optimizer/):
  subclass `BaseOptimizer`, decorate with `@register_optimizer(...)`, add the
  import to `pof/optimizers/__init__.py`'s `_load_all()`.
- **New dataset** → [`examples/04_custom_dataset/`](examples/04_custom_dataset/):
  arbitrary local JSON works with zero code changes; a named dataset needs
  an `elif` branch in `pof/datasets/loader.py`.
- **New LLM backend** → same if/elif dispatch pattern, in `pof/llm/factory.py`.

## Architecture

```
pof/
├── core/          # Types (PromptRecord, OptimizationResult), exceptions — no outward deps
├── audit/         # Tracker, history, lineage exporters
├── config/        # Pydantic v2 schemas, YAML/JSON loader with env-var substitution
├── llm/           # Backends (HuggingFace, OpenAI-compatible, Ollama), factory
├── evaluation/    # Scoring, answer extraction, Hoeffding racing
├── datasets/      # Dataset loading and management
├── optimizers/    # All 27 registered APO algorithms
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

Coverage today: `pof/core/`, `pof/config/`, `pof/evaluation/scoring.py`, and
`apex`'s bandit logic are tested. Most optimizer implementations, `pof/llm/`,
`pof/datasets/loader.py`, and `pof/orchestration/runner.py` are not yet —
contributions welcome (see [`CONTRIBUTING.md`](CONTRIBUTING.md)).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for commit conventions and how to
add an optimizer or dataset.

## License

MIT — see [`LICENSE`](LICENSE).
