# POF — Prompt Optimization Framework

Low-cost, auditable prompt evolution with full lineage tracking.

## Overview

POF merges the best of two implementations:
- **Performance & Efficiency**: Ported from Projeto's best SEE implementation — proven with small models (qwen3-4B) at low computational cost
- **Traceability & Architecture**: Full per-candidate lineage tracking with SHA-256 hashing, generation snapshots, and auditable JSON/CSV exports

## Features

- **6 Optimization Methods**: SEE, SWIFT, APEX, GAAPO, CAPO, GEPA
- **Full Audit Trail**: Every candidate tracked with UUID, SHA-256 hash, parent lineage, operator used, generation info
- **Efficient Evaluation**: Hoeffding racing for early termination of inferior candidates
- **Batch Inference**: Efficient batched generation for local HuggingFace models
- **Flexible Backends**: HuggingFace (local) and OpenAI (API) support
- **Pydantic Config**: Type-safe configuration with YAML/JSON loading and env-var substitution
- **CLI Interface**: Run optimizations and benchmarks from the command line

## Installation

```bash
pip install -e .
```

## Quick Start

### CLI Usage

```bash
# List available optimizers
pof list

# Run SEE optimization on a BigBench-Hard task
pof run -m see --dataset bbh --task boolean_expressions --model Qwen/Qwen2.5-3B-Instruct

# Benchmark multiple methods
pof benchmark --methods see swift apex --dataset bbh --task navigate

# Use a config file
pof run -c config.yaml
```

### Python API

```python
from pof.config import load_config
from pof.orchestration import RunOrchestrator

# Load config and run
config = load_config("config.yaml")
orchestrator = RunOrchestrator(config)
result = orchestrator.run()

print(f"Best score: {result.best_score:.4f}")
print(f"Best prompt: {result.best_prompt}")
```

### Programmatic Usage

```python
from pof.llm import create_llm
from pof.datasets import load_dataset_by_name
from pof.evaluation import Evaluator
from pof.optimizers import get_optimizer
from pof.config.schemas import LLMConfig

# Setup
llm = create_llm(LLMConfig(model_name="Qwen/Qwen2.5-3B-Instruct"))
dataset = load_dataset_by_name("bbh", task="boolean_expressions")
evaluator = Evaluator(llm, task_type=dataset.task_type)

# Run optimizer
SEE = get_optimizer("see")
optimizer = SEE(
    llm=llm,
    dataset=dataset,
    evaluator=evaluator,
    seed_prompt="Evaluate the boolean expression and output True or False.",
)
result = optimizer.optimize()

# Access audit trail
tracker = optimizer.tracker
tracker.save_json()  # Full lineage in JSON
tracker.save_csv()   # Generation metrics in CSV
```

## Configuration

Create a `config.yaml`:

```yaml
llm:
  backend: huggingface
  model_name: Qwen/Qwen2.5-3B-Instruct
  device: auto
  thinking_mode: false

evaluation:
  sample_size: 50
  max_new_tokens: 32
  racing_enabled: true

optimizer:
  method: see
  population_size: 5
  seed_prompt: "Solve the following task step by step."

dataset:
  name: bbh
  task: boolean_expressions
  num_samples: 100

output_dir: outputs
seed: 42
```

## Methods

| Method    | Type                          | Description                                                            |
| --------- | ----------------------------- | ---------------------------------------------------------------------- |
| **SEE**   | Paper (Cui et al.)            | 4-phase: Init → Feedback → Fusion → Semantic. Best proven performance. |
| **SWIFT** | Proposed                      | Failure-guided + trajectory + racing. Budget-comparable to SEE.        |
| **APEX**  | Proposed                      | Adaptive expert personas + operator selection.                         |
| **GSPE**  | Proposed                      | Grammar-guided structured prompt evolution with formal field specs.    |
| **GAAPO** | Reference (Sécheresse et al.) | Genetic algorithm with LLM-based operators.                            |
| **CAPO**  | Reference (Zehle et al.)      | Confidence-aware with Hoeffding bounds.                                |
| **GEPA**  | Proposed                      | Component-level evolution with free-form decomposition.                |

## Architecture

```
pof/
├── core/          # Types, exceptions (no outward deps)
├── audit/         # Tracker, history, lineage
├── config/        # Pydantic schemas, YAML/JSON loader
├── llm/           # Backends (HF, OpenAI), factory
├── evaluation/    # Scoring, evaluator, racing
├── datasets/      # BigBench loader, JSON datasets
├── optimizers/    # All APO algorithms
├── orchestration/ # Run management, benchmarking
└── cli.py         # Command-line interface
```

## Audit & Traceability

Every optimization run produces:
- **JSON audit trail**: Full candidate lineage with hashes, scores, operators, parent IDs
- **CSV metrics**: Per-generation best/mean scores, population size, operator counts
- **PromptRecord**: Each candidate gets a UUID, SHA-256 text hash, and full parent chain

```python
# Trace lineage of the best prompt
lineage = tracker.history.get_lineage(best_record.id)
for record in lineage:
    print(f"  {record.operator} → score={record.score:.3f}")
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check pof/
```

## License

MIT