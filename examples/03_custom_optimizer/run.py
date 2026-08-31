"""Run my_optimizer.py against a real task, in-process.

Importing my_optimizer BEFORE building the orchestrator is what triggers
its @register_optimizer("my_method") decorator — `pof run` on the CLI runs
in a separate process and would not see it without adding the import to
pof/optimizers/__init__.py's _load_all() (see my_optimizer.py's docstring).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import my_optimizer  # noqa: F401  (import triggers registration)

from pof.config.loader import load_config
from pof.orchestration.runner import RunOrchestrator

if __name__ == "__main__":
    config = load_config(str(Path(__file__).parent / "config.yaml"))
    result = RunOrchestrator(config).run()
    print(f"\nbest_score={result.best_score:.4f}  test_score={result.test_score:.4f}")
    print(f"best_prompt={result.best_prompt[:120]!r}...")
