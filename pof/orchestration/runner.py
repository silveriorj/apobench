"""RunOrchestrator — manages full optimization runs and benchmarks.

Provides:
- Single-run execution with full audit trail
- Multi-method benchmarking
- Result aggregation and export
"""
from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pof.config.schemas import RunConfig
from pof.core.types import OptimizationResult
from pof.datasets.loader import TaskDataset, load_dataset_by_name
from pof.evaluation.evaluator import Evaluator
from pof.evaluation.scoring import create_score_function
from pof.llm.base import BaseLLM
from pof.llm.factory import create_llm
from pof.optimizers import get_optimizer
from pof.core.budget import BudgetManager

logger = logging.getLogger(__name__)


class RunOrchestrator:
    """Orchestrates optimization runs with full configuration management.

    Usage:
        config = load_config("config.yaml")
        orchestrator = RunOrchestrator(config)
        result = orchestrator.run()
    """

    def __init__(self, config: RunConfig):
        self.config = config
        self._llm: Optional[BaseLLM] = None
        self._dataset: Optional[TaskDataset] = None
        self._evaluator: Optional[Evaluator] = None

    def run(self) -> OptimizationResult:
        """Execute a single optimization run.

        Returns:
            OptimizationResult with best prompt, scores, and audit trail.
        """
        # Set random seed
        random.seed(self.config.seed)
        try:
            import numpy as np
            np.random.seed(self.config.seed)
        except ImportError:
            pass

        # Initialize components
        llm = self._get_llm()
        dataset = self._get_dataset()
        evaluator = self._get_evaluator(llm, dataset)

        # Get optimizer class
        optimizer_cls = get_optimizer(self.config.optimizer.method)

        # Create optimizer instance
        # Note: num_iterations is passed via params if needed; some optimizers
        # (SWIFT, APEX) hardcode their own iteration count.
        init_kwargs = {
            "llm": llm,
            "dataset": dataset,
            "evaluator": evaluator,
            "population_size": self.config.optimizer.population_size,
            "seed_prompt": self.config.optimizer.seed_prompt,
            "eval_sample_size": self.config.evaluation.sample_size,
            "output_dir": self.config.output_dir,
            **self.config.optimizer.params,
        }
        # Only pass num_iterations if explicitly set (non-default) and not already in params
        if self.config.optimizer.num_iterations != 3 and "num_iterations" not in self.config.optimizer.params:
            init_kwargs["num_iterations"] = self.config.optimizer.num_iterations
        optimizer = optimizer_cls(**init_kwargs)

        # Run optimization
        logger.info(
            f"Starting run: method={self.config.optimizer.method}, "
            f"dataset={dataset.name}, model={self.config.llm.model_name}"
        )
        result = optimizer.optimize()

        # Final test evaluation on held-out samples
        test_samples = dataset.get_eval_samples("test", n=self.config.evaluation.full_eval_size)
        if test_samples and result.best_prompt:
            logger.info(
                f"[Test eval] {len(test_samples)} samples on best prompt "
                f"(dev score={result.best_score:.4f})"
            )
            test_result = evaluator.evaluate(result.best_prompt, test_samples)
            result.test_score = test_result.score
            logger.info(f"[Test eval] test_score={result.test_score:.4f}")
        else:
            logger.warning("[Test eval] skipped — no test samples or no best prompt")

        # Save audit trail
        optimizer.tracker.save_json()
        optimizer.tracker.save_csv()

        # Save result summary
        self._save_result(result)

        return result

    def benchmark(
        self, methods: Optional[List[str]] = None
    ) -> Dict[str, OptimizationResult]:
        """Run multiple methods on the same dataset for comparison.

        Args:
            methods: List of method names. If None, uses all registered.

        Returns:
            Dict mapping method name to OptimizationResult.
        """
        from pof.optimizers import list_optimizers

        if methods is None:
            methods = list_optimizers()

        results: Dict[str, OptimizationResult] = {}
        dataset = self._get_dataset()

        for method_name in methods:
            logger.info(f"\n{'='*60}\nBenchmarking: {method_name}\n{'='*60}")

            # Reset LLM usage for each method
            llm = self._get_llm()
            llm.reset_usage()
            evaluator = self._get_evaluator(llm, dataset)

            try:
                optimizer_cls = get_optimizer(method_name)
                optimizer = optimizer_cls(
                    llm=llm,
                    dataset=dataset,
                    evaluator=evaluator,
                    population_size=self.config.optimizer.population_size,
                    seed_prompt=self.config.optimizer.seed_prompt,
                    eval_sample_size=self.config.evaluation.sample_size,
                    output_dir=self.config.output_dir,
                )
                result = optimizer.optimize()
                results[method_name] = result

                # Save individual audit
                optimizer.tracker.save_json()

                logger.info(
                    f"  {method_name}: score={result.best_score:.4f}, "
                    f"time={result.total_time:.1f}s, "
                    f"calls={result.llm_usage.total_calls if result.llm_usage else 0}"
                )
            except Exception as e:
                logger.error(f"  {method_name} FAILED: {e}")
                results[method_name] = OptimizationResult(
                    method_name=method_name,
                    dataset_name=dataset.name,
                    best_prompt="",
                    best_score=0.0,
                    config={"error": str(e)},
                )

        # Save benchmark summary
        self._save_benchmark_summary(results)
        return results

    def _get_llm(self) -> BaseLLM:
        """Get or create LLM instance and attach budget manager."""
        if self._llm is None:
            self._llm = create_llm(self.config.llm)
            # Attach budget manager (hard caps)
            try:
                budget_mgr = BudgetManager(self.config.budget)
                budget_mgr.attach_llm(self._llm)
                self._llm.attach_budget(budget_mgr)
            except Exception as e:
                logger.warning(f"Failed to attach budget manager: {e}")
        return self._llm

    def _get_dataset(self) -> TaskDataset:
        """Get or create dataset instance."""
        if self._dataset is None:
            self._dataset = load_dataset_by_name(
                name=self.config.dataset.name,
                task=self.config.dataset.task,
                num_samples=self.config.dataset.num_samples,
                seed=self.config.seed,
            )
        return self._dataset

    def _get_evaluator(self, llm: BaseLLM, dataset: TaskDataset) -> Evaluator:
        """Create evaluator instance."""
        score_fn = create_score_function(dataset.task_type)
        return Evaluator(
            llm=llm,
            score_fn=score_fn,
            task_type=dataset.task_type,
            max_new_tokens=self.config.evaluation.max_new_tokens,
            temperature=self.config.evaluation.temperature,
            batch_size=self.config.evaluation.batch_size,
        )

    def _save_result(self, result: OptimizationResult) -> None:
        """Save optimization result to JSON."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"result_{result.method_name}_{result.dataset_name}.json"
        path = output_dir / filename

        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Result saved to: {path}")

    def _save_benchmark_summary(
        self, results: Dict[str, OptimizationResult]
    ) -> None:
        """Save benchmark comparison summary."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "config": {
                "model": self.config.llm.model_name,
                "dataset": self.config.dataset.name,
                "task": self.config.dataset.task,
            },
            "results": {
                name: {
                    "best_score": r.best_score,
                    "total_time": r.total_time,
                    "llm_calls": r.llm_usage.total_calls if r.llm_usage else 0,
                    "total_tokens": r.llm_usage.total_tokens if r.llm_usage else 0,
                    "best_prompt": r.best_prompt[:200],
                }
                for name, r in results.items()
            },
        }

        path = output_dir / "benchmark_summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Print comparison table
        logger.info("\n" + "=" * 70)
        logger.info(f"{'Method':<12} {'Score':<8} {'Time(s)':<10} {'LLM Calls':<12} {'Tokens':<10}")
        logger.info("-" * 70)
        for name, r in sorted(results.items(), key=lambda x: x[1].best_score, reverse=True):
            calls = r.llm_usage.total_calls if r.llm_usage else 0
            tokens = r.llm_usage.total_tokens if r.llm_usage else 0
            logger.info(f"{name:<12} {r.best_score:<8.4f} {r.total_time:<10.1f} {calls:<12} {tokens:<10}")
        logger.info("=" * 70)