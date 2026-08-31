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
from pof.core.exceptions import BudgetExceeded
from pof.core.types import OptimizationResult
from pof.datasets.loader import TaskDataset, load_dataset_by_name
from pof.evaluation.evaluator import Evaluator
from pof.evaluation.power import describe_power, minimum_detectable_effect
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
        # Note: population_size and num_iterations are only passed when explicitly
        # overridden away from the schema default; otherwise each optimizer keeps
        # its own paper-calibrated default (e.g. GAAPO's population_size=8).
        init_kwargs = {
            "llm": llm,
            "dataset": dataset,
            "evaluator": evaluator,
            "seed_prompt": self.config.optimizer.seed_prompt,
            "eval_sample_size": self.config.evaluation.sample_size,
            "output_dir": self.config.output_dir,
            **self.config.optimizer.params,
        }
        # Only pass population_size if explicitly set (non-default) and not already in params
        if self.config.optimizer.population_size != 5 and "population_size" not in self.config.optimizer.params:
            init_kwargs["population_size"] = self.config.optimizer.population_size
        # Only pass num_iterations if explicitly set (non-default) and not already in params
        if self.config.optimizer.num_iterations != 3 and "num_iterations" not in self.config.optimizer.params:
            init_kwargs["num_iterations"] = self.config.optimizer.num_iterations
        optimizer = optimizer_cls(**init_kwargs)

        # Run optimization
        logger.info(
            f"Starting run: method={self.config.optimizer.method}, "
            f"dataset={dataset.name}, model={self.config.llm.model_name}"
        )
        try:
            result = optimizer.optimize()
        except BudgetExceeded as be:
            # Defensive: well-behaved optimizers catch BudgetExceeded internally.
            # If one leaks it, recover from the tracker rather than losing the run.
            logger.warning(
                f"[Budget] {be.kind} exhausted before optimize() returned; "
                "recovering best result from tracker."
            )
            result = optimizer.tracker.to_optimization_result()

        # Detach budget from the LLM so the test eval is never blocked by
        # exhausted time/call/token caps from the optimization phase.
        llm.attach_budget(None)

        # Final test evaluation on held-out samples
        test_samples = dataset.get_eval_samples("test", n=self.config.evaluation.full_eval_size)
        if not result.best_prompt and self.config.optimizer.seed_prompt:
            logger.warning(
                "[Test eval] best prompt is empty — falling back to the seed prompt. "
                "This indicates an optimizer bug (empty operator output?)"
            )
            result.best_prompt = self.config.optimizer.seed_prompt
        if not test_samples:
            logger.warning("[Test eval] skipped — dataset has no test samples")
        elif not result.best_prompt:
            logger.warning(
                "[Test eval] skipped — best prompt is empty and no seed prompt to fall back to"
            )
        else:
            source = result.config.get(
                "selection_score_source",
                f"dev@{self.config.evaluation.sample_size}",
            )
            logger.info(
                f"[Test eval] {len(test_samples)} samples on best prompt "
                f"(selection score {source}={result.best_score:.4f})"
            )
            # Truncation is measured across the test eval specifically: a score
            # produced from answers that never finished reports the decode
            # budget, not the prompt, and the two are indistinguishable once the
            # run is saved. Snapshot the backend counter around the call.
            trunc_before = llm.truncated_generations
            test_result = evaluator.evaluate(result.best_prompt, test_samples)
            trunc_during = llm.truncated_generations - trunc_before
            result.test_score = test_result.score
            result.test_per_sample_details = test_result.per_sample_details
            result.config["test_truncation_rate"] = (
                trunc_during / len(test_samples) if test_samples else 0.0
            )
            result.config["test_truncated_generations"] = trunc_during
            logger.info(f"[Test eval] test_score={result.test_score:.4f}")
            # The selection score is chosen as the max over finalists on a small
            # slice, so it is upward-biased by construction — it ranks prompts,
            # it does not estimate performance. Measured on GPT-4o/HumanEval it
            # overstated test by 9.8 to 14.1pp on three consecutive runs while
            # being printed as the headline "dev=" figure. Record the gap so the
            # number carries its own health warning.
            if result.best_score is not None and result.test_score is not None:
                gap = result.best_score - result.test_score
                result.config["selection_minus_test"] = round(gap, 4)
                if gap > 0.05:
                    logger.warning(
                        f"[Selection bias] the selection score "
                        f"({source}={result.best_score:.4f}) overstates test "
                        f"({result.test_score:.4f}) by {gap*100:.1f}pp — it ranks "
                        "candidates, it is not an estimate of performance"
                    )
            if trunc_during:
                rate = trunc_during / len(test_samples)
                level = logger.warning if rate > 0.1 else logger.info
                level(
                    f"[Test eval] {trunc_during}/{len(test_samples)} generations "
                    f"({rate:.1%}) hit the token cap and were cut off"
                    + (" — this score reflects the decode budget, not the prompt"
                       if rate > 0.5 else "")
                )
            # State the resolution of the instrument next to its reading, so
            # a null result is never mistaken for an underpowered one.
            logger.info(f"[Power] {describe_power(result.test_score, len(test_samples))}")
            result.config["minimum_detectable_effect"] = minimum_detectable_effect(
                result.test_score, len(test_samples)
            )

        optimizer.tracker.test_score = result.test_score

        # Enrich result config with the full run settings for audit.
        #
        # eval_system_prompt is recorded because it silently defines a separate
        # experimental condition: --strip-system-prompt (override "") and
        # --simple-system-prompt produce different answer formats for the same
        # task, and a run scored under one is not comparable with a run scored
        # under another. Without it in the saved config, downstream analysis
        # cannot tell those runs apart and pools them as if they were replicas.
        # None means "the evaluator's task_type default was used".
        override = self.config.evaluation.system_prompt_override
        result.config.update({
            "model": self.config.llm.model_name,
            "eval_max_new_tokens": self.config.evaluation.max_new_tokens,
            "eval_temperature": self.config.evaluation.temperature,
            "eval_sample_size": self.config.evaluation.sample_size,
            "full_eval_size": self.config.evaluation.full_eval_size,
            "eval_system_prompt": override,
            "eval_task_type": getattr(evaluator, "task_type", "") or "",
            "dataset_num_samples": self.config.dataset.num_samples,
            "dataset_task": self.config.dataset.task or "",
            "run_seed": self.config.seed,
        })

        # A run whose measurement is compromised should say so once, in one
        # place, rather than requiring every downstream reader to separately
        # notice truncation and content-filter rates. Thresholds match
        # experiments/probe_addressability.py's INVALID verdict.
        trunc_rate = result.config.get("test_truncation_rate") or 0.0
        filtered = getattr(llm, "_filtered_generations", 0)
        filtered_rate = filtered / max(1, len(test_samples)) if test_samples else 0.0
        invalid_reasons = []
        if trunc_rate > 0.10:
            invalid_reasons.append(f"truncation_rate={trunc_rate:.1%}")
        if filtered_rate > 0.05:
            invalid_reasons.append(f"content_filtered_rate={filtered_rate:.1%}")
        result.config["condition_invalid"] = bool(invalid_reasons)
        if invalid_reasons:
            logger.warning(
                f"[Condition] run marked invalid for comparison: "
                f"{', '.join(invalid_reasons)} — the score reflects measurement "
                "artifacts, not the prompt"
            )

        # Cost belongs next to the score, not in a footnote. Methods in this
        # project run at very different budgets — a config may allow 2000 calls
        # for one and 5000 for another — so a bare score comparison silently
        # rewards whichever method was allowed to spend more. Recording the
        # normalised figure makes that visible in every result file.
        calls = result.llm_usage.total_calls if result.llm_usage else 0
        result.config["llm_calls"] = calls
        result.config["test_score_per_1k_calls"] = (
            round(result.test_score / (calls / 1000.0), 4)
            if calls and result.test_score is not None else None
        )
        if calls:
            logger.info(
                f"[Cost] {calls} LLM calls for test_score={result.test_score:.4f} "
                f"({result.config['test_score_per_1k_calls']} per 1k calls)"
            )

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
                optimizer_kwargs = {
                    "llm": llm,
                    "dataset": dataset,
                    "evaluator": evaluator,
                    "seed_prompt": self.config.optimizer.seed_prompt,
                    "eval_sample_size": self.config.evaluation.sample_size,
                }
                if self.config.optimizer.population_size != 5:
                    optimizer_kwargs["population_size"] = self.config.optimizer.population_size
                optimizer = optimizer_cls(
                    **optimizer_kwargs,
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
                dev_test_split=self.config.dataset.dev_test_split,
            )
        return self._dataset

    def cleanup(self) -> None:
        """Release GPU memory after a run."""
        if self._llm is not None:
            try:
                self._llm.cleanup()
            except Exception:
                pass
            self._llm = None
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    def _get_evaluator(self, llm: BaseLLM, dataset: TaskDataset) -> Evaluator:
        """Create evaluator instance."""
        # Config override takes precedence over dataset auto-detection.
        task_type = self.config.dataset.task_type or dataset.task_type
        score_fn = create_score_function(task_type)
        return Evaluator(
            llm=llm,
            score_fn=score_fn,
            task_type=task_type,
            max_new_tokens=self.config.evaluation.max_new_tokens,
            temperature=self.config.evaluation.temperature,
            batch_size=self.config.evaluation.batch_size,
            system_prompt=self.config.evaluation.system_prompt_override,
            racing_enabled=self.config.evaluation.racing_enabled,
            racing_confidence=self.config.evaluation.racing_confidence,
            racing_min_samples=self.config.evaluation.racing_min_samples,
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