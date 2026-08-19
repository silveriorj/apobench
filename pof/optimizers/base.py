"""Base optimizer — template method pattern with shared techniques.

All optimizers inherit from BaseOptimizer and implement:
- _init_population(): Generate initial candidate pool
- _step(): One optimization iteration/phase

The base class provides:
- Audit integration (PromptRecord creation, history tracking)
- Evaluation helpers (with racing support)
- Shared generation techniques (Lamarckian, crossover, mutation, etc.)
"""
from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pof.audit.tracker import AuditTracker
from pof.core.types import EvalResult, GenerationConfig, OptimizationResult, PromptRecord, rank_key
from pof.datasets.loader import TaskDataset
from pof.evaluation.evaluator import Evaluator
from pof.llm.base import BaseLLM
from pof.optimizers._failure_view import render_failures
from pof.core.exceptions import BudgetExceeded

logger = logging.getLogger(__name__)

# System prompts injected per operator category so the model knows its role
# and keeps output format tight.

_GENERATE_SYSTEM_PROMPT = (
    "You are an expert prompt engineer. "
    "Output only the instruction text requested — no preamble, no labels, no commentary."
)

_CRITIQUE_SYSTEM_PROMPT = (
    "You are an expert prompt engineer specializing in failure analysis. "
    "Diagnose why the instruction fails on the given examples. Be concise and specific."
)

_IMPROVE_SYSTEM_PROMPT = (
    "You are an expert prompt engineer. "
    "Given a failure analysis, rewrite the instruction to fix the identified issues. "
    "Output only the improved instruction text — no preamble, no labels."
)

# Shared addition to the failure-review meta-prompts (base.py's
# _feedback_improve and swift.py's _structured_improve): separates "the
# model is reasoning wrong" from "the reasoning is fine but the format or
# a surface shortcut is the real problem," so the fix targets the actual
# cause instead of defaulting to more reasoning guidance.
_FAILURE_DIAGNOSIS_CHECKLIST = (
    "When diagnosing, first classify the failure: (a) the reasoning/logic "
    "itself is wrong or missing task-relevant steps, or (b) the reasoning "
    "is likely fine but the output format isn't being followed correctly "
    "(wrong shape, missing required label/prefix, extra text), or (c) the "
    "input contains a surface cue that looks like it states the answer "
    "(e.g. a phrase that sounds like a verdict) and the model may be "
    "keying off that instead of the actual content. Fix whichever is the "
    "true cause -- do not add more reasoning guidance to fix a formatting "
    "problem, and if (c) applies, explicitly instruct the model to ignore "
    "misleading surface phrasing and judge the actual content instead."
)


def format_exemplar(evaluator: Any, sample: Dict[str, str], max_input: Optional[int] = None) -> str:
    """Render one few-shot demonstration in the OUTPUT FORMAT the scorer expects.

    Demonstrations must match the evaluator's required answer shape (JSON for
    BBH, "The answer is X" for math, raw code for HumanEval) — a mismatched
    format measurably hurts accuracy since the model splits the difference
    between what it's shown and what it's told to produce.
    """
    text = str(sample.get("input", ""))
    if max_input:
        text = text[:max_input]
    target = str(sample.get("target", ""))
    task_type = getattr(evaluator, "task_type", "") or ""

    if task_type == "code":
        answer = target
    elif task_type in ("math", "cot", "dyck"):
        answer = f"The answer is {target}" if task_type == "math" else f"So the answer is {target}"
    else:
        answer = '{"answer": "%s"}' % target.replace('"', '\\"')
    return f"Input: {text}\nOutput: {answer}"


class BaseOptimizer(ABC):
    """Abstract base for all prompt optimizers.

    Template method: optimize() calls _init_population() then _step() in a loop.
    Subclasses implement the specific optimization logic.
    """

    # Class-level name for registry
    name: str = "base"

    def __init__(
        self,
        llm: BaseLLM,
        dataset: TaskDataset,
        evaluator: Evaluator,
        population_size: int = 5,
        num_iterations: int = 3,
        seed_prompt: str = "",
        eval_sample_size: int = 50,
        output_dir: str = "outputs",
        **kwargs: Any,
    ):
        self.llm = llm
        self.dataset = dataset
        self.evaluator = evaluator
        self.population_size = population_size
        self.num_iterations = num_iterations
        self.seed_prompt = seed_prompt
        self.eval_sample_size = eval_sample_size
        self.kwargs = kwargs
        # Guards _finalize() against a double call: a subclass's _step() may
        # call it directly before raising StopIteration, and optimize()'s
        # finally block also calls it unconditionally — see _finalize().
        self._finalized: bool = False

        # Audit tracker
        self.tracker = AuditTracker(
            method_name=self.name,
            dataset_name=dataset.name,
            config=self._get_config_dict(),
            output_dir=output_dir,
        )

        # Population state
        self.population: List[PromptRecord] = []
        self.generation: int = 0
        self.best_record: Optional[PromptRecord] = None

    def optimize(self) -> OptimizationResult:
        """Run the full optimization loop.

        Returns:
            OptimizationResult with best prompt, scores, and audit trail.
        """
        self.tracker.start()
        logger.info(f"Starting {self.name} optimization on {self.dataset.name}")

        try:
            # Phase 0: Initialize population
            self.population = self._init_population()
            self._evaluate_population(self.population)
            self._update_best()
            self.tracker.record_generation(self.generation, self.population)
            logger.info(
                f"[Gen {self.generation}] Population initialized: "
                f"{len(self.population)} candidates, best={self.best_record.score:.4f}"
            )

            # Optimization loop (budget-aware)
            budget_mgr = self._get_budget_mgr()
            effective_iters = self.num_iterations
            if budget_mgr and getattr(budget_mgr, "config", None) and getattr(budget_mgr.config, "max_generations", None):
                try:
                    effective_iters = min(self.num_iterations, int(budget_mgr.config.max_generations))
                except Exception:
                    pass
            patience = 0
            if budget_mgr and getattr(budget_mgr, "config", None):
                try:
                    patience = int(getattr(budget_mgr.config, "early_stop_patience", 0) or 0)
                except Exception:
                    patience = 0
            last_best = self.best_record.score if self.best_record else 0.0
            no_improve = 0

            for i in range(effective_iters):
                # should_stop_for_search(), not should_stop(): reserves a
                # slice of the time budget for _finalize() (e.g. holdout
                # re-ranking) so it never runs out of time to correct
                # selection at the end.
                stop_reason = budget_mgr.should_stop_for_search() if budget_mgr else None
                if stop_reason is not None:
                    logger.info(
                        f"Budget exhausted before generation {self.generation+1} "
                        f"({stop_reason}), stopping optimization"
                    )
                    break

                self.generation += 1
                try:
                    new_population = self._step()
                except StopIteration:
                    logger.info(f"Optimizer signaled early stop at generation {self.generation}")
                    break
                except BudgetExceeded as be:
                    logger.info(f"Budget exhausted during generation {self.generation}: {be.kind}")
                    break

                if new_population:
                    self.population = new_population
                self._update_best()
                self.tracker.record_generation(self.generation, self.population)
                logger.info(
                    f"[Gen {self.generation}] best={self.best_record.score:.4f}, "
                    f"pop_size={len(self.population)}"
                )

                # Early stopping based on patience (no improvement)
                if patience > 0:
                    if self.best_record and self.best_record.score > last_best + 1e-12:
                        last_best = self.best_record.score
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= patience:
                            logger.info(f"Early stopping: no improvement for {patience} consecutive generations")
                            break

        except Exception as e:
            self.tracker.add_note(f"ERROR: {e}")
            logger.error(f"Optimization failed: {e}", exc_info=True)
            raise
        finally:
            # Must run regardless of which exit path was taken (natural
            # exhaustion, patience, or budget), since held-out selection
            # correction depends on it. No-op by default and guarded by
            # self._finalized, so this is safe even if a subclass's _step()
            # already called it.
            if not self._finalized:
                try:
                    self._finalize()
                except Exception as e:
                    logger.error(f"_finalize() failed: {e}", exc_info=True)
            self.tracker.end()
            self.tracker.usage = self.llm.get_usage()

        result = self.tracker.to_optimization_result()
        logger.info(
            f"Optimization complete: best_score={result.best_score:.4f}, "
            f"time={result.total_time:.1f}s, "
            f"llm_calls={self.llm.usage.total_calls}"
        )
        return result

    @abstractmethod
    def _init_population(self) -> List[PromptRecord]:
        """Generate initial population of candidates.

        Returns:
            List of PromptRecord candidates (unevaluated).
        """
        ...

    @abstractmethod
    def _step(self) -> List[PromptRecord]:
        """Execute one optimization step/phase.

        Returns:
            Updated population. Raise StopIteration to end early.
        """
        ...

    def _finalize(self) -> None:
        """Called exactly once, right before the result is built, regardless
        of why the optimization loop stopped. No-op here; subclasses override
        it to run held-out selection. If calling this from more than one
        place, guard with `if self._finalized: return` and set
        `self._finalized = True` (optimize()'s own call already checks the
        flag before calling).
        """
        pass

    def _maybe_stop_if_perfect(self, threshold: float = 1.0) -> None:
        """Raise StopIteration if `best_record.score` already reached
        `threshold`. Not called automatically -- a subclass opts in by
        calling this at the top of its own `_step()`.

        Avoids grinding through remaining phases with zero possible
        improvement once a perfect dev score is reached. Assumes `.score` is
        a plain accuracy fraction; optimizers whose score can be
        barrier-penalized below the true EM (e.g. FUNNEL) should check their
        own `scores["dev"]` instead.
        """
        if self.best_record and self.best_record.score >= threshold:
            note = (
                f"perfect dev score ({threshold:.3f}) already reached -- "
                f"stopping early rather than spending more budget confirming it"
            )
            logger.info(f"[{self.name}] {note}")
            self.tracker.add_note(note)
            raise StopIteration

    # --- Shared helpers ---

    def _sample_dev(self, n: Optional[int], seed: int = 42) -> List[Dict[str, str]]:
        """Sample dev-set instances for candidate evaluation.

        Draws from `self.dataset`'s full dev split by default. Subclasses
        reserving a held-out selection slice (e.g. `HoldoutSelectionMixin`)
        override this to sample only the search-visible portion — the
        held-out slice must stay unseen by every operator/gate/population
        eval or the final holdout re-ranking loses its purpose.
        """
        return self.dataset.get_eval_samples("dev", n=n, seed=seed)

    def _get_budget_mgr(self) -> Optional[Any]:
        """Fetch the attached BudgetManager, if any."""
        get_budget = getattr(self.llm, "get_budget", None)
        return get_budget() if callable(get_budget) else None

    def _timed_evaluate(self, prompt: str, samples: List[Dict[str, str]]) -> EvalResult:
        """self.evaluator.evaluate(), timed, feeding the observed duration to
        the budget manager's rolling average so its time-remaining estimates
        adapt to how slow evaluation actually is on this task."""
        budget_mgr = self._get_budget_mgr()
        start = time.time()
        result = self.evaluator.evaluate(prompt, samples)
        if budget_mgr is not None:
            budget_mgr.record_eval_duration(time.time() - start)
        return result

    def _evaluate_population(self, population: List[PromptRecord]) -> None:
        """Evaluate all candidates in the population."""
        samples = self._sample_dev(self.eval_sample_size)
        to_eval = [r for r in population if r.score == 0.0 and r.text]
        logger.debug(f"[Pop eval] {len(to_eval)} candidates to evaluate")
        for idx, record in enumerate(to_eval, start=1):
            logger.debug(
                f"[Pop eval] candidate {idx}/{len(to_eval)}"
                f" op={record.operator} prompt={record.text[:60]!r}..."
            )
            result = self._timed_evaluate(record.text, samples)
            record.score = result.score
            record.performance_vector = result.performance_vector
            record.per_sample_details = result.per_sample_details
            record.scores["dev"] = result.score
            logger.debug(f"[Pop eval] candidate {idx}/{len(to_eval)} → score={result.score:.3f}")

    def _evaluate_with_racing(
        self, candidates: List[PromptRecord], baseline_score: float
    ) -> List[PromptRecord]:
        """Evaluate candidates with racing against baseline."""
        samples = self._sample_dev(self.eval_sample_size)
        budget_mgr = self._get_budget_mgr()
        for record in candidates:
            if record.text:
                start = time.time()
                result = self.evaluator.evaluate_with_racing(
                    record.text, samples, baseline_score
                )
                if budget_mgr is not None:
                    budget_mgr.record_eval_duration(time.time() - start)
                record.score = result.score
                record.performance_vector = result.performance_vector
                record.per_sample_details = result.per_sample_details
                record.scores["dev"] = result.score
        return candidates

    def _evaluate_with_minibatch_gate(
        self,
        candidates: List[PromptRecord],
        baseline_score: float,
        minibatch_size: int = 16,
        slack: float = 0.10,
    ) -> List[PromptRecord]:
        """GEPA-style two-stage evaluation (Agrawal et al. 2025).

        Stage 1: each candidate is scored on a FRESH random dev minibatch
        (resampled every call, so selection never overfits one fixed subset).
        Stage 2: only candidates within `slack` of the baseline on the
        minibatch get the full dev evaluation used for selection.

        Rejected candidates keep their minibatch score (they rank low and
        drop out at selection) and are marked in metadata for the audit.

        Checks has_time_for_another_eval() before EACH Stage-2 eval (not
        just once per generation) so a generation with many gate-passed
        candidates can bail mid-list instead of exhausting the whole
        finalize time reserve on its own.

        Stage-2 eval uses evaluate_with_batch_racing (Hoeffding early-stop
        between batches) so a candidate clearly below the racing threshold
        is eliminated before paying for all eval_sample_size samples. The
        racing threshold is the current best record's score minus `slack`
        (falling back to `baseline_score` before any best_record exists) —
        a continuously tightening bar that still leaves near-best
        candidates a fair shot rather than eliminating anything short of a
        new best, which would prematurely narrow population diversity.
        """
        minibatch = self._sample_dev(minibatch_size, seed=random.randint(0, 10**6))
        full_samples = self._sample_dev(self.eval_sample_size)
        budget_mgr = self._get_budget_mgr()
        racing_threshold = (
            self.best_record.score - slack if self.best_record else baseline_score
        )

        n_pass = 0
        n_budget_skipped = 0
        for record in candidates:
            if not record.text:
                continue
            mb = self.evaluator.evaluate(record.text, minibatch)
            if mb.score + slack >= baseline_score:
                if budget_mgr is not None and not budget_mgr.has_time_for_another_eval():
                    record.score = mb.score
                    record.scores["minibatch"] = mb.score
                    record.metadata["gate"] = "budget_skipped"
                    n_budget_skipped += 1
                    logger.info(
                        f"[Gate] op={record.operator}: minibatch={mb.score:.3f} "
                        f"passed slack but SKIPPED full eval (finalize reserve low)"
                    )
                    continue
                start = time.time()
                result = self.evaluator.evaluate_with_batch_racing(
                    record.text, full_samples, threshold=racing_threshold
                )
                if budget_mgr is not None:
                    budget_mgr.record_eval_duration(time.time() - start)
                record.score = result.score
                record.performance_vector = result.performance_vector
                record.per_sample_details = result.per_sample_details
                record.scores["dev"] = result.score
                record.scores["minibatch"] = mb.score
                n_pass += 1
                raced_note = " [raced]" if result.metadata.get("racing_terminated") else ""
                logger.info(
                    f"[Gate] op={record.operator}: minibatch={mb.score:.3f} "
                    f"→ dev={result.score:.3f} (n={result.num_total}/{self.eval_sample_size}"
                    f"{raced_note}, selection uses dev)"
                )
            else:
                record.score = mb.score
                record.scores["minibatch"] = mb.score
                record.metadata["gate"] = "rejected"
                logger.info(
                    f"[Gate] op={record.operator}: minibatch={mb.score:.3f} "
                    f"< baseline−slack — rejected, no full eval"
                )
        logger.info(
            f"[Minibatch gate] {n_pass}/{len(candidates)} candidates passed "
            f"(baseline={baseline_score:.3f}, slack={slack})"
            + (f", {n_budget_skipped} skipped (reserve low)" if n_budget_skipped else "")
        )
        return candidates

    def _select_top_k(
        self, candidates: List[PromptRecord], k: Optional[int] = None
    ) -> List[PromptRecord]:
        """Select top-k candidates by score.

        Uses rank_key (pof/core/types.py), not raw `.score` -- gate-rejected
        candidates carry a noisy minibatch score on `.score` while
        gate-passed candidates carry a full-dev score, so ranking on
        `.score` alone could let a minibatch fluke outrank a real best.
        """
        k = k or self.population_size
        sorted_candidates = sorted(candidates, key=rank_key, reverse=True)
        return sorted_candidates[:k]

    def _update_best(self) -> None:
        """Update best_record from current population."""
        if self.population:
            current_best = max(self.population, key=rank_key)
            if self.best_record is None or rank_key(current_best) > rank_key(self.best_record):
                self.best_record = current_best

    def _is_duplicate(self, text: str) -> bool:
        """True if an identical prompt was already generated this run.

        Duplicate candidates waste a full (racing) evaluation each — operators
        like paraphrase and crossover regenerate near-identical text often.
        """
        if not text:
            return True
        text_hash = PromptRecord._compute_hash(text.strip())
        return self.tracker.history.get_by_hash(text_hash) is not None

    def _create_record(
        self,
        text: str,
        operator: str,
        parent_ids: Optional[List[str]] = None,
        **metadata: Any,
    ) -> PromptRecord:
        """Create a new PromptRecord with lineage info."""
        record = PromptRecord(
            text=text,
            operator=operator,
            parent_ids=parent_ids or [],
            generation_created=self.generation,
            metadata=metadata,
        )
        self.tracker.add_record(record)
        return record

    def _generate_prompt(
        self,
        instruction: str,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate text using the LLM."""
        config = GenerationConfig(
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
        )
        return self.llm.generate(instruction, config, system_prompt=system_prompt)

    def _get_config_dict(self) -> Dict[str, Any]:
        """Get optimizer configuration as dict."""
        return {
            "method": self.name,
            "population_size": self.population_size,
            "num_iterations": self.num_iterations,
            "eval_sample_size": self.eval_sample_size,
            "seed_prompt": self.seed_prompt,
            **{k: v for k, v in self.kwargs.items() if isinstance(v, (str, int, float, bool))},
        }

    # --- Shared generation techniques ---

    def _lamarckian_generate(
        self, samples: List[Dict[str, str]], n: int = 1, _label: str = "lamarckian"
    ) -> List[str]:
        """Lamarckian operator: reverse-engineer instruction from I/O pairs.

        Given input-output examples, ask the LLM to infer what instruction
        would produce those outputs.
        """
        few_shot = samples[:5]
        examples_text = "\n".join(
            format_exemplar(self.evaluator, s) for s in few_shot
        )

        meta_prompt = (
            "Given the following input-output examples, reverse-engineer the instruction "
            "that would produce these outputs from the inputs. Write ONLY the instruction, "
            "nothing else.\n\n"
            f"Examples:\n{examples_text}\n\n"
            "Instruction:"
        )

        results = []
        for _ in range(n):
            result = self._generate_prompt(meta_prompt, temperature=0.8, system_prompt=_GENERATE_SYSTEM_PROMPT)
            if result.strip():
                results.append(result.strip())
        return results

    def _semantic_variation(self, prompt: str, n: int = 1) -> List[str]:
        """Generate semantic variations of a prompt (paraphrasing)."""
        meta_prompt = (
            "Rewrite the following instruction in a different way while preserving "
            "its exact meaning and intent. Use different words and structure.\n\n"
            f"Original instruction:\n{prompt}\n\n"
            "Rewritten instruction:"
        )

        results = []
        for _ in range(n):
            result = self._generate_prompt(meta_prompt, temperature=0.9, system_prompt=_GENERATE_SYSTEM_PROMPT)
            if result.strip():
                results.append(result.strip())
        return results

    def _crossover(self, prompt_a: str, prompt_b: str) -> str:
        """LLM-based crossover of two prompts."""
        meta_prompt = (
            "Combine the best elements of these two instructions into a single, "
            "improved instruction that captures the strengths of both.\n\n"
            f"Instruction A:\n{prompt_a}\n\n"
            f"Instruction B:\n{prompt_b}\n\n"
            "Combined instruction:"
        )
        return self._generate_prompt(meta_prompt, temperature=0.7, system_prompt=_GENERATE_SYSTEM_PROMPT)

    def _feedback_improve(
        self, prompt: str, failures: List[Dict[str, Any]]
    ) -> str:
        """Improve a prompt based on failure analysis."""
        failure_text = render_failures(failures, max_failures=5)

        meta_prompt = (
            "The following instruction was used but produced incorrect outputs "
            "for some inputs. Analyze the failures and rewrite the instruction "
            "to fix these issues.\n\n"
            f"{_FAILURE_DIAGNOSIS_CHECKLIST}\n\n"
            f"Current instruction:\n{prompt}\n\n"
            f"Failures:\n{failure_text}\n\n"
            "Improved instruction:"
        )
        return self._generate_prompt(meta_prompt, temperature=0.7, system_prompt=_IMPROVE_SYSTEM_PROMPT)

    def _eda_generate(self, prompts: List[str]) -> str:
        """EDA operator: generate from distribution of existing prompts."""
        prompts_text = "\n---\n".join(prompts[:5])
        meta_prompt = (
            "Here are several instructions that all aim to accomplish the same task. "
            "Generate a NEW instruction that captures the common patterns and best "
            "elements from all of them, but is distinct from each.\n\n"
            f"Existing instructions:\n{prompts_text}\n\n"
            "New instruction:"
        )
        return self._generate_prompt(meta_prompt, temperature=0.8, system_prompt=_GENERATE_SYSTEM_PROMPT)