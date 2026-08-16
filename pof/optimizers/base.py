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
# _feedback_improve, used by apex's failure_guided operator; swift.py's
# _structured_improve, used by SWIFT's own failure-guided phase and
# swift_v2's second pass). Added 2026-08-15 after inspecting real winning
# prompts from a BBH strip run: the search had independently discovered
# this exact distinction on its own (formal_fallacies' winner explicitly
# tells the model not to key off the phrase "perfectly valid argument" and
# to judge actual logical structure instead) -- this instruction makes
# that a standing part of the diagnostic step instead of leaving it to
# chance, so failure review reliably separates "the model is reasoning
# wrong" from "the model is reasoning fine but the output shape or a
# surface shortcut is the real problem," and fixes the one that's
# actually broken rather than always reaching for more reasoning guidance.
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

    Every optimizer previously wrote demonstrations as `Input: X / Output: (B)`
    regardless of task, while the evaluator's system prompt demands a specific
    shape — JSON for BBH, "The answer is X" for math, raw code for HumanEval.
    A prompt therefore showed the model one answer format while instructing it
    to produce another, and the model splits the difference: measured on
    bbh_boolean_expressions, literal `A: False` exemplars scored 0.7652 against
    0.8435 for the identical exemplars written as `{"answer": "False"}` — a
    7.8-point loss caused purely by the mismatch.

    Aligning the demonstration with the required output removes that. The gain
    is task-dependent (it helped 4 of 8 BBH tasks, mean +0.010), so this is a
    correctness fix rather than a uniform improvement.
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
        # Guards _finalize() against a double call: FUNNELv2's own _step()
        # calls it before raising StopIteration on natural phase exhaustion,
        # but optimize()'s other exit paths (patience, budget) previously
        # skipped it entirely -- see _finalize()'s docstring.
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
                # Global budget stop check. Uses should_stop_for_search(),
                # not should_stop(), so a fixed slice of the time budget is
                # always left over for _finalize() (e.g. holdout
                # re-ranking) below -- otherwise a search that runs to the
                # true budget edge starves _finalize() of any time to run,
                # silently falling back to uncorrected selection.
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
            # Bug fix (2026-07-29): every exit path used to reach this point
            # EXCEPT the StopIteration one (natural phase exhaustion), which
            # is the only one that previously ran a subclass's _finalize().
            # Patience-based and budget-exhaustion stops skipped it entirely,
            # meaning FUNNELv2+'s held-out selection correction -- built
            # specifically to counter a measured winner's-curse bias between
            # dev and test scores -- silently never ran for any run that
            # stopped that way. _finalize() is a no-op by default and guarded
            # by self._finalized, so calling it here is always safe whether
            # or not a subclass's _step() already called it.
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
        of why the optimization loop stopped (phase exhaustion, patience,
        budget exhaustion). No-op here; FUNNELv2+ overrides it to run held-out
        selection. Guard with `if self._finalized: return` / set
        `self._finalized = True` if overriding and calling this from more
        than one place (see FUNNELv2Optimizer._finalize for the pattern) --
        `optimize()`'s own call already checks the flag before calling.
        """
        pass

    def _maybe_stop_if_perfect(self, threshold: float = 1.0) -> None:
        """Raise StopIteration if `best_record.score` already reached
        `threshold`. Not called automatically -- a subclass opts in by
        calling this at the top of its own `_step()`.

        Motivated by SWIFT reaching a perfect dev score in `_init_population`
        (Gen 0, before any optimization operator even ran) and then still
        grinding through its full fixed Phase 1/2/3 schedule for 2 more hours
        with zero possible improvement, until the external time budget cut it
        off mid-Phase-3 -- the same waste FUNNELv2+'s `_maybe_stop_if_perfect`
        (funnel_v4d.py) was built to avoid, generalized here for any
        optimizer whose `.score` is a plain accuracy fraction with no
        barrier-penalty or other non-monotonic scoring quirk (FUNNEL
        subclasses check their own `scores["dev"]` instead, since `.score`
        there can be barrier-penalized below the true EM).
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
        that reserve a held-out selection slice (e.g. an optimizer using
        `HoldoutSelectionMixin`) override this to sample only from the
        search-visible portion, keeping the held-out slice genuinely unseen
        by every operator/gate/population eval -- otherwise the final
        `_select_on_holdout` re-ranking would be picking among finalists on
        data the search had already touched, defeating the point.
        """
        return self.dataset.get_eval_samples("dev", n=n, seed=seed)

    def _get_budget_mgr(self) -> Optional[Any]:
        """Fetch the attached BudgetManager, if any (mirrors optimize()'s
        own lookup so eval helpers can record durations / check the
        mid-generation reserve without duplicating the getattr dance)."""
        get_budget = getattr(self.llm, "get_budget", None)
        return get_budget() if callable(get_budget) else None

    def _timed_evaluate(self, prompt: str, samples: List[Dict[str, str]]) -> EvalResult:
        """self.evaluator.evaluate(), timed, feeding the observed duration
        to the budget manager's rolling average (see BudgetManager.
        record_eval_duration) so remaining_search_time()'s finalize reserve
        can adapt to how slow evaluation actually is on this task, and
        has_time_for_another_eval() has real data to check against."""
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

        Bug fix (2026-08-15, found via the BBH v2-strip validation run):
        the per-generation budget check (should_stop_for_search(), in
        optimize()'s loop) only runs once BEFORE a generation starts -- a
        single generation with several gate-passed candidates, each
        needing a full-dev eval, could still consume the ENTIRE finalize
        reserve on its own before the next check ever ran (observed: 3 of
        5 BBH tasks in one run hit BudgetExceeded during _finalize()
        despite the 2026-08-13 reserve fix). Now checks
        has_time_for_another_eval() before EACH full-dev eval in this
        loop -- once time is short, remaining gate-passed candidates are
        skipped (kept at their minibatch score, flagged in metadata) rather
        than evaluated regardless of cost, so a generation can bail
        mid-list instead of only being caught at its own end.

        Cost/speed improvement (2026-08-15, SOTA-APO research pass): the
        Stage-2 full-dev eval now runs via evaluate_with_batch_racing
        (Hoeffding early-stop BETWEEN batches, not the older per-sample
        evaluate_with_racing which loses _batch_generate's batching
        throughput -- see that method's own docstring for why the
        between-batch variant exists). A gate-passed candidate that turns
        out to be clearly worse than `baseline_score` once enough of the
        full-dev set has been seen is eliminated early instead of always
        paying for all eval_sample_size samples -- this is the main lever
        for cutting the multi-hour BBH run times seen in this session's
        validation sweeps, without changing what "passing the gate" means.
        """
        minibatch = self._sample_dev(minibatch_size, seed=random.randint(0, 10**6))
        full_samples = self._sample_dev(self.eval_sample_size)
        budget_mgr = self._get_budget_mgr()

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
                    record.text, full_samples, threshold=baseline_score
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
        candidates carry a noisy 16-sample minibatch score on `.score` while
        gate-passed candidates carry a full-dev score; ranking on `.score`
        alone let a minibatch fluke outrank a genuine best candidate.
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
        # Bug found via livebench_coding: unlike input/prediction, 'target'
        # was embedded unbounded -- fine for short answer strings (BBH/math)
        # but for task_type="code", target is a JSON blob carrying test
        # cases, and one dataset (LiveCodeBench's large private_test_cases)
        # blew a meta-prompt out to 22.5M tokens. Truncate like the other
        # two fields; a clipped JSON blob is no less informative to the LLM
        # than the full one (neither is human-readable as "the expected
        # answer" for code tasks), so this loses no real signal.
        failure_text = "\n".join(
            f"- Input: {f.get('input', '')[:80]}\n"
            f"  Expected: {str(f.get('target', ''))[:80]}\n"
            f"  Got: {f.get('prediction', '')[:80]}"
            for f in failures[:5]
        )

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