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

# Used by SEEC's synthesis phase and GEPA-Micro's final rewrite step: both
# feed the LLM a scored history of candidates (not just the current one) and
# need it to weigh generalization over dev-set memorization when synthesizing
# a new instruction from that history.
_SYNTHESIS_SYSTEM_PROMPT = (
    "You are an expert prompt engineer performing evidence-based synthesis "
    "across generations. Prefer prompts that generalize over ones that "
    "memorize. Be suspicious of long, specific clauses that only marginally "
    "improve score — when two candidates score similarly, always prefer the "
    "shorter one. Output only the new instruction text — no preamble, no labels."
)

# Used by GEPA-Micro to consolidate several independent persona critiques
# (logic, format, generalization) of the same prompt into one prioritized
# diagnosis before synthesis — a "second opinion" step that catches
# contradictions a single critique call would miss.
_REVIEWER_CONSOLIDATION_SYSTEM_PROMPT = (
    "You are a senior prompt engineering reviewer. You receive multiple "
    "independent critiques of the same prompt from different perspectives. "
    "Consolidate them into a single, prioritized diagnosis — resolve any "
    "contradictions, drop redundant points, and rank issues by impact on "
    "generalization. Output only the consolidated diagnosis — no preamble."
)

# Shared addition to failure-review meta-prompts: separates "the model is
# reasoning wrong" from "the reasoning is fine but the format or a surface
# shortcut is the real problem," so the fix targets the actual cause instead
# of defaulting to more reasoning guidance.
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


def _screen_coverage(scored: List[Any]) -> Dict[str, int]:
    """How many screen instances each candidate solves that not everyone solves.

    An instance every candidate gets right separates nobody, so it contributes
    no coverage. What counts is an instance a candidate solves while at least
    one rival misses it — that is the evidence a tie on aggregate score hides.
    """
    if len(scored) < 2:
        return {}
    vectors = [(rec.id, vec) for _, vec, rec in scored if vec]
    if len(vectors) < 2:
        return {}
    length = min(len(v) for _, v in vectors)
    counts: Dict[str, int] = {}
    for i in range(length):
        solved = [rid for rid, v in vectors if v[i] > 0]
        if not solved or len(solved) == len(vectors):
            continue  # nobody solved it, or everybody did — no signal either way
        for rid in solved:
            counts[rid] = counts.get(rid, 0) + 1
    return counts


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

    # What kind of evidence this method's numbers are. Not decoration — a
    # researcher comparing methods needs to know which comparisons are citable
    # before running anything, not after finding out the hard way.
    #
    #   "contribution" — the method under study.
    #   "baseline"     — a reimplementation of a published method, cited in the
    #                    class docstring. Legitimate as a comparison point.
    #   "in_house"     — designed within this project. Comparing against it is
    #                    comparing against ourselves, not the literature —
    #                    useful for ablation, not for "beats published work".
    tier: str = "in_house"

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
            self._discriminability = self._report_discriminability()

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
            # Generations actually completed, as distinct from num_iterations,
            # which is what was asked for. Under a matched call budget the two
            # diverge — a method whose gate costs more finishes fewer rounds —
            # and comparing methods without that number compares different
            # amounts of search.
            self._generations_completed = 0
            self._search_stop_reason = "completed_all_generations"

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
                    self._search_stop_reason = f"budget:{stop_reason}"
                    break

                self.generation += 1
                try:
                    new_population = self._step()
                except StopIteration:
                    logger.info(f"Optimizer signaled early stop at generation {self.generation}")
                    self._search_stop_reason = "optimizer_stop_iteration"
                    break
                except BudgetExceeded as be:
                    logger.info(f"Budget exhausted during generation {self.generation}: {be.kind}")
                    self._search_stop_reason = f"budget_mid_generation:{be.kind}"
                    break

                if new_population:
                    self.population = new_population
                self._update_best()
                self._generations_completed = self.generation
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
                            self._search_stop_reason = "patience_exhausted"
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
            # The tracker's config snapshot was taken in __init__, before any
            # search ran. Refresh the two fields that are only knowable now.
            try:
                self.tracker.config["generations_completed"] = getattr(
                    self, "_generations_completed", 0)
                self.tracker.config["search_stop_reason"] = getattr(
                    self, "_search_stop_reason", "did_not_reach_search_loop")
            except Exception:
                pass

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
        minibatch_size: int = 16,  # retained for API compat; no longer used
        slack: float = 0.10,
    ) -> List[PromptRecord]:
        """Adaptive streaming gate — evaluate on the full dev pool with
        Hoeffding-bounded early termination.

        Replaces the GEPA-style two-stage approach (fixed-size pre-filter then
        full eval) with a single streaming pass via evaluate_with_batch_racing:

          • Every candidate starts evaluating on the full dev pool immediately.
          • A candidate is stopped early only when the Hoeffding lower bound
            on its running score falls below threshold — requiring sustained
            underperformance across many samples, not a single unlucky
            mini-batch result.
          • Candidates that stay at or above threshold all the way through
            complete the full eval. "False-perfect" from a lucky short gate
            (the root cause of EXP-033b's −2.4pp mean) is structurally
            impossible: a mediocre prompt cannot maintain a high running score
            once enough samples are seen.

        When evaluator.racing_enabled is False, evaluate_with_batch_racing
        falls through to a plain full eval — no early termination at all.

        threshold = elite_score − slack (or baseline_score − slack before any
        best_record exists). The slack keeps near-best candidates viable and
        prevents the bar from collapsing to a single dominant prompt.
        """
        full_samples = self._sample_dev(self.eval_sample_size)
        budget_mgr = self._get_budget_mgr()
        racing_threshold = (
            self.best_record.score - slack if self.best_record else baseline_score - slack
        )

        n_passed = 0
        n_rejected = 0
        n_budget_skipped = 0
        for record in candidates:
            if not record.text:
                continue
            if budget_mgr is not None and not budget_mgr.has_time_for_another_eval():
                record.metadata["gate"] = "budget_skipped"
                n_budget_skipped += 1
                logger.info(
                    f"[Gate] op={record.operator}: SKIPPED (finalize reserve low)"
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
            if result.metadata.get("racing_terminated"):
                record.metadata["gate"] = "rejected"
                n_rejected += 1
                logger.info(
                    f"[Gate] op={record.operator}: score={result.score:.3f} "
                    f"(n={result.num_total}/{self.eval_sample_size}, early stop)"
                )
            else:
                n_passed += 1
                logger.info(
                    f"[Gate] op={record.operator}: score={result.score:.3f} "
                    f"(n={result.num_total}/{self.eval_sample_size}, full eval)"
                )
        logger.info(
            f"[Adaptive gate] {n_passed} full / {n_rejected} early-stop"
            f" / {n_budget_skipped} skipped (threshold={racing_threshold:.3f})"
        )
        return candidates

    def _screen_and_evaluate(
        self,
        candidates: List[PromptRecord],
        keep_n: int,
        screen_size: int = 16,
        seed: int = 7777,
        escalations: int = 2,
    ) -> List[PromptRecord]:
        """Two-stage init evaluation: cheap screen, full dev only on survivors.

        Initialization is the single largest cost block in the phase-based
        optimizers -- every generated candidate gets a full dev evaluation even
        though most are discarded moments later by `_select_top_k`. Screening on
        a small shared sample first and spending the full evaluation only on the
        survivors buys the same starting pool for materially fewer calls, which
        is what makes a wider init affordable in the first place.

        Screened-out candidates keep `score == 0.0` and never get a "dev" entry
        in `scores`, so they stay out of holdout finalist ranking (which filters
        on that key) exactly like any other unevaluated record.

        Returns the survivors, already full-dev evaluated.
        """
        usable = [r for r in candidates if r.text]
        if len(usable) <= keep_n:
            self._evaluate_population(usable)
            return usable

        pool = list(self._sample_dev(None))
        random.Random(seed).shuffle(pool)
        cursor = 0

        batch = pool[cursor:cursor + screen_size]
        cursor += len(batch)
        scored = []
        for record in usable:
            result = self._timed_evaluate(record.text, batch)
            scored.append((result.score, result.performance_vector, record))

        # A small screen saturates against a strong model: measured on
        # GPT-4o/HumanEval, five of six candidates tied at 1.000 for three slots,
        # and the pattern repeated across seeds, so it is structural rather than
        # luck. Ranking on score then cuts arbitrarily among equals.
        #
        # Enlarging the screen for everyone would buy resolution on candidates
        # that were never near the cut. Escalate instead: give fresh instances
        # only to the group straddling the cutoff, until it separates or the
        # rounds run out. Candidates clearly above or below keep their place.
        for _ in range(max(0, escalations)):
            scored.sort(key=lambda t: t[0], reverse=True)
            boundary = scored[keep_n - 1][0]
            tied = [t for t in scored if t[0] == boundary]
            n_above_cut = sum(1 for t in scored[:keep_n] if t[0] == boundary)
            if len(tied) < 2 or n_above_cut == len(tied):
                break  # separated, or the tie sits wholly inside the kept set
            extra = pool[cursor:cursor + screen_size]
            cursor += len(extra)
            if not extra:
                break
            logger.info(
                f"[Init screen] {len(tied)} candidates tied at {boundary:.3f} "
                f"across the cut — escalating those on {len(extra)} fresh instances"
            )
            n_seen = len(batch) + len(extra)
            refreshed = []
            for s, vec, rec in tied:
                res = self._timed_evaluate(rec.text, extra)
                merged = list(vec or []) + list(res.performance_vector or [])
                pooled = (s * len(batch) + res.score * len(extra)) / n_seen
                refreshed.append((pooled, merged, rec))
            tied_ids = {r.id for _, _, r in tied}
            scored = [t for t in scored if t[2].id not in tied_ids] + refreshed

        # Whatever escalation could not separate falls back to coverage — does
        # this candidate solve anything the others miss — and only then to
        # length. Length alone was tried first and is biased at init: Lamarckian
        # candidates are systematically terser than persona ones, so preferring
        # the shorter prompt filters by generator rather than by merit.
        coverage = _screen_coverage(scored)
        scored.sort(
            key=lambda t: (t[0], coverage.get(t[2].id, 0), -len(t[2].text)),
            reverse=True,
        )
        survivors = [r for _, _, r in scored[:keep_n]]
        logger.info(
            f"[Init screen] {len(usable)} candidates on {cursor} instances -> "
            f"top-{len(survivors)} to full dev eval "
            f"(scores: {[round(s, 3) for s, _, _ in scored]}, "
            f"coverage: {[coverage.get(r.id, 0) for _, _, r in scored]})"
        )
        self._evaluate_population(survivors)
        return survivors

    def _report_discriminability(self) -> Dict[str, float]:
        """Warn when this model/task pair cannot separate candidates.

        Measured on GPT-4o/HumanEval: the baseline already scored 82.5%, the init
        screen tied 6 of 6 candidates, and parent plus four children all scored
        1.000 on 32 gate instances round after round. Twelve runs were spent
        establishing that the comparison was measuring noise. Every symptom was
        visible right after initialization, from numbers already computed.

        Costs nothing — reads the init pool's scores and vectors. Advisory only:
        a saturated pair is still worth running deliberately, just not worth
        mistaking for a method comparison.
        """
        pool = [r for r in self.population if r.score > 0]
        stats = {"best": 0.0, "tied_fraction": 0.0, "headroom": 1.0}
        if not pool:
            return stats

        best = max(r.score for r in pool)
        tied = sum(1 for r in pool if r.score == best)
        stats["best"] = best
        stats["tied_fraction"] = tied / len(pool)
        stats["headroom"] = 1.0 - best

        warnings = []
        if best >= 0.95:
            warnings.append(
                f"init best is {best:.1%} — almost no headroom for search to work in"
            )
        if len(pool) > 1 and tied / len(pool) >= 0.7:
            warnings.append(
                f"{tied}/{len(pool)} init candidates tie at {best:.1%} — "
                "the dev pool barely separates them"
            )
        vec = next((r.performance_vector for r in pool if r.performance_vector), None)
        if vec:
            misses = sum(1 for v in vec if v <= 0)
            if misses <= 1:
                warnings.append(
                    f"the best candidate misses {misses} of {len(vec)} dev "
                    "instances — a gate has almost nothing to target"
                )
        if warnings:
            logger.warning(
                "[Discriminability] this model/task pair looks saturated; "
                "differences between methods here will be hard to distinguish "
                "from run-to-run noise:"
            )
            for w in warnings:
                logger.warning(f"[Discriminability]   - {w}")
        else:
            logger.info(
                f"[Discriminability] init best={best:.1%}, "
                f"{tied}/{len(pool)} tied — usable headroom"
            )
        return stats

    def _sample_details(self, record: PromptRecord) -> List[Dict[str, Any]]:
        """Per-instance results for `record`, evaluating only if not cached.

        `_evaluate_population` already stores `per_sample_details` on every
        record it scores, drawn from `_sample_dev` with the default seed. Any
        operator that re-evaluates the same record on the same pool to read its
        failures is recomputing an identical result at full dev cost — measured
        at ~15% of a SEEC run and ~25% of a GEPA-Micro run.

        Populates score and performance_vector too when it does have to
        evaluate, so the record is left consistent either way.
        """
        if record.per_sample_details:
            return record.per_sample_details
        if not record.text:
            return []
        result = self._timed_evaluate(record.text, self._sample_dev(self.eval_sample_size))
        record.per_sample_details = result.per_sample_details
        record.performance_vector = result.performance_vector
        record.score = result.score
        record.scores["dev"] = result.score
        return record.per_sample_details

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
        max_new_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate text using the LLM.

        max_new_tokens defaults to the LLM's own default_max_new_tokens (set
        from llm.max_new_tokens in the run config) rather than a hardcoded
        constant. This used to be hardcoded to 512 regardless of what a run's
        YAML set llm.max_new_tokens to -- that field was accepted by the
        config schema and never read by any backend, so every operator call
        (rewrite, critique, crossover, ...) silently generated at 512 no
        matter what the config asked for.
        """
        config = GenerationConfig(
            temperature=temperature,
            max_new_tokens=max_new_tokens
            if max_new_tokens is not None
            else self.llm.default_max_new_tokens,
            do_sample=temperature > 0,
        )
        return self.llm.generate(instruction, config, system_prompt=system_prompt)

    def _get_config_dict(self) -> Dict[str, Any]:
        """Get optimizer configuration as dict."""
        return {
            "method": self.name,
            "population_size": self.population_size,
            "num_iterations": self.num_iterations,
            # What was requested vs what the budget allowed. Two methods at the
            # same max_calls can finish very different amounts of search.
            "generations_completed": getattr(self, "_generations_completed", None),
            "search_stop_reason": getattr(self, "_search_stop_reason", None),
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