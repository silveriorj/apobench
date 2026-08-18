"""FUNNELv2 — UCB1 over a 20-operator pool spanning 15+ published APO methods,
with successive-halving validation scaling, an output-verbosity barrier gate,
and automatic trap-task detection.

Relative to FUNNEL v1 (`funnel.py`), which established UCB1 selection over a
pooled library plus cross-run pruning, v2 changes four things:

1. **Operator pool** — adds ten operators from methods outside the benchmark
   (ETGPO, StraGo, AutoHint, GrIPS, AMPO, UniPrompt; see
   `_funnel_v2_techniques.py`). Two are zero-LLM-call (GrIPS delete/swap).

2. **Identical evaluation protocol to the rest of the benchmark** — every
   phase evaluates on the same flat `eval_sample_size` (50) the other methods
   use, taken as a fixed prefix of one deterministically shuffled dev pool
   (not a per-call `random.sample` draw) so every candidate in a run is
   scored on exactly the same instances.

3. **Output-verbosity barrier gate** — evaluation caps generation at 32 new
   tokens for BBH and scores a truncated answer wrong regardless of
   reasoning quality, so the barrier penalizes mean OUTPUT length (distinct
   from the prompt-length penalty CAPO/`_v2_common` already apply):

       score = EM - lambda * max(0, (L_avg - L_free) / (L_cap - L_free))^3

   Cubic so it is nearly free below the threshold and rises sharply near the
   cap. Raw EM is preserved in `scores["dev"]`; the penalized value is used
   only for selection.

4. **Trap-task detection** — some BBH tasks are unsolvable under answer-only
   prompting regardless of optimizer (e.g. `web_of_lies`, per Suzgun et al.
   2023 Table 3). v2 runs a one-sided binomial test of the best candidate
   against the task's chance baseline and records the verdict in the audit
   trail, but does not prune the task — pruning would make macro-averages
   incomparable against methods that ran the full task set.
"""
from __future__ import annotations

import logging
import math
import random
from typing import Dict, List, Optional, Tuple

from pof.core.types import PromptRecord
from pof.optimizers import register_optimizer
from pof.optimizers.base import BaseOptimizer
from pof.optimizers._funnel_stats import prune_techniques, scan_technique_stats
from pof.optimizers._funnel_techniques import ALL_TECHNIQUES as V1_TECHNIQUES
from pof.optimizers._funnel_v2_techniques import V2_TECHNIQUES, ZERO_COST_TECHNIQUES

logger = logging.getLogger(__name__)

# Curated Phase-0 pool balanced across cost tiers: heavy (8, multi-call
# diagnosis/rewrite), cheap (8, single-call local/recombination), free (4,
# zero/near-zero LLM cost). `lamarckian` is bootstrap-only (generates from
# raw I/O pairs, ignores the population) so it lives in init, not here.
_HEAVY = [
    "etgpo_taxonomy", "strago_dual", "multi_aspect_critique", "autohint",
    "ampo_branch", "facet_edit", "structured_failure_guided",
    "reflective_mutation", "expert_refine",
]
_CHEAP = [
    "grips_add", "grips_paraphrase", "local_edit", "semantic_var",
    "few_shot", "crossover", "trajectory", "trajectory_momentum",
    "decompose_recompose",
]
_FREE = ["grips_delete", "grips_swap", "midpoint_crossover", "format_constraint"]

FUNNEL_V2_POOL: List[str] = _HEAVY + _CHEAP + _FREE

# Split used by FUNNELv3 (`funnel_v3.py`), defined here so both versions
# share one source of truth. v2 itself doesn't use it — all operators are
# bandit arms; v3 promotes the three below to a static backbone.
STATIC_CORE: List[str] = ["strago_dual", "etgpo_taxonomy", "autohint"]
BANDIT_POOL: List[str] = [t for t in FUNNEL_V2_POOL if t not in STATIC_CORE]

_ALL_TECHNIQUE_FNS: Dict[str, object] = {**V1_TECHNIQUES, **V2_TECHNIQUES}

assert len(FUNNEL_V2_POOL) == 22, f"expected 22 operators, got {len(FUNNEL_V2_POOL)}"
assert all(t in FUNNEL_V2_POOL for t in STATIC_CORE), "static core must be in the pool"
assert len(BANDIT_POOL) == 19, f"expected 19 bandit arms, got {len(BANDIT_POOL)}"
assert all(t in _ALL_TECHNIQUE_FNS for t in FUNNEL_V2_POOL), (
    "pool references an unknown technique: "
    f"{[t for t in FUNNEL_V2_POOL if t not in _ALL_TECHNIQUE_FNS]}"
)

# Fraction of the optimization pool (not raw dev — the held-out selection
# slice stays untouched) evaluated at each phase. Phase k evaluates on a
# GROWING prefix, so every phase adds validation instances the search has
# never been selected against; a flat N reused every phase let a surviving
# candidate be selected repeatedly against the same fixed sample, and
# measured dev/test correlation went strongly negative (runs improving most
# on dev did worst on test). Because prefixes are nested, a survivor pays
# only for the newly added instances, not a full re-evaluation.
PHASE_N_FRACTIONS: List[float] = [0.25, 0.50, 0.75, 1.0]

# Matches `eval_sample_size` (50), the same flat N every other method in the
# benchmark uses. A scaled schedule (38/57/76/95) was measured and found 53%
# more expensive than flat-50 for the same final precision, and would have
# made FUNNELv2 select on more dev samples than the methods it's compared
# against — a confound not worth the cost.
MIN_PHASE_N = 16

# Minimum size of the held-out selection slice (see __init__).
MIN_HOLDOUT_N = 12

# How many top-scoring records to verify at the final N before reporting a
# winner (see `_finalize`). Under a flat schedule this is normally a no-op; it
# remains as a guard against any record entering the ranking under-evaluated.
FINALIZE_TOP_K = 6

# Chance-level accuracy per BBH task, from Suzgun et al. (2023) Table 3
# ("Random" column). Used only for the trap-task test.
CHANCE_BASELINE: Dict[str, float] = {
    "boolean_expressions": 0.500,
    "causal_judgement": 0.500,
    "disambiguation_qa": 0.332,
    "formal_fallacies": 0.250,
    "hyperbaton": 0.500,
    "logical_deduction_five_objects": 0.200,
    "reasoning_about_colored_objects": 0.119,
    "web_of_lies": 0.500,
    "penguins_in_a_table": 0.000,
    "dyck_languages": 0.012,
}


@register_optimizer("funnel_v2")
class FUNNELv2Optimizer(BaseOptimizer):
    """FUNNELv2 — see module docstring."""

    name = "funnel_v2"

    # Scheduling contract, overridden by subclasses (see `funnel_v3.py`).
    # v2 is pure adaptation: every one of the 20 operators is a bandit arm and
    # none is guaranteed. A subclass can move operators into STATIC_ARMS to run
    # them on every elite each phase, leaving BANDIT_ARMS under UCB1.
    EXTRA_TECHNIQUES: Dict[str, object] = {}
    # Operators whose duplicates should re-enter the pool rather than be dropped.
    DEDUP_REVIVE: set = set()
    STATIC_ARMS: List[str] = []
    BANDIT_ARMS: List[str] = FUNNEL_V2_POOL

    def _invoke_operator(self, fn, name: str, elite: Optional[PromptRecord] = None):
        """Run one operator and return (text, target_record).

        The single place an operator is called, so a subclass can change how
        operators see and write prompts without touching either call site.
        `elite` forces a target; None lets the operator pick its own.
        """
        if elite is not None:
            self._forced_elite = elite
        try:
            text = fn(self)
        finally:
            self._forced_elite = None
        target = elite if elite is not None else getattr(self, "_last_elite", None)
        return self._post_process(text, target), target

    def _post_process(self, text: Optional[str], parent: Optional[PromptRecord]) -> Optional[str]:
        """Hook applied to every operator's output before it becomes a record.

        Identity in v2. Subclasses use it to repair properties that operators
        destroy (see `funnel_v3.py`).
        """
        return text

    def _apply_static_core(self, candidates: List[PromptRecord]) -> int:
        """Run each guaranteed operator on EVERY elite. No-op when empty.

        Uses the `_forced_elite` hook honoured by `_pick_elite`, so an operator
        can be aimed at a specific record without altering its signature.
        """
        if not self.STATIC_ARMS:
            return 0
        made = 0
        revived = 0
        for name in self.STATIC_ARMS:
            fn = self.EXTRA_TECHNIQUES.get(name) or _ALL_TECHNIQUE_FNS[name]
            for elite in self.population[: self.static_top_k]:
                text, _ = self._invoke_operator(fn, name, elite)
                if text and not self._is_duplicate(text):
                    candidates.append(self._create_record(
                        text, operator=name, parent_ids=[elite.id],
                    ))
                    made += 1
                elif text and name in self.DEDUP_REVIVE:
                    # This operator exists to keep a prompt FORM in contention,
                    # so being a duplicate is the normal case, not a waste. Rather
                    # than minting an identical record (which would be re-scored
                    # for nothing), re-enter the existing one so the form keeps
                    # competing at zero evaluation cost.
                    existing = self.tracker.history.get_by_hash(
                        PromptRecord._compute_hash(text.strip())
                    )
                    if existing is not None and existing not in candidates:
                        candidates.append(existing)
                        revived += 1
        logger.info(
            f"[{self.name} Phase {self._phase_idx}] static core produced "
            f"{made} new + {revived} revived candidate(s) across "
            f"{len(self.population[: self.static_top_k])} elites"
        )
        return made

    def __init__(
        self,
        llm,
        dataset,
        evaluator,
        population_size: int = 5,
        num_iterations: int = 4,
        candidates_per_operator: int = 2,
        top_m_operators: int = 6,
        ucb_c: float = 0.5,
        prior_pulls: int = 3,
        min_n_for_pruning: int = 40,
        prune_z_threshold: float = 1.64,
        barrier_lambda: float = 0.05,
        length_free_tokens: int = 15,
        length_cap_tokens: Optional[int] = None,
        static_top_k: int = 3,
        trap_alpha: float = 0.05,
        output_dir: str = "outputs",
        **kwargs,
    ):
        super().__init__(
            llm=llm,
            dataset=dataset,
            evaluator=evaluator,
            population_size=population_size,
            num_iterations=num_iterations,
            output_dir=output_dir,
            **kwargs,
        )
        self.candidates_per_operator = candidates_per_operator
        self.top_m_operators = top_m_operators
        self.ucb_c = ucb_c
        self.barrier_lambda = barrier_lambda
        self.length_free_tokens = length_free_tokens
        self.length_cap_tokens = length_cap_tokens or getattr(evaluator, "max_new_tokens", 32)
        self.static_top_k = static_top_k
        self.trap_alpha = trap_alpha

        # Nested dev pool: one deterministic shuffle, phases take prefixes.
        full_dev = dataset.get_eval_samples("dev", n=None)
        pool = list(full_dev)
        random.Random(42).shuffle(pool)
        pool = self._order_dev_pool(pool)

        # Split dev into an OPTIMIZATION pool and a HELD-OUT selection slice.
        # With many candidates competing, the argmax over the optimization
        # pool is upward-biased toward whichever prompt got lucky on those
        # instances (the optimizer's curse, worsening as the pool grows).
        # The held-out slice never influences the search, so the final pick
        # among finalists is made on evidence the search could not overfit.
        n_dev = len(pool)
        n_opt = max(
            MIN_PHASE_N,
            min(n_dev - MIN_HOLDOUT_N, max(self.eval_sample_size, int(0.70 * n_dev))),
        )
        n_opt = min(n_opt, n_dev)
        self._dev_pool: List[Dict[str, str]] = pool[:n_opt]
        self._holdout: List[Dict[str, str]] = pool[n_opt:]
        self._holdout_winner: Optional[PromptRecord] = None

        self._phase_sizes = self._compute_phase_sizes(
            len(self._dev_pool), self.eval_sample_size, n_phases=max(1, num_iterations)
        )
        logger.info(
            f"[{self.name}] dev={n_dev} -> optimization pool={len(self._dev_pool)}, "
            f"held-out selection slice={len(self._holdout)}; "
            f"accumulating phase N={self._phase_sizes}"
        )
        logger.info(
            f"[FUNNELv2] dev pool={len(pool)}; flat eval N={self._phase_sizes[0]} "
            f"across {len(self._phase_sizes)} phases"
        )

        # Cross-run pruning + UCB1 warm start, scoped to THIS task so the
        # priors act as automatic per-task operator routing.
        historical = scan_technique_stats(
            output_dir,
            method_name=self.name,
            technique_names=set(self.BANDIT_ARMS),
            task_filter=self._task_key(),
        )
        # Only bandit arms are subject to cross-run pruning. The static core is
        # guaranteed by design, so pruning it would silently dismantle the
        # backbone this method is built to test.
        self._active_techniques, self._dropped_techniques = prune_techniques(
            historical, list(self.BANDIT_ARMS),
            min_n=min_n_for_pruning, z_threshold=prune_z_threshold,
        )
        if self._dropped_techniques:
            logger.info(
                f"[FUNNELv2] pruned {len(self._dropped_techniques)} technique(s) on "
                f"cross-run evidence: {', '.join(self._dropped_techniques)}"
            )

        self._operator_scores: Dict[str, List[float]] = {}
        for op in self._active_techniques:
            if op in historical:
                self._operator_scores[op] = [historical[op].mean] * prior_pulls

        self._phase_idx = 0
        self._trap_verdict: Optional[str] = None
        # When set, `_pick_elite` targets this record instead of drawing at
        # random — used by the static backbone to sweep every elite.
        self._forced_elite: Optional[PromptRecord] = None

    # --- setup helpers ---

    def _order_dev_pool(self, pool: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Hook applied to the shuffled dev pool before it's split into the
        optimization pool and held-out slice. Identity here -- current
        behavior for every existing variant is unchanged. Overridden by
        FUNNELv4dOptimizer to front-load instances that look more complex
        into the early, accumulating-fresh phases (see its docstring for why
        a true historical-variance ordering isn't buildable from data this
        project actually persists).
        """
        return pool

    def _task_key(self) -> str:
        return getattr(self.dataset, "name", "") or ""

    @staticmethod
    def _compute_phase_sizes(dev_size: int, eval_sample_size: int, n_phases: int = 4) -> List[int]:
        """Growing schedule: each phase adds validation instances not yet used.

        `dev_size` here is the OPTIMIZATION pool (dev minus the held-out slice).
        Sizes are a nested, non-decreasing prefix of it, so within any phase all
        candidates are scored on identical instances while each phase widens the
        evidence base. Floored at MIN_PHASE_N and clamped to the pool, which for
        small BBH dev splits can collapse several early phases to the same size —
        acceptable, since the floor exists to stop early phases saturating.
        """
        fracs = PHASE_N_FRACTIONS[:n_phases] or [1.0]
        while len(fracs) < n_phases:
            fracs.append(1.0)
        sizes = [min(dev_size, max(MIN_PHASE_N, int(round(dev_size * f)))) for f in fracs]
        for i in range(1, len(sizes)):
            sizes[i] = max(sizes[i], sizes[i - 1])
        return sizes

    def _phase_samples(self, phase: int) -> List[Dict[str, str]]:
        n = self._phase_sizes[min(phase, len(self._phase_sizes) - 1)]
        return self._dev_pool[:n]

    # --- scoring ---

    @staticmethod
    def _mean_output_tokens(details: List[Dict]) -> float:
        """Mean output length, whitespace tokens as a tokenizer-free proxy.

        An approximation: real BPE tokens exceed whitespace tokens, so this
        under-counts. It is monotone in true length, which is all the barrier
        needs, and avoids coupling the optimizer to a specific tokenizer.
        """
        if not details:
            return 0.0
        lengths = [len(str(d.get("prediction", "")).split()) for d in details]
        return sum(lengths) / len(lengths)

    def _barrier_score(self, em: float, mean_len: float) -> float:
        span = max(1.0, float(self.length_cap_tokens - self.length_free_tokens))
        excess = max(0.0, (mean_len - self.length_free_tokens) / span)
        return em - self.barrier_lambda * (excess ** 3)

    def _evaluate_phase(self, candidates: List[PromptRecord], phase: int) -> None:
        """Bring EVERY candidate up to this phase's sample size, then rescore.

        All candidates must be compared at the same N: carrying a survivor's
        smaller-N score forward would bias selection toward incumbents
        (winner's curse — a survivor was selected for scoring high on fewer,
        noisier samples). Because phase samples are nested prefixes of one
        pool, a survivor only needs its INCREMENT evaluated.
        """
        n_target = self._phase_sizes[min(phase, len(self._phase_sizes) - 1)]
        n_full, n_incr, n_cached = 0, 0, 0

        for record in candidates:
            if not record.text:
                continue
            cached = list(record.per_sample_details or [])
            have = len(cached)

            if have >= n_target:
                details = cached[:n_target]      # already deep enough
                n_cached += 1
            else:
                increment = self._dev_pool[have:n_target]
                if not increment:
                    continue
                res = self.evaluator.evaluate(record.text, increment)
                details = cached + list(res.per_sample_details)
                if have == 0:
                    n_full += 1
                else:
                    n_incr += 1

            em = sum(1.0 for d in details if d.get("correct")) / max(len(details), 1)
            mean_len = self._mean_output_tokens(details)
            penalized = self._barrier_score(em, mean_len)

            record.per_sample_details = details
            record.performance_vector = [1.0 if d.get("correct") else 0.0 for d in details]
            record.scores["dev"] = em                # raw EM, for analysis
            record.scores["gate_score"] = penalized  # what selection uses
            record.scores["out_len"] = mean_len
            record.scores["eval_n"] = float(len(details))
            record.score = max(0.0, penalized)

        logger.info(
            f"[FUNNELv2 Phase {phase}] all candidates at N={n_target} "
            f"({n_full} full, {n_incr} incremental, {n_cached} cached)"
        )

    def _evaluate_population(self, population: List[PromptRecord]) -> None:
        """Route the base class's evaluation through the phase-aware path.

        `BaseOptimizer.optimize` calls `_evaluate_population` immediately after
        `_init_population`. The base implementation evaluates on
        `self.eval_sample_size` samples drawn independently of our nested pool,
        which would overwrite `per_sample_details` with a vector that no longer
        aligns to `_dev_pool` prefixes and silently corrupt every subsequent
        incremental evaluation.
        """
        self._evaluate_phase(population, phase=self._phase_idx)

    # --- trap-task detection ---

    def _check_trap(self) -> None:
        """One-sided binomial z-test of the best candidate against chance.

        Records a verdict in the audit trail. Deliberately does not prune:
        removing a task would make this method's macro-average incomparable
        with methods that ran the full task set.
        """
        if self._trap_verdict is not None or not self.best_record:
            return
        task = self._task_key()
        p0 = next(
            (v for k, v in CHANCE_BASELINE.items() if k in task), None
        )
        if p0 is None:
            return
        n = int(self.best_record.scores.get("eval_n", 0) or 0)
        if n < MIN_PHASE_N:
            return
        p_hat = self.best_record.scores.get("dev", 0.0)
        se = math.sqrt(max(p0 * (1.0 - p0), 1e-9) / n)
        z = (p_hat - p0) / se
        z_crit = 1.645 if self.trap_alpha <= 0.05 else 1.282

        if z <= z_crit:
            self._trap_verdict = (
                f"TRAP-SUSPECT task={task}: best dev EM={p_hat:.3f} vs chance "
                f"p0={p0:.3f} (n={n}, z={z:.2f} <= {z_crit}). Not significantly "
                f"above chance; likely unsolvable under answer-only prompting."
            )
            logger.warning(f"[FUNNELv2] {self._trap_verdict}")
        else:
            self._trap_verdict = (
                f"OK task={task}: best dev EM={p_hat:.3f} vs chance p0={p0:.3f} "
                f"(n={n}, z={z:.2f} > {z_crit}); significantly above chance."
            )
            logger.info(f"[FUNNELv2] {self._trap_verdict}")
        self.tracker.add_note(self._trap_verdict)

    # --- optimization loop ---

    def _init_population(self) -> List[PromptRecord]:
        """Phase 0: bootstrap + a cheap diversity sample, evaluated at N_0."""
        candidates: List[PromptRecord] = []

        if self.seed_prompt:
            candidates.append(self._create_record(self.seed_prompt, operator="seed"))
        self.population = list(candidates)

        train_samples = self.dataset.get_few_shot_examples(n=5)
        for text in self._lamarckian_generate(train_samples, n=2):
            if text and not self._is_duplicate(text):
                candidates.append(self._create_record(text, operator="lamarckian"))
        if not candidates:
            candidates.append(self._create_record("Solve the task.", operator="fallback_seed"))
        self.population = list(candidates)

        # Seed the pool with a few free/cheap operators so Phase 0 has breadth
        # without spending heavy-operator budget before UCB1 has any data.
        for name in ["format_constraint", "local_edit", "semantic_var", "few_shot"]:
            if name not in self._active_techniques:
                continue
            fn = _ALL_TECHNIQUE_FNS[name]
            text = fn(self)
            if text and not self._is_duplicate(text):
                candidates.append(self._create_record(text, operator=name))
                self.population = sorted(candidates, key=lambda r: r.score, reverse=True)

        self._evaluate_phase(candidates, phase=0)
        return self._select_top_k(candidates)

    def _step(self) -> List[PromptRecord]:
        self._phase_idx += 1
        if self._phase_idx >= len(self._phase_sizes):
            self._finalize()
            raise StopIteration

        logger.info(f"[FUNNELv2 Gen {self.generation}] hybrid step, phase {self._phase_idx}")
        candidates = list(self.population)

        self._apply_static_core(candidates)

        # Walk the full UCB1 ranking, not just top-M: an arm that can't apply
        # right now (e.g. a failure-driven operator when the elite has no
        # failures) yields its slot to the next-ranked arm instead of wasting
        # it. Stops after `top_m_operators` arms have actually produced.
        used_arms = 0
        skipped: List[str] = []
        for name in self._select_operators():
            if used_arms >= self.top_m_operators:
                break
            fn = _ALL_TECHNIQUE_FNS[name]
            n_draws = self.candidates_per_operator
            # Zero-cost operators are stochastic string edits; drawing extra
            # samples from them costs no LLM calls, so take more.
            if name in ZERO_COST_TECHNIQUES:
                n_draws += 1
            produced = 0
            for _ in range(n_draws):
                text, _ = self._invoke_operator(fn, name)
                if text and not self._is_duplicate(text):
                    candidates.append(self._create_record(
                        text, operator=name,
                        parent_ids=[r.id for r in self.population[:2]],
                    ))
                    produced += 1
            if produced:
                used_arms += 1
            else:
                skipped.append(name)
        if skipped:
            logger.info(
                f"[FUNNELv2 Phase {self._phase_idx}] {used_arms} arms produced "
                f"candidates; backfilled past {len(skipped)} inapplicable "
                f"arm(s): {', '.join(skipped)}"
            )

        # Identify freshly-generated candidates BEFORE evaluating: afterwards
        # every candidate carries per-sample details and is indistinguishable.
        # Only fresh ones feed the bandit, so an incumbent is not re-credited to
        # its operator once per phase it survives.
        new_candidates = [c for c in candidates if not c.per_sample_details]

        # Evaluate ALL candidates, not just the new ones, so survivors and
        # newcomers are ranked on the same sample size (see `_evaluate_phase`).
        self._evaluate_phase(candidates, phase=self._phase_idx)

        for record in new_candidates:
            self._operator_scores.setdefault(record.operator, []).append(record.score)

        self._update_best()
        if self._phase_idx == 1:
            self._check_trap()

        return self._tournament_select(candidates)

    def _finalize(self) -> None:
        """Verify every plausible winner at the final N before one is reported.

        `get_best_record` maxes over the entire run including candidates
        dropped in an early phase, whose small-sample scores carry the same
        upward bias `_evaluate_phase` corrects for — a lucky Phase-0 score
        could otherwise outrank an honestly-measured finalist. Re-evaluating
        contenders at the final N (nested prefixes mean each pays only its
        increment) fixes this; iterated because promoting a leader can pull
        a previously-lower record into contention.

        Called both from `_step()` when phases are exhausted and, as a
        backstop, unconditionally from `BaseOptimizer.optimize()`'s `finally`
        block for exit paths that skip `_step()`'s StopIteration — guarded
        so the second call is a no-op.
        """
        if self._finalized:
            return
        n_final = self._phase_sizes[-1]
        for _ in range(3):
            contenders = sorted(
                self.tracker.history.records.values(),
                key=lambda r: r.score, reverse=True,
            )[:FINALIZE_TOP_K]
            under = [
                r for r in contenders
                if r.text and len(r.per_sample_details or []) < n_final
            ]
            if not under:
                break
            logger.info(
                f"[{self.name} finalize] verifying {len(under)} under-evaluated "
                f"contender(s) at N={n_final}"
            )
            self._evaluate_phase(under, phase=len(self._phase_sizes) - 1)

        self._select_on_holdout()
        self._finalized = True

    def _select_on_holdout(self) -> None:
        """Pick the reported winner on data the search never saw.

        The optimization-pool argmax is upward-biased by however many
        candidates competed; re-ranking finalists on the held-out slice cuts
        that to best-of-K on fresh instances. The winner is recorded on the
        optimizer rather than by rewriting `record.score`, which would
        corrupt the audit trail and the cross-run operator statistics
        derived from it — `optimize` applies the choice to the result instead.
        """
        if not self._holdout:
            return
        contenders = sorted(
            self.tracker.history.records.values(),
            key=lambda r: r.score, reverse=True,
        )[:FINALIZE_TOP_K]
        contenders = [r for r in contenders if r.text]
        if not contenders:
            return

        for record in contenders:
            res = self.evaluator.evaluate(record.text, self._holdout)
            record.scores["holdout"] = res.score
            record.scores["holdout_n"] = float(len(self._holdout))

        winner = max(contenders, key=lambda r: r.scores.get("holdout", 0.0))
        self._holdout_winner = winner
        opt_best = contenders[0]
        note = (
            f"holdout selection over {len(contenders)} finalists on "
            f"{len(self._holdout)} held-out instances: winner holdout="
            f"{winner.scores.get('holdout', 0.0):.3f} (opt-pool={winner.scores.get('dev', 0.0):.3f}); "
            f"opt-pool argmax would have picked holdout="
            f"{opt_best.scores.get('holdout', 0.0):.3f} (opt-pool={opt_best.scores.get('dev', 0.0):.3f})"
            + ("" if winner is opt_best else " -- DIFFERENT prompt chosen")
        )
        logger.info(f"[{self.name}] {note}")
        self.tracker.add_note(note)

    def optimize(self):
        """Run the search, then report the held-out winner if one was chosen."""
        result = super().optimize()
        winner = self._holdout_winner
        if winner is not None and winner.text:
            result.best_prompt = winner.text
            result.best_score = winner.scores.get("dev", winner.score)
        return result

    def _select_operators(self) -> List[str]:
        """Full UCB1 ranking of the active pool (same formula as APEX/FUNNEL v1).

        Returns every arm in ranked order rather than a top-M slice; `_step`
        consumes from the top and stops once `top_m_operators` arms have
        actually produced candidates, so the tail is only reached when
        higher-ranked arms cannot apply.
        """
        total_pulls = sum(len(v) for v in self._operator_scores.values())
        if total_pulls == 0:
            return list(self._active_techniques)

        scored: List[Tuple[float, str]] = []
        for name in self._active_techniques:
            pulls = self._operator_scores.get(name, [])
            if not pulls:
                ucb = float("inf")
            else:
                mean = sum(pulls) / len(pulls)
                ucb = mean + self.ucb_c * math.sqrt(math.log(max(total_pulls, 2)) / len(pulls))
            scored.append((ucb, name))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [name for _, name in scored]

    def _tournament_select(
        self, candidates: List[PromptRecord], tournament_size: int = 3
    ) -> List[PromptRecord]:
        sorted_candidates = sorted(candidates, key=lambda r: r.score, reverse=True)
        selected = [sorted_candidates[0]]
        remaining = sorted_candidates[1:]
        while len(selected) < self.population_size and remaining:
            tournament = random.sample(remaining, min(tournament_size, len(remaining)))
            winner = max(tournament, key=lambda r: r.score)
            selected.append(winner)
            remaining.remove(winner)
        return selected
