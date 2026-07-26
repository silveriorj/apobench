"""FUNNELv2 — UCB1 over a 20-operator pool spanning 15+ published APO methods,
with successive-halving validation scaling, an output-verbosity barrier gate,
and automatic trap-task detection.

Relationship to FUNNEL v1 (`funnel.py`): v1 established UCB1 selection over a
pooled library plus cross-run pruning. v2 keeps both and changes four things,
each motivated by a measured result from this project or a specific published
method:

1. **Operator pool** — v1 pooled operators from the six methods already in our
   benchmark. v2 adds ten operators from methods outside it (ETGPO, StraGo,
   AutoHint, GrIPS, AMPO, UniPrompt) — see `_funnel_v2_techniques.py`. Two are
   ZERO-LLM-call (GrIPS delete/swap), buying free exploration under a call cap.

2. **Identical evaluation protocol to the rest of the benchmark** — every phase
   evaluates on a flat `eval_sample_size` (50) sample, the same value the other
   methods use, so the operator pool and the barrier gate are the only things
   that differ. A scaled schedule was tried and measured first; it cost 53%
   more than flat-50 and introduced a third protocol difference, so it was
   dropped (see the note above `MIN_PHASE_N`).

   The sample is a fixed prefix of one deterministically shuffled dev pool
   rather than a per-call draw. `TaskDataset.get_eval_samples` uses
   `random.sample`, whose outputs are not nested across different `n` and which
   re-draws per call; taking a stable prefix guarantees every candidate in a
   run is scored on exactly the same instances, which is what makes the
   candidate-vs-candidate comparison in selection meaningful.

3. **Output-verbosity barrier gate** — evaluation caps generation at 32 new
   tokens for BBH, and a truncated answer is always scored wrong. Prompts that
   induce verbose answers therefore fail for a reason unrelated to reasoning
   quality. The barrier penalizes mean OUTPUT length (not prompt length, which
   is what CAPO and `_v2_common` already penalize):

       score = EM - lambda * max(0, (L_avg - L_free) / (L_cap - L_free))^3

   Cubic so it is nearly free below the threshold and rises sharply near the
   cap. Raw EM is preserved in `scores["dev"]`; the penalized value is used
   only for selection.

4. **Trap-task detection** — some BBH tasks are unsolvable under answer-only
   prompting regardless of optimizer: every large model in Suzgun et al. (2023,
   Table 3) scores ~51.6 on `web_of_lies` answer-only and 92-100 with CoT. v2
   runs a one-sided binomial test of the best candidate against the task's
   chance baseline and records the verdict in the audit trail. It does NOT
   prune the task: pruning would make macro-averages incomparable against
   methods that ran all eight tasks. Detection is reported, not acted on.
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

# The curated 20-operator Phase-0 pool, balanced across cost tiers so UCB1 has
# both cheap breadth and expensive depth to arbitrate between.
#
#   heavy (8)  — multi-call, high-yield diagnosis/rewrite operators
#   cheap (8)  — single-call local or recombination edits
#   free  (4)  — zero or near-zero LLM cost
#
# `lamarckian` is deliberately NOT an arm: it is bootstrap-only (it ignores the
# population and generates from raw I/O pairs), so it belongs in init, not in
# the per-iteration arm set.
_HEAVY = [
    "etgpo_taxonomy", "strago_dual", "autohint", "ampo_branch",
    "facet_edit", "structured_failure_guided", "reflective_mutation",
    "expert_refine",
]
_CHEAP = [
    "grips_add", "grips_paraphrase", "local_edit", "semantic_var",
    "few_shot", "crossover", "trajectory", "decompose_recompose",
]
_FREE = ["grips_delete", "grips_swap", "midpoint_crossover", "format_constraint"]

FUNNEL_V2_POOL: List[str] = _HEAVY + _CHEAP + _FREE

# --- Split used by FUNNELv3 (see `funnel_v3.py`), defined here so both
# --- versions share one source of truth for the operator names.
#
# v2 itself does NOT use this split: it is pure adaptation, with all 20
# operators as bandit arms and none guaranteed. v3 promotes the three below to
# a static backbone run on every elite each phase, leaving 17 under UCB1.
STATIC_CORE: List[str] = ["strago_dual", "etgpo_taxonomy", "autohint"]
BANDIT_POOL: List[str] = [t for t in FUNNEL_V2_POOL if t not in STATIC_CORE]

_ALL_TECHNIQUE_FNS: Dict[str, object] = {**V1_TECHNIQUES, **V2_TECHNIQUES}

assert len(FUNNEL_V2_POOL) == 20, f"expected 20 operators, got {len(FUNNEL_V2_POOL)}"
assert all(t in FUNNEL_V2_POOL for t in STATIC_CORE), "static core must be in the pool"
assert len(BANDIT_POOL) == 17, f"expected 17 bandit arms, got {len(BANDIT_POOL)}"
assert all(t in _ALL_TECHNIQUE_FNS for t in FUNNEL_V2_POOL), (
    "pool references an unknown technique: "
    f"{[t for t in FUNNEL_V2_POOL if t not in _ALL_TECHNIQUE_FNS]}"
)

# Fraction of the available dev split evaluated at each phase. Rising fractions
# concentrate evaluation on the later, higher-stakes selection steps.
#
# The Phase-0 fraction has a floor for a reason found empirically: at 0.15 of a
# 127-example dev split (N=19), easy tasks such as boolean_expressions saturate
# at EM=1.000 in Phase 0. A saturated score is measurement noise rather than a
# solved task, and it starves every failure-driven operator of input — seven of
# the eight heavy operators no-op when the elite has no failures, collapsing the
# pool to its cheap tier exactly when budget is most available. Starting at 0.30
# keeps early scores off the ceiling.
# Evaluation uses a FLAT sample size, equal to `eval_sample_size` (50 by
# default, set from `evaluation.sample_size` in the run config) — the same
# value every other method in the benchmark uses.
#
# An earlier design scaled N across phases (38/57/76/95 on a 127-example dev
# split), on the successive-halving argument that precision matters most at the
# final selection. Measured, that trade did not pay for itself here:
#
#   schedule                sample-evals   final N
#   scaled 38/57/76/95           4,199        95
#   flat 95                      5,225        95
#   flat 50                      2,750        50
#
# Scaling was 20% cheaper than a flat 95 for identical final precision, but 53%
# MORE expensive than the flat 50 the rest of the benchmark runs at, roughly
# doubling wall-clock time. It also made FUNNELv2 select on 95 dev samples while
# every method it is compared against selects on 50 — a third simultaneous
# protocol difference (alongside the operator pool and the barrier gate), which
# would make any observed difference hard to attribute to the thing under test.
#
# Flat N keeps the comparison clean. The nested-prefix pool below is retained
# because it still guarantees every candidate is scored on an identical sample.
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
    STATIC_ARMS: List[str] = []
    BANDIT_ARMS: List[str] = FUNNEL_V2_POOL

    def _apply_static_core(self, candidates: List[PromptRecord]) -> int:
        """Run each guaranteed operator on EVERY elite. No-op when empty.

        Uses the `_forced_elite` hook honoured by `_pick_elite`, so an operator
        can be aimed at a specific record without altering its signature.
        """
        if not self.STATIC_ARMS:
            return 0
        made = 0
        for name in self.STATIC_ARMS:
            fn = _ALL_TECHNIQUE_FNS[name]
            for elite in self.population[: self.static_top_k]:
                self._forced_elite = elite
                try:
                    text = fn(self)
                finally:
                    self._forced_elite = None
                if text and not self._is_duplicate(text):
                    candidates.append(self._create_record(
                        text, operator=name, parent_ids=[elite.id],
                    ))
                    made += 1
        logger.info(
            f"[{self.name} Phase {self._phase_idx}] static core produced "
            f"{made} candidate(s) across {len(self.population[: self.static_top_k])} elites"
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

        # Split dev into an OPTIMIZATION pool and a HELD-OUT selection slice.
        #
        # Measured motivation: with a flat N=50 and ~70 candidates competing,
        # the argmax over dev reliably picks whichever prompt was luckiest on
        # those particular 50 instances. On bbh_boolean_expressions this
        # produced dev 0.927 +/- 0.025 but test 0.838 +/- 0.062, with a dev-test
        # correlation of -0.46: the seed scoring HIGHEST on dev scored LOWEST on
        # test. That is the optimizer's curse, and it worsens as the operator
        # pool grows, because more candidates means more draws in the lottery.
        #
        # The held-out slice never influences the search, so the final choice
        # among a handful of finalists is made on evidence the search could not
        # overfit. Selection bias scales with how many things you choose
        # between: best-of-70 on a reused sample is badly biased, best-of-6 on
        # fresh instances is not.
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
            f"held-out selection slice={len(self._holdout)}"
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

    def _task_key(self) -> str:
        return getattr(self.dataset, "name", "") or ""

    @staticmethod
    def _compute_phase_sizes(dev_size: int, eval_sample_size: int, n_phases: int = 4) -> List[int]:
        """Flat schedule: every phase evaluates on the same sample size.

        Clamped to the dev split, which for BBH holds only 64-127 examples, so
        a configured 50 is reachable on every task.
        """
        n = min(max(MIN_PHASE_N, eval_sample_size), dev_size)
        return [n] * n_phases

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

        Selection compares candidates against each other, so they must all be
        measured on the SAME sample. Carrying a survivor's earlier small-N score
        forward would bias selection toward incumbents: small samples have
        higher variance, and a survivor was *selected for scoring high*, so its
        small-N estimate is upward-biased (winner's curse). Ranking that
        inflated estimate against an honest larger-N estimate of a new candidate
        systematically favours the incumbent, and the population ossifies around
        whichever candidates got lucky early.

        Standard successive halving avoids this by re-running survivors at each
        new budget. Because our phase samples are nested prefixes of one pool, a
        survivor only needs the INCREMENT evaluated — its cached per-sample
        results for the prefix stay valid — so correctness here costs only the
        extra samples, not a full re-evaluation.
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

        # Walk the full UCB1 ranking, not just the top-M. An operator that
        # cannot apply right now (failure-driven operators return None when the
        # elite has no failures on the current sample) yields its slot to the
        # next-ranked arm rather than silently wasting it. On a healthy step
        # this stops after `top_m_operators` arms and behaves identically to
        # taking the top-M directly; it only backfills when arms no-op.
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
                text = fn(self)
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

        `AuditHistory.get_best_record` returns `max(records, key=score)` over
        the ENTIRE run, including candidates dropped in an early phase. Those
        were scored on a small sample and never re-evaluated, so their scores
        carry the same upward bias that motivated `_evaluate_phase`: a Phase-0
        candidate that got lucky on 38 samples can outrank a finalist honestly
        measured on 95 and be reported as the run's best prompt — which is then
        what the runner sends to the held-out test evaluation.

        Re-evaluating the top contenders at the final N removes the bias where
        it actually matters. Nested prefixes mean each one pays only its
        increment. Iterated, because correcting the leaders can promote a
        previously-lower record into contention.
        """
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

    def _select_on_holdout(self) -> None:
        """Pick the reported winner on data the search never saw.

        The optimization pool drove every selection decision, so the argmax over
        it is upward-biased by however many candidates competed — measured at
        dev 0.927 +/- 0.025 against test 0.838 +/- 0.062, correlation -0.46.
        Re-ranking a handful of finalists on the held-out slice cuts that bias
        to best-of-K on fresh instances.

        The winner is recorded on the optimizer rather than by rewriting scores:
        `AuditHistory.get_best_record` ranks by `record.score`, and mutating
        scores to force an outcome would corrupt the audit trail and the
        cross-run operator statistics derived from it. `optimize` applies the
        choice to the returned result instead.
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
