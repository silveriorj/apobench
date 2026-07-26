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

2. **Successive-halving validation scaling** — v1 evaluated every candidate on
   the same fixed 50-sample dev subset at every phase. v2 scales the validation
   size across phases (small N when there are many candidates and only coarse
   ranking is needed; full dev when few candidates remain and selection
   precision determines the final answer), so evaluation budget concentrates on
   exploitation rather than being spent uniformly on early exploration.

   Sample sets are strict NESTED PREFIXES of one shuffled dev pool. This
   matters: `TaskDataset.get_eval_samples` uses `random.sample`, whose outputs
   are NOT nested across different `n` (CPython switches selection algorithms
   at a size threshold), so naively varying `n` would make phase-to-phase score
   differences reflect subset composition as much as sample size.

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

_ALL_TECHNIQUE_FNS: Dict[str, object] = {**V1_TECHNIQUES, **V2_TECHNIQUES}

assert len(FUNNEL_V2_POOL) == 20, f"expected 20 operators, got {len(FUNNEL_V2_POOL)}"
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
PHASE_N_FRACTIONS: List[float] = [0.30, 0.45, 0.70, 1.0]
MIN_PHASE_N = 16

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

    def __init__(
        self,
        llm,
        dataset,
        evaluator,
        population_size: int = 5,
        num_iterations: int = 4,
        candidates_per_operator: int = 2,
        top_m_operators: int = 8,
        ucb_c: float = 0.5,
        prior_pulls: int = 3,
        min_n_for_pruning: int = 40,
        prune_z_threshold: float = 1.64,
        barrier_lambda: float = 0.05,
        length_free_tokens: int = 15,
        length_cap_tokens: Optional[int] = None,
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
        self.trap_alpha = trap_alpha

        # Nested dev pool: one deterministic shuffle, phases take prefixes.
        full_dev = dataset.get_eval_samples("dev", n=None)
        pool = list(full_dev)
        random.Random(42).shuffle(pool)
        self._dev_pool: List[Dict[str, str]] = pool
        self._phase_sizes = self._compute_phase_sizes(len(pool))
        logger.info(
            f"[FUNNELv2] dev pool={len(pool)}; phase N schedule={self._phase_sizes}"
        )

        # Cross-run pruning + UCB1 warm start, scoped to THIS task so the
        # priors act as automatic per-task operator routing.
        historical = scan_technique_stats(
            output_dir,
            method_name=self.name,
            technique_names=set(FUNNEL_V2_POOL),
            task_filter=self._task_key(),
        )
        self._active_techniques, self._dropped_techniques = prune_techniques(
            historical, list(FUNNEL_V2_POOL),
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

    # --- setup helpers ---

    def _task_key(self) -> str:
        return getattr(self.dataset, "name", "") or ""

    @staticmethod
    def _compute_phase_sizes(dev_size: int) -> List[int]:
        sizes = []
        for frac in PHASE_N_FRACTIONS:
            n = max(MIN_PHASE_N, int(round(dev_size * frac)))
            sizes.append(min(n, dev_size))
        # Enforce monotonic non-decreasing after clamping.
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
        """Evaluate on this phase's nested sample; store raw EM and penalized score."""
        samples = self._phase_samples(phase)
        to_eval = [r for r in candidates if r.score == 0.0 and r.text]
        for record in to_eval:
            result = self.evaluator.evaluate(record.text, samples)
            mean_len = self._mean_output_tokens(result.per_sample_details)
            penalized = self._barrier_score(result.score, mean_len)

            record.performance_vector = result.performance_vector
            record.per_sample_details = result.per_sample_details
            record.scores["dev"] = result.score          # raw EM, for analysis
            record.scores["gate_score"] = penalized      # what selection uses
            record.scores["out_len"] = mean_len
            record.scores["eval_n"] = float(len(samples))
            record.score = max(0.0, penalized)
        if to_eval:
            logger.info(
                f"[FUNNELv2 Phase {phase}] evaluated {len(to_eval)} candidates "
                f"on N={len(samples)}"
            )

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
            raise StopIteration

        logger.info(f"[FUNNELv2 Gen {self.generation}] UCB1 step, phase {self._phase_idx}")
        candidates = list(self.population)

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

        new_candidates = [c for c in candidates if c.score == 0.0]
        self._evaluate_phase(new_candidates, phase=self._phase_idx)

        for record in new_candidates:
            self._operator_scores.setdefault(record.operator, []).append(record.score)

        self._update_best()
        if self._phase_idx == 1:
            self._check_trap()

        return self._tournament_select(candidates)

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
