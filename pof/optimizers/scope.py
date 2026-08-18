"""SCOPE — Signal-Calibrated Optimization with Pareto Evolution.

Most APO methods assume the dev score they optimize is a usable estimate of
prompt quality. In this project it demonstrably is not: measured dev/test
correlation is **-0.695** (dev gains anti-predict test gains) and the
seed-to-seed noise floor averages **5.48pp**, larger than nearly every
effect any method here has produced. Under those conditions a search is
mostly fitting noise, which is the single mechanism that explains why
SWIFT's fixes, three GEPA-Pareto ports, and FUNNEL-v7 all failed to move.

The variance of a prompt's measured score decomposes into variance among
responses (generation stochasticity — noise) and variance among prompts
(the signal selection needs). Search only works where the latter dominates
(cf. p1, arXiv:2604.08801). Critically, adding dev instances does not fix
this: an instance every candidate answers identically contributes zero
discriminative information while still consuming evaluation budget, so a
larger pool can *dilute* the signal it was meant to strengthen.

SCOPE therefore calibrates *what it measures against* before it searches,
then runs only mechanisms with evidence behind them:

M1  Signal-calibrated dev pool. After generation 0, per-instance
    discrimination is Var(outcome) across candidates; unanimous instances
    are dropped (subject to MIN_SIGNAL_POOL). Zero extra LLM calls -- it is
    arithmetic over performance vectors already collected -- and it makes
    every later generation *cheaper*, since the pool it evaluates on is
    smaller.

M2  Retention. Fraction of a parent's solved instances the child still
    solves. 84% of whole-prompt rewrites here destroyed parent content;
    retention makes that visible per candidate, for free.

M3  Guarded acceptance (cf. SPEAR, arXiv:2605.26275). A child is rejected
    only when it both fails to improve on its parent AND falls below the
    retention floor -- i.e. it lost content and bought nothing with it.
    Cheap monotonicity against the measured cases where an operator family
    scored *worse than doing nothing*, without suppressing exploration.

M4  Per-instance Pareto frontier (GEPA, arXiv:2507.19457) over a
    many-generation bandit loop -- the only search mechanism in this project
    that clears the noise floor with replication (+4.93pp, 9/9 pairwise seed
    dominance). The three failed SWIFT ports established it needs exactly
    this shape: many generations, not a fixed short pipeline.

M5  Cost as a real objective (cf. MO-CAPO, arXiv:2605.18869), not a cap.
    Cost is scored on the *prompt being selected* -- its own length plus the
    output length it induces -- because that is what a deployed prompt bills
    on every future query. Selection runs on an accuracy x cost front, so a
    cheaper prompt of equal accuracy wins.

Deliberately excluded, each on measured evidence: whole-prompt rewriting as
the default edit (84% destructive), `trajectory` (weakest operator by mean
and win-rate across 181 runs), `decompose_recompose` (dead, 0/6 fire rate),
and added reflection steps (harmful in arXiv:2605.26046 and flagged in this
project's own survey).

Reported metrics never use the calibrated pool. M1 changes what the search
optimizes against; the held-out selection slice and the test split are
untouched, so the number that gets published is measured the same way as
every other method's.
"""
from __future__ import annotations

import logging
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pof.core.types import PromptRecord, pareto_frontier_coverage, rank_key
from pof.optimizers import register_optimizer
from pof.optimizers.base import BaseOptimizer
from pof.optimizers.holdout import HoldoutSelectionMixin
from pof.optimizers._funnel_techniques import ALL_TECHNIQUES
from pof.optimizers._funnel_v2_techniques import V2_TECHNIQUES, ZERO_COST_TECHNIQUES

logger = logging.getLogger(__name__)

_TECHNIQUES: Dict[str, Any] = {**ALL_TECHNIQUES, **V2_TECHNIQUES}

# Curated pool. Every exclusion below is on measured evidence, not taste --
# see the module docstring. `lamarckian` is init-only by design and so is
# never a bandit arm.
SCOPE_OPERATORS: List[str] = [
    # strongest by this project's own operator audits
    "structured_failure_guided",   # most reliable operator (181-run SWIFT audit)
    "local_edit",                  # most common source of the final winner
    "semantic_var",
    "expert_refine",
    "crossover",
    # zero-LLM-call operators: free candidates, always worth a draw
    "few_shot",
    "format_constraint",
    "midpoint_crossover",
    "grips_delete",
    "grips_swap",
    # published techniques with distinct mechanisms
    "strago_dual",                 # learns from correct AND failed cases
    "etgpo_taxonomy",
    "autohint",
    "grips_add",
    "grips_paraphrase",
    "ampo_branch",
]

# Free operators get an extra draw: they cost no LLM call, so declining to
# sample them is pure loss.
_FREE_OPERATORS = set(ZERO_COST_TECHNIQUES) | {
    "few_shot", "format_constraint", "midpoint_crossover",
}


@register_optimizer("scope")
class SCOPEOptimizer(HoldoutSelectionMixin, BaseOptimizer):
    """Signal-calibrated, cost-aware prompt optimizer.

    MRO note: `HoldoutSelectionMixin` must precede `BaseOptimizer` so its
    `_finalize`/`optimize` overrides take effect. SCOPE is deliberately NOT
    a FUNNEL subclass -- FUNNEL defines its own holdout machinery and mixing
    the two would split dev twice.
    """

    name = "scope"

    # --- M1: signal calibration ---
    MIN_SIGNAL_POOL: int = 12      # never calibrate below this many instances
    # --- M3: guarded acceptance ---
    # Deliberately permissive: the guard exists to catch *catastrophic*
    # rewrites (the measured 84% destroy-the-parent's-content case), not to
    # referee normal exploration. On an M1-calibrated pool -- which by
    # construction keeps the instances candidates disagree on -- ordinary
    # retention runs low, so a high floor silently starves the search.
    RETENTION_FLOOR: float = 0.50
    # --- M5: cost objective (whitespace-token proxy, tokenizer-free) ---
    COST_INPUT_WEIGHT: float = 1.0
    COST_OUTPUT_WEIGHT: float = 1.0

    def __init__(
        self,
        llm,
        dataset,
        evaluator,
        population_size: int = 5,
        num_iterations: int = 6,
        use_holdout_selection: bool = True,
        candidates_per_operator: int = 1,
        top_m_operators: int = 5,
        ucb_c: float = 0.5,
        frontier_pull_prob: float = 0.5,
        **kwargs,
    ):
        super().__init__(
            llm=llm,
            dataset=dataset,
            evaluator=evaluator,
            population_size=population_size,
            num_iterations=num_iterations,
            **kwargs,
        )
        self.candidates_per_operator = candidates_per_operator
        self.top_m_operators = top_m_operators
        self.ucb_c = ucb_c
        self.frontier_pull_prob = frontier_pull_prob
        self._operator_scores: Dict[str, List[float]] = {}
        self._calibrated = False

        self._init_holdout(use_holdout_selection=use_holdout_selection)

        # The fixed, ordered evaluation pool. Index alignment across
        # candidates is load-bearing for M1/M2/M4 -- every record must be
        # scored on THIS list, in THIS order, so `performance_vector[i]`
        # refers to the same instance for every candidate. `_opt_pool` comes
        # from the holdout mixin (search-visible portion only); the fallback
        # is the raw dev split for holdout-disabled runs.
        pool = self._opt_pool if self._opt_pool is not None else \
            self.dataset.get_eval_samples("dev", n=None)
        if self.eval_sample_size and self.eval_sample_size < len(pool):
            pool = list(pool)[: self.eval_sample_size]
        self._signal_pool: List[Dict[str, str]] = list(pool)
        logger.info(
            f"[{self.name}] signal pool initialized: {len(self._signal_pool)} instances"
        )

    # ------------------------------------------------------------------
    # Evaluation on the fixed pool
    # ------------------------------------------------------------------

    def _evaluate_on_signal_pool(self, records: Sequence[PromptRecord]) -> None:
        """Score records on the fixed signal pool, in fixed order.

        `num_samples` is left None so `Evaluator.evaluate` never reshuffles
        (it only samples when `num_samples` is given) -- that is what keeps
        `performance_vector` index-comparable across candidates.

        Also populates `per_sample_details` for every record, which
        incidentally prevents the technique library's `_ensure_details` from
        firing: that helper would re-evaluate against a *different*,
        freshly-drawn sample set and silently desync the alignment M1
        depends on.
        """
        for record in records:
            if not record.text or record.per_sample_details:
                continue
            result = self._timed_evaluate(record.text, self._signal_pool)
            record.score = result.score
            record.scores["dev"] = result.score
            record.performance_vector = list(result.performance_vector)
            record.per_sample_details = result.per_sample_details

    # ------------------------------------------------------------------
    # M1 — signal calibration
    # ------------------------------------------------------------------

    @staticmethod
    def _discrimination(vectors: List[List[int]], n: int) -> List[float]:
        """Per-instance Bernoulli variance of outcomes across candidates.

        Zero exactly when every candidate agreed on that instance (all
        solved or none solved) -- i.e. the instance cannot distinguish any
        two candidates and contributes nothing to selection.
        """
        out: List[float] = []
        for i in range(n):
            col = [v[i] for v in vectors]
            mean = sum(col) / len(col)
            out.append(mean * (1.0 - mean))
        return out

    def _calibrate_signal_pool(self, records: Sequence[PromptRecord]) -> None:
        """Drop instances that cannot discriminate between candidates.

        Runs once, after generation 0. Costs no LLM calls and shrinks every
        subsequent evaluation. Existing performance vectors and per-sample
        details are re-indexed onto the surviving instances so alignment is
        preserved -- skipping that would silently corrupt every later
        comparison, which is the sharpest failure mode in this design.
        """
        if self._calibrated:
            return
        n = len(self._signal_pool)
        vectors = [
            list(r.performance_vector) for r in records
            if len(r.performance_vector) == n
        ]
        if len(vectors) < 2 or n == 0:
            self._calibrated = True
            return

        disc = self._discrimination(vectors, n)
        keep = [i for i, d in enumerate(disc) if d > 0.0]

        if len(keep) < self.MIN_SIGNAL_POOL:
            # Too few discriminating instances to estimate reliably. Top up
            # with the remainder (stable order) rather than shrinking into a
            # high-variance estimate -- guarding against trading one noise
            # source for another.
            filler = [i for i, d in enumerate(disc) if d <= 0.0]
            keep = sorted(keep + filler[: self.MIN_SIGNAL_POOL - len(keep)])
        if len(keep) >= n:
            self._calibrated = True
            logger.info(f"[{self.name}] M1: all {n} instances discriminate; pool unchanged")
            return

        keep = sorted(keep)
        self._signal_pool = [self._signal_pool[i] for i in keep]
        for r in records:
            if len(r.performance_vector) == n:
                r.performance_vector = [r.performance_vector[i] for i in keep]
                if len(r.per_sample_details) == n:
                    r.per_sample_details = [r.per_sample_details[i] for i in keep]
                if r.performance_vector:
                    r.score = sum(r.performance_vector) / len(r.performance_vector)
                    r.scores["dev"] = r.score

        self._calibrated = True
        note = (
            f"M1 signal calibration: dev pool {n} -> {len(keep)} instances "
            f"({n - len(keep)} carried no discriminative signal); "
            f"every later generation evaluates on the smaller pool"
        )
        logger.info(f"[{self.name}] {note}")
        self.tracker.add_note(note)

    # ------------------------------------------------------------------
    # M2 / M3 — retention and guarded acceptance
    # ------------------------------------------------------------------

    @staticmethod
    def _retention(child: Sequence[int], parent: Sequence[int]) -> float:
        """Fraction of the parent's solved instances the child still solves.

        A parent that solved nothing cannot be regressed against, so it
        returns 1.0 -- the guard should never block a child for failing to
        preserve successes that never existed.
        """
        solved = [i for i, v in enumerate(parent) if v]
        if not solved:
            return 1.0
        kept = sum(1 for i in solved if i < len(child) and child[i])
        return kept / len(solved)

    def _admits(self, child: PromptRecord, parents: Sequence[PromptRecord]) -> bool:
        """M3: reject only genuinely destructive edits.

        A destructive edit is one that loses parent content *without*
        compensating gain, so the gate needs BOTH conditions: the child
        scores no better than its parent AND it dropped below the retention
        floor. That is SPEAR's actual framing -- rollback is triggered by
        metric regression, with the guard metric as a secondary floor.

        Retention alone must not gate. M1 deliberately keeps the instances
        candidates *disagree* on, so retention measured against a calibrated
        pool is structurally low for every candidate; using it as the primary
        gate rejects almost everything and the search stops exploring. (This
        was measured: a retention-only gate rejected 43 of 44 candidates in
        the mocked end-to-end run.)

        Compared against the parent it retains most of, so a crossover is
        judged on the lineage it actually preserved rather than punished for
        not preserving both.
        """
        vecs = [p.performance_vector for p in parents if p.performance_vector]
        if not child.performance_vector or not vecs:
            return True
        best = max(self._retention(child.performance_vector, v) for v in vecs)
        child.metadata["retention"] = round(best, 4)
        parent_best = max(p.score for p in parents)
        if child.score > parent_best:
            return True                      # improved: never a destructive edit
        return best >= self.RETENTION_FLOOR   # no gain, so demand preservation

    # ------------------------------------------------------------------
    # M5 — cost objective
    # ------------------------------------------------------------------

    def _cost_of(self, record: PromptRecord) -> float:
        """Deploy-time cost proxy: prompt length plus induced output length.

        Whitespace tokens, deliberately tokenizer-free (the same proxy the
        FUNNEL barrier uses). It under-counts real BPE tokens but is monotone
        in true length, which is all a selection objective needs. This scores
        the *prompt*, not the search: a deployed prompt pays its own length
        on every future query, which is the cost that actually compounds.
        """
        in_tokens = len(record.text.split()) if record.text else 0
        details = record.per_sample_details or []
        if details:
            out_tokens = sum(
                len(str(d.get("prediction", "")).split()) for d in details
            ) / len(details)
        else:
            out_tokens = 0.0
        return self.COST_INPUT_WEIGHT * in_tokens + self.COST_OUTPUT_WEIGHT * out_tokens

    def _cost_front(self, records: Sequence[PromptRecord]) -> List[PromptRecord]:
        """Non-dominated set on (accuracy up, cost down).

        A record is dominated when another is at least as accurate AND at
        least as cheap, and strictly better on one of the two.
        """
        scored = [(r, r.score, self._cost_of(r)) for r in records if r.text]
        front: List[PromptRecord] = []
        for rec, acc, cost in scored:
            dominated = any(
                (o_acc >= acc and o_cost <= cost) and (o_acc > acc or o_cost < cost)
                for other, o_acc, o_cost in scored
                if other is not rec
            )
            if not dominated:
                front.append(rec)
        return front

    # ------------------------------------------------------------------
    # M4 — bandit + Pareto-widened selection
    # ------------------------------------------------------------------

    def _select_operators(self) -> List[Tuple[str, Any]]:
        """UCB1 over the curated operator pool; unpulled arms come first."""
        total = sum(len(v) for v in self._operator_scores.values())
        if total == 0:
            return [(n, _TECHNIQUES[n]) for n in SCOPE_OPERATORS]
        ranked: List[Tuple[float, str]] = []
        for name in SCOPE_OPERATORS:
            pulls = self._operator_scores.get(name, [])
            if not pulls:
                ranked.append((float("inf"), name))
                continue
            mean = sum(pulls) / len(pulls)
            ranked.append(
                (mean + self.ucb_c * math.sqrt(math.log(max(total, 2)) / len(pulls)), name)
            )
        ranked.sort(key=lambda t: t[0], reverse=True)
        chosen = [n for _, n in ranked[: self.top_m_operators]]
        # Free operators cost nothing, so never let UCB rank them out.
        for name in SCOPE_OPERATORS:
            if name in _FREE_OPERATORS and name not in chosen:
                chosen.append(name)
        return [(n, _TECHNIQUES[n]) for n in chosen]

    def _select_next_population(
        self, candidates: List[PromptRecord]
    ) -> List[PromptRecord]:
        """Elitism + cost-front + per-instance Pareto tournament.

        Three sources, in order of precedence: the best scalar record
        (elitism, never lost); the accuracy-vs-cost front (M5, so a cheaper
        equal-accuracy prompt survives); then tournament draws widened by
        the per-instance frontier (M4), which is the mechanism that
        replicated at +4.93pp on a many-generation loop.
        """
        ranked = sorted(candidates, key=rank_key, reverse=True)
        if not ranked:
            return list(self.population)

        selected: List[PromptRecord] = [ranked[0]]
        chosen_ids = {ranked[0].id}

        for rec in self._cost_front(ranked):
            if len(selected) >= self.population_size:
                break
            if rec.id not in chosen_ids:
                selected.append(rec)
                chosen_ids.add(rec.id)

        coverage = pareto_frontier_coverage(ranked)
        remaining = [r for r in ranked if r.id not in chosen_ids]
        while len(selected) < self.population_size and remaining:
            bout = random.sample(remaining, min(3, len(remaining)))
            if coverage and random.random() < self.frontier_pull_prob:
                pool = [r for r in remaining if r.id in coverage]
                if pool:
                    weights = [coverage[r.id] for r in pool]
                    pulled = random.choices(pool, weights=weights, k=1)[0]
                    if pulled not in bout:
                        bout.append(pulled)
            winner = max(bout, key=rank_key)
            selected.append(winner)
            chosen_ids.add(winner.id)
            remaining.remove(winner)

        selected.sort(key=rank_key, reverse=True)
        return selected

    # ------------------------------------------------------------------
    # Search loop
    # ------------------------------------------------------------------

    def _init_population(self) -> List[PromptRecord]:
        """Generation 0: a deliberately diverse, mostly-free seed set.

        Diversity here is what makes M1 work -- per-instance discrimination
        can only be measured if candidates actually disagree, so the seed
        set spans instruction rewrites, a format contract and a few-shot
        variant rather than near-duplicates of one prompt.
        """
        candidates: List[PromptRecord] = []
        if self.seed_prompt:
            candidates.append(self._create_record(self.seed_prompt, operator="seed"))

        train = self.dataset.get_few_shot_examples(n=5)
        for text in self._lamarckian_generate(train, n=2):
            if text and not self._is_duplicate(text):
                candidates.append(self._create_record(text, operator="lamarckian"))

        base = self.seed_prompt or (candidates[0].text if candidates else "Solve the task.")
        for text in self._semantic_variation(base, n=2):
            if text and not self._is_duplicate(text):
                candidates.append(self._create_record(text, operator="semantic_var"))

        # Free seeds: zero LLM calls, and they widen the disagreement M1 reads.
        self.population = list(candidates)
        for name in ("format_constraint", "few_shot"):
            self._forced_elite = candidates[0] if candidates else None
            try:
                text = _TECHNIQUES[name](self)
            finally:
                self._forced_elite = None
            if text and not self._is_duplicate(text):
                candidates.append(self._create_record(text, operator=name))

        self._evaluate_on_signal_pool(candidates)
        self._calibrate_signal_pool(candidates)
        logger.info(
            f"[{self.name}] gen 0: {len(candidates)} candidates, "
            f"best={max((c.score for c in candidates), default=0.0):.3f}"
        )
        return self._select_next_population(candidates)

    def _step(self) -> List[PromptRecord]:
        """One generation: bandit-selected operators, guarded admission."""
        candidates = list(self.population)
        produced: List[PromptRecord] = []

        for name, fn in self._select_operators():
            draws = self.candidates_per_operator + (1 if name in _FREE_OPERATORS else 0)
            for _ in range(draws):
                parent = self._pick_parent()
                self._forced_elite = parent
                try:
                    text = fn(self)
                except Exception as exc:  # one bad operator must not kill the run
                    logger.warning(f"[{self.name}] operator {name} raised: {exc}")
                    text = None
                finally:
                    self._forced_elite = None
                if not text or self._is_duplicate(text):
                    self._operator_scores.setdefault(name, []).append(0.0)
                    continue
                rec = self._create_record(
                    text, operator=name,
                    parent_ids=[parent.id] if parent else [],
                )
                rec.metadata["_parent_obj_ids"] = [parent.id] if parent else []
                produced.append(rec)

        if not produced:
            logger.info(f"[{self.name}] gen {self.generation}: no new candidates")
            return list(self.population)

        self._evaluate_on_signal_pool(produced)

        baseline = self.best_record.score if self.best_record else 0.0
        admitted: List[PromptRecord] = []
        for rec in produced:
            parents = [
                p for p in self.population if p.id in (rec.parent_ids or [])
            ]
            if self._admits(rec, parents):
                admitted.append(rec)
            else:
                rec.metadata["gate"] = "retention_rejected"
            # Bandit credit: improvement over the incumbent best, so arms are
            # scored against one shared reference rather than each parent's
            # own bar (operators draw parents from different strata).
            self._operator_scores.setdefault(rec.operator, []).append(
                rec.score - baseline
            )

        n_rejected = len(produced) - len(admitted)
        if not admitted and produced:
            # Safety valve: a generation that admits nothing has burned its
            # LLM calls for no candidate at all. Readmit the single best
            # rejected record so the generation still contributes something
            # -- selection downstream can still decline to keep it.
            best_rejected = max(produced, key=rank_key)
            best_rejected.metadata["gate"] = "retention_rejected_readmitted"
            admitted = [best_rejected]
            logger.info(
                f"[{self.name}] gen {self.generation}: all {len(produced)} candidates "
                f"failed the retention guard; readmitting the best one"
            )
        elif n_rejected:
            logger.info(
                f"[{self.name}] gen {self.generation}: M3 rejected {n_rejected}/"
                f"{len(produced)} (no score gain and retention < {self.RETENTION_FLOOR})"
            )
        candidates.extend(admitted)
        return self._select_next_population(candidates)

    def _pick_parent(self) -> Optional[PromptRecord]:
        if not self.population:
            return None
        return random.choice(self.population[: max(1, min(3, len(self.population)))])
