"""PACT — Prompt Advancement via Contracted Transformations.

Twenty-one experiments in this project optimized *search* — Pareto frontiers,
bandits, racing, gating. None validated what was being searched over. The
codebase's entire check on an optimizer response is
`result.strip() if result.strip() else None`, so a preamble, a leaked Step-1
diagnosis, or an outright refusal becomes a candidate prompt and gets
evaluated. Measured on this project's own audit records: **7.9% of stored
candidates are unusable as instructions** (and that is a lower bound — 68.7%
of records are text-stripped, and the stripped ones are the lower-scoring
ones). Separately, **84% of whole-prompt rewrites destroyed their parent's
demonstrations**.

PACT makes every mutating call a verified contract:

1. **Constrained emission** — the model returns JSON whose `analysis` field
   is free-form and emitted first, followed by at most `MAX_EDITS` anchored
   edits. Reasoning therefore completes before any constraint binds, which
   is the documented resolution to "structured output hurts reasoning": the
   damage comes from forcing an answer field early, not from constraint
   itself (arXiv:2501.10868 measures real constraint engines as
   accuracy-neutral-to-positive and *faster* than unconstrained decoding).
2. **Anchored edits, not rewrites** — each edit names a verbatim span of the
   parent. Whole-prompt destruction becomes impossible by construction, and
   the demonstration block is explicitly protected.
3. **Programmatic verification** — anchors, protected regions, growth bound
   and no-op detection are checked in Python, because a grammar cannot know
   what spans exist in the parent.
4. **One retry naming the violation**, then discard. A rejected candidate is
   never evaluated, so it costs no eval budget.

The meta-prompt carries an explicit step-by-step reasoning template and
**no persona**: PE2 (arXiv:2311.05661) ablated both, finding the reasoning
template worth -5/-7 points when removed while role framing was +1/-5 and
inconsistent. This is a deliberate departure from APEX's expert personas.

Search is deliberately minimal — GEPA-Pareto tournament selection plus
held-out final selection, both already validated in this project, and
nothing else. SCOPE (EXP-020) stacked five unvalidated mechanisms and scored
*exactly* baseline; the restraint here is a direct response to that.

**Honest scope**: 7.9% waste cannot by itself explain a ~5pp method gap, so
the claim is reliability and wasted-call elimination, with a novel
contract-validity metric — not an accuracy win. Accuracy is reported as
secondary and expected to be flat.
"""
from __future__ import annotations

import logging
import random
from collections import Counter
from typing import Any, Dict, List, Optional

from pof.core.types import GenerationConfig, PromptRecord, pareto_frontier_coverage, rank_key
from pof.llm.structured import make_decoder
from pof.optimizers import register_optimizer
from pof.optimizers.base import BaseOptimizer, format_exemplar
from pof.optimizers.holdout import HoldoutSelectionMixin
from pof.optimizers._pact_contract import (
    CONTRACT_SCHEMA,
    MAX_EDITS,
    apply_contract,
    build_meta_prompt,
    build_retry_prompt,
)

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You revise task instructions. You return only a JSON object matching the "
    "requested schema."
)


@register_optimizer("pact")
class PACTOptimizer(HoldoutSelectionMixin, BaseOptimizer):
    """Contract-verified prompt optimizer."""

    name = "pact"

    def __init__(
        self,
        llm,
        dataset,
        evaluator,
        population_size: int = 5,
        num_iterations: int = 5,
        use_holdout_selection: bool = True,
        frontier_pull_prob: float = 0.5,
        max_edits: int = MAX_EDITS,
        constrained_decoding: bool = True,
        allow_retry: bool = True,
        **kwargs,
    ):
        super().__init__(
            llm=llm, dataset=dataset, evaluator=evaluator,
            population_size=population_size, num_iterations=num_iterations,
            **kwargs,
        )
        self.frontier_pull_prob = frontier_pull_prob
        self.max_edits = max_edits
        self.allow_retry = allow_retry
        self._init_holdout(use_holdout_selection=use_holdout_selection)
        self._decoder = make_decoder(llm, CONTRACT_SCHEMA, enabled=constrained_decoding)
        # The measurement this method exists to produce.
        self.contract_stats: Counter = Counter()
        logger.info(
            f"[{self.name}] contract enforcement: {self._decoder.name} "
            f"(max_edits={self.max_edits}, retry={'on' if allow_retry else 'off'})"
        )

    # ------------------------------------------------------------------
    # The contract call
    # ------------------------------------------------------------------

    def _generate_constrained(self, meta_prompt: str) -> str:
        """One generation with structural enforcement, if available.

        The logits processor is attached for exactly this call and cleared
        afterwards: it carries parser position state, so leaking it into a
        later call would leave that call mid-parse.
        """
        procs = self._decoder.logits_processors()
        if procs:
            setattr(self.llm, "_logits_processors", procs)
        try:
            return self._generate_prompt(
                meta_prompt,
                temperature=0.7,
                max_new_tokens=768,
                system_prompt=_SYSTEM,
            )
        finally:
            if procs:
                setattr(self.llm, "_logits_processors", None)

    def _contract_edit(self, parent: PromptRecord) -> Optional[PromptRecord]:
        """Ask for anchored edits to `parent`, verify, and apply.

        Returns a new record, or None when the contract could not be
        satisfied — in which case nothing is evaluated, so a bad call costs
        one generation and zero eval budget.
        """
        failures = [d for d in (parent.per_sample_details or []) if not d.get("correct")]
        meta = build_meta_prompt(parent.text, failures, max_edits=self.max_edits)

        raw = self._generate_constrained(meta)
        result = apply_contract(parent.text, raw, max_edits=self.max_edits)
        self.contract_stats["attempts"] += 1

        if not result.ok and self.allow_retry:
            self.contract_stats[f"reject_{result.violation}"] += 1
            self.contract_stats["retries"] += 1
            raw = self._generate_constrained(build_retry_prompt(meta, result.detail))
            result = apply_contract(parent.text, raw, max_edits=self.max_edits)

        if not result.ok:
            self.contract_stats["failed"] += 1
            self.contract_stats[f"final_{result.violation}"] += 1
            logger.debug(f"[{self.name}] contract failed: {result.violation}")
            return None

        self.contract_stats["ok"] += 1
        self.contract_stats["edits"] += result.edits_applied
        if self._is_duplicate(result.text):
            self.contract_stats["duplicate"] += 1
            return None
        rec = self._create_record(
            result.text, operator="contract_edit", parent_ids=[parent.id],
            edits_applied=result.edits_applied,
        )
        rec.metadata["analysis"] = result.analysis[:400]
        return rec

    def validity_report(self) -> Dict[str, Any]:
        """Contract compliance — the method's primary reported metric."""
        s = self.contract_stats
        attempts = s.get("attempts", 0)
        return {
            "enforcement": self._decoder.name,
            "attempts": attempts,
            "satisfied": s.get("ok", 0),
            "failed": s.get("failed", 0),
            "retries": s.get("retries", 0),
            "duplicates": s.get("duplicate", 0),
            "validity_rate": (s.get("ok", 0) / attempts) if attempts else 0.0,
            "retry_rate": (s.get("retries", 0) / attempts) if attempts else 0.0,
            "mean_edits": (s.get("edits", 0) / max(s.get("ok", 0), 1)),
            "violations": {k[7:]: v for k, v in s.items() if k.startswith("reject_")},
        }

    # ------------------------------------------------------------------
    # Search — minimal, reusing only validated mechanisms
    # ------------------------------------------------------------------

    def _select_next_population(self, candidates: List[PromptRecord]) -> List[PromptRecord]:
        """Elitism + GEPA-Pareto-widened tournament.

        The only selection mechanism in this project that cleared the noise
        floor with replication (+4.93pp, 9/9 pairwise seed dominance in
        APEX). Nothing is layered on top of it -- that was SCOPE's mistake.
        """
        ranked = sorted(candidates, key=rank_key, reverse=True)
        if not ranked:
            return list(self.population)
        selected = [ranked[0]]
        coverage = pareto_frontier_coverage(ranked)
        remaining = ranked[1:]
        while len(selected) < self.population_size and remaining:
            bout = random.sample(remaining, min(3, len(remaining)))
            if coverage and random.random() < self.frontier_pull_prob:
                pool = [r for r in remaining if r.id in coverage]
                if pool:
                    pulled = random.choices(
                        pool, weights=[coverage[r.id] for r in pool], k=1)[0]
                    if pulled not in bout:
                        bout.append(pulled)
            winner = max(bout, key=rank_key)
            selected.append(winner)
            remaining.remove(winner)
        selected.sort(key=rank_key, reverse=True)
        return selected

    def _init_population(self) -> List[PromptRecord]:
        """Seed plus zero-LLM-call variants.

        Initialization is exempt from the contract: there is no parent to
        anchor edits against. It uses the free generators, so the population
        starts diverse at almost no cost and every subsequent change goes
        through the contract.
        """
        candidates: List[PromptRecord] = []
        if self.seed_prompt:
            candidates.append(self._create_record(self.seed_prompt, operator="seed"))

        train = self.dataset.get_few_shot_examples(n=5)
        for text in self._lamarckian_generate(train, n=2):
            if text and not self._is_duplicate(text):
                candidates.append(self._create_record(text, operator="lamarckian"))

        base = self.seed_prompt or (candidates[0].text if candidates else "Solve the task.")
        if train:
            shots = "\n\n".join(format_exemplar(self.evaluator, s) for s in train[:2])
            fs = f"{base}\n\nExamples:\n{shots}"
            if not self._is_duplicate(fs):
                candidates.append(self._create_record(fs, operator="few_shot", num_few_shots=2))
            targets = ", ".join(repr(s["target"])[:30] for s in train[:3])
            fmt = (f"{base.rstrip()}\n\nAnswer with ONLY the final answer, "
                   f"exactly in the same format as these examples: {targets}. "
                   f"No explanation.")
            if not self._is_duplicate(fmt):
                candidates.append(self._create_record(fmt, operator="format_constraint"))

        self._evaluate_population(candidates)
        logger.info(f"[{self.name}] gen 0: {len(candidates)} candidates")
        return self._select_next_population(candidates)

    def _step(self) -> List[PromptRecord]:
        """One generation: a contract edit per elite."""
        candidates = list(self.population)
        produced: List[PromptRecord] = []
        for parent in self.population[: self.population_size]:
            if not parent.text:
                continue
            rec = self._contract_edit(parent)
            if rec is not None:
                produced.append(rec)

        if not produced:
            logger.info(
                f"[{self.name}] gen {self.generation}: no candidate satisfied "
                f"the contract"
            )
            return list(self.population)

        self._evaluate_population(produced)
        candidates.extend(produced)
        rep = self.validity_report()
        logger.info(
            f"[{self.name}] gen {self.generation}: {len(produced)} accepted, "
            f"validity={rep['validity_rate']:.2f} retry={rep['retry_rate']:.2f}"
        )
        return self._select_next_population(candidates)

    def _finalize(self) -> None:
        if not self._finalized:
            rep = self.validity_report()
            note = (
                f"contract[{rep['enforcement']}]: {rep['satisfied']}/{rep['attempts']} "
                f"satisfied ({rep['validity_rate']:.1%}), {rep['retries']} retries, "
                f"mean {rep['mean_edits']:.1f} edits, violations={rep['violations']}"
            )
            logger.info(f"[{self.name}] {note}")
            try:
                self.tracker.add_note(note)
            except Exception:
                pass
        super()._finalize()
