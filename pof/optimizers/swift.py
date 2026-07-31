"""SWIFT — Sequential Weighted Improvement with Failure-guided Tuning and Racing.

A proposed method designed to match SEE's budget (~34 LLM calls) while
outperforming it through:

1. Failure-guided improvement with ProTeGi-style expansion: each textual
   gradient also spawns a paraphrase, exploring around the fix at zero
   extra diagnosis cost (Pryzant et al. 2023)
2. Trajectory-augmented generation (OPRO-style): FULL instructions plus
   task exemplars in the meta-prompt (Yang et al. 2023)
3. Semantic crossover of top performers
4. Polish phase: PLUM local edits + few-shot exemplar augmentation
   (joint instruction+ICL search, cf. CAPO/SEE) + full eval
5. GEPA-style minibatch gate (Agrawal et al. 2025): candidates are screened
   on a FRESH random dev minibatch each phase, and only survivors get the
   full dev evaluation — selection never overfits one fixed subset
6. Hash dedup so no candidate is evaluated twice

Budget breakdown (population_size K=5):
  Phase 0 Init:       K+2 diverse seeds → Full eval → top K
  Phase 1 Failure:    K improvements + K paraphrases → Gate → top K
  Phase 2 Trajectory: 2 OPRO-style + 3 crossovers → Gate → top K
  Phase 3 Polish:     K local edits + 2 few-shot variants → Full eval → final
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pof.core.types import GenerationConfig, PromptRecord
from pof.optimizers import register_optimizer
from pof.optimizers.base import format_exemplar, BaseOptimizer, _GENERATE_SYSTEM_PROMPT, _IMPROVE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


@register_optimizer("swift")
class SWIFTOptimizer(BaseOptimizer):
    """SWIFT optimizer — failure-guided improvement with racing.

    Proposed method: needs validation.
    """

    name = "swift"

    def __init__(self, llm, dataset, evaluator, population_size: int = 8, **kwargs):
        super().__init__(
            llm=llm,
            dataset=dataset,
            evaluator=evaluator,
            population_size=population_size,
            num_iterations=3,
            **kwargs,
        )
        self._phase_idx = 0

    def _init_population(self) -> List[PromptRecord]:
        """Phase 0: Diverse seed generation.

        Generates seeds via multiple strategies:
        - Lamarckian reverse-engineering (×2)
        - Zero-order paraphrasing (×2)
        - Local edit variant (×1)
        - Few-shot variant (×1)
        - Original seed
        """
        candidates: List[PromptRecord] = []
        train_samples = self.dataset.get_few_shot_examples(n=5)

        logger.debug("[Init] Lamarckian generation ×2 ...")
        lamarckian = self._lamarckian_generate(train_samples, n=2)
        for text in lamarckian:
            candidates.append(self._create_record(text, operator="lamarckian_init"))

        base = self.seed_prompt or (candidates[0].text if candidates else "Solve the task.")
        logger.debug("[Init] Semantic variation ×2 ...")
        variations = self._semantic_variation(base, n=2)
        for text in variations:
            candidates.append(self._create_record(text, operator="zero_order_init"))

        if self.seed_prompt:
            local_edit = self._local_edit(self.seed_prompt)
            if local_edit:
                candidates.append(self._create_record(local_edit, operator="local_edit_init"))

        logger.debug("[Init] Few-shot variant ...")
        few_shot_prompt = self.dataset.format_few_shot_prompt(
            self.seed_prompt or "Solve the following task:", n_examples=3
        )
        candidates.append(self._create_record(few_shot_prompt, operator="few_shot_init"))

        # Format-constraint variant: explicit output-format rule from real
        # targets — strict extraction penalizes format drift more than
        # reasoning errors on small models (Sclar et al. 2023).
        base_text = self.seed_prompt or (candidates[0].text if candidates else "")
        if base_text:
            targets = [s["target"] for s in train_samples[:3]]
            sample_answers = ", ".join(repr(t)[:30] for t in targets)
            fmt_text = (
                f"{base_text.rstrip()}\n\nAnswer with ONLY the final answer, "
                f"exactly in the same format as these examples: {sample_answers}. "
                f"No explanation."
            )
            candidates.append(self._create_record(fmt_text, operator="format_constraint_init"))

        if self.seed_prompt:
            candidates.append(self._create_record(self.seed_prompt, operator="seed"))

        logger.info(f"[Init] {len(candidates)} candidates generated — starting evaluation ...")
        self._evaluate_population(candidates)
        top = self._select_top_k(candidates)
        logger.info(
            f"[Init] top-{len(top)} selected:"
            + "".join(f"\n  [{r.score:.3f}] op={r.operator} {r.text[:60]!r}..." for r in top)
        )
        return top

    def _step(self) -> List[PromptRecord]:
        """Execute SWIFT phases 1-3."""
        self._maybe_stop_if_perfect()
        self._phase_idx += 1
        if self._phase_idx == 1:
            return self._phase_failure_guided()
        elif self._phase_idx == 2:
            return self._phase_trajectory_crossover()
        elif self._phase_idx == 3:
            result = self._phase_polish()
            raise StopIteration
        else:
            raise StopIteration

    def _phase_failure_guided(self) -> List[PromptRecord]:
        """Phase 1: Structured failure-guided improvement.

        Uses cached per_sample_details from init evaluation — no re-evaluation.
        """
        logger.info("[SWIFT Phase 1] Failure-guided improvement")
        candidates = list(self.population)

        for record in self.population[:self.population_size]:
            # Reuse details stored during _evaluate_population; only re-evaluate
            # if the record was somehow created without them (edge case).
            details = record.per_sample_details
            if not details and record.text:
                samples = self.dataset.get_eval_samples("dev", n=self.eval_sample_size)
                result = self.evaluator.evaluate(record.text, samples)
                details = result.per_sample_details
                record.per_sample_details = details
                record.score = result.score
                record.performance_vector = result.performance_vector

            failures = [d for d in details if not d["correct"]]
            if failures:
                improved = self._structured_improve(record.text, failures)
                if improved and not self._is_duplicate(improved):
                    candidates.append(self._create_record(
                        improved, operator="failure_guided", parent_ids=[record.id]
                    ))
                    # ProTeGi-style expansion: a paraphrase of each improvement
                    # explores around the textual gradient at zero extra
                    # diagnosis cost (Pryzant et al. 2023).
                    variants = self._semantic_variation(improved, n=1)
                    if variants and not self._is_duplicate(variants[0]):
                        candidates.append(self._create_record(
                            variants[0], operator="failure_guided_var",
                            parent_ids=[record.id],
                        ))

        # Few-shot augmentation of the current top-2: the strongest known
        # lever (CAPO/SEE) competes from Phase 1, not only in the final polish.
        import random as _random
        train = self.dataset.get_few_shot_examples(n=6)
        for record in self.population[:2]:
            if not train:
                break
            k = _random.randint(1, min(3, len(train)))
            shots = _random.sample(train, k)
            examples = "\n\n".join(
                format_exemplar(self.evaluator, s) for s in shots
            )
            fs_text = f"{record.text}\n\nExamples:\n{examples}"
            if not self._is_duplicate(fs_text):
                candidates.append(self._create_record(
                    fs_text, operator="few_shot_phase1", parent_ids=[record.id],
                    num_few_shots=k,
                ))

        new_candidates = [c for c in candidates if c.score == 0.0]
        baseline = self.best_record.score if self.best_record else 0.0
        self._evaluate_with_minibatch_gate(new_candidates, baseline)
        return self._select_top_k(candidates)

    def _phase_trajectory_crossover(self) -> List[PromptRecord]:
        """Phase 2: OPRO-style trajectory + crossover.

        - 2 trajectory climbs using score history as context
        - 3 pairwise LLM crossovers of elite prompts
        """
        logger.info("[SWIFT Phase 2] Trajectory + Crossover")
        candidates = list(self.population)

        # Trajectory climbs (OPRO-style)
        trajectory_context = self._build_trajectory_context()
        for _ in range(2):
            new_text = self._trajectory_generate(trajectory_context)
            if new_text and not self._is_duplicate(new_text):
                candidates.append(self._create_record(
                    new_text, operator="trajectory_climb",
                    parent_ids=[r.id for r in self.population[:3]]
                ))

        # Pairwise crossovers
        for i in range(min(3, len(self.population) - 1)):
            a, b = self.population[i], self.population[i + 1]
            cross = self._crossover(a.text, b.text)
            if cross.strip() and not self._is_duplicate(cross):
                candidates.append(self._create_record(
                    cross, operator="crossover", parent_ids=[a.id, b.id]
                ))

        new_candidates = [c for c in candidates if c.score == 0.0]
        baseline = self.best_record.score if self.best_record else 0.0
        self._evaluate_with_minibatch_gate(new_candidates, baseline)
        return self._select_top_k(candidates)

    def _phase_polish(self) -> List[PromptRecord]:
        """Phase 3: Local edit polish + full evaluation."""
        logger.info("[SWIFT Phase 3] Polish (local edits + few-shot + full eval)")
        candidates = list(self.population)
        import random as _random

        for record in self.population[:self.population_size]:
            # Local edit
            edited = self._local_edit(record.text)
            if edited and not self._is_duplicate(edited):
                candidates.append(self._create_record(
                    edited, operator="local_edit_polish", parent_ids=[record.id]
                ))

        # Few-shot augmentation of the top-2 (joint instruction+ICL search,
        # cf. CAPO/SEE — exemplars are the strongest single lever on BBH).
        train = self.dataset.get_few_shot_examples(n=6)
        for record in self.population[:2]:
            if not train:
                break
            k = _random.randint(1, min(3, len(train)))
            shots = _random.sample(train, k)
            examples = "\n\n".join(
                format_exemplar(self.evaluator, s) for s in shots
            )
            fs_text = f"{record.text}\n\nExamples:\n{examples}"
            if not self._is_duplicate(fs_text):
                candidates.append(self._create_record(
                    fs_text, operator="few_shot_polish", parent_ids=[record.id],
                    num_few_shots=k,
                ))

        # Full evaluation (no racing) for precise final selection
        new_candidates = [c for c in candidates if c.score == 0.0]
        self._evaluate_population(new_candidates)
        return self._select_top_k(candidates)

    # --- SWIFT-specific techniques ---

    def _structured_improve(self, prompt: str, failures: List[Dict]) -> Optional[str]:
        """Diagnose failures then engineer an improved prompt."""
        failure_text = "\n".join(
            f"- Input: {f.get('input', '')[:60]} | Expected: {f.get('target', '')} | Got: {f.get('prediction', '')[:60]}"
            for f in failures[:5]
        )

        meta_prompt = (
            "You are an expert prompt engineer. Analyze why this instruction fails "
            "on certain inputs, then write an improved version.\n\n"
            f"Current instruction:\n{prompt}\n\n"
            f"Failure cases:\n{failure_text}\n\n"
            "Step 1: Diagnose the root cause of failures.\n"
            "Step 2: Write an improved instruction that addresses these issues.\n\n"
            "Improved instruction:"
        )
        result = self._generate_prompt(meta_prompt, temperature=0.7, system_prompt=_IMPROVE_SYSTEM_PROMPT)
        return result.strip() if result.strip() else None

    def _local_edit(self, prompt: str) -> Optional[str]:
        """PLUM-style local edit: small targeted modification."""
        meta_prompt = (
            "Make a small, targeted improvement to this instruction. "
            "Change only 1-2 sentences to make it clearer or more precise. "
            "Do NOT rewrite the entire instruction.\n\n"
            f"Instruction:\n{prompt}\n\n"
            "Slightly improved instruction:"
        )
        result = self._generate_prompt(meta_prompt, temperature=0.5, system_prompt=_GENERATE_SYSTEM_PROMPT)
        return result.strip() if result.strip() else None

    def _build_trajectory_context(self) -> str:
        """Build OPRO-style trajectory context: FULL instructions, not
        fragments (Yang et al. 2023 keep up to 20 complete instructions)."""
        entries = []
        for record in sorted(self.population, key=lambda r: r.score):
            entries.append(f"Score: {record.score:.3f}\nInstruction: {record.text}\n")
        return "\n".join(entries)

    def _trajectory_generate(self, context: str) -> Optional[str]:
        """Generate a new prompt using trajectory context (OPRO-style).

        Includes task exemplars in the meta-prompt, per OPRO's ablations."""
        exemplars = self.dataset.get_few_shot_examples(n=2)
        exemplar_text = "\n".join(
            format_exemplar(self.evaluator, s, max_input=150) for s in exemplars
        )
        meta_prompt = (
            "Below are instructions for a task, sorted by performance score "
            "(ascending). Task examples:\n"
            f"{exemplar_text}\n\n"
            f"Instructions and scores:\n{context}\n"
            "Write a NEW instruction that would score higher than all of the above.\n\n"
            "New higher-scoring instruction:"
        )
        result = self._generate_prompt(meta_prompt, temperature=0.8, system_prompt=_GENERATE_SYSTEM_PROMPT)
        return result.strip() if result.strip() else None