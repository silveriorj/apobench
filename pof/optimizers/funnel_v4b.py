"""FUNNELv4b — FUNNELv3 with a guaranteed static-baseline floor.

Diagnosis motivating this (Qwen3-4B, 8 BBH tasks, 3 seeds, all four earlier
fixes applied — accumulating-fresh validation, fixed test split, exemplar
format, held-out selection): v3 wins big on hyperbaton (+17pts) and
disambiguation_qa (+5.5pts), but loses SIGNIFICANTLY to the plain static
`json_3shot` baseline on formal_fallacies (t=-4.9), and sits numerically BELOW
even `instruction_only` on logical_deduction_five_objects (0.557 vs. 0.591) —
worse than doing nothing. v3's guaranteed decomposition/enrichment family
apparently disrupts prompt structure on tasks that reward a plain,
well-formatted few-shot prompt more than reformulation.

**The fix.** Seed the initial population with the exact static `json_3shot`
baseline (canonical BBH instruction + the official 3 exemplars, answers
rewritten in our JSON format — see `experiments/bbh_reference_baseline.py`,
which this borrows its construction from). It competes in selection like any
other candidate: equal-N tournament selection, held-out final ranking. This
does not fix WHY v3's machinery underperforms on these tasks (that is v4a's
job); it only guarantees the search can never finish worse than a baseline we
already know beats it on some tasks. Cheap: it costs one eval per phase for
one candidate, no LLM calls to construct it.

Network fetch (`fetch_bbh_prompt`) is cached to `.cache/prompts/bbh/`, so this
does not add meaningfully to run time — the same file is already used to
build the seed prompt for every method in the benchmark.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

from pof.core.types import PromptRecord
from pof.optimizers import register_optimizer
from pof.optimizers.funnel_v2 import _ALL_TECHNIQUE_FNS
from pof.optimizers.funnel_v3 import FUNNELv3Optimizer
from pof.prompts.loader import extract_instruction, fetch_bbh_prompt

logger = logging.getLogger(__name__)

_ANSWER_RE = re.compile(r"So the answer is\s*(.+?)\.\s*$", re.MULTILINE)


def _build_answer_only(prompt_text: str) -> str:
    """Reconstruct Suzgun's answer-only exemplars from the official CoT file.

    Mirrors `experiments/bbh_reference_baseline.build_answer_only` exactly —
    duplicated rather than imported since that module is a standalone script,
    not a package the optimizer should depend on.
    """
    text = prompt_text.strip()
    head, _, body = text.partition("\n\n")
    instruction = head.strip()

    blocks = [b for b in body.split("\n\nQ:") if b.strip()]
    rebuilt: List[str] = []
    for i, block in enumerate(blocks):
        block = block if i == 0 else "Q:" + block
        if not block.lstrip().startswith("Q:"):
            block = "Q: " + block.lstrip()
        question, sep, answer_part = block.partition("\nA:")
        if not sep:
            continue
        m = _ANSWER_RE.search(answer_part)
        if not m:
            continue
        rebuilt.append(f"{question.strip()}\nA: {m.group(1).strip()}")

    if not rebuilt:
        return instruction
    return instruction + "\n\n" + "\n\n".join(rebuilt)


def _build_json_3shot(prompt_text: str) -> str:
    """Answer-only exemplars rewritten in our evaluator's JSON answer format."""
    ao = _build_answer_only(prompt_text)
    out = []
    for line in ao.split("\n"):
        if line.startswith("A: "):
            answer = line[3:].strip()
            out.append('A: {"answer": "%s"}' % answer.replace('"', '\\"'))
        else:
            out.append(line)
    return "\n".join(out)


@register_optimizer("funnel_v4b")
class FUNNELv4bOptimizer(FUNNELv3Optimizer):
    """FUNNELv3 with a guaranteed static json_3shot baseline candidate."""

    name = "funnel_v4b"

    def _static_baseline_prompt(self) -> Optional[str]:
        task = (self.dataset.metadata or {}).get("task")
        if not task or (self.dataset.metadata or {}).get("source") != "bigbenchhard":
            return None
        try:
            raw = fetch_bbh_prompt(task)
            return _build_json_3shot(raw)
        except Exception as e:
            logger.warning(f"[{self.name}] could not build static baseline for {task}: {e!r}")
            return None

    def _init_population(self) -> List[PromptRecord]:
        """Same as FUNNELv2's Phase 0, plus one guaranteed static-baseline seed."""
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

        for name in ["format_constraint", "local_edit", "semantic_var", "few_shot"]:
            if name not in self._active_techniques:
                continue
            fn = _ALL_TECHNIQUE_FNS[name]
            text = fn(self)
            if text and not self._is_duplicate(text):
                candidates.append(self._create_record(text, operator=name))
                self.population = sorted(candidates, key=lambda r: r.score, reverse=True)

        baseline_text = self._static_baseline_prompt()
        if baseline_text and not self._is_duplicate(baseline_text):
            candidates.append(self._create_record(baseline_text, operator="static_json3shot_baseline"))

        self._evaluate_phase(candidates, phase=0)
        return self._select_top_k(candidates)
