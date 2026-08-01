"""APEX-Lean — APEXv2 plus the one guaranteed cheap mechanism proven to help.

`funnel_prime`'s `causal_judgement` results (mean test=0.586 across 3 seeds,
below both the AO baseline ~0.649 and SWIFT/APEX's own ~0.62-0.65) confirm
what the FUNNEL lineage's own measurements already showed piecemeal: most of
FUNNEL's guaranteed-operator machinery (multi-aspect critique, trajectory
momentum, demonstration-split family, mode family) does not clearly pay for
its complexity relative to a plain adaptive search. Only one FUNNEL mechanism
has an unambiguous, cheap, always-safe win: the static-baseline floor
(`funnel_v4b.py`) — seed the population with the canonical `json_3shot`
baseline so the search can never finish worse than a prompt we already know
is competitive. It costs exactly one extra eval per run, no LLM calls to
construct.

Base class is `APEXv2Optimizer`, not plain APEX: v2 already carries two
measured, cheap wins over v1 (`format_constraint` applied for free instead of
competing for bandit slots; variance-aware UCB1 exploration bonus) from a
177-run bandit audit -- no reason to leave that on the table.

APEX-Lean adds nothing else: no guaranteed critique/momentum operators
(unmeasured net-positive, folded into FUNNEL-Prime's underperformance), no
demonstration-split family (FUNNEL-specific, not part of APEX's operator
set), no mode family (adaptive per-candidate eval-mode selection was
FUNNEL-specific machinery; APEX/SWIFT already run under whichever mode the
harness sets globally).

Also inherits `BaseOptimizer._maybe_stop_if_perfect()` unchanged (already
wired into APEX's `_step()`), so a lucky early population still exits
immediately rather than paying for full-budget confirmation.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from pof.core.types import PromptRecord
from pof.optimizers import register_optimizer
from pof.optimizers.apex_v2 import APEXv2Optimizer
from pof.prompts.loader import fetch_bbh_prompt

logger = logging.getLogger(__name__)


def _build_json_3shot(prompt_text: str) -> str:
    """Reconstruct the canonical answer-only baseline in our JSON answer format.

    Duplicated from `funnel_v4b._build_json_3shot` rather than imported --
    that module pulls in the whole FUNNELv3 chain as a side effect of import,
    which this class deliberately does not depend on.
    """
    import re

    answer_re = re.compile(r"So the answer is\s*(.+?)\.\s*$", re.MULTILINE)
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
        m = answer_re.search(answer_part)
        if not m:
            continue
        rebuilt.append(f"{question.strip()}\nA: {m.group(1).strip()}")

    ao = instruction if not rebuilt else instruction + "\n\n" + "\n\n".join(rebuilt)

    out = []
    for line in ao.split("\n"):
        if line.startswith("A: "):
            answer = line[3:].strip()
            out.append('A: {"answer": "%s"}' % answer.replace('"', '\\"'))
        else:
            out.append(line)
    return "\n".join(out)


@register_optimizer("apex_lean")
class APEXLeanOptimizer(APEXv2Optimizer):
    """APEX plus a guaranteed static-baseline floor candidate."""

    name = "apex_lean"

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
        candidates = super()._init_population()
        baseline_text = self._static_baseline_prompt()
        if baseline_text and not self._is_duplicate(baseline_text):
            record = self._create_record(baseline_text, operator="static_json3shot_baseline")
            self._evaluate_population([record])
            candidates.append(record)
            candidates = self._select_top_k(candidates)
        return candidates


register_optimizer("swift_apex_lean")(APEXLeanOptimizer)
