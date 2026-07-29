"""FUNNELv5 — FUNNELv4d ("FUNNEL-Lean") with eval mode as a searchable
candidate property: answer-only, brief CoT, or full CoT/thinking.

**Motivation.** Every prior FUNNEL variant fixes the eval mode (system prompt
+ token budget) for the whole run, chosen up front by the experimenter. But
measured results show the best mode is TASK-dependent, not something you can
guess correctly in advance: on `boolean_expressions`, full CoT reaches
near-perfect accuracy (0.997) versus the best answer-only result all session
(0.887); on `causal_judgement`, full CoT is WORSE than answer-only (0.583 vs.
0.649) -- and that reversal independently reproduces a finding already in
Suzgun et al. (2023) Table 3, where CoT measurably hurts causal_judgement for
much larger models too. Picking one mode per run means guessing which
tasks look like the first case and which look like the second.

**The mechanism.** Eval mode becomes a first-class, heritable property of a
`PromptRecord` (`metadata["mode"]` — one of "ao", "cot", "thinking"), not a
run-level setting:

- New candidates inherit their parent's mode by default (`_create_record`).
- A guaranteed "mode family" mints, for each top elite each phase, sibling
  candidates in whichever of the other two modes it doesn't already have a
  sibling in — same instruction text, different eval configuration. Zero LLM
  calls, same principle as v3's few-shot split (only the identical
  demonstration/instruction split, generalized to eval mode).
- Selection (equal-N tournament + the held-out final check, both inherited
  unchanged) then does the actual work: whichever mode's siblings score
  higher on THIS task survive and keep propagating. A task that answers this
  quickly stops paying for the losing mode's LLM cost -- CoT/thinking siblings
  of a clearly-losing lineage just don't get selected forward, so the
  population converges toward whichever mode is winning without an explicit
  gate (the same self-pruning logic v4a's adaptive gate applies explicitly,
  happening here as an emergent property of ordinary tournament selection).

**Why bypass the normal duplicate check for mode siblings.** `_is_duplicate`
hashes only text, so a same-text sibling in a new mode would be dropped as a
"duplicate" of the elite it was minted from -- exactly the case this family
exists to create. Mode siblings are tracked and deduplicated on
(text_hash, mode) instead, via `_mode_pairs_seen`.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from pof.core.types import PromptRecord
from pof.evaluation.evaluator import SYSTEM_PROMPT_BY_TASK_TYPE
from pof.optimizers import register_optimizer
from pof.optimizers.funnel_v4d import FUNNELv4dOptimizer

logger = logging.getLogger(__name__)

# Duplicated rather than imported from experiments/run_swift_apex.py (that
# module is a script, not something pof/ should depend on) -- same values as
# COT_BRIEF_MAX_NEW_TOKENS / COT_MAX_NEW_TOKENS there.
MODE_CONFIGS: Dict[str, Tuple[Optional[str], Optional[int]]] = {
    "ao": (None, None),  # None = defer to the evaluator's own per-run defaults
    "cot": (SYSTEM_PROMPT_BY_TASK_TYPE["cot"], 256),
    "thinking": (SYSTEM_PROMPT_BY_TASK_TYPE["thinking"], 1536),
}
MODE_BY_OPERATOR = {f"mode_{m}": m for m in MODE_CONFIGS}


@register_optimizer("funnel_v5")
class FUNNELv5Optimizer(FUNNELv4dOptimizer):
    """FUNNELv4d ("FUNNEL-Lean") with a guaranteed AO/CoT/thinking mode family."""

    name = "funnel_v5"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mode_pairs_seen: set = set()

    def _eval_overrides(self, record: PromptRecord):
        mode = record.metadata.get("mode", "ao")
        return MODE_CONFIGS.get(mode, (None, None))

    def _create_record(self, text, operator, parent_ids=None, **metadata):
        if "mode" not in metadata:
            parent = None
            if parent_ids:
                parent = self.tracker.history.records.get(parent_ids[0])
            if operator in MODE_BY_OPERATOR:
                metadata["mode"] = MODE_BY_OPERATOR[operator]
            elif parent is not None:
                metadata["mode"] = parent.metadata.get("mode", "ao")
            else:
                metadata["mode"] = "ao"
        record = super()._create_record(text, operator, parent_ids=parent_ids, **metadata)
        self._mode_pairs_seen.add((record.text_hash, record.metadata["mode"]))
        return record

    def _apply_static_core(self, candidates: List[PromptRecord]) -> int:
        made = super()._apply_static_core(candidates)
        made += self._apply_mode_family(candidates)
        return made

    def _apply_mode_family(self, candidates: List[PromptRecord]) -> int:
        """For each top elite, mint siblings in modes it doesn't already have."""
        made = 0
        elites = self.population[: self.static_top_k] if self.population else []
        for elite in elites:
            if not elite.text:
                continue
            current_mode = elite.metadata.get("mode", "ao")
            for mode in MODE_CONFIGS:
                if mode == current_mode:
                    continue
                key = (elite.text_hash, mode)
                if key in self._mode_pairs_seen:
                    continue
                record = self._create_record(
                    elite.text, operator=f"mode_{mode}", parent_ids=[elite.id], mode=mode,
                )
                candidates.append(record)
                made += 1
        if made:
            logger.info(
                f"[{self.name} Phase {self._phase_idx}] mode family produced "
                f"{made} sibling(s) across {len(elites)} elite(s)"
            )
        return made


# Alias: paper-facing name, same class. Launch WITHOUT --cot/--cot-brief --
# this variant manages eval mode per-candidate internally; the run-level flag
# would only set the "ao" mode's own fallback default, not add anything.
register_optimizer("funnel_wide")(FUNNELv5Optimizer)
