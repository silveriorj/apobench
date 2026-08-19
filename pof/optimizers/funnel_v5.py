"""FUNNELv5 — FUNNELv4d ("FUNNEL-Lean") with eval mode as a searchable
candidate property: answer-only, brief CoT, or full CoT/thinking.

Prior FUNNEL variants fix the eval mode (system prompt + token budget) for
the whole run, but the best mode is task-dependent — e.g. full CoT helps
`boolean_expressions` but hurts `causal_judgement` relative to answer-only,
matching a reversal already reported in Suzgun et al. (2023) Table 3 for
larger models. Rather than guess per task, eval mode becomes a heritable
`PromptRecord` property (`metadata["mode"]`): new candidates inherit their
parent's mode, and a guaranteed "mode family" mints zero-LLM-call siblings
in each of the other two modes for every top elite each phase. Ordinary
tournament selection then does the work — whichever mode scores higher on
a given task keeps propagating, without an explicit gate.

Mode siblings share text with their elite, so `_is_duplicate` (which hashes
only text) would otherwise drop them; they're deduplicated on
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
    # `cot` is deliberately the SHORT arm -- one-line-per-step, ending in
    # "So the answer is X". Its 256-token cap is the hypothesis it exists to
    # test (does most of full CoT's gain survive a much shorter trace?), so
    # raising it would collapse this mode into `thinking` and destroy the
    # comparison. Kept small on purpose.
    "cot": (SYSTEM_PROMPT_BY_TASK_TYPE["cot"], 256),
    # `thinking` is the generous arm: full reasoning to \boxed{}. Matches
    # COT_MAX_NEW_TOKENS in experiments/run_swift_apex.py.
    "thinking": (SYSTEM_PROMPT_BY_TASK_TYPE["thinking"], 2048),
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
