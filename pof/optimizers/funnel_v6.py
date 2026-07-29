"""FUNNELv6 — FUNNELv5 ("FUNNEL-Wide") with an EDA-informed prior on which
eval modes are worth exploring per task, adaptively corrected by real
evidence.

**The index.** From this session's FUNNEL results: on `boolean_expressions`,
full CoT reaches 0.997 vs. 0.896 best answer-only; on `causal_judgement`,
full CoT is WORSE than answer-only (0.583 vs. 0.649) -- and that reversal
independently reproduces Suzgun et al. (2023) Table 3's own
CoT-hurts-causal_judgement finding. Only 2 tasks are validated head-to-head;
the rest are hypotheses by task-property analogy (symbolic/formal-logic
tasks lean toward reasoning helping; judgment/pattern-recognition tasks lean
toward answer-only winning). Unknown tasks default to exploring fully --
the index is a cost-saving PRIOR, not a hard rule, precisely because it's
thin.

**Why adaptive, not a hard skip.** v5's mode family always mints siblings in
every mode a candidate doesn't already have. That's correct but expensive on
a task where CoT/thinking predictably won't pay off. v6 restricts the
guaranteed family to answer-only siblings ONLY when BOTH: (a) the index
predicts answer-only wins for this task, AND (b) by phase 2 the actual
measured evidence still agrees (no CoT/thinking-mode candidate has beaten
the best answer-only one yet). Condition (b) is what makes this
self-correcting rather than a blind assumption baked in: if the index is
wrong for a given seed/task, real evidence overrides it and full exploration
resumes -- gating only ever narrows an already-losing family, the same
principle as v4a's adaptive gate, applied to eval mode instead of the
decomposition operator family.
"""
from __future__ import annotations

import logging
from typing import Dict, List

from pof.core.types import PromptRecord
from pof.optimizers import register_optimizer
from pof.optimizers.funnel_v5 import FUNNELv5Optimizer

logger = logging.getLogger(__name__)

# task name -> whether CoT/thinking reasoning modes are worth exploring at
# all, vs. sticking to answer-only. True/False values validated on the 2
# tasks noted above; the rest are hypotheses, not measurements -- see the
# module docstring. Unknown tasks default to True (explore) in __init__.
REASONING_TASK_INDEX: Dict[str, bool] = {
    "boolean_expressions": True,       # validated: CoT 0.997 vs AO 0.896
    "dyck_languages": True,            # hypothesis: symbolic stack simulation
    "formal_fallacies": True,          # hypothesis: formal logic
    "logical_deduction_five_objects": True,   # hypothesis: constraint reasoning
    "reasoning_about_colored_objects": True,  # hypothesis: multi-step attribute tracking
    "causal_judgement": False,         # validated: CoT 0.583 vs AO 0.649 (matches Suzgun Table 3)
    "hyperbaton": False,               # hypothesis: grammar pattern recognition
    "disambiguation_qa": False,        # hypothesis: reading comprehension
    "web_of_lies": False,              # near-chance for every mode tried; not worth paying for CoT
    "sports_understanding": False,     # hypothesis: commonsense plausibility, causal_judgement-like
}

# Phase before which the index is never consulted -- every task gets at
# least this many phases of unbiased, full 3-mode exploration regardless of
# what the index predicts, so a wrong prediction gets a real chance to be
# overturned by evidence before gating can kick in.
MIN_PHASES_BEFORE_GATING = 2


@register_optimizer("funnel_v6")
class FUNNELv6Optimizer(FUNNELv5Optimizer):
    """FUNNELv5 with an adaptive, EDA-informed prior on eval-mode exploration."""

    name = "funnel_v6"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        task = (self.dataset.metadata or {}).get("task", "")
        self._favor_reasoning = REASONING_TASK_INDEX.get(task, True)
        if task in REASONING_TASK_INDEX:
            logger.info(
                f"[{self.name}] task='{task}' reasoning-mode index: "
                f"{'favor exploring CoT/thinking' if self._favor_reasoning else 'favor answer-only, gate reasoning modes if evidence agrees'}"
            )

    def _apply_mode_family(self, candidates: List[PromptRecord]) -> int:
        if self._favor_reasoning or not self._reasoning_modes_still_losing():
            return super()._apply_mode_family(candidates)

        # Index predicts answer-only wins here, and by MIN_PHASES_BEFORE_GATING
        # real evidence still agrees -- restrict the guaranteed family to
        # answer-only siblings only, skipping the CoT/thinking ones v5 would
        # otherwise mint every phase regardless of how they're doing.
        made = 0
        elites = self.population[: self.static_top_k] if self.population else []
        for elite in elites:
            if not elite.text or elite.metadata.get("mode", "ao") == "ao":
                continue
            key = (elite.text_hash, "ao")
            if key in self._mode_pairs_seen:
                continue
            record = self._create_record(
                elite.text, operator="mode_ao", parent_ids=[elite.id], mode="ao",
            )
            candidates.append(record)
            made += 1
        if made:
            logger.info(
                f"[{self.name} Phase {self._phase_idx}] mode family restricted "
                f"to answer-only ({made} sibling(s)) -- index predicts reasoning "
                f"won't help here and evidence so far agrees"
            )
        return made

    def _reasoning_modes_still_losing(self) -> bool:
        """True if AO's best score is still >= the best CoT/thinking score.

        Only meaningful from phase MIN_PHASES_BEFORE_GATING onward -- before
        that, every task explores fully regardless of the index.
        """
        if self._phase_idx < MIN_PHASES_BEFORE_GATING:
            return False
        records = self.tracker.history.records.values()
        ao_best = max(
            (r.score for r in records if r.metadata.get("mode", "ao") == "ao"),
            default=0.0,
        )
        reasoning_best = max(
            (r.score for r in records if r.metadata.get("mode") in ("cot", "thinking")),
            default=0.0,
        )
        return reasoning_best <= ao_best


# Alias: paper-facing name, same class.
register_optimizer("funnel_indexed")(FUNNELv6Optimizer)
