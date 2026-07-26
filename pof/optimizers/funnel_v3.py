"""FUNNELv3 — curated static backbone plus adaptive remainder.

FUNNELv3 is FUNNELv2 with one change: three operators are promoted from bandit
arms to a guaranteed backbone applied to EVERY elite, every phase. Everything
else — the 20-operator pool, the flat N=50 evaluation, the output-verbosity
barrier, trap detection, equal-N selection, cross-run pruning — is inherited
unchanged, so a v2-vs-v3 comparison isolates the scheduling contract alone.

**The problem it addresses.** v2 (like APEX before it) selects operators purely
by bandit, and each selected operator then acts on ONE elite drawn at random
from the top-3. Coverage is therefore stochastic and incomplete: with two
attempts per operator the best prompt has a 44% chance of never receiving any
given operator, and elites 4-5 are never touched at all, which effectively
reduces the population to three. The operators most damaged by that are exactly
the ones v2 exists to test — success-anchored feedback, error-taxonomy
guidance, aggregated hints — because their value comes from being applied
consistently to the prompts that are actually competing.

**Where it sits in the study.** The paper's scheduling axis previously had two
endpoints: SWIFT is pure curation (fixed phase order, every elite covered) and
APEX pure adaptation (bandit allocation, random coverage). Neither tests the
middle. FUNNELv3 is that middle: a curated backbone guaranteeing coverage of
the operators we have the strongest prior reason to trust, with UCB1 arbitrating
the remaining, more speculative ones. Whether the hybrid beats either endpoint
is the empirical question; the design does not presume it.

**Why these three.** They are the highest-yield additions from the 2024-2026
methods surveyed (see `_funnel_v2_techniques.py`) and are also the ones whose
mechanism is least tolerant of patchy application:

- `strago_dual` (StraGo, EMNLP 2024 Findings) — the only operator in either
  pool that reads correct predictions as well as failures, explicitly to
  prevent "prompt drifting", where fixing failures degrades cases that already
  worked. Directly targets the dev-to-test generalization gap measured on
  swift_v2/apex_v2 in this project.
- `etgpo_taxonomy` (ETGPO) — builds a frequency-ranked taxonomy of failure
  modes before rewriting, so the rewrite attacks the modal error rather than
  whichever handful of failures happened to be sampled.
- `autohint` (AutoHint) — aggregates all failures into one generalized hint,
  the cheapest failure-driven operator available at a single call.

They are exempt from cross-run pruning: pruning the backbone would silently
dismantle the very contract under test.

Note this is a genuine cost increase, roughly +38% candidates per phase
(~16 to ~22), because guaranteed coverage means the static operators fire
`len(population)` times per phase instead of once or twice.
"""
from __future__ import annotations

import logging
from typing import List

from pof.optimizers import register_optimizer
from pof.optimizers.funnel_v2 import (
    BANDIT_POOL,
    STATIC_CORE,
    FUNNELv2Optimizer,
)

logger = logging.getLogger(__name__)


@register_optimizer("funnel_v3")
class FUNNELv3Optimizer(FUNNELv2Optimizer):
    """FUNNELv2 plus a guaranteed static backbone — see module docstring."""

    name = "funnel_v3"

    # The entire behavioural difference from v2 is these two lines: three
    # operators move from the bandit arm set into the guaranteed backbone.
    # `_apply_static_core` (inherited) sweeps STATIC_ARMS over every elite;
    # `BANDIT_ARMS` is what UCB1 selects from and what pruning may touch.
    STATIC_ARMS: List[str] = STATIC_CORE
    BANDIT_ARMS: List[str] = BANDIT_POOL
