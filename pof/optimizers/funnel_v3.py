"""FUNNELv3 — FUNNELv2 with guaranteed few-shot exploration.

v3 is v2 with one change: `few_shot` is promoted from a bandit arm to a
guaranteed operator applied to the top elites every phase. Everything else —
the 20-operator pool, flat N=50, the output-verbosity barrier, trap detection,
equal-N selection, held-out final selection, cross-run pruning — is inherited
unchanged, so a v2-vs-v3 comparison isolates that single decision.

**Why few-shot specifically.** Inspecting the winning prompt of all six
completed runs on `bbh_boolean_expressions` (Qwen3-4B, v1 and v2, three seeds
each) showed one clean split:

    method  seed  test    winning operator              prompt form
    v1      123   0.8609  lamarckian                    bare instruction
    v1      42    0.8522  local_edit                    bare instruction
    v1      7     0.8522  mutator_structural_variation  bare instruction
    v2      123   0.8522  grips_swap                    bare instruction
    v2      42    0.9217  few_shot                      instruction + examples
    v2      7     0.8435  lamarckian                    bare instruction

Every bare-instruction winner landed in 0.843-0.861. The one run whose winner
carried worked examples reached 0.9217 — a ~0.06 gap unrelated to scheduling,
operator diversity or selection bias. It is simply whether demonstrations
survive into the final prompt. That matches CAPO and SEE, which both treat
exemplars as part of the genome rather than as one mutation among many.

**Why guaranteeing it is the right response.** The argument is NOT that few-shot
prompts always win — on five of six runs they did not. It is that few-shot
should always be *tested*. Under pure bandit selection it is one arm of 20
applied to a randomly drawn elite with a random 1-3 examples, so it is sampled
about three times per run (and on one v1 seed, zero times). Three scattershot
draws are a thin probe of a subspace that, when it pays, pays by 0.06. Sweeping
it across the top elites every phase raises that to ~12 systematic attempts.

**Why it is nearly free.** `t_few_shot` makes ZERO LLM calls — it strips any
existing demonstration block and concatenates training examples as strings. The
only cost is evaluating the extra candidates, unlike the previous v3 whose
static core used three multi-call diagnosis operators and doubled runtime for
no measurable gain (see `proposta_funnel_v2.md` §3.9).

This supersedes the earlier v3 (static StraGo/ETGPO/AutoHint backbone), which
measured no benefit at 2.1x the cost. That negative result is recorded in the
proposal document and in commit history; its run directories are removed
because `funnel_v3` now denotes a different method.
"""
from __future__ import annotations

import logging
from typing import List

from pof.optimizers import register_optimizer
from pof.optimizers.funnel_v2 import FUNNEL_V2_POOL, FUNNELv2Optimizer

logger = logging.getLogger(__name__)

# Promoted to a guaranteed sweep; everything else stays under UCB1.
V3_STATIC: List[str] = ["few_shot"]
V3_BANDIT: List[str] = [t for t in FUNNEL_V2_POOL if t not in V3_STATIC]

assert len(V3_BANDIT) == 19, f"expected 19 bandit arms, got {len(V3_BANDIT)}"


@register_optimizer("funnel_v3")
class FUNNELv3Optimizer(FUNNELv2Optimizer):
    """FUNNELv2 with few-shot augmentation guaranteed — see module docstring."""

    name = "funnel_v3"

    STATIC_ARMS: List[str] = V3_STATIC
    BANDIT_ARMS: List[str] = V3_BANDIT
