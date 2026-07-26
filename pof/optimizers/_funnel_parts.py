"""Structured prompt representation for FUNNELv3.

A prompt is stored as an ordered composition of named parts rather than as one
opaque string:

    role         who the model should act as
    instruction  the core task statement
    reasoning    how to work through the problem
    format       the required output shape
    caveats      mistakes to avoid
    examples     worked demonstrations

The rendered prompt is those parts concatenated in that order; empty parts
vanish. This follows GSPE's grammar (`gspe.py` PROMPT_GRAMMAR) with one
addition that matters here: GSPE has no `examples` field, and CAPO/SEE both
show demonstrations belong in the genome rather than being one mutation among
many.

**Why the representation, not just an operator.** Measured on six real runs,
84% of rewrites destroyed a parent's demonstrations, because every operator
receives the whole prompt and returns a whole prompt — so a rewrite of the
instruction silently drops whatever else was there. `funnel_v3` previously
patched this by re-attaching examples afterwards, which works but is a repair
applied after the damage. With parts, an instruction operator is handed ONLY
the instruction and its result is written back to ONLY that field, so the other
parts are untouched by construction. Destruction becomes impossible rather than
corrected.

Two further consequences, both free:

- Operator meta-prompts shrink. A rewriter no longer sees the demonstration
  block, which is both cheaper in tokens and removes the risk of the model
  treating solved `Input:/Output:` pairs as instructions to obey.
- Each operator declares which part it writes, so credit is attributable to a
  specific component — the per-facet attribution GSPE lacks (it mutates a
  randomly chosen field and scores only the whole prompt).
"""
from __future__ import annotations

import re
from typing import Dict, Optional

# Rendering order. Also the canonical field list.
PART_ORDER = ["role", "instruction", "reasoning", "format", "caveats", "examples"]

EXAMPLES_MARKER = "Examples:"

# Which part each operator writes. Anything unlisted rewrites `instruction`,
# which is the correct default: the overwhelming majority of operators are
# instruction rewriters.
PART_WRITER: Dict[str, str] = {
    "few_shot": "examples",
    "few_shot_fixed": "examples",
    "few_shot_aug": "examples",
    "format_constraint": "format",
    "facet_enrich": "reasoning",
}


def render(parts: Dict[str, str]) -> str:
    """Compose the parts into the prompt text actually sent to the model."""
    chunks = [
        (parts.get(field) or "").strip()
        for field in PART_ORDER
    ]
    return "\n\n".join(c for c in chunks if c).strip()


def split_raw(text: str) -> Dict[str, str]:
    """Best-effort parts for a prompt that arrived as an opaque string.

    Only the demonstration block can be recovered reliably, since it has an
    explicit marker; everything else becomes `instruction`. That is the honest
    decomposition — inventing role/format boundaries from free text would be
    guesswork, and a wrong split is worse than no split.
    """
    text = text or ""
    head, sep, tail = text.partition("\n\n" + EXAMPLES_MARKER)
    parts = {"instruction": head.strip()}
    if sep and tail.strip():
        parts["examples"] = (EXAMPLES_MARKER + tail).strip()
    return parts


def extract_written(field: str, produced: str, previous: Dict[str, str]) -> Optional[str]:
    """Normalise an operator's raw output into the value for `field`.

    Operators return whole prompts even when they conceptually write one part,
    so the block is isolated here rather than requiring every operator to be
    rewritten.
    """
    produced = (produced or "").strip()
    if not produced:
        return None
    if field != "examples":
        # An instruction rewriter handed only the instruction should return only
        # the instruction; strip a demonstration block if it echoed one anyway.
        return re.split(r"\n\n" + EXAMPLES_MARKER, produced)[0].strip() or None
    if EXAMPLES_MARKER in produced:
        idx = produced.index(EXAMPLES_MARKER)
        return produced[idx:].strip()
    return None
