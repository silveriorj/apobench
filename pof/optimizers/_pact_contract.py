"""The PACT contract: schema, meta-prompt, parser, validator, edit applier.

Every mutating operator in PACT is a *contract call*: the model must return
structured output containing free-form analysis followed by a small number of
anchored edits, and the result is verified programmatically before it is ever
evaluated.

Two layers, because neither alone is sufficient:

- **Grammar** (constrained decoding, applied by the caller) guarantees the
  *shape* -- valid JSON, required fields, edit-count cap.
- **These functions** guarantee the *semantics* -- that each `find` span
  actually occurs verbatim in the parent, that protected regions are
  untouched, that the result differs from the parent. A grammar cannot know
  what text exists in the parent prompt, so this half cannot be delegated to
  it.

Everything here is a pure function over strings, so the whole contract is
unit-testable with no model loaded.

Measured context for the design (see `Dissertacao/experiments_log.md`):
7.9% of stored candidates in this project are unusable as instructions
(preamble, leaked scaffolding, refusals) and were evaluated anyway, because
the only validation in the codebase is `result.strip() if result.strip()`.
Separately, 84% of whole-prompt rewrites destroyed their parent's
demonstrations. Anchored edits make the second failure impossible by
construction and the first detectable.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# At most this many edits per call. A cap is what makes an edit "targeted";
# it also bounds how much damage one bad call can do.
MAX_EDITS = 2

# Appended demonstration blocks are owned by the few-shot operators and must
# survive any instruction rewrite -- this is the region whose destruction the
# 84% measurement was about.
PROTECTED_MARKERS = ("Examples:", "\nInput:")

# A replacement may not blow the prompt up; runaway growth is the documented
# "prompt distributional overfitting" failure mode.
MAX_GROWTH_RATIO = 2.0

# An edit may not rewrite the whole instruction: if a model is allowed to
# target every span at once it has simply performed a whole-prompt rewrite
# through the contract, which is the failure this design exists to prevent.
MAX_SPAN_FRACTION = 0.5

CONTRACT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    # Property order matters: it fixes generation order, so `analysis` is
    # emitted BEFORE any constrained field. Reasoning therefore happens
    # free-form and only the final emission is constrained -- the documented
    # resolution to "structure hurts reasoning" (the effect comes from
    # forcing an answer before reasoning completes, not from constraint).
    #
    # Edits address a span by INDEX, not by quoting it. Measured on
    # Qwen3-0.6B, verbatim quoting failed on ~87% of calls: the model either
    # paraphrased the span (unanchorable) or quoted the entire instruction
    # (a whole-prompt rewrite that passes an anchor check). An integer index
    # is trivially reproducible, is checkable against a known range, and is
    # exactly the kind of constraint a grammar can enforce -- which is what
    # makes constrained decoding worth applying here at all.
    "properties": {
        "analysis": {"type": "string"},
        "edits": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_EDITS,
            "items": {
                "type": "object",
                "properties": {
                    "span_id": {"type": "integer", "minimum": 0},
                    "replace": {"type": "string"},
                },
                "required": ["span_id", "replace"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["analysis", "edits"],
    "additionalProperties": False,
}


def split_spans(prompt: str) -> List[str]:
    """Split the editable instruction into addressable spans.

    Sentence-ish granularity: small enough that replacing one is a targeted
    change, large enough to be a meaningful unit. The appended demonstration
    block is deliberately excluded -- it is never editable, so it is never
    given an index and cannot be addressed at all.
    """
    editable = prompt
    guarded = protected_span(prompt)
    if guarded:
        editable = prompt[: prompt.find(guarded)]
    parts = re.split(r"(?<=[.!?:])\s+|\n+", editable)
    return [p.strip() for p in parts if p and p.strip()]


@dataclass
class ContractResult:
    """Outcome of one contract call."""

    ok: bool
    text: Optional[str] = None          # the edited prompt, when ok
    violation: Optional[str] = None     # machine-named reason, when not ok
    detail: str = ""                    # human/retry-facing explanation
    edits_applied: int = 0
    analysis: str = ""
    raw: str = field(default="", repr=False)


# --------------------------------------------------------------------------
# Meta-prompt
# --------------------------------------------------------------------------

# PE2 (arXiv:2311.05661) ablated meta-prompt components: removing the
# step-by-step reasoning template cost -5/-7 points, the largest single
# effect, while removing role/persona framing was +1/-5 (inconsistent). So
# this template carries an explicit procedure and deliberately carries NO
# persona. Phrasing also avoids "make sure to"-style meta-instructions,
# measured at -0.103 on math tasks (arXiv:2605.26655).
_METHOD_BLOCK = """\
Work through these steps in order:
1. For each failure, name its single root cause:
   REASONING - the required steps are wrong, missing, or in the wrong order.
   FORMAT    - the answer content is right but its shape is wrong.
   SURFACE   - the input contains a phrase that resembles an answer, and the
               model followed it instead of the actual content.
2. Identify which root cause occurs most often across the failures.
3. Quote the exact span of the current instruction responsible for it.
4. Write the smallest replacement for that span that removes the cause."""

_OUTPUT_BLOCK = """\
Return a JSON object with two fields:
- "analysis": your reasoning from the four steps above.
- "edits": an array of at most {max_edits} edit objects, each with:
    "span_id" - the number of the span to replace, from the list above.
    "replace" - the new text for that span.

Edit the fewest spans that remove the root cause. Leave every other span
untouched by not listing it."""


def build_meta_prompt(
    prompt: str,
    failures: List[Dict[str, Any]],
    max_edits: int = MAX_EDITS,
    max_failures: int = 5,
    max_field: int = 160,
) -> str:
    """Assemble the contract meta-prompt.

    Sections are delimited so the model cannot confuse instruction with data
    -- the failure inputs are themselves task text and would otherwise read
    as commands.
    """
    lines = []
    for i, f in enumerate(failures[:max_failures], 1):
        got = str(f.get("prediction", ""))[:max_field]
        want = str(f.get("target", ""))[:max_field]
        inp = str(f.get("input", ""))[:max_field]
        lines.append(f"{i}. INPUT: {inp}\n   EXPECTED: {want}\n   GOT: {got}")
    failure_block = "\n".join(lines) if lines else "(none recorded)"

    spans = split_spans(prompt)
    span_block = "\n".join(f"[{i}] {s}" for i, s in enumerate(spans)) or "[0] (empty)"

    return (
        "# TASK\n"
        "Improve a task instruction that is failing on specific cases, by "
        "replacing as few spans as possible.\n\n"
        "# CURRENT INSTRUCTION, SPLIT INTO NUMBERED SPANS\n"
        f"{span_block}\n\n"
        "# OBSERVED FAILURES\n"
        f"{failure_block}\n\n"
        "# METHOD\n"
        f"{_METHOD_BLOCK}\n\n"
        "# OUTPUT\n"
        f"{_OUTPUT_BLOCK.format(max_edits=max_edits)}\n"
    )


def build_retry_prompt(original: str, violation_detail: str) -> str:
    """Re-ask, naming the concrete violation.

    Naming the specific failure is materially more effective than a blind
    "try again", and costs the same single call.
    """
    return (
        f"{original}\n\n"
        "# PREVIOUS ATTEMPT REJECTED\n"
        f"{violation_detail}\n"
        "Return corrected JSON in the same format. Use only \"span_id\" values "
        "that appear in the numbered list above."
    )


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def parse_response(raw: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Extract the contract object from a model response.

    Tolerates the two things that survive even constrained decoding in the
    fallback path: markdown fences, and prose surrounding the object. Returns
    `(obj, None)` or `(None, reason)`.
    """
    if not raw or not raw.strip():
        return None, "empty response"
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass
    m = _JSON_OBJ.search(text)
    if not m:
        return None, "no JSON object found in response"
    try:
        return json.loads(m.group(0)), None
    except json.JSONDecodeError as e:
        return None, f"malformed JSON: {e.msg}"


# --------------------------------------------------------------------------
# Validation + application
# --------------------------------------------------------------------------

def protected_span(prompt: str) -> Optional[str]:
    """The appended demonstration block, if the prompt carries one."""
    for marker in PROTECTED_MARKERS:
        idx = prompt.find(marker)
        if idx != -1:
            return prompt[idx:]
    return None


def apply_contract(
    prompt: str,
    raw: str,
    max_edits: int = MAX_EDITS,
) -> ContractResult:
    """Parse, validate and apply a contract response against `prompt`.

    Every rejection path names a distinct `violation` so failures can be
    counted by cause rather than lumped into "invalid" -- the point of the
    exercise is to learn *how* models break the contract.
    """
    obj, err = parse_response(raw)
    if obj is None:
        return ContractResult(False, violation="unparseable", detail=err or "", raw=raw)

    analysis = obj.get("analysis")
    if not isinstance(analysis, str):
        analysis = ""

    edits = obj.get("edits")
    if not isinstance(edits, list) or not edits:
        return ContractResult(
            False, violation="no_edits",
            detail="The \"edits\" array was missing or empty.",
            analysis=analysis, raw=raw)
    if len(edits) > max_edits:
        return ContractResult(
            False, violation="too_many_edits",
            detail=f"{len(edits)} edits returned; at most {max_edits} allowed.",
            analysis=analysis, raw=raw)

    guarded = protected_span(prompt)
    spans = split_spans(prompt)
    if not spans:
        return ContractResult(
            False, violation="no_spans",
            detail="The instruction has no editable spans.",
            analysis=analysis, raw=raw)

    replacements: Dict[int, str] = {}
    for i, edit in enumerate(edits, 1):
        if not isinstance(edit, dict):
            return ContractResult(
                False, violation="bad_edit_type",
                detail=f"Edit {i} is not an object.",
                analysis=analysis, raw=raw)
        sid = edit.get("span_id")
        replace = edit.get("replace")
        if isinstance(sid, bool) or not isinstance(sid, int):
            return ContractResult(
                False, violation="bad_span_id",
                detail=f"Edit {i}: \"span_id\" must be an integer.",
                analysis=analysis, raw=raw)
        if not 0 <= sid < len(spans):
            return ContractResult(
                False, violation="span_out_of_range",
                detail=(f"Edit {i}: span_id {sid} does not exist; valid ids "
                        f"are 0..{len(spans) - 1}."),
                analysis=analysis, raw=raw)
        if not isinstance(replace, str):
            return ContractResult(
                False, violation="bad_replace",
                detail=f"Edit {i} has a non-string \"replace\".",
                analysis=analysis, raw=raw)
        if sid in replacements:
            return ContractResult(
                False, violation="duplicate_span",
                detail=f"Edit {i}: span {sid} was already edited.",
                analysis=analysis, raw=raw)
        replacements[sid] = replace

    # Targeting every span is a whole-prompt rewrite wearing a contract.
    if len(spans) > 1 and len(replacements) > max(1, int(len(spans) * MAX_SPAN_FRACTION)):
        return ContractResult(
            False, violation="rewrite_disguised_as_edit",
            detail=(f"{len(replacements)} of {len(spans)} spans targeted; at "
                    f"most {MAX_SPAN_FRACTION:.0%} may be replaced in one edit."),
            analysis=analysis, raw=raw)

    rebuilt = " ".join(
        replacements.get(i, s) for i, s in enumerate(spans)
        if replacements.get(i, s).strip()
    )
    working = rebuilt + (("\n\n" + guarded) if guarded else "")
    applied = len(replacements)

    if guarded and guarded not in working:
        return ContractResult(
            False, violation="protected_destroyed",
            detail="The worked-examples block did not survive the edits.",
            analysis=analysis, raw=raw)
    if not working.strip():
        return ContractResult(
            False, violation="empty_result",
            detail="The edits emptied the instruction.",
            analysis=analysis, raw=raw)
    if working.strip() == prompt.strip():
        return ContractResult(
            False, violation="no_op",
            detail="The edits left the instruction unchanged.",
            analysis=analysis, raw=raw)
    if len(working) > max(80, len(prompt) * MAX_GROWTH_RATIO):
        return ContractResult(
            False, violation="excessive_growth",
            detail=(f"Result grew from {len(prompt)} to {len(working)} "
                    f"characters; at most {MAX_GROWTH_RATIO}x is allowed."),
            analysis=analysis, raw=raw)

    return ContractResult(
        True, text=working.strip(), edits_applied=applied,
        analysis=analysis, raw=raw)
