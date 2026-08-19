"""Render evaluation failures into meta-prompt evidence.

Every optimizer in this project shows failing cases to an LLM and asks it to
diagnose a root cause. Until now each one did that inline, with its own
character cap, and on code tasks all of them were wrong in the same two ways.

**Defect 1 — the "expected" field was never an expected answer.** For
`task_type="code"`, `datasets/loader.py` stores `target` as a JSON blob of
unit tests (`{"prompt", "test", "entry_point"}`), because that is what the
execution scorer consumes. Optimizers rendered it verbatim, so every
`Expected:` line was a serialized record whose first key restates the input.
The blob was noticed (see the old comment at `base.py:_feedback_improve`) and
the response was to truncate it *harder*, which kept the defect and hid it.

**Defect 2 — truncation was silent.** Caps ran 80-160 characters. Measured on
a real HumanEval failure: input 415 chars, target 1209, prediction 3366. A
HumanEval prediction echoes the signature and docstring before the body, so
under a 160-char cap the input, the expected value and the prediction are
*near-identical strings*. Asked for a root cause given three mutilated
look-alikes, PACT (EXP-022) recorded this diagnosis verbatim:

    "The model incorrectly truncated the prompt mid-sentence
     ('replaces all vowels i' vs 'replaces all vowels')"

It had compared two truncations of the same string, attributed the cut to the
model under test, and written the "fix" into the prompt — the seed-42 winner
gained *"preserving the original wording and structure"* and lost both the
word `Python` and its output-format constraint.

So the two rules here are: **never show a scoring blob as if it were an
answer**, and **never let an elision be readable as a symptom** — every cut
carries an explicit marker saying how much was removed.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

# Budgets are per field, in characters. They are ~10x the old caps: the whole
# point is that the model can see where a solution actually diverges. A
# 5-failure block at these sizes is roughly 2-3k tokens, negligible against
# the context of any model this project runs.
INPUT_BUDGET = 900
EXPECTED_BUDGET = 400
PREDICTION_BUDGET = 900

# How many assertions to lift out of a HumanEval-style test harness. The
# first few pin the contract; the rest are usually edge cases that cost
# tokens without changing the diagnosis.
MAX_ASSERTIONS = 4

_ELISION = "\n[… {n} characters omitted …]\n"


def _elide(text: str, budget: int, keep: str = "head") -> str:
    """Trim `text` to `budget`, always announcing what was removed.

    `keep` selects which end survives: "head" for inputs (the task statement
    leads), "tail" for code predictions (the signature and docstring are
    echoed boilerplate; the divergence is in the body), "both" when the
    middle is the throwaway.

    A truncation that does not announce itself is indistinguishable from a
    model that stopped early -- that confusion is what this module exists to
    prevent, so the marker is not optional.
    """
    text = text or ""
    if len(text) <= budget:
        return text
    omitted = len(text) - budget
    if keep == "tail":
        return _ELISION.format(n=omitted) + text[-budget:]
    if keep == "both":
        half = budget // 2
        return text[:half] + _ELISION.format(n=omitted) + text[-half:]
    return text[:budget] + _ELISION.format(n=omitted)


def _render_expected(target: Any, budget: int = EXPECTED_BUDGET) -> str:
    """Turn a stored `target` into something that reads as a specification.

    Dispatches on target shape exactly as `evaluation/scoring.py:_score_code`
    does, so this never disagrees with how the sample is actually graded.

    Code targets become their assertions rather than their canonical
    solution: assertions *are* the definition of correctness for an
    execution-scored task, and unlike a reference solution they cannot leak a
    memorised answer into an optimised instruction.
    """
    if not isinstance(target, str):
        return _elide(str(target), budget)

    try:
        meta = json.loads(target)
    except (ValueError, TypeError):
        return _elide(target, budget)          # plain-text task
    if not isinstance(meta, dict):
        return _elide(target, budget)

    # LiveCodeBench-style: explicit input/output pairs.
    if "test_cases" in meta:
        cases = meta.get("test_cases") or []
        lines = []
        for c in cases[:MAX_ASSERTIONS]:
            if isinstance(c, dict):
                lines.append(f"input {c.get('input', '')!r} -> {c.get('output', '')!r}")
        body = "\n".join(lines) or "(test cases unavailable)"
        return _elide(f"the function must satisfy:\n{body}", budget)

    # HumanEval-style: a `check(candidate)` harness carrying assertions.
    if "test" in meta or "entry_point" in meta:
        entry = meta.get("entry_point", "")
        asserts = [
            ln.strip()
            for ln in str(meta.get("test", "")).splitlines()
            if ln.strip().startswith("assert")
        ][:MAX_ASSERTIONS]
        head = f"a function named `{entry}` that satisfies:" if entry else "a function that satisfies:"
        body = "\n".join(asserts) if asserts else "(assertions unavailable)"
        return _elide(f"{head}\n{body}", budget)

    return _elide(target, budget)


def render_failures(
    failures: List[Dict[str, Any]],
    max_failures: int = 5,
    input_budget: int = INPUT_BUDGET,
    expected_budget: int = EXPECTED_BUDGET,
    prediction_budget: int = PREDICTION_BUDGET,
    prediction_label: str = "Got",
    show_correct: bool = False,
    numbered: bool = False,
    empty: str = "(none recorded)",
) -> str:
    """Render failing cases as diagnosable evidence.

    The surface parameters exist only so the five existing call sites keep
    the prompt shape they already had: `numbered` gives PACT's
    "1. INPUT: …" form, `prediction_label="Model output"` and
    `show_correct` give the GEPA/FUNNEL trace form. Changing wording *and*
    content at once would leave EXP-023 unable to attribute its result.
    """
    if not failures:
        return empty

    blocks = []
    for i, f in enumerate(failures[:max_failures], 1):
        inp = _elide(str(f.get("input", "")), input_budget, keep="head")
        want = _render_expected(f.get("target", ""), expected_budget)
        # Tail-biased: a code prediction restates the signature and docstring
        # before the body, so its head is the least informative part of it.
        got = _elide(str(f.get("prediction", "")), prediction_budget, keep="tail")
        if numbered:
            block = f"{i}. INPUT: {inp}\n   EXPECTED: {want}\n   GOT: {got}"
        else:
            block = (f"- Input: {inp}\n  Expected: {want}\n"
                     f"  {prediction_label}: {got}")
        if show_correct:
            block += f"\n  Correct: {f.get('correct')}"
        blocks.append(block)
    return "\n".join(blocks)
