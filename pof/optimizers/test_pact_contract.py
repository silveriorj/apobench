"""Offline tests for the PACT contract — no LLM, no dataset.

The contract is the method's central claim (verified calls beat unverified
ones), so every rejection path is asserted individually: the point is to
count failures *by cause*, and a violation label that never fires, or fires
for the wrong reason, would silently corrupt that measurement.
"""
import json

import pytest

from pof.optimizers._pact_contract import (
    MAX_EDITS,
    apply_contract,
    build_meta_prompt,
    build_retry_prompt,
    parse_response,
    protected_span,
)

PROMPT = "Solve the task carefully. Answer with a single number."
WITH_EX = PROMPT + "\n\nExamples:\nInput: 2+2\nOutput: 4"


def _resp(edits, analysis="because"):
    return json.dumps({"analysis": analysis, "edits": edits})


# ------------------------------------------------------------- parsing ----

class TestParse:
    def test_plain_json(self):
        obj, err = parse_response('{"analysis":"a","edits":[]}')
        assert err is None and obj["analysis"] == "a"

    def test_markdown_fenced(self):
        obj, err = parse_response('```json\n{"analysis":"a","edits":[]}\n```')
        assert err is None and obj["analysis"] == "a"

    def test_json_embedded_in_prose(self):
        """The fallback path still sees preambles; extract rather than fail."""
        obj, err = parse_response('Sure! Here you go:\n{"analysis":"a","edits":[]}\nHope that helps.')
        assert err is None and obj["analysis"] == "a"

    def test_empty_is_rejected(self):
        obj, err = parse_response("   ")
        assert obj is None and "empty" in err

    def test_no_json_is_rejected(self):
        obj, err = parse_response("I cannot help with that request.")
        assert obj is None and "no JSON" in err

    def test_malformed_json_is_rejected(self):
        obj, err = parse_response('{"analysis": "a", "edits": [}')
        assert obj is None and "malformed" in err.lower()


# ---------------------------------------------------------- happy path ----

class TestApply:
    def test_single_edit_applies(self):
        r = apply_contract(PROMPT, _resp([{"find": "carefully", "replace": "step by step"}]))
        assert r.ok
        assert r.text == "Solve the task step by step. Answer with a single number."
        assert r.edits_applied == 1
        assert r.analysis == "because"

    def test_two_edits_apply_in_order(self):
        r = apply_contract(PROMPT, _resp([
            {"find": "carefully", "replace": "step by step"},
            {"find": "a single number", "replace": "one integer"},
        ]))
        assert r.ok and r.edits_applied == 2
        assert r.text == "Solve the task step by step. Answer with one integer."

    def test_only_first_occurrence_is_replaced(self):
        p = "repeat repeat repeat"
        r = apply_contract(p, _resp([{"find": "repeat", "replace": "once"}]))
        assert r.ok and r.text == "once repeat repeat"

    def test_examples_block_survives_an_instruction_edit(self):
        """The 84%-destructive-rewrite failure must be impossible here."""
        r = apply_contract(WITH_EX, _resp([{"find": "carefully", "replace": "precisely"}]))
        assert r.ok
        assert "Examples:\nInput: 2+2\nOutput: 4" in r.text


# ---------------------------------------------------------- rejections ----

class TestViolations:
    def test_unparseable(self):
        r = apply_contract(PROMPT, "not json at all")
        assert not r.ok and r.violation == "unparseable"

    def test_no_edits(self):
        r = apply_contract(PROMPT, _resp([]))
        assert not r.ok and r.violation == "no_edits"

    def test_too_many_edits(self):
        edits = [{"find": f"x{i}", "replace": "y"} for i in range(MAX_EDITS + 1)]
        r = apply_contract(PROMPT, _resp(edits))
        assert not r.ok and r.violation == "too_many_edits"
        assert str(MAX_EDITS) in r.detail

    def test_anchor_missing_is_the_key_check(self):
        """A grammar cannot know the parent's spans; this is why Python must."""
        r = apply_contract(PROMPT, _resp([{"find": "nowhere in the prompt", "replace": "x"}]))
        assert not r.ok and r.violation == "anchor_missing"
        assert "nowhere in the prompt" in r.detail

    def test_empty_find(self):
        r = apply_contract(PROMPT, _resp([{"find": "", "replace": "x"}]))
        assert not r.ok and r.violation == "empty_find"

    def test_non_string_replace(self):
        r = apply_contract(PROMPT, _resp([{"find": "carefully", "replace": 5}]))
        assert not r.ok and r.violation == "bad_replace"

    def test_edit_targeting_protected_region_is_refused(self):
        r = apply_contract(WITH_EX, _resp([{"find": "Input: 2+2", "replace": "Input: 3+3"}]))
        assert not r.ok and r.violation == "protected_region"

    def test_no_op_edit_is_refused(self):
        r = apply_contract(PROMPT, _resp([{"find": "carefully", "replace": "carefully"}]))
        assert not r.ok and r.violation == "no_op"

    def test_emptying_the_prompt_is_refused(self):
        r = apply_contract(PROMPT, _resp([{"find": PROMPT, "replace": ""}]))
        assert not r.ok and r.violation in ("empty_result", "no_op")

    def test_excessive_growth_is_refused(self):
        r = apply_contract(PROMPT, _resp([{"find": "carefully", "replace": "x" * 500}]))
        assert not r.ok and r.violation == "excessive_growth"

    def test_edits_not_a_list(self):
        r = apply_contract(PROMPT, json.dumps({"analysis": "a", "edits": "nope"}))
        assert not r.ok and r.violation == "no_edits"

    def test_edit_not_an_object(self):
        r = apply_contract(PROMPT, json.dumps({"analysis": "a", "edits": ["oops"]}))
        assert not r.ok and r.violation == "bad_edit_type"

    def test_missing_analysis_is_tolerated(self):
        """Analysis aids the model's reasoning; its absence is not fatal."""
        r = apply_contract(PROMPT, json.dumps(
            {"edits": [{"find": "carefully", "replace": "quickly"}]}))
        assert r.ok and r.analysis == ""


# -------------------------------------------------------- meta-prompts ----

class TestMetaPrompt:
    def test_contains_required_sections_and_no_persona(self):
        mp = build_meta_prompt(PROMPT, [{"input": "2+2", "target": "4", "prediction": "5"}])
        for section in ("# TASK", "# CURRENT INSTRUCTION", "# OBSERVED FAILURES",
                        "# METHOD", "# OUTPUT"):
            assert section in mp
        assert PROMPT in mp
        # PE2: the reasoning template is the load-bearing component...
        assert "REASONING" in mp and "FORMAT" in mp and "SURFACE" in mp
        # ...while persona framing measured inconsistent, so it is absent.
        assert "You are a" not in mp and "expert prompt engineer" not in mp

    def test_delimits_instruction_from_data(self):
        """Failure inputs are task text and would otherwise read as commands."""
        mp = build_meta_prompt(PROMPT, [])
        assert "<<<" in mp and ">>>" in mp

    def test_failures_are_truncated(self):
        mp = build_meta_prompt(PROMPT, [{"input": "x" * 900, "target": "t", "prediction": "p"}])
        assert "x" * 900 not in mp

    def test_handles_no_failures(self):
        assert "(none recorded)" in build_meta_prompt(PROMPT, [])

    def test_retry_names_the_violation(self):
        r = apply_contract(PROMPT, _resp([{"find": "absent span", "replace": "x"}]))
        retry = build_retry_prompt(build_meta_prompt(PROMPT, []), r.detail)
        assert "PREVIOUS ATTEMPT REJECTED" in retry
        assert "absent span" in retry


class TestProtectedSpan:
    def test_detects_examples_block(self):
        assert protected_span(WITH_EX).startswith("Examples:")

    def test_none_when_absent(self):
        assert protected_span(PROMPT) is None
