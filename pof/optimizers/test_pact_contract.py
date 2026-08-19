"""Offline tests for the PACT contract — no LLM, no dataset.

The contract is the method's central claim (verified calls beat unverified
ones), so every rejection path is asserted individually: the point is to
count failures *by cause*, and a violation label that never fires, or fires
for the wrong reason, would silently corrupt that measurement.

Edits address spans by INDEX rather than by quoting them. That is not a
stylistic choice — measured on Qwen3-0.6B, verbatim quoting failed on ~87%
of calls, either by paraphrasing the span or by quoting the entire
instruction (a whole-prompt rewrite that passes an anchor check).
"""
import json

import pytest

from pof.optimizers._pact_contract import (
    MAX_EDITS,
    MAX_SPAN_FRACTION,
    apply_contract,
    build_meta_prompt,
    build_retry_prompt,
    parse_response,
    protected_span,
    split_spans,
)

PROMPT = "Solve the task carefully. Answer with a single number."
WITH_EX = PROMPT + "\n\nExamples:\nInput: 2+2\nOutput: 4"
LONG = "One. Two. Three. Four. Five. Six."


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
        obj, err = parse_response('Sure! Here you go:\n{"analysis":"a","edits":[]}\nDone.')
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


# -------------------------------------------------------------- spans ----

class TestSplitSpans:
    def test_splits_on_sentence_boundaries(self):
        assert split_spans(PROMPT) == [
            "Solve the task carefully.", "Answer with a single number."]

    def test_demonstration_block_is_never_addressable(self):
        """Excluded from the span list, so no span_id can reach it."""
        spans = split_spans(WITH_EX)
        assert spans == split_spans(PROMPT)
        assert not any("Output: 4" in s for s in spans)

    def test_single_sentence_yields_one_span(self):
        assert split_spans("Do the thing.") == ["Do the thing."]


# ---------------------------------------------------------- happy path ----

class TestApply:
    def test_single_edit_applies(self):
        r = apply_contract(PROMPT, _resp([
            {"span_id": 0, "replace": "Solve the task step by step."}]))
        assert r.ok
        assert r.text == "Solve the task step by step. Answer with a single number."
        assert r.edits_applied == 1
        assert r.analysis == "because"

    def test_edit_to_a_later_span(self):
        r = apply_contract(PROMPT, _resp([
            {"span_id": 1, "replace": "Answer with one integer."}]))
        assert r.ok
        assert r.text == "Solve the task carefully. Answer with one integer."

    def test_two_edits_on_a_long_prompt(self):
        r = apply_contract(LONG, _resp([
            {"span_id": 0, "replace": "Uno."}, {"span_id": 5, "replace": "Seis."}]))
        assert r.ok and r.edits_applied == 2
        assert r.text == "Uno. Two. Three. Four. Five. Seis."

    def test_demonstration_block_survives(self):
        """The 84%-destructive-rewrite failure must be impossible here."""
        r = apply_contract(WITH_EX, _resp([
            {"span_id": 0, "replace": "Solve the task precisely."}]))
        assert r.ok
        assert "Examples:\nInput: 2+2\nOutput: 4" in r.text
        assert r.text.startswith("Solve the task precisely.")

    def test_empty_replacement_drops_that_span(self):
        r = apply_contract(PROMPT, _resp([{"span_id": 0, "replace": ""}]))
        assert r.ok and r.text == "Answer with a single number."


# ---------------------------------------------------------- rejections ----

class TestViolations:
    def test_unparseable(self):
        r = apply_contract(PROMPT, "not json at all")
        assert not r.ok and r.violation == "unparseable"

    def test_no_edits(self):
        r = apply_contract(PROMPT, _resp([]))
        assert not r.ok and r.violation == "no_edits"

    def test_too_many_edits(self):
        edits = [{"span_id": i, "replace": "y."} for i in range(MAX_EDITS + 1)]
        r = apply_contract(LONG, _resp(edits))
        assert not r.ok and r.violation == "too_many_edits"
        assert str(MAX_EDITS) in r.detail

    def test_span_out_of_range(self):
        """The index replaces verbatim quoting precisely so this is checkable."""
        r = apply_contract(PROMPT, _resp([{"span_id": 99, "replace": "x"}]))
        assert not r.ok and r.violation == "span_out_of_range"
        assert "0..1" in r.detail

    def test_negative_span_rejected(self):
        r = apply_contract(PROMPT, _resp([{"span_id": -1, "replace": "x"}]))
        assert not r.ok and r.violation == "span_out_of_range"

    def test_non_integer_span_id(self):
        r = apply_contract(PROMPT, _resp([{"span_id": "zero", "replace": "x"}]))
        assert not r.ok and r.violation == "bad_span_id"

    def test_bool_is_not_an_integer_span_id(self):
        """bool subclasses int in Python; True must not address span 1."""
        r = apply_contract(PROMPT, _resp([{"span_id": True, "replace": "x"}]))
        assert not r.ok and r.violation == "bad_span_id"

    def test_non_string_replace(self):
        r = apply_contract(PROMPT, _resp([{"span_id": 0, "replace": 5}]))
        assert not r.ok and r.violation == "bad_replace"

    def test_duplicate_span(self):
        r = apply_contract(LONG, _resp([
            {"span_id": 1, "replace": "a."}, {"span_id": 1, "replace": "b."}]))
        assert not r.ok and r.violation == "duplicate_span"

    def test_rewriting_every_span_is_refused(self):
        """Targeting the whole prompt is a rewrite wearing a contract."""
        r = apply_contract(PROMPT, _resp([
            {"span_id": 0, "replace": "A."}, {"span_id": 1, "replace": "B."}]))
        assert not r.ok and r.violation == "rewrite_disguised_as_edit"
        assert f"{MAX_SPAN_FRACTION:.0%}" in r.detail

    def test_no_op_edit_is_refused(self):
        r = apply_contract(PROMPT, _resp([
            {"span_id": 0, "replace": "Solve the task carefully."}]))
        assert not r.ok and r.violation == "no_op"

    def test_excessive_growth_is_refused(self):
        r = apply_contract(PROMPT, _resp([{"span_id": 0, "replace": "x" * 500}]))
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
            {"edits": [{"span_id": 0, "replace": "Solve it quickly."}]}))
        assert r.ok and r.analysis == ""


# -------------------------------------------------------- meta-prompts ----

class TestMetaPrompt:
    def test_contains_required_sections_and_no_persona(self):
        mp = build_meta_prompt(PROMPT, [{"input": "2+2", "target": "4", "prediction": "5"}])
        for section in ("# TASK", "# CURRENT INSTRUCTION, SPLIT INTO NUMBERED SPANS",
                        "# OBSERVED FAILURES", "# METHOD", "# OUTPUT"):
            assert section in mp
        # PE2: the reasoning template is the load-bearing component...
        assert "REASONING" in mp and "FORMAT" in mp and "SURFACE" in mp
        # ...while persona framing measured inconsistent, so it is absent.
        assert "You are a" not in mp and "expert prompt engineer" not in mp

    def test_spans_are_numbered_for_addressing(self):
        mp = build_meta_prompt(PROMPT, [])
        assert "[0] Solve the task carefully." in mp
        assert "[1] Answer with a single number." in mp

    def test_asks_for_span_id_not_verbatim_quoting(self):
        mp = build_meta_prompt(PROMPT, [])
        assert "span_id" in mp
        assert "character-for-character" not in mp

    def test_failures_are_truncated(self):
        mp = build_meta_prompt(PROMPT, [{"input": "x" * 900, "target": "t", "prediction": "p"}])
        assert "x" * 900 not in mp

    def test_handles_no_failures(self):
        assert "(none recorded)" in build_meta_prompt(PROMPT, [])

    def test_retry_names_the_violation(self):
        r = apply_contract(PROMPT, _resp([{"span_id": 99, "replace": "x"}]))
        retry = build_retry_prompt(build_meta_prompt(PROMPT, []), r.detail)
        assert "PREVIOUS ATTEMPT REJECTED" in retry
        assert "span_id 99" in retry


class TestProtectedSpan:
    def test_detects_examples_block(self):
        assert protected_span(WITH_EX).startswith("Examples:")

    def test_none_when_absent(self):
        assert protected_span(PROMPT) is None
