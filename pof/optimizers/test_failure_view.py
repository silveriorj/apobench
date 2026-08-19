"""Offline tests for the failure renderer — no LLM, no dataset.

These assert the two properties that EXP-022 violated: a code task's
EXPECTED must never be the JSON scoring blob, and no field may be cut
without saying so. The second is the load-bearing one — a silent cut is
what let PACT diagnose "the model truncated the prompt mid-sentence" when
the meta-prompt builder had done the truncating.
"""
import json

from pof.optimizers._failure_view import (
    INPUT_BUDGET,
    PREDICTION_BUDGET,
    _elide,
    render_failures,
)

MARKER = "characters omitted"

HUMANEVAL_TARGET = json.dumps({
    "prompt": "def vowels_count(s):\n    \"\"\"Write a function which "
              "replaces all vowels in a string.\"\"\"\n",
    "test": (
        "METADATA = {}\n\n\n"
        "def check(candidate):\n"
        "    assert candidate('abcde') == 2\n"
        "    assert candidate('ACEDY') == 3\n"
        "    assert candidate('') == 0\n"
    ),
    "entry_point": "vowels_count",
})

LCB_TARGET = json.dumps({
    "test_cases": [{"input": "1 2", "output": "3"}, {"input": "4 5", "output": "9"}],
})


def _failure(**kw):
    base = {"input": "i", "target": "t", "prediction": "p", "correct": False}
    base.update(kw)
    return base


# ------------------------------------------------------------- elision ----

class TestElide:
    def test_short_text_is_untouched(self):
        assert _elide("hello", 100) == "hello"

    def test_exact_budget_is_untouched(self):
        assert _elide("x" * 50, 50) == "x" * 50

    def test_head_keeps_the_start_and_announces_the_cut(self):
        out = _elide("a" * 100, 10, keep="head")
        assert out.startswith("a" * 10)
        assert "90 characters omitted" in out

    def test_tail_keeps_the_end_and_announces_the_cut(self):
        out = _elide("a" * 90 + "TAIL", 4, keep="tail")
        assert out.endswith("TAIL")
        assert "90 characters omitted" in out

    def test_both_keeps_each_end(self):
        out = _elide("HEAD" + "x" * 100 + "TAIL", 8, keep="both")
        assert out.startswith("HEAD") and out.endswith("TAIL")
        assert MARKER in out

    def test_omitted_count_is_exact(self):
        out = _elide("z" * 1000, 300)
        assert "700 characters omitted" in out

    def test_none_is_tolerated(self):
        assert _elide(None, 10) == ""


# ------------------------------------------------ code target rendering ----

class TestExpectedOnCodeTasks:
    def test_humaneval_blob_is_never_shown_raw(self):
        """The D1 defect: EXPECTED was a serialized scoring record."""
        out = render_failures([_failure(target=HUMANEVAL_TARGET)])
        assert '"entry_point"' not in out
        assert '"test":' not in out
        assert "METADATA" not in out

    def test_humaneval_renders_the_assertions(self):
        out = render_failures([_failure(target=HUMANEVAL_TARGET)])
        assert "assert candidate('abcde') == 2" in out
        assert "assert candidate('ACEDY') == 3" in out

    def test_humaneval_names_the_entry_point(self):
        out = render_failures([_failure(target=HUMANEVAL_TARGET)])
        assert "vowels_count" in out

    def test_livecodebench_renders_io_pairs(self):
        out = render_failures([_failure(target=LCB_TARGET)])
        assert "'1 2'" in out and "'3'" in out
        assert "test_cases" not in out

    def test_plain_text_target_passes_through(self):
        out = render_failures([_failure(target="42")])
        assert "Expected: 42" in out

    def test_json_scalar_target_is_not_mistaken_for_a_blob(self):
        """`json.loads("42")` succeeds; only dicts get the code treatment."""
        out = render_failures([_failure(target="12345")])
        assert "Expected: 12345" in out


# ------------------------------------------------------------ rendering ----

class TestRenderFailures:
    def test_empty_is_labelled(self):
        assert render_failures([]) == "(none recorded)"

    def test_empty_label_is_overridable(self):
        assert render_failures([], empty="") == ""

    def test_respects_max_failures(self):
        out = render_failures([_failure(input=f"case{i}") for i in range(10)],
                              max_failures=3)
        assert "case2" in out and "case3" not in out

    def test_every_truncated_field_announces_itself(self):
        """The core guarantee: three cuts, three markers, never a silent one."""
        out = render_failures([_failure(
            input="i" * (INPUT_BUDGET + 500),
            target="t" * 5000,
            prediction="p" * (PREDICTION_BUDGET + 500),
        )])
        assert out.count(MARKER) == 3

    def test_untruncated_fields_carry_no_marker(self):
        assert MARKER not in render_failures([_failure()])

    def test_prediction_keeps_its_tail(self):
        """A code prediction's head is echoed boilerplate; the body is at the end."""
        out = render_failures([_failure(
            prediction="def f():\n" + "#pad\n" * 500 + "    return WRONG")])
        assert "return WRONG" in out

    def test_input_keeps_its_head(self):
        out = render_failures([_failure(input="THE TASK IS" + "x" * 5000)])
        assert "THE TASK IS" in out

    def test_budgets_are_an_order_of_magnitude_above_the_old_caps(self):
        """EXP-022 showed 160 chars leaves input/expected/got indistinguishable."""
        assert INPUT_BUDGET >= 800 and PREDICTION_BUDGET >= 800


class TestSurfaceForms:
    def test_numbered_form_for_the_pact_contract(self):
        out = render_failures([_failure()], numbered=True)
        assert out.startswith("1. INPUT: ")
        assert "EXPECTED: " in out and "GOT: " in out

    def test_bullet_form_is_the_default(self):
        out = render_failures([_failure()])
        assert out.startswith("- Input: ")
        assert "  Got: " in out

    def test_prediction_label_is_overridable(self):
        out = render_failures([_failure()], prediction_label="Model output")
        assert "  Model output: p" in out

    def test_correct_flag_is_optional(self):
        assert "Correct:" not in render_failures([_failure()])
        assert "Correct: False" in render_failures([_failure()], show_correct=True)

    def test_multiple_failures_are_separated(self):
        out = render_failures([_failure(input="A"), _failure(input="B")])
        assert "- Input: A" in out and "- Input: B" in out
