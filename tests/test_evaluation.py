"""Tests for evaluation and scoring."""
from pof.evaluation.scoring import (
    create_score_function,
    _extract_number,
    _extract_choice,
    _extract_boolean,
    _normalize_text,
)


class TestScoring:
    def test_math_exact(self):
        fn = create_score_function("math")
        assert fn("42", "42") == 1
        assert fn("3.14", "3.14") == 1
        assert fn("-5", "-5") == 1

    def test_math_extraction(self):
        fn = create_score_function("math")
        assert fn("The answer is 42", "42") == 1
        assert fn("result: 3.14", "3.14") == 1
        assert fn("#### 7", "7") == 1

    def test_math_wrong(self):
        fn = create_score_function("math")
        assert fn("43", "42") == 0
        assert fn("The answer is 5", "10") == 0

    def test_mcq_exact(self):
        fn = create_score_function("mcq")
        assert fn("A", "A") == 1
        assert fn("B", "B") == 1
        assert fn("A", "B") == 0

    def test_mcq_extraction(self):
        fn = create_score_function("mcq")
        assert fn("The answer is B", "B") == 1
        assert fn("(C)", "C") == 1
        assert fn("choice: D", "D") == 1

    def test_boolean(self):
        fn = create_score_function("boolean")
        assert fn("yes", "yes") == 1
        assert fn("true", "true") == 1
        assert fn("no", "no") == 1
        assert fn("yes", "no") == 0
        assert fn("True", "true") == 1

    def test_text_normalization(self):
        fn = create_score_function("text")
        # Case and whitespace are normalized...
        assert fn("Hello World", "hello world") == 1
        assert fn("  spaces  ", "spaces") == 1
        assert fn("UPPER", "upper") == 1
        # ...but punctuation is NOT stripped -- see _normalize_text's
        # docstring: doing so would collapse bracket-sequence tasks (e.g.
        # BBH's dyck_languages) to an empty string.
        assert fn("Hello World!", "hello world") == 0

    def test_auto_detection(self):
        fn = create_score_function("auto")
        # Boolean
        assert fn("yes", "yes") == 1
        assert fn("no", "yes") == 0
        # MCQ
        assert fn("A", "A") == 1
        # Numeric
        assert fn("42", "42") == 1


class TestExtractors:
    def test_extract_number(self):
        assert _extract_number("42") == 42.0
        assert _extract_number("3.14") == 3.14
        assert _extract_number("-5") == -5.0
        assert _extract_number("The answer is 7") == 7.0
        assert _extract_number("#### 100") == 100.0
        assert _extract_number("no number here") is None

    def test_extract_choice(self):
        assert _extract_choice("A") == "A"
        assert _extract_choice("(B)") == "B"
        assert _extract_choice("The answer is C") == "C"
        assert _extract_choice("random text") is None

    def test_extract_boolean(self):
        assert _extract_boolean("yes") is True
        assert _extract_boolean("no") is False
        assert _extract_boolean("true") is True
        assert _extract_boolean("false") is False
        assert _extract_boolean("maybe") is None

    def test_normalize_text(self):
        # Lowercase + whitespace collapse only -- punctuation is preserved
        # on purpose (stripping it would collapse bracket-sequence answers,
        # e.g. BBH's dyck_languages, to an empty string).
        assert _normalize_text("Hello, World!") == "hello, world!"
        assert _normalize_text("  Multiple   Spaces  ") == "multiple spaces"
        assert _normalize_text("") == ""
        assert _normalize_text("[ { ( ) } ]") == "[ { ( ) } ]"