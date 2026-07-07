"""Score functions — answer extraction and comparison logic.

Ported from Projeto's TaskDataset with robust extraction for:
- Mathematical reasoning (numeric extraction)
- Multiple choice (A/B/C/D)
- Boolean (yes/no, true/false)
- Free-form text matching
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional

# Type alias for score functions
ScoreFunction = Callable[[str, str], int]


def create_score_function(task_type: str = "auto") -> ScoreFunction:
    """Create a task-appropriate score function.

    Args:
        task_type: One of 'math', 'mcq', 'boolean', 'text', 'auto'.
            'auto' tries all extractors in order.

    Returns:
        A function (prediction, target) -> 0 or 1.
    """
    if task_type == "math":
        return _score_math
    elif task_type == "mcq":
        return _score_mcq
    elif task_type == "boolean":
        return _score_boolean
    elif task_type == "text":
        return _score_text
    else:
        return _score_auto


def _score_auto(prediction: str, target: str) -> int:
    """Auto-detect task type and score accordingly."""
    target_clean = target.strip().lower()

    # Try boolean first
    if target_clean in ("true", "false", "yes", "no"):
        return _score_boolean(prediction, target)

    # Try MCQ (single letter)
    if len(target_clean) == 1 and target_clean.isalpha():
        return _score_mcq(prediction, target)

    # Try numeric
    target_num = _extract_number(target)
    if target_num is not None:
        return _score_math(prediction, target)

    # Fallback to text matching
    return _score_text(prediction, target)


def _score_math(prediction: str, target: str) -> int:
    """Score mathematical/numeric answers."""
    pred_num = _extract_number(prediction)
    target_num = _extract_number(target)

    if pred_num is None or target_num is None:
        # Fallback: string comparison
        return 1 if _normalize_text(prediction) == _normalize_text(target) else 0

    # Allow small floating point tolerance
    if abs(pred_num - target_num) < 1e-6:
        return 1
    # Integer comparison
    if pred_num == target_num:
        return 1
    return 0


def _score_mcq(prediction: str, target: str) -> int:
    """Score multiple choice answers (A/B/C/D/E)."""
    pred_choice = _extract_choice(prediction)
    target_choice = _extract_choice(target)

    if pred_choice and target_choice:
        return 1 if pred_choice == target_choice else 0

    # Fallback: normalized text comparison
    return 1 if _normalize_text(prediction) == _normalize_text(target) else 0


def _score_boolean(prediction: str, target: str) -> int:
    """Score boolean answers (yes/no, true/false)."""
    pred_bool = _extract_boolean(prediction)
    target_bool = _extract_boolean(target)

    if pred_bool is not None and target_bool is not None:
        return 1 if pred_bool == target_bool else 0

    return 1 if _normalize_text(prediction) == _normalize_text(target) else 0


def _score_text(prediction: str, target: str) -> int:
    """Score free-form text with normalization."""
    return 1 if _normalize_text(prediction) == _normalize_text(target) else 0


# --- Extraction helpers ---


def _extract_number(text: str) -> Optional[float]:
    """Extract a numeric value from text."""
    if not text:
        return None

    text = text.strip()

    # Direct parse attempt
    try:
        return float(text)
    except ValueError:
        pass

    # Common patterns: "the answer is 42", "result: 3.14", "= -5"
    patterns = [
        r"(?:answer|result|solution|output)\s*(?:is|=|:)\s*(-?\d+\.?\d*)",
        r"####\s*(-?\d+\.?\d*)",
        r"\\boxed\{(-?\d+\.?\d*)\}",
        r"=\s*(-?\d+\.?\d*)\s*$",
        r"(-?\d+\.?\d*)\s*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue

    return None


def _extract_choice(text: str) -> Optional[str]:
    """Extract a multiple choice letter (A-E) from text."""
    if not text:
        return None

    text = text.strip()

    # Direct single letter
    if len(text) == 1 and text.upper() in "ABCDE":
        return text.upper()

    # Patterns: "(A)", "A)", "A.", "answer: A", "The answer is B"
    patterns = [
        r"(?:answer|choice)\s*(?:is|:)\s*\(?([A-Ea-e])\)?",
        r"^\s*\(?([A-Ea-e])\)?\s*[.\s]",
        r"\(([A-Ea-e])\)",
        r"([A-Ea-e])\s*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()

    return None


def _extract_boolean(text: str) -> Optional[bool]:
    """Extract a boolean value from text."""
    if not text:
        return None

    text = text.strip().lower()

    # Direct matches
    true_values = {"true", "yes", "correct", "valid", "1"}
    false_values = {"false", "no", "incorrect", "invalid", "0"}

    if text in true_values:
        return True
    if text in false_values:
        return False

    # Pattern: "the answer is yes/no"
    match = re.search(
        r"(?:answer|result)\s*(?:is|:)\s*(yes|no|true|false)",
        text,
        re.IGNORECASE,
    )
    if match:
        val = match.group(1).lower()
        return val in ("yes", "true")

    return None


def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    # Lowercase, strip, collapse whitespace, remove punctuation
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()