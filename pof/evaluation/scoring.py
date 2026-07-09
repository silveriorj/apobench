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
    elif task_type == "code":
        return _score_code
    else:
        return _score_auto


def _score_auto(prediction: str, target: str) -> int:
    """Auto-detect task type and score accordingly."""
    # First, try to extract the final answer from CoT output
    extracted = _extract_cot_answer(prediction)
    if extracted:
        prediction = extracted

    target_clean = target.strip().lower()

    # Try boolean first (includes valid/invalid, e.g. BBH formal_fallacies)
    if target_clean in ("true", "false", "yes", "no", "valid", "invalid"):
        return _score_boolean(prediction, target)

    # Try MCQ: single letter with optional parens — "A", "(A)", "a)"
    if re.fullmatch(r"\(?([A-Za-z])\)?", target_clean):
        return _score_mcq(prediction, target)

    # Try numeric
    target_num = _extract_number(target)
    if target_num is not None:
        return _score_math(prediction, target)

    # Fallback to text matching (JSON answer extracted first if present)
    json_match = re.search(r'"answer"\s*:\s*"([^"]*)"', prediction)
    if json_match:
        prediction = json_match.group(1)
    return _score_text(prediction, target)


def _score_math(prediction: str, target: str) -> int:
    """Score mathematical answers (final_em style, \\boxed{} aware).

    The final answer is extracted from \\boxed{...} (last occurrence, per
    LiveBench/Meta's 0-shot CoT convention) or from a JSON answer field; the
    extracted answer is compared numerically when both sides parse as numbers,
    otherwise by normalized exact match (handles expressions, tuples, letters).
    """
    pred_final = _extract_boxed(prediction)
    if pred_final is None:
        json_match = re.search(r'"answer"\s*:\s*"([^"]*)"', prediction)
        pred_final = json_match.group(1) if json_match else None
    target_final = _extract_boxed(target) or target

    # Exact match on the extracted final answer (LaTeX-normalized)
    if pred_final is not None:
        if _normalize_math(pred_final) == _normalize_math(target_final):
            return 1

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


def _score_code(prediction: str, target: str) -> int:
    """Score code by executing it against unit tests (HumanEval pass@1).

    The target is a JSON blob with 'prompt' (signature + docstring), 'test'
    (the check function), and 'entry_point'. The candidate program is
    assembled and run in a subprocess with a timeout; score 1 iff it exits 0.
    """
    import json as _json
    import subprocess
    import sys as _sys

    try:
        meta = _json.loads(target)
        test_code = meta["test"]
        entry_point = meta["entry_point"]
        problem_prompt = meta.get("prompt", "")
    except (ValueError, KeyError, TypeError):
        # Not a HumanEval-style target — plain text comparison
        return _score_text(prediction, target)

    completion = _extract_code(prediction)
    if not completion.strip():
        return 0

    # Models either return the function body (HumanEval convention) or a
    # full function definition — handle both.
    if f"def {entry_point}" in completion:
        program = completion
        # Keep imports/helpers from the original prompt header
        header = problem_prompt.split(f"def {entry_point}")[0]
        program = header + "\n" + completion
    else:
        body = completion if completion.startswith((" ", "\t")) else _indent(completion)
        program = problem_prompt + body

    program = f"{program}\n\n{test_code}\n\ncheck({entry_point})\n"

    try:
        proc = subprocess.run(
            [_sys.executable, "-c", program],
            capture_output=True,
            timeout=15,
        )
        return 1 if proc.returncode == 0 else 0
    except (subprocess.TimeoutExpired, OSError):
        return 0


def _extract_code(text: str) -> str:
    """Extract code from model output, stripping markdown fences and chatter."""
    if not text:
        return ""
    # Prefer fenced code blocks if present
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    return text


def _indent(code: str, prefix: str = "    ") -> str:
    """Indent every line (turn a flush-left body into a function body)."""
    return "\n".join(prefix + line if line.strip() else line
                     for line in code.split("\n"))


def _normalize_math(text: str) -> str:
    """Normalize a LaTeX/math answer for exact-match comparison.

    Removes presentation-only markup (mirroring LiveBench's scorer):
    dollar signs, \\left/\\right, LaTeX spacing (\\, \\; \\!, escaped
    spaces), and all whitespace. Case-insensitive.
    """
    if not text:
        return ""
    text = text.strip().lower()
    text = text.replace("$", "")
    text = re.sub(r"\\left|\\right", "", text)
    text = re.sub(r"\\[,;! ]", "", text)   # LaTeX spacing commands incl. '\ '
    text = re.sub(r"\s+", "", text)
    return text


def _extract_boxed(text: str) -> Optional[str]:
    """Extract the content of the LAST \\boxed{...} in the text.

    Handles one level of nested braces (e.g. \\boxed{\\frac{1}{2}}).
    """
    if not text:
        return None
    matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    return matches[-1].strip() if matches else None


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
    """Extract a multiple choice letter from text.

    BBH tasks like reasoning_about_colored_objects have >5 options (colors go
    up to letter R or beyond), so we match the full A-Z range rather than A-E.
    JSON format {"answer": "C"} is checked first per Qwen's output recommendation.
    """
    if not text:
        return None

    text = text.strip()

    # Highest priority: JSON format  {"answer": "C"}  or  "answer": "C"
    json_match = re.search(r'"answer"\s*:\s*"([A-Za-z])"', text)
    if json_match:
        return json_match.group(1).upper()

    # Direct single letter
    if len(text) == 1 and text.upper().isalpha():
        return text.upper()

    # Patterns (broadened to A-Z): "(A)", "A)", "A.", "answer: A", "The answer is B"
    patterns = [
        r"(?:answer|choice)\s*(?:is|:)\s*\(?([A-Za-z])\)?",
        r"^\s*\(?([A-Za-z])\)?\s*[.\s]",
        r"\(([A-Za-z])\)",
        r"([A-Za-z])\s*$",
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

    # Highest priority: JSON format  {"answer": "Yes"}
    json_match = re.search(r'"answer"\s*:\s*"(\w+)"', text, re.IGNORECASE)
    if json_match:
        val = json_match.group(1).lower()
        if val in ("yes", "true", "valid", "correct"):
            return True
        if val in ("no", "false", "invalid", "incorrect"):
            return False

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
        r"(?:answer|result)\s*(?:is|:)\s*(yes|no|true|false|valid|invalid)",
        text,
        re.IGNORECASE,
    )
    if match:
        val = match.group(1).lower()
        return val in ("yes", "true", "valid")

    return None


def _extract_cot_answer(text: str) -> Optional[str]:
    """Extract the final answer from Chain-of-Thought output.

    Looks for common CoT answer patterns:
    - "the answer is X"
    - "So the answer is X"
    - "#### X"
    - "Answer: X"
    - Last line after reasoning
    """
    if not text:
        return None

    # Pattern: "the answer is <answer>" (most common in BBH/GSM8K CoT)
    match = re.search(
        r"(?:so\s+)?the\s+answer\s+is\s+(.+?)(?:\.|$)",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    if match:
        return match.group(1).strip()

    # Pattern: "#### <answer>" (GSM8K format)
    match = re.search(r"####\s*(.+?)$", text, re.MULTILINE)
    if match:
        return match.group(1).strip()

    # Pattern: "Answer: <answer>"
    match = re.search(r"Answer\s*:\s*(.+?)(?:\.|$)", text, re.IGNORECASE | re.MULTILINE)
    if match:
        return match.group(1).strip()

    # Pattern: "Therefore, <answer>"
    match = re.search(
        r"(?:therefore|thus|hence|so),?\s+(.+?)(?:\.|$)",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    if match:
        candidate = match.group(1).strip()
        # Only use if it's short (likely an answer, not a sentence)
        if len(candidate) < 100:
            return candidate

    return None


def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase and collapse whitespace only.

    Punctuation is NOT stripped — bracket/symbol sequences (e.g. dyck_languages)
    would all collapse to "" if we removed non-word characters.
    """
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text
