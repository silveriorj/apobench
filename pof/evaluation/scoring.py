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


def _extract_answer_is(text: str) -> Optional[str]:
    """Extract final answer from common answer-marker patterns.

    Matches (in priority order):
    - 'The answer is X' / 'the answer is X'
    - 'Answer: X' (LiveBench olympiad format)
    - 'answer: X' (case-insensitive variant)
    """
    text = text.strip()
    m = re.search(r'[Tt]he answer is\s+(.+?)(?:\s*[.\n]|$)', text)
    if m:
        return m.group(1).strip()
    # LiveBench olympiad: "Answer: 1,6,7,2,3,4,5"
    m = re.search(r'(?:^|\n)Answer:\s*(.+?)(?:\s*\n|$)', text)
    if m:
        return m.group(1).strip()
    return None


def _score_math(prediction: str, target: str) -> int:
    """Score mathematical answers (final_em style).

    Extraction priority: \\boxed{} → 'The answer is X' → JSON answer field.
    Comparison: normalized exact match, then sympy equivalence.
    """
    pred_final = _extract_boxed(prediction) or _extract_answer_is(prediction)
    if pred_final is None:
        json_match = re.search(r'"answer"\s*:\s*"([^"]*)"', prediction)
        pred_final = json_match.group(1) if json_match else None
    target_final = _extract_boxed(target) or _extract_answer_is(target) or target

    # Exact match on the extracted final answer (LaTeX-normalized)
    if pred_final is not None:
        if _normalize_math(pred_final) == _normalize_math(target_final):
            return 1
        # Mathematical equivalence (LiveBench-style leniency): 0.5 == \frac{1}{2}
        if _math_equiv(pred_final, target_final):
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

    A second target shape (LiveCodeBench-style, 'test_cases' key present
    instead of 'test'/'entry_point') is dispatched to _score_livecodebench
    -- kept as one task_type ("code") rather than a second one so no other
    layer (runner.py, evaluator.py's SYSTEM_PROMPT_BY_TASK_TYPE) needs to
    know about the new dataset.
    """
    import json as _json
    import subprocess
    import sys as _sys

    try:
        meta = _json.loads(target)
    except (ValueError, TypeError):
        return _score_text(prediction, target)

    if "test_cases" in meta:
        return _score_livecodebench(prediction, meta)

    try:
        test_code = meta["test"]
        entry_point = meta["entry_point"]
        problem_prompt = meta.get("prompt", "")
    except (KeyError, TypeError):
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
            # Matches inspect_evals' HumanEval VERIFY_TIMEOUT (30s), not an
            # arbitrary choice -- this project's prior 15s cap risked false
            # negatives on correct-but-slower canonical solutions for
            # reasons unrelated to model/prompt quality.
            timeout=30,
        )
        return 1 if proc.returncode == 0 else 0
    except (subprocess.TimeoutExpired, OSError):
        return 0


def _score_livecodebench(prediction: str, meta: dict) -> int:
    """Score LiveCodeBench-style problems: run ALL test cases, pass@1 needs
    every one to pass (matches LiveCodeBench's own all-or-nothing scoring).

    `meta['test_cases']` is a list of {'input', 'output', 'testtype'} dicts.
    'functional' tests call `meta['func_name']` on the (LeetCode-style)
    `Solution` class built from `meta['starter_code']` + the completion,
    passing ast.literal_eval'd args and comparing literal-evaluated return
    values. Any other/missing testtype is treated as stdin/stdout: the
    program is run as a script with `input` piped to stdin, stdout compared
    to `output` as text. Each case runs as its own subprocess (isolation +
    per-case timeout, same rationale as the HumanEval path above).
    """
    import ast as _ast
    import json as _json
    import subprocess
    import sys as _sys

    test_cases = meta.get("test_cases") or []
    if not test_cases:
        return 0

    completion = _extract_code(prediction)
    if not completion.strip():
        return 0

    starter_code = meta.get("starter_code") or ""
    func_name = meta.get("func_name") or ""
    header = (
        "import sys, math, itertools, collections, functools, heapq, bisect, re\n"
        "from typing import List, Dict, Tuple, Optional, Set, Any\n"
        "from collections import defaultdict, Counter, deque\n"
    )

    is_functional = bool(starter_code.strip()) and bool(func_name)

    if is_functional:
        if "class Solution" in completion:
            program_body = completion
        else:
            body = completion if completion.startswith((" ", "\t")) else _indent(completion)
            program_body = starter_code.rstrip("\n") + "\n" + body
        program = header + program_body

        total = passed = 0
        for case in test_cases:
            if case.get("testtype", "functional") != "functional":
                continue
            total += 1
            try:
                args = _ast.literal_eval(case["input"])
            except (ValueError, SyntaxError):
                continue
            if not isinstance(args, tuple):
                args = (args,)
            try:
                expected = _ast.literal_eval(case["output"])
            except (ValueError, SyntaxError):
                expected = case["output"]

            call = (
                "\nimport json as _json\n"
                "_sol = Solution()\n"
                f"_res = _sol.{func_name}(*{args!r})\n"
                "print(_json.dumps(_res))\n"
            )
            try:
                proc = subprocess.run(
                    [_sys.executable, "-c", program + call],
                    capture_output=True, timeout=10, text=True,
                )
                if proc.returncode != 0 or not proc.stdout.strip():
                    continue
                got = _json.loads(proc.stdout.strip().splitlines()[-1])
                if got == expected:
                    passed += 1
            except (subprocess.TimeoutExpired, OSError, ValueError):
                continue
        return 1 if total > 0 and passed == total else 0

    # stdin/stdout style: no starter_code/func_name — completion is a script.
    program = header + completion
    total = passed = 0
    for case in test_cases:
        if case.get("testtype") not in (None, "stdin", "stdio"):
            continue
        total += 1
        try:
            proc = subprocess.run(
                [_sys.executable, "-c", program],
                input=str(case.get("input", "")),
                capture_output=True, timeout=10, text=True,
            )
            if proc.stdout.strip() == str(case.get("output", "")).strip():
                passed += 1
        except (subprocess.TimeoutExpired, OSError):
            continue
    return 1 if total > 0 and passed == total else 0


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


def _latex_to_expr(s: str) -> str:
    """Convert common LaTeX math markup to a sympy-parseable expression."""
    s = s.strip().strip("$")
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"\\[,;! ]", "", s)          # LaTeX spacing commands
    # \sqrt first (turns its braces into parens so nested \frac args match)
    s = re.sub(r"\\sqrt\[([^\]]+)\]\{([^{}]+)\}", r"((\2)**(1/(\1)))", s)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    # \frac{a}{b} → ((a)/(b)) — repeat to unwrap nesting
    frac = re.compile(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}")
    while frac.search(s):
        s = frac.sub(r"((\1)/(\2))", s)
    s = s.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\pi", "pi")
    s = s.replace("^", "**")
    s = re.sub(r"\{([^{}]*)\}", r"(\1)", s)  # remaining braces → parens
    s = re.sub(r"\\([a-zA-Z]+)", r"\1", s)   # strip leftover LaTeX commands
    return s


def _math_equiv(a: str, b: str) -> bool:
    """Check mathematical equivalence via sympy (0.5 == \\frac{1}{2}).

    Mirrors the leniency of LiveBench's official final_em scorer. Numeric
    comparison first (fast, robust); symbolic simplification as fallback.
    Any parse failure returns False (falls back to stricter checks).
    """
    try:
        from sympy.parsing.sympy_parser import (
            parse_expr,
            standard_transformations,
            implicit_multiplication_application,
        )
        import sympy

        transformations = standard_transformations + (
            implicit_multiplication_application,
        )
        ea = parse_expr(_latex_to_expr(a), transformations=transformations)
        eb = parse_expr(_latex_to_expr(b), transformations=transformations)

        # Numeric comparison when both evaluate to a number
        va, vb = ea.evalf(), eb.evalf()
        if va.is_number and vb.is_number:
            return abs(complex(va) - complex(vb)) < 1e-9

        # Symbolic fallback
        return sympy.simplify(ea - eb) == 0
    except Exception:
        return False


def _normalize_math(text: str) -> str:
    """Normalize a LaTeX/math answer for exact-match comparison.

    Removes presentation-only markup (mirroring LiveBench's scorer):
    dollar signs, \\left/\\right, LaTeX spacing (\\, \\; \\!, escaped
    spaces), and all whitespace. Case-insensitive. Answers that don't match
    textually get a second chance via sympy equivalence (_math_equiv).
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

    Uses brace counting, so arbitrarily nested LaTeX is handled
    (e.g. \\boxed{\\frac{\\sqrt{2}}{2}}).
    """
    if not text:
        return None
    result = None
    for match in re.finditer(r"\\boxed\{", text):
        start = match.end()
        depth = 1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    result = text[start:i].strip()
                    break
    return result


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
    #
    # Letters must be followed by a real delimiter or end-of-clause, not bare
    # whitespace: naive patterns here previously misfired on ordinary prose
    # (e.g. the article "A" at a sentence start, or the trailing letter of
    # words like "mat"/"in"/"a"). The lookaheads below guard against that.
    patterns = [
        r"(?:answer|choice)\s*(?:is|:)\s*\(?([A-Za-z])\)?(?=[.,;:!?)]|\s*(?:\n|$))",
        r"^\s*\(([A-Za-z])\)",
        r"^\s*([A-Za-z])[.):]",
        r"\(([A-Za-z])\)",
        r"\b([A-Za-z])\s*$",
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

    # Pattern: \boxed{<answer>} — used by the "thinking" system prompt, which
    # explicitly asks for this format instead of "the answer is X".
    boxed = _extract_boxed(text)
    if boxed:
        return boxed

    # Pattern: "the answer is <answer>" (most common in BBH/GSM8K CoT).
    # Use the LAST match — models sometimes self-correct ("So the answer is X — wait, no!")
    # and the final occurrence is the authoritative one.
    matches = list(re.finditer(
        r"(?:so\s+)?the\s+answer\s+is\s+(.+?)(?:\.|$)",
        text,
        re.IGNORECASE | re.MULTILINE,
    ))
    if matches:
        return matches[-1].group(1).strip()

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

    # Pattern: Dyck/bracket sequence — last line that looks like closing brackets.
    # Catches CoT outputs that end with the answer but lack an "answer is" marker.
    # A valid dyck answer is one or more bracket tokens (], }, ), >, separated by spaces).
    bracket_line = re.search(
        r"^([\]\}\)\>](?:\s+[\]\}\)\>])*)$",
        text.strip(),
        re.MULTILINE,
    )
    if bracket_line:
        return bracket_line.group(1).strip()

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
