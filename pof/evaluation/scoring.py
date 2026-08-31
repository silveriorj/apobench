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
    # Third-party scorers first. A new task family usually needs its own notion
    # of correctness, and without this the only way to add one was to edit this
    # chain — which meant the benchmark could not evaluate anything its authors
    # had not anticipated.
    from pof.plugins import SCORER_GROUP, discover

    external = discover(SCORER_GROUP)
    if task_type and task_type.lower() in external:
        return external[task_type.lower()]

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

    Two LiveBench-specific paths:
    - math_comp (AMC/AIME): target is N digits; extract trailing N digits from
      the prediction tail (prompt instructs model to end with the number).
    - AMPS_Hard integrals: strip integration constants (+C, +K) before
      sympy comparison — indefinite integrals match up to a constant.
    """
    pred_final = _extract_boxed(prediction) or _extract_answer_is(prediction)
    if pred_final is None:
        json_match = re.search(r'"answer"\s*:\s*"([^"]*)"', prediction)
        pred_final = json_match.group(1) if json_match else None
    target_final = _extract_boxed(target) or _extract_answer_is(target) or target

    target_stripped = target_final.strip()

    # AMC math_comp: target is a single letter (A–E multiple choice).
    # _score_math normally doesn't call _extract_choice; add it here so
    # LiveBench AMC problems don't fall through to string/numeric comparison.
    if re.fullmatch(r"[A-Ea-e]", target_stripped):
        pred_choice = _extract_choice(pred_final or prediction)
        if pred_choice:
            return 1 if pred_choice.upper() == target_stripped.upper() else 0

    # AIME math_comp: target is a zero-padded integer string like "025".
    # Prompt tells model "have the N digits as the last part of the response."
    if re.fullmatch(r"\d{1,4}", target_stripped):
        n = len(target_stripped)
        trailing = _extract_trailing_digits(prediction, n)
        if trailing is not None:
            if trailing == target_stripped:
                return 1
            # Handle leading-zero stripping: "025" == "25" numerically
            try:
                if int(trailing) == int(target_stripped):
                    return 1
            except ValueError:
                pass
        # Also try any trailing number (e.g., model writes "25" not "025")
        m = re.search(r"(\d+)\s*$", prediction.strip())
        if m:
            try:
                if int(m.group(1)) == int(target_stripped):
                    return 1
            except ValueError:
                pass
        # Fall back to the extracted final answer before giving up.
        #
        # This branch was written for LiveBench AIME, whose prompt instructs
        # the model to end its response with the answer's digits -- but its
        # guard (a 1-4 digit target) also captures essentially all of GSM8K.
        # There, a model that writes a well-formed sentence ending in a period
        # ("The answer is 21.") has no trailing digits, so both attempts above
        # miss and the original unconditional `return 0` scored a correct
        # answer as wrong. Measured on GPT-4o/GSM8K: init scores of 0.000 to
        # 0.023 where the model was in fact answering correctly. Qwen3 happened
        # to omit the trailing period, which is why this stayed invisible.
        if pred_final is not None:
            try:
                if int(float(_normalize_math(pred_final))) == int(target_stripped):
                    return 1
            except (ValueError, TypeError):
                pass
        return 0

    # Exact match on the extracted final answer (LaTeX-normalized)
    if pred_final is not None:
        # For AMPS_Hard integrals: strip indefinite-integral constant (+C/+K)
        # before comparison. The constant is implicit in the target.
        pred_clean = _strip_integration_constant(pred_final)
        if _normalize_math(pred_clean) == _normalize_math(target_final):
            return 1
        if _math_equiv(pred_clean, target_final):
            return 1
        # Try the original (in case constant is genuinely part of the answer)
        if _normalize_math(pred_final) == _normalize_math(target_final):
            return 1
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

    SECURITY: this runs LLM-generated code locally, unsandboxed, as the calling
    user (`subprocess.run([sys.executable, "-c", program], timeout=30)`).
    Standard practice for HumanEval-style harnesses, but a caller in an
    untrusted setting (an unknown model, an adversarial prompt source) should
    run the whole benchmark inside a container or VM rather than expect this
    function to contain anything.

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
    task_format = meta.get("task_format", "generation")
    partial_solution = meta.get("partial_solution") or ""
    header = (
        "import sys, math, itertools, collections, functools, heapq, bisect, re\n"
        "from typing import List, Dict, Tuple, Optional, Set, Any\n"
        "from collections import defaultdict, Counter, deque\n"
    )

    is_functional = bool(starter_code.strip()) and bool(func_name)

    if is_functional:
        if "class Solution" in completion:
            program_body = completion
        elif task_format == "completion" and partial_solution:
            # The prompt shows partial_solution (far more than the bare
            # `starter_code` signature -- often several lines into the
            # method body) and tells the model to write ONLY what comes
            # next, to be appended directly. No re-indent, no assumption
            # about where a line boundary falls: the model's completion
            # picks up exactly where partial_solution's last character
            # left off, same as the prompt instructs.
            program_body = partial_solution + completion
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

            # Args go over stdin, pickled+base64, rather than embedded via
            # repr() into the -c script text: some LiveCodeBench test cases
            # carry arrays up to 1e5 elements, and repr()'ing that directly
            # into the command line hit Windows' ~32KB CreateProcess limit
            # (FileNotFoundError -- silently swallowed by the except clause
            # below, since OSError covers it, so every such case counted as
            # "failed" with no visible error). pickle, not json, because args
            # can contain tuples LeetCode-style problems use for coordinates,
            # which json can't round-trip.
            import base64 as _b64
            import pickle as _pickle
            call = (
                "\nimport sys, pickle as _pickle, base64 as _b64, json as _json\n"
                "_args = _pickle.loads(_b64.b64decode(sys.stdin.read()))\n"
                "_sol = Solution()\n"
                f"_res = _sol.{func_name}(*_args)\n"
                "print(_json.dumps(_res))\n"
            )
            try:
                payload = _b64.b64encode(_pickle.dumps(args)).decode("ascii")
                proc = subprocess.run(
                    [_sys.executable, "-c", program + call],
                    input=payload,
                    capture_output=True, timeout=10, text=True,
                )
                if proc.returncode != 0 or not proc.stdout.strip():
                    continue
                got = _json.loads(proc.stdout.strip().splitlines()[-1])
                if got == expected:
                    passed += 1
            except (subprocess.TimeoutExpired, OSError, ValueError, _pickle.PicklingError):
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


def _strip_integration_constant(expr: str) -> str:
    """Remove a trailing integration constant (+C, +K, etc.) from a LaTeX expr."""
    return re.sub(r"\s*\+\s*[CcKk]\s*$", "", expr).strip()


def _extract_trailing_digits(text: str, n: int) -> Optional[str]:
    """Extract the last n-digit sequence from the tail of `text`.

    Used for math_comp (AMC/AIME) where the prompt instructs the model to
    place the N-digit answer at the very end of its response.
    """
    m = re.search(r"(\d{" + str(n) + r"})\s*$", text.strip())
    return m.group(1) if m else None


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
    """Extract a numeric value from text.

    Mirrors lm-eval's NumberParseRegexFilter: digit regex first, then
    word2number fallback ("five" → 5, "twelve" → 12) for object_counting
    and similar tasks where the model writes out the number as a word.
    """
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

    # Currency/comma fallback: "$1,234" → 1234, "1,200 dollars" → 1200.
    # Mirrors lm-eval GSM8K regexes_to_ignore (strip '$' and ',').
    # Matches the LAST currency-or-comma-number in the text so "She had $3
    # left after spending $1,200" returns 1200, not 3.
    currency_matches = list(re.finditer(r"[$€£¥]?\d[\d,]*\.?\d*", text))
    if currency_matches:
        raw = currency_matches[-1].group(0)
        cleaned = re.sub(r"[$€£¥]", "", raw).replace(",", "")
        try:
            return float(cleaned)
        except ValueError:
            pass

    # Word-number fallback: "five" → 5.0, "twelve" → 12.0.
    # Covers object_counting where models write out the count as a word.
    try:
        from word2number import w2n
        import regex as _regex
        _english_num_re = _regex.compile(
            r"((?:(?:zero|one|two|three|four|five|(?:twen|thir|for|fif|six|seven|nine)"
            r"(?:|teen|ty)|eight(?:|een|y)|ten|eleven|twelve|fourteen|hundred|thousand|"
            r"(?:m|b|tr)illion)(?:zero|one|two|three|four|five|(?:twen|thir|for|fif|six|"
            r"seven|nine)(?:|teen|ty)|eight(?:|een|y)|ten|eleven|twelve|fourteen|hundred|"
            r"thousand|(?:m|b|tr)illion|[^\S\r\n]|,|and|&)+)?(?:zero|one|two|three|four|"
            r"five|(?:twen|thir|for|fif|six|seven|nine)(?:|teen|ty)|eight(?:|een|y)|ten|"
            r"eleven|twelve|fourteen|hundred|thousand|(?:m|b|tr)illion))",
            _regex.IGNORECASE,
        )
        m = _english_num_re.search(text.lower())
        if m:
            return float(w2n.word_to_num(m.group(0)))
    except Exception:
        pass

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

    # Pattern: "the answer is yes/no" or "answer: valid" etc.
    match = re.search(
        r"(?:answer|result)\s*(?:is|:)\s*(yes|no|true|false|valid|invalid)",
        text,
        re.IGNORECASE,
    )
    if match:
        val = match.group(1).lower()
        return val in ("yes", "true", "valid")

    # web_of_lies phrases (lm-eval MapRegexFilter mapping).
    # Must run before the generic tail scan so "tells the truth" is not absorbed
    # by the "true" keyword match at the wrong position.
    for phrase, is_true in (
        ("does not tell the truth", False),
        ("is not telling the truth", False),
        ("tells the truth", True),
        ("is telling the truth", True),
    ):
        if phrase in text:
            return is_true

    # Tail fallback: at 2048 tokens BBH CoT sometimes concludes with phrasing
    # like "the argument is valid" or "this is false" without the standard
    # "the answer is X" marker. Scan the last 200 chars for a keyword.
    # Check "invalid" before "valid" since "invalid" contains "valid" as a substr.
    tail = text[-200:].lower()
    for word, is_true in (
        ("invalid", False), ("valid", True),
        ("false", False), ("true", True),
        ("no", False), ("yes", True),
    ):
        if re.search(r"\b" + word + r"\b", tail):
            return is_true

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

    # Pattern: <solution>ANSWER</solution> — LiveBench's primary extraction format.
    # Check before boxed: LiveBench wraps the final answer in these tags.
    m = re.search(r"<solution>\s*(.+?)\s*</solution>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

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

    # Pattern: Dyck/bracket sequence — mirrors lm-eval's dyck_languages extractor.
    # lm-eval uses `(?<= )([" \[\(<{}>)\]]+)` which grabs any sequence of
    # bracket/quote characters preceded by a space anywhere in the response.
    # This is broader than requiring a dedicated bracket-only line at the end,
    # and correctly handles "The answer is ] ) }" style outputs.
    bracket_match = re.search(r"(?<= )([\"\s\[\]()<>{}\\\|]+)", text)
    if bracket_match:
        candidate = bracket_match.group(1).strip()
        # Only return if it looks like a bracket sequence (has at least one bracket char)
        if re.search(r"[\[\]()<>{}]", candidate):
            return candidate

    return None


def _normalize_text(text: str) -> str:
    """Normalize text for comparison.

    Mirrors lm-eval's BBH `regexes_to_ignore`: strips trailing period and comma,
    then lowercases and collapses whitespace.  Bracket/symbol sequences
    (dyck_languages) survive because we only strip trailing punctuation, not
    all non-word characters.
    """
    if not text:
        return ""
    text = text.strip()
    # Strip trailing period, comma, backslash, quote — lm-eval regexes_to_ignore.
    text = re.sub(r'[.,\\"]+$', "", text)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text
