"""Additional operators for FUNNELv2, drawn from methods outside the original
six-method pool (see `_funnel_techniques.py` for that pool).

Each operator below is distilled from a specific published method and is
GENUINELY distinct from the mechanisms already in the v1 pool — this module
deliberately does not re-wrap paraphrase/crossover/failure-rewrite under new
names. Sources, and what each contributes that v1 lacks:

- **ETGPO** (error-taxonomy-guided prompt optimization): collects failures,
  builds a *frequency-ranked taxonomy* of failure modes, then writes guidance
  targeting the most prevalent categories in a single pass. v1's
  `structured_failure_guided` reacts to a handful of raw failures; ETGPO
  categorizes first and attacks the modal category. Reported to use roughly a
  third the token budget of iterative failure-driven methods.

- **StraGo** (Wu et al., EMNLP 2024 Findings): derives strategy from BOTH
  correct and incorrect predictions, explicitly to counter "prompt drifting" —
  where fixing failures degrades cases that already worked. Every feedback
  operator in the v1 pool looks only at failures, so the signal in what the
  prompt already gets right is discarded. This is directly relevant to the
  dev→test generalization gap measured on swift_v2/apex_v2 in this project.

- **AutoHint**: aggregates hints across many wrong answers into one
  generalizable hint, rather than rewriting per-failure. One summarization
  call instead of N rewrite calls.

- **GrIPS** (Prasad et al., EACL 2023): phrase-level edit search with four
  operations — delete, add, swap, paraphrase. Delete and swap are pure string
  operations requiring ZERO LLM calls, which is why they are valuable under a
  tight call budget: they buy exploration for free.

- **AMPO**: conditional if-then-else branching, so one prompt can carry
  case-specific handling. Suits tasks whose failures split into distinct
  sub-cases (logical deduction, colored objects) rather than one uniform
  weakness.

- **UniPrompt** (Juneja et al., ACL 2025 Findings) + **GSPE** (this project's
  own optimizer, `gspe.py`): both decompose a prompt into semantically
  independent sections. GSPE contributes the concrete 5-field grammar
  (preamble / task_definition / output_spec / reasoning_guide /
  error_prevention) and per-field mutation, but scores only whole prompts —
  it has NO per-component credit assignment. UniPrompt contributes exactly
  that missing half: feedback aggregated per facet, proposing add/edit/delete
  at section granularity. `t_facet_edit` below is the combination.

- **CRISPO** (Zhang et al., 2024): critiques failures along several NAMED
  aspects independently, rather than one holistic pass. `strago_dual`
  already reads both correct and incorrect cases, but produces a single
  blended 3-5 sentence strategy — a precise, fixable issue about (say)
  output-format compliance can get buried under general commentary about
  reasoning approach. `t_multi_aspect_critique` below scores each aspect
  separately first, then rewrites addressing only the aspects that actually
  flagged a problem, which is what CRISPO's ablations report as the source
  of its improvement over single-pass critique.

Every function has signature `(opt) -> Optional[str]` matching the v1 pool,
so both pools compose into one UCB1 arm set.
"""
from __future__ import annotations

import random
import re
from typing import Callable, Dict, List, Optional

from pof.optimizers.base import (
    _CRITIQUE_SYSTEM_PROMPT,
    _GENERATE_SYSTEM_PROMPT,
    _IMPROVE_SYSTEM_PROMPT,
)
from pof.optimizers._funnel_techniques import _ensure_details, _pick_elite, _split_sentences

# GSPE's prompt grammar (gspe.py PROMPT_GRAMMAR), reused so FUNNELv2's
# decomposition is directly comparable to GSPE's rather than a parallel
# invention. Descriptions kept verbatim; the token hints are advisory.
_FACETS: Dict[str, str] = {
    "preamble": "Context-setting opening (role, expertise, context)",
    "task_definition": "Clear statement of what to do",
    "output_spec": "Expected output format and constraints",
    "reasoning_guide": "Step-by-step reasoning instructions",
    "error_prevention": "Common mistakes to avoid",
}


def _format_cases(cases: List[Dict], n: int = 6, label_correct: bool = False) -> str:
    lines = []
    for c in cases[:n]:
        line = (
            f"- Input: {str(c.get('input', ''))[:100]}\n"
            f"  Expected: {c.get('target', '')}\n"
            f"  Model output: {str(c.get('prediction', ''))[:100]}"
        )
        if label_correct:
            line += f"\n  Correct: {c.get('correct')}"
        lines.append(line)
    return "\n".join(lines)


# --- ETGPO: error-taxonomy-guided ---

def t_etgpo_taxonomy(opt) -> Optional[str]:
    """Build a frequency-ranked failure taxonomy, then target the modal category.

    Two calls: one to categorize, one to rewrite. Distinct from
    `structured_failure_guided`, which rewrites directly off raw failures
    without ever forming categories.
    """
    record = _pick_elite(opt)
    if not record:
        return None
    details = _ensure_details(opt, record)
    failures = [d for d in details if not d.get("correct")] if details else []
    if not failures:
        return None

    taxonomy_prompt = (
        "Below are cases where an instruction produced wrong answers. "
        "Group them into distinct error CATEGORIES. For each category give: "
        "a short name, what goes wrong, why it produces a wrong answer, and "
        "how many of the listed cases it covers. Order categories by how many "
        "cases they cover, most frequent first.\n\n"
        f"Failing cases:\n{_format_cases(failures, n=10)}\n\n"
        "Error taxonomy:"
    )
    taxonomy = opt._generate_prompt(
        taxonomy_prompt, temperature=0.4, system_prompt=_CRITIQUE_SYSTEM_PROMPT
    )
    if not taxonomy.strip():
        return None

    rewrite_prompt = (
        "Rewrite the instruction so it closes the MOST FREQUENT error "
        "categories in the taxonomy below. Add explicit rules that prevent "
        "those specific errors. Keep everything that already works. Output "
        "only the new instruction.\n\n"
        f"Current instruction:\n{record.text}\n\n"
        f"Error taxonomy:\n{taxonomy.strip()}\n\n"
        "Improved instruction:"
    )
    result = opt._generate_prompt(
        rewrite_prompt, temperature=0.7, system_prompt=_IMPROVE_SYSTEM_PROMPT
    )
    return result.strip() or None


# --- StraGo: success-anchored dual feedback ---

def t_strago_dual(opt) -> Optional[str]:
    """Derive strategy from correct AND incorrect cases (anti-drift).

    The only operator in either pool that reads successful predictions.
    Its explicit goal is to stop a fix for failures from degrading cases
    the prompt already handles.
    """
    record = _pick_elite(opt)
    if not record:
        return None
    details = _ensure_details(opt, record)
    if not details:
        return None
    correct = [d for d in details if d.get("correct")]
    failures = [d for d in details if not d.get("correct")]
    if not correct or not failures:
        return None

    strategy_prompt = (
        "An instruction succeeds on some inputs and fails on others. "
        "Compare the two groups. State (a) what the instruction is doing "
        "RIGHT that must be preserved, and (b) what specifically to change "
        "to fix the failures WITHOUT breaking the successes. Be concrete and "
        "actionable, 3-5 sentences.\n\n"
        f"SUCCESSFUL cases:\n{_format_cases(correct, n=5)}\n\n"
        f"FAILING cases:\n{_format_cases(failures, n=5)}\n\n"
        "Strategy:"
    )
    strategy = opt._generate_prompt(
        strategy_prompt, temperature=0.5, system_prompt=_CRITIQUE_SYSTEM_PROMPT
    )
    if not strategy.strip():
        return None

    rewrite_prompt = (
        "Apply this strategy to the instruction. Preserve what the strategy "
        "says is working; change only what it says to change. Output only "
        "the new instruction.\n\n"
        f"Current instruction:\n{record.text}\n\n"
        f"Strategy:\n{strategy.strip()}\n\n"
        "Improved instruction:"
    )
    result = opt._generate_prompt(
        rewrite_prompt, temperature=0.7, system_prompt=_IMPROVE_SYSTEM_PROMPT
    )
    return result.strip() or None


# --- CRISPO: multi-aspect critique ---

# Named independently so the critique step can't blend a precise, fixable
# issue on one axis into vague commentary about another. Chosen for
# short-answer BBH-style tasks specifically, not CRISPO's original
# style/precision/content-alignment triad (built for longer generative
# outputs where "style" is meaningful; not here).
_CRITIQUE_ASPECTS: List[str] = [
    "answer_format",       # does the instruction make the required output format unambiguous?
    "reasoning_approach",  # does it point at the right method/strategy for the task?
    "edge_case_handling",  # does it address the specific edge cases these failures show?
    "instruction_clarity", # is any part of it ambiguous or missing necessary detail?
]


def t_multi_aspect_critique(opt) -> Optional[str]:
    """Critique failures along named aspects independently, then rewrite.

    One structured critique call (all aspects at once, each scored
    independently) plus one rewrite call -- same 2-call cost as
    `strago_dual`, but the critique can't let a precise fixable issue on one
    aspect get buried under general commentary about another, which is what
    CRISPO's ablations report as the source of its improvement over
    single-pass critique.
    """
    record = _pick_elite(opt)
    if not record:
        return None
    details = _ensure_details(opt, record)
    failures = [d for d in details if not d.get("correct")] if details else []
    if not failures:
        return None

    aspect_list = "\n".join(f"- {a}" for a in _CRITIQUE_ASPECTS)
    critique_prompt = (
        "Below are cases where an instruction produced wrong answers. "
        "Evaluate the instruction against EACH of these aspects independently:\n"
        f"{aspect_list}\n\n"
        "For each aspect, either write 'N/A' (this aspect is not the "
        "problem) or a single specific, actionable sentence describing "
        "exactly what to change. Do not blend aspects together — keep each "
        "one's issue (or N/A) separate and attributed to its own aspect "
        "name.\n\n"
        f"Failing cases:\n{_format_cases(failures, n=8)}\n\n"
        "Per-aspect critique:"
    )
    critique = opt._generate_prompt(
        critique_prompt, temperature=0.4, system_prompt=_CRITIQUE_SYSTEM_PROMPT
    )
    if not critique.strip():
        return None

    rewrite_prompt = (
        "Rewrite the instruction to fix ONLY the aspects below that flagged "
        "a real problem (skip any marked N/A). Address each flagged aspect "
        "with a concrete change; do not touch parts of the instruction the "
        "critique didn't flag. Output only the new instruction.\n\n"
        f"Current instruction:\n{record.text}\n\n"
        f"Per-aspect critique:\n{critique.strip()}\n\n"
        "Improved instruction:"
    )
    result = opt._generate_prompt(
        rewrite_prompt, temperature=0.7, system_prompt=_IMPROVE_SYSTEM_PROMPT
    )
    return result.strip() or None


# --- AutoHint: aggregated hint synthesis ---

def t_autohint(opt) -> Optional[str]:
    """Summarize all failures into ONE generalizable hint, then append it."""
    record = _pick_elite(opt)
    if not record:
        return None
    details = _ensure_details(opt, record)
    failures = [d for d in details if not d.get("correct")] if details else []
    if not failures:
        return None

    hint_prompt = (
        "Read all these failing cases together and write ONE short, general "
        "hint (1-2 sentences) that would help avoid this whole class of "
        "mistakes. Do not describe individual cases; generalize across them.\n\n"
        f"Failing cases:\n{_format_cases(failures, n=10)}\n\n"
        "Hint:"
    )
    hint = opt._generate_prompt(
        hint_prompt, temperature=0.6, system_prompt=_CRITIQUE_SYSTEM_PROMPT
    )
    hint = hint.strip()
    if not hint:
        return None
    return f"{record.text.rstrip()}\n\nHint: {hint}"


# --- GrIPS: phrase-level edits (delete/swap are zero-LLM-call) ---

def t_grips_delete(opt) -> Optional[str]:
    """Delete one phrase. ZERO LLM calls."""
    record = _pick_elite(opt)
    if not record:
        return None
    sents = _split_sentences(record.text)
    if len(sents) < 2:
        return None
    idx = random.randrange(len(sents))
    kept = sents[:idx] + sents[idx + 1:]
    return " ".join(kept).strip() or None


def t_grips_swap(opt) -> Optional[str]:
    """Swap two phrases. ZERO LLM calls."""
    record = _pick_elite(opt)
    if not record:
        return None
    sents = _split_sentences(record.text)
    if len(sents) < 2:
        return None
    i, j = random.sample(range(len(sents)), 2)
    sents[i], sents[j] = sents[j], sents[i]
    return " ".join(sents).strip() or None


def t_grips_add(opt) -> Optional[str]:
    """Add one clarifying phrase at a random position. One LLM call."""
    record = _pick_elite(opt)
    if not record:
        return None
    meta_prompt = (
        "Write ONE short additional sentence that would make this instruction "
        "clearer or more precise. Output only that single sentence.\n\n"
        f"Instruction:\n{record.text}\n\n"
        "Additional sentence:"
    )
    addition = opt._generate_prompt(
        meta_prompt, temperature=0.8, system_prompt=_GENERATE_SYSTEM_PROMPT
    ).strip()
    if not addition:
        return None
    sents = _split_sentences(record.text)
    pos = random.randint(0, len(sents))
    sents.insert(pos, addition)
    return " ".join(sents).strip() or None


def t_grips_paraphrase(opt) -> Optional[str]:
    """Paraphrase ONE phrase in place, leaving the rest byte-identical.

    Distinct from whole-prompt `semantic_var`: the edit is localized, so the
    search step is much smaller.
    """
    record = _pick_elite(opt)
    if not record:
        return None
    sents = _split_sentences(record.text)
    if not sents:
        return None
    idx = random.randrange(len(sents))
    meta_prompt = (
        "Rewrite this single sentence to be clearer, preserving its exact "
        "meaning. Output only the rewritten sentence.\n\n"
        f"Sentence:\n{sents[idx]}\n\n"
        "Rewritten:"
    )
    new_sent = opt._generate_prompt(
        meta_prompt, temperature=0.8, system_prompt=_GENERATE_SYSTEM_PROMPT
    ).strip()
    if not new_sent:
        return None
    sents[idx] = new_sent
    return " ".join(sents).strip() or None


# --- AMPO: conditional if-then-else branching ---

def t_ampo_branch(opt) -> Optional[str]:
    """Add a case-specific conditional branch for a recurring failure mode.

    Targets tasks whose errors split into distinct sub-cases rather than one
    uniform weakness — e.g. logical deduction, colored objects.
    """
    record = _pick_elite(opt)
    if not record:
        return None
    details = _ensure_details(opt, record)
    failures = [d for d in details if not d.get("correct")] if details else []
    if not failures:
        return None

    meta_prompt = (
        "The instruction below fails on a recognizable SUBSET of inputs. "
        "Identify what distinguishes those inputs, then rewrite the "
        "instruction to include an explicit conditional rule of the form "
        "\"If <condition>, then <handling>. Otherwise, <default handling>.\" "
        "Keep the conditional concrete and checkable from the input alone. "
        "Output only the new instruction.\n\n"
        f"Current instruction:\n{record.text}\n\n"
        f"Failing cases:\n{_format_cases(failures, n=8)}\n\n"
        "Instruction with conditional branch:"
    )
    result = opt._generate_prompt(
        meta_prompt, temperature=0.7, system_prompt=_IMPROVE_SYSTEM_PROMPT
    )
    return result.strip() or None


# --- UniPrompt x GSPE: facet decomposition with per-facet feedback ---

def _decompose(opt, text: str) -> Dict[str, str]:
    """Split a free-form prompt into GSPE's 5-field grammar. One LLM call."""
    field_list = "\n".join(f"- {k.upper()}: {v}" for k, v in _FACETS.items())
    meta_prompt = (
        "Decompose this instruction into structured fields. Use exactly the "
        "format 'FIELD_NAME: content', one field per line. Leave a field out "
        "if the instruction does not cover it.\n\n"
        f"Fields:\n{field_list}\n\n"
        f"Instruction:\n{text}\n\n"
        "Decomposition:"
    )
    raw = opt._generate_prompt(meta_prompt, temperature=0.3, system_prompt=_GENERATE_SYSTEM_PROMPT)

    structure: Dict[str, str] = {}
    current: Optional[str] = None
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        matched = None
        for field in _FACETS:
            if stripped.upper().startswith(field.upper() + ":"):
                matched = field
                break
        if matched:
            current = matched
            structure[current] = stripped[len(matched) + 1:].strip()
        elif current:
            structure[current] = (structure[current] + " " + stripped).strip()
    if not structure:
        structure["task_definition"] = text.strip()
    return structure


def _render(structure: Dict[str, str]) -> str:
    """Recombine facets in grammar order (matches gspe.py `_render_structure`)."""
    parts = [structure[f] for f in _FACETS if structure.get(f)]
    return "\n".join(parts).strip()


def t_facet_edit(opt) -> Optional[str]:
    """Decompose into facets, then add/edit/delete ONE facet using feedback
    attributed to that facet.

    This is the piece GSPE lacks: GSPE mutates a randomly-chosen field with no
    per-field credit signal, so it cannot tell which section is actually
    responsible for the errors. Here the model is shown the failures alongside
    the full decomposition and must name the responsible facet before editing
    it — UniPrompt's per-facet feedback attribution over GSPE's grammar.
    """
    record = _pick_elite(opt)
    if not record:
        return None
    details = _ensure_details(opt, record)
    failures = [d for d in details if not d.get("correct")] if details else []
    if not failures:
        return None

    structure = _decompose(opt, record.text)
    present = "\n".join(f"{k.upper()}: {v}" for k, v in structure.items() if v)
    missing = [f for f in _FACETS if not structure.get(f)]
    missing_note = (
        f"\nFields currently ABSENT (you may ADD one): {', '.join(missing)}"
        if missing else ""
    )

    meta_prompt = (
        "An instruction has been decomposed into sections. Given the failing "
        "cases, decide which SINGLE section is most responsible for the "
        "errors, then either rewrite that section or add a missing one.\n\n"
        f"Current sections:\n{present}{missing_note}\n\n"
        f"Failing cases:\n{_format_cases(failures, n=6)}\n\n"
        "Reply with exactly one line in the format 'FIELD_NAME: new content'."
    )
    raw = opt._generate_prompt(
        meta_prompt, temperature=0.7, system_prompt=_IMPROVE_SYSTEM_PROMPT
    ).strip()
    if not raw:
        return None

    target, new_content = None, None
    for line in raw.splitlines():
        s = line.strip()
        for field in _FACETS:
            if s.upper().startswith(field.upper() + ":"):
                target = field
                new_content = s[len(field) + 1:].strip()
                break
        if target:
            break
    if not target or not new_content:
        return None

    structure[target] = new_content
    rendered = _render(structure)
    return rendered if rendered and rendered != record.text else None


def t_decompose_recompose(opt) -> Optional[str]:
    """Decompose, drop empty/redundant facets, fill missing REQUIRED facets.

    A structural-repair operator (GSPE's `_grammar_refine`), useful when a
    prompt has drifted into an unstructured blob after many edits.
    """
    record = _pick_elite(opt)
    if not record:
        return None
    structure = _decompose(opt, record.text)
    required = ["task_definition", "output_spec"]
    missing = [f for f in required if not structure.get(f)]
    if not missing:
        return None

    meta_prompt = (
        f"This instruction is missing these components: {', '.join(missing)}. "
        "Write ONLY the missing components, one per line, in the format "
        "'FIELD_NAME: content'. Base them on the existing instruction.\n\n"
        f"Instruction:\n{record.text}\n\n"
        "Missing components:"
    )
    raw = opt._generate_prompt(
        meta_prompt, temperature=0.6, system_prompt=_GENERATE_SYSTEM_PROMPT
    )
    for line in raw.splitlines():
        s = line.strip()
        for field in missing:
            if s.upper().startswith(field.upper() + ":"):
                structure[field] = s[len(field) + 1:].strip()
    rendered = _render(structure)
    return rendered if rendered and rendered != record.text else None


V2_TECHNIQUES: Dict[str, Callable] = {
    "etgpo_taxonomy": t_etgpo_taxonomy,
    "strago_dual": t_strago_dual,
    "multi_aspect_critique": t_multi_aspect_critique,
    "autohint": t_autohint,
    "grips_delete": t_grips_delete,
    "grips_swap": t_grips_swap,
    "grips_add": t_grips_add,
    "grips_paraphrase": t_grips_paraphrase,
    "ampo_branch": t_ampo_branch,
    "facet_edit": t_facet_edit,
    "decompose_recompose": t_decompose_recompose,
}

# Operators requiring no LLM call at all — free exploration under a call cap.
ZERO_COST_TECHNIQUES = {"grips_delete", "grips_swap"}
