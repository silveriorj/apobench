"""Shared technique pool for FUNNEL — the top-20 operators pooled across
every method evaluated in this study (GAAPO, SEE, CAPO, SWIFT, APEX, GEPA).

Selection methodology (audit, 2026-07-25): every candidate ever produced by
a completed run of any method in `outputs/proposal_comparison_3bench_qwen3-4-instruct`,
`outputs/multi_model_cot_4bs`, `outputs/multi_model_cot_8bs`, and
`outputs/multi_model_4bs` (889 audited runs total) was pooled by its
recorded operator label, scored two ways: mean dev score, and win-rate
(the fraction of its own appearances where it produced the run's final
best prompt — a candidate tried once and winning once outranks a candidate
tried a thousand times and winning rarely). Labels that are literally the
same underlying mechanism under a different per-method name (e.g. SWIFT's
"zero_order_init", APEX's "semantic_var", and GAAPO's "semantic_init" all
call the same `BaseOptimizer._semantic_variation`) were deduplicated by
hand before ranking, since counting them as separate "techniques" would
have inflated the list with duplicates rather than genuinely distinct
mechanisms.

The 20 techniques below are the result: every technique that survived
dedup and ranked in the combined top tier by at least one of the two
scores, with GAAPO's full eight-mutator suite kept together (rather than
hand-picking only its historically-strongest members) so FUNNEL's own
narrowing schedule — not a prior audit — decides which of the eight earn
their place, since deciding that empirically, live, is the entire point
of a funnel-down design.

Every function below has the signature `(opt) -> Optional[str]`, where
`opt` is a FUNNELOptimizer instance. Functions either operate on a single
elite drawn from `opt.population` (most of them), bootstrap from raw
training I/O pairs (`lamarckian`), or need two-or-more existing candidates
(`crossover`, `midpoint_crossover`, `trajectory`, `eda`) — the funnel
schedule runs single-record techniques before pair/history techniques so
the latter always have material to work with (see `FUNNELOptimizer`).
"""
from __future__ import annotations

import random
import re
from typing import Callable, Dict, List, Optional

from pof.optimizers.base import (
    format_exemplar,
    _CRITIQUE_SYSTEM_PROMPT,
    _GENERATE_SYSTEM_PROMPT,
    _IMPROVE_SYSTEM_PROMPT,
)

# The eight GAAPO random-mutator instructions (Sécheresse et al. 2025),
# ported verbatim from gaapo.py's _MUTATION_TECHNIQUES.
_GAAPO_MUTATIONS: Dict[str, str] = {
    "role_assignment": (
        "Prepend an appropriate role assignment (e.g. 'You are a ...') to this "
        "instruction."
    ),
    "expert_persona": (
        "Rewrite this instruction as if written by a domain expert, injecting "
        "expert framing (e.g. 'As an expert in ...')."
    ),
    "concise_optimization": (
        "Make this instruction more concise without losing important information."
    ),
    "task_decomposition": (
        "Rewrite this instruction as a short sequence of sub-steps to follow."
    ),
    "structural_variation": (
        "Restructure this instruction: reorder its parts, or convert prose "
        "into steps / steps into prose, keeping the same meaning."
    ),
    "constraint_addition": (
        "Add one useful constraint or rule to this instruction."
    ),
    "instruction_expansion": (
        "Expand this instruction with additional helpful detail or clarification."
    ),
    "creative_backstory": (
        "Add a brief motivating context or scenario to this instruction."
    ),
}

_EXPERT_PERSONAS = [
    "a concise technical writer",
    "a patient teacher explaining to a student",
    "a rigorous logician focused on precision",
    "a creative problem solver",
]


def _pick_elite(opt, top_k: int = 3):
    """Choose the record an operator acts on.

    Normally a random draw from the top-`top_k` elites. When the caller has set
    `opt._forced_elite`, that record is returned instead — this lets a scheduler
    apply an operator to a SPECIFIC elite (e.g. sweeping every member of the
    population) without changing any operator's signature.
    """
    forced = getattr(opt, "_forced_elite", None)
    if forced is not None:
        opt._last_elite = forced
        return forced
    pool = opt.population[:top_k] if opt.population else []
    chosen = random.choice(pool) if pool else None
    # Record the target so a scheduler can post-process the result against the
    # record it was derived from. Inert for optimizers that never read it.
    opt._last_elite = chosen
    return chosen


def _ensure_details(opt, record):
    """Fetch per-sample details for a record, evaluating on demand if absent."""
    details = record.per_sample_details
    if not details and record.text:
        samples = opt.dataset.get_eval_samples("dev", n=opt.eval_sample_size)
        result = opt.evaluator.evaluate(record.text, samples)
        details = result.per_sample_details
        record.per_sample_details = details
        record.score = result.score
        record.performance_vector = result.performance_vector
    return details


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


# --- Single-record techniques ---

def t_semantic_var(opt) -> Optional[str]:
    """Paraphrase an elite (BaseOptimizer._semantic_variation)."""
    record = _pick_elite(opt)
    if not record:
        return None
    out = opt._semantic_variation(record.text, n=1)
    return out[0] if out else None


def t_local_edit(opt) -> Optional[str]:
    """PLUM-style small, clarity-preserving rephrase (ported from swift.py)."""
    record = _pick_elite(opt)
    if not record:
        return None
    meta_prompt = (
        "Make a small, targeted improvement to this instruction. "
        "Change only 1-2 sentences to make it clearer or more precise. "
        "Do NOT rewrite the entire instruction.\n\n"
        f"Instruction:\n{record.text}\n\n"
        "Slightly improved instruction:"
    )
    result = opt._generate_prompt(meta_prompt, temperature=0.5, system_prompt=_GENERATE_SYSTEM_PROMPT)
    return result.strip() if result.strip() else None


def t_structured_failure_guided(opt) -> Optional[str]:
    """SWIFT-style two-step diagnose-then-rewrite (ported from swift.py)."""
    record = _pick_elite(opt)
    if not record:
        return None
    details = _ensure_details(opt, record)
    failures = [d for d in details if not d["correct"]] if details else []
    if not failures:
        return None
    failure_text = "\n".join(
        f"- Input: {f.get('input', '')[:60]} | Expected: {f.get('target', '')} | Got: {f.get('prediction', '')[:60]}"
        for f in failures[:5]
    )
    meta_prompt = (
        "You are an expert prompt engineer. Analyze why this instruction fails "
        "on certain inputs, then write an improved version.\n\n"
        f"Current instruction:\n{record.text}\n\n"
        f"Failure cases:\n{failure_text}\n\n"
        "Step 1: Diagnose the root cause of failures.\n"
        "Step 2: Write an improved instruction that addresses these issues.\n\n"
        "Improved instruction:"
    )
    result = opt._generate_prompt(meta_prompt, temperature=0.7, system_prompt=_IMPROVE_SYSTEM_PROMPT)
    return result.strip() if result.strip() else None


def t_expert_refine(opt) -> Optional[str]:
    """APEX-style persona-based refinement (ported from apex.py)."""
    record = _pick_elite(opt)
    if not record:
        return None
    persona = random.choice(_EXPERT_PERSONAS)
    meta_prompt = (
        f"You are {persona}. Improve this instruction to make it more effective. "
        f"Maintain the core intent but enhance clarity and precision.\n\n"
        f"Original instruction:\n{record.text}\n\n"
        f"Improved instruction:"
    )
    result = opt._generate_prompt(meta_prompt, temperature=0.7, system_prompt=_IMPROVE_SYSTEM_PROMPT)
    return result.strip() if result.strip() else None


def t_few_shot(opt) -> Optional[str]:
    """Append k labeled demonstrations to an elite (idempotent: strips any
    existing block first, so demonstrations never accumulate)."""
    record = _pick_elite(opt)
    if not record:
        return None
    base_text = re.split(r"\n\nExamples:\n", record.text)[0]
    train = opt.dataset.get_few_shot_examples(n=8, seed=random.randint(0, 10**6))
    if not train:
        return None
    k = random.randint(1, min(3, len(train)))
    shots = random.sample(train, k)
    examples = "\n\n".join(format_exemplar(opt.evaluator, s) for s in shots)
    return f"{base_text}\n\nExamples:\n{examples}"


def t_format_constraint(opt) -> Optional[str]:
    """Append an explicit output-format rule derived from real targets."""
    record = _pick_elite(opt)
    if not record:
        return None
    targets = [s["target"] for s in opt.dataset.get_few_shot_examples(n=4)]
    if not targets:
        return None
    sample_answers = ", ".join(repr(t)[:30] for t in targets[:3])
    return (
        f"{record.text.rstrip()}\n\nAnswer with ONLY the final answer, exactly "
        f"in the same format as these examples: {sample_answers}. No explanation."
    )


def _mutator(technique: str) -> Callable:
    instruction = _GAAPO_MUTATIONS[technique]

    def _fn(opt) -> Optional[str]:
        record = _pick_elite(opt)
        if not record:
            return None
        meta_prompt = f"{instruction}\n\nInstruction:\n{record.text}\n\nModified instruction:"
        result = opt._generate_prompt(meta_prompt, temperature=0.9, system_prompt=_GENERATE_SYSTEM_PROMPT)
        return result.strip() if result.strip() else None

    _fn.__name__ = f"t_mutator_{technique}"
    return _fn


def t_reflective_mutation(opt) -> Optional[str]:
    """GEPA-style two-step reflect-then-rewrite (ported from gepa.py)."""
    record = _pick_elite(opt)
    if not record:
        return None
    details = _ensure_details(opt, record)
    if not details:
        return None
    failures = [d for d in details if not d.get("correct")]
    shown = failures[:4] if failures else details[:4]
    trace_text = "\n".join(
        f"- Input: {t.get('input', '')[:120]}\n"
        f"  Expected: {t.get('target', '')}\n"
        f"  Model output: {t.get('prediction', '')[:120]}\n"
        f"  Correct: {t.get('correct')}"
        for t in shown
    )
    reflection_prompt = (
        "An AI assistant used the instruction below and produced these "
        "results. Reflect on what the instruction fails to convey: what "
        "task rules, edge cases, or output format details is it missing? "
        "Answer in 2-4 sentences.\n\n"
        f"Instruction:\n{record.text}\n\n"
        f"Execution traces:\n{trace_text}\n\n"
        "Reflection:"
    )
    reflection = opt._generate_prompt(reflection_prompt, temperature=0.7, system_prompt=_CRITIQUE_SYSTEM_PROMPT)
    if not reflection.strip():
        return None
    rewrite_prompt = (
        "Rewrite the instruction to fix the issues identified in the "
        "reflection. Keep what works; add the missing task rules or "
        "format details. Output only the new instruction.\n\n"
        f"Current instruction:\n{record.text}\n\n"
        f"Reflection:\n{reflection.strip()}\n\n"
        "Improved instruction:"
    )
    result = opt._generate_prompt(rewrite_prompt, temperature=0.7, system_prompt=_IMPROVE_SYSTEM_PROMPT)
    return result.strip() if result.strip() else None


# --- Bootstrap technique (no population needed) ---

def t_lamarckian(opt) -> Optional[str]:
    train_samples = opt.dataset.get_few_shot_examples(n=5)
    out = opt._lamarckian_generate(train_samples, n=1)
    return out[0] if out else None


# --- Pair / multi-candidate techniques (need >= 2 existing candidates) ---

def t_crossover(opt) -> Optional[str]:
    """LLM-blended semantic crossover (BaseOptimizer._crossover)."""
    if len(opt.population) < 2:
        return None
    a, b = random.sample(opt.population[:4], 2)
    result = opt._crossover(a.text, b.text)
    return result.strip() if result.strip() else None


def t_midpoint_crossover(opt) -> Optional[str]:
    """GAAPO's positional midpoint splice — genuinely distinct from the LLM
    blend above: a string-level operation, not an LLM call."""
    if len(opt.population) < 2:
        return None
    a, b = random.sample(opt.population[:4], 2)
    sents_a = _split_sentences(a.text)
    sents_b = _split_sentences(b.text)
    half_a = " ".join(sents_a[: max(1, len(sents_a) // 2)])
    half_b = " ".join(sents_b[len(sents_b) // 2:]) if len(sents_b) > 1 else b.text
    text = f"{half_a} {half_b}".strip()
    return text or None


def t_trajectory(opt) -> Optional[str]:
    """OPRO-style: condition generation on the ranked (prompt, score) history."""
    if len(opt.population) < 2:
        return None
    ranked = sorted(opt.population, key=lambda r: r.score)
    context = "\n".join(f"Score: {r.score:.3f}\nInstruction: {r.text}\n" for r in ranked)
    meta_prompt = (
        "Below are instructions for a task, sorted by performance score "
        "(ascending). Write a NEW instruction that would score higher than "
        "all of the above.\n\n"
        f"Instructions and scores:\n{context}\n\n"
        "New higher-scoring instruction:"
    )
    result = opt._generate_prompt(meta_prompt, temperature=0.8, system_prompt=_GENERATE_SYSTEM_PROMPT)
    return result.strip() if result.strip() else None


def t_eda(opt) -> Optional[str]:
    """EDA: generate from the distribution of existing top prompts."""
    if len(opt.population) < 2:
        return None
    prompts = [r.text for r in opt.population[:5]]
    result = opt._eda_generate(prompts)
    return result.strip() if result.strip() else None


# --- The pool ---
# Order matters: single-record and bootstrap techniques run before
# pair/history techniques, so the latter always have material to draw on
# (see FUNNELOptimizer._phase_broad).

SINGLE_RECORD_TECHNIQUES: Dict[str, Callable] = {
    "semantic_var": t_semantic_var,
    "local_edit": t_local_edit,
    "structured_failure_guided": t_structured_failure_guided,
    "expert_refine": t_expert_refine,
    "few_shot": t_few_shot,
    "format_constraint": t_format_constraint,
    "reflective_mutation": t_reflective_mutation,
    **{f"mutator_{k}": _mutator(k) for k in _GAAPO_MUTATIONS},
}

BOOTSTRAP_TECHNIQUES: Dict[str, Callable] = {
    "lamarckian": t_lamarckian,
}

PAIR_TECHNIQUES: Dict[str, Callable] = {
    "crossover": t_crossover,
    "midpoint_crossover": t_midpoint_crossover,
    "trajectory": t_trajectory,
    "eda": t_eda,
}

ALL_TECHNIQUES: Dict[str, Callable] = {
    **BOOTSTRAP_TECHNIQUES,
    **SINGLE_RECORD_TECHNIQUES,
    **PAIR_TECHNIQUES,
}

assert len(ALL_TECHNIQUES) == 20, f"expected 20 techniques, got {len(ALL_TECHNIQUES)}"
