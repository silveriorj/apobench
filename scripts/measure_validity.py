"""Measure how many stored candidate prompts are actually usable instructions.

The optimizer's entire validation of an LLM response is
`result.strip() if result.strip() else None` -- non-empty is the only test.
So a preamble ("Sure! Here's an improved instruction: ..."), a leftover
Step-1 diagnosis, or a refusal all become candidate prompts and get
evaluated. This script quantifies how often that happens on data already on
disk. Zero LLM calls.

Restricted to `is_complete` records: incomplete ones carry a text_hash but
no persisted text, so counting them would compare real prompts against empty
strings (the contamination pattern found in the sibling project's feature
analysis). The excluded count is reported, as the aggregation pipeline does
for its own exclusions.

    python scripts/measure_validity.py [--root outputs] [--examples N]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

# A response that opens by talking to the user rather than stating the task.
_PREAMBLE = re.compile(
    r"^\s*(sure|certainly|of course|here(?:'s| is)|i'd be happy|absolutely|"
    r"okay|ok,|got it|great|understood|below is|the improved|improved version)"
    r"\b", re.IGNORECASE)

# Scaffolding from the meta-prompt that leaked into the candidate itself.
_META_LEAK = re.compile(
    r"(step\s*1\s*:|step\s*2\s*:|diagnosis\s*:|root cause\s*:|"
    r"improved instruction\s*:|new instruction\s*:|analysis\s*:|"
    r"here is the improved|revised instruction\s*:)", re.IGNORECASE)

# The model declining or commenting instead of producing an instruction.
_REFUSAL = re.compile(
    r"^\s*(i (?:cannot|can't|am unable|do not|don't)|as an ai|"
    r"the (?:instruction|prompt) (?:is|seems) (?:already|correct|fine))",
    re.IGNORECASE)

_TERMINAL = tuple(".!?:)]}\"'`")


# Operators that build their text by string concatenation, with no LLM call.
# Their output is not model-generated, so format compliance does not apply.
_ZERO_CALL_OPS = {
    "few_shot", "few_shot_init", "few_shot_phase1", "few_shot_polish",
    "few_shot_augment", "few_shot_fixed", "few_shot_aug", "format_constraint",
    "format_constraint_init", "midpoint_crossover", "grips_delete",
    "grips_swap", "instruction_only", "O_ICL",
}


def _instruction_part(text: str) -> str:
    """Strip appended exemplar blocks before judging format compliance.

    Demonstrations are concatenated verbatim from the dataset and routinely
    end without terminal punctuation (code, or `Output: Yes`), so judging the
    whole string would flag every exemplar-bearing prompt as truncated.
    """
    head = text.split("Examples:")[0]
    head = head.split("\nInput:")[0]
    head = head.split("\nQ:")[0]
    return head.strip()


def classify(text: str) -> str:
    """Return the single most severe problem with this candidate prompt."""
    t = (text or "").strip()
    if not t:
        return "empty"
    if _REFUSAL.search(t):
        return "refusal"
    if _META_LEAK.search(t):
        return "meta_leak"
    if _PREAMBLE.search(t):
        return "preamble"
    if not t.endswith(_TERMINAL) and len(t.split()) > 12:
        # Long and stopping mid-sentence: consistent with hitting the cap.
        return "truncated"
    return "clean"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs")
    ap.add_argument("--examples", type=int, default=2,
                    help="sample offending prompts to print per category")
    a = ap.parse_args()

    files = glob.glob(os.path.join(a.root, "**", "audit_*.json"), recursive=True)
    if not files:
        print(f"no audit_*.json under {a.root!r}")
        return 1

    counts: Counter = Counter()
    by_operator: dict = defaultdict(Counter)
    samples: dict = defaultdict(list)
    n_records = n_incomplete = n_files = 0

    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        n_files += 1
        # Records live under `history.records` (the audit export nests the
        # OptimizationHistory dict under a top-level `history` key).
        records = (data.get("history") or {}).get("records") or data.get("records") or {}
        for rec in records.values():
            n_records += 1
            # Incomplete records have a hash but no text -- counting them
            # would compare real prompts against empty strings.
            if not rec.get("is_complete", True) or not rec.get("text"):
                n_incomplete += 1
                continue
            op = rec.get("operator") or "?"
            # Seeds and zero-LLM-call operators aren't model output, so
            # format compliance is not a meaningful question for them.
            if op in ("seed", "init") or op in _ZERO_CALL_OPS:
                continue
            # Judge only the generated instruction, not appended exemplars.
            label = classify(_instruction_part(rec["text"]))
            counts[label] += 1
            by_operator[op][label] += 1
            if label != "clean" and len(samples[label]) < a.examples:
                samples[label].append((op, rec["text"][:180].replace("\n", " ")))

    total = sum(counts.values())
    if not total:
        print(f"scanned {n_files} files, {n_records} records, "
              f"but none carried text (all is_complete=False)")
        return 1

    print(f"audit files scanned : {n_files}")
    print(f"records total       : {n_records}")
    print(f"excluded (no text)  : {n_incomplete} "
          f"({100*n_incomplete/max(n_records,1):.1f}%)")
    print(f"classified          : {total}")
    print()
    unclean = total - counts["clean"]
    print(f"{'category':14s} {'count':>7s}  {'share':>7s}")
    print("-" * 32)
    for label, c in counts.most_common():
        print(f"{label:14s} {c:7d}  {100*c/total:6.1f}%")
    print("-" * 32)
    print(f"{'UNCLEAN':14s} {unclean:7d}  {100*unclean/total:6.1f}%")

    print("\nworst operators by unclean share (min 20 candidates):")
    rows = []
    for op, c in by_operator.items():
        tot = sum(c.values())
        if tot >= 20:
            rows.append((1 - c["clean"] / tot, tot, op))
    for share, tot, op in sorted(rows, reverse=True)[:10]:
        print(f"   {op:28s} {100*share:5.1f}%  (n={tot})")

    if samples:
        print("\nexamples:")
        for label, items in samples.items():
            for op, txt in items:
                print(f"   [{label}] ({op}) {txt!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
