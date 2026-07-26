"""Measure the canonical BBH baseline prompts on OUR model and harness.

Why this exists. Our optimized scores were compared against Table 3 of Suzgun
et al. (2023), but that comparison is not like-for-like: their "answer-only"
(AO) column is **3-shot** — three worked exemplars plus a task description and
answer options — whereas our optimized prompts start from a bare instruction
and only carry demonstrations if the optimizer added them. Their numbers also
come from 175B-540B models measured with a different answer extractor.

This script removes every one of those confounds at once by evaluating the
canonical prompts on our model, our test split and our scorer:

    instruction_only  line 1 of the official prompt file (what we seed with)
    answer_only_3shot the official 3 exemplars with reasoning stripped to the
                      final answer — Suzgun's AO setup, reconstructed

The result is the number our optimized prompts should actually be compared
against: same model, same 115-instance test split, same greedy decoding, same
32-token cap, same JSON extraction. No optimization is run.

The chain-of-thought column is deliberately NOT reproduced: our protocol caps
generation at 32 tokens, which forbids reasoning by construction, so a CoT
measurement here would say nothing about our setup.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from pof.datasets.loader import load_dataset_by_name
from pof.evaluation.evaluator import Evaluator
from pof.llm.factory import create_llm
from pof.config.schemas import LLMConfig
from pof.prompts.loader import fetch_bbh_prompt, extract_instruction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ANSWER_RE = re.compile(r"So the answer is\s*(.+?)\.\s*$", re.MULTILINE)


def build_answer_only(prompt_text: str) -> str:
    """Reconstruct Suzgun's answer-only prompt from the official CoT file.

    The file is: instruction, then blocks of `Q: ... A: <prompt> <reasoning>
    ... So the answer is (X).`. Answer-only keeps each question and replaces
    the reasoning with the final answer, which is exactly the AO/CoT contrast
    the paper draws (same exemplars, reasoning removed).
    """
    text = prompt_text.strip()
    head, _, body = text.partition("\n\n")
    instruction = head.strip()

    blocks = [b for b in body.split("\n\nQ:") if b.strip()]
    rebuilt: List[str] = []
    for i, block in enumerate(blocks):
        block = block if i == 0 else "Q:" + block
        if not block.lstrip().startswith("Q:"):
            block = "Q: " + block.lstrip()
        question, sep, answer_part = block.partition("\nA:")
        if not sep:
            continue
        m = ANSWER_RE.search(answer_part)
        if not m:
            continue
        rebuilt.append(f"{question.strip()}\nA: {m.group(1).strip()}")

    if not rebuilt:
        return instruction
    return instruction + "\n\n" + "\n\n".join(rebuilt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--test-n", type=int, default=115)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--out", default="outputs/bbh_reference_baseline.json")
    args = ap.parse_args()

    llm = create_llm(LLMConfig(backend="huggingface", model_name=args.model,
                              device="auto", dtype="auto", max_new_tokens=512))
    results: Dict[str, Dict[str, float]] = {}

    for task in args.tasks:
        raw = fetch_bbh_prompt(task)
        variants = {
            "instruction_only": extract_instruction(raw),
            "answer_only_3shot": build_answer_only(raw),
        }
        dataset = load_dataset_by_name("bbh", task=task, num_samples=100000, seed=42)
        samples = dataset.get_eval_samples("test", n=args.test_n)
        evaluator = Evaluator(llm, task_type="auto",
                              max_new_tokens=args.max_new_tokens,
                              temperature=0.0, batch_size=args.batch_size)

        results[task] = {}
        for name, prompt in variants.items():
            res = evaluator.evaluate(prompt, samples)
            results[task][name] = res.score
            logger.info("%-34s %-18s n=%d  acc=%.4f  (prompt %d chars)",
                        task, name, len(samples), res.score, len(prompt))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
