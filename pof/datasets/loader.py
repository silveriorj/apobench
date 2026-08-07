"""Dataset loader — BigBench and custom JSON datasets.

Ported from Projeto's TaskDataset with unified interface for:
- BigBench-Hard (BBH) tasks via HuggingFace datasets
- Custom JSON datasets (local files)
- Train/dev/test split management
- Few-shot example selection
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pof.core.exceptions import DatasetError

logger = logging.getLogger(__name__)

# Seed used to carve the held-out TEST split. Deliberately CONSTANT: the
# test set must be identical across run seeds so their scores are
# comparable. Run seeds still vary train/dev, which is the optimizer-facing
# randomness the seed sweep is meant to measure. Value 42 preserves the
# test set that seed-42 runs already used.
TEST_SPLIT_SEED = 42

# BigBench-Hard tasks
BBH_TASKS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]


class TaskDataset:
    """Unified dataset interface for prompt optimization tasks.

    Manages train/dev/test splits and provides samples in a standard format:
    [{"input": str, "target": str}, ...]
    """

    def __init__(
        self,
        name: str,
        train_samples: List[Dict[str, str]],
        dev_samples: List[Dict[str, str]],
        test_samples: List[Dict[str, str]],
        task_type: str = "auto",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.train_samples = train_samples
        self.dev_samples = dev_samples
        self.test_samples = test_samples
        self.task_type = task_type
        self.metadata = metadata or {}

    @property
    def num_train(self) -> int:
        return len(self.train_samples)

    @property
    def num_dev(self) -> int:
        return len(self.dev_samples)

    @property
    def num_test(self) -> int:
        return len(self.test_samples)

    def get_few_shot_examples(self, n: int = 3, seed: int = 42) -> List[Dict[str, str]]:
        """Get n few-shot examples from training set."""
        rng = random.Random(seed)
        if n >= len(self.train_samples):
            return self.train_samples[:]
        return rng.sample(self.train_samples, n)

    def get_eval_samples(
        self, split: str = "dev", n: Optional[int] = None, seed: int = 42
    ) -> List[Dict[str, str]]:
        """Get evaluation samples from specified split.

        Args:
            split: 'train', 'dev', or 'test'.
            n: Number of samples (None = all).
            seed: Random seed for sampling.
        """
        if split == "train":
            samples = self.train_samples
        elif split == "dev":
            samples = self.dev_samples
        elif split == "test":
            samples = self.test_samples
        else:
            raise DatasetError(f"Unknown split: {split}")

        if n is not None and n < len(samples):
            rng = random.Random(seed)
            return rng.sample(samples, n)
        return samples[:]

    def format_few_shot_prompt(
        self, instruction: str, n_examples: int = 3, seed: int = 42
    ) -> str:
        """Format instruction with few-shot examples appended."""
        examples = self.get_few_shot_examples(n_examples, seed)
        if not examples:
            return instruction

        example_text = "\n\n".join(
            f"Input: {ex['input']}\nOutput: {ex['target']}"
            for ex in examples
        )
        return f"{instruction}\n\nExamples:\n{example_text}"


def load_dataset_by_name(
    name: str,
    task: str = "",
    num_samples: int = 100,
    seed: int = 42,
    dev_test_split: float = 0.0,
) -> TaskDataset:
    """Load a dataset by name.

    Args:
        name: Dataset name ('bbh', 'json', or path to JSON file).
        task: Specific task within dataset (e.g., BBH task name).
        num_samples: Total samples to load.
        seed: Random seed.
        dev_test_split: Fraction of the post-train pool given to dev (rest to
            test). 0.0 (default) preserves the original fixed-size split
            (test capped at 115, dev gets the remainder). Only applied by
            BBH loading; other datasets ignore it.

    Returns:
        TaskDataset instance.
    """
    if name.lower() == "bbh":
        return _load_bbh(task, num_samples, seed, dev_test_split=dev_test_split)
    elif name.lower() in ("livebench_math", "livebench/math"):
        return _load_livebench_math(task, num_samples, seed)
    elif name.lower() == "gsm8k":
        return _load_gsm8k(num_samples, seed)
    elif name.lower() == "svamp":
        return _load_svamp(num_samples, seed)
    elif name.lower() == "humaneval":
        return _load_humaneval(num_samples, seed)
    elif name.endswith(".json") or Path(name).exists():
        return _load_json(name, num_samples, seed)
    else:
        raise DatasetError(
            f"Unknown dataset: {name}. Use 'bbh', 'gsm8k', 'svamp', 'livebench_math', or a JSON file path."
        )


def _load_bbh(task: str, num_samples: int, seed: int, dev_test_split: float = 0.0) -> TaskDataset:
    """Load a BigBench-Hard task from HuggingFace."""
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise DatasetError("'datasets' package required. Run: pip install datasets")

    if not task:
        # Pick a random task
        rng = random.Random(seed)
        task = rng.choice(BBH_TASKS)
        logger.info(f"No task specified, randomly selected: {task}")

    if task not in BBH_TASKS:
        raise DatasetError(
            f"Unknown BBH task: {task}. Available: {BBH_TASKS}"
        )

    # Prefer a script-free dataset to comply with datasets>=5 restrictions
    try:
        # Community mirror with per-task subsets and no loading script
        dataset = hf_load_dataset("lukaemon/bbh", task)
    except Exception as e1:
        # Fallback to original repo if available (may fail on newer datasets versions)
        try:
            dataset = hf_load_dataset("maveriq/bigbenchhard", task)
        except Exception as e2:
            raise DatasetError(
                f"Failed to load BBH task '{task}': lukaemon/bbh error: {e1}; "
                f"maveriq/bigbenchhard error: {e2}. "
                "Tip: use datasets<3.0 for script-based loaders or switch to lukaemon/bbh."
            ) from e2

    # BBH has a single 'train' split
    split_data = dataset["train"] if "train" in dataset else dataset[list(dataset.keys())[0]]

    # Convert to standard format
    all_samples = []
    for item in split_data:
        sample = {
            "input": item.get("input", item.get("question", "")),
            "target": item.get("target", item.get("answer", "")),
        }
        if sample["input"] and sample["target"]:
            all_samples.append(sample)

    # Shuffle with all available samples (no cap)
    # Test split is carved with a FIXED seed so the held-out set is identical
    # across run seeds; only train/dev vary with the run seed (see module note).
    rng = random.Random(TEST_SPLIT_SEED)
    rng.shuffle(all_samples)

    if dev_test_split > 0.0:
        # Even split mode: train takes a fixed few-shot pool, dev/test split
        # the remainder by the requested ratio (e.g. 0.5 -> 50/50). Exists to
        # test whether the small default dev pool (see fixed-size branch
        # below) was starving the optimizer of statistical power -- APO gains
        # measured on ~50-instance dev did not transfer to test (corr
        # -0.695); a larger, evenly-sized dev pool is the direct fix.
        n_train = 8 if len(all_samples) >= 8 + 20 else 3
        pool = len(all_samples) - n_train
        n_dev = max(1, round(pool * dev_test_split))
        n_test = pool - n_dev
    elif len(all_samples) >= 8 + 50 + 115:
        # Fixed-size splits: test is reserved first (held-out, fixed at 115),
        # train (few-shot) takes 8, dev gets everything else (≥50 for candidate eval).
        # BBH tasks have 187-250 examples, so dev ends up with 64-127 samples.
        # Small tasks (e.g. penguins_in_a_table, 146 examples) shrink train to 3
        # and the dev floor to 33 so the held-out test stays as close to 115 as possible.
        n_train = 8
        n_test = min(115, max(1, len(all_samples) - n_train - 50))
        n_dev = len(all_samples) - n_test - n_train
    else:
        n_train = 3
        n_test = min(115, max(1, len(all_samples) - n_train - 33))
        n_dev = len(all_samples) - n_test - n_train

    test = all_samples[:n_test]
    _rest = all_samples[n_test:]
    random.Random(seed).shuffle(_rest)   # run seed varies train/dev only
    train = _rest[:n_train]
    dev = _rest[n_train:]

    # Detect task type
    task_type = _detect_task_type(train)

    return TaskDataset(
        name=f"bbh_{task}",
        train_samples=train,
        dev_samples=dev,
        test_samples=test,
        task_type=task_type,
        metadata={"source": "bigbenchhard", "task": task},
    )


def _load_livebench_math(task: str, num_samples: int, seed: int) -> TaskDataset:
    """Load LiveBench math (HF: livebench/math) — zero-shot competition math.

    Subtasks: AMPS_Hard (numeric/expression answers), math_comp (multiple
    choice), olympiad (fill-in). Pass `task` to filter to one subtask, or
    leave empty for all. Targets come from `ground_truth`; scoring uses
    final_em-style exact match on the extracted final answer (\\boxed{} aware).
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise DatasetError("'datasets' package required. Run: pip install datasets")

    try:
        dataset = hf_load_dataset("livebench/math", split="test")
    except Exception as e:
        raise DatasetError(f"Failed to load livebench/math: {e}") from e

    all_samples = []
    for item in dataset:
        if task and item.get("task", "") != task:
            continue
        turns = item.get("turns") or []
        question = turns[0] if turns else item.get("question", "")
        target = item.get("ground_truth", "")
        if question and target:
            all_samples.append({"input": question, "target": str(target)})

    if not all_samples:
        raise DatasetError(
            f"livebench/math returned no samples (task filter: {task!r})"
        )

    # Test split is carved with a FIXED seed so the held-out set is identical
    # across run seeds; only train/dev vary with the run seed (see module note).
    rng = random.Random(TEST_SPLIT_SEED)
    rng.shuffle(all_samples)

    # Same fixed-size splits as BBH (small-task fallback included)
    if len(all_samples) >= 8 + 50 + 115:
        n_train = 8
        n_test = min(115, max(1, len(all_samples) - n_train - 50))
    else:
        n_train = 3
        n_test = min(115, max(1, len(all_samples) - n_train - 33))

    test = all_samples[:n_test]
    _rest = all_samples[n_test:]
    random.Random(seed).shuffle(_rest)   # run seed varies train/dev only
    train = _rest[:n_train]
    dev = _rest[n_train:]

    name = f"livebench_math_{task}" if task else "livebench_math"
    return TaskDataset(
        name=name,
        train_samples=train,
        dev_samples=dev,
        test_samples=test,
        task_type="math",
        metadata={"source": "livebench/math", "task": task},
    )


def _load_gsm8k(num_samples: int, seed: int) -> TaskDataset:
    """Load GSM8K (HF: openai/gsm8k) — grade-school math word problems.

    HF train split (7473) → few-shot train examples (8 samples).
    HF test split (1319)  → shuffled, then split into test (115) and dev (rest).
    Targets are the integer answer extracted from the '#### N' suffix in the
    HF answer field, so the scorer compares against "The answer is N" format.
    """
    import re as _re

    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise DatasetError("'datasets' package required. Run: pip install datasets")

    try:
        hf_train = hf_load_dataset("openai/gsm8k", "main", split="train")
        hf_test = hf_load_dataset("openai/gsm8k", "main", split="test")
    except Exception as e:
        raise DatasetError(f"Failed to load openai/gsm8k: {e}") from e

    def _extract_answer(answer_text: str) -> str:
        m = _re.search(r"####\s*([0-9,]+)", answer_text)
        return m.group(1).replace(",", "") if m else answer_text.strip()

    # Few-shot train examples: sample 8 from HF train split
    rng = random.Random(seed)
    train_indices = rng.sample(range(len(hf_train)), min(8, len(hf_train)))
    train_samples = [
        {"input": hf_train[i]["question"], "target": _extract_answer(hf_train[i]["answer"])}
        for i in train_indices
    ]

    # Dev + test from HF test split
    test_samples_raw = [
        {"input": item["question"], "target": _extract_answer(item["answer"])}
        for item in hf_test
    ]
    # Fixed seed carves the held-out test set identically for every run seed;
    # only dev is re-shuffled per run seed (train came from the HF train split).
    random.Random(TEST_SPLIT_SEED).shuffle(test_samples_raw)

    n_test = min(115, max(1, len(test_samples_raw) - 50))
    test = test_samples_raw[:n_test]
    dev = test_samples_raw[n_test:]
    random.Random(seed).shuffle(dev)

    logger.info(f"GSM8K loaded: {len(train_samples)} train, {len(dev)} dev, {len(test)} test")

    return TaskDataset(
        name="gsm8k",
        train_samples=train_samples,
        dev_samples=dev,
        test_samples=test,
        task_type="math",
        metadata={"source": "openai/gsm8k"},
    )


def _load_svamp(num_samples: int, seed: int) -> TaskDataset:
    """Load SVAMP (HF: ChilleD/SVAMP) — 1-2 step arithmetic word problems.

    A faster, simpler complement to GSM8K (2-8 reasoning steps): shorter
    questions and shorter chains of thought, so both generation and
    evaluation are cheaper per sample. HF train split (700) feeds few-shot
    examples; HF test split (300) is shuffled then split into test (115)
    and dev (rest). Targets are plain numeric strings (int-formatted when
    whole), matching GSM8K's '#### N' convention so the shared 'math'
    scorer (_score_math) needs no changes.
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise DatasetError("'datasets' package required. Run: pip install datasets")

    try:
        hf_train = hf_load_dataset("ChilleD/SVAMP", split="train")
        hf_test = hf_load_dataset("ChilleD/SVAMP", split="test")
    except Exception as e:
        raise DatasetError(f"Failed to load ChilleD/SVAMP: {e}") from e

    def _format_answer(value) -> str:
        num = float(value)
        return str(int(num)) if num.is_integer() else str(num)

    rng = random.Random(seed)
    train_indices = rng.sample(range(len(hf_train)), min(8, len(hf_train)))
    train_samples = [
        {"input": hf_train[i]["question_concat"], "target": _format_answer(hf_train[i]["Answer"])}
        for i in train_indices
    ]

    test_samples_raw = [
        {"input": item["question_concat"], "target": _format_answer(item["Answer"])}
        for item in hf_test
    ]
    # Fixed seed carves the held-out test set identically for every run seed;
    # only dev is re-shuffled per run seed (train came from the HF train split).
    random.Random(TEST_SPLIT_SEED).shuffle(test_samples_raw)

    n_test = min(115, max(1, len(test_samples_raw) - 50))
    test = test_samples_raw[:n_test]
    dev = test_samples_raw[n_test:]
    random.Random(seed).shuffle(dev)

    logger.info(f"SVAMP loaded: {len(train_samples)} train, {len(dev)} dev, {len(test)} test")

    return TaskDataset(
        name="svamp",
        train_samples=train_samples,
        dev_samples=dev,
        test_samples=test,
        task_type="math",
        metadata={"source": "ChilleD/SVAMP"},
    )


def _load_humaneval(num_samples: int, seed: int) -> TaskDataset:
    """Load HumanEval (HF: openai/openai_humaneval) — code generation, pass@1.

    The sample `input` is the function signature + docstring; the `target`
    is a JSON blob carrying the unit tests and entry point, consumed by the
    'code' score function which executes the completion against the tests.
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        raise DatasetError("'datasets' package required. Run: pip install datasets")

    try:
        dataset = hf_load_dataset("openai/openai_humaneval", split="test")
    except Exception as e:
        raise DatasetError(f"Failed to load openai/openai_humaneval: {e}") from e

    all_samples = []
    for item in dataset:
        all_samples.append({
            "input": item["prompt"],
            "target": json.dumps({
                "prompt": item["prompt"],
                "test": item["test"],
                "entry_point": item["entry_point"],
            }),
            "_canonical": item["canonical_solution"],
        })

    # Test split is carved with a FIXED seed so the held-out set is identical
    # across run seeds; only train/dev vary with the run seed (see module note).
    rng = random.Random(TEST_SPLIT_SEED)
    rng.shuffle(all_samples)

    # 164 problems total → small-task split: train=3, dev=46, test=115
    if len(all_samples) >= 8 + 50 + 115:
        n_train = 8
        n_test = min(115, max(1, len(all_samples) - n_train - 50))
    else:
        n_train = 3
        n_test = min(115, max(1, len(all_samples) - n_train - 33))

    test = all_samples[:n_test]
    _rest = all_samples[n_test:]
    random.Random(seed).shuffle(_rest)   # run seed varies train/dev only
    train = _rest[:n_train]
    dev = _rest[n_train:]

    # Train samples feed few-shot/Lamarckian operators — show the canonical
    # solution as the target, not the JSON test blob used for scoring.
    train = [
        {"input": s["input"], "target": s["_canonical"]} for s in train
    ]
    test = [{"input": s["input"], "target": s["target"]} for s in test]
    dev = [{"input": s["input"], "target": s["target"]} for s in dev]

    return TaskDataset(
        name="humaneval",
        train_samples=train,
        dev_samples=dev,
        test_samples=test,
        task_type="code",
        metadata={"source": "openai/openai_humaneval"},
    )


def _load_json(path: str, num_samples: int, seed: int) -> TaskDataset:
    """Load dataset from a JSON file.

    Expected format:
    [{"input": "...", "target": "..."}, ...] or
    {"train": [...], "dev": [...], "test": [...]}
    """
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Dataset file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise DatasetError(f"Invalid JSON in {path}: {e}") from e

    if isinstance(data, list):
        # Flat list — split it
        rng = random.Random(seed)
        rng.shuffle(data)
        data = data[:num_samples]

        n_train = max(3, len(data) // 5)
        n_remaining = len(data) - n_train
        n_dev = n_remaining // 2

        train = data[:n_train]
        dev = data[n_train: n_train + n_dev]
        test = data[n_train + n_dev:]
    elif isinstance(data, dict):
        train = data.get("train", [])[:num_samples // 5]
        dev = data.get("dev", data.get("validation", []))[:num_samples // 2]
        test = data.get("test", [])[:num_samples // 2]
    else:
        raise DatasetError(f"Unexpected JSON structure in {path}")

    task_type = _detect_task_type(train) if train else "auto"

    return TaskDataset(
        name=path.stem,
        train_samples=train,
        dev_samples=dev,
        test_samples=test,
        task_type=task_type,
        metadata={"source": "json", "path": str(path)},
    )


def _detect_task_type(samples: List[Dict[str, str]]) -> str:
    """Auto-detect task type from sample targets."""
    if not samples:
        return "auto"

    targets = [s["target"].strip().lower() for s in samples[:20]]

    # Check boolean (includes valid/invalid, e.g. BBH formal_fallacies)
    bool_values = {"true", "false", "yes", "no", "valid", "invalid"}
    if all(t in bool_values for t in targets):
        return "boolean"

    # Check MCQ: single letter with optional parens — "A", "(A)", "a)"
    import re
    if all(re.fullmatch(r"\(?[a-z]\)?", t) for t in targets):
        return "mcq"

    # Check numeric
    numeric_count = 0
    for t in targets:
        try:
            float(t)
            numeric_count += 1
        except ValueError:
            pass
    if numeric_count > len(targets) * 0.8:
        return "math"

    return "auto"