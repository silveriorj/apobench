# ENIAC 2026 — accepted paper results

Raw result data underlying the APOBench paper accepted at ENIAC 2026
(`eniac2026_raul.pdf`, included here). Seven published APO methods
(GAAPO, SEE, CriSPO, DSGE-Evo, PromptBreeder, PLUM, HPSS — this repo's
`gaapo`/`see`/`capo`/`gepa`/`baseline_seed` are later, separately-tiered
reimplementations and are **not** the methods measured here) plus a
zero-search baseline, evaluated on BBH, GSM8K, HumanEval, and MMLU-Pro with
`Qwen/Qwen3-4B-Instruct-2507`.

## Files and what they feed

| File | Protocol | Feeds |
|---|---|---|
| `bbh_5seed_summary.csv` | BBH, 8 tasks, 5 seeds | Table 5's BBH column, Table 6 (per-task BBH) |
| `gsm8k_humaneval_mmlu_3seed_summary.csv` | GSM8K/HumanEval/MMLU-Pro, 3 seeds | Table 5's GSM8K and HumanEval columns |
| `baseline_3seed_summary.csv` | Zero-search baseline, 3 seeds, per benchmark | Table 5's Baseline row |

Each row is a `(optimizer, benchmark, task)` cell: mean/std test score across
seeds, mean tokens and wall-clock time, mean generations, and error count.
Table 5 in the paper reports the mean across the 8 BBH tasks; the per-task
numbers here are the ones that mean is computed from.

Spot-checked against the published paper: GAAPO — BBH 0.5997→0.600, GSM8K
0.8533→0.853, HumanEval 0.7778→0.778; baseline — GSM8K 0.46, HumanEval
0.6667→0.667. All match Table 5 exactly.

**One gap, stated plainly:** the paper's baseline BBH figure (0.344) isn't
reproduced byte-for-byte by any raw file recovered for this release —
`baseline_3seed_summary.csv`'s BBH row is a single aggregated task type from
an earlier smoke-test-scale run, not the 8-task breakdown the other two files
use, and doesn't average to 0.344. The other five baseline/method numbers in
Table 5 all verified exactly, so this is very likely a file that wasn't
retained rather than an error in the paper — noted here instead of silently
omitted or overclaimed.

## Citing

See [`../../CITATION.cff`](../../CITATION.cff) at the repo root.
