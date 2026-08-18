# Ad-hoc scripts

Historical, one-off scripts kept for research provenance. They are **not**
officially supported entry points into APOBench — flags, hardcoded paths, and
assumptions may be specific to a particular past run. If you're looking for a
documented, generic starting point, see [`examples/`](../../examples/) instead.

| Script | What it was for |
|---|---|
| `bbh_reference_baseline.py` | Reproduces the Suzgun et al. (2023) BBH Table 3 comparison on this harness. |
| `convert_full_benchmark.py` | One-off converter between two result-directory layouts. |
| `repair_test_scores.py` | Retroactively backfills missing test-set scores into existing result files. |
| `run_baseline_gsm8k.py` | Zero-search seed-prompt baseline across the three original benchmarks. |
| `run_fill_gaps.py` | Fills specific missing `(method, dataset, seed)` slots in one named results file. |
| `run_gsm8k_rerun.py` | GSM8K + HumanEval re-run with corrected token limits. |
| `run_swift_apex_rerun.py` | Re-run variant listing specific bug fixes it incorporates. |

All of these import from `experiments/run_swift_apex.py` and expect to be run
from the repository root, e.g. `python experiments/adhoc/run_fill_gaps.py ...`.
