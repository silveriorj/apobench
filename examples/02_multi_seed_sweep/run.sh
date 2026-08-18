#!/bin/bash
# APOBench sweep: the core methods x datasets x seeds protocol.
#
# This is the generic shape every real experiment in this project follows —
# fix a config (shared LLM/eval/budget settings), pick methods, datasets/tasks,
# and seeds, and run the full matrix. Compare two or more optimizers under
# IDENTICAL conditions; that comparison, repeated across seeds, is what makes
# a result citable as "under the APOBench protocol" rather than a one-off run.
set -e
cd "$(dirname "$0")/../.."   # repo root

python experiments/run_swift_apex.py \
    --methods see apex \
    --datasets bbh \
    --tasks boolean_expressions causal_judgement \
    --config examples/02_multi_seed_sweep/config.yaml \
    --seeds 42 123 7 \
    --output-dir outputs/02_multi_seed_sweep \
    "$@"

echo "=== SWEEP COMPLETE — see outputs/02_multi_seed_sweep/experiment_results.json ==="
