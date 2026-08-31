#!/bin/bash
# APOBench sweep: methods x seeds under one fixed config.
#
# This is the generic shape every experiment in this project follows -- fix a
# config (shared LLM/eval/budget settings), pick methods and seeds, and run
# the full matrix under IDENTICAL conditions. That repetition across seeds is
# what makes a result citable as "under the APOBench protocol" rather than a
# one-off run.
set -e
cd "$(dirname "$0")/../.."   # repo root

METHODS="see gaapo"
SEEDS="42 123 7"
BASE_CONFIG="examples/02_multi_seed_sweep/config.yaml"
OUTDIR="outputs/02_multi_seed_sweep"
TMP_CONFIG=$(mktemp --suffix=.yaml)
trap 'rm -f "$TMP_CONFIG"' EXIT

for method in $METHODS; do
    for seed in $SEEDS; do
        sed "s/^seed: .*/seed: $seed/" "$BASE_CONFIG" > "$TMP_CONFIG"
        pof run -c "$TMP_CONFIG" -m "$method" \
            -o "$OUTDIR/$method/seed_$seed"
    done
done

echo "=== SWEEP COMPLETE -- see $OUTDIR/ ==="
