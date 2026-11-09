#!/bin/sh
# Rerun the residual with the training prior cross-fitted (5a49012), against
# the leaky prior, at prior strength 1.0 and 0.75: a 2x2 on Tiny Shakespeare's
# canonical split, then on Gutenberg 5M. One run at a time (a Gutenberg run
# peaks near 2.6 GB of RAM and takes about 20 minutes on a GPU). Each corpus's first
# cross-fitted run builds its prior cache; the rest reuse it. After each
# corpus, scripts/prior_row_ppl.py reports the prior on training rows
# against held-out for both caches.
#
#   scripts/rerun_prior_leak.sh [tiny|gutenberg|all]   (default all)
set -e
which=${1:-all}
cd "$(dirname "$0")/.."
run() { python scripts/run_segment_memory_model.py --config "configs/$1.json"; }

if [ "$which" = tiny ] || [ "$which" = all ]; then
for c in tiny_canonical_leaky_ps075_e1 tiny_canonical_leaky_ps100_e1 \
         tiny_canonical_crossfit20_ps075_e1 tiny_canonical_crossfit20_ps100_e1; do
  run "$c"
done
python scripts/prior_row_ppl.py --config configs/tiny_canonical_leaky_ps075_e1.json
python scripts/prior_row_ppl.py --config configs/tiny_canonical_crossfit20_ps075_e1.json
fi

if [ "$which" = gutenberg ] || [ "$which" = all ]; then
for c in gutenberg5m_crossfit20_ps075_e1 gutenberg5m_crossfit20_ps100_e1 \
         gutenberg5m_leaky_ps100_e1 gutenberg5m_priorstrength075_residual025_l2_005_e1; do
  run "$c"
done
python scripts/prior_row_ppl.py --config configs/gutenberg5m_leaky_ps100_e1.json
python scripts/prior_row_ppl.py --config configs/gutenberg5m_crossfit20_ps075_e1.json
fi
