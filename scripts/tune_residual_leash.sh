#!/bin/sh
# The residual's leash, retuned without the leak. Its logit scale (0.25) and
# L2 (0.05) were chosen alongside prior strength 0.75 with the leaky training
# prior, and 0.75 turned out wrong (README, "Important Lesson"). Cross-fitted
# prior (20 folds), strength 1.0, seed 0, 1 epoch. One run at a time.
#
#   scripts/tune_residual_leash.sh tiny
#     Tiny canonical: scale 0.25, 0.5, 1.0 x L2 0.05, 0.01. The 0.25/0.05
#     cell is tiny_canonical_crossfit20_ps100_e1, already run. ~10 min a run.
#   scripts/tune_residual_leash.sh gutenberg SCALE L2 [EPOCHS]
#     Gutenberg at one setting: epoch 1, then resumed to EPOCHS (default 3).
#     ~21 min an epoch on a GPU.
set -e
which=${1:?tiny or gutenberg}
cd "$(dirname "$0")/.."
tag() { printf 'rs%03d_l2%03d' "$(echo "$1 * 100 / 1" | bc)" "$(echo "$2 * 100 / 1" | bc)"; }

if [ "$which" = tiny ]; then
  for setting in "0.5 0.05" "1.0 0.05" "0.25 0.01" "0.5 0.01" "1.0 0.01"; do
    set -- $setting
    id="tiny_canonical_crossfit20_ps100_$(tag "$1" "$2")_e1"
    python scripts/run_segment_memory_model.py --config configs/tiny_canonical_crossfit20_ps100_e1.json \
      --run-id "$id" --output "runs/$id.csv" --prior-residual-scale "$1" --prior-residual-l2 "$2"
  done
elif [ "$which" = gutenberg ]; then
  scale=${2:?scale}; l2=${3:?l2}; epochs=${4:-3}
  id="gutenberg5m_crossfit20_ps100_$(tag "$scale" "$l2")"
  python scripts/run_segment_memory_model.py --config configs/gutenberg5m_crossfit20_ps100_e1.json \
    --run-id "${id}_e1" --output "runs/${id}_e1.csv" --prior-residual-scale "$scale" --prior-residual-l2 "$l2"
  python scripts/run_segment_memory_model.py --config configs/gutenberg5m_crossfit20_ps100_e1.json \
    --run-id "${id}_e$epochs" --output "runs/${id}_e$epochs.csv" --prior-residual-scale "$scale" --prior-residual-l2 "$l2" \
    --epochs "$epochs" \
    --resume-checkpoint "runs/${id}_e1_${id}_e1.pt"
else
  echo "usage: $0 tiny | gutenberg SCALE L2 [EPOCHS]" >&2
  exit 2
fi
