#!/bin/sh
# rnd/pair-level, added after the first results: the best 256-window arms of
# the pair model and the character GRU 128 (both 1e-3) again, their final
# held-out scores split by place in a pair (even, odd). Side by side, ~0.7 GB
# each, CPU with the GPU hidden, about 5 minutes.
#   sh rnd/pair-level/run_by_place.sh   (from the repo; writes results/)
renice -n 19 -p $$ >/dev/null
ionice -c 3 -p $$
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
cd "$(dirname "$0")"
mkdir -p results
python3 tiny_pairs.py --window 256 --lr 1e-3 --train-chars 2000000 \
  --out results/w256_lr1e-3_by_place.json > results/w256_lr1e-3_by_place.log 2>&1 &
a=$!
python3 ../two-char-updates/tiny_tbptt.py --window 256 --lr 1e-3 --eval-group 2 --train-chars 2000000 \
  --out results/gru128_k256_lr1e-3_by_place.json > results/gru128_k256_lr1e-3_by_place.log 2>&1 &
b=$!
wait $a $b
echo "all done $(date -Iseconds)"
