#!/bin/sh
# rnd/pair-level, equal time checked under one load: the GRU 154 and the
# pair model at level 1 280 / decoder 343 (both again) and at 240 / 294,
# window 256, 1e-3, side by side. ~0.8 GB each, CPU with the GPU hidden,
# about 8 minutes.
#   sh rnd/pair-level/run_time_check.sh   (from the repo; writes results/)
renice -n 19 -p $$ >/dev/null
ionice -c 3 -p $$
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
cd "$(dirname "$0")"
mkdir -p results
python3 ../two-char-updates/tiny_tbptt.py --window 256 --lr 1e-3 --train-chars 2000000 --hidden 154 --eval-group 2 \
  --out results/gru154_w256_lr1e-3_together.json > results/gru154_w256_lr1e-3_together.log 2>&1 &
a=$!
python3 tiny_pairs.py --window 256 --lr 1e-3 --train-chars 2000000 --upper 280 --local 343 \
  --out results/u280_l343_w256_lr1e-3_together.json > results/u280_l343_w256_lr1e-3_together.log 2>&1 &
b=$!
python3 tiny_pairs.py --window 256 --lr 1e-3 --train-chars 2000000 --upper 240 --local 294 \
  --out results/u240_l294_w256_lr1e-3.json > results/u240_l294_w256_lr1e-3.log 2>&1 &
c=$!
wait $a $b $c
echo "all done $(date -Iseconds)"
