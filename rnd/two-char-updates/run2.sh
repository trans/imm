#!/bin/sh
# rnd/two-char-updates, stage 2: level 1 on top of the character GRU, each
# level updated on its own two steps (tiny_levels.py), against a character
# GRU of the same size updated every 2 characters (tiny_tbptt.py --hidden 215),
# each at learning rates 3e-5 and 1e-4. The four arms run side by side,
# 2 threads and about 1 GB each, CPU. About 40 minutes.
#   sh rnd/two-char-updates/run2.sh   (from the repo; writes results/)
renice -n 19 -p $$ >/dev/null
ionice -c 3 -p $$
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$(dirname "$0")"
mkdir -p results
for lr in 3e-5 1e-4; do
  python3 tiny_levels.py --lr "$lr" --train-chars 2000000 --out "results/levels_lr$lr.json" > "results/levels_lr$lr.log" 2>&1 &
  python3 tiny_tbptt.py --window 2 --hidden 215 --lr "$lr" --train-chars 2000000 --out "results/k2_h215_lr$lr.json" > "results/k2_h215_lr$lr.log" 2>&1 &
done
wait
echo "all done $(date -Iseconds)"
