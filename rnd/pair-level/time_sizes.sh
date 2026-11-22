#!/bin/sh
# rnd/pair-level, sizing for equal wall-clock time: the GRU 154 and pair
# models at the widest shape's ratio (level 1 : decoder = 80 : 98) at three
# scales, each over the first 250,000 training characters at window 256 and
# 1e-3, three at a time as in run_compute.sh. Prints each one's seconds to
# its first held-out score. CPU with the GPU hidden, a few minutes.
#   sh rnd/pair-level/time_sizes.sh   (from the repo; writes results/timing/)
renice -n 19 -p $$ >/dev/null
ionice -c 3 -p $$
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
cd "$(dirname "$0")"
mkdir -p results/timing
t() {  # name, then the command's flags
  name=$1
  shift
  "$@" --window 256 --lr 1e-3 --train-chars 250002 --eval-every 250000 --out "results/timing/$name.json" > "results/timing/$name.log" 2>&1
}
t gru154 python3 ../two-char-updates/tiny_tbptt.py --hidden 154 --eval-group 2 &
a=$!
t u120_l147 python3 tiny_pairs.py --upper 120 --local 147 &
b=$!
t u160_l196 python3 tiny_pairs.py --upper 160 --local 196 &
c=$!
wait $a $b $c
t u200_l245 python3 tiny_pairs.py --upper 200 --local 245 &
a=$!
t u240_l294 python3 tiny_pairs.py --upper 240 --local 294 &
b=$!
t gru154_again python3 ../two-char-updates/tiny_tbptt.py --hidden 154 --eval-group 2 &
c=$!
wait $a $b $c
for f in results/timing/*.json; do
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1].split('/')[-1][:-5], d['params'], d['curve'][1]['sec'], round(d['curve'][1]['heldout_ppl'], 3))" "$f"
done
