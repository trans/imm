#!/bin/sh
# rnd/two-char-updates: a tiny character GRU (93,024 parameters), one stream,
# the first 2M characters of the 100M split's training text, its weights
# updated every 2 characters, against a 256-character window, three learning
# rates each. CPU, 2 threads, about 1 GB: no GPU slot. About an hour.
#   sh rnd/two-char-updates/run.sh   (from the repo; writes results/k<K>_lr<LR>.json)
renice -n 19 -p $$ >/dev/null
ionice -c 3 -p $$
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$(dirname "$0")"
mkdir -p results
for spec in "256 3e-4" "256 1e-3" "256 3e-3" "2 1e-4" "2 3e-4" "2 1e-3" "2 3e-5"; do
  set -- $spec
  echo "start window=$1 lr=$2 $(date -Iseconds)"
  python3 tiny_tbptt.py --window "$1" --lr "$2" --train-chars 2000000 --out "results/k$1_lr$2.json"
  rc=$?
  echo "end window=$1 lr=$2 exit=$rc $(date -Iseconds)"
done
echo "all done $(date -Iseconds)"
