#!/bin/sh
# rnd/pair-level, the equal-compute arms: the pair model at the GRU 128's
# compute (level 1 80, decoder 98) and the character GRU of its size (154),
# each at window 256 (3e-4, 1e-3, 3e-3) and window 4 (3e-5, 1e-4, 3e-4).
# Three lanes, one a rate pair, each running its window-256 arms then its
# window-4 arms: ~0.7 GB a lane, CPU with the GPU hidden, about an hour. A
# guard stops the arms if the machine's available memory falls under 1.9 GB.
#   sh rnd/pair-level/run_compute.sh   (from the repo; writes results/)
renice -n 19 -p $$ >/dev/null
ionice -c 3 -p $$
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
cd "$(dirname "$0")"
mkdir -p results
rm -f results/run_compute.stopped

pair() {  # window, rate
  [ -e results/run_compute.stopped ] && return
  python3 tiny_pairs.py --window "$1" --lr "$2" --train-chars 2000000 --upper 80 --local 98 \
    --out "results/u80_l98_w$1_lr$2.json" > "results/u80_l98_w$1_lr$2.log" 2>&1
}

gru() {  # window, rate
  [ -e results/run_compute.stopped ] && return
  python3 ../two-char-updates/tiny_tbptt.py --window "$1" --lr "$2" --train-chars 2000000 --hidden 154 --eval-group 2 \
    --out "results/gru154_w$1_lr$2.json" > "results/gru154_w$1_lr$2.log" 2>&1
}

lane() {  # the window-256 rate, the window-4 rate
  pair 256 "$1"
  gru 256 "$1"
  pair 4 "$2"
  gru 4 "$2"
}

guard() {
  while :; do
    kb=$(awk '/^MemAvailable/ {print $2}' /proc/meminfo)
    if [ "$kb" -lt 1900000 ]; then
      echo "$(date -Iseconds) MemAvailable $kb kB: stopping the equal-compute arms" | tee results/run_compute.stopped
      pkill -f 'tiny_pairs.py --window'
      pkill -f 'tiny_tbptt.py --window [0-9]* --lr [0-9e.-]* --train-chars 2000000 --hidden 154'
      return
    fi
    sleep 5
  done
}

guard &
g=$!
lane 3e-4 3e-5 &
a=$!
lane 1e-3 1e-4 &
b=$!
lane 3e-3 3e-4 &
c=$!
wait $a $b $c
kill $g 2>/dev/null
echo "all done $(date -Iseconds)"
