#!/bin/sh
# rnd/pair-level: the pair model (tiny_pairs.py) at window 4 (3e-5, 1e-4,
# 3e-4) and window 256 (3e-4, 1e-3, 3e-3). Four lanes: the three window-4
# arms side by side, and the three window-256 arms in turn in the fourth.
# ~0.7 GB a lane, CPU with the GPU hidden, about 25 minutes. A guard stops the
# arms if the machine's available memory falls under 1.9 GB.
#   sh rnd/pair-level/run.sh   (from the repo; writes results/)
renice -n 19 -p $$ >/dev/null
ionice -c 3 -p $$
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
cd "$(dirname "$0")"
mkdir -p results
rm -f results/run.stopped

arm() {
  w=$1
  lr=$2
  python3 tiny_pairs.py --window "$w" --lr "$lr" --train-chars 2000000 \
    --out "results/w${w}_lr$lr.json" > "results/w${w}_lr$lr.log" 2>&1
}

long_lane() {
  for lr in 3e-4 1e-3 3e-3; do
    [ -e results/run.stopped ] && return
    arm 256 "$lr"
  done
}

guard() {
  while :; do
    kb=$(awk '/^MemAvailable/ {print $2}' /proc/meminfo)
    if [ "$kb" -lt 1900000 ]; then
      echo "$(date -Iseconds) MemAvailable $kb kB: stopping the pair-level arms" | tee results/run.stopped
      pkill -f 'tiny_pairs.py --window'
      return
    fi
    sleep 5
  done
}

guard &
g=$!
arm 4 3e-5 &
a=$!
arm 4 1e-4 &
b=$!
arm 4 3e-4 &
c=$!
long_lane &
d=$!
wait $a $b $c $d
kill $g 2>/dev/null
echo "all done $(date -Iseconds)"
