#!/bin/sh
# rnd/pair-level, equal wall-clock time: the pair model at level 1 280,
# decoder 343 (sized by time_sizes.sh to train in about the GRU 154's time),
# window 256, at 3e-4, 1e-3 and 3e-3, three at once as the GRU 154's arms
# ran. ~0.8 GB each, CPU with the GPU hidden, about 7 minutes. A guard stops
# the arms if the machine's available memory falls under 1.9 GB.
#   sh rnd/pair-level/run_time.sh   (from the repo; writes results/)
renice -n 19 -p $$ >/dev/null
ionice -c 3 -p $$
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
cd "$(dirname "$0")"
mkdir -p results
rm -f results/run_time.stopped
arm() {
  python3 tiny_pairs.py --window 256 --lr "$1" --train-chars 2000000 --upper 280 --local 343 \
    --out "results/u280_l343_w256_lr$1.json" > "results/u280_l343_w256_lr$1.log" 2>&1
}
guard() {
  while :; do
    kb=$(awk '/^MemAvailable/ {print $2}' /proc/meminfo)
    if [ "$kb" -lt 1900000 ]; then
      echo "$(date -Iseconds) MemAvailable $kb kB: stopping the equal-time arms" | tee results/run_time.stopped
      pkill -f 'tiny_pairs.py --window'
      return
    fi
    sleep 5
  done
}
guard &
g=$!
arm 3e-4 &
a=$!
arm 1e-3 &
b=$!
arm 3e-3 &
c=$!
wait $a $b $c
kill $g 2>/dev/null
echo "all done $(date -Iseconds)"
