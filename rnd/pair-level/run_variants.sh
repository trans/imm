#!/bin/sh
# rnd/pair-level, the decoder screen: four shapes of the pair model at the
# GRU 128's size, window 256, at 3e-4, 1e-3 and 3e-3. One lane a shape, its
# three rates in turn: four lanes, ~0.7 GB each, CPU with the GPU hidden,
# about 12 minutes. A guard stops the arms if the machine's available memory
# falls under 1.9 GB.
#   sh rnd/pair-level/run_variants.sh   (from the repo; writes results/)
renice -n 19 -p $$ >/dev/null
ionice -c 3 -p $$
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
cd "$(dirname "$0")"
mkdir -p results
rm -f results/run_variants.stopped

lane() {  # name, then the shape's flags
  name=$1
  shift
  for lr in 3e-4 1e-3 3e-3; do
    [ -e results/run_variants.stopped ] && return
    python3 tiny_pairs.py --window 256 --lr "$lr" --train-chars 2000000 "$@" \
      --out "results/${name}_w256_lr$lr.json" > "results/${name}_w256_lr$lr.log" 2>&1
  done
}

guard() {
  while :; do
    kb=$(awk '/^MemAvailable/ {print $2}' /proc/meminfo)
    if [ "$kb" -lt 1900000 ]; then
      echo "$(date -Iseconds) MemAvailable $kb kB: stopping the shape arms" | tee results/run_variants.stopped
      pkill -f 'tiny_pairs.py --window'
      return
    fi
    sleep 5
  done
}

guard &
g=$!
lane u80_l64 --upper 80 --local 64 &
a=$!
lane u66_l80 --upper 66 --local 80 &
b=$!
lane u90_l42_feed --upper 90 --local 42 --feed &
c=$!
lane u75_l64_feed --upper 75 --local 64 --feed &
d=$!
wait $a $b $c $d
kill $g 2>/dev/null
echo "all done $(date -Iseconds)"
