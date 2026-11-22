#!/bin/sh
# rnd/two-char-updates, stage 4: stage 3's model with level 1 attached, so
# its error also trains the character GRU (tiny_levels.py --char-window 4
# --level-window 4 --attach), at learning rates 3e-5 and 1e-4. Read against
# stage 3's control and detached model. The two arms run side by side,
# 2 threads and ~0.9 GB each, CPU, about 35 minutes. A guard stops them if
# the machine's available memory falls under 1.9 GB.
#   sh rnd/two-char-updates/run4.sh   (from the repo; writes results/)
renice -n 19 -p $$ >/dev/null
ionice -c 3 -p $$
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$(dirname "$0")"
mkdir -p results
rm -f results/run4.stopped

guard() {
  while :; do
    kb=$(awk '/^MemAvailable/ {print $2}' /proc/meminfo)
    if [ "$kb" -lt 1900000 ]; then
      echo "$(date -Iseconds) MemAvailable $kb kB: stopping the stage-4 arms" | tee results/run4.stopped
      pkill -f 'tiny_levels.py --char-window 4 --level-window 4 --attach'
      return
    fi
    sleep 5
  done
}

arm() {
  lr=$1
  python3 tiny_levels.py --char-window 4 --level-window 4 --attach --eval-group 2 --lr "$lr" --train-chars 2000000 \
    --out "results/joint4att_lr$lr.json" > "results/joint4att_lr$lr.log" 2>&1
}

guard &
g=$!
arm 3e-5 &
a=$!
arm 1e-4 &
b=$!
wait $a $b
kill $g 2>/dev/null
echo "all done $(date -Iseconds)"
