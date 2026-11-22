#!/bin/sh
# rnd/two-char-updates, stage 3: the stage-2 model with both levels updated
# together every 4 characters (tiny_levels.py --char-window 4 --level-window 4),
# against a character GRU of the same size updated every 4 characters
# (tiny_tbptt.py --window 4 --hidden 215), each at learning rates 3e-5 and
# 1e-4. Two lanes side by side, one per rate, each running its level-1 arm
# and then its control: 2 threads and ~0.9 GB a lane, CPU, about 50 minutes.
# A guard stops the arms, and the run, if the machine's available memory
# falls under 1.9 GB, to leave room for other projects' jobs.
#   sh rnd/two-char-updates/run3.sh   (from the repo; writes results/)
renice -n 19 -p $$ >/dev/null
ionice -c 3 -p $$
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$(dirname "$0")"
mkdir -p results
rm -f results/run3.stopped

lane() {
  lr=$1
  python3 tiny_levels.py --char-window 4 --level-window 4 --eval-group 2 --lr "$lr" --train-chars 2000000 \
    --out "results/joint4_lr$lr.json" > "results/joint4_lr$lr.log" 2>&1
  [ -e results/run3.stopped ] && return
  python3 tiny_tbptt.py --window 4 --hidden 215 --eval-group 2 --lr "$lr" --train-chars 2000000 \
    --out "results/k4_h215_lr$lr.json" > "results/k4_h215_lr$lr.log" 2>&1
}

guard() {
  while :; do
    kb=$(awk '/^MemAvailable/ {print $2}' /proc/meminfo)
    if [ "$kb" -lt 1900000 ]; then
      echo "$(date -Iseconds) MemAvailable $kb kB: stopping the stage-3 arms" | tee results/run3.stopped
      pkill -f 'tiny_levels.py --char-window 4 --level-window 4'
      pkill -f 'tiny_tbptt.py --window 4 --hidden 215'
      return
    fi
    sleep 5
  done
}

guard &
g=$!
lane 3e-5 &
a=$!
lane 1e-4 &
b=$!
wait $a $b
kill $g 2>/dev/null
echo "all done $(date -Iseconds)"
