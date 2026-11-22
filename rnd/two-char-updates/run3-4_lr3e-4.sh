#!/bin/sh
# rnd/two-char-updates, stages 3 and 4 at learning rate 3e-4: both stage-3
# models were best at the higher of their two rates (1e-4), so each model of
# stages 3 and 4 also runs at the next rate up. Two lanes side by side, ~0.9
# GB each: the detached model then the control, and the attached model. CPU,
# about 50 minutes. A guard stops the arms if the machine's available memory
# falls under 1.9 GB.
#   sh rnd/two-char-updates/run3-4_lr3e-4.sh   (from the repo; writes results/)
renice -n 19 -p $$ >/dev/null
ionice -c 3 -p $$
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$(dirname "$0")"
mkdir -p results
rm -f results/run3-4.stopped
lr=3e-4

lane_a() {
  python3 tiny_levels.py --char-window 4 --level-window 4 --eval-group 2 --lr "$lr" --train-chars 2000000 \
    --out "results/joint4_lr$lr.json" > "results/joint4_lr$lr.log" 2>&1
  [ -e results/run3-4.stopped ] && return
  python3 tiny_tbptt.py --window 4 --hidden 215 --eval-group 2 --lr "$lr" --train-chars 2000000 \
    --out "results/k4_h215_lr$lr.json" > "results/k4_h215_lr$lr.log" 2>&1
}

lane_b() {
  python3 tiny_levels.py --char-window 4 --level-window 4 --attach --eval-group 2 --lr "$lr" --train-chars 2000000 \
    --out "results/joint4att_lr$lr.json" > "results/joint4att_lr$lr.log" 2>&1
}

guard() {
  while :; do
    kb=$(awk '/^MemAvailable/ {print $2}' /proc/meminfo)
    if [ "$kb" -lt 1900000 ]; then
      echo "$(date -Iseconds) MemAvailable $kb kB: stopping the 3e-4 arms" | tee results/run3-4.stopped
      pkill -f 'tiny_levels.py --char-window 4 --level-window 4'
      pkill -f 'tiny_tbptt.py --window 4 --hidden 215'
      return
    fi
    sleep 5
  done
}

guard &
g=$!
lane_a &
a=$!
lane_b &
b=$!
wait $a $b
kill $g 2>/dev/null
echo "all done $(date -Iseconds)"
