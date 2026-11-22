#!/bin/sh
# rnd/two-char-updates, added after stages 3-4: the 128-wide character GRU
# alone (93,024 parameters), updated every 4 characters, at learning rates
# 3e-5, 1e-4 and 3e-4: the model stages 3-4 put level 1 on, without it. The
# three arms run side by side, 2 threads and ~0.9 GB each, CPU, about 15
# minutes. A guard stops them if the machine's available memory falls under
# 1.9 GB.
#   sh rnd/two-char-updates/run3_h128.sh   (from the repo; writes results/)
renice -n 19 -p $$ >/dev/null
ionice -c 3 -p $$
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$(dirname "$0")"
mkdir -p results
rm -f results/run3_h128.stopped

arm() {
  lr=$1
  python3 tiny_tbptt.py --window 4 --hidden 128 --eval-group 2 --lr "$lr" --train-chars 2000000 \
    --out "results/k4_lr$lr.json" > "results/k4_lr$lr.log" 2>&1
}

guard() {
  while :; do
    kb=$(awk '/^MemAvailable/ {print $2}' /proc/meminfo)
    if [ "$kb" -lt 1900000 ]; then
      echo "$(date -Iseconds) MemAvailable $kb kB: stopping the h128 arms" | tee results/run3_h128.stopped
      pkill -f 'tiny_tbptt.py --window 4 --hidden 128'
      return
    fi
    sleep 5
  done
}

guard &
g=$!
arm 3e-5 &
a=$!
arm 1e-4 &
b=$!
arm 3e-4 &
c=$!
wait $a $b $c
kill $g 2>/dev/null
echo "all done $(date -Iseconds)"
