#!/bin/sh
# imm's slot $1: the depth-width screen. Slot 137's 10.8M imm (depth records,
# 1,024 character records, no prior, dropout 0, lr 1e-3 cosine to 1e-4 over 8
# epochs, --bf16, scored in fp32) with its parameters spent on depth: 6 blocks
# at width 344 (8 heads, word tier 301; 10,730,974 trained parameters) instead
# of 3 at 448 (10,913,495). Stopped after epoch 3 (--stop-after-epoch 3), read
# against slot 137's epoch 3, 2.9342 (epoch 2: 3.0100).
#  1. probe_depth_width.py: the --max-update-chars cap under 6.9 GiB, the
#     run's text on the GPU, and the time an update at that cap.
#     (2026-11-22: stops if the 3 epochs would take over 4 hours.)
#  2. the screen, at the cap; the epoch-3 best's per-character losses in fp32.
# Fixed worktree $S/imm-dw at 6ce0302 (branch depth-width: depth-records
# 5ae5c64 plus --stop-after-epoch).
# Stops and ends the slot at any failure. RSS and GPU memory every minute,
# with a guard that stops the run below 800 MB available.
# Laptop limits: nice 19, ionice idle, 2 CPU threads.
renice -n 19 -p $$ >/dev/null
ionice -c 3 -p $$
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
export TMPDIR=$S PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SLOTS="sh /home/trans/Projects/air/tools/slots"
sid=$1
finish() { kill $rss 2>/dev/null; $SLOTS done @imm; echo "slot ended $(date -Is)"; exit $1; }
while ! $SLOTS list 2>/dev/null | grep -q "^in use: $sid @imm,"; do
  if ! $SLOTS list 2>/dev/null | grep -qE "^(in use: |[0-9]+\. )$sid @imm,"; then
    echo "slot $sid is neither queued nor in use $(date -Is)"; exit 1
  fi
  sleep 20
done
echo "slot $sid granted $(date -Is)"
cd $S/imm-dw || exit 1
echo "code $(git rev-parse --short HEAD) $(git status --porcelain --untracked-files=no | wc -l) changed files"
(while :; do
  avail=$(free -m | awk '/^Mem/ {print $7}')
  line="$(date -u +%H:%M:%S) available=${avail}MB $(free -m | awk '/^Swap/ {printf "swap_used=%dMB", $3}') gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr -d ' ')"
  for p in $(pgrep -f "run_segment_memory_model.py --config"); do
    [ -r /proc/$p/status ] && line="$line | pid=$p $(awk '/^(VmRSS|RssAnon|VmSwap):/ {printf "%s=%dMB ", $1, $2/1024}' /proc/$p/status)"
  done
  echo "$line"
  if [ "$avail" -lt 800 ]; then
    echo "GUARD: available ${avail}MB < 800MB, stopping the run"; pkill -f "run_segment_memory_model.py --config"
  fi
  sleep 60
done) > $S/dw/rss_$sid.log 2>&1 &
rss=$!

echo "probe start $(date -Is)"
python3 -B $S/dw/probe_depth_width.py $S/s10/worst_window.txt > $S/dw/probe_$sid.log 2>&1
echo "probe exit=$? $(date -Is)"
grep -v "InitGoogleLogging\|CUDACachingAllocator" $S/dw/probe_$sid.log
choice=$(grep '^CHOICE' $S/dw/probe_$sid.log)
case "$choice" in
  "CHOICE h=344 "*) ;;
  *) echo "no cap fits ($choice): stopping"; finish 1 ;;
esac
cap=$(echo "$choice" | sed 's/.*cap=\([0-9]*\).*/\1/')
echo "cap=$cap"
# A small cap splits every update into more groups, which is slower. Past 4
# hours for the 3 epochs, stop: a screen that slow waits for the user.
hours=$(grep -o '3 epochs ~[0-9.]* h' $S/dw/probe_$sid.log | head -1 | sed 's/3 epochs ~\([0-9.]*\) h/\1/')
echo "3 epochs ~${hours} h at cap $cap"
case "$hours" in
  ''|*[!0-9.]*) echo "no time estimate: stopping"; finish 1 ;;
esac
if [ "$(echo "$hours > 4.0" | bc)" = 1 ]; then
  echo "3 epochs would take ~${hours} h, over 4: stopping"; finish 1
fi

cfg=configs/pg19split100m_noprior.json
base="--count-prior none --prior-residual-gate none --prior-input none --prior-residual-state-features= --prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --eval-streams 64 --lr 1e-3 --segment-rule word --max-memory 128 --segments-per-update 64 --dropout 0.0 --lr-schedule cosine --min-lr 1e-4 --carry-bound gru --memory-state gru --prefetch-ahead 4 --tier-own-step --char-memory 1024 --documents data/pg19/imm-split-100m/manifest.json --epochs 8"

# A bf16 run, then its best checkpoint's per-character losses in fp32.
train() {  # run id, model flags, training-only flags
  echo "start imm $1 $(date -Is)"
  echo "  args: $2 --bf16 $3"
  python -B scripts/run_segment_memory_model.py --config $cfg --run-id $1 --output runs/$1.csv $2 --bf16 $3 > runs/$1.log 2>&1
  rc=$?
  echo "end imm $1 exit=$rc $(date -Is) $(grep -o 'best_val_ppl=[0-9.]*' runs/$1.log | tail -1)"
  [ $rc -eq 0 ] || return $rc
  mkdir -p $S/charloss
  [ -f runs/${1}_${1}.best.pt ] || return 1
  python -B scripts/run_segment_memory_model.py --config $cfg --run-id cl_$1 --output $S/charloss/cl_$1.csv $2 \
    --eval-only --resume-checkpoint runs/${1}_${1}.best.pt --char-loss-output runs/char_losses_${1}_best.npz > $S/charloss/$1.log 2>&1
  rc=$?
  echo "char losses $1 (fp32) exit=$rc $(grep -o 'char_losses heldout_ppl=[0-9.]*' $S/charloss/$1.log) $(date -Is)"
  return $rc
}
train pg19s100m_h344_par_l6_w301_own_np_doc_mm128_do0_cm1024_depth_bf16_e8s3 \
  "$base --hidden-size 344 --embedding-size 344 --attention-layers 6 --attention-heads 8 --word-tier 301 --depth-records" \
  "--max-update-chars $cap --stop-after-epoch 3" || finish 1
finish 0
