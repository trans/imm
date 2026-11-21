#!/bin/sh
# imm's slot $1: the GRU-only tree screen (rnd/gru-tree; the user approved
# the two arms on 2026-11-21). No prior, no attention, no records, dropout 0,
# lr 1e-3 cosine to 1e-4 over 8 epochs, stopped after epoch 3, --bf16, the
# epoch-3 best scored in fp32.
#   T: the binary tree, h192 e192, --tree-levels 10 --tree-width 132,
#      1,593,298 trained parameters.
#   G: the plain character GRU at the same size, h498 e498, no tree,
#      1,587,222 trained: the control.
# Read against each other, and against epoch 3 of slot 127 (3.233), slot
# 130's depth records (3.178), slot 142's attention alone (3.423) and slot
# 130's GRUs without attention (3.532). Watch T's tree_gates per level.
#  1. probe_tree.py: each arm's --max-update-chars cap under 6.9 GiB.
#  2. T, then G, at their caps; per-character losses in fp32.
# Fixed worktree $S/imm-tree (branch gru-tree, 8818093).
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
cd $S/imm-tree || exit 1
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
done) > $S/tree/rss_$sid.log 2>&1 &
rss=$!

echo "probe start $(date -Is)"
python3 -B $S/tree/probe_tree.py $S/s10/worst_window.txt T G > $S/tree/probe_tree_$sid.log 2>&1
echo "probe exit=$? $(date -Is)"
grep -v "InitGoogleLogging\|CUDACachingAllocator" $S/tree/probe_tree_$sid.log
capT=$(sed -n 's/^CHOICE arm=T cap=\([0-9][0-9]*\)$/\1/p' $S/tree/probe_tree_$sid.log)
capG=$(sed -n 's/^CHOICE arm=G cap=\([0-9][0-9]*\)$/\1/p' $S/tree/probe_tree_$sid.log)
[ -n "$capT" ] && [ -n "$capG" ] || { echo "no cap fits for an arm (T=$capT G=$capG): stopping"; finish 1; }
echo "capT=$capT capG=$capG"

cfg=configs/pg19split100m_noprior.json
base="--count-prior none --prior-residual-gate none --prior-input none --prior-residual-state-features= --prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --eval-streams 64 --lr 1e-3 --segment-rule word --max-memory 128 --segments-per-update 64 --dropout 0.0 --lr-schedule cosine --min-lr 1e-4 --carry-bound gru --memory-state gru --prefetch-ahead 4 --documents data/pg19/imm-split-100m/manifest.json --epochs 8"
gru="$base --attention-layers 0 --no-word-records"

# A bf16 run, then the best checkpoint's per-character losses in fp32.
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
train pg19s100m_tree_h192_t10w132_np_doc_do0_bf16_e8s3 \
  "$gru --hidden-size 192 --embedding-size 192 --tree-levels 10 --tree-width 132" \
  "--max-update-chars $capT --stop-after-epoch 3" || finish 1
grep "tree_gates=" runs/pg19s100m_tree_h192_t10w132_np_doc_do0_bf16_e8s3.log
train pg19s100m_gru_h498_np_doc_do0_bf16_e8s3 \
  "$gru --hidden-size 498 --embedding-size 498" \
  "--max-update-chars $capG --stop-after-epoch 3" || finish 1
finish 0
