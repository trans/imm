#!/bin/sh
# imm's slot $1: the attention-only screens (the user, 2026-11-20: explore
# hierarchical attention on its own, read at epoch 3). No character GRU
# (--rnn-core none): each block reads character records rebuilt from the
# block below (--depth-records), h192 x 4 blocks, 4 heads of 48, the 1.84M
# transformer's shape, 1,892,264 trained parameters in both arms; no prior,
# dropout 0, lr 1e-3 cosine to 1e-4 over 8 epochs, stopped after epoch 3
# (--stop-after-epoch 3), --bf16, the epoch-3 best scored in fp32.
#   A: 1,024 character records (in effect a transformer reaching 1,024).
#   B: 512 character records, and word records for the words before them
#      (--words-beyond-chars), the last 600 words, each the mean of the
#      block's input over the word (--word-pool mean, --memory-record raw:
#      no writer, so no new parameters). About the same records per
#      character as A, reaching ~3,300 characters.
#  1. $S/arch/probe_ao.py: each arm's --max-update-chars cap under 6.9 GiB.
#  2. A, then B, at their caps; per-character losses in fp32.
# B against A at epoch 3 is the screen.
# Fixed worktree $S/imm-ao (branch attention-only).
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
cd $S/imm-ao || exit 1
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
done) > $S/arch/rss_$sid.log 2>&1 &
rss=$!

echo "probe start $(date -Is)"
python3 -B $S/arch/probe_ao.py $S/s10/worst_window.txt A B > $S/arch/probe_ao_$sid.log 2>&1
echo "probe exit=$? $(date -Is)"
grep -v "InitGoogleLogging\|CUDACachingAllocator" $S/arch/probe_ao_$sid.log
capA=$(sed -n 's/^CHOICE arm=A cap=\([0-9][0-9]*\)$/\1/p' $S/arch/probe_ao_$sid.log)
capB=$(sed -n 's/^CHOICE arm=B cap=\([0-9][0-9]*\)$/\1/p' $S/arch/probe_ao_$sid.log)
[ -n "$capA" ] && [ -n "$capB" ] || { echo "no cap fits for an arm (A=$capA B=$capB): stopping"; finish 1; }
echo "capA=$capA capB=$capB"

cfg=configs/pg19split100m_noprior.json
base="--count-prior none --prior-residual-gate none --prior-input none --prior-residual-state-features= --prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --eval-streams 64 --lr 1e-3 --segment-rule word --max-memory 128 --segments-per-update 64 --dropout 0.0 --lr-schedule cosine --min-lr 1e-4 --carry-bound gru --memory-state gru --prefetch-ahead 4 --char-memory 1024 --documents data/pg19/imm-split-100m/manifest.json --epochs 8"
alone="$base --hidden-size 192 --embedding-size 192 --attention-layers 4 --attention-heads 4 --rnn-core none --depth-records"
h=192 heads=4
[ $((h / heads % 8)) -eq 0 ] || { echo "head width $((h / heads)) is not a multiple of 8"; finish 1; }

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
train pg19s100m_ao_h192_l4_cm1024_depth_bf16_e8s3 \
  "$alone --no-word-records" "--max-update-chars $capA --stop-after-epoch 3" || finish 1
train pg19s100m_ao_h192_l4_cm512_w600mean_beyond_raw_depth_bf16_e8s3 \
  "$alone --char-memory 512 --max-memory 600 --word-pool mean --words-beyond-chars --memory-record raw" \
  "--max-update-chars $capB --stop-after-epoch 3" || finish 1
finish 0
