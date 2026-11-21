#!/bin/sh
# imm's slot $1: a 1,024-character window (--char-memory 1024, read in runs
# of 128: --char-chunk, branch char-records fe7dfb9; the user, 2026-11-19,
# chose to keep to the plan). Slot 113's imm (3.1519 with 256) with each
# character's window 1,024 long, as the block-1,024 transformer's. First
# chunk_probe.py: ms/update and peak GPU memory of the old and new reads;
# the run only if N=1024 C=128 fits. Then its per-character held-out
# losses. From worktree imm-char. RSS and GPU memory every minute, with a
# guard that stops the run below 800 MB available. Ends the slot.
# Laptop limits: nice 19, ionice idle, 2 CPU threads.
renice -n 19 -p $$ >/dev/null
ionice -c 3 -p $$
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
export TMPDIR=$S
SLOTS="sh /home/trans/Projects/air/tools/slots"
sid=$1
while ! $SLOTS list 2>/dev/null | grep -q "^in use: $sid @imm,"; do
  if ! $SLOTS list 2>/dev/null | grep -qE "^(in use: |[0-9]+\. )$sid @imm,"; then
    echo "slot $sid is neither queued nor in use $(date -Is)"; exit 1
  fi
  sleep 20
done
echo "slot $sid granted $(date -Is)"
cd $S/imm-char || exit 1
echo "code $(git rev-parse --short HEAD) $(git status --porcelain --untracked-files=no | wc -l) changed files"
python -B $S/scale/chunk_probe.py > $S/scale/chunk_probe_$sid.log 2>&1
echo "probe exit=$? $(date -Is)"
cat $S/scale/chunk_probe_$sid.log
if ! grep -q "^N= 1024 C= 128: .*ms/update" $S/scale/chunk_probe_$sid.log; then
  echo "N=1024 C=128 didn't run in the probe: no run"; $SLOTS done @imm; echo "slot ended $(date -Is)"; exit 1
fi
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
done) > $S/scale/rss_$sid.log 2>&1 &
rss=$!
cfg=configs/pg19split100m_noprior.json
id=pg19s100m_h192_par_l2_w168_own_np_doc_mm128_do0_cm1024_e8
common="--count-prior none --prior-residual-gate none --prior-input none --prior-residual-state-features= --prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --eval-streams 64 --lr 1e-3 --segment-rule word --max-memory 128 --segments-per-update 64 --dropout 0.0 --lr-schedule cosine --min-lr 1e-4 --carry-bound gru --memory-state gru --hidden-size 192 --embedding-size 192 --prefetch-ahead 4 --attention-layers 2 --word-tier 168 --tier-own-step --char-memory 1024 --char-chunk 128 --documents data/pg19/imm-split-100m/manifest.json --epochs 8"
echo "start imm $id $(date -Is)"
echo "  args: $common"
python -B scripts/run_segment_memory_model.py --config $cfg --run-id $id --output runs/$id.csv $common > runs/$id.log 2>&1
rc=$?
echo "end imm exit=$rc $(date -Is) $(grep -o 'best_val_ppl=[0-9.]*' runs/$id.log | tail -1)"
mkdir -p $S/charloss
if [ $rc -eq 0 ] && [ -f runs/${id}_${id}.best.pt ]; then
  python -B scripts/run_segment_memory_model.py --config $cfg --run-id cl_$id --output $S/charloss/cl_$id.csv $common \
    --eval-only --resume-checkpoint runs/${id}_${id}.best.pt --char-loss-output runs/char_losses_${id}_best.npz > $S/charloss/$id.log 2>&1
  echo "char losses imm exit=$? $(grep -o 'char_losses heldout_ppl=[0-9.]*' $S/charloss/$id.log) $(date -Is)"
fi
kill $rss
$SLOTS done @imm
echo "slot ended $(date -Is)"
