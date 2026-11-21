#!/bin/sh
# imm's slot $1: the first no-prior pair with both sides at dropout 0.
# Slot 105: the 1.84M transformer at dropout 0.0 scores 3.2065, against
# 3.5788 at 0.2 and imm's 3.3927 at 0.2. At 0.2 both models' training loss
# stayed above held-out.
#   imm: slot 101's m128 run (100M split, no prior, 8 epochs) at
#        --dropout 0.0, then its per-character held-out losses;
#   tx:  the user-approved block-1,024 transformer (batch 16) at dropout
#        0.0; at 0.2 it scored 3.6878.
# From worktree imm-run (topic-tiers 91c3c55). RSS sampling every minute,
# with a guard that stops the runs below 800 MB available. Ends the slot.
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
cd $S/imm-run || exit 1
echo "code $(git rev-parse --short HEAD) $(git status --porcelain --untracked-files=no | wc -l) changed files"
(while :; do
  avail=$(free -m | awk '/^Mem/ {print $7}')
  line="$(date -u +%H:%M:%S) available=${avail}MB $(free -m | awk '/^Swap/ {printf "swap_used=%dMB", $3}')"
  for p in $(pgrep -f "run_segment_memory_model.py --config|prior_transformer.py --config"); do
    [ -r /proc/$p/status ] && line="$line | pid=$p $(awk '/^(VmRSS|RssAnon|VmSwap):/ {printf "%s=%dMB ", $1, $2/1024}' /proc/$p/status)"
  done
  echo "$line"
  if [ "$avail" -lt 800 ]; then
    echo "GUARD: available ${avail}MB < 800MB, stopping the runs"; pkill -f "run_segment_memory_model.py --config|prior_transformer.py --config"
  fi
  sleep 60
done) > $S/scale/rss_$sid.log 2>&1 &
rss=$!
cfg=configs/pg19split100m_noprior.json
id=pg19s100m_h192_par_l2_w168_own_np_doc_mm128_do0_e8
common="--count-prior none --prior-residual-gate none --prior-input none --prior-residual-state-features= --prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --eval-streams 64 --lr 1e-3 --segment-rule word --max-memory 128 --segments-per-update 64 --dropout 0.0 --lr-schedule cosine --min-lr 1e-4 --carry-bound gru --memory-state gru --hidden-size 192 --embedding-size 192 --prefetch-ahead 4 --attention-layers 2 --word-tier 168 --tier-own-step --documents data/pg19/imm-split-100m/manifest.json --epochs 8"
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
tx=tx1.84m_np_100m_b1024_do0
echo "start tx $tx $(date -Is)"
(cd /home/trans/Projects/air/experiments/prior_scaling && python3 -B prior_transformer.py --config $S/imm-run/$cfg --output $S/scale/$tx.json --prior none \
  --layers 4 --heads 4 --width 192 --block 1024 --dropout 0.0 --batch 16 --iters 48728 --lr 1e-3 --min-lr 1e-4 --warmup 100 --weight-decay 0.1 \
  --eval-every 2500 --seed 1337 --save $S/scale/$tx.pt > $S/scale/$tx.log 2>&1)
echo "end tx exit=$? $(date -Is) $(grep '^result' $S/scale/$tx.log)"
kill $rss
$SLOTS done @imm
echo "slot ended $(date -Is)"
CUDA_VISIBLE_DEVICES= nice -n 19 python3 -B $S/scale/score_tx_noprior.py $S/scale/$tx.pt $S/imm-run/$cfg $S/scale/char_losses_$tx.npz > $S/scale/${tx}_cl.log 2>&1
echo "char losses tx exit=$? $(head -1 $S/scale/${tx}_cl.log) $(date -Is)"
