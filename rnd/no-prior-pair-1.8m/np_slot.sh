#!/bin/sh
# imm's slot $1: the first pair of the no-prior scale test (the user,
# 2026-11-18, paraphrased: one imm run at ~1.85M and one transformer of
# the same size, compared before anything bigger). Both without any prior,
# on mobius's 100M split (imm-split-100m: the 20M pieces plus 205 whole
# books, the same 50 held-out pieces), the same characters seen:
#   imm: the m128 model (h192, word tier 168 own step, 2 attention blocks,
#        128 records), --documents, 8 epochs;
#   tx:  air's prior_transformer.py --prior none at 1.84M (4 x 192, 4 heads,
#        block 256, batch 64), iterations set to imm's characters seen.
# From worktree imm-run (topic-tiers 91c3c55). Then per-character held-out
# losses for both. /proc sampling every minute to $S/scale/rss_$1.log, with
# a guard that stops the run below 800 MB available. Ends the slot.
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
id=pg19s100m_h192_par_l2_w168_own_np_doc_mm128_e8
common="--count-prior none --prior-residual-gate none --prior-input none --prior-residual-state-features= --prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --eval-streams 64 --lr 1e-3 --segment-rule word --max-memory 128 --segments-per-update 64 --dropout 0.2 --lr-schedule cosine --min-lr 1e-4 --carry-bound gru --memory-state gru --hidden-size 192 --embedding-size 192 --prefetch-ahead 4 --attention-layers 2 --word-tier 168 --tier-own-step --documents data/pg19/imm-split-100m/manifest.json --epochs 8"
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
# The transformer sees as many characters as imm did: 8 epochs of imm's
# tokens per epoch (its profile line), in batches of 64 x 256.
tok=$(grep -o 'epoch=1 profile updates=[0-9]* tokens=[0-9]*' runs/$id.log | grep -o 'tokens=[0-9]*' | cut -d= -f2)
if [ -n "$tok" ]; then iters=$(( (tok * 8 + 8192) / 16384 )); else iters=48700; fi
echo "imm tokens per epoch=${tok:-unknown}; transformer iters=$iters ($(( iters * 16384 )) characters)"
tx=tx1.84m_np_100m
echo "start tx $tx $(date -Is)"
(cd /home/trans/Projects/air/experiments/prior_scaling && python3 -B prior_transformer.py --config $S/imm-run/$cfg --output $S/scale/$tx.json --prior none \
  --layers 4 --heads 4 --width 192 --block 256 --dropout 0.2 --batch 64 --iters $iters --lr 1e-3 --min-lr 1e-4 --warmup 100 --weight-decay 0.1 \
  --eval-every 2500 --seed 1337 --save $S/scale/$tx.pt > $S/scale/$tx.log 2>&1)
echo "end tx exit=$? $(date -Is) $(grep '^result' $S/scale/$tx.log)"
kill $rss
$SLOTS done @imm
echo "slot ended $(date -Is)"
# CPU only from here (light): the transformer's per-character losses.
CUDA_VISIBLE_DEVICES= nice -n 19 python3 -B $S/scale/score_tx_noprior.py $S/scale/$tx.pt $S/imm-run/$cfg $S/scale/char_losses_$tx.npz > $S/scale/${tx}_cl.log 2>&1
echo "char losses tx exit=$? $(head -1 $S/scale/${tx}_cl.log) $(date -Is)"
