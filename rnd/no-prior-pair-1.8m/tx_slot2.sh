#!/bin/sh
# imm's slot $1: two more transformers for the first no-prior pair (slot
# 101: imm 3.3927, the 1.84M transformer at block 256 3.5788).
#  1. block 1,024, batch 16 (16,384 characters a step, as slot 101's).
#     The user, via @air 4f5cb25a, agreed to add the longer
#     transformer run. Dropout 0.2, as slot 101's.
#  2. block 256 at dropout 0.0 and 0.1. Slot 101's transformer ends with
#     training loss 1.340 above its held-out 1.275: it isn't overfitting.
# Everything else is slot 101's: the 100M split, 48,728 iterations (imm's
# 798M characters seen), seed 1337. Then per-character held-out losses on
# the CPU (windows of the model's block, stride half). Ends the slot.
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
cfg=$S/imm-run/configs/pg19split100m_noprior.json
arms="tx1.84m_np_100m_b1024:1024:16:0.2 tx1.84m_np_100m_do0:256:64:0.0 tx1.84m_np_100m_do1:256:64:0.1"
for arm in $arms; do
  tx=${arm%%:*}; r=${arm#*:}; block=${r%%:*}; r=${r#*:}; batch=${r%%:*}; dr=${r#*:}
  echo "start $tx block=$block batch=$batch dropout=$dr $(date -Is)"
  (cd /home/trans/Projects/air/experiments/prior_scaling && python3 -B prior_transformer.py --config $cfg --output $S/scale/$tx.json --prior none \
    --layers 4 --heads 4 --width 192 --block $block --dropout $dr --batch $batch --iters 48728 --lr 1e-3 --min-lr 1e-4 --warmup 100 --weight-decay 0.1 \
    --eval-every 2500 --seed 1337 --save $S/scale/$tx.pt > $S/scale/$tx.log 2>&1)
  echo "end $tx exit=$? $(date -Is) $(grep '^result' $S/scale/$tx.log)"
done
$SLOTS done @imm
echo "slot ended $(date -Is)"
for arm in $arms; do
  tx=${arm%%:*}
  CUDA_VISIBLE_DEVICES= nice -n 19 python3 -B $S/scale/score_tx_noprior.py $S/scale/$tx.pt $cfg $S/scale/char_losses_$tx.npz > $S/scale/${tx}_cl.log 2>&1
  echo "char losses $tx exit=$? $(head -1 $S/scale/${tx}_cl.log) $(date -Is)"
done
