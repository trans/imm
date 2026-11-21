#!/bin/sh
# imm's slot $1, the 10.8M pair's first part (approved by the user,
# 2026-11-19). 1. $S/s10/probe.py: imm at ~10.8M with 1,024 character records,
# speed and GPU memory, to size its run. 2. Air's 10.8M transformer
# (6 layers x 384, 6 heads; air's s10.8m_none) with no prior at block
# 1,024, batch 16, at dropout 0.0 and 0.1: the 100M split, 48,728
# iterations (imm's 798M characters seen), lr 1e-3, seed 1337, as the
# 1.84M pairs. Ends the slot, then per-character held-out losses on the
# CPU. Laptop limits: nice 19, ionice idle, 2 CPU threads.
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
echo "probe start $(date -Is)"
(cd $S/imm-char && echo "code $(git rev-parse --short HEAD)" && python3 -B $S/s10/probe.py) > $S/s10/probe_$sid.log 2>&1
echo "probe exit=$? $(date -Is)"
cfg=$S/imm-run/configs/pg19split100m_noprior.json
arms="tx10.8m_np_100m_b1024_do0:0.0 tx10.8m_np_100m_b1024_do1:0.1"
for arm in $arms; do
  tx=${arm%%:*}; dr=${arm#*:}
  echo "start $tx dropout=$dr $(date -Is)"
  (cd /home/trans/Projects/air/experiments/prior_scaling && python3 -B prior_transformer.py --config $cfg --output $S/s10/$tx.json --prior none \
    --layers 6 --heads 6 --width 384 --block 1024 --dropout $dr --batch 16 --iters 48728 --lr 1e-3 --min-lr 1e-4 --warmup 100 --weight-decay 0.1 \
    --eval-every 2500 --seed 1337 --save $S/s10/$tx.pt > $S/s10/$tx.log 2>&1)
  echo "end $tx exit=$? $(date -Is) $(grep '^result' $S/s10/$tx.log)"
done
$SLOTS done @imm
echo "slot ended $(date -Is)"
for arm in $arms; do
  tx=${arm%%:*}
  CUDA_VISIBLE_DEVICES= nice -n 19 python3 -B $S/scale/score_tx_noprior.py $S/s10/$tx.pt $cfg $S/s10/char_losses_$tx.npz > $S/s10/${tx}_cl.log 2>&1
  echo "char losses $tx exit=$? $(head -1 $S/s10/${tx}_cl.log) $(date -Is)"
done
