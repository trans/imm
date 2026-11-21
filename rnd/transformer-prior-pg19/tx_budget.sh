#!/bin/sh
# The user (2026-11-13) chose the cheap check first. @air's 477k transformer
# with the books prior in its input (runs/s477k_prior_in.json: 50,000 iters
# x 64 x 256 = 819M chars, 3.1282), rerun at imm's budget: 20 epochs x 20M =
# 400M chars = 24,414 iters. Same config otherwise (air's followup_queue.sh,
# S2, seed 1337). air's prior_transformer.py run as is; output only in
# imm's scratchpad; -B so no bytecode is written anywhere.
# Waits for imm's slot SLOT_ID to be granted, then runs and ends the slot.
renice -n 19 -p $$ >/dev/null
ionice -c 3 -p $$
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
export TMPDIR=$S
SLOTS="sh /home/trans/Projects/air/tools/slots"
id=$1
while ! $SLOTS list 2>/dev/null | grep -q "^in use: $id @imm,"; do
  if ! $SLOTS list 2>/dev/null | grep -qE "^(in use: |[0-9]+\. )$id @imm,"; then
    echo "slot $id is neither queued nor in use $(date -Is)"; exit 1
  fi
  sleep 20
done
echo "slot $id granted $(date -Is)"
E=/home/trans/Projects/air/experiments/prior_scaling
B=/home/trans/Projects/air/mobius/runs/imm/pg19split20m_gated_d16_pg200m
mkdir -p $S/tx
cd $E || exit 1
python -B prior_transformer.py --config $E/pg19split20m_cfg.json \
  --output $S/tx/s477k_prior_in_i24414.json --iters 24414 --eval-every 2500 --seed 1337 \
  --prior $B --prior-input --layers 4 --heads 4 --width 96 --dropout 0.0 > $S/tx/s477k_prior_in_i24414.log 2>&1
echo "end exit=$? $(date -Is)"
$SLOTS done @imm
echo "slot ended $(date -Is)"
