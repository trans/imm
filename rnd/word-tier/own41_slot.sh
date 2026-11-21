#!/bin/sh
# imm's slot $1 (the user, 2026-11-15, asked for the 820M run): two
# blocks + --word-tier 256 + --tier-own-step (slot 50: 3.1155 at 20 epochs)
# for 41 epochs = 820M characters, the budget of @air's s1.84m_prior_in
# (3.0987) and of slot 37's word tier without own step (3.1084); cosine over
# the 41 epochs. Then its take-the-prior-away test and each held-out
# character's loss. Worktree imm-own at tier-own-step 55b8128. Free memory
# logged every 5 minutes to $S/own41_mem.log. Ends the slot.
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
cd $S/imm-own || exit 1
echo "code $(git rev-parse --short HEAD) $(git status --porcelain --untracked-files=no | wc -l) changed files"
(while :; do echo "$(date -u +%H:%M) $(free -m | awk '/^Mem/ {print "available=" $7 "MB cache=" $6 "MB"} /^Swap/ {print "swap_used=" $3 "MB"}' | tr '\n' ' ')"; sleep 300; done) > $S/own41_mem.log 2>&1 &
mem=$!
M=/home/trans/Projects/air/mobius/runs/imm
common="--prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows $M/pg19split20m_kn --segment-rule word --max-memory 64 --segments-per-update 64 --dropout 0.2 --eval-streams 64 --lr-schedule cosine --min-lr 1e-4 --epochs 41 --carry-bound gru --memory-state gru --prior-rows $M/pg19split20m_gated_d16_pg200m --prior-residual-state-features reliability,depth_norm --hidden-size 256 --embedding-size 256 --prefetch-ahead 4 --attention-layers 2 --word-tier 256 --tier-own-step"
cfg=configs/pg19split20m_crossfit20_ps100_e1.json
id=pg19s20m_h256_par_l2_w256_own_e41
echo "start $id $(date -Is)"
python -B scripts/run_segment_memory_model.py --config $cfg --run-id $id --output runs/$id.csv $common > runs/$id.log 2>&1
rc=$?
echo "end $id exit=$rc $(date -Is)"
if [ $rc -eq 0 ] && [ -f runs/${id}_${id}.best.pt ]; then
  python -B scripts/run_segment_memory_model.py --config $cfg --run-id abl_$id --output $S/abl_$id.csv $common \
    --eval-only --resume-checkpoint runs/${id}_${id}.best.pt --prior-ablation-output runs/prior_ablation_${id}_best.json > runs/prior_ablation_${id}_best.log 2>&1
  echo "ablation $id exit=$? $(date -Is)"
  mkdir -p $S/charloss
  python -B scripts/run_segment_memory_model.py --config $cfg --run-id cl_$id --output $S/charloss/cl_$id.csv $common \
    --eval-only --resume-checkpoint runs/${id}_${id}.best.pt --char-loss-output runs/char_losses_${id}_best.npz > $S/charloss/$id.log 2>&1
  echo "char losses $id exit=$? $(grep -o 'char_losses heldout_ppl=[0-9.]*' $S/charloss/$id.log) $(date -Is)"
fi
kill $mem
$SLOTS done @imm
echo "slot ended $(date -Is)"
python -B scripts/tier_position_losses.py word41=runs/char_losses_pg19s20m_h256_par_l2_w256_e41_best.npz \
  word41_own=runs/char_losses_${id}_best.npz word20_own=runs/char_losses_pg19s20m_h256_par_l2_w256_own_e20_best.npz \
  --json runs/tier_positions_own41_pg19s20m.json > $S/own41_positions.txt 2>&1
echo "positions exit=$? -> $S/own41_positions.txt"
