#!/bin/sh
# imm's slot $1 (the user, 2026-11-15, asked for a phrase level, as another
# tier): two blocks + --word-tier 256 + --phrase-tier 256 (marks ",;:.!?"),
# 20 epochs, against slot 36's word tier alone (3.1274 at 20 epochs). Then
# its take-the-prior-away test, and each held-out character's loss for four
# models (two blocks; + word tier; + phrase tier; word tier at 41 epochs)
# split by place in word, phrase and sentence. Worktree imm-phrase at
# phrase-tier 573cea5. Free memory logged every 5 minutes. Ends the slot.
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
cd $S/imm-phrase || exit 1
echo "code $(git rev-parse --short HEAD) $(git status --porcelain --untracked-files=no | wc -l) changed files"
(while :; do echo "$(date -u +%H:%M) $(free -m | awk '/^Mem/ {print "available=" $7 "MB cache=" $6 "MB"} /^Swap/ {print "swap_used=" $3 "MB"}' | tr '\n' ' ')"; sleep 300; done) > $S/phrase_mem.log 2>&1 &
mem=$!
M=/home/trans/Projects/air/mobius/runs/imm
common="--prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows $M/pg19split20m_kn --segment-rule word --max-memory 64 --segments-per-update 64 --dropout 0.2 --eval-streams 64 --lr-schedule cosine --min-lr 1e-4 --carry-bound gru --memory-state gru --prior-rows $M/pg19split20m_gated_d16_pg200m --prior-residual-state-features reliability,depth_norm --hidden-size 256 --embedding-size 256 --prefetch-ahead 4 --attention-layers 2"
cfg=configs/pg19split20m_crossfit20_ps100_e1.json
id=pg19s20m_h256_par_l2_w256_p256_e20
echo "start $id $(date -Is)"
python -B scripts/run_segment_memory_model.py --config $cfg --run-id $id --output runs/$id.csv $common --epochs 20 --word-tier 256 --phrase-tier 256 > runs/$id.log 2>&1
rc=$?
echo "end $id exit=$rc $(date -Is)"
if [ $rc -eq 0 ] && [ -f runs/${id}_${id}.best.pt ]; then
  echo "start ablation $id $(date -Is)"
  python -B scripts/run_segment_memory_model.py --config $cfg --run-id abl_$id --output $S/abl_$id.csv $common --epochs 20 --word-tier 256 --phrase-tier 256 \
    --eval-only --resume-checkpoint runs/${id}_${id}.best.pt --prior-ablation-output runs/prior_ablation_${id}_best.json > runs/prior_ablation_${id}_best.log 2>&1
  echo "end ablation $id exit=$? $(date -Is)"
fi
# Each held-out character's loss: label, run id, epochs, tier flags.
mkdir -p $S/charloss
for spec in "base pg19s20m_h256_par_l2_e20 20" "word pg19s20m_h256_par_l2_w256_e20 20 --word-tier 256" \
  "phrase pg19s20m_h256_par_l2_w256_p256_e20 20 --word-tier 256 --phrase-tier 256" "word41 pg19s20m_h256_par_l2_w256_e41 41 --word-tier 256"; do
  set -- $spec
  label=$1 run=$2 epochs=$3
  shift 3
  if [ -f runs/${run}_${run}.best.pt ]; then
    python -B scripts/run_segment_memory_model.py --config $cfg --run-id cl_$run --output $S/charloss/cl_$run.csv $common --epochs $epochs "$@" \
      --eval-only --resume-checkpoint runs/${run}_${run}.best.pt --char-loss-output runs/char_losses_${run}_best.npz > $S/charloss/$label.log 2>&1
    echo "char losses $label exit=$? $(grep -o 'char_losses heldout_ppl=[0-9.]*' $S/charloss/$label.log) $(date -Is)"
  else
    echo "char losses $label: no checkpoint runs/${run}_${run}.best.pt"
  fi
done
kill $mem
$SLOTS done @imm
echo "slot ended $(date -Is)"
python -B scripts/tier_position_losses.py base=runs/char_losses_pg19s20m_h256_par_l2_e20_best.npz \
  word=runs/char_losses_pg19s20m_h256_par_l2_w256_e20_best.npz phrase=runs/char_losses_pg19s20m_h256_par_l2_w256_p256_e20_best.npz \
  word41=runs/char_losses_pg19s20m_h256_par_l2_w256_e41_best.npz --json runs/tier_positions_pg19s20m.json > $S/tier_positions.txt 2>&1
echo "positions exit=$? -> $S/tier_positions.txt"
