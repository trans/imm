#!/bin/sh
# imm's slot $1, runs 1-2 of the tier plan (mail to @air 2026-11-15;
# the user approved them): --tier-own-step, a segment's last
# character sees the tier state its own segment makes. Run 1: two blocks +
# --word-tier 256, against the word tier without it (3.1274 at 20 epochs).
# Run 2: + --phrase-tier 256, against the phrase tier without it (3.1326).
# 20 epochs each, then the take-the-prior-away test and each held-out
# character's loss for both. Worktree imm-own at tier-own-step 55b8128.
# Free memory logged every 5 minutes. Ends the slot.
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
(while :; do echo "$(date -u +%H:%M) $(free -m | awk '/^Mem/ {print "available=" $7 "MB cache=" $6 "MB"} /^Swap/ {print "swap_used=" $3 "MB"}' | tr '\n' ' ')"; sleep 300; done) > $S/own_mem.log 2>&1 &
mem=$!
M=/home/trans/Projects/air/mobius/runs/imm
common="--prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows $M/pg19split20m_kn --segment-rule word --max-memory 64 --segments-per-update 64 --dropout 0.2 --eval-streams 64 --lr-schedule cosine --min-lr 1e-4 --carry-bound gru --memory-state gru --prior-rows $M/pg19split20m_gated_d16_pg200m --prior-residual-state-features reliability,depth_norm --hidden-size 256 --embedding-size 256 --prefetch-ahead 4 --attention-layers 2"
cfg=configs/pg19split20m_crossfit20_ps100_e1.json
mkdir -p $S/charloss
for spec in "pg19s20m_h256_par_l2_w256_own_e20 --word-tier 256 --tier-own-step" \
  "pg19s20m_h256_par_l2_w256_p256_own_e20 --word-tier 256 --phrase-tier 256 --tier-own-step"; do
  set -- $spec
  id=$1
  shift
  echo "start $id $(date -Is)"
  python -B scripts/run_segment_memory_model.py --config $cfg --run-id $id --output runs/$id.csv $common --epochs 20 "$@" > runs/$id.log 2>&1
  echo "end $id exit=$? $(date -Is)"
  if [ -f runs/${id}_${id}.best.pt ]; then
    python -B scripts/run_segment_memory_model.py --config $cfg --run-id abl_$id --output $S/abl_$id.csv $common --epochs 20 "$@" \
      --eval-only --resume-checkpoint runs/${id}_${id}.best.pt --prior-ablation-output runs/prior_ablation_${id}_best.json > runs/prior_ablation_${id}_best.log 2>&1
    echo "ablation $id exit=$? $(date -Is)"
    python -B scripts/run_segment_memory_model.py --config $cfg --run-id cl_$id --output $S/charloss/cl_$id.csv $common --epochs 20 "$@" \
      --eval-only --resume-checkpoint runs/${id}_${id}.best.pt --char-loss-output runs/char_losses_${id}_best.npz > $S/charloss/$id.log 2>&1
    echo "char losses $id exit=$? $(grep -o 'char_losses heldout_ppl=[0-9.]*' $S/charloss/$id.log) $(date -Is)"
  else
    echo "no checkpoint runs/${id}_${id}.best.pt"
  fi
done
kill $mem
$SLOTS done @imm
echo "slot ended $(date -Is)"
R=runs
python -B scripts/tier_position_losses.py word=$R/char_losses_pg19s20m_h256_par_l2_w256_e20_best.npz \
  word_own=$R/char_losses_pg19s20m_h256_par_l2_w256_own_e20_best.npz \
  --json runs/tier_positions_own_word_pg19s20m.json > $S/own_positions_word.txt 2>&1
python -B scripts/tier_position_losses.py phrase=$R/char_losses_pg19s20m_h256_par_l2_w256_p256_e20_best.npz \
  phrase_own=$R/char_losses_pg19s20m_h256_par_l2_w256_p256_own_e20_best.npz word_own=$R/char_losses_pg19s20m_h256_par_l2_w256_own_e20_best.npz \
  --json runs/tier_positions_own_phrase_pg19s20m.json > $S/own_positions_phrase.txt 2>&1
echo "positions exit=$? -> $S/own_positions_*.txt"
