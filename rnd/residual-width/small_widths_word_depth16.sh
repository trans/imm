#!/bin/sh
# The user's plan via @air (2026-11-09), with @air's correction: can a smaller
# residual plus the prior match the 474k transformer? Hidden 64 and 128,
# imm's best setup: word segments at 16 records, mobius's depth-16 evidence,
# --prior-input both, lr 1e-3; otherwise the segword_m16 run's settings
# (3.1316). From the word-segments-depth-evidence worktree. One at a time.
cd /home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad/imm-combined || exit 1
for spec in "64 40" "128 25"; do
  set -- $spec
  id=gutenberg5m_crossfit20_ps100_rs100_l2005_s64_lr1e-3_h$1_e$2_both_mobius16_segword_m16
  echo "start $id $(date -Is)"
  python scripts/run_segment_memory_model.py --config configs/gutenberg5m_crossfit20_ps100_e1.json \
    --run-id $id --output runs/$id.csv --prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 \
    --lr 1e-3 --epochs $2 --hidden-size $1 --embedding-size $1 --prior-input both \
    --prior-evidence-rows /home/trans/air/mobius/runs/imm/gutenberg5m_kn --segment-rule word --max-memory 16 > runs/$id.log 2>&1
  echo "end $id exit=$? $(date -Is)"
done
