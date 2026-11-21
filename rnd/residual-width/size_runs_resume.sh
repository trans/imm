#!/bin/sh
# The size runs (the user's plan via @air, 2026-11-09), restarted on the user's
# word (2026-11-10): hidden 64 resumed from its epoch-27 checkpoint to 40, then
# hidden 128 for 25. Word segments at 16, mobius's depth 16, --prior-input both,
# lr 1e-3. From the word-segments-depth-evidence worktree. One at a time.
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
cd $S/imm-combined || exit 1
common="--prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows /home/trans/air/mobius/runs/imm/gutenberg5m_kn --segment-rule word --max-memory 16"
id=gutenberg5m_crossfit20_ps100_rs100_l2005_s64_lr1e-3_h64_e40_both_mobius16_segword_m16
echo "start $id resume $(date -Is)"
python scripts/run_segment_memory_model.py --config $S/gutenberg5m_crossfit20_ps100_e1_resume.json \
  --run-id $id --output runs/$id.csv --epochs 40 --hidden-size 64 --embedding-size 64 $common \
  --resume-checkpoint runs/${id}_$id.pt >> runs/$id.log 2>&1
echo "end $id exit=$? $(date -Is)"
id=gutenberg5m_crossfit20_ps100_rs100_l2005_s64_lr1e-3_h128_e25_both_mobius16_segword_m16
echo "start $id $(date -Is)"
python scripts/run_segment_memory_model.py --config configs/gutenberg5m_crossfit20_ps100_e1.json \
  --run-id $id --output runs/$id.csv --epochs 25 --hidden-size 128 --embedding-size 128 $common > runs/$id.log 2>&1
echo "end $id exit=$? $(date -Is)"
