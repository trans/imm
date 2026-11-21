#!/bin/sh
# h128 dropout 0.2, 25 epochs, on mobius's combined prior (gutenberg5m_mix_units,
# mobius 8737818: 1.8156 bits/char on training rows against imm's d8 1.8396),
# with mobius's KN depth evidence. The user approved the run (2026-11-10).
# Compared with the same run on imm's own prior: 3.1079 at epoch 25.
# State features by name: reliability,depth_norm are imm's columns 0,3.
# From the imm-dropout worktree (prior-rows-loader, f210ae7).
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
cd $S/imm-dropout || exit 1
common="--prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows /home/trans/air/mobius/runs/imm/gutenberg5m_kn --segment-rule word --max-memory 16 --dropout 0.2 --eval-streams 64 --drop-evidence-cache"
id=gutenberg5m_crossfit20_ps100_rs100_l2005_s64_lr1e-3_h128_e25_both_mobius16_segword_m16_do02_mixunits
echo "start $id $(date -Is)"
python scripts/run_segment_memory_model.py --config configs/gutenberg5m_crossfit20_ps100_e1.json \
  --run-id $id --output runs/$id.csv --epochs 25 --hidden-size 128 --embedding-size 128 $common \
  --prior-rows /home/trans/air/mobius/runs/imm/gutenberg5m_mix_units \
  --prior-residual-state-features reliability,depth_norm > runs/$id.log 2>&1
echo "end $id exit=$? $(date -Is)"
