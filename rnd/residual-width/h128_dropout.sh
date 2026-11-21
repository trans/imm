#!/bin/sh
# imm dropout at hidden 128 (the user, via @air, 2026-11-10, asked for dropout
# at hidden 128). The h128 size run's settings plus --dropout 0.2, so the
# difference is the dropout alone. From the residual-dropout worktree.
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
cd $S/imm-dropout || exit 1
common="--prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows /home/trans/air/mobius/runs/imm/gutenberg5m_kn --segment-rule word --max-memory 16"
id=gutenberg5m_crossfit20_ps100_rs100_l2005_s64_lr1e-3_h128_e25_both_mobius16_segword_m16_do02
echo "start $id $(date -Is)"
python scripts/run_segment_memory_model.py --config configs/gutenberg5m_crossfit20_ps100_e1.json \
  --run-id $id --output runs/$id.csv --epochs 25 --hidden-size 128 --embedding-size 128 $common --dropout 0.2 > runs/$id.log 2>&1
echo "end $id exit=$? $(date -Is)"
