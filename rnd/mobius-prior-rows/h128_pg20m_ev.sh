#!/bin/sh
# h128 dropout 0.2, 25 epochs, seed 1337, on mobius's pg20m prior rows, now
# with depth evidence that also counts the 59 PG-19 books
# (../mobius/runs/imm/gutenberg5m_kn_pg20m, mobius d7ac253). Paired with the
# 3.0315 run: the same command, only --prior-evidence-rows changed.
# The user chose this run (2026-11-10). From the imm-dropout worktree (lr-schedule
# 3955731, constant lr, the default).
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
cd $S/imm-dropout || exit 1
common="--prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows /home/trans/air/mobius/runs/imm/gutenberg5m_kn_pg20m --segment-rule word --max-memory 16 --dropout 0.2 --eval-streams 64 --drop-evidence-cache"
id=gutenberg5m_crossfit20_ps100_rs100_l2005_s64_lr1e-3_h128_e25_both_mobius16_segword_m16_do02_pg20m_ev
echo "start $id $(date -Is)"
python scripts/run_segment_memory_model.py --config configs/gutenberg5m_crossfit20_ps100_e1.json \
  --run-id $id --output runs/$id.csv --epochs 25 --hidden-size 128 --embedding-size 128 $common \
  --prior-rows /home/trans/air/mobius/runs/imm/gutenberg5m_mix_units_pg20m \
  --prior-residual-state-features reliability,depth_norm > runs/$id.log 2>&1
echo "end $id exit=$? $(date -Is)"
