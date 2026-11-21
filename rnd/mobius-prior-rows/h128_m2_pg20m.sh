#!/bin/sh
# h128 dropout 0.2, 25 epochs, on mobius's m2 and pg20m prefixes (mobius
# 37962d7): the same single model, pg20m also counting 19.7M characters of
# PG-19. Paired: same command and seed (1337) as the 3.0927 (mix_units) and
# 3.1079 (imm's prior) runs. The user chose this run (2026-11-10).
# From the imm-dropout worktree, now lr-schedule 3955731 (constant lr, the
# default, reproduces earlier runs exactly).
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
cd $S/imm-dropout || exit 1
common="--prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows /home/trans/air/mobius/runs/imm/gutenberg5m_kn --segment-rule word --max-memory 16 --dropout 0.2 --eval-streams 64 --drop-evidence-cache"
for p in m2 pg20m; do
  id=gutenberg5m_crossfit20_ps100_rs100_l2005_s64_lr1e-3_h128_e25_both_mobius16_segword_m16_do02_$p
  echo "start $id $(date -Is)"
  python scripts/run_segment_memory_model.py --config configs/gutenberg5m_crossfit20_ps100_e1.json \
    --run-id $id --output runs/$id.csv --epochs 25 --hidden-size 128 --embedding-size 128 $common \
    --prior-rows /home/trans/air/mobius/runs/imm/gutenberg5m_mix_units_$p \
    --prior-residual-state-features reliability,depth_norm > runs/$id.log 2>&1
  echo "end $id exit=$? $(date -Is)"
done
