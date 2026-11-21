#!/bin/sh
# The two runs planned after h128 dropout 0.2 (3.1079, still improving),
# run after the speedups (batched evaluation, background row reads), as the
# user asked (2026-11-10). From the batched-eval worktree. One at a time.
#   1. h128 dropout 0.2 resumed from epoch 25 to 50 (constant lr, as before);
#      its CSV starts as a copy of the 25-epoch one, so it holds all 50.
#   2. h256 dropout 0.2, 40 epochs: the same-size test against @air's 3.0542.
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
cd $S/imm-dropout || exit 1
common="--prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows /home/trans/air/mobius/runs/imm/gutenberg5m_kn --segment-rule word --max-memory 16 --dropout 0.2 --eval-streams 64 --drop-evidence-cache"
old=gutenberg5m_crossfit20_ps100_rs100_l2005_s64_lr1e-3_h128_e25_both_mobius16_segword_m16_do02
id=gutenberg5m_crossfit20_ps100_rs100_l2005_s64_lr1e-3_h128_e50_both_mobius16_segword_m16_do02
[ -e runs/${id}_$id.csv ] || cp runs/${old}_$old.csv runs/${id}_$id.csv
echo "start $id $(date -Is)"
python scripts/run_segment_memory_model.py --config $S/gutenberg5m_crossfit20_ps100_e1_resume.json \
  --run-id $id --output runs/$id.csv --epochs 50 --hidden-size 128 --embedding-size 128 $common \
  --resume-checkpoint runs/${old}_$old.pt > runs/$id.log 2>&1
echo "end $id exit=$? $(date -Is)"
id=gutenberg5m_crossfit20_ps100_rs100_l2005_s64_lr1e-3_h256_e40_both_mobius16_segword_m16_do02
echo "start $id $(date -Is)"
python scripts/run_segment_memory_model.py --config configs/gutenberg5m_crossfit20_ps100_e1.json \
  --run-id $id --output runs/$id.csv --epochs 40 --hidden-size 256 --embedding-size 256 $common > runs/$id.log 2>&1
echo "end $id exit=$? $(date -Is)"
