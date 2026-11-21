#!/bin/sh
# h256 dropout 0.2, 80 epochs, cosine learning-rate decay 1e-3 -> 1e-4 (no
# warm-up), seed 1337. First on mobius's pg20m prior (h128: 3.0315), then on
# imm's own prior (h256 40 epochs constant lr: 3.0217 / 3.0395 over two seeds).
# The user asked for 80 epochs (2026-11-10). From the imm-dropout worktree,
# branch lr-schedule 3955731.
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
cd $S/imm-dropout || exit 1
common="--prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows /home/trans/air/mobius/runs/imm/gutenberg5m_kn --segment-rule word --max-memory 16 --dropout 0.2 --eval-streams 64 --drop-evidence-cache --lr-schedule cosine --min-lr 1e-4 --hidden-size 256 --embedding-size 256 --epochs 80"
base=gutenberg5m_crossfit20_ps100_rs100_l2005_s64_lr1e-3_h256_e80_both_mobius16_segword_m16_do02_cos
id=${base}_pg20m
echo "start $id $(date -Is)"
python scripts/run_segment_memory_model.py --config configs/gutenberg5m_crossfit20_ps100_e1.json --run-id $id --output runs/$id.csv $common \
  --prior-rows /home/trans/air/mobius/runs/imm/gutenberg5m_mix_units_pg20m --prior-residual-state-features reliability,depth_norm > runs/$id.log 2>&1
echo "end $id exit=$? $(date -Is)"
id=$base
echo "start $id $(date -Is)"
python scripts/run_segment_memory_model.py --config configs/gutenberg5m_crossfit20_ps100_e1.json --run-id $id --output runs/$id.csv $common > runs/$id.log 2>&1
echo "end $id exit=$? $(date -Is)"
