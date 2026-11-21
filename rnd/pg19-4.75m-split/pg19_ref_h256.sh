#!/bin/sh
# Reference run on the PG-19 split: imm's own gated prior (theta fitted on
# the split), mobius's KN depth evidence to 16 without the books
# (../mobius/runs/imm/pg19split_kn, mobius 1417d28), h256, dropout 0.2,
# 80 epochs, cosine 1e-3 -> 1e-4, seed 1337. The same command as the
# Gutenberg 5M 3.0076 run apart from config and evidence. Branch pg19-split
# d65044d.
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
cd $S/imm-dropout || exit 1
common="--prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows /home/trans/air/mobius/runs/imm/pg19split_kn --segment-rule word --max-memory 16 --dropout 0.2 --eval-streams 64 --drop-evidence-cache --lr-schedule cosine --min-lr 1e-4 --hidden-size 256 --embedding-size 256 --epochs 80"
id=pg19split_crossfit20_ps100_rs100_l2005_s64_lr1e-3_h256_e80_both_mobius16_segword_m16_do02_cos
echo "start $id $(date -Is)"
python scripts/run_segment_memory_model.py --config configs/pg19split_crossfit20_ps100_e1.json --run-id $id --output runs/$id.csv $common > runs/$id.log 2>&1
echo "end $id exit=$? $(date -Is)"
