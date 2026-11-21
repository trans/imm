#!/bin/sh
# Longer reach: the gd16_pg200m run (best 3.3220) with 64 memory records and
# 64 word segments per update (about 360 characters each, against 90 and
# 180), everything else the same: mobius's depth-16 prior with 200M of books,
# pg19split_kn evidence, h256, dropout 0.2, 80 epochs cosine, seed 1337.
# Word starts gained only 0.029 bits/char there; this tests whether reach is
# where the residual adds on top of a strong prior. Branch pg19-split.
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
cd $S/imm-dropout || exit 1
common="--prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows /home/trans/air/mobius/runs/imm/pg19split_kn --segment-rule word --max-memory 64 --segments-per-update 64 --dropout 0.2 --eval-streams 64 --drop-evidence-cache --lr-schedule cosine --min-lr 1e-4 --hidden-size 256 --embedding-size 256 --epochs 80"
id=pg19split_crossfit20_ps100_rs100_l2005_s64_lr1e-3_h256_e80_both_mobius16_segword_m64_spu64_do02_cos_gd16_pg200m
echo "start $id $(date -Is)"
python scripts/run_segment_memory_model.py --config configs/pg19split_crossfit20_ps100_e1.json --run-id $id --output runs/$id.csv $common \
  --prior-rows /home/trans/air/mobius/runs/imm/pg19split_gated_d16_pg200m --prior-residual-state-features reliability,depth_norm > runs/$id.log 2>&1
echo "end $id exit=$? $(date -Is)"
