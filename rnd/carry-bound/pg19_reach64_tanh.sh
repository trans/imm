#!/bin/sh
# The reach-64 run (64 records, 64 segments per update; best 3.3155 at
# epoch 13, then a runaway at epoch 19) with --carry-bound tanh, and the
# depth evidence read from the compact layout (runs/pg19split_kn_compact3,
# the same bytes as pg19split_kn). Otherwise identical, seed 1337. Branch
# pg19-carry-compact 2fc6d00.
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
cd $S/imm-run || exit 1
common="--prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows runs/pg19split_kn_compact3 --segment-rule word --max-memory 64 --segments-per-update 64 --dropout 0.2 --eval-streams 64 --lr-schedule cosine --min-lr 1e-4 --hidden-size 256 --embedding-size 256 --epochs 80 --carry-bound tanh"
id=pg19split_h256_e80_both_segword_m64_spu64_do02_cos_gd16_pg200m_cbtanh_ev3
echo "start $id $(date -Is)"
python scripts/run_segment_memory_model.py --config configs/pg19split_crossfit20_ps100_e1.json --run-id $id --output runs/$id.csv $common \
  --prior-rows /home/trans/air/mobius/runs/imm/pg19split_gated_d16_pg200m --prior-residual-state-features reliability,depth_norm > runs/$id.log 2>&1
echo "end $id exit=$? $(date -Is)"
