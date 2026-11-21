#!/bin/sh
# imm h256 on the 20M PG-19 split: mobius's gated depth-16 rows with 200M
# of books counted (pg19split20m_gated_d16_pg200m, held-out 1.8205 alone)
# and compact KN depth evidence (pg19split20m_kn). Reach-64 (64 records, 64
# segments per update) with --carry-bound norm --carry-scale 32, 20
# epochs (about as many characters as 80 on 4.75M), cosine 1e-3 -> 1e-4,
# dropout 0.2, seed 1337. No prior dropout: the baseline for it. Branch
# pg19-20m ba91341.
renice -n 19 -p $$ >/dev/null
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
cd $S/imm-run || exit 1
M=/home/trans/Projects/air/mobius/runs/imm
common="--prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows $M/pg19split20m_kn --segment-rule word --max-memory 64 --segments-per-update 64 --dropout 0.2 --eval-streams 64 --lr-schedule cosine --min-lr 1e-4 --hidden-size 256 --embedding-size 256 --epochs 20 --carry-bound norm --carry-scale 32"
id=pg19s20m_h256_m64_cbn32_e20
echo "start $id $(date -Is)"
python scripts/run_segment_memory_model.py --config configs/pg19split20m_crossfit20_ps100_e1.json --run-id $id --output runs/$id.csv $common \
  --prior-rows $M/pg19split20m_gated_d16_pg200m --prior-residual-state-features reliability,depth_norm > runs/$id.log 2>&1
echo "end $id exit=$? $(date -Is)"
