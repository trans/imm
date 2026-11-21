#!/bin/sh
# The take-the-prior-away test (the user, via @air f32f3b40) on the 4.75M
# PG-19 split's h256 runs, on the CPU at nice 19: books prior (best 3.3220)
# and imm's own prior (best 3.7059), each scored in full, residual alone,
# no prior, and two fresh-head probes. Branch prior-ablation 733f101.
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
renice -n 19 -p $$ >/dev/null
cd $S/imm-ablate || exit 1
export CUDA_VISIBLE_DEVICES= TMPDIR=$S
M=/home/trans/Projects/air/mobius/runs/imm
common="--prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows $M/pg19split_kn --segment-rule word --max-memory 16 --dropout 0.2 --eval-streams 64 --lr-schedule cosine --min-lr 1e-4 --hidden-size 256 --embedding-size 256 --epochs 80 --eval-only"
books=pg19split_crossfit20_ps100_rs100_l2005_s64_lr1e-3_h256_e80_both_mobius16_segword_m16_do02_cos_gd16_pg200m
own=pg19split_crossfit20_ps100_rs100_l2005_s64_lr1e-3_h256_e80_both_mobius16_segword_m16_do02_cos
echo "start books $(date -Is)"
nice -n 19 python scripts/run_segment_memory_model.py --config configs/pg19split_crossfit20_ps100_e1.json --run-id abl_books --output $S/abl_books.csv $common \
  --prior-rows $M/pg19split_gated_d16_pg200m --prior-residual-state-features reliability,depth_norm \
  --resume-checkpoint runs/${books}_${books}.best.pt --prior-ablation-output runs/prior_ablation_pg19split_h256_pg200m_best.json > runs/prior_ablation_pg19split_h256_pg200m_best.log 2>&1
echo "end books exit=$? $(date -Is)"
echo "start own $(date -Is)"
nice -n 19 python scripts/run_segment_memory_model.py --config configs/pg19split_crossfit20_ps100_e1.json --run-id abl_own --output $S/abl_own.csv $common \
  --resume-checkpoint runs/${own}_${own}.best.pt --prior-ablation-output runs/prior_ablation_pg19split_h256_own_best.json > runs/prior_ablation_pg19split_h256_own_best.log 2>&1
echo "end own exit=$? $(date -Is)"
