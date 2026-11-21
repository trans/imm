#!/bin/sh
# After the 20M h256 run (pg19_20m_h256.sh) exits: its take-the-prior-away
# test on the CPU, then the same run with --prior-dropout 0.5 (@air's rate)
# on the GPU, then its test. Branch prior-dropout 57f1033. Waits for the GPU
# to have no compute process before the training run.
renice -n 19 -p $$ >/dev/null
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
cd $S/imm-pdrop || exit 1
M=/home/trans/Projects/air/mobius/runs/imm
common="--prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows $M/pg19split20m_kn --segment-rule word --max-memory 64 --segments-per-update 64 --dropout 0.2 --eval-streams 64 --lr-schedule cosine --min-lr 1e-4 --hidden-size 256 --embedding-size 256 --epochs 20 --carry-bound norm --carry-scale 32 --prior-rows $M/pg19split20m_gated_d16_pg200m --prior-residual-state-features reliability,depth_norm"
cfg=configs/pg19split20m_crossfit20_ps100_e1.json
base=pg19s20m_h256_m64_cbn32_e20
pd=pg19s20m_h256_m64_cbn32_pd05_e20
ablate() {
  echo "start ablation $1 $(date -Is)"
  CUDA_VISIBLE_DEVICES= TMPDIR=$S python scripts/run_segment_memory_model.py --config $cfg --run-id abl_$2 --output $S/abl_$2.csv $common \
    --eval-only --resume-checkpoint runs/${1}_${1}.best.pt --prior-ablation-output runs/prior_ablation_${1}_best.json > runs/prior_ablation_${1}_best.log 2>&1
  echo "end ablation $1 exit=$? $(date -Is)"
}
while pgrep -f "run-id $base " >/dev/null; do sleep 30; done
echo "base run gone $(date -Is)"
[ -f runs/${base}_${base}.best.pt ] && ablate $base base
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 30; done
echo "start $pd $(date -Is)"
python scripts/run_segment_memory_model.py --config $cfg --run-id $pd --output runs/$pd.csv $common --prior-dropout 0.5 > runs/$pd.log 2>&1
echo "end $pd exit=$? $(date -Is)"
[ -f runs/${pd}_${pd}.best.pt ] && ablate $pd pd
