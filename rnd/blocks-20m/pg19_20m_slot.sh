#!/bin/sh
# One of imm's slots (start only when @air says the slot is open; @air
# queued them as two slots, 537f3a00): `pd` is the 20M h256 run with
# --prior-dropout 0.5, `deep` the deeper run, two attention blocks at
# width 192 (the user asked to try depth first). Each is followed at once by its
# take-the-prior-away test, on the GPU, inside the slot (only if training
# exited 0). `pair` (the user, 2026-11-12: two blocks at h256, added to the
# queue, then three blocks at 128 too) runs h256 x2 then h128 x3.
# Laptop limits: nice 19, ionice idle, 2 CPU threads. Branch prior-dropout
# 57f1033.
renice -n 19 -p $$ >/dev/null
ionice -c 3 -p $$
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
export TMPDIR=$S
cd $S/imm-pdrop || exit 1
M=/home/trans/Projects/air/mobius/runs/imm
common="--prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows $M/pg19split20m_kn --segment-rule word --max-memory 64 --segments-per-update 64 --dropout 0.2 --eval-streams 64 --lr-schedule cosine --min-lr 1e-4 --epochs 20 --carry-bound norm --carry-scale 32 --prior-rows $M/pg19split20m_gated_d16_pg200m --prior-residual-state-features reliability,depth_norm"
cfg=configs/pg19split20m_crossfit20_ps100_e1.json
run() {  # id, extra flags
  id=$1; shift
  echo "start $id $(date -Is)"
  python scripts/run_segment_memory_model.py --config $cfg --run-id $id --output runs/$id.csv $common "$@" > runs/$id.log 2>&1
  rc=$?
  echo "end $id exit=$rc $(date -Is)"
  if [ $rc -eq 0 ] && [ -f runs/${id}_${id}.best.pt ]; then
    echo "start ablation $id $(date -Is)"
    python scripts/run_segment_memory_model.py --config $cfg --run-id abl_$id --output $S/abl_$id.csv $common "$@" \
      --eval-only --resume-checkpoint runs/${id}_${id}.best.pt --prior-ablation-output runs/prior_ablation_${id}_best.json > runs/prior_ablation_${id}_best.log 2>&1
    echo "end ablation $id exit=$? $(date -Is)"
  fi
}
case "$1" in
  pd) run pg19s20m_h256_m64_cbn32_pd05_e20 --hidden-size 256 --embedding-size 256 --prior-dropout 0.5 ;;
  deep) run pg19s20m_h192_l2_m64_cbn32_e20 --hidden-size 192 --embedding-size 192 --attention-layers 2 ;;
  wide2) run pg19s20m_h256_l2_m64_cbn32_e20 --hidden-size 256 --embedding-size 256 --attention-layers 2 ;;
  narrow3) run pg19s20m_h128_l3_m64_cbn32_e20 --hidden-size 128 --embedding-size 128 --attention-layers 3 ;;
  pair) run pg19s20m_h256_l2_m64_cbn32_e20 --hidden-size 256 --embedding-size 256 --attention-layers 2
        run pg19s20m_h128_l3_m64_cbn32_e20 --hidden-size 128 --embedding-size 128 --attention-layers 3 ;;
  *) echo "usage: $0 pd|deep|wide2|narrow3|pair" >&2; exit 2 ;;
esac
echo "slot work done $(date -Is)"
