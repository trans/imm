#!/bin/sh
# imm's slot $1 (the user approved it, 2026-11-13): memory off on the 20M
# split, the base run's flags (1 x h256 memory on, 64 x 64, 20 epochs,
# 3.1517) with --mixing none: no records, no cross-attention, one GRU call
# over each stream's span. h256 (895,147 trained parameters: the same GRU
# without the memory), then h432 (1,964,347: the base's 1,949,357). Each
# followed by its take-the-prior-away test. Branch memory-off 3c8ca62.
# Ends the slot when done. Laptop limits: nice 19, ionice idle, 2 CPU threads.
renice -n 19 -p $$ >/dev/null
ionice -c 3 -p $$
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
export TMPDIR=$S
SLOTS="sh /home/trans/Projects/air/tools/slots"
sid=$1
while ! $SLOTS list 2>/dev/null | grep -q "^in use: $sid @imm,"; do
  if ! $SLOTS list 2>/dev/null | grep -qE "^(in use: |[0-9]+\. )$sid @imm,"; then
    echo "slot $sid is neither queued nor in use $(date -Is)"; exit 1
  fi
  sleep 20
done
echo "slot $sid granted $(date -Is)"
cd $S/imm-nomem || exit 1
echo "code $(git rev-parse --short HEAD) $(git status --porcelain --untracked-files=no | wc -l) changed files"
M=/home/trans/Projects/air/mobius/runs/imm
common="--prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows $M/pg19split20m_kn --segment-rule word --max-memory 64 --segments-per-update 64 --dropout 0.2 --eval-streams 64 --lr-schedule cosine --min-lr 1e-4 --epochs 20 --carry-bound norm --carry-scale 32 --prior-rows $M/pg19split20m_gated_d16_pg200m --prior-residual-state-features reliability,depth_norm --mixing none"
cfg=configs/pg19split20m_crossfit20_ps100_e1.json
run() {  # id, extra flags
  id=$1; shift
  echo "start $id $(date -Is)"
  python -B scripts/run_segment_memory_model.py --config $cfg --run-id $id --output runs/$id.csv $common "$@" > runs/$id.log 2>&1
  rc=$?
  echo "end $id exit=$rc $(date -Is)"
  if [ $rc -eq 0 ] && [ -f runs/${id}_${id}.best.pt ]; then
    echo "start ablation $id $(date -Is)"
    python -B scripts/run_segment_memory_model.py --config $cfg --run-id abl_$id --output $S/abl_$id.csv $common "$@" \
      --eval-only --resume-checkpoint runs/${id}_${id}.best.pt --prior-ablation-output runs/prior_ablation_${id}_best.json > runs/prior_ablation_${id}_best.log 2>&1
    echo "end ablation $id exit=$? $(date -Is)"
  fi
}
run pg19s20m_h256_nomem_e20 --hidden-size 256 --embedding-size 256
run pg19s20m_h432_nomem_e20 --hidden-size 432 --embedding-size 432
$SLOTS done @imm
echo "slot ended $(date -Is)"
