#!/bin/sh
# imm's slot $1 (the user approved the plan, 2026-11-14,
# step 4): the word-level GRU tier (--word-tier 256, branch word-tier
# df1e3c9, on reader b38374b: compact rows and readahead on by default) on
# the two-block parallel model (pg19s20m_h256_par_l2_e20, 3.1360; three
# blocks 3.1351), then its take-the-prior-away test. Worktree imm-word.
# Its reader wait in epochs 4-20 also measures readahead over a long run.
# Ends the slot. Laptop limits: nice 19, ionice idle, 2 CPU threads.
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
cd $S/imm-word || exit 1
echo "code $(git rev-parse --short HEAD) $(git status --porcelain --untracked-files=no | wc -l) changed files"
M=/home/trans/Projects/air/mobius/runs/imm
common="--prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows $M/pg19split20m_kn --segment-rule word --max-memory 64 --segments-per-update 64 --dropout 0.2 --eval-streams 64 --lr-schedule cosine --min-lr 1e-4 --epochs 20 --carry-bound gru --memory-state gru --prior-rows $M/pg19split20m_gated_d16_pg200m --prior-residual-state-features reliability,depth_norm --hidden-size 256 --embedding-size 256 --prefetch-ahead 4 --attention-layers 2 --word-tier 256"
cfg=configs/pg19split20m_crossfit20_ps100_e1.json
id=pg19s20m_h256_par_l2_w256_e20
echo "start $id $(date -Is)"
python -B scripts/run_segment_memory_model.py --config $cfg --run-id $id --output runs/$id.csv $common > runs/$id.log 2>&1
rc=$?
echo "end $id exit=$rc $(date -Is)"
if [ $rc -eq 0 ] && [ -f runs/${id}_${id}.best.pt ]; then
  echo "start ablation $id $(date -Is)"
  python -B scripts/run_segment_memory_model.py --config $cfg --run-id abl_$id --output $S/abl_$id.csv $common \
    --eval-only --resume-checkpoint runs/${id}_${id}.best.pt --prior-ablation-output runs/prior_ablation_${id}_best.json > runs/prior_ablation_${id}_best.log 2>&1
  echo "end ablation $id exit=$? $(date -Is)"
fi
$SLOTS done @imm
echo "slot ended $(date -Is)"
