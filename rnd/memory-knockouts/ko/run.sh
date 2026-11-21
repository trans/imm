#!/bin/sh
# Knockouts on slot 63's best checkpoint (the user approved them, 2026-11-17):
# full, word GRU off (word_tier.gate = 0), memory attention off (both
# gate_attn = 0, MLPs kept), neither. On the GPU inside imm's slot (the full load needs over
# 2 GB); nice 19, ionice idle, 2 CPU threads.
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
M=/home/trans/Projects/air/mobius/runs/imm
cd /home/trans/Projects/air/imm || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TMPDIR=$S
common="--prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows $M/pg19split20m_kn --segment-rule word --max-memory 64 --segments-per-update 64 --dropout 0.2 --eval-streams 64 --lr-schedule cosine --min-lr 1e-4 --epochs 41 --carry-bound gru --memory-state gru --prior-rows $M/pg19split20m_gated_d24_pg10g --prior-residual-state-features reliability,depth_norm --hidden-size 192 --embedding-size 192 --prefetch-ahead 4 --attention-layers 2 --word-tier 168 --tier-own-step --prior-strength-gate"
for arm in ${ARMS:-full noword nomem neither}; do
  t0=$(date +%s)
  nice -n 19 ionice -c 3 python -B $S/ko/maxrss.py python -B scripts/run_segment_memory_model.py \
    --config configs/pg19split20m_crossfit20_ps100_e1.json --run-id ko_$arm --output $S/ko/$arm.csv $common \
    --eval-only --resume-checkpoint $S/ko/$arm.pt --char-loss-output $S/ko/char_losses_$arm.npz > $S/ko/$arm.log 2>&1
  echo "$arm exit=$? $(grep -o 'char_losses heldout_ppl=[0-9.]*' $S/ko/$arm.log) $(grep -o 'maxrss_kb=[0-9]*' $S/ko/$arm.log) $(( $(date +%s) - t0 ))s"
done
echo "done $(date -u +%H:%M)"
