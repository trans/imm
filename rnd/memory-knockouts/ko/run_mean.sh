#!/bin/sh
# Mean ablations on slot 63's best checkpoint (follow-up to the zero knockouts,
# which put the word-GRU-off model below uniform at sentence ends): collect the
# averages, then the word tier's add, the attention outputs, both, replaced by
# them. On the GPU inside imm's slot; nice 19, ionice idle, 2 CPU threads.
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
M=/home/trans/Projects/air/mobius/runs/imm
cd /home/trans/Projects/air/imm || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TMPDIR=$S KO_MEANS=$S/ko/means.pt
common="--prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows $M/pg19split20m_kn --segment-rule word --max-memory 64 --segments-per-update 64 --dropout 0.2 --eval-streams 64 --lr-schedule cosine --min-lr 1e-4 --epochs 41 --carry-bound gru --memory-state gru --prior-rows $M/pg19split20m_gated_d24_pg10g --prior-residual-state-features reliability,depth_norm --hidden-size 192 --embedding-size 192 --prefetch-ahead 4 --attention-layers 2 --word-tier 168 --tier-own-step --prior-strength-gate"
for pair in collect:full word:noword mem:nomem both:neither; do
  mode=${pair%%:*}; arm=mean_${pair#*:}
  t0=$(date +%s)
  KO_MODE=$mode nice -n 19 ionice -c 3 python -B $S/ko/maxrss.py python -B $S/ko/mean_ablate.py \
    --config configs/pg19split20m_crossfit20_ps100_e1.json --run-id ko_$arm --output $S/ko/$arm.csv $common \
    --eval-only --resume-checkpoint $S/ko/full.pt --char-loss-output $S/ko/char_losses_$arm.npz > $S/ko/$arm.log 2>&1
  rc=$?
  echo "$arm exit=$rc $(grep -o 'char_losses heldout_ppl=[0-9.]*' $S/ko/$arm.log) $(grep -o 'maxrss_kb=[0-9]*' $S/ko/$arm.log) $(( $(date +%s) - t0 ))s"
  grep "^mean \|^ko_mode" $S/ko/$arm.log
  [ $rc -eq 0 ] || break
done
echo "done $(date -u +%H:%M)"
