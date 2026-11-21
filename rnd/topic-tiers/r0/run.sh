#!/bin/sh
# R0 of the topic-tier plan (the user's go-ahead via @air e7072a98): slot 63's
# best checkpoint scored one stream per held-out piece, each from its start
# with zeros and an empty memory (--documents, topic-tiers 4616a6a), against
# today's 64 streams with a 256-segment warm-up. Eval only; in imm's slot.
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
M=/home/trans/Projects/air/mobius/runs/imm
cd $S/imm-topic || exit 1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TMPDIR=$S
echo "code $(git rev-parse --short HEAD) $(git status --porcelain --untracked-files=no | wc -l) changed files"
common="--prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows $M/pg19split20m_kn --segment-rule word --max-memory 64 --segments-per-update 64 --dropout 0.2 --eval-streams 64 --lr-schedule cosine --min-lr 1e-4 --epochs 41 --carry-bound gru --memory-state gru --prior-rows $M/pg19split20m_gated_d24_pg10g --prior-residual-state-features reliability,depth_norm --hidden-size 192 --embedding-size 192 --prefetch-ahead 4 --attention-layers 2 --word-tier 168 --tier-own-step --prior-strength-gate"
t0=$(date +%s)
nice -n 19 ionice -c 3 python -B $S/ko/maxrss.py python -B scripts/run_segment_memory_model.py \
  --config configs/pg19split20m_crossfit20_ps100_e1.json --run-id r0 --output $S/r0/r0.csv $common \
  --documents data/pg19/imm-split-20m/manifest.json --eval-only \
  --resume-checkpoint runs/pg19s20m_h192_par_l2_w168_own_pg_d24_e41_pg19s20m_h192_par_l2_w168_own_pg_d24_e41.best.pt \
  --char-loss-output $S/r0/char_losses_r0.npz > $S/r0/r0.log 2>&1
echo "r0 exit=$? $(grep -o 'char_losses heldout_ppl=[0-9.]*' $S/r0/r0.log) $(grep -o 'maxrss_kb=[0-9]*' $S/r0/r0.log) $(( $(date +%s) - t0 ))s"
grep "^documents\|^resume" $S/r0/r0.log
