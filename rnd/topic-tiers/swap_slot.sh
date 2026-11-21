#!/bin/sh
# imm's slot $1: the topic-tier plan's swap test (the user via @air e7072a98).
# For each trained stack model, eval only: the held-out pieces scored one
# stream per piece, with the sentence tier's state, then the paragraph tier's,
# passed to the next piece's stream before every chunk (--swap-tier, topic-tiers
# 07f370d). If the loss doesn't rise, that tier's state carries nothing of
# its piece. Worktree imm-swap. nice 19, ionice idle, 2 CPU threads.
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
cd $S/imm-swap || exit 1
echo "code $(git rev-parse --short HEAD)"
M=/home/trans/Projects/air/mobius/runs/imm
cfg=configs/pg19split20m_crossfit20_ps100_e1.json
base="--prior-residual-scale 1.0 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows $M/pg19split20m_kn --segment-rule word --max-memory 64 --segments-per-update 64 --dropout 0.2 --eval-streams 64 --lr-schedule cosine --min-lr 1e-4 --epochs 41 --carry-bound gru --memory-state gru --prior-rows $M/pg19split20m_gated_d24_pg10g --prior-residual-state-features reliability,depth_norm --hidden-size 192 --embedding-size 192 --prefetch-ahead 4 --attention-layers 2 --tier-own-step --prior-strength-gate --word-tier 128 --sentence-tier 48 --paragraph-tier 48 --tier-bptt 32 --prior-residual-l2 0.05 --documents data/pg19/imm-split-20m/manifest.json"
mkdir -p $S/swap
for model in r2c:pg19s20m_h192_par_l2_w128s48p48_bptt32_cent_own_pg_d24_doc_e41:"--tier-input end+centroid" \
             r3:pg19s20m_h192_par_l2_w128s48p48_bptt32_end_centroid_td_own_pg_d24_doc_e41:"--tier-input end+centroid --top-down" \
             r3e:pg19s20m_h192_par_l2_w128s48p48_bptt32_end_td_own_pg_d24_doc_e41:"--tier-input end --top-down"; do
  name=${model%%:*}; rest=${model#*:}; id=${rest%%:*}; extra=$(echo ${rest#*:} | tr -d '"')
  ck=runs/${id}_${id}.best.pt
  [ -f $ck ] || { echo "$name: no $ck, skipped"; continue; }
  for tier in none sentence paragraph; do
    swap=""; [ $tier = none ] || swap="--swap-tier $tier"
    python -B scripts/run_segment_memory_model.py --config $cfg --run-id swap_${name}_$tier --output $S/swap/${name}_$tier.csv $base $extra \
      --drop-row-cache --eval-only --resume-checkpoint $ck $swap --char-loss-output $S/swap/${name}_$tier.npz > $S/swap/${name}_$tier.log 2>&1
    echo "$name swap=$tier exit=$? $(grep -o 'char_losses heldout_ppl=[0-9.]*' $S/swap/${name}_$tier.log) $(date -Is)"
  done
done
$SLOTS done @imm
echo "slot ended $(date -Is)"
