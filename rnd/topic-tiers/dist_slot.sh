#!/bin/sh
# imm's slot $1: the held-out text's full next-character distributions, for
# @air's post-hoc topic-prior test (the user via @air 0ffdfd38, 2026-11-18,
# approved asking imm). Two eval-only runs of dist_dump.py from worktree imm-run
# (c732057), one stream per piece:
#   r0: slot 63's best checkpoint (as R0, 2.7735) -> runs/dist_slot63_doc.npz
#   s1: S1's best checkpoint (2.7751)             -> runs/dist_s1_doc.npz
# Laptop limits: nice 19, ionice idle, 2 CPU threads. Ends the slot.
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
cd $S/imm-run || exit 1
echo "code $(git rev-parse --short HEAD) $(git status --porcelain --untracked-files=no | wc -l) changed files"
M=/home/trans/Projects/air/mobius/runs/imm
cfg=configs/pg19split20m_crossfit20_ps100_e1.json
docs="--documents data/pg19/imm-split-20m/manifest.json"
base="--prior-residual-scale 1.0 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows $M/pg19split20m_kn --segment-rule word --max-memory 64 --segments-per-update 64 --dropout 0.2 --eval-streams 64 --lr-schedule cosine --min-lr 1e-4 --epochs 41 --carry-bound gru --memory-state gru --prior-rows $M/pg19split20m_gated_d24_pg10g --prior-residual-state-features reliability,depth_norm --hidden-size 192 --embedding-size 192 --prefetch-ahead 4 --attention-layers 2 --tier-own-step --prior-strength-gate"
mkdir -p $S/dist
export VOCAB_TXT=data/pg19/imm-split-20m/vocab.txt
for arm in r0 s1; do
  case $arm in
    r0) ck=runs/pg19s20m_h192_par_l2_w168_own_pg_d24_e41_pg19s20m_h192_par_l2_w168_own_pg_d24_e41.best.pt
        out=runs/dist_slot63_doc.npz; flags="--word-tier 168" ;;
    s1) id=pg19s20m_h192_par_l2_w168s48_bptt32_cent_own_pg_d24_doc_e41; ck=runs/${id}_${id}.best.pt
        out=runs/dist_s1_doc.npz; flags="--word-tier 168 --sentence-tier 48 --tier-bptt 32 --tier-input end+centroid" ;;
  esac
  DIST_OUT=$out python -B $S/dist_dump.py --config $cfg --run-id dist_$arm --output $S/dist/$arm.csv \
    $base $flags --prior-residual-l2 0.05 $docs --drop-row-cache --eval-only \
    --resume-checkpoint $ck --char-loss-output $S/dist/char_losses_$arm.npz > $S/dist/$arm.log 2>&1
  echo "$arm exit=$? $(grep -o 'char_losses heldout_ppl=[0-9.]*' $S/dist/$arm.log) $(grep -o 'dist_dump n=.*' $S/dist/$arm.log) $(date -Is)"
done
$SLOTS done @imm
echo "slot ended $(date -Is)"
