#!/bin/sh
# imm's slot $1 (the user approved it via @air 21b43990: C, the no-prior
# row): slot 50's model (two blocks + --word-tier 256 + --tier-own-step),
# trained and scored with no prior at all (no rows, no evidence, no prior
# input, no count prior), 20 epochs, against 3.1155 with the prior (and
# 3.0616 with the learned prior strength). Then each held-out character's
# loss. Worktree imm-gate at prior-strength-gate 91939ef. Ends the slot.
# Laptop limits: nice 19, ionice idle, 2 CPU threads.
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
cd $S/imm-gate || exit 1
echo "code $(git rev-parse --short HEAD) $(git status --porcelain --untracked-files=no | wc -l) changed files"
(while :; do echo "$(date -u +%H:%M) $(free -m | awk '/^Mem/ {print "available=" $7 "MB cache=" $6 "MB"} /^Swap/ {print "swap_used=" $3 "MB"}' | tr '\n' ' ')"; sleep 300; done) > $S/noprior_mem.log 2>&1 &
mem=$!
cfg=configs/pg19split20m_crossfit20_ps100_e1.json
id=pg19s20m_h256_par_l2_w256_own_noprior_e20
run() {
  python -B scripts/run_segment_memory_model.py --config $cfg --count-prior none --prior-residual-gate none --prior-input none \
    --prior-residual-state-features "" --prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 \
    --segment-rule word --max-memory 64 --segments-per-update 64 --dropout 0.2 --eval-streams 64 --lr-schedule cosine --min-lr 1e-4 \
    --carry-bound gru --memory-state gru --hidden-size 256 --embedding-size 256 --prefetch-ahead 4 --attention-layers 2 \
    --word-tier 256 --tier-own-step --epochs 20 "$@"
}
echo "start $id $(date -Is)"
run --run-id $id --output runs/$id.csv > runs/$id.log 2>&1
echo "end $id exit=$? $(date -Is)"
if [ -f runs/${id}_${id}.best.pt ]; then
  mkdir -p $S/charloss
  run --run-id cl_$id --output $S/charloss/cl_$id.csv --eval-only --resume-checkpoint runs/${id}_${id}.best.pt \
    --char-loss-output runs/char_losses_${id}_best.npz > $S/charloss/$id.log 2>&1
  echo "char losses $id exit=$? $(grep -o 'char_losses heldout_ppl=[0-9.]*' $S/charloss/$id.log) $(date -Is)"
fi
kill $mem
$SLOTS done @imm
echo "slot ended $(date -Is)"
python -B $S/compare_tx.py . runs/char_losses_${id}_best.npz runs/char_losses_pg19s20m_h256_par_l2_w256_own_e20_best.npz > $S/noprior_vs_own.txt 2>&1
echo "buckets exit=$? -> $S/noprior_vs_own.txt"
