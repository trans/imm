#!/bin/sh
# imm's slot $1 (the user approved them via @air 21b43990, 2026-11-16): runs
# A and B of the bottleneck plan, each 20 epochs on the word tier with own
# step, against 3.1155 (slot 50). A: the residual's leash off
# (--prior-residual-l2 0). B: --prior-strength-gate, the leash as it was.
# Each then gets the take-the-prior-away test and each held-out character's
# loss. /proc RSS (VmRSS, RssAnon, RssFile, RssShmem) of the training
# process every minute to $S/gate_rss.log. Worktree imm-gate at
# prior-strength-gate 91939ef. Ends the slot.
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
(while :; do echo "$(date -u +%H:%M) $(free -m | awk '/^Mem/ {print "available=" $7 "MB cache=" $6 "MB"} /^Swap/ {print "swap_used=" $3 "MB"}' | tr '\n' ' ')"; sleep 300; done) > $S/gate_mem.log 2>&1 &
mem=$!
(while :; do
  for p in $(pgrep -f "scripts/run_segment_memory_model.py --config"); do
    [ -r /proc/$p/status ] && echo "$(date -u +%H:%M:%S) pid=$p $(awk '/^(VmRSS|RssAnon|RssFile|RssShmem|VmSwap):/ {printf "%s=%dMB ", $1, $2/1024}' /proc/$p/status)"
  done
  sleep 60
done) > $S/gate_rss.log 2>&1 &
rss=$!
M=/home/trans/Projects/air/mobius/runs/imm
common="--prior-residual-scale 1.0 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows $M/pg19split20m_kn --segment-rule word --max-memory 64 --segments-per-update 64 --dropout 0.2 --eval-streams 64 --lr-schedule cosine --min-lr 1e-4 --carry-bound gru --memory-state gru --prior-rows $M/pg19split20m_gated_d16_pg200m --prior-residual-state-features reliability,depth_norm --hidden-size 256 --embedding-size 256 --prefetch-ahead 4 --attention-layers 2 --word-tier 256 --tier-own-step --epochs 20"
cfg=configs/pg19split20m_crossfit20_ps100_e1.json
mkdir -p $S/charloss
for spec in "pg19s20m_h256_par_l2_w256_own_l2off_e20 --prior-residual-l2 0" \
  "pg19s20m_h256_par_l2_w256_own_pg_e20 --prior-residual-l2 0.05 --prior-strength-gate"; do
  set -- $spec
  id=$1
  shift
  echo "start $id $(date -Is)"
  python -B scripts/run_segment_memory_model.py --config $cfg --run-id $id --output runs/$id.csv $common "$@" > runs/$id.log 2>&1
  echo "end $id exit=$? $(date -Is)"
  if [ -f runs/${id}_${id}.best.pt ]; then
    python -B scripts/run_segment_memory_model.py --config $cfg --run-id abl_$id --output $S/abl_$id.csv $common "$@" \
      --eval-only --resume-checkpoint runs/${id}_${id}.best.pt --prior-ablation-output runs/prior_ablation_${id}_best.json > runs/prior_ablation_${id}_best.log 2>&1
    echo "ablation $id exit=$? $(date -Is)"
    python -B scripts/run_segment_memory_model.py --config $cfg --run-id cl_$id --output $S/charloss/cl_$id.csv $common "$@" \
      --eval-only --resume-checkpoint runs/${id}_${id}.best.pt --char-loss-output runs/char_losses_${id}_best.npz > $S/charloss/$id.log 2>&1
    echo "char losses $id exit=$? $(grep -o 'char_losses heldout_ppl=[0-9.]*' $S/charloss/$id.log) $(date -Is)"
  else
    echo "no checkpoint runs/${id}_${id}.best.pt"
  fi
done
kill $mem $rss
$SLOTS done @imm
echo "slot ended $(date -Is)"
B=runs/char_losses_pg19s20m_h256_par_l2_w256_own_e20_best.npz
for id in pg19s20m_h256_par_l2_w256_own_l2off_e20 pg19s20m_h256_par_l2_w256_own_pg_e20; do
  python -B $S/compare_tx.py . runs/char_losses_${id}_best.npz $B > $S/gate_vs_own_${id}.txt 2>&1
  echo "buckets $id exit=$? -> $S/gate_vs_own_${id}.txt"
done
