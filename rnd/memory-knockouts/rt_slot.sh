#!/bin/sh
# imm's slot $1 (the user, 2026-11-17, asked for both to be queued): what
# the word GRU and the memory attention are worth at slot 63's size, by
# retraining slot 63's run (h192, --word-tier 168, --tier-own-step,
# --prior-strength-gate, d24 rows, 41 epochs = 820M characters, cosine)
# without each, back to back:
#   noword: no --word-tier, no --tier-own-step;
#   noattn: --no-memory-attention (the blocks see no records, MLPs kept).
# Then each best checkpoint's per-character held-out losses, for pairing
# against slot 63's. /proc sampling every minute to $S/rt_rss.log. Worktree
# imm-noattn, branch memory-attention-off 3989c87. Ends the slot.
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
cd $S/imm-noattn || exit 1
echo "code $(git rev-parse --short HEAD) $(git status --porcelain --untracked-files=no | wc -l) changed files"
M=/home/trans/Projects/air/mobius/runs/imm
rowfiles="$M/pg19split20m_gated_d24_pg10g.train_log_probs.dat $M/pg19split20m_gated_d24_pg10g.train_gate_features.dat $M/pg19split20m_kn.train_ev_ptr.dat $M/pg19split20m_kn.train_ev_pos.dat $M/pg19split20m_kn.train_ev_val.dat $M/pg19split20m_kn.train_ev_shallow_index.dat"
(while :; do
  line="$(date -u +%H:%M:%S) $(free -m | awk '/^Mem/ {printf "available=%dMB ", $7} /^Swap/ {printf "swap_used=%dMB ", $3}')"
  line="$line mem_some_avg60=$(awk '/^some/ {split($3, a, "="); print a[2]}' /proc/pressure/memory)"
  line="$line rows_cached=$(fincore -b -n -o RES $rowfiles 2>/dev/null | awk '{s += $1} END {printf "%dMB", s / 1048576}')"
  for p in $(pgrep -f "scripts/run_segment_memory_model.py --config"); do
    [ -r /proc/$p/status ] && line="$line | pid=$p $(awk '/^(VmRSS|RssAnon|RssFile|VmSwap):/ {printf "%s=%dMB ", $1, $2/1024}' /proc/$p/status)"
  done
  echo "$line"
  sleep 60
done) > $S/rt_rss.log 2>&1 &
rss=$!
base="--prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows $M/pg19split20m_kn --segment-rule word --max-memory 64 --segments-per-update 64 --dropout 0.2 --eval-streams 64 --lr-schedule cosine --min-lr 1e-4 --epochs 41 --carry-bound gru --memory-state gru --prior-rows $M/pg19split20m_gated_d24_pg10g --prior-residual-state-features reliability,depth_norm --hidden-size 192 --embedding-size 192 --prefetch-ahead 4 --attention-layers 2 --prior-strength-gate --drop-row-cache"
cfg=configs/pg19split20m_crossfit20_ps100_e1.json
mkdir -p $S/charloss
for arm in noword noattn; do
  case $arm in
    noword) id=pg19s20m_h192_par_l2_noword_pg_d24_e41; common="$base" ;;
    noattn) id=pg19s20m_h192_par_l2_w168_own_noattn_pg_d24_e41; common="$base --word-tier 168 --tier-own-step --no-memory-attention" ;;
  esac
  echo "start $id $(date -Is)"
  python -B scripts/run_segment_memory_model.py --config $cfg --run-id $id --output runs/$id.csv $common > runs/$id.log 2>&1
  rc=$?
  echo "end $id exit=$rc $(date -Is) $(grep -o 'best_val_ppl=[0-9.]*' runs/$id.log | tail -1)"
  if [ $rc -eq 0 ] && [ -f runs/${id}_${id}.best.pt ]; then
    python -B scripts/run_segment_memory_model.py --config $cfg --run-id cl_$id --output $S/charloss/cl_$id.csv $common \
      --eval-only --resume-checkpoint runs/${id}_${id}.best.pt --char-loss-output runs/char_losses_${id}_best.npz > $S/charloss/$id.log 2>&1
    echo "char losses $id exit=$? $(grep -o 'char_losses heldout_ppl=[0-9.]*' $S/charloss/$id.log) $(date -Is)"
  fi
done
kill $rss
$SLOTS done @imm
echo "slot ended $(date -Is)"
# Paired against slot 63, by bucket.
mkdir -p $S/rt
ln -sf /home/trans/Projects/air/imm/runs/char_losses_pg19s20m_h192_par_l2_w168_own_pg_d24_e41_best.npz $S/rt/char_losses_full.npz
ln -sf /home/trans/Projects/air/imm/runs/char_losses_pg19s20m_h192_par_l2_noword_pg_d24_e41_best.npz $S/rt/char_losses_noword.npz
ln -sf /home/trans/Projects/air/imm/runs/char_losses_pg19s20m_h192_par_l2_w168_own_noattn_pg_d24_e41_best.npz $S/rt/char_losses_nomem.npz
CUDA_VISIBLE_DEVICES= nice -n 19 python -B $S/ko_buckets.py /home/trans/Projects/air/imm $S/rt > $S/rt/buckets.txt 2>&1
echo "buckets exit=$? -> $S/rt/buckets.txt"
