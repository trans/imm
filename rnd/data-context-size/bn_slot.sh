#!/bin/sh
# imm's slot $1: where the bottleneck is (the user, 2026-11-18, asked where
# the real bottleneck is, and approved queueing these). Three runs of the 128-record model (R1 with --max-memory 128,
# 2.7619), each changing one thing, from worktree imm-run (topic-tiers 543ef84):
#   half: the first 10M training characters only (--train-prefix-chars), 82
#         epochs, so the same number of updates: how much the residual uses data;
#   m256: 256 records: how much more context gives;
#   h256: the character GRU, embedding and records at 256 wide: how much size gives.
# Each best checkpoint then gives per-character held-out losses (by piece).
# /proc sampling every minute to $S/bn_rss_$1.log. Ends the slot.
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
cd $S/imm-run || exit 1
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
done) > $S/bn_rss_$sid.log 2>&1 &
rss=$!
cfg=configs/pg19split20m_crossfit20_ps100_e1.json
docs="--documents data/pg19/imm-split-20m/manifest.json"
base="--prior-residual-scale 1.0 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows $M/pg19split20m_kn --segment-rule word --max-memory 64 --segments-per-update 64 --dropout 0.2 --eval-streams 64 --lr-schedule cosine --min-lr 1e-4 --epochs 41 --carry-bound gru --memory-state gru --prior-rows $M/pg19split20m_gated_d24_pg10g --prior-residual-state-features reliability,depth_norm --hidden-size 192 --embedding-size 192 --prefetch-ahead 4 --attention-layers 2 --tier-own-step --prior-strength-gate"
stack="--word-tier 128 --sentence-tier 48 --paragraph-tier 48"
mkdir -p $S/charloss
b128=$(echo "$base" | sed "s/--max-memory 64/--max-memory 128/")
for arm in half m256 h256; do
  case $arm in
    half) id=pg19s20m_h192_par_l2_w168_own_pg_d24_doc_mm128_tp10m_e82
          common=$(echo "$b128 --word-tier 168 --prior-residual-l2 0.05 $docs --train-prefix-chars 10000000" | sed "s/--epochs 41/--epochs 82/") ;;
    m256) id=pg19s20m_h192_par_l2_w168_own_pg_d24_doc_mm256_e41
          common=$(echo "$base --word-tier 168 --prior-residual-l2 0.05 $docs" | sed "s/--max-memory 64/--max-memory 256/") ;;
    h256) id=pg19s20m_h256_par_l2_w168_own_pg_d24_doc_mm128_e41
          common=$(echo "$b128 --word-tier 168 --prior-residual-l2 0.05 $docs" | sed "s/--hidden-size 192 --embedding-size 192/--hidden-size 256 --embedding-size 256/") ;;
  esac
  echo "start $arm $id $(date -Is)"
  echo "  args: $common"
  python -B scripts/run_segment_memory_model.py --config $cfg --run-id $id --output runs/$id.csv $common --drop-row-cache > runs/$id.log 2>&1
  rc=$?
  echo "end $arm exit=$rc $(date -Is) $(grep -o 'best_val_ppl=[0-9.]*' runs/$id.log | tail -1)"
  if [ $rc -eq 0 ] && [ -f runs/${id}_${id}.best.pt ]; then
    python -B scripts/run_segment_memory_model.py --config $cfg --run-id cl_$id --output $S/charloss/cl_$id.csv $common --drop-row-cache \
      --eval-only --resume-checkpoint runs/${id}_${id}.best.pt --char-loss-output runs/char_losses_${id}_best.npz > $S/charloss/$id.log 2>&1
    echo "char losses $arm exit=$? $(grep -o 'char_losses heldout_ppl=[0-9.]*' $S/charloss/$id.log) $(date -Is)"
  fi
done
kill $rss
$SLOTS done @imm
echo "slot ended $(date -Is)"
