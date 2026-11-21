#!/bin/sh
# imm's slot $1 (the user approved it via @air 0bec2f0f): the smaller
# imm with the learned prior strength at 820M, about the 1.86M
# transformer's size: two blocks, hidden 192, --word-tier 168,
# --tier-own-step, --prior-strength-gate; 1,852,908 parameters used,
# against the transformer's 1,859,532. 41 epochs = 820M characters, cosine
# over the 41, as slot 57 (h256: 3.0563) and air's s1.84m_prior_in (3.0987).
# Then its take-the-prior-away test and each held-out character's loss.
# --drop-row-cache (@air asked for the evidence cache drop): every training
# row file's pages leave the page cache once read. Every minute: /proc of
# the training process, free memory and swap, memory pressure, and the
# row files' cached MB, to $S/small_rss.log. Worktree imm-drop at
# drop-row-cache ccacad2. Ends the slot.
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
cd $S/imm-drop || exit 1
echo "code $(git rev-parse --short HEAD) $(git status --porcelain --untracked-files=no | wc -l) changed files"
M=/home/trans/Projects/air/mobius/runs/imm
rowfiles="$M/pg19split20m_gated_d16_pg200m.train_log_probs.dat $M/pg19split20m_gated_d16_pg200m.train_gate_features.dat $M/pg19split20m_kn.train_ev_ptr.dat $M/pg19split20m_kn.train_ev_pos.dat $M/pg19split20m_kn.train_ev_val.dat $M/pg19split20m_kn.train_ev_shallow_index.dat"
(while :; do
  line="$(date -u +%H:%M:%S) $(free -m | awk '/^Mem/ {printf "available=%dMB ", $7} /^Swap/ {printf "swap_used=%dMB ", $3}')"
  line="$line mem_some_avg60=$(awk '/^some/ {split($3, a, "="); print a[2]}' /proc/pressure/memory)"
  line="$line rows_cached=$(fincore -b -n -o RES $rowfiles 2>/dev/null | awk '{s += $1} END {printf "%dMB", s / 1048576}')"
  for p in $(pgrep -f "scripts/run_segment_memory_model.py --config"); do
    [ -r /proc/$p/status ] && line="$line | pid=$p $(awk '/^(VmRSS|RssAnon|RssFile|VmSwap):/ {printf "%s=%dMB ", $1, $2/1024}' /proc/$p/status)"
  done
  echo "$line"
  sleep 60
done) > $S/small_rss.log 2>&1 &
rss=$!
common="--prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows $M/pg19split20m_kn --segment-rule word --max-memory 64 --segments-per-update 64 --dropout 0.2 --eval-streams 64 --lr-schedule cosine --min-lr 1e-4 --epochs 41 --carry-bound gru --memory-state gru --prior-rows $M/pg19split20m_gated_d16_pg200m --prior-residual-state-features reliability,depth_norm --hidden-size 192 --embedding-size 192 --prefetch-ahead 4 --attention-layers 2 --word-tier 168 --tier-own-step --prior-strength-gate --drop-row-cache"
cfg=configs/pg19split20m_crossfit20_ps100_e1.json
id=pg19s20m_h192_par_l2_w168_own_pg_e41
echo "start $id $(date -Is)"
python -B scripts/run_segment_memory_model.py --config $cfg --run-id $id --output runs/$id.csv $common > runs/$id.log 2>&1
rc=$?
echo "end $id exit=$rc $(date -Is)"
if [ $rc -eq 0 ] && [ -f runs/${id}_${id}.best.pt ]; then
  python -B scripts/run_segment_memory_model.py --config $cfg --run-id abl_$id --output $S/abl_$id.csv $common \
    --eval-only --resume-checkpoint runs/${id}_${id}.best.pt --prior-ablation-output runs/prior_ablation_${id}_best.json > runs/prior_ablation_${id}_best.log 2>&1
  echo "ablation $id exit=$? $(date -Is)"
  mkdir -p $S/charloss
  python -B scripts/run_segment_memory_model.py --config $cfg --run-id cl_$id --output $S/charloss/cl_$id.csv $common \
    --eval-only --resume-checkpoint runs/${id}_${id}.best.pt --char-loss-output runs/char_losses_${id}_best.npz > $S/charloss/$id.log 2>&1
  echo "char losses $id exit=$? $(grep -o 'char_losses heldout_ppl=[0-9.]*' $S/charloss/$id.log) $(date -Is)"
fi
kill $rss
$SLOTS done @imm
echo "slot ended $(date -Is)"
for base in char_losses_pg19s20m_h256_par_l2_w256_own_pg_e41_best.npz:pg41 $S/char_losses_tx1.84m.npz:tx; do
  f=${base%%:*}; label=${base##*:}
  case $f in /*) ;; *) f=runs/$f ;; esac
  python -B $S/compare_tx.py . runs/char_losses_${id}_best.npz $f > $S/small_vs_$label.txt 2>&1
  echo "buckets vs $label exit=$? -> $S/small_vs_$label.txt"
done
