#!/bin/sh
# imm's slot $1, the 10.8M pair's imm side (approved by the user,
# 2026-11-19; then bf16 for imm). Branch bf16 1edcb56, worktree imm-char.
# Slot 120's probe found no size whose worst update fits, so now:
#  1. $S/s10/probe_split.py: under --bf16, at h448 x 3 blocks, h512 x 2,
#     h384 x 4 in that order, the largest --max-update-chars cap (64 x L,
#     L from 704 down) whose largest unsplit update stays under 7.0 GiB,
#     then the worst update split at it, and scoring's worst case in fp32
#     and bf16. It picks the first size where all of these fit.
#  2. Check bf16 on slot 115's model (h192, 2 blocks, 1,024 records): one
#     epoch of its 8-epoch schedule, against fp32's 3.711 after epoch 1. If
#     it's more than 0.5% worse, stop.
#  3. The chosen ~10.8M imm, dropout 0, 8 epochs, --bf16, --max-update-chars
#     at the cap, otherwise slot 115's settings; then its per-character
#     losses scored in fp32.
# Stops and ends the slot at any failure. RSS and GPU memory every minute,
# with a guard that stops the run below 800 MB available.
# Laptop limits: nice 19, ionice idle, 2 CPU threads.
renice -n 19 -p $$ >/dev/null
ionice -c 3 -p $$
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
export TMPDIR=$S PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SLOTS="sh /home/trans/Projects/air/tools/slots"
sid=$1
finish() { kill $rss 2>/dev/null; $SLOTS done @imm; echo "slot ended $(date -Is)"; exit $1; }
while ! $SLOTS list 2>/dev/null | grep -q "^in use: $sid @imm,"; do
  if ! $SLOTS list 2>/dev/null | grep -qE "^(in use: |[0-9]+\. )$sid @imm,"; then
    echo "slot $sid is neither queued nor in use $(date -Is)"; exit 1
  fi
  sleep 20
done
echo "slot $sid granted $(date -Is)"
cd $S/imm-char || exit 1
echo "code $(git rev-parse --short HEAD) $(git status --porcelain --untracked-files=no | wc -l) changed files"
(while :; do
  avail=$(free -m | awk '/^Mem/ {print $7}')
  line="$(date -u +%H:%M:%S) available=${avail}MB $(free -m | awk '/^Swap/ {printf "swap_used=%dMB", $3}') gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr -d ' ')"
  for p in $(pgrep -f "run_segment_memory_model.py --config"); do
    [ -r /proc/$p/status ] && line="$line | pid=$p $(awk '/^(VmRSS|RssAnon|VmSwap):/ {printf "%s=%dMB ", $1, $2/1024}' /proc/$p/status)"
  done
  echo "$line"
  if [ "$avail" -lt 800 ]; then
    echo "GUARD: available ${avail}MB < 800MB, stopping the run"; pkill -f "run_segment_memory_model.py --config"
  fi
  sleep 60
done) > $S/s10/rss_$sid.log 2>&1 &
rss=$!

echo "probe start $(date -Is)"
python3 -B $S/s10/probe_split.py $S/s10/worst_window.txt > $S/s10/probe_split_$sid.log 2>&1
echo "probe exit=$? $(date -Is)"
grep -v "InitGoogleLogging\|CUDACachingAllocator" $S/s10/probe_split_$sid.log
choice=$(grep '^CHOICE' $S/s10/probe_split_$sid.log)
case "$choice" in
  "CHOICE h="*) ;;
  *) echo "no size fits ($choice): stopping"; finish 1 ;;
esac
h=$(echo "$choice" | sed 's/.*h=\([0-9]*\).*/\1/'); l=$(echo "$choice" | sed 's/.* l=\([0-9]*\).*/\1/')
heads=$(echo "$choice" | sed 's/.*heads=\([0-9]*\).*/\1/'); tier=$(echo "$choice" | sed 's/.*tier=\([0-9]*\).*/\1/')
cap=$(echo "$choice" | sed 's/.*cap=\([0-9]*\).*/\1/')
echo "chose h=$h l=$l heads=$heads tier=$tier cap=$cap"

cfg=configs/pg19split100m_noprior.json
base="--count-prior none --prior-residual-gate none --prior-input none --prior-residual-state-features= --prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --eval-streams 64 --lr 1e-3 --segment-rule word --max-memory 128 --segments-per-update 64 --dropout 0.0 --lr-schedule cosine --min-lr 1e-4 --carry-bound gru --memory-state gru --prefetch-ahead 4 --tier-own-step --char-memory 1024 --documents data/pg19/imm-split-100m/manifest.json --epochs 8"

vid=pg19s100m_h192_par_l2_w168_own_np_doc_mm128_do0_cm1024_bf16_check
echo "check start $vid $(date -Is)"
python -B scripts/run_segment_memory_model.py --config $cfg --run-id $vid --output runs/$vid.csv $base \
  --hidden-size 192 --embedding-size 192 --attention-layers 2 --attention-heads 4 --word-tier 168 --bf16 > runs/$vid.log 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null && ! grep -q "^epoch=1 done" runs/$vid.log; do sleep 10; done
kill $pid 2>/dev/null; wait $pid 2>/dev/null
line=$(grep "^epoch=1 done" runs/$vid.log)
val=$(echo "$line" | grep -o 'val_ppl=[0-9.]*' | cut -d= -f2)
echo "check end $(date -Is) epoch 1 val_ppl=$val (fp32: 3.711) $(grep '^epoch=1 profile' runs/$vid.log | grep -o 'train_sec=[0-9.]*')"
if [ -z "$val" ] || ! awk -v v="$val" 'BEGIN { exit !(v <= 3.711 * 1.005) }'; then
  echo "bf16 check failed: stopping"; finish 1
fi

id=pg19s100m_h${h}_par_l${l}_w${tier}_own_np_doc_mm128_do0_cm1024_bf16_e8
args="$base --hidden-size $h --embedding-size $h --attention-layers $l --attention-heads $heads --word-tier $tier"
echo "start imm $id $(date -Is)"
echo "  args: $args --bf16 --max-update-chars $cap"
python -B scripts/run_segment_memory_model.py --config $cfg --run-id $id --output runs/$id.csv $args --bf16 \
  --max-update-chars $cap > runs/$id.log 2>&1
rc=$?
echo "end imm exit=$rc $(date -Is) $(grep -o 'best_val_ppl=[0-9.]*' runs/$id.log | tail -1)"
mkdir -p $S/charloss
if [ $rc -eq 0 ] && [ -f runs/${id}_${id}.best.pt ]; then
  python -B scripts/run_segment_memory_model.py --config $cfg --run-id cl_$id --output $S/charloss/cl_$id.csv $args \
    --eval-only --resume-checkpoint runs/${id}_${id}.best.pt --char-loss-output runs/char_losses_${id}_best.npz > $S/charloss/$id.log 2>&1
  echo "char losses imm (fp32) exit=$? $(grep -o 'char_losses heldout_ppl=[0-9.]*' $S/charloss/$id.log) $(date -Is)"
fi
finish $rc
