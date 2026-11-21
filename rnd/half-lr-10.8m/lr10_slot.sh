#!/bin/sh
# imm's slot $1: slot 126's 10.8M imm at half the learning rate (the user,
# in air's session, 2026-11-20, asked to queue them; air's mail 1c535e8a):
# --lr 5e-4 --min-lr 5e-5 cosine, otherwise unchanged (h448, 3 blocks, 8
# heads, word tier 392, 1,024 character records, no prior, dropout 0,
# 8 epochs, --bf16 at slot 126's cap 36864, scored in fp32). Against slot
# 126's 2.8188 at 1e-3. The transformer ran at 1e-3 too; its 5e-4 run is
# not yet approved.
# Fixed worktree $S/imm-q at 5ae5c64 (branch depth-records; the default
# path is unchanged from slot 126's 00b6a52: CPU smoke 15.409).
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
cd $S/imm-q || exit 1
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
done) > $S/arch/rss_$sid.log 2>&1 &
rss=$!

cfg=configs/pg19split100m_noprior.json
base="--count-prior none --prior-residual-gate none --prior-input none --prior-residual-state-features= --prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --eval-streams 64 --lr 1e-3 --segment-rule word --max-memory 128 --segments-per-update 64 --dropout 0.0 --lr-schedule cosine --min-lr 1e-4 --carry-bound gru --memory-state gru --prefetch-ahead 4 --tier-own-step --char-memory 1024 --documents data/pg19/imm-split-100m/manifest.json --epochs 8"
small="$base --hidden-size 192 --embedding-size 192 --attention-layers 2 --attention-heads 4 --word-tier 168"

# An 8-epoch bf16 run, then its per-character losses in fp32.
train() {  # run id, model flags, training-only flags
  echo "start imm $1 $(date -Is)"
  echo "  args: $2 --bf16 $3"
  python -B scripts/run_segment_memory_model.py --config $cfg --run-id $1 --output runs/$1.csv $2 --bf16 $3 > runs/$1.log 2>&1
  rc=$?
  echo "end imm $1 exit=$rc $(date -Is) $(grep -o 'best_val_ppl=[0-9.]*' runs/$1.log | tail -1)"
  [ $rc -eq 0 ] || return $rc
  mkdir -p $S/charloss
  [ -f runs/${1}_${1}.best.pt ] || return 1
  python -B scripts/run_segment_memory_model.py --config $cfg --run-id cl_$1 --output $S/charloss/cl_$1.csv $2 \
    --eval-only --resume-checkpoint runs/${1}_${1}.best.pt --char-loss-output runs/char_losses_${1}_best.npz > $S/charloss/$1.log 2>&1
  rc=$?
  echo "char losses $1 (fp32) exit=$rc $(grep -o 'char_losses heldout_ppl=[0-9.]*' $S/charloss/$1.log) $(date -Is)"
  return $rc
}
half=$(echo "$base" | sed -e 's/ --lr 1e-3 / --lr 5e-4 /' -e 's/ --min-lr 1e-4 / --min-lr 5e-5 /')
case "$half" in *" --lr 5e-4 "*" --min-lr 5e-5 "*) ;; *) echo "lr not set: $half"; finish 1 ;; esac
train pg19s100m_h448_par_l3_w392_own_np_doc_mm128_do0_cm1024_rel_lr5e-4_bf16_e8 \
  "$half --hidden-size 448 --embedding-size 448 --attention-layers 3 --attention-heads 8 --word-tier 392" \
  "--max-update-chars 36864" || finish 1
finish 0
