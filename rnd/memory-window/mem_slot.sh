#!/bin/sh
# imm's slot $1, running the topic-tier plan's arms $2 back to back (the user
# approved the plan via @air e7072a98, 2026-11-17, with the overtrust arm
# included). From worktree imm-run, detached at a fixed commit, so a run
# never picks up code still being written. Every arm is slot 63's model (h192,
# two blocks, --prior-strength-gate, d24 rows, 41 epochs, cosine), changed as:
#   r0:  slot 63's best checkpoint itself, scored one stream per piece (eval only);
#   r1:  --documents (each piece read from its start);
#   ot:  r1 with --prior-residual-l2 0 (the overtrust arm, A + B);
#   r2:  r1 with the stack: --word-tier 128 --sentence-tier 48 --paragraph-tier 48 --tier-bptt 32;
#   r2c: r2 with --tier-input end+centroid;
#   r2s: r2 without the long windows (--tier-bptt 0);
#   r3:  r2 (or r2c, R3_INPUT=end+centroid) with --top-down.
#   s1:  the sentence tier alone, added to slot 63's word tier 168 (not paid for
#        from it): --sentence-tier 48 --tier-bptt 32 --tier-input end+centroid
#        (the user, 2026-11-17, wanted sentences working before paragraphs
#        come into it, and found R2's shrinking of the word tier an unfair
#        comparison).
#   s2:  s1 with --tier-position-gate: each position scales the sentence tier's
#        add by a gate read from its own state (the user, 2026-11-18, chose to
#        build the per-position gate first);
#   s2map: s2's best checkpoint scored again, with the gate's factors kept by
#        position (gate_map.py) and bucketed (gate_buckets.py) to $S/s2/.
#   m32, m128: r1 with the attention reading the last 32 or 128 word records
#        instead of 64 (the user, 2026-11-18, asked to try 128 records, and 32,
#        and see what changes). The window applies in training and when scoring.
# Each trained arm's best checkpoint then gives per-character held-out losses
# (by piece). /proc sampling every minute to $S/topic_rss_$1.log. Ends the slot.
# Laptop limits: nice 19, ionice idle, 2 CPU threads.
renice -n 19 -p $$ >/dev/null
ionice -c 3 -p $$
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
export TMPDIR=$S
SLOTS="sh /home/trans/Projects/air/tools/slots"
sid=$1
arms=$2
while ! $SLOTS list 2>/dev/null | grep -q "^in use: $sid @imm,"; do
  if ! $SLOTS list 2>/dev/null | grep -qE "^(in use: |[0-9]+\. )$sid @imm,"; then
    echo "slot $sid is neither queued nor in use $(date -Is)"; exit 1
  fi
  sleep 20
done
echo "slot $sid granted $(date -Is), arms: $arms"
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
done) > $S/topic_rss_$sid.log 2>&1 &
rss=$!
cfg=configs/pg19split20m_crossfit20_ps100_e1.json
docs="--documents data/pg19/imm-split-20m/manifest.json"
base="--prior-residual-scale 1.0 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows $M/pg19split20m_kn --segment-rule word --max-memory 64 --segments-per-update 64 --dropout 0.2 --eval-streams 64 --lr-schedule cosine --min-lr 1e-4 --epochs 41 --carry-bound gru --memory-state gru --prior-rows $M/pg19split20m_gated_d24_pg10g --prior-residual-state-features reliability,depth_norm --hidden-size 192 --embedding-size 192 --prefetch-ahead 4 --attention-layers 2 --tier-own-step --prior-strength-gate"
stack="--word-tier 128 --sentence-tier 48 --paragraph-tier 48"
mkdir -p $S/r0 $S/charloss
for arm in $arms; do
  case $arm in
    r0)
      python -B $S/ko/maxrss.py python -B scripts/run_segment_memory_model.py --config $cfg --run-id r0 --output $S/r0/r0.csv \
        $base --word-tier 168 --prior-residual-l2 0.05 $docs --eval-only \
        --resume-checkpoint runs/pg19s20m_h192_par_l2_w168_own_pg_d24_e41_pg19s20m_h192_par_l2_w168_own_pg_d24_e41.best.pt \
        --char-loss-output $S/r0/char_losses_r0.npz > $S/r0/r0.log 2>&1
      echo "r0 exit=$? $(grep -o 'char_losses heldout_ppl=[0-9.]*' $S/r0/r0.log) $(grep -o 'maxrss_kb=[0-9]*' $S/r0/r0.log) $(date -Is)"
      continue ;;
    r1) id=pg19s20m_h192_par_l2_w168_own_pg_d24_doc_e41; common="$base --word-tier 168 --prior-residual-l2 0.05 $docs" ;;
    ot) id=pg19s20m_h192_par_l2_w168_own_pg_d24_doc_l2z_e41; common="$base --word-tier 168 --prior-residual-l2 0 $docs" ;;
    r2) id=pg19s20m_h192_par_l2_w128s48p48_bptt32_own_pg_d24_doc_e41; common="$base $stack --tier-bptt 32 --prior-residual-l2 0.05 $docs" ;;
    r2c) id=pg19s20m_h192_par_l2_w128s48p48_bptt32_cent_own_pg_d24_doc_e41; common="$base $stack --tier-bptt 32 --tier-input end+centroid --prior-residual-l2 0.05 $docs" ;;
    r2s) id=pg19s20m_h192_par_l2_w128s48p48_own_pg_d24_doc_e41; common="$base $stack --prior-residual-l2 0.05 $docs" ;;
    r3) id=pg19s20m_h192_par_l2_w128s48p48_bptt32_${R3_INPUT:-end}_td_own_pg_d24_doc_e41; common="$base $stack --tier-bptt 32 --tier-input ${R3_INPUT:-end} --top-down --prior-residual-l2 0.05 $docs" ;;
    s1) id=pg19s20m_h192_par_l2_w168s48_bptt32_cent_own_pg_d24_doc_e41; common="$base --word-tier 168 --sentence-tier 48 --tier-bptt 32 --tier-input end+centroid --prior-residual-l2 0.05 $docs" ;;
    s2) id=pg19s20m_h192_par_l2_w168s48_bptt32_cent_pgate_own_pg_d24_doc_e41; common="$base --word-tier 168 --sentence-tier 48 --tier-bptt 32 --tier-input end+centroid --tier-position-gate --prior-residual-l2 0.05 $docs" ;;
    s2map)
      id=pg19s20m_h192_par_l2_w168s48_bptt32_cent_pgate_own_pg_d24_doc_e41
      mkdir -p $S/s2
      GATE_MAP=$S/s2/gate_map.npz python -B $S/gate_map.py --config $cfg --run-id gm_$id --output $S/s2/gm.csv \
        $base --word-tier 168 --sentence-tier 48 --tier-bptt 32 --tier-input end+centroid --tier-position-gate --prior-residual-l2 0.05 $docs \
        --drop-row-cache --eval-only --resume-checkpoint runs/${id}_${id}.best.pt --char-loss-output $S/s2/gm.npz > $S/s2/gate_map.log 2>&1
      echo "s2map exit=$? $(grep -o 'gate_map n=[0-9]* mean=[0-9.]*' $S/s2/gate_map.log) $(date -Is)"
      python3 $S/gate_buckets.py $S/imm-run $S/s2/gate_map.npz $S/s2/gm.npz > $S/s2/gate_buckets.txt 2>&1
      continue ;;
    m32|m128)
      mem=${arm#m}
      # the window is in $base: re-read it with this arm's size
      base=$(echo "$base" | sed "s/--max-memory [0-9]*/--max-memory $mem/")
      id=pg19s20m_h192_par_l2_w168_own_pg_d24_doc_mm${mem}_e41; common="$base --word-tier 168 --prior-residual-l2 0.05 $docs" ;;
    *) echo "unknown arm $arm"; continue ;;
  esac
  echo "start $arm $id $(date -Is)"
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
if [ -f $S/r0/char_losses_r0.npz ] && [ ! -f $S/r0/r0_vs_usual.txt ]; then
  python3 $S/r0/compare.py runs/char_losses_pg19s20m_h192_par_l2_w168_own_pg_d24_e41_best.npz $S/r0/char_losses_r0.npz > $S/r0/r0_vs_usual.txt 2>&1
  echo "r0 vs usual exit=$? -> $S/r0/r0_vs_usual.txt"
fi
