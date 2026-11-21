#!/bin/sh
# imm's slot $1 (the user approved it, 2026-11-14): the mean-ablation
# of the attention. Re-score the base (pg19s20m_h256_m64_cbn32_e20, 3.1517)
# and parallel-memory (pg19s20m_h256_par_e20, 3.1518) best checkpoints with
# the gated block's attention output replaced (attn_mean.py, the training
# script unchanged, branch parallel-memory 787ee25): collect (unchanged, the
# control; the means), zero (must give 3.3486 / 3.3521, gate_attn = 0),
# global (its overall mean), position (its mean at each position in the
# word). Eval only. Ends the slot.
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
cd $S/imm-par || exit 1
echo "code $(git rev-parse --short HEAD) $(git status --porcelain --untracked-files=no | wc -l) changed files"
O=$S/attn_mean
mkdir -p $O
M=/home/trans/Projects/air/mobius/runs/imm
R=/home/trans/air/imm/runs
common="--prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows $M/pg19split20m_kn --segment-rule word --max-memory 64 --segments-per-update 64 --dropout 0.2 --eval-streams 64 --lr-schedule cosine --min-lr 1e-4 --epochs 20 --prior-rows $M/pg19split20m_gated_d16_pg200m --prior-residual-state-features reliability,depth_norm --hidden-size 256 --embedding-size 256 --eval-only"
cfg=configs/pg19split20m_crossfit20_ps100_e1.json
model() {  # name, run id, extra flags
  name=$1; id=$2; shift 2
  for m in collect zero global position; do
    echo "start ${name}_$m $(date -Is)"
    ABL_MODE=$m ABL_FILE=$O/means_$name.pt python -B $S/attn_mean.py --config $cfg --run-id am_${name}_$m \
      --output $O/${name}_$m.csv $common "$@" --resume-checkpoint $R/${id}_${id}.best.pt > $O/${name}_$m.log 2>&1
    echo "end ${name}_$m exit=$? $(grep -o 'eval_only checkpoint=[^ ]* val_ppl=[0-9.]*' $O/${name}_$m.log | sed 's/.*val_ppl=//') $(grep -o 'block_calls=.*' $O/${name}_$m.log)"
  done
}
model base pg19s20m_h256_m64_cbn32_e20 --carry-bound norm --carry-scale 32
model par pg19s20m_h256_par_e20 --carry-bound gru --memory-state gru
$SLOTS done @imm
echo "slot ended $(date -Is)"
