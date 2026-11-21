#!/bin/sh
# imm's slot $1 (only if the user says yes): how much the trained models
# lean on the history. Re-score the base (pg19s20m_h256_m64_cbn32_e20,
# 3.1517) and the parallel-memory run (pg19s20m_h256_par_e20) on the
# held-out text with the block's attention gate set to 0 (history off, the
# MLP kept), then with its MLP gate set to 0 (the MLP off, history kept).
# Eval only, no training; branch parallel-memory 787ee25 (the default path
# is bit-identical to the code the base trained on). Ends the slot.
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
mkdir -p $S/gate0
M=/home/trans/Projects/air/mobius/runs/imm
R=/home/trans/air/imm/runs
common="--prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows $M/pg19split20m_kn --segment-rule word --max-memory 64 --segments-per-update 64 --dropout 0.2 --eval-streams 64 --lr-schedule cosine --min-lr 1e-4 --epochs 20 --prior-rows $M/pg19split20m_gated_d16_pg200m --prior-residual-state-features reliability,depth_norm --hidden-size 256 --embedding-size 256"
cfg=configs/pg19split20m_crossfit20_ps100_e1.json
score() {  # name, checkpoint, gate to zero (none|attn|mlp), extra flags
  name=$1; ck=$2; gate=$3; shift 3
  python -B -c "
import torch, sys
ck = torch.load('$ck', map_location='cpu', weights_only=False)
sd = ck['model_state_dict']
if '$gate' != 'none':
    key = 'gated_xattn_blocks.0.gate_$gate'
    print(key, 'tanh was', float(torch.tanh(sd[key])))
    sd[key] = torch.zeros_like(sd[key])
torch.save(ck, '$S/gate0/$name.pt')
"
  echo "start $name $(date -Is)"
  python -B scripts/run_segment_memory_model.py --config $cfg --run-id g0_$name --output $S/gate0/$name.csv $common "$@" \
    --eval-only --resume-checkpoint $S/gate0/$name.pt > $S/gate0/$name.log 2>&1
  echo "end $name exit=$? $(date -Is)"
  rm -f $S/gate0/$name.pt
}
base=pg19s20m_h256_m64_cbn32_e20
par=pg19s20m_h256_par_e20
score base_attn0 $R/${base}_${base}.best.pt attn --carry-bound norm --carry-scale 32
score base_mlp0 $R/${base}_${base}.best.pt mlp --carry-bound norm --carry-scale 32
score par_attn0 $R/${par}_${par}.best.pt attn --carry-bound gru --memory-state gru
score par_mlp0 $R/${par}_${par}.best.pt mlp --carry-bound gru --memory-state gru
$SLOTS done @imm
echo "slot ended $(date -Is)"
