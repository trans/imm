#!/bin/sh
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
cd $S/imm-dropout || exit 1
common="--prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows /home/trans/air/mobius/runs/imm/pg19split_kn --segment-rule word --max-memory 64 --segments-per-update 64 --dropout 0.2 --eval-streams 64 --drop-evidence-cache --lr-schedule cosine --min-lr 1e-4 --hidden-size 256 --embedding-size 256 --epochs 80"
id=spike_epoch5_trace_$1
python $S/spike_trace.py $S/imm-dropout $S/$id.npz --config configs/pg19split_crossfit20_ps100_e1.json --run-id $id --output $S/$id.csv $common \
  --prior-rows /home/trans/air/mobius/runs/imm/pg19split_gated_d16_pg200m --prior-residual-state-features reliability,depth_norm \
  --resume-checkpoint $2 --eval-only 2>&1 | grep -E 'val_ppl|recorded|Error|Traceback' 
