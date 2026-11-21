#!/bin/sh
# Re-score the reach-64 run's epoch-5 weights (held-out 129.17 during training,
# training loss smooth) with the run's own arguments. DEVICE=cpu or cuda.
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
cd $S/imm-dropout || exit 1
common="--prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows /home/trans/air/mobius/runs/imm/pg19split_kn --segment-rule word --max-memory 64 --segments-per-update 64 --dropout 0.2 --eval-streams 64 --drop-evidence-cache --lr-schedule cosine --min-lr 1e-4 --hidden-size 256 --embedding-size 256 --epochs 80"
id=spike_epoch5_eval_$1
python scripts/run_segment_memory_model.py --config configs/pg19split_crossfit20_ps100_e1.json --run-id $id --output $S/$id.csv $common \
  --prior-rows /home/trans/air/mobius/runs/imm/pg19split_gated_d16_pg200m --prior-residual-state-features reliability,depth_norm \
  --resume-checkpoint $S/spike_epoch5.pt --eval-only 2>&1 | grep -v '^epoch=.*profile' | tail -5
