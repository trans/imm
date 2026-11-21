#!/bin/sh
# Build imm's cross-fitted prior cache on the PG-19 split and score the
# prior alone (epoch 0, then 1 epoch of h256 to see the pipeline runs end to
# end on 96 characters). Branch pg19-split d65044d.
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
cd $S/imm-dropout || exit 1
id=pg19split_prior_check_h256_e1
python scripts/run_segment_memory_model.py --config configs/pg19split_crossfit20_ps100_e1.json --run-id $id \
  --output runs/$id.csv --epochs 1 --hidden-size 256 --embedding-size 256 \
  --prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --prior-input log-probs \
  --segment-rule word --max-memory 16 --dropout 0.2 --eval-streams 64 > runs/$id.log 2>&1
echo "exit=$?"
