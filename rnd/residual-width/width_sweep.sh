#!/bin/sh
# Waits for the hidden-64 100-epoch run (PID 1892980) to exit, then runs the
# width sweep on Gutenberg one at a time: hidden/embedding 128, then 256.
cd /home/trans/air/imm || exit 1
while kill -0 1892980 2>/dev/null; do sleep 30; done
for h in 128 256; do
  id=gutenberg5m_crossfit20_ps100_rs100_l2005_s64_lr3e-3_h${h}_e40
  echo "start $id $(date -Is)"
  python scripts/run_segment_memory_model.py --config configs/gutenberg5m_crossfit20_ps100_e1.json \
    --run-id $id --output runs/$id.csv \
    --prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 3e-3 --epochs 40 \
    --hidden-size $h --embedding-size $h > runs/$id.log 2>&1
  echo "end $id exit=$? $(date -Is)"
done
