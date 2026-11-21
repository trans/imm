#!/bin/sh
# Waits for the hidden-256 lr 1e-3 Gutenberg run (PID 1976535), then runs the
# segment-rule grid on Tiny canonical from the segment-rules worktree, one
# at a time. Baseline settings: 16 streams, lr 3e-3, scale 1.0, L2 0.05,
# 10 epochs, hidden 32 (tiny_..._s16_lr3e-3_e10, 3.6769).
cd /home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad/imm-segments || exit 1
while kill -0 1976535 2>/dev/null; do sleep 30; done
for spec in "unique 8" "word 8" "word 16" "hybrid 8"; do
  set -- $spec
  id=tiny_canonical_crossfit20_ps100_rs100_l2005_s16_lr3e-3_e10_seg$1_m$2
  echo "start $id $(date -Is)"
  python scripts/run_segment_memory_model.py --config configs/tiny_canonical_crossfit20_ps100_e1.json \
    --run-id $id --output runs/$id.csv --epochs 10 --lr 3e-3 --prior-residual-scale 1.0 --train-streams 16 \
    --segment-rule $1 --max-memory $2 > runs/$id.log 2>&1
  echo "end $id exit=$? $(date -Is)"
done
