#!/bin/sh
# 1. Seeds 1 and 2 of the segment grid's unique-8 and word-16 (Tiny), from the
#    segment-rules worktree. 2. The three depth-evidence runs (Gutenberg,
#    hidden 256, lr 1e-3, 12 epochs, --prior-input both), from the
#    external-evidence worktree. One at a time.
cd /home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad/imm-segments || exit 1
for seed in 1 2; do
  for spec in "unique 8" "word 16"; do
    set -- $spec
    id=tiny_canonical_crossfit20_ps100_rs100_l2005_s16_lr3e-3_e10_seg$1_m$2_seed$seed
    echo "start $id $(date -Is)"
    python scripts/run_segment_memory_model.py --config configs/tiny_canonical_crossfit20_ps100_e1.json \
      --run-id $id --output runs/$id.csv --epochs 10 --lr 3e-3 --prior-residual-scale 1.0 --train-streams 16 \
      --segment-rule $1 --max-memory $2 --seed $seed > runs/$id.log 2>&1
    echo "end $id exit=$? $(date -Is)"
  done
done
cd /home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad/imm-evidence || exit 1
for spec in "own8 -" "mobius8 runs/mobius_gutenberg5m_kn_d8" "mobius16 /home/trans/air/mobius/runs/imm/gutenberg5m_kn"; do
  set -- $spec
  id=gutenberg5m_crossfit20_ps100_rs100_l2005_s64_lr1e-3_h256_e12_both_$1
  extra=""
  [ "$2" != "-" ] && extra="--prior-evidence-rows $2"
  echo "start $id $(date -Is)"
  python scripts/run_segment_memory_model.py --config configs/gutenberg5m_crossfit20_ps100_e1.json \
    --run-id $id --output runs/$id.csv --prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 \
    --lr 1e-3 --epochs 12 --hidden-size 256 --embedding-size 256 --prior-input both $extra > runs/$id.log 2>&1
  echo "end $id exit=$? $(date -Is)"
done
