#!/bin/sh
# Perfect repair (the user's go-ahead, 2026-11-10): each typo queue model scored
# with the prior, evidence, segments and targets on the clean held-out text and
# only the residual reading the typo'd copy. From the typo-perfect-repair worktree.
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
cd $S/imm-repair || exit 1
R=runs
model="--prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --hidden-size 256 --embedding-size 256 --prior-input both --segment-rule word --max-memory 16"
base=gutenberg5m_crossfit20_ps100_rs100_l2005_s64_lr1e-3_h256_e12_both_mobius16_segword_m16
for n in 0 1 2 5; do
  if [ $n = 0 ]; then ck=$R/${base}_$base.best.pt; else ck=$R/${base}_noise${n}_${base}_noise$n.best.pt; fi
  for q in 1 2 5; do
    id=typo_eval_noise${n}_heldout${q}_perfectrepair
    if [ -s $R/typo_evals/${id}_$id.csv ]; then echo "done already $id"; continue; fi
    echo "start $id $(date -Is)"
    python scripts/run_segment_memory_model.py --config configs/gutenberg5m_crossfit20_ps100_e1.json \
      --run-id $id --output $R/typo_evals/$id.csv --eval-only --resume-checkpoint $ck $model \
      --prior-evidence-rows /home/trans/air/mobius/runs/imm/gutenberg5m_kn \
      --eval-residual-input data/typos/gutenberg5m_heldout_typo${q}_seed1.txt > $R/typo_evals/$id.log 2>&1
    echo "end $id exit=$? $(date -Is)"
  done
done
