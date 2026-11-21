#!/bin/sh
# Typo noise (the user's idea via @mobius; typo'd held-out text too, the user,
# 2026-11-10). From the typo-noise worktree. One job at a time.
#   A. the 3.1316 model (no training noise) scored on each typo'd held-out text,
#      on the typo'd text itself and with clean targets (builds each text's
#      prior cache the first time);
#   B. the same setup trained with --input-noise 0.01, 0.02, 0.05;
#   C. those models scored as in A.
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
cd $S/imm-typo || exit 1
R=runs
CLEAN=data/.splits/4bfd5a43d446644a/heldout_corpus.txt
model="--prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --hidden-size 256 --embedding-size 256 --prior-input both --segment-rule word --max-memory 16"
base=gutenberg5m_crossfit20_ps100_rs100_l2005_s64_lr1e-3_h256_e12_both_mobius16_segword_m16
mkdir -p $R/typo_evals

score() {  # score <model tag> <checkpoint>
  for q in 1 2 5; do
    for how in typo clean; do
      id=typo_eval_${1}_heldout${q}_${how}targets
      extra=""
      [ $how = clean ] && extra="--eval-clean-targets $CLEAN"
      echo "start $id $(date -Is)"
      python scripts/run_segment_memory_model.py --config configs/gutenberg5m_crossfit20_ps100_e1_heldout_typo${q}_seed1.json \
        --run-id $id --output $R/typo_evals/$id.csv --eval-only --resume-checkpoint $2 $model \
        --prior-evidence-rows /home/trans/air/mobius/runs/imm/gutenberg5m_kn_typo$q $extra > $R/typo_evals/$id.log 2>&1
      echo "end $id exit=$? $(date -Is)"
    done
  done
}

score noise0 $R/${base}_$base.best.pt
for q in 1 2 5; do
  id=${base}_noise$q
  echo "start $id $(date -Is)"
  python scripts/run_segment_memory_model.py --config configs/gutenberg5m_crossfit20_ps100_e1.json \
    --run-id $id --output $R/$id.csv --epochs 12 $model --input-noise 0.0$q \
    --prior-evidence-rows /home/trans/air/mobius/runs/imm/gutenberg5m_kn > $R/$id.log 2>&1
  echo "end $id exit=$? $(date -Is)"
  score noise$q $R/${id}_$id.best.pt
done
