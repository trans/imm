#!/bin/sh
# The rest of typo_queue.sh after the low-memory stop on
# 2026-11-10, then the perfect-repair scoring. Skips what already finished,
# so it can be rerun. Waits for the GPU to be idle first. One job at a time.
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
cd $S/imm-typo || exit 1
R=runs
CLEAN=data/.splits/4bfd5a43d446644a/heldout_corpus.txt
model="--prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --hidden-size 256 --embedding-size 256 --prior-input both --segment-rule word --max-memory 16"
base=gutenberg5m_crossfit20_ps100_rs100_l2005_s64_lr1e-3_h256_e12_both_mobius16_segword_m16

while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 60; done
echo "gpu idle $(date -Is)"

score() {  # score <model tag> <checkpoint>
  for q in 1 2 5; do
    for how in typo clean; do
      id=typo_eval_${1}_heldout${q}_${how}targets
      if [ -s $R/typo_evals/${id}_$id.csv ]; then echo "done already $id"; continue; fi
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

score noise2 $R/${base}_noise2_${base}_noise2.best.pt
id=${base}_noise5
if grep -q '^12,' $R/${id}_$id.csv 2>/dev/null; then echo "done already $id"; else
  echo "start $id $(date -Is)"
  python scripts/run_segment_memory_model.py --config configs/gutenberg5m_crossfit20_ps100_e1.json \
    --run-id $id --output $R/$id.csv --epochs 12 $model --input-noise 0.05 \
    --prior-evidence-rows /home/trans/air/mobius/runs/imm/gutenberg5m_kn > $R/$id.log 2>&1
  echo "end $id exit=$? $(date -Is)"
fi
score noise5 $R/${id}_$id.best.pt
sh $S/typo_repair_evals.sh
