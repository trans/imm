#!/bin/sh
# After the h128 mix_units run, two checks imm-06 asked for (2026-11-10)
# on a request later found to be meant for @agpt (imm-06, 2026-11-10); kept as a seed check:
#   1. the h256 dropout best checkpoint (3.0217, scored batched) rescored one-stream;
#   2. a second seed of h256 dropout 0.2, 40 epochs (--seed 1; the first used 1337).
# From the imm-dropout worktree, now prior-rows-loader f210ae7 (the first seed
# ran on batched-eval 74188ec; f210ae7 only adds --prior-rows and named features).
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
cd $S/imm-dropout || exit 1
while pgrep -f "python scripts/run_segment_memory_model.py.*do02_mixunits" > /dev/null; do sleep 10; done
common="--prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 1e-3 --prior-input both --prior-evidence-rows /home/trans/air/mobius/runs/imm/gutenberg5m_kn --segment-rule word --max-memory 16 --dropout 0.2 --drop-evidence-cache"
old=gutenberg5m_crossfit20_ps100_rs100_l2005_s64_lr1e-3_h256_e40_both_mobius16_segword_m16_do02
mkdir -p runs/eval_checks
tag=h256_do02_best_s1
echo "start eval_check_$tag $(date -Is)"
python scripts/run_segment_memory_model.py --config configs/gutenberg5m_crossfit20_ps100_e1.json --run-id eval_check_$tag \
  --output runs/eval_checks/eval_check_$tag.csv --eval-only --resume-checkpoint runs/${old}_$old.best.pt \
  --hidden-size 256 --embedding-size 256 $common --eval-streams 1 > runs/eval_checks/eval_check_$tag.log 2>&1
echo "end eval_check_$tag exit=$? $(date -Is) nll,ppl=$(tail -1 runs/eval_checks/eval_check_${tag}_eval_check_$tag.csv | cut -d, -f4,5)"
id=${old}_seed1
echo "start $id $(date -Is)"
python scripts/run_segment_memory_model.py --config configs/gutenberg5m_crossfit20_ps100_e1.json \
  --run-id $id --output runs/$id.csv --epochs 40 --hidden-size 256 --embedding-size 256 $common --eval-streams 64 \
  --seed 1 > runs/$id.log 2>&1
echo "end $id exit=$? $(date -Is)"
