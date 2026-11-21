#!/bin/sh
# Fit imm's gated count prior (depth 8, entropy_delta + suffix_stats) on
# mobius's PG-19 split (data/pg19/imm-split, mobius cbb6d4c). Valid ratio
# 10/190: the gate's fitting text is the last ten 25K pieces, starting at a
# piece boundary (mobius's suggestion). Same settings as the Gutenberg theta
# (runs/gutenberg_count_gate_d8_entropy_suffix.json) otherwise.
S=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
cd $S/imm-dropout || exit 1
D=data/pg19/imm-split
python -m imm.count_gate --train $D/train_corpus.txt --heldout $D/heldout_corpus.txt --vocab-file $D/vocab.txt \
  --depth 8 --valid-ratio 0.05263157894736842 --epochs 100 --lr 0.05 --max-fit-positions 50000 --seed 1337 \
  --extra-features entropy_delta,suffix_stats --output runs/pg19split_count_gate_d8_entropy_suffix.json
echo "exit=$?"
