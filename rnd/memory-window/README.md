---
title:    How many word records the attention reads
kind:     experiment
status:   concluded
outcome:  positive
question: >-
  imm's attention reads the last 64 word records. Does a longer or shorter
  window change the score, with nothing else changed and no parameters
  added?
answer: >-
  128 records helps once the model has the d24 prior and the word tier:
  2.7619 against 2.7738 (−0.43% ±0.06), the gain all past a piece's first
  256 characters, so it is reach. 32 records costs +0.12% ±0.05, and 256
  records +0.22% against 128. On the earlier d16 parallel model, 128
  records gave only 0.09% (3.1491 against 3.1518).
opened:   2026-11-14
updated:  2026-11-18
code: >-
  slot 31 at 787ee25 (branch parallel-memory, deleted; the commit is on
  main); slots 84 and 93 at c732057 and 543ef84 (branch topic-tiers, kept,
  not merged into main).
data: >-
  PG-19 20M split, data/pg19/imm-split-20m (250,000 held-out characters).
  Slot 31: mobius's pg19split20m_gated_d16_pg200m prior (alone 1.8205
  bits/char). Slots 84 and 93: pg19split20m_gated_d24_pg10g (alone 3.068),
  scored by piece. KN evidence pg19split20m_kn throughout.
slots:    [31, 84, 93]
tags:     [memory, attention, context, reach, pg19]
related:  [parallel-memory, topic-tiers, data-context-size, imm-1.85m, blocks-20m]
---

# How many word records the attention reads

## Question

`--max-memory` sets how many past word records each character's attention
can see. At about 5.6 characters a word, 64 records reach about 360
characters. Is the window what limits imm? The user asked (2026-11-18) to
try 128 records, and 32.

## Design

- **Slot 31:** the 20M parallel model (`parallel-memory`: 1 × h256, d16
  prior, 20 epochs) with `--max-memory 128`. Before it, `par_wide_check.py`
  (in `blocks-20m`) checked that the parallel path equals the loop with a
  window wider than an update.
- **Slot 84:** R1 of `topic-tiers` (slot 63's model trained by piece:
  h192, two blocks, word tier 168, learned prior strength, d24 rows, 41
  epochs, 2.7738) with `--max-memory 32` and 128. Nothing else changed.
- **Slot 93:** the 128-record model with 256 records (one of three arms in
  `data-context-size`).

## How to rerun

| run | script | commit | slot | proposed tag |
|---|---|---|---|---|
| 128 records, d16 parallel model | `../blocks-20m/par_push_slot.sh 31` (first run) | `787ee25` | 31 | `rnd/memory-window/slot31` |
| 32 and 128 records | `mem_slot.sh 84 "m32 m128"` (here) | `c732057` | 84 | `rnd/memory-window/slot84` |
| 256 records | `../data-context-size/bn_slot.sh 93` (the m256 arm) | `543ef84` | 93 | `rnd/data-context-size/slot93` |

`mem_slot.sh` is `topic-tiers`'s `topic_slot.sh` with the `m32` and `m128`
arms added. Comments that quoted the user were paraphrased; every command
is as it ran. Paths are the original machine's.

Paired comparisons here (`scripts/rnd/rpair.sh`; "imm" is A, "tx" is B):
`results/mm32_vs_r1.txt`, `results/mm128_vs_r1.txt`,
`results/mm128_vs_s2.txt`.

## Results

### 2026-11-14: 128 records on the d16 parallel model (slot 31)

| | parallel, 64 records | 128 records |
|---|---:|---:|
| held-out, best (epoch 20) | 3.1518 | 3.1491 |
| epoch 1 / 4 / 8 / 12 / 16 | 3.3404 / 3.2326 / 3.1889 / 3.1681 / 3.1572 | 3.3375 / 3.2310 / 3.1882 / 3.1654 / 3.1535 |
| wall time, 20 epochs | 2,779 s | 3,283 s |

0.09%, ahead at every epoch by 0.001–0.004 but within any seed spread, for
18% more wall time. Then, the window wasn't what limited the model.

### 2026-11-18: 32 and 128 records on the d24 word-tier model (slot 84)

| records | held-out, by piece | change against R1 | epochs 10 / 20 / 30 / 41 | train NLL | wall time |
|---:|---:|---:|---|---:|---:|
| 32 | 2.7771 | +0.12% ±0.05 | 2.801 / 2.785 / 2.779 / 2.778 | 0.9788 | 3,781 s |
| 64 (R1) | 2.7738 | | 2.805 / 2.784 / 2.778 / 2.774 | 0.9786 | 3,853 s |
| 128 | **2.7619** | **−0.43% ±0.06** | 2.796 / 2.771 / 2.768 / 2.762 | 0.9747 | 4,192 s |

- 128 records was the largest gain since the word tier and the attention
  themselves: −0.42% against S2, −1.82% against the d24 transformer's
  2.8131 (R1: −1.40%).
- It isn't overfitting: train and held-out fall together (gap 0.0414
  against 0.0416).
- Where the window can't matter, it doesn't: in a piece's first 256
  characters (about 45 records) +0.00% ±0.26; from 256 to 1,000 −0.48%;
  past 2,500 −0.38%. So the gain is reach.
- Repeated words gain most (−0.73% against −0.21% on new words): 1–16 words
  back −1.31% ±0.18, 16–64 −0.74% ±0.18, 64–256 −0.45% ±0.17, 256+ −0.25%
  ±0.21. Near repeats gain most even though both windows reach them; one
  untested reading is that the longer window also holds the earlier copies
  of a word repeated close by.
- Capitals and sentence ends gain most: uppercase −2.94% ±0.58, capitalised
  repeats −3.66% ±0.81, sentence ends −2.59% ±0.97.
- 32 records costs about a quarter of what 128 gains, on repeats and
  capitals.

### 2026-11-18: 256 records (slot 93)

2.7679, +0.22% against 128 records, mostly inside words (+0.26%); word
ends level (+0.19% ±0.14). Details in `data-context-size`.

## What we learned

- With the d24 prior and the word tier, 128 records is better than 64, and
  256 no better than 128. The 128-record model (2.7619) became the base
  for `data-context-size`.
- The memory's gain is reach to repeats and names.
