---
title:    A word-level GRU tier, and its own step
kind:     experiment
status:   concluded
outcome:  positive
question: >-
  Does a second GRU that steps once a word, over the character GRU's state
  at each word's end, help the two-block parallel model; and does letting a
  word's last character see the word state its own word makes help more?
answer: >-
  Yes, both. The word tier took the two-block model from 3.1360 to 3.1274
  (−0.27%) at 400M characters, beating a third block (3.1351), and 3.1084
  at 820M. Own step took it to 3.1155 at 400M (−0.38%, most of it before a
  new phrase, −4.4%, or sentence, −8.5%) and 3.1011 at 820M, 0.08% behind
  the 1.86M transformer with the prior in its input (3.0987), with 1.7x its
  parameters.
opened:   2026-11-14
updated:  2026-11-15
code: >-
  --word-tier at df1e3c9 (branch word-tier; slot 37 ran from main at
  df1e3c9); --tier-own-step at 55b8128 (branch tier-own-step). Branches
  deleted; the commits are on main.
data: >-
  PG-19 20M split, data/pg19/imm-split-20m (250,000 held-out characters).
  Mobius's pg19split20m_gated_d16_pg200m prior (alone 1.8205 bits/char)
  with compact pg19split20m_kn depth evidence.
slots:    [36, 37, 50, 52]
tags:     [hierarchy, word-tier, gru, pg19, transformer-comparison]
related:  [blocks-20m, parallel-memory, phrase-tier, memory-knockouts, learned-prior-strength]
---

# A word-level GRU tier, and its own step

## Question

The user's idea was a binary tree of GRUs, each tier at half the clock of
the one below. The plan's step 4 (the user approved it, 2026-11-14) was its
first tier: a GRU that steps once a word. Then, after the phrase tier
showed that a tier's state arrives one segment late (`phrase-tier`), would
giving a segment's last character its own segment's tier state help?

## Design

- **`--word-tier WIDTH`** (`df1e3c9`): a second GRU steps once a segment,
  over the character GRU's state at each segment's last character, its
  state carried from update to update. Every character of segment k gets
  the word GRU's state after segment k − 1, projected to the hidden size,
  gated by tanh(gate) (from 0), and added to its GRU state before the
  attention blocks. Parallel memory path only. A planted leak fails the
  tests.
- **`--tier-own-step`** (`55b8128`): a segment's last character (the one
  that predicts the next word's first) gets the word state after its own
  segment, and the last character of a phrase-ending segment the phrase
  state after that phrase. Causal: that state reads the character GRU's
  state at that very character. No new parameters.
- **Base:** the two-block parallel model of `blocks-20m` (3.1360), 20
  epochs (400M characters) or 41 (820M, the transformer's budget), cosine
  over the epochs, seed 1337. Word tier width 256.
- Paired per-character comparisons by place in the word, phrase and
  sentence (`paired_buckets2.py`).

## How to rerun

Scripts here, as they ran; paths are the original machine's. Comments that
quoted the user were paraphrased; every command is as it ran.

| run | script | commit | slot | proposed tag |
|---|---|---|---|---|
| word tier, 20 epochs | `word_slot.sh 36` | `df1e3c9` | 36 | `rnd/word-tier/slot36` |
| word tier, 41 epochs | `word41_slot.sh 37` | `df1e3c9` | 37 | `rnd/word-tier/slot37` |
| own step, word and word + phrase, 20 epochs | `own_slot.sh 50` | `55b8128` | 50 | `rnd/word-tier/slot50` |
| own step, word, 41 epochs | `own41_slot.sh 52` | `55b8128` | 52 | `rnd/word-tier/slot52` |

The breakdowns came from `paired_buckets2.py` on the runs'
`--char-loss-output` files. Results here: `results/own_positions_word.txt`
(own step against none, 400M), `results/own41_paired.txt` and
`results/own41_positions.txt` (the same at 820M). The phrase arm of slot 50
is in `phrase-tier`.

## Results

### 2026-11-14: the word tier at 400M characters (slot 36)

| | two blocks | three blocks | **two blocks + word tier** |
|---|---:|---:|---:|
| trained parameters | 2.74M | 3.53M | 3.20M |
| held-out, best (epoch 20) | 3.1360 | 3.1351 | **3.1274** |
| epoch 1 / 4 / 8 / 12 / 16 | 3.3313 / 3.2246 / 3.1757 / 3.1574 / 3.1417 | 3.3315 / 3.2167 / 3.1699 / 3.1519 / 3.1398 | 3.3280 / 3.2220 / 3.1739 / 3.1523 / 3.1349 |
| training NLL, epoch 20 | 1.0956 | 1.0885 | 1.0900 |
| first character / later | 3.425 / 3.078 | 3.414 / 3.079 | 3.408 / 3.071 |
| take-away: own, prior in input / none | 59.60 / 95.21 | 57.41 / 95.11 | 58.40 / 100.32 |
| wall time, 20 epochs | 3,202 s | 3,940 s | **2,256 s** |

- The word tier is worth 0.27%, against the third block's 0.03%, with
  0.46M parameters against its 0.79M. One seed; the GPU's run-to-run
  spread is up to 0.15%.
- It helps most at the first position that sees the new word state.
- Wall time fell 30% against two blocks alone, from the new row reader
  (`b38374b`), not the tier.

### 2026-11-14: at the transformer's budget, 820M characters (slot 37)

| at 819–820M characters | params | held-out |
|---|---:|---:|
| transformer 4 × 192, books prior in the input | 1.86M | **3.0987** |
| **imm 2 × h256 + word tier 256** | 3.20M | **3.1084** (1.6362 bits/char) |
| transformer 4 × 96, books prior in the input | 487k | 3.1282 |

- 0.31% behind the 1.86M transformer, with 1.7x its parameters. Doubling
  the training gained imm 0.61%, the transformer 1.9%. The last 11 epochs
  gained 0.17%; 41 passes over 20M begin to overfit.
- 4,635 s of training.

### 2026-11-15: own step at 400M (slot 50)

| | word tier | **word, own step** | word + phrase | word + phrase, own step |
|---|---:|---:|---:|---:|
| held-out, best (epoch 20) | 3.1274 | **3.1155** | 3.1326 | 3.1140 |
| epoch 4 / 8 / 12 / 16 / 19 | 3.222 / 3.174 / 3.152 / 3.135 / 3.128 | 3.201 / 3.155 / 3.134 / 3.121 / 3.117 | 3.217 / 3.168 / 3.148 / 3.137 / 3.133 | 3.202 / 3.160 / 3.131 / 3.119 / 3.115 |
| tanh of the tier gates | 0.076 | 0.085 | 0.066, 0.039 | 0.074, 0.036 |
| wall time, 20 epochs | 2,256 s | 2,285 s | 2,449 s | 2,391 s |

Paired change in perplexity, ± the standard error over held-out characters:

| bucket | chars | word: own vs not | word own vs two blocks |
|---|---:|---:|---:|
| all | 249,999 | **−0.38 ±0.06** | −0.65 ±0.06 |
| at a word's first character | 43,854 | +0.36 ±0.13 | −0.13 ±0.13 |
| inside a word | 162,344 | −0.43 ±0.08 | −0.57 ±0.08 |
| at a word's last (predicting the next word's first) | 43,854 | −0.96 ±0.17 | −1.56 ±0.17 |
| the same, the word not ending a phrase | 37,022 | −0.31 ±0.15 | −0.81 ±0.15 |
| a phrase's last (predicting the next phrase's first) | 6,832 | **−4.43 ±0.69** | −5.57 ±0.69 |
| a sentence's last (predicting the next sentence's first) | 2,998 | **−8.52 ±1.31** | −11.01 ±1.30 |
| first word of a sentence, not last | 13,427 | −0.79 ±0.44 | −0.01 ±0.45 |

- Own step works where it was meant to: the gain lands on the position
  that predicts the next unit's first character, and grows with the unit.
  The breakdown sits where the mechanism predicts, so the gain isn't noise.
- The sentence-end gain comes from the word tier alone: the word just
  finished carries the mark.
- It also helps inside words (−0.43%) and costs +0.36% at a word's first
  character: effects of training, since nothing changes there mechanically.
- imm at 400M was then 1.3% ahead of the 1.86M transformer at the same
  budget (3.1155 against 3.1578).

### 2026-11-15: own step at 820M (slot 52)

| at 820M characters | params | held-out |
|---|---:|---:|
| 1.86M transformer, prior in the input | 1.86M | **3.0987** |
| **imm, 2 × h256 + word tier, own step** | 3.20M | **3.1011** |
| imm, 2 × h256 + word tier (slot 37) | 3.20M | 3.1084 |
| 487k transformer, prior in the input | 487k | 3.1282 |

- 0.08% behind the 1.86M transformer, within the run-to-run spread, with
  1.7x its parameters. Best at epoch 38.
- Doubling the training gained 0.46% with own step, against 1.9% for the
  transformer: imm saturates on the 20M split, the transformer doesn't yet.
- Own step's gain shrinks with training: −0.24% ±0.07 at 820M against
  −0.38% at 400M. Before a word that doesn't end a phrase, −0.10% ±0.17
  (was −0.31%). Before a new phrase −5.15% ±0.75, before a new sentence
  −10.05% ±1.43: those stay or grow.
- 4,648 s.

## What we learned

- One GRU tier at word rate is worth more than another attention block, and
  own step is worth keeping. Both stayed in imm's model (`imm-1.85m`).
- Retrained without it later, the word tier was worth 0.57% at 1.85M
  parameters (`memory-knockouts`), close to these figures.
- Its gain is at boundaries: before new phrases and sentences.
