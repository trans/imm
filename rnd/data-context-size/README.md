---
title:    "Where the remaining loss is: data, context or size"
kind:     experiment
status:   concluded
outcome:  mixed
question: >-
  After the 128-record model (2.7619), where is the remaining loss, and
  which limits it at 20M characters: the training text, the memory's
  reach, or the model's width?
answer: >-
  62% of the loss is a word's first two letters, choosing the next word;
  from the third letter on the prior has nearly settled the spelling. Data
  binds: half the training text costs +2.46% (2.8299), more than imm's
  whole lead over the transformer, and most at word ends (+4.19%).
  256 records cost +0.22% (2.7679) and h256 +0.79% (2.7836, it overfits).
  The lever is more text, not width or reach.
opened:   2026-11-18
updated:  2026-11-18
code: >-
  543ef84 on branch topic-tiers (kept, not merged into main), which adds
  --train-prefix-chars. The bottleneck analysis read the slot 84 runs'
  per-character losses (c732057).
data: >-
  PG-19 20M split, data/pg19/imm-split-20m, scored by piece (249,999
  held-out targets). Mobius's pg19split20m_gated_d24_pg10g prior (alone
  3.068) with pg19split20m_kn evidence. The half arm trains on the first
  10,000,000 training characters.
slots:    [93]
tags:     [bottleneck, data, context, size, pg19]
related:  [memory-window, topic-tiers, imm-1.85m, memory-knockouts]
---

# Where the remaining loss is: data, context or size

## Question

The user asked (2026-11-18) where the real bottleneck is. Two parts: where
in the text the remaining loss sits, and which single change to the
128-record model (`memory-window`, 2.7619) moves it.

## Design

- **Where the loss is (CPU):** `bottleneck.py` pairs per-target losses of
  the 128-record model (and R1) against the d24 transformer and the d24
  prior alone, by the letter's place in its word and by whether the word
  was seen before in the piece, inside or beyond the transformer's
  256-character window.
- **Slot 93:** three changes to the 128-record model, one at a time, same
  seed and schedule:
  - half: `--train-prefix-chars 10000000`, 82 epochs (the same number of
    updates). The flag trains on the text's first N characters and keeps
    the whole text and its prior rows loaded (`--max-train-chars` can't,
    because the rows' digests refuse a shortened text);
  - m256: `--max-memory 256`;
  - h256: `--hidden-size 256 --embedding-size 256`.

## How to rerun

`bn_slot.sh 93` here, from a worktree at `543ef84`; proposed tag
`rnd/data-context-size/slot93`. Its comment quoted the user; it was
paraphrased. Every command is as it ran; paths are the original machine's.
`bottleneck.py IMM.npz TX.npz PRIOR_ROWS` gives the place-in-word table;
the CPU analysis ran it on the slot 84 runs (proposed tag
`rnd/data-context-size/bottleneck`, on `c732057`).

Results here: `results/mm128_bottleneck.txt` (the 128-record model and R1
against the transformer), `results/{half,m256,h256}_vs_mm128.txt` (each arm
paired against the 128-record model, `scripts/rnd/rpair.sh`) and
`results/{half,m256,h256}_bottleneck.txt`.

## Results

### 2026-11-18: where the remaining loss is (CPU)

| targets | share of m128's loss | m128 | transformer | prior alone | m128 against the transformer |
|---|---:|---:|---:|---:|---:|
| a word's first letter (which word comes next) | 43.3% | 11.99 | 12.37 | 13.75 | −3.11% ±0.20 |
| second letter | 18.3% | 2.99 | 3.05 | 3.15 | −1.79% ±0.16 |
| third letter | 12.8% | 2.55 | 2.58 | 2.65 | −1.19% ±0.18 |
| fourth letter on | 12.1% | 1.55 | 1.56 | 1.58 | −0.64% ±0.13 |
| not in a word (spaces, punctuation) | 13.4% | 1.79 | 1.84 | 2.34 | −2.62% ±0.21 |

- 62% of the loss is the first two letters of a word. On first letters both
  networks take only about 13% off the prior (13.75 → 11.99).
- Copying rare spellings is where imm trails: on third-and-later letters of
  a repeated word the prior can't spell (≥ 3 nats), the 128-record model is
  behind the transformer by +10.3% ±6.3 within its window and +14.1% ±4.8
  at 256–1,000 characters. A word record is a lossy copy of its letters.
  But those letters are only about 2% of the loss.

### 2026-11-18: one change at a time (slot 93)

| arm | change | held out | vs m128 | train NLL (m128 0.9747) | vs the transformer |
|---|---|---:|---:|---:|---:|
| half | `--train-prefix-chars 10000000`, 82 epochs | 2.8299 | **+2.46%** | 0.9565 | +0.60% |
| m256 | `--max-memory 256` | 2.7679 | +0.22% | 0.9763 | −1.60% |
| h256 | `--hidden-size 256 --embedding-size 256` | 2.7836 | +0.79% | 0.9643 | −1.05% |

- **Data binds.** Half the text costs 2.46%, more than all of imm's lead
  over the transformer (−1.82%). Its training loss is lower: it overfits.
- **Size doesn't pay at 20M.** h256 fits training better (−0.010 nats) and
  held-out worse. The same dropout (0.2) was used, untuned.
- **Context past 128 words doesn't pay.** m256 is +0.22%, mostly inside
  words.
- **Data moves next-word choice most.** Half against m128: +4.19% ±0.18 at
  a word's last letter (predicting the next word's first), +10.7% at
  sentence ends, +14.1% on punctuation, +1.87% inside words. h256's loss is
  in the same places, smaller.

## What we learned

- At 20M characters the 62% of the loss that is choosing the next word is
  data-limited. This pointed the work to the larger PG-19 splits (100M,
  from slot 101; other entries).
- At this size neither a wider model nor a longer memory helps; a wider
  model needs its regularisation retuned before that can be read as a
  size limit.
