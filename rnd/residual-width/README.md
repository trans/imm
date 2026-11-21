---
title:    Residual width, size and dropout on Gutenberg 5M
kind:     experiment
status:   concluded
outcome:  positive
question: >-
  With batched streams, does a wider residual help; how small can imm be
  and still beat the transformer of its size; and is the wide residual
  short of capacity or unregularised?
answer: >-
  Width helps but overfits 5M characters: h256 at lr 1e-3 scored 3.2711
  (epoch 10) against h64's 3.3078 after 100 epochs; lr 3e-3 diverged at
  h256. With word segments and mobius's depth 16, h64 (178,833 trained
  parameters) scored 3.1606, 3.3% better than the 474k transformer.
  Dropout was the missing piece: h128 went from 3.1464 to 3.0671 (50
  epochs), and h256 to 3.0217 (seed 1337) and 3.0395 (seed 1), against
  3.0542 for @air's 1.83M dropout transformer.
opened:   2026-11-09
updated:  2026-11-10
code: >-
  width 50e3b79 (branch residual-capacity); sizes 37321f1 (branch
  word-segments-depth-evidence); h128 dropout 95f6645 (branch
  residual-dropout); h256 dropout 74188ec (branch batched-eval); its second
  seed f210ae7 (branch prior-rows-loader). The branches are deleted; every
  commit is on main.
data: >-
  Gutenberg 5M split 4bfd5a43d446644a. imm's depth-8 gated prior,
  cross-fitted, strength 1.0 (prior alone 3.5716). The width sweep has no
  prior input; the size and dropout runs add --prior-input both with
  mobius's KN depth-16 evidence (gutenberg5m_kn).
slots:    none (before the slot system)
tags:     [width, size, dropout, overfitting, gutenberg]
related:  [batched-streams, depth-evidence-input, word-segments, transformer-baseline-gutenberg5m, mobius-prior-rows]
---

# Residual width, size and dropout on Gutenberg 5M

## Question

At hidden 64, 64 streams, imm flattened at about 3.31 from epoch 30
(`batched-streams`). Three questions followed:

1. Does a wider residual help?
2. The user's question (2026-11-09, via @air): imm is judged by size. Can a
   smaller residual plus the prior match the 474k transformer?
3. The wide residuals overfit. Are they short of capacity, or
   unregularised? The transformer had dropout 0.2 and weight decay 0.1;
   imm had no dropout and weight decay 0.01.

## Design

All at 64 streams, residual scale 1.0, L2 0.05, seed 1337 unless stated.
Hidden and embedding sizes are set together. Trained parameters are the sum
of `exp_avg` sizes in the checkpoint's optimizer state.

1. **Width** (`50e3b79`): h128 and h256 at lr 3e-3 constant, 40 epochs, no
   prior input, against the h64 100-epoch run. Then h256 at lr 1e-3. New in
   `50e3b79`: each epoch row logs the epoch's mean training NLL.
2. **Size** (`37321f1`): imm's best setup then (word segments at 16
   records, mobius's depth 16, `--prior-input both`, lr 1e-3) at h64 for 40
   epochs and h128 for 25, against h256 (3.1316, `word-segments`).
3. **Dropout** (`95f6645`, `--dropout`): dropout on the GRU's input and the
   state the head reads, and inside the cross-attention blocks; never on
   the carried state, the records or the prior's log-probs. h128 at 0.2
   for 25 epochs, resumed to 50; h256 at 0.2 for 40 epochs, two seeds.

## How to rerun

Scripts here, as they ran (paths are the original machine's; `cd` into a
worktree at the commit named):

| run | script | commit | proposed tag |
|---|---|---|---|
| h128, h256 at lr 3e-3 | `width_sweep.sh` | `50e3b79` | `rnd/residual-width/lr3e-3` |
| h256 at lr 1e-3 | not kept: `width_sweep.sh`'s command with `--lr 1e-3 --hidden-size 256 --embedding-size 256`, run id `..._lr1e-3_h256_e40` | `50e3b79` | `rnd/residual-width/h256-lr1e-3` |
| h64, h128, word, depth 16 | `small_widths_word_depth16.sh`, then `size_runs_resume.sh` (h64 resumed from epoch 27, then h128) | `37321f1` | `rnd/residual-width/size` |
| h128 dropout 0.2, 25 epochs | `h128_dropout.sh` | `95f6645` | `rnd/residual-width/h128-dropout` |
| h128 dropout to 50 epochs; h256 dropout | `dropout_queue.sh` | `74188ec` | `rnd/residual-width/h256-dropout` |
| h256 dropout, seed 1 | `after_mixunits.sh` (second half) | `f210ae7` | `rnd/residual-width/h256-dropout-seed1` |

`gutenberg5m_crossfit20_ps100_e1_resume.json` is the config the resumed runs
read (the usual config with `eval_initial` off). One comment in
`h128_dropout.sh` quoted the user; it was paraphrased. Every command is as
it ran.

## Results

### 2026-11-09: width at lr 3e-3, no prior input

| epoch | h64 held-out | h128 held-out | h128 train | h256 held-out | h256 train |
|---:|---:|---:|---:|---:|---:|
| 1 | 3.4508 | 3.4480 | 3.5149 | 3.4307 | 3.4896 |
| 5 | 3.3892 | 3.4371 | 3.4370 | 3.3509 | 3.3712 |
| 10 | 3.3473 | 3.3580 | 3.3702 | 3.3033 | 3.2971 |
| 12 | 3.3390 | 3.3462 | 3.3543 | 3.2992 | 3.2671 |
| 15 | 3.3287 | 3.3434 | 3.3355 | inf | 4.4611 |
| 20 | 3.3212 | 3.3325 | 3.3133 | 3.6171 | 3.5798 |
| 30 | 3.3156 | 3.3341 | 3.2800 | 3.5736 | 3.5787 |
| 40 | 3.3154 | 3.3412 | 3.2636 | 9.6567 | 3.6279 |

Best held-out: h64 3.3154 (epoch 40; 3.3078 by epoch 84), h128 3.3286
(epoch 21), **h256 3.2883 (epoch 13)**.

- lr 3e-3 is too high for the wider residuals: h256 diverged at epoch 15,
  fell back to the prior alone (about 3.57), and blew up at epoch 40. The
  cross-attention context norm grew all along (h256: 3.2 at epoch 1, 15 at
  epoch 15). Gradients were already clipped at 1.0.
- The wider residuals overfit: training PPL falls below held-out by epoch
  15 at h128 and by epochs 9–12 at h256.

### 2026-11-09: h256 at lr 1e-3

| epoch | held-out | train | context norm |
|---:|---:|---:|---:|
| 1 | 3.4441 | 3.4980 | 4.3 |
| 5 | 3.3258 | 3.3484 | 4.4 |
| 8 | 3.2786 | 3.2582 | 5.8 |
| 10 | **3.2711** | 3.2115 | 6.3 |
| 12 | 3.2856 | 3.1706 | 7.2 |
| 20 | 3.3350 | 3.0729 | 8.7 |
| 40 | 3.4349 | 2.9505 | 9.9 |

- 3.2711 at epoch 10 (9.8 min) ties the 474k transformer (3.2694, 7.1 min)
  and is 1.1% better than h64 after 100 epochs.
- Stable at 1e-3: the context norm levels off near 9–10.
- It overfits from epoch 8 (2.95 training against 3.43 held out at epoch
  40). On Gutenberg, wide runs are compared at their best epoch from here
  on, chosen on held-out text (slightly optimistic, the same for every
  run).

### 2026-11-09/10: size, word segments, mobius's depth 16

| model | trained | best held-out (epoch) | last held-out (epoch) |
|---|---:|---:|---:|
| imm h64 | **178,833** | 3.1606 (37) | 3.1615 (40) |
| imm h128 | 535,505 | 3.1464 (11) | 3.1634 (25) |
| imm h256 (`word-segments`) | 1,789,521 | 3.1316 (7) | 3.1714 (12) |
| transformer 4 × 96 | 474k | 3.2694 | |
| transformer 4 × 192, dropout 0 (@air) | 1,832,832 | 3.2239 (step 10k) | 4.1457 (step 50k) |

- h64 beats the 474k transformer by 3.3% with 38% of its parameters. It
  passed 3.2694 at epoch 6.
- h128, about the transformer's size, is 3.8% better at its best (epoch 11)
  and overfits after.
- Width buys little at the best epoch: about 0.45% for each tripling of
  parameters. The wider residuals peak sooner (epochs 37, 11, 7).
- The h64 run was cut at epoch 27 of 40 when the machine ran low on memory,
  and resumed from its epoch-27 checkpoint (`size_runs_resume.sh`). Peak RSS
  rose from 4.7 GB to 9.0 GB at epochs 13–17, much of it pages of the 10 GB
  depth-16 evidence file.

### 2026-11-10: dropout 0.2

| model | trained | best held-out (epoch) | last held-out (epoch) |
|---|---:|---:|---:|
| imm h128, dropout 0.2 | 535,505 | 3.1079 (25) | 3.1079 (25) |
| imm h128, dropout 0.2, resumed to 50 epochs | 535,505 | **3.0671** (48) | **3.0676** (50) |
| imm h256, dropout 0.2, 40 epochs | 1,789,521 | **3.0217** (39) | **3.0247** (40) |
| imm h256, dropout 0.2, 40 epochs, seed 1 | 1,789,521 | 3.0395 (40) | 3.0395 (40) |
| transformer 4 × 192, dropout 0.2 (@air) | 1,832,832 | 3.0542 (step 47.5k) | 3.0594 (step 50k) |
| nanoGPT char config | 10.7M | 3.0212 | |

- h128: dropout makes it 1.2% better at the best epoch and 1.8% at the
  last. No overfitting: training PPL with dropout on 3.157 against 3.108
  held out at epoch 25. 58 min, peak RSS 6.8 GB. Resumed to 50 epochs at
  constant lr, it was still improving about 0.0014 an epoch over epochs
  40–50, 0.42% behind the 1.83M dropout transformer with 29% of its
  parameters.
- h256, seed 1337: 1.1% better than the 1.83M dropout transformer at the
  same size, and level with nanoGPT's 10.7M (3.0212). 31 min (46.5 s an
  epoch), peak RSS 2.9 GB. Selected epochs: 3.1299 (7), 3.0989 (10),
  3.0705 (15), 3.0549 (20), 3.0443 (25), 3.0327 (30), 3.0302 (35).
- h256, seed 1: behind seed 1337 at every epoch from 1 on. Two seeds: mean
  best 3.0306, mean last 3.0321, 0.8% and 0.9% ahead of the transformer
  (one seed). The seed spread is 0.59% at the best, wider than the
  0.15–0.3% seen at smaller sizes. This seed was a check another imm
  session asked for, not a request of the user's.

## What we learned

- On 5M characters the wide residuals were unregularised, not short of
  capacity, as @air read it. Dropout 0.2 became the default for imm's runs.
- The smallest residual (h64) already beat the transformer of 2.7x its
  size, so the prior was worth a lot at small sizes.
- Overfitting made Gutenberg 5M too small to compare architectures. The
  work moved to PG-19 next (`pg19-4.75m-split`).
