---
title:    Mobius's prior rows under imm's residual
kind:     experiment
status:   concluded
outcome:  positive
question: >-
  imm's residual had run on imm's own depth-8 gated prior. Does a better
  prior from mobius (imm's gate mixed with Kneser–Ney, unit orders on top),
  or one that also counts 19.7M characters of PG-19 books, still help once
  the residual is trained?
answer: >-
  The better mix helps a little: h128 at 25 epochs scored 3.0927 against
  3.1079 (−0.49%); most of the prior's 0.0253 bits/char lead was won back
  by the residual. Counting the books helps more, and the gain survives
  training: 3.0315 against m2's 3.0944 (−2.03%). At h256 with 80 epochs of
  cosine decay, pg20m scored 2.9514 against 3.0076 on imm's prior (−1.87%
  at the last epoch). Depth evidence that also counts the books adds
  0.12%.
opened:   2026-11-10
updated:  2026-11-10
code: >-
  --prior-rows: mix_units at f210ae7 (branch prior-rows-loader); m2, pg20m,
  pg20m evidence and the 80-epoch runs at 3955731 (branch lr-schedule,
  which adds --lr-schedule cosine). Branches deleted; the commits are on
  main. The rows come from mobius (8737818, 37962d7, d7ac253).
data: >-
  Gutenberg 5M split 4bfd5a43d446644a, held-out PPL. Priors: imm's depth-8
  gated prior (alone 3.5716); mobius's gutenberg5m_mix_units (alone
  3.5094), _m2 (3.5001) and _pg20m (3.4575; also counts 19.7M characters
  of PG-19, 59 books). Depth evidence: mobius's KN rows gutenberg5m_kn,
  or gutenberg5m_kn_pg20m.
slots:    none (before the slot system)
tags:     [prior, mobius, outside-text, depth, gutenberg, lr-schedule]
related:  [residual-width, depth-evidence-input, pg19-4.75m-split]
---

# Mobius's prior rows under imm's residual

## Question

The division of labour (the user, 2026-11-09): mobius makes the priors, imm
the neural model. With `--prior-rows`, imm reads a prior's log-probs and
gate features from mobius's files instead of building its own. Two
questions:

1. Does a better prior on the same text help once the residual has trained,
   or does the residual learn the same corrections itself?
2. Does a prior that counts outside text (PG-19 books) help more?

## Design

Each run is the h128 dropout run of `residual-width` (3.1079 at epoch 25 on
imm's prior): word segments at 16 records, `--prior-input both`, lr 1e-3,
64 streams, dropout 0.2, 25 epochs, seed 1337. Only `--prior-rows` changes,
and the state features are named (`reliability,depth_norm`, imm's columns
0 and 3). The same seed means the same initialisation, data order and
dropout masks, so each pair is a paired comparison.

- `gutenberg5m_mix_units`: imm's gated prior and Kneser–Ney mixed per row,
  unit orders on top (mobius `8737818`).
- `_m2`: mix_units with small fixes (mobius `37962d7`).
- `_pg20m`: m2, also counting 19.7M characters of PG-19 (59 books) in every
  part of the prior. Mobius found no held-out 40-letter run in them.
- pg20m with `--prior-evidence-rows gutenberg5m_kn_pg20m` (mobius
  `d7ac253`), depth evidence that counts the books too.
- h256, 80 epochs, `--lr-schedule cosine --min-lr 1e-4` (no warm-up), on
  pg20m and on imm's prior.

## How to rerun

Scripts here, as they ran, from a worktree at the commit named. Comments
that quoted the user were paraphrased; every command is as it ran. Paths
are the original machine's, and the rows are in mobius's `runs/imm/`.

| run | script | commit | proposed tag |
|---|---|---|---|
| h128 on mix_units | `h128_mixunits.sh` | `f210ae7` | `rnd/mobius-prior-rows/mixunits` |
| h128 on m2, pg20m | `h128_m2_pg20m.sh` | `3955731` | `rnd/mobius-prior-rows/pg20m` |
| h128 on pg20m, books in the evidence | `h128_pg20m_ev.sh` | `3955731` | `rnd/mobius-prior-rows/pg20m-ev` |
| h256, 80 epochs, cosine | `h256_cos80.sh` | `3955731` | `rnd/mobius-prior-rows/h256-cos80` |

## Results

### 2026-11-10: mix_units against imm's prior (h128, 25 epochs)

| epoch | imm's prior | mobius's mix_units | difference | bits/char |
|---|---:|---:|---:|---:|
| 0 (prior alone) | 3.5716 | 3.5094 | −1.74% | 0.0253 |
| 1 | 3.3635 | 3.3378 | −0.76% | 0.0111 |
| 5 | 3.2352 | 3.2116 | −0.73% | 0.0105 |
| 10 | 3.1728 | 3.1513 | −0.68% | 0.0098 |
| 15 | 3.1474 | 3.1249 | −0.72% | 0.0104 |
| 20 | 3.1253 | 3.1092 | −0.52% | 0.0075 |
| 25 | **3.1079** | **3.0927** | **−0.49%** | 0.0071 |

- The better prior helps at every epoch, but the residual makes up most of
  the difference: 0.0253 bits/char before training, 0.007 by epoch 25 and
  still narrowing.
- 0.49% is above the 0.3% seed spread seen before, with the same sign at
  all 26 evaluations. 951 s, peak RSS 3.0 GB.

### 2026-11-10: outside text in the prior (h128, 25 epochs)

| epoch | imm's prior | mix_units | m2 | **pg20m** | pg20m vs m2 | bits/char |
|---|---:|---:|---:|---:|---:|---:|
| 0 (prior alone) | 3.5716 | 3.5094 | 3.5001 | 3.4575 | −1.22% | 0.0177 |
| 1 | 3.3635 | 3.3378 | 3.3242 | 3.2569 | −2.03% | 0.0295 |
| 5 | 3.2352 | 3.2116 | 3.2071 | 3.1514 | −1.74% | 0.0253 |
| 10 | 3.1728 | 3.1513 | 3.1467 | 3.0962 | −1.61% | 0.0234 |
| 15 | 3.1474 | 3.1249 | 3.1210 | 3.0659 | −1.77% | 0.0257 |
| 20 | 3.1253 | 3.1092 | 3.1050 | 3.0441 | −1.96% | 0.0286 |
| 25 | 3.1079 | 3.0927 | 3.0944 | **3.0315** | **−2.03%** | 0.0296 |

- The books' gain grows under the residual instead of shrinking: 0.018
  bits/char for the prior alone, 0.030 after training. A reading, not a
  measurement: Kneser–Ney adds what the residual can learn from the
  training text itself, while the books add statistics the training text
  doesn't hold.
- The residual didn't over-trust the pg20m prior (mean residual α 0.361
  against m2's 0.364).
- m2's fixes wash out (3.0944 against 3.0927).
- h128 on pg20m (535,505 trained parameters) is ahead of the 1.83M
  transformer (3.0542 best, 3.0594 last) by 0.74% and 0.91%. The
  transformer trained on the 4.75M training characters only, while this
  prior counts 24.4M.

### 2026-11-10: depth evidence that counts the books too

pg20m with `--prior-evidence-rows gutenberg5m_kn_pg20m` (contexts found at
lengths 8 / 12 / 16 for 85 / 42 / 12% of held-out rows, against 75 / 29 /
7%): **best 3.0279 (epoch 24), last 3.0300**, against 3.0315 / 3.0315.
−0.12% best, −0.05% last; lower in 24 of 25 epochs, mean −0.18%. Well under
the seed spread, one pair. The gain is all at each segment's first
character (3.2971 against 3.3064); later characters are equal (2.9745
both).

### 2026-11-10: how deep the counts reach, by corpus size

The user expected the average matched depth to grow by about 2 for every
10x of corpus. Checked on the two evidence files: per held-out character,
the longest context (1–16) the counts have seen at least n times.

| seen at least | 4.75M chars (`kn`) | 24.4M chars (`kn_pg20m`) | per 10× |
|---|---:|---:|---:|
| 1 | 9.89 (7.5% at 16) | 10.93 (12.5% at 16) | +1.47 |
| 2 | 8.95 | 10.07 | +1.57 |
| 5 | 7.92 | 9.08 | +1.63 |
| 10 | 7.22 | 8.40 | +1.66 |

1.5–1.7 per 10x here; the 16-character cap trims the top. At about 2 per
10x, 1 TB of text gives mean depths of about 17–20. The user decided that
depth 16 is enough for now (2026-11-10).

### 2026-11-10: h256, 80 epochs, cosine decay

| epoch | imm's prior | pg20m | difference | bits/char |
|---|---:|---:|---:|---:|
| 0 (prior alone) | 3.5716 | 3.4575 | −3.19% | 0.0468 |
| 10 | 3.1084 | 3.0382 | −2.26% | 0.0329 |
| 20 | 3.0621 | 2.9962 | −2.15% | 0.0314 |
| 40 | 3.0243 | 2.9653 | −1.95% | 0.0284 |
| 60 | 3.0108 | 2.9541 | −1.88% | 0.0274 |
| 80 | 3.0088 | 2.9524 | −1.87% | 0.0273 |

- pg20m: best 2.9514 (epoch 69), last 2.9524, 3.4% / 3.5% ahead of the
  1.83M transformer and 2.3% ahead of nanoGPT's 10.7M (3.0212). 60.5 min.
- imm's prior: best 3.0076 (epoch 74), last 3.0088. 58.5 min.
- Splitting 3.0217 → 2.9514 (−2.3%), seed 1337 throughout: the schedule
  (80 epochs of cosine against 40 at constant lr) is worth about 0.5%,
  within the seed spread at h256; the prior is worth 1.9% at the end, about
  three times the seed spread. 58% of the prior's 0.047 bits/char lead
  survives at h256 after 80 epochs; at h128 after 25, 77% survived.
- imm on its own prior is now 1.5% / 1.65% ahead of the 1.83M transformer
  and 0.45% ahead of nanoGPT's 10.7M, each side counting and training on
  the same 4.75M characters.

## What we learned

- A better smoother on the same text helps little after training; text the
  model never trains on helps a lot, and the gain survives. That made the
  prior's corpus, not its smoothing, the thing to grow, and it led to PG-19
  (`pg19-4.75m-split`), where the prior and the residual can share one
  corpus.
- The comparison with the transformer here is uneven (the prior counts
  24.4M characters, the transformer saw 4.75M). The PG-19 entries fix that.
- Cosine decay became the default schedule.
