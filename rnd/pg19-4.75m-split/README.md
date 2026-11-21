---
title:    "One corpus: imm on mobius's 4.75M PG-19 split"
kind:     experiment
status:   concluded
outcome:  reference
question: >-
  With the prior and the residual on one corpus (PG-19, the text mobius
  counts), where does imm stand on its own prior, and how much does
  mobius's depth-16 prior with 200M characters of books add?
answer: >-
  imm's gated prior alone scores 2.1183 bits/char (PPL 4.3417). h256 on it
  reaches 3.7059 at best (epoch 55), 0.8% ahead of the 1.84M transformer
  trained on the same text (3.7364). On mobius's 200M-book prior it
  reaches 3.3220 (epoch 16), 0.158 bits/char better; the residual's own
  gain shrinks from 0.228 to 0.095 bits/char, and the split's 4.75M
  training characters become the limit (0.20 bits/char train/held-out
  gap). Against a 1.84M transformer with the same prior in its input
  (3.2887), imm is 1.0% behind.
opened:   2026-11-10
updated:  2026-11-12
code: >-
  d65044d (branch pg19-split, deleted; the commit is on main): configs
  for mobius's split.
data: >-
  PG-19 4.75M split, data/pg19/imm-split (mobius cbb6d4c): 190 pieces of
  25,000 characters from 190 training books; 50 held-out pieces of 5,000
  from 50 test books (250,000 characters, hash 095cbd13…); 96 characters.
  Priors: imm's depth-8 gated prior fitted on the split; mobius's
  pg19split_gated_d16_pg200m (depth 16, 200M characters of books; alone
  1.8266 bits/char). Depth evidence: mobius's pg19split_kn (training text
  only).
slots:    none (before the slot system)
tags:     [pg19, corpus, prior, books, baseline, overfitting]
related:  [mobius-prior-rows, carry-bound, prior-dropout, transformer-prior-pg19, blocks-20m]
---

# One corpus: imm on mobius's 4.75M PG-19 split

## Question

On Gutenberg 5M the best prior (pg20m) counted books the residual never
trained on, so imm's lead over the transformer was uneven
(`mobius-prior-rows`). The user wanted imm to train on part of the same
English text that mobius counts, judging English, not one corpus
(relayed by mobius, 2026-11-10). Mobius built a PG-19 split for it.

## Design

- **The split:** 190 training pieces of 25,000 characters from 190 PG-19
  training books, 50 held-out pieces of 5,000 from 50 test books, 96
  characters. Cross-fitting leaves out each row's own book.
- **imm's prior:** depth 8, theta refitted with the last ten pieces as the
  gate's fitting text (`pg19_theta.sh`).
- **Reference run:** h256 on imm's own prior, mobius's KN depth evidence to
  16 without the books. The same command as the Gutenberg 5M 3.0076 run:
  word segments at 16 records, `--prior-input both`, dropout 0.2, 80
  epochs, cosine 1e-3 to 1e-4, seed 1337.
- **Books prior:** the same command on mobius's
  `pg19split_gated_d16_pg200m` (mobius `f684bd6`). The depth evidence stays
  `pg19split_kn`, so the prior is the only change.

## How to rerun

Scripts here, as they ran, from a worktree at `d65044d`. Paths are the
original machine's; mobius's rows are in its `runs/imm/`.

| step | script | proposed tag |
|---|---|---|
| fit imm's theta on the split | `pg19_theta.sh` | `rnd/pg19-4.75m-split/ref-h256` |
| build the cross-fitted cache, one-epoch check | `pg19_prior_check.sh` | (same) |
| reference run, imm's prior | `pg19_ref_h256.sh` | `rnd/pg19-4.75m-split/ref-h256` |
| mobius's 200M-book prior | `pg19_pg200m_h256.sh` | `rnd/pg19-4.75m-split/pg200m-h256` |

Both tags are on `d65044d`. The config is
`configs/pg19split_crossfit20_ps100_e1.json`.

## Results

### 2026-11-10: the priors alone

imm's gated prior: 2.1183 bits/char held out (PPL 4.3417). Witten–Bell
2.6221, unigram 4.5797. Mobius's KN (limit 6): 2.1328.

### 2026-11-10: reference run, h256 on imm's prior

Best 3.7059 (epoch 55), last 3.7144: 1.8898 / 1.8931 bits/char. 1,949,357
trained parameters (the 96-character vocabulary enlarges the input and
output layers). 63.8 min, peak RSS 3.2 GB.

| epoch | held-out | train |
|---:|---:|---:|
| 0 (prior alone) | 4.3417 | |
| 1 | 4.0650 | 4.0423 |
| 10 | 3.7797 | 3.5571 |
| 20 | 3.7550 | 3.4360 |
| 30 | 3.7243 | 3.3693 |
| 40 | 3.7142 | 3.3223 |
| 50 | 3.7088 | 3.2857 |
| 60 | 3.7101 | 3.2567 |
| 80 | 3.7144 | 3.2288 |

- The residual gains 0.228 bits/char over the prior (Gutenberg 5M, same
  command: 0.248).
- The training text is the limit now: training PPL ends at 3.23 against
  3.71 held out, 0.20 bits/char apart (0.06 on Gutenberg 5M). 190 pieces
  are too few for this width to generalise to unseen books.
- A one-epoch spike at epoch 11 (held-out 8876, then 3.7657 at epoch 12),
  with smooth training loss and context norm. Not explained then; it
  didn't recur. `carry-bound` later found the cause in a similar spike.
- First / later characters of each segment: 3.8961 / 3.6768.

### 2026-11-11: on mobius's depth-16 prior with 200M characters of books

Best 3.3220 (epoch 16), last 3.3440: 1.7320 / 1.7416 bits/char. 119 min
(another project's jobs shared the GPU), peak RSS 3.1 GB.

| epoch | held-out | bits/char | train bits/char |
|---:|---:|---:|---:|
| 0 (prior alone) | 3.5471 | 1.8266 | |
| 1 | 3.411 | 1.7703 | 1.7489 |
| 10 | 3.338 | 1.7390 | 1.6566 |
| 16 (best) | 3.3220 | 1.7320 | 1.6319 |
| 30 | 3.3352 | 1.7378 | 1.5948 |
| 50 | 3.3322 | 1.7365 | 1.5633 |
| 80 | 3.3440 | 1.7416 | 1.5420 |

- 0.158 bits/char better than the reference run, on the same held-out
  text.
- The residual gains less on the stronger prior: 0.095 bits/char at its
  best (0.085 at the end), against 0.228. It peaks at epoch 16, not 55.
- The gain sits almost all on later characters. Segment-first characters
  (word starts): 3.6971 → 3.6221, 0.029 bits/char. Later characters:
  3.5160 → 3.2614, 0.108. Mobius's prior is already much better within
  words than at word starts; word starts depend most on context from
  further back than 16 characters.
- A training spike at epoch 5 (training loss 1.692 → 1.798 bits/char,
  context norm 4.2 → 7.1); held-out barely moved.

### 2026-11-11/12: the transformer on the same split (@air)

| model, trained on | params | best | last | bits/char (best) |
|---|---:|---:|---:|---:|
| transformer, 4.75M | 1.84M | 3.7364 (step 47,500) | 3.7404 | 1.9017 |
| transformer + the 200M-book prior in its input | 1.86M | **3.2887** (step 30k) | 3.3186 | |
| transformer, the prior on its logits only | 1.86M | 3.3540 (step 27.5k) | 3.3859 | |
| imm h256 on its own prior | 1.95M | 3.7059 | 3.7144 | 1.8898 |
| imm h256 on the 200M-book prior | 1.95M | 3.3220 | 3.3440 | 1.7320 |

- Same text on both sides (imm's own prior): imm leads by 0.8% at best and
  0.7% at last. On Gutenberg 5M the lead was 1.5%.
- With the books prior on both sides, imm is 1.0% behind the
  prior-in-input transformer at best (0.8% at last), and 1.0% ahead of the
  logits-only form. Level within about twice the seed spread. An earlier
  "11% ahead" had set imm against the transformer without the prior; the
  fair bar is the transformer with the prior. Details in
  `transformer-prior-pg19`.

## What we learned

- On PG-19 the 4.75M training characters are too few: the model overfits,
  and the deeper prior leaves the residual little to add. The 20M split
  came next (`blocks-20m`).
- The fair comparison is a transformer given the same prior. Every later
  comparison with a prior uses one.
