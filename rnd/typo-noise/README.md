---
title:    Typos in the held-out text, and training noise
kind:     experiment
status:   concluded
outcome:  negative
question: >-
  How much does a typo in the history cost imm, compared with the count
  prior alone, and does training the residual on typo'd input make it more
  tolerant?
answer: >-
  A typo costs imm what it costs Kneser–Ney alone: with clean targets about
  14.0, 13.8 and 13.3 bits per typo at 1%, 2% and 5% substitutions,
  against KN's 14.0, 13.6 and 13.0. The residual wins none of it back.
  Training noise at 1% acts as a regulariser (best 3.1244 against 3.1316)
  but not as typo tolerance; at 2% it is slightly worse on typo'd text. The
  user judged the rest not worth running in this form.
opened:   2026-11-09
updated:  2026-11-10
code: >-
  --input-noise, --eval-clean-targets, --eval-only at 1230878; the typo'd
  held-out configs at df20f03 (branch typo-noise); --eval-residual-input
  (perfect repair) at 8ec595d (branch typo-perfect-repair, on typo-noise).
  Branches deleted; the commits are on main.
data: >-
  Gutenberg 5M split 4bfd5a43d446644a. Held-out text with unigram
  substitutions at 1%, 2%, 5% (seed 1: 2,506, 5,057 and 12,507 of 250,000
  characters changed). imm's depth-8 gated prior, cross-fitted, strength
  1.0; mobius's KN depth-16 evidence, rebuilt by mobius for each typo'd
  text (gutenberg5m_kn_typo1/2/5).
slots:    none (before the slot system)
tags:     [noise, typos, robustness, prior, gutenberg]
related:  [edit-repair, word-segments]
---

# Typos in the held-out text, and training noise

## Question

The user's idea (2026-11-09, via @mobius): put typo noise in the history.
Real prompts have typos. Exact counts lose every longer context after a
typo, so a count prior should suffer. Does the residual protect imm from
that, and can training on noisy input make it more tolerant?

## Design

- **Model:** the 3.1316 setup (`word-segments`: h256, word segments at 16
  records, mobius's depth 16, `--prior-input both`, lr 1e-3), scored at its
  best epoch.
- **Scoring** on each typo'd held-out text, two ways:
  - "typo'd": everything reads the typo'd text and scores its characters;
  - "clean targets" (`--eval-clean-targets`): the same reading, scored on
    the original's characters.
- **Training noise** (`--input-noise 0.0n`): the residual reads freshly
  typo'd input each epoch, with the prior, evidence and targets clean.
  Noise 1, 2 and 5, 12 epochs each.
- **Bits per typo** = (bits/char − clean bits/char) × 250,000 / characters
  changed.
- **Perfect repair** (`--eval-residual-input`, built, not run): the prior,
  evidence, segments and targets on the clean text, only the residual
  reading the typo'd copy. It is the ceiling for a typo-tolerant prior,
  with the plain typo'd scoring the floor.

## How to rerun

Scripts here, from a worktree at `df20f03` (proposed tag
`rnd/typo-noise/queue`). Paths are the original machine's.

- `typo_queue.sh` ran: the noise-0 model scored on each text, then each
  noise level trained and scored.
- `typo_queue_rest.sh` (the rest, skipping what finished) and
  `typo_repair_evals.sh` (perfect repair, from a worktree at `8ec595d`) were
  written and never run. One comment in `typo_queue_rest.sh` gave the clock
  time of the stop; it was cut to the date. Every command is as written.
- The typo'd texts are made by `scripts/make_typo_text.py`; configs
  `configs/gutenberg5m_crossfit20_ps100_e1_heldout_typo{1,2,5}_seed1.json`.

## Results

### 2026-11-10

| model | clean | 1% typo'd | 1% clean tgt | 2% typo'd | 2% clean tgt | 5% typo'd | 5% clean tgt |
|---|---:|---:|---:|---:|---:|---:|---:|
| noise 0 (epoch 7) | 3.1316 | 3.6844 | 3.4528 | 4.3071 | 3.7994 | 6.5036 | 4.9592 |
| noise 1 (epoch 7) | **3.1244** | **3.6819** | **3.4466** | **4.3067** | **3.7927** | 6.5238 | **4.9588** |
| noise 2 (epoch 9) | 3.1294 | 3.7080 | 3.4735 | 4.3571 | 3.8433 | | |

- **A typo costs the model what it costs Kneser–Ney alone.** With clean
  targets: 14.0, 13.8, 13.3 bits per typo (noise 0), against KN's 14.0,
  13.6, 13.0 (mobius's numbers). Scored on the typo'd text: 23.4, 22.7,
  21.1 against 23.5, 22.8, 21.1. The residual keeps its 0.21 bits/char
  lead over KN (1.6469 against 1.8528 on clean text) and wins back none of
  the typo's cost: the loss is all in the prior's contexts.
- **Training noise regularises; it doesn't give typo tolerance.** Noise 1:
  best 3.1244 (−0.23%), last 3.1432 against 3.1714 (−0.9%). Noise 2: 3.1294
  at epoch 9, last 3.1438. On typo'd text noise 1 equals noise 0 (within
  0.3%), and noise 2 is 0.6–1.2% worse (15.0 and 14.7 bits per typo with
  clean targets). A likely reason: in training the prior read clean text,
  so the residual learned to fix its own input, not a prior that a typo
  had fooled.
- Noise 2's mean residual gate is lower (0.457 against 0.519), so it leans
  on the prior more, which hurts once the prior reads typos. A guess from
  one number.
- **Not run:** noise 2 on the 5% text, noise 5, and the 12 perfect-repair
  scorings. The queue was stopped by a low-memory reaper during noise 2's
  5% scoring. Eval CSVs are in `runs/typo_evals/`.

### 2026-11-10: the user's decision

The user found the worse scores expected and judged it unnecessary to
pursue typos this way. imm's answer: a model does need typo tolerance for
real prompts, and for imm the weak part is the prior. It proposed two
smaller checks in place of the rest of the queue: perfect repair for the
3.1316 model only, and the transformer scored on the same typo'd texts.
`research-lines.md` records that as waiting for the user's choice; neither
has run.

## What we learned, and what's open

- imm's typo weakness is its prior: exact counts lose the longer contexts,
  and the residual, trained against a clean prior, doesn't make up for it.
- Input noise at 1% is a mild regulariser (0.9% at the last epoch here),
  later overtaken by dropout.
- Open, unrun: how much a perfect repair would win back (the ceiling for
  edit-neighbour priors; see `edit-repair`), and how a transformer fares on
  the same texts.
