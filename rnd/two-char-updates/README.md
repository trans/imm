---
title:    Weight updates every two characters
kind:     experiment
status:   running
outcome:  n/a
question: >-
  If a character GRU's weights are updated every 2 characters, so the error
  reaches back only 2 steps, how much worse does it predict than with a
  256-character window?
answer:   running
opened:   2026-11-21
updated:  2026-11-21
code:     standalone, this directory (tiny_tbptt.py); no branch
data:     PG-19 100M split (rnd/README.md), its first 2M training characters
slots:    none (CPU, about 1 GB)
tags:     [gru, truncation, training]
related:  [gru-tree]
---

# Weight updates every two characters

## Question

imm trains with truncated backpropagation: the state carries from update to
update, but the error from a prediction reaches back only to the start of its
update. In the `gru-tree` discussion the user asked for the simplest case: one
stream, a plain character GRU, its weights updated every other character. How
bad is it?

The state still carries the whole way, so the GRU reads the same text either
way. What changes is what it can learn: with a 2-character window, nothing
ever rewards keeping something in the state for more than 2 steps.

## Design

- **Model:** a tiny character GRU: embedding 64, GRU 128, a linear output
  layer; 93,024 parameters. No dropout.
- **Text:** one stream reads the first 2,000,000 characters of the 100M
  split's training text (its first 80 pieces of 25,000), once. The state is
  carried and detached between updates and reset to zeros at each piece's
  start; no update crosses a piece's start.
- **Updates:** every K characters, the mean loss over those K predictions,
  Adam at a constant learning rate, gradient norm clipped at 1.0.
  - **K = 2** (the question), at learning rates 1e-4, 3e-4 and 1e-3: it makes
    a million updates, so a smaller rate may suit it.
  - **K = 256** (the control), at 3e-4, 1e-3 and 3e-3.
  - Each window is read at its best learning rate.
- **Score:** the 50 held-out pieces, each from a zero state, every 250,000
  training characters: perplexity per character over characters 2 to 5,000
  of each piece (249,950 predictions). The model is tiny and sees 2M
  characters, so the numbers aren't comparable to imm's runs, only to each
  other.

## How to rerun

From the repo, on CPU: `sh rnd/two-char-updates/run.sh` runs the six arms in
turn and writes `results/k<K>_lr<LR>.json` (the curve, the final score).
`tiny_tbptt.py --help` gives the single-arm flags.

## Results
