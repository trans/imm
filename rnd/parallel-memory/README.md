---
title:    "Parallel memory: the GRU carries its own state"
kind:     experiment
status:   concluded
outcome:  positive
question: >-
  If the GRU carries its own state to the next word and writes records
  from it, so the attention never feeds back into the GRU, can a whole
  update run at once, and does the model lose anything?
answer: >-
  It loses nothing and runs 3.2x faster on model time: 3.1518 against the
  loop's 3.1517 on the 20M split, 276k training characters a second
  against 77.6k (2,779 s against 6,969 s of wall time for 20 epochs). It
  was ahead by 0.2–0.4% over epochs 2–12 and level from epoch 14. Every
  later imm run uses this path.
opened:   2026-11-13
updated:  2026-11-13
code: >-
  787ee25 (branch parallel-memory, deleted; the commit is on main).
data: >-
  PG-19 20M split, data/pg19/imm-split-20m (mobius e3622a7). Mobius's
  pg19split20m_gated_d16_pg200m prior (alone 1.8205 bits/char) with
  compact pg19split20m_kn depth evidence.
slots:    [23]
tags:     [speed, parallel, memory, gru, pg19]
related:  [blocks-20m, memory-knockouts, carry-bound, memory-window, word-tier]
---

# Parallel memory: the GRU carries its own state

## Question

In imm's loop, a stream carried the state after the attention block into
the next segment's GRU, so each segment had to wait for the last one's
attention. That made the training a Python loop over segments. If the GRU
carries its own state instead, and records are written from the GRU's own
states, the attention never reaches the GRU. Then an update can run at
once. Is the feedback from attention into the GRU worth anything?

## Design

- `--carry-bound gru`: the GRU carries its own last state to the next
  word.
- `--memory-state gru` (new): each word's record is written from the GRU's
  own state at the word's last character.
- With both, the batched path (`parallel_memory_streams`) runs an update
  at once: one GRU call over each stream's span gives every state and
  every record; each attention block then runs over every position, with a
  mask showing a position in word k the records the per-segment loop would
  hold (those carried in and those of earlier words, the last 64).
- Tests check it against the loop and against separate single-stream runs;
  the default path stays bit-identical to `a353202`.
- Every other flag as the 20M base run (`blocks-20m`: 1 × h256, 64 records,
  64 × 64, 20 epochs, dropout 0.2, seed 1337), in place of `--carry-bound
  norm --carry-scale 32`.

## How to rerun

`par_slot.sh 23` here, from a worktree at `787ee25`; proposed tag
`rnd/parallel-memory/slot23`. It runs the model, then its take-the-prior-away
test, then ends the slot. One comment quoted the user; it was paraphrased.
Every command is as it ran. `same_grads_par.py` was the CPU check that the
default (loop) path still gives `a353202`'s loss and gradients bit for bit.

## Results

### 2026-11-13: slot 23

| | base (loop, carry after attention) | parallel |
|---|---:|---:|
| held-out, best (epoch 20) | 3.1517 | **3.1518** |
| training NLL, epoch 20 | 1.1117 | 1.1070 |
| first character / later | 3.451 / 3.092 | 3.440 / 3.094 |
| model time, characters/s | 77.6k (85.7k with the grid fix) | **276k** |
| wall time, 20 epochs | 6,969 s | 2,779 s |
| reader wait | 0 | 1,096 s of 2,755 s training |
| take-away: own, prior in input / none | 62.86 / 98.06 | 61.41 / 99.45 |
| probes, prior in input / none | 4.29 / 9.42 | 4.32 / 9.58 |
| tanh(gate_attn), tanh(gate_mlp) | −0.015, 0.36 | −0.022, 0.75 |

- The same score, 3.2x faster on model time (2.5x in wall time: the
  prior-row reader became the limit, 40% of training spent waiting for rows
  under memory pressure).
- Training NLL is lower: the feedback into the GRU wasn't buying anything
  measurable.
- One seed each.

## What we learned

- The feedback path from attention into the GRU can go. The parallel path
  made the later depth, word-tier and knockout runs affordable (`blocks-20m`,
  `word-tier`, `memory-knockouts`).
- The remaining bottleneck became the disk reads of the prior rows, fixed
  separately (the row reader, `b38374b`; engineering, no entry).
