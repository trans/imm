---
title:    Batched training streams
kind:     experiment
status:   concluded
outcome:  positive
question: >-
  imm trained one stream at about 3,400 characters a second. If the
  training text is cut into N contiguous streams, each with its own memory
  and carried state, trained in one batched GRU call per segment index, is
  the training budget the limit?
answer: >-
  Yes. On Tiny canonical, 16 streams at lr 3e-3 for 10 epochs scored
  3.6769 against 3.7734 for one stream and one epoch (2.6% better), in less
  time (255 s against 326 s). On Gutenberg 5M, 64 streams at lr 3e-3 gave
  3.3460 after 10 epochs (8.8 min) and 3.3121 after 100 (best 3.3078 at
  epoch 84), against the one-stream best 3.4359 (about 65 min).
opened:   2026-11-09
updated:  2026-11-09
code: >-
  15c5011 (branch batched-streams, deleted; the commit is on main).
data: >-
  Tiny Shakespeare canonical 4fa9aec1db6b3aea (prior alone 3.8595);
  Gutenberg 5M 4bfd5a43d446644a (prior alone 3.5716). imm's depth-8 gated
  prior, cross-fitted, strength 1.0. No prior input to the residual.
slots:    none (before the slot system)
tags:     [training, throughput, streams, gutenberg, tiny]
related:  [transformer-baseline-gutenberg5m, residual-width, depth-evidence-input]
---

# Batched training streams

## Question

The transformer comparison (`transformer-baseline-gutenberg5m`) showed imm's
shape, one stream fed 16 characters at a time, cost 600x in throughput. The
segments of one stream can't be fused, because each segment's start depends
on the last one's attention. Independent streams can be batched. Does more
training, at the same wall time, close the gap?

## Design

`--train-streams N` cuts the training segments into N equal contiguous
runs. Each keeps its own memory and carried state, and each update takes
`--segments-per-update` (32) segments from every stream, one batched GRU
call per segment index. Evaluation stays one stream. Tests check that one
stream reproduces the old loss and that B streams match B separate runs.

Arms, all at residual scale 1.0, L2 0.05, no prior input:

- Tiny canonical: 16 or 64 streams, lr 1e-3 or 3e-3, 10 epochs, against one
  stream at lr 3e-4 for one epoch.
- Gutenberg 5M: 64 streams, lr 3e-3, 10 epochs and 100 epochs, hidden 64.

## How to rerun

Commit `15c5011` (branch `batched-streams`, deleted; the commit is on
`main`). Proposed tags `rnd/batched-streams/tiny-s16` and
`rnd/batched-streams/gutenberg-s64` (both on `15c5011`).

No run script was kept. The flags, from the run ids and the notes:

```sh
# Tiny, 16 streams, lr 3e-3, 10 epochs (the 3.6769 baseline)
python scripts/run_segment_memory_model.py --config configs/tiny_canonical_crossfit20_ps100_e1.json \
  --prior-residual-scale 1.0 --train-streams 16 --lr 3e-3 --epochs 10
# Gutenberg, 64 streams, lr 3e-3, 10 or 100 epochs
python scripts/run_segment_memory_model.py --config configs/gutenberg5m_crossfit20_ps100_e1.json \
  --prior-residual-scale 1.0 --prior-residual-l2 0.05 --train-streams 64 --lr 3e-3 --epochs 100
```

`--run-id` and `--output` were set per run (`..._s16_lr3e-3_e10`,
`..._s64_lr3e-3_e100`).

## Results

### 2026-11-09: Tiny canonical

Times include an 8 s single-stream evaluation per epoch.

| streams, lr | training chars/s | epoch 1 | 2 | 5 | 10 | total |
|---|---:|---:|---:|---:|---:|---:|
| 1, 3e-4 (before) | ~3.4k | 3.7734 | | | | 326 s |
| 16, 1e-3 | 63k | 3.8502 | 3.7603 | 3.7387 | 3.6956 | 252 s |
| 16, 3e-3 | 63k | 3.7628 | 3.7458 | 3.7008 | **3.6769** | 255 s |
| 64, 1e-3 | 244k | 3.8568 | 3.8372 | 3.7497 | 3.7284 | 132 s |
| 64, 3e-3 | 244k | 3.8297 | 3.7649 | 3.7358 | 3.6909 | 132 s |

- The training budget was a limit: 10 epochs at 16 streams beat one epoch
  at one stream by 2.6%, and beat `--prior-input both`'s 3.7341, in less
  time. Every run was still improving at epoch 10.
- At 64 streams an epoch is 56 updates; lr 3e-3 makes up most of the
  difference.

### 2026-11-09: Gutenberg 5M, 64 streams, lr 3e-3

10 epochs (216 updates each), held-out PPL by epoch from 0: 3.5716 (prior),
3.4483, 3.4309, 3.4152, 3.3958, 3.3816, 3.3716, 3.3624, 3.3578, 3.3522,
**3.3460** (1.742 bits/char). 285k training characters a second; 8.8 min in
all, of which 5.2 min was the single-stream evaluation (31 s each).

- 2.7% better than the one-stream best (3.4359, about 65 min) in an eighth
  of the time. Epoch 2 alone (3.4309) beats it.
- Still 2.3% behind the imm-sized transformer (3.2694, 7.1 min).

100 epochs (80 min): epoch 1 3.4508, 2 3.4363, 5 3.3892, 10 3.3474, 20
3.3212, 30 3.3157, 50 3.3088, 70 3.3157, 90 3.3102, **100 3.3121** (1.728
bits/char); best 3.3078 at epoch 84.

- Flat from epoch 30, without overfitting: at hidden 64 and constant lr
  3e-3, more epochs stop helping. 1.3% behind the 474k transformer.
- The same settings gave 3.4508 after epoch 1 here and 3.4483 in the
  10-epoch run: GPU nondeterminism, about 0.07%.

## What we learned

- Every later imm run trains batched streams (16 on Tiny, 64 on the larger
  corpora).
- With the budget no longer the limit, the next limits were the residual's
  width, its context and the constant learning rate; none was separated
  here. Width came next (`residual-width`).
