---
title:    Held-out spikes, and bounding the carried state
kind:     experiment
status:   concluded
outcome:  positive
question: >-
  Why did single epochs read held-out perplexities of 129 or 8876 while
  training stayed smooth, and does longer reach (64 records) help on a
  strong prior once that is fixed?
answer: >-
  One held-out stream's carried state ran away: its norm grew from 9 to
  3165 over the text, because the state carried between segments is taken
  after the attention block, which adds to it every segment. Rescoring the
  epoch-5 weights with the carried state squashed by tanh gave 3.380
  instead of 121.1. Trained with tanh, the reach-64 run had no spike in 50
  epochs (best 3.3250) but trailed the unbounded run by about 0.5%, ending
  level with the 16-record run (3.3220). The softer cap (--carry-bound
  norm --carry-scale 32) became the default.
opened:   2026-11-11
updated:  2026-11-11
code: >-
  reach-64 run at d65044d (branch pg19-split); --carry-bound at db210ad
  (branch carry-bound); trained with tanh at 2fc6d00 (branch
  pg19-carry-compact, which adds compact depth evidence, 9007354);
  --carry-scale and --carry-bound norm at 2b58589. Branches deleted; the
  commits are on main.
data: >-
  PG-19 4.75M split, data/pg19/imm-split (mobius cbb6d4c; 250,000
  held-out characters). Mobius's pg19split_gated_d16_pg200m prior (alone
  1.8266 bits/char, PPL 3.5471) with pg19split_kn depth evidence.
slots:    none (before the slot system)
tags:     [stability, carried-state, reach, bug, pg19]
related:  [pg19-4.75m-split, blocks-20m]
---

# Held-out spikes, and bounding the carried state

## Question

Two runs on the 4.75M PG-19 split had single epochs with huge held-out
perplexity and smooth training: the reference run's epoch 11 (8876) and
the reach-64 run's epoch 5 (129.17). The reach-64 run then blew up in
training at epoch 19. What causes it? And with it fixed, does a longer
reach help on mobius's strong prior, where the residual's gain at word
starts was small (`pg19-4.75m-split`)?

## Design

- **Reach 64:** the books-prior run of `pg19-4.75m-split` (3.3220) with
  `--max-memory 64 --segments-per-update 64`, so memory and the training
  window both reach about 360 characters, against 90 and 180.
- **Diagnosis:** epoch 5's weights were saved before epoch 6 overwrote
  them. Rescored with the run's own arguments on the CPU (`spike_eval.sh`),
  then traced per character and per stream (`spike_trace.py`, which wraps
  `segment_loss_streams` in the runner's `--eval-only` path).
- **Fix** (`--carry-bound`, `db210ad`): `tanh` squashes the carried state
  into the GRU's range; `gru` carries the GRU's own last state. Default
  `none`. First applied only at evaluation, to the epoch-5 weights; then
  trained with `tanh`.
- **Softer bounds** (`2b58589`): `--carry-scale c` gives c·tanh(x/c), and
  `--carry-bound norm` caps the carried state's length, set above a
  healthy stream's 8–26.

## How to rerun

Scripts here, as they ran. Paths are the original machine's;
`spike_epoch5.pt` was a scratch copy of the epoch-5 weights and is gone.

| step | script | commit | proposed tag |
|---|---|---|---|
| reach-64 run | `pg19_reach64.sh` | `d65044d` | `rnd/carry-bound/reach64` |
| rescore epoch 5 | `spike_eval.sh cpu` | `d65044d` | (same) |
| trace per character | `spike_trace.sh TAG CHECKPOINT` with `spike_trace.py` | `d65044d` | (same) |
| rescore with a bound | `spike_trace_bound.sh TAG CHECKPOINT tanh\|gru` | `db210ad` | `rnd/carry-bound/rescore` |
| trained with tanh | `pg19_reach64_tanh.sh` | `2fc6d00` | `rnd/carry-bound/tanh` |

`pg19_reach64_tanh.sh` reads the depth evidence from the compact layout
(`runs/pg19split_kn_compact3`, the same bytes as `pg19split_kn`), which
`9007354` added.

## Results

### 2026-11-11: reach 64, unbounded

Held-out: 3.434 (epoch 1), 3.358 (4), 3.322 (10), **best 3.3155 at epoch
13** (1.7292 bits/char, 0.003 better than the 16-record run's 1.7320, three
epochs earlier). Epoch 5 read 129.17. At epoch 19 the runaway reached
training: training loss 4883.5 bits/char, context norm 4.75 → 35.1,
held-out about 1e117. The run was stopped at epoch 19.

### 2026-11-11: the spike sits in one stream

- Epoch 5's weights rescored on the CPU: 121.1. So the weights cause it,
  not a race in evaluation.
- Held-out piece 47 averages 260 bits/char; every other piece is normal
  (median 1.69). The loss climbs about linearly from 33 to 750 bits/char
  over about 3,500 characters, then falls to 1.6 at position ~238,500,
  where evaluation stream 60 ends. Piece 47 is ordinary prose.
- Stream 60's carried state norm per evaluation chunk: 9.0, 10.2, 14.0,
  52.8, 199.5, 394, 662, …, 3165. The other 63 streams stay at 8.3–26
  (median about 10.5).
- **Why:** a stream carries the state *after* the attention block into the
  next segment's GRU. The block adds to the GRU's output
  (x + tanh(g)·f(LN(x))). Once the GRU's update gate passes a large carried
  state through, the block adds a similar step every segment, and nothing
  normalises the state that runs through time. It grows linearly, and the
  logits with it.

Epoch 5's weights rescored with a bound only at evaluation:

| carry | held-out |
|---|---:|
| none (as trained) | 121.1 |
| tanh | **3.380** |
| gru | 3.447 |

With tanh it is near epoch 4's 3.358.

### 2026-11-11: trained with `--carry-bound tanh`

- Epoch 0 is 3.547, the same as with the dense evidence.
- **Best 3.3250 at epoch 25 (1.7334 bits/char), no spike in 50 epochs.**
- The job was stopped after epoch 50 when the system ran low on memory, and
  not restarted.
- Epoch for epoch it trails the unbounded reach-64 run by about 0.5%, on
  training loss too (epoch 10: 3.3392 against 3.3223; training 1.6659
  against 1.6566). The squash throws away information; it ends level with
  the 16-record run, and the reach gain is gone.
- Held-out: 3.3392 (10), 3.3324 (20), 3.3298 (30), 3.3308 (40), 3.3339
  (50).

## What we learned

- The earlier unexplained spikes (the reference run's epoch 11, the
  16-record run's epoch-5 training spike) are likely the same runaway; not
  checked, since those weights are gone.
- A bound is needed, but tanh at scale 1 costs about 0.5%. The softer
  `--carry-bound norm --carry-scale 32` (`2b58589`) leaves healthy states
  as they are and stops only the runaway. It is what the 20M runs used
  (`blocks-20m`), and parallel memory later moved to `--carry-bound gru`
  (`parallel-memory`).
- Reach 64 bought 0.003 bits/char at best on this split, within noise; the
  question of reach came back at larger scale (`memory-window`).
