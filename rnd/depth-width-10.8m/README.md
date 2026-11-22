---
title:    Depth against width at 10.8M
kind:     experiment
status:   planned
outcome:  n/a
question: >-
  At 10.8M imm trails the transformer by 1.99%, and the gap is in plain
  prediction (new words, lowercase letters), not in what the records copy.
  The transformer spends its parameters on 6 layers of 384; imm on one GRU
  of 448 and 3 attention blocks. With the same parameters, do 6 blocks at
  width 344 beat 3 at 448?
answer: >-
  Not run yet. The check before it: imm's held-out excess over the
  transformer at epoch 8 (+0.0196 nats per character) is +0.0118 from a
  higher training loss and +0.0077 from a larger train-to-held-out gap, so
  imm fits the training text less well more than it generalizes worse.
opened:   2026-11-22
updated:  2026-11-22
code: >-
  branch depth-width 6ce0302: depth-records 5ae5c64 (slot 137's code) plus
  --stop-after-epoch, as 0ff2938 added it on attention-only.
data:     PG-19 100M split (rnd/README.md), configs/pg19split100m_noprior.json
slots:    []
tags:     [depth, width, depth-records, scale, no-prior, screen]
related:  [depth-records-10.8m, scale-10.8m, blocks-20m, half-lr-10.8m, reading-early]
---

# Depth against width at 10.8M

## Question

imm leads the transformer at about 1.6M parameters (depth records, −5.50%
against the 1.84M) and trails it at 10.8M (+1.99% ±0.13,
[depth-records-10.8m](../depth-records-10.8m/)). From 1.59M to 10.8M imm
gains 7.7%, the transformer 13.8%. At 10.8M the gap is on new words
(+3.8%) and lowercase letters (+2.7%); imm still leads in a piece's first
256 characters (−6.2%) and on repeats 64–256 words back (−1.7%). So the
records do their job, and the plain prediction is what falls behind.

The transformer is 6 layers of width 384. imm is one GRU of width 448
feeding 3 attention blocks of the same width. In imm almost all the
parameters past the GRU are in the blocks, so the question is how to spend
them: on width or on depth.

The earlier block-count results don't settle it. On the 20M split, with
the books prior, at width 256, a second block gave 0.40% and a third
0.03% ([blocks-20m](../blocks-20m/)). That was before character records
and before depth records. With depth records each block past the first
reads records written from the block before it, so a block's depth now
reaches the keys and values too, not only the queries.

### The check before it: does imm fit less or generalize worse?

`fit_gap.py` (`results/fit_gap_10.8m.txt`) compares, per epoch, the
training loss with the held-out loss for slot 137 and the 10.8M
transformer at dropout 0, in nats per character.

| epoch 8 | training | held-out | gap |
|---|---:|---:|---:|
| imm, slot 137 | 0.9205 | 1.0298 | +0.1093 |
| transformer, 10.8M | 0.9086 | 1.0102 | +0.1016 |

- imm's held-out excess, +0.0196 nats, is +0.0118 from a higher training
  loss and +0.0077 from a larger gap: about 60% fitting less, 40%
  generalizing worse. Both parts carry the transformer's ±0.007 (below),
  so the split is rough.
- Both gaps grow every epoch from epoch 2, as the 100M characters repeat.
- The transformer at dropout 0.1 scores 2.7877 against 2.7458 at 0
  (+1.5%), so regularization hurt it at this size and amount of data.
- So imm doesn't overfit the way a dropout run would cure; mostly, with
  the same parameters, it fits the training text less well. That points
  at how the parameters are spent.
- The two training losses aren't measured the same way. imm's is the mean
  over every training character of the epoch, with its carried state. The
  transformer's is one batch (16 × 1,024 characters) logged every 500
  steps, 12 or 13 an eighth of the run. At epoch 8 the 12 batches have a
  standard deviation of 0.023, so their mean's standard error is 0.007.

## Design

- **Base:** slot 137's 10.8M imm. GRU and embedding 448, 3 blocks of 8
  heads, word tier 392, 128 word records and 1,024 character records,
  `--depth-records`, no prior, dropout 0, learning rate 1e-3 cosine to 1e-4
  over 8 epochs, trained in bf16, scored in fp32. 10,913,495 trained
  parameters. Epoch 2 3.0100, epoch 3 2.9342, epoch 8 2.8004.
- **The arm:** the same, at GRU and embedding 344, 6 blocks of 8 heads,
  word tier 301 (392/448 of the width, as in the base). 10,730,974 trained
  parameters, 1.7% fewer than the base (counted by one CPU update with
  `$S/s10/size.py`).
  - Widths that keep 8 heads and the tier's ratio: 336 gives 10.24M
    (−6.2%), 352 gives 11.23M (+2.9%). 344 is the closest, with 43
    dimensions a head. The rotary positions turn 42 of them, which the
    code handles.
  - The GRU narrows with the blocks, since one `--hidden-size` sets both.
    So the arm tests depth bought with width, everywhere: the GRU's
    included.
- **A screen:** the 8-epoch schedule stopped after epoch 3, read against
  slot 137's epoch 3 ([reading-early](../reading-early/): gaps of about 1%
  or more at epoch 3 kept their sign; smaller ones moved by up to 0.5%).
  - A gain of 1% or more (2.905 or lower) earns a full run, at the user's
    say.
  - Within 1%, a full run is needed to tell.
  - The epoch 2 → 3 trend is read too.
- **GPU memory:** every block writes its own records, so 6 blocks need
  more memory than 3. `probe_depth_width.py` finds the largest update cap
  under 6.9 GiB, as `probe_depth.py` did for slot 137 (64 × 352), searching
  down to 64 × 128, and times an update there.
- One seed, the default, as in slot 137.

## How to rerun

From a worktree at `rnd/depth-width-10.8m/slot<N>` (rnd/README.md), with
`$S/dw/` holding a copy of `probe_depth_width.py`:

- `dw_slot.sh <slot>`: waits for the slot, runs the probe, then the
  screen at the cap the probe chose, then the epoch-3 checkpoint's
  per-character losses in fp32.
- The probe reads `../scale-10.8m/results/worst_window.txt`, the training text's
  longest 64-word window (the script's path is the original
  `$S/s10/worst_window.txt`).
- `fit_gap.py`: the check before the screen, from slot 137's CSV and the
  transformer's log and JSON.

**Results files** (`results/`): `fit_gap_10.8m.txt`.

**Checkpoints and per-character losses**, in `runs/`:
`pg19s100m_h344_par_l6_w301_own_np_doc_mm128_do0_cm1024_depth_bf16_e8s3`
(`.log`, `.csv`, `.pt`, `.best.pt`) and its `char_losses_…_best.npz`.

## Results

None yet.

## What we learned, or what's open

- Open: the screen.
- Not tested here: depth at a fixed GRU width (it needs the blocks'
  width set apart from the GRU's), or 4 blocks between the two.
