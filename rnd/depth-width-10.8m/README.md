---
title:    Depth against width at 10.8M
kind:     experiment
status:   parked
outcome:  n/a
question: >-
  At 10.8M imm trails the transformer by 1.99%, and the gap is in plain
  prediction (new words, lowercase letters), not in what the records copy.
  The transformer spends its parameters on 6 layers of 384; imm on one GRU
  of 448 and 3 attention blocks. With the same parameters, do 6 blocks at
  width 344 beat 3 at 448?
answer: >-
  Not run: 6 blocks at 344 fit this GPU only at a 64 x 96 cap, at 3.2x the
  base's time an update (3 epochs ~5.2 h), and scoring doesn't fit (slot
  165). The arm waits for a decision. The check before it: imm's held-out excess over the
  transformer at epoch 8 (+0.0196 nats per character) is +0.0118 from a
  higher training loss and +0.0077 from a larger train-to-held-out gap, so
  imm fits the training text less well more than it generalizes worse.
opened:   2026-11-22
updated:  2026-11-22
code: >-
  branch depth-width 6ce0302: depth-records 5ae5c64 (slot 137's code) plus
  --stop-after-epoch, as 0ff2938 added it on attention-only.
data:     PG-19 100M split (rnd/README.md), configs/pg19split100m_noprior.json
slots:    [162, 164, 165]
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

**Results files** (`results/`): `fit_gap_10.8m.txt`; `probe_162.log`,
`probe_164.log`, `probe_165.log` and the slots' logs; `saved_bytes.txt`.

**Checkpoints and per-character losses**, in `runs/`:
`pg19s100m_h344_par_l6_w301_own_np_doc_mm128_do0_cm1024_depth_bf16_e8s3`
(`.log`, `.csv`, `.pt`, `.best.pt`) and its `char_losses_…_best.npz`.

## Results

### 2026-11-22: slot 162, out of GPU memory in the probe's warm-up

The slot opened at 09:34 UTC. The probe ran out of GPU memory in its first
warm-up updates at 64 × 288, before the cap search began. The GPU was
otherwise empty (62 MiB in use), and the slot ended at once. Nothing
trained.

- **Why:** most of an update's memory is attention scores, heads × records
  a character in every block, which a narrower width doesn't shrink. So 6
  blocks need about twice the memory a character that 3 blocks do. Slot
  137's 3 blocks already peaked at 6.78 GiB at 64 × 352.
- **Fixed in the scripts** (the versions that ran in slot 162 are in git
  at `96e059f`):
  - The probe warms up at 64 × 96 and searches down to 64 × 64.
  - The cap only splits an update into groups of streams, so it changes
    the speed, not the result.
  - `dw_slot.sh` stops if the probe puts the 3 epochs over 4 hours. A
    screen that slow waits for the user.

### 2026-11-22: slot 164, the probe's cap search crashed

The slot opened at 10:46 UTC. Nothing trained; the slot ended after 11
seconds.

- **The warm-up fit:** at 64 × 96 it peaked at 5.94 GiB, leaving about 1
  GiB under the limit.
- **Unsplit updates didn't:** every one from 64 × 576 down to 64 × 256 ran
  out of GPU memory.
- **Then the probe crashed.** It tests a cap of 64 × L with 64 streams that
  are each a 64-segment window of at most L characters. 64 words don't fit
  in under about 250 characters, so below 64 × 256 there was no window to
  test with.
- **Fixed** (the slot-164 version is in git at `0fed229`): a cap is now
  tested the way the run uses it. Three updates of the text and the worst
  update are each split at the cap, from 64 × 320 down to 64 × 64. The
  warm-up is now 64 × 64. On the CPU, with a limit nothing meets, the
  search runs to 64 × 64 and reports no cap.

### 2026-11-22: slot 165, 6 blocks at 344 don't fit at a usable speed

The slot opened at 10:49 UTC. The probe ran and found no workable cap, so
nothing trained (`results/probe_165.log`).

- **The cap search:**
  - The warm-up at 64 × 64 peaked at 4.85 GiB.
  - Every cap from 64 × 320 down to 64 × 128 ran out of GPU memory.
  - 64 × 96 passed: three updates and the worst split into 5, 5, 5 and 10
    groups, peaking at 6.67 GiB.
- **The speed there: 1,361 ms an update**, 3.2× slot 137's 424 ms.
  - That puts 3 epochs at about 5.2 hours and 8 at about 14.
  - The script's ceiling is 4 hours for the screen.
- **It doesn't run at all, either.** After the timed updates, the worst
  update at that cap ran out of memory, and so did scoring's worst case (64
  streams of the longest window, unsplit, no gradient).
- **Memory per character, compared:**
  - On the GPU, 3 blocks at 448 added 0.26 GiB per 2,048 more characters
    an update (slot 137's probe, unsplit updates).
  - 6 blocks at 344 added about 1.09 GiB going from a cap of 64 × 64 to
    64 × 96 (slots 164 and 165, split updates). That's about 4×, though
    the two slopes aren't measured the same way.
- **What autograd saves doesn't explain it** (`saved_bytes.py`,
  `results/saved_bytes.txt`; CPU, fp32, one unsplit update of 4 streams):
  - 3 × 448 with depth records saves 322 KiB a character, 6 × 344 saves
    471: 1.46× as much.
  - Depth records add 17–23% at either depth.
  - So most of the extra GPU memory is something that splitting the
    update doesn't reduce. It isn't found yet.

## What we learned, or what's open

- **As built, 6 blocks at 344 can't be screened on this GPU** (8 GB, 6.9
  GiB usable): it fits only at 64 × 96, there it runs at 3.2× the base's
  time an update, and scoring doesn't fit.
- **Waiting for the user.** The ways on:
  - **Depth at 1.59M,** where memory is ample: slot 127's 2 blocks at 192
    against 4 blocks at 152 (1,561,962 trained parameters, −1.6%, about
    the 10.8M arm's −1.7%; `$S/dw/size_depth.py`). A screen would take
    under an hour. It tells whether depth beats width in imm at all, at
    the smaller size.
  - **Find the memory the cap doesn't split,** then try 6 blocks at 10.8M
    again. It would still be slower per update than the base.
  - **4 blocks at 10.8M**, at a width that keeps 10.9M parameters: less
    memory than 6, still a test of depth.
- Not tested here: depth at a fixed GRU width (it needs the blocks'
  width set apart from the GRU's), or 4 blocks between the two.
