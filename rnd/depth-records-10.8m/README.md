---
title:    Depth records at 10.8M
kind:     experiment
status:   concluded
outcome:  positive
question: >-
  Depth records gave −1.45% at 1.59M with no new parameters. Does the gain
  hold at 10.8M, and how much of the gap to the 10.8M transformer does it
  close?
answer: >-
  Less than half of it holds. 2.8004, −0.65% ±0.11 against slot 126's
  2.8188, ahead at every epoch and in every bucket. Against the 10.8M
  transformer (2.7458) imm is now +1.99% ±0.13, down from +2.66%.
opened:   2026-11-20
updated:  2026-11-21
code: >-
  branch depth-records 5ae5c64 (tag rnd/depth-records-10.8m/slot137). The
  commit also adds --no-word-records, unused here; the default path is
  unchanged.
data:     PG-19 100M split (rnd/README.md), configs/pg19split100m_noprior.json
slots:    137
tags:     [depth-records, scale, no-prior]
related:  [architecture-1.59m, scale-10.8m, half-lr-10.8m]
---

# Depth records at 10.8M

## Question

At 1.59M, depth records (each block past the first reads records rebuilt
from the block before it) gave −1.45% ±0.10 with no new parameters
([architecture-1.59m](../architecture-1.59m/)). The lead agent had told
the user the 10.8M run would follow if the gain was clear.

## Design

- Slot 126's 10.8M imm ([scale-10.8m](../scale-10.8m/)): GRU and
  embedding 448, 3 blocks of 8 heads, word tier 392, 128 word records and
  1,024 character records, no prior, dropout 0, lr 1e-3 cosine to 1e-4, 8
  epochs, trained in bf16 and scored in fp32. Plus `--depth-records`.
- Every block past the first writes its own records, so an update needs
  more GPU memory. `probe_depth.py` found the largest update cap under
  6.9 GiB with the training text on the GPU: 22,528 (64 × 352), against
  slot 126's 64 × 576. Every update splits; 424 ms an update.
- Slot 137 ran 2026-11-20 18:03–22:26 UTC. Training took 15,707 s, 1.07×
  slot 126's 14,642 s.

## How to rerun

From a worktree at `rnd/depth-records-10.8m/slot137` (rnd/README.md):

- `d10_slot.sh`: `probe_depth.py`, then the run at the cap the probe
  chose, then its fp32 per-character losses.
- `probe_depth.py` reads `../scale-10.8m/worst_window.txt`, the training
  text's longest 64-word window (the script's path is the original
  `$S/s10/worst_window.txt`).
- Pairings: `scripts/rnd/rpair.sh A.npz B.npz OUT.txt`.

**Results files** (`results/`): `depth_vs_rel10.8m.txt` (against slot
126), `depth_vs_tx10.8m.txt` (against the 10.8M transformer),
`slot_137.log`, `probe_depth_137.log`, `rss_137.log`.

**Checkpoints and per-character losses**, in `runs/`:
`pg19s100m_h448_par_l3_w392_own_np_doc_mm128_do0_cm1024_depth_bf16_e8`
(`.log`, and `…_e8_pg19s100m_…_e8.csv`, `.pt`, `.best.pt`) and
`char_losses_pg19s100m_h448_par_l3_w392_own_np_doc_mm128_do0_cm1024_depth_bf16_e8_best.npz`.

## Results

### 2026-11-20: slot 137

Percentages are A's change in held-out perplexity per character against B
(`rpair.sh`), ± one standard error.

- **2.8004, −0.65% ±0.11 against slot 126's 2.8188.** That is under half
  the −1.45% at 1.59M.
  - Ahead at every epoch: 3.174 / 3.010 / 2.934 / 2.887 / 2.850 / 2.821 /
    2.806 / 2.800 against 3.182 / 3.024 / 2.951 / 2.905 / 2.868 / 2.841 /
    2.823 / 2.819.
  - Ahead in every bucket. The most on uppercase letters (−2.9%), after a
    word (−1.2%) and on repeats 1–16 words back (−1.2%); the least inside
    words (−0.5%).
- **Against the 10.8M transformer (2.7458): +1.99% ±0.13**, down from slot
  126's +2.66%.
  - The gap is still on new words (+3.8%) and lowercase letters (+2.7%).
  - imm leads in a piece's first 256 characters (−6.2%) and on repeats
    64–256 words back (−1.7%).
  - It trails on repeats more than 256 words back (+4.0%), past its
    records' reach.

## What we learned

- Depth records help at 10.8M too, but about half as much as at 1.59M;
  imm still trails the transformer by ~2%.
- **One seed.** Never tested: depth records with a second seed, or at
  10.8M with other learning rates (the half-rate run,
  [half-lr-10.8m](../half-lr-10.8m/), used slot 126's model without them).
