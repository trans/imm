---
title:    Reading a run early, the epoch-3 screen
kind:     design
status:   concluded
outcome:  reference
question: >-
  How early in the 8-epoch schedule can a pair of runs be read, and which
  gaps can be trusted that early?
answer: >-
  At epoch 3, for architecture changes. In eight past pairs every gap of
  about 1 or more (in percent of the base's held-out loss) at epoch 3 kept
  its sign to epoch 8; among the architecture pairs its size was off by up
  to 0.66. Smaller gaps moved by up to ~0.5 and one flipped sign, so a full
  run decides those. Epoch 1 misleads. Dropout's effect grows to the end,
  so regularisation, and likely the learning rate, needs full runs. The
  user chose epoch 3 (2026-11-20).
opened:   2026-11-20
updated:  2026-11-21
code:     none; read from the runs' per-epoch logs in runs/ (no slot, no tag)
data:     PG-19 100M split (rnd/README.md)
slots:    none
tags:     [method, screening, evaluation]
related:  [attention-only-words, architecture-1.59m, simplified-no-words, char-records, no-prior-pair-1.8m, depth-records-10.8m, half-lr-10.8m]
---

# Reading a run early, the epoch-3 screen

## Question

By 2026-11-20 about 11 hours of imm runs were queued, and the GPU is
shared. The user asked for a way to test imm's ideas without such long
runs. The lead agent proposed reading screens at epoch 2 of the usual 8;
the user chose epoch 3, for a little more margin.

## Design

Every 8-epoch run on the 100M split logs its held-out score after each
epoch. For pairs that had already run to the end, compare the gap at
epochs 1, 2, 3 and 4 with the final one at epoch 8. All runs use the
cosine schedule from 1e-3 to 1e-4 over 8 epochs.

The pairs (run ids in `runs/`, A against B):

| pair | A | B |
|---|---|---|
| depth records vs slot 127 (1.59M) | `…_cm1024_depth_bf16_e8` (slot 130) | `…_cm1024_rel_bf16_e8` (slot 127) |
| char-kv vs slot 127 | `…_cm1024_charkv_bf16_e8` (slot 131) | slot 127 |
| no attention vs slot 127 | `…_noattn_bf16_e8` (slot 130) | slot 127 |
| 256 vs 1,024 character records | `…_do0_cm256_e8` (slot 113) | `…_do0_cm1024_e8` (slot 115) |
| no character records vs 1,024 | `…_do0_e8` (slot 107) | slot 115 |
| dropout 0.2 vs 0 | `…_mm128_e8` (slot 101) | slot 107 |
| depth records vs slot 126 (10.8M) | `pg19s100m_h448_…_depth_bf16_e8` (slot 137) | `pg19s100m_h448_…_rel_bf16_e8` (slot 126) |
| simplified vs slot 127 | `pg19s100m_h224_e160_…_nowords_bf16_e8` (slot 139) | slot 127 |

`…` is `pg19s100m_h192_par_l2_w168_own_np_doc_mm128` except where the id is
given whole. All but the last pair differ in one change.

## How to rerun

No script survives from the original reading. `gaps_by_epoch.awk`, written
for this record, recomputes the table from the runs' CSVs:

```sh
cd runs
awk -f ../rnd/reading-early/gaps_by_epoch.awk A_A.csv B_B.csv
```

where `A_A.csv` is `<id>_<id>.csv`. It prints the gap three ways: in
points (100 × the difference in mean held-out loss, nats per character),
in this table's unit (below) and as a change in perplexity.

## Results

### 2026-11-20: the gaps by epoch

The unit is the percent change in the mean held-out loss (nats per
character), 100 × (ln ppl_A / ln ppl_B − 1), from the 3-decimal
perplexities in the logs. (The notes call it "the gap in nats" and
"points"; see the 2026-11-21 section.)

| pair | e1 | e2 | e3 | e4 | e8 |
|---|---:|---:|---:|---:|---:|
| depth records vs slot 127 (1.59M) | −0.71 | −1.63 | −1.46 | −1.30 | −1.30 |
| char-kv vs slot 127 | −1.69 | −0.65 | −0.66 | −0.44 | −0.29 |
| no attention vs slot 127 | +1.52 | +6.79 | +7.54 | +7.77 | +8.20 |
| 256 vs 1,024 character records | +0.65 | +3.53 | +3.48 | +3.36 | +3.13 |
| no character records vs 1,024 | +0.94 | +5.92 | +6.58 | +6.72 | +6.87 |
| dropout 0.2 vs 0 | +0.80 | +1.60 | +1.79 | +2.03 | +2.69 |
| depth records vs slot 126 (10.8M) | −0.22 | −0.42 | −0.53 | −0.58 | −0.65 |
| simplified (slot 139) vs slot 127 | −3.83 | −0.47 | −0.08 | +0.08 | +0.44 |

- **Epoch 1 misleads.** Changes that speed early learning look big
  (char-kv −1.69 against −0.29 at the end), and changes that cost capacity
  look small (no attention +1.52 against +8.20).
- **At epoch 3, a screen:**
  - every gap of 1 or more has the final sign; its size is off by up to
    0.66 (no attention, 8% of it);
  - smaller gaps move by up to ~0.5 to the end: char-kv −0.66 → −0.29, and
    the simplified model −0.08 → +0.44, whose sign flipped;
  - so a screen tells effects of about 1 or more; under that, a full run
    decides;
  - a gap still moving between epochs 2 and 3 keeps moving the same way
    (the simplified model +0.39, then +0.52 more to the end), so read the
    trend as well.
- **Dropout grows to the end** (+1.60 → +2.69), so regularisation needs
  the full run; the learning rate, untested here, likely does too.
- One seed each; seed noise is ~0.37 (slot 127 against slot 115, a
  perplexity change).
- Epoch 3 is 3/8 of the run: ~33 min at 1.59M, ~1.6 h at 10.8M. The base's
  epoch-3 score is already in its log, so a screen needs no rerun of the
  base.
- **A correction made the same day:** the first version of this reading
  said that from epoch 2 every pair was within ~0.4 of its final size. That
  was wrong for the large effects: no attention was off by 1.41 at epoch 2,
  no character records by 0.95.

### 2026-11-21: notes on the units

Recomputed from the CSVs with `gaps_by_epoch.awk`, every number in the
table above is reproduced as 100 × (ln ppl_A / ln ppl_B − 1). The notes'
labels for it don't match:

- The notes call the table "the gap in nats" and its bullets speak of
  "points". In [attention-only-words](../attention-only-words/) and
  [half-lr-10.8m](../half-lr-10.8m/), a point is 100 × the difference in
  mean loss in nats, which is this table's number times the base's mean
  loss (1.04–1.32 nats over these pairs and epochs). For example, depth
  records vs slot 127 at epoch 8 is −1.30 here, −1.46 points, and −1.45%
  in perplexity (`rpair.sh`).
- So at epoch 3, where the bases' losses are 1.08–1.24 nats, this
  screen's line of 1 is about 1.1–1.25 points. The perplexity change is
  close to the points.
- The same pairs in points (100 × the difference in `val_nll`):

  | pair | e1 | e2 | e3 | e4 | e8 |
  |---|---:|---:|---:|---:|---:|
  | depth records vs slot 127 | −0.95 | −1.96 | −1.71 | −1.52 | −1.46 |
  | char-kv vs slot 127 | −2.25 | −0.78 | −0.77 | −0.52 | −0.34 |
  | no attention vs slot 127 | +2.00 | +8.19 | +8.85 | +8.93 | +9.13 |
  | 256 vs 1,024 character records | +0.85 | +4.22 | +4.07 | +3.87 | +3.48 |
  | no character records vs 1,024 | +1.23 | +7.07 | +7.68 | +7.72 | +7.66 |
  | dropout 0.2 vs 0 | +1.07 | +2.04 | +2.21 | +2.50 | +3.19 |
  | depth records vs slot 126 | −0.26 | −0.47 | −0.55 | −0.60 | −0.66 |
  | simplified vs slot 127 | −5.08 | −0.57 | −0.08 | +0.09 | +0.49 |

  The conclusions hold in points: every epoch-3 gap of 1 point or more
  keeps its sign; the char-kv and simplified pairs still move by 0.43 and
  0.57.
- "Off by up to 0.66" holds for the architecture pairs. The dropout pair
  moves more, 0.90 (+1.79 → +2.69), which is why regularisation needs the
  full run.

## What we learned

- Architecture ideas are screened at epoch 3 of the 8-epoch schedule
  against the base's own epoch-3 score. A gap of about 1% or more is
  trusted; a smaller one needs a full run.
- Learning rate and dropout aren't screened this way.
- **Open:** the spread between seeds at epoch 3 was never measured; the
  ~0.37 is an 8-epoch figure between two code versions.
