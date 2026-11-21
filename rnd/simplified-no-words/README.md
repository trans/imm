---
title:    The simplified imm, character records only
kind:     experiment
status:   concluded
outcome:  negative
question: >-
  Does imm need its word level? Without word records or the word tier,
  only the character GRU and attention over the last 1,024 characters'
  records, at slot 127's trained size, is imm as good?
answer: >-
  Not quite. 3.0705, +0.50% ±0.11 against slot 127's 3.0552 (+1.98%
  against depth records' 3.0109; −3.63% against the 1.84M transformer). It
  loses at word and piece starts and gains a little on repeats. So the word
  level (tier and records together) is worth about 0.5% at 1.59M without
  the prior. As it wasn't level, its 10.8M run wasn't proposed.
opened:   2026-11-20
updated:  2026-11-21
code: >-
  branch depth-records 5ae5c64. Slot 136 (GRU 216, out of GPU memory in its
  first update; tag rnd/simplified-no-words/slot136), slot 139 (GRU 224;
  tag rnd/simplified-no-words/slot139).
data:     PG-19 100M split (rnd/README.md), configs/pg19split100m_noprior.json
slots:    136, 139
tags:     [simplification, word-records, word-tier, ablation, no-prior]
related:  [architecture-1.59m, attention-only-words, reading-early, gru-tree]
---

# The simplified imm, character records only

## Question

The user, via air, suggested imm might need simplifying; this design is
air's. In air's session the user asked for it and the other proposed runs
to be queued (2026-11-20). The question is
whether imm's word level earns its place once the attention can read the
last 1,024 characters' records. The user's condition, via air: the 10.8M
run follows only if the simplified model is level.

## Design

- Slot 127's model ([rope-far-positions](../rope-far-positions/)) with
  **`--no-word-records`** (`5ae5c64`): the attention blocks read the
  character records only. No `--word-tier`, so no word tier and no
  `--tier-own-step`.
- Otherwise slot 127's settings: no prior, dropout 0, 2 blocks of 4 heads,
  1,024 character records, 8 epochs on the 100M split, trained in bf16 and
  scored in fp32.
- Widened to slot 127's trained size: GRU 224, embedding 160 (head width
  56), 1,593,988 trained parameters by `size_nw.py`, against slot 127's
  1,587,797 by the same count.
- **Slot 136** (2026-11-20 18:01–18:02 UTC, `nw_slot.sh`) at GRU 216,
  embedding 256 (1,581,388 trained) ran out of GPU memory in its first
  update. 216 over 4 heads is a head width of 54. The fused attention
  kernel needs a head width that's a multiple of 8 in bf16, so attention
  fell back to storing the full weights.
- **Slot 139** (22:26–23:48 UTC, `nw2_slot.sh`) at GRU 224, embedding 160.

## How to rerun

From a worktree at the slot's tag (rnd/README.md):

| script | slot | what it ran |
|---|---|---|
| `nw_slot.sh` | 136 | GRU 216, embedding 256 (out of GPU memory) |
| `nw2_slot.sh` | 139 | GRU 224, embedding 160, 8 epochs bf16, then fp32 per-character losses |
| `size_nw.py` | none | trained parameters of a config, with or without word records (one update on the CPU) |

- Pairings: `scripts/rnd/rpair.sh A.npz B.npz OUT.txt`.
- Comments that quoted people word for word are paraphrased in the copied
  scripts. No command line changed.

**Results files** (`results/`): `nowords_vs_rel1.59m.txt` (against slot
127), `nowords_vs_depth1.59m.txt` (against depth records),
`nowords_vs_tx1.84m.txt` (against the 1.84M block-1,024 transformer), the
slot logs (`slot_136.log`, `slot_139.log`) and memory samples
(`rss_136.log`, `rss_139.log`).

**Checkpoints and per-character losses**, in `runs/`:

- slot 139: `pg19s100m_h224_e160_par_l2_np_doc_do0_cm1024_nowords_bf16_e8`
  (`.log`, `…_e8_pg19s100m_…_e8.csv`, `.pt`, `.best.pt`) and
  `char_losses_pg19s100m_h224_e160_par_l2_np_doc_do0_cm1024_nowords_bf16_e8_best.npz`;
- slot 136: `pg19s100m_h216_e256_par_l2_np_doc_do0_cm1024_nowords_bf16_e8.log`
  (with its `.csv` and the untrained `.best.pt`).

## Results

### 2026-11-20: slot 139

Percentages are A's change in held-out perplexity per character against B
(`rpair.sh`), ± one standard error.

- **3.0705, +0.50% ±0.11 against slot 127's 3.0552**; +1.98% against depth
  records (3.0109); −3.63% against the 1.84M transformer (3.1862).
- **It loses at word and piece starts:** predicting a word's first
  character +1.49%; uppercase letters +4.74%; a piece's first 256
  characters +4.89%; new words +0.53%.
- **It gains on repeats:** words seen before −0.66%, 1–16 words back
  −0.97%. Its GRU is wider (224 against 192).
- So the word level (tier and records together) is worth about 0.5% at
  1.59M without the prior, at boundaries, as slot 72 found for the word
  tier with the prior (+0.57% ±0.06). It isn't level, so its 10.8M run
  isn't proposed.
- **It led early, then fell behind:** 3.563 / 3.321 / 3.230 / 3.172 /
  3.135 / 3.104 / 3.081 / 3.070 against slot 127's 3.748 / 3.340 / 3.233 /
  3.169 / 3.126 / 3.093 / 3.067 / 3.055. This pair is the one whose sign
  flipped after epoch 3 in [reading-early](../reading-early/).

### 2026-11-21: a note on the record

Counted from the best checkpoint's optimizer state
(`../no-prior-pair-1.8m/used_params.py`), slot 139 trains 1,594,212
parameters and slot 127 1,588,157. `size_nw.py`'s one-update count gives
224 and 360 fewer.

## What we learned

- The word level is worth only ~0.5% at 1.59M once the attention reads
  1,024 characters' records. The lead agent's reading, untested: word
  summaries add reach, not information the character states lack, and the
  128 word records (~740 characters) sit inside the 1,024-character
  window.
- After this result the user chose to step back to simpler designs
  (2026-11-21); the first is [gru-tree](../gru-tree/).
- **One seed.** Not measured: the word tier alone without the prior. This
  run drops the word tier and the word records together.
