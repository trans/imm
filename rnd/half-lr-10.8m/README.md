---
title:    Half the learning rate at 10.8M
kind:     experiment
status:   concluded
outcome:  negative
question: >-
  imm's learning rate (1e-3 cosine to 1e-4) was set at 1.59M. Is the 10.8M
  imm held back by it?
answer: >-
  No. At 5e-4 cosine to 5e-5 it scores 2.8230, +0.15% ±0.11 against slot
  126's 2.8188 at 1e-3: no gain, inside seed noise. The lower rate starts
  7.53 points behind after epoch 1, closes to within 0.2 by epoch 4 and
  holds; the epoch-8 training loss is level. 1e-3 stays. The transformer at
  5e-4 was to be proposed only if imm gained, so it wasn't.
opened:   2026-11-20
updated:  2026-11-21
code: >-
  branch depth-records 5ae5c64 (tag rnd/half-lr-10.8m/slot145). The run
  uses none of that branch's new flags; its default path is unchanged from
  slot 126's 00b6a52 (the CPU smoke gives 15.409 on both).
data:     PG-19 100M split (rnd/README.md), configs/pg19split100m_noprior.json
slots:    145 (queued before as 138, then 141; neither ran)
tags:     [learning-rate, scale, no-prior]
related:  [scale-10.8m, depth-records-10.8m, reading-early]
---

# Half the learning rate at 10.8M

## Question

At 10.8M imm trails the transformer by 2.66% per character, and its
dropout and learning rate were never tuned at that size
([scale-10.8m](../scale-10.8m/)). In air's session the user asked for the
proposed runs to be queued (2026-11-20); this is the learning-rate one.

## Design

- Slot 126's 10.8M imm as it was (relative positions, no depth records):
  GRU and embedding 448, 3 blocks of 8 heads, word tier 392, 128 word
  records and 1,024 character records, no prior, dropout 0, 8 epochs,
  trained in bf16 at slot 126's cap of 36,864 characters an update, scored
  in fp32.
- One change: `--lr 5e-4 --min-lr 5e-5`, half of 1e-3 → 1e-4.
- If it gained, the transformer would need the same test before reading
  the gap as closed. That run wasn't approved in advance.
- **Queueing:** first queued as slot 138 on 2026-11-20. The simplified run
  (slot 139) was queued ahead of it. It was re-queued as slot 141 at 22:41
  UTC, behind a shorter run of another project. The user then pointed out
  that ~11 hours of imm runs were queued, keeping the GPU from others, and
  asked for this long run to go later. Slot 141 was dropped at 23:12 UTC,
  and a small wrapper queued the run again on 2026-11-21 at 09:00 UTC as
  slot 145.
- Slot 145 ran 2026-11-21 09:00–13:07 UTC. Training took 14,724 s.

## How to rerun

From a worktree at `rnd/half-lr-10.8m/slot145` (rnd/README.md):

- `lr10_slot.sh`: the run, then its fp32 per-character losses.
- The wrapper that waited until the set time, queued the slot and then ran
  `lr10_slot.sh` isn't copied; it did nothing else.
- Pairing: `scripts/rnd/rpair.sh A.npz B.npz OUT.txt`.
- Comments that quoted people word for word are paraphrased in the copied
  script. No command line changed.

**Results files** (`results/`): `lr5e-4_vs_rel10.8m.txt` (against slot
126), `slot_145.log`, `rss_145.log`, and the logs of the dropped slots
(`slot_138.log`, `slot_141.log`: each script found its slot gone and
stopped).

**Checkpoints and per-character losses**, in `runs/`:
`pg19s100m_h448_par_l3_w392_own_np_doc_mm128_do0_cm1024_rel_lr5e-4_bf16_e8`
(`.log`, and `…_e8_pg19s100m_…_e8.csv`, `.pt`, `.best.pt`) and
`char_losses_pg19s100m_h448_par_l3_w392_own_np_doc_mm128_do0_cm1024_rel_lr5e-4_bf16_e8_best.npz`.

## Results

### 2026-11-21: slot 145

- **2.8230, +0.15% ±0.11 against slot 126's 2.8188** (the change in
  held-out perplexity per character, ± one standard error): no gain,
  inside seed noise.
- **The gap closes and then holds.** In points (100 × the difference in
  mean held-out loss, nats per character), epochs 1 to 8: +7.53 / +1.20 /
  +0.45 / +0.20 / +0.10 / +0.02 / +0.16 / +0.15. The epoch-8 training loss
  ends level (−0.04 points), so the lower rate neither under-trains nor
  overfits less.
- It gains a little at word starts (−0.6%) and on new words (−0.6%), and
  loses on punctuation (+5.9%) and capitals (+2.1%).
- Held-out by epoch, from the run's CSV: 3.431 / 3.060 / 2.964 / 2.910 /
  2.871 / 2.842 / 2.828 / 2.823.

## What we learned

- 1e-3 stays imm's learning rate at 10.8M.
- A learning-rate change shows only late: at epoch 3 this run was still
  0.45 points behind and closing. It is the kind of change
  [reading-early](../reading-early/) says needs a full run.
- **One seed.** Never tested: a higher rate, imm's dropout at 10.8M, and
  the transformer at 5e-4.
