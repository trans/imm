---
title:    "What the memory and the word GRU are worth: knockouts and retraining"
kind:     experiment
status:   concluded
outcome:  mixed
question: >-
  How much does imm owe its memory (the attention over word records) and,
  later, its word-level GRU? Is switching a trained branch off a fair
  measure, or must the model be retrained without it?
answer: >-
  Switching off measures reliance, retraining measures worth, and they
  differ a lot. On the 20M d16 model, memory off retrained costs 2.1% at
  the same GRU (3.2172 against 3.1517) and 1.0% at the same size (3.1821),
  while zeroing the trained attention costs 6.2%. On slot 63's model
  (2.7757, d24 prior) zeroing the word GRU costs 12.3% and the attention
  1.3%, but retrained without them the model loses only 0.57% (2.7915) and
  0.52% (2.7903). Each is worth about half a percent.
opened:   2026-11-13
updated:  2026-11-17
code: >-
  memory off 3c8ca62 (branch memory-off); gate and mean ablations 787ee25
  (branch parallel-memory); slot 63 knockouts ccacad2 (branch
  drop-row-cache); those branches are deleted and the commits are on main.
  No attention, retrained: 3989c87 on branch memory-attention-off (kept,
  not merged into main).
data: >-
  PG-19 20M split, data/pg19/imm-split-20m (250,000 held-out characters).
  Slots 22–29: mobius's pg19split20m_gated_d16_pg200m prior (alone 1.8205
  bits/char) with pg19split20m_kn evidence. Slots 69–72: slot 63's model
  on mobius's pg19split20m_gated_d24_pg10g prior (alone PPL 3.068).
slots:    [22, 28, 29, 69, 71, 72]
tags:     [ablation, memory, attention, word-tier, reliance]
related:  [parallel-memory, blocks-20m, word-tier, imm-1.85m, memory-window]
---

# What the memory and the word GRU are worth: knockouts and retraining

## Question

The user asked (2026-11-13, in air's session) what happens if the memory
records are taken out. Later (2026-11-17) the user asked how much each
part contributes: the word GRU or the word-level attention memory.
Two ways to measure: switch a branch off in a trained model, or train a
model without it.

## Design

- **Memory off, retrained** (slot 22, `3c8ca62`, `--mixing none`): no
  records, no cross-attention, the head on the GRU state, the prior and its
  input as before. h256 (895,147 trained parameters, the same GRU) and h432
  (1,964,347, the base's size). Every other flag as the 20M base run
  (`blocks-20m`). Caveat found after the runs: this removed the whole gated
  block, its MLP too, not only the attention.
- **Gate zero** (slot 28, `787ee25`): the base and the parallel model
  (`parallel-memory`) rescored with `gate_attn` = 0 (attention off, MLP
  kept) or `gate_mlp` = 0. Nothing retrained.
- **Mean instead of zero** (slot 29): the attention output replaced by its
  mean over held-out positions that see a record, or by its mean at the
  character's position in its word (`attn_mean.py`, which runs the training
  script unchanged with the output replaced).
- **Slot 63's model** (`imm-1.85m`: h192, word tier 168 with own step,
  learned prior strength, d24 rows, 2.7757), rescored with the word tier's
  gate at 0, both blocks' `gate_attn` at 0, or all three (slot 69); then
  with the word tier's add or the attention outputs replaced by their
  held-out averages (slot 71).
- **Slot 63 retrained** (slot 72), 41 epochs, seed 1337: without
  `--word-tier 168 --tier-own-step`; and with `--no-memory-attention`
  (`3989c87`), which gives the blocks no records so they keep their MLPs.
- Breakdowns by place in the word, what is predicted, and repeats
  (`ko_buckets.py`), paired per held-out character against the full model.

## How to rerun

Scripts here, as they ran; paths are the original machine's. Comments that
quoted the user were paraphrased; every command is as it ran.

| run | script | commit | slot | proposed tag |
|---|---|---|---|---|
| memory off, h256 and h432 | `nomem_slot.sh 22` (`count_params_nomem.py` sized h432) | `3c8ca62` | 22 | `rnd/memory-knockouts/slot22` |
| gate zero | `gate0_slot.sh 28` | `787ee25` | 28 | `rnd/memory-knockouts/slot28` |
| mean instead of zero | `attn_mean_slot.sh 29` with `attn_mean.py` | `787ee25` | 29 | `rnd/memory-knockouts/slot29` |
| slot 63 zeroed | `ko/slot.sh 69` (runs `ko/run.sh`) | `ccacad2` | 69 | `rnd/memory-knockouts/slot69` |
| slot 63 mean | `ko/slot_mean.sh 71` (runs `ko/run_mean.sh`, `ko/mean_ablate.py`) | `ccacad2` | 71 | `rnd/memory-knockouts/slot71` |
| slot 63 retrained without each | `rt_slot.sh 72` | `ccacad2` (no word tier), `3989c87` (no attention) | 72 | `rnd/memory-knockouts/slot72` (on `3989c87`; the no-word-tier arm is `ccacad2`) |

- `ko/run.sh` scores checkpoint copies with the gates set to zero. Those
  copies were made by a few lines of Python that weren't kept; each copy is
  the best checkpoint with `word_tier.gate` and/or both blocks' `gate_attn`
  set to 0, optimizer state kept (`--resume-checkpoint` needs it).
- `ko/maxrss.py` reports a run's peak memory.
- Results here: `results/ko_buckets.txt` (zero), `results/ko_buckets_mean.txt`
  (mean), `results/rt_buckets.txt` (retrained), from `ko_buckets.py`.

## Results

### 2026-11-13: memory off, retrained (slot 22)

| run | trained params | held-out | vs base | training NLL | first char / later |
|---|---:|---:|---:|---:|---:|
| base, memory on, 1 × h256 | 1,949,357 | **3.1517** | | 1.1117 | 3.451 / 3.092 |
| memory off, h256 | 895,147 | 3.2172 | +2.1% | 1.1454 | 3.546 / 3.151 |
| memory off, h432 | 1,964,347 | 3.1821 | +1.0% | 1.1260 | 3.495 / 3.119 |
| (memory on, 3 × h128, for size) | 1.01M | 3.1668 | +0.5% | | |

- The block is worth 2.1% at the same GRU and 1.0% at the same parameter
  count. Of its 1.05M parameters, the MLP is about 525k, the attention's
  projections about 263k, the record writer about 263k, so how much of
  the 2.1% is attention and how much the MLP's depth wasn't separated here.
- The gap opens over epochs 1–5 and then holds; it shows in training NLL
  too.
- Memory off runs at 554k (h256) and 443k (h432) characters a second,
  against 77.6k for the memory loop then.

### 2026-11-14: gate zero and mean ablation (slots 28 and 29)

| | base (loop) | parallel |
|---|---:|---:|
| as trained | 3.1517 | 3.1518 |
| attention off (gate 0) | 3.3486 (+6.2%) | 3.3521 (+6.4%) |
| MLP off (gate 0) | 3.3711 (+7.0%) | 3.3975 (+7.8%) |
| attention output replaced by its overall mean | 3.2490 (+3.1%) | 3.3387 (+5.9%) |
| … by its mean at the position in the word | 3.2525 (+3.2%) | 3.3325 (+5.7%) |

- The trained models lean on the history, though tanh(gate_attn) is only
  −0.015 / −0.022: a gate's size says little about what its branch adds.
- **Reliance isn't worth.** Zeroing the trained attention costs 3x what
  retraining without the whole block costs.
- In the parallel model the loss is the history's content (the mean brings
  back 7–10% of it). In the base half of it was an offset (the mean brings
  back 3.1% of the 6.2%), since the base feeds the attention output into
  the carried state and the records.
- A correction recorded 2026-11-15: these runs split "first / later" by
  where the prediction is made, so "later" includes each next word's first
  character. The first reading, that the records don't mainly say which
  word comes next, doesn't follow (`lessons.md`).

### 2026-11-17: slot 63's model switched off (slots 69 and 71)

Unmodified, it scores 2.7757 both times.

| switched off | zero | mean |
|---|---:|---:|
| word GRU | 3.1181 (+12.3%) | 3.2468 (+17.0%) |
| memory attention | 2.8118 (+1.30%) | 2.8016 (+0.93%) |
| both | 3.2929 (+18.6%) | 3.3132 (+19.4%) |
| overlap (both minus the two) | +4.25% | +1.10% |

- 83% of the word GRU's zero cost falls on a word's last character, where
  the next word's first is predicted (17.5% of targets); 45% of the total
  on sentence ends alone (1.2% of targets). There, switched off, it is
  worse than a uniform guess (6.95 nats zero, 7.6 mean, against 4.56).
- The mean is worse than zero for the word GRU: the typical word state
  isn't neutral; the model reads it as mid-sentence.
- The memory attention is relied on little, and where expected: spaces and
  newlines +3.7% (mean), uppercase +2.7%, repeats 1–16 words back +1.8%,
  16–64 back +0.8%, beyond 64 nothing.

### 2026-11-17: retrained without each (slot 72)

| | held-out, best (epoch 41) | change | against the d24 transformer (2.8131) | parameters saved | wall time |
|---|---:|---:|---:|---:|---:|
| slot 63 | 2.7757 | | −1.33% | | 3,854 s |
| no word tier | 2.7915 | **+0.57% ±0.06** | −0.77% | 214,897 | 3,654 s |
| no attention | 2.7903 | **+0.52% ±0.05** | −0.81% | 297,986 | 2,616 s |

Epochs 10 / 20 / 30: slot 63 2.806 / 2.785 / 2.777; no word tier 2.824 /
2.802 / 2.795; no attention 2.823 / 2.801 / 2.794.

- Each is worth about half a percent, about 40% of imm's lead over the
  transformer then. Without either, imm still led by 0.77–0.81%. (That
  lead was later qualified: see the `no-prior-pair-1.8m` entry on the
  transformer's dropout.)
- The word GRU is relied on 20–30x its worth. What it still adds is at
  boundaries: +10.3% ±1.3 before a sentence's first character, +4.5% at a
  phrase end, +3.7% on uppercase letters; +0.02% on lowercase letters.
- The attention is worth about half what it is relied on for: repeats 1–16
  words back +1.82% ±0.18, 16–64 back +0.83%, beyond 64 within noise;
  punctuation and digits +3.67% ±0.55.
- The attention costs 32% of the wall time for its 0.52%; the word tier
  costs 5% for its 0.57%.
- One seed each. No model was retrained without both.

## What we learned

- Zeroing a trained branch measures reliance; to measure worth, retrain
  without it. The two differed by 3x (attention, d16) to 20–30x (word GRU).
- The memory attention earns its keep mostly on repeats within about 64
  words, and the word GRU at sentence and phrase boundaries. Neither is
  where most of the loss is (`data-context-size`).
