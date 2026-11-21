---
title:    imm at the transformer's size, on the depth-16 and depth-24 priors
kind:     experiment
status:   concluded
outcome:  positive
question: >-
  Does a smaller imm, at the 1.86M transformer's parameter count, still
  beat it when both have the same prior; and how much does mobius's 10 GB
  depth-24 prior add?
answer: >-
  Yes on both priors. At 1,852,908 parameters imm scored 3.0575 on the d16
  rows, 1.34% ahead of the transformer (3.0987) and level with the 3.2M
  imm (3.0563): on the 20M split the data, not the size, is the limit. On
  the d24 rows (prior alone 3.0684) imm scored 2.7757, 9.2% better, and
  1.33% ahead of the transformer on the same rows (2.8131). The lead is at
  boundaries and on repeated words; imm trails where the d24 prior is
  badly wrong. The transformer's dropout was not tuned (see below).
opened:   2026-11-16
updated:  2026-11-19
code: >-
  ccacad2 (branch drop-row-cache, deleted; the commit is on main), which
  adds --drop-row-cache.
data: >-
  PG-19 20M split, data/pg19/imm-split-20m (249,999 held-out targets).
  Slot 60: mobius's pg19split20m_gated_d16_pg200m (depth 16 over 200M
  characters, 500 books; alone 3.5320). Slot 63:
  pg19split20m_gated_d24_pg10g (depth 24 over 10.4G characters, 26,497
  books; alone 3.0684). KN evidence pg19split20m_kn in both.
slots:    [60, 63]
tags:     [size, prior, depth-24, transformer-comparison, pg19, baseline]
related:  [learned-prior-strength, word-tier, memory-knockouts, topic-tiers, memory-window, transformer-prior-pg19]
---

# imm at the transformer's size, on the depth-16 and depth-24 priors

## Question

The user's question since 2026-11-09 was whether a smaller model plus the
prior matches the transformer (`residual-width`). imm at 3.2M was 1.37%
ahead of the 1.86M transformer (`learned-prior-strength`), with 1.7x its
parameters. At the same size? And then on mobius's much larger prior?

## Design

- **Slot 60:** B at 820M (`learned-prior-strength`) made smaller: hidden
  192, word tier 168, two blocks, own step, the learned prior strength, 41
  epochs, cosine, seed 1337, `--drop-row-cache` (drops the row files'
  cached pages as they are read). 1,852,908 parameters used, counted from
  the model built at that width, against the transformer's 1,859,532 (both
  with their prior-input layers).
- **Slot 63:** slot 60's command with only `--prior-rows` changed to
  `pg19split20m_gated_d24_pg10g`. Same layout, vocabulary and digests.
- **The transformer on d24** is @air's slot 64 run
  (`d24_s1.84m_prior_in`, air `fbd853e`), rescored per character on the CPU
  (`../learned-prior-strength/score_transformer_chars.py`): 2.8130.
- Paired per-character comparisons with `scripts/rnd/compare_tx.py`
  (`PRIOR_ROWS` set to the d24 rows where stated), and by place in the
  piece and repeated words with `scripts/rnd/topic_buckets.py`.

## How to rerun

Scripts here, as they ran; paths are the original machine's. Comments that
quoted the user were paraphrased; every command is as it ran.

| run | script | commit | slot | proposed tag |
|---|---|---|---|---|
| h192 on d16 | `small_slot.sh 60` | `ccacad2` | 60 | `rnd/imm-1.85m/slot60` |
| h192 on d24 | `d24_slot.sh 63` | `ccacad2` | 63 | `rnd/imm-1.85m/slot63` |

Paired comparisons here (`compare_tx.py` unless stated; "imm" is A, "tx" is
B):

- `results/small_vs_tx.txt`: slot 60 against the transformer on d16;
- `results/small_vs_pg41.txt`: slot 60 against the 3.2M imm (slot 57);
- `results/d24_vs_d16.txt`: slot 63 against slot 60;
- `results/d24_vs_tx.txt`: slot 63 against the transformer on d16 (not an
  equal-prior comparison);
- `results/d24_vs_txd24.txt`: slot 63 against the transformer on d24,
  buckets from the d24 prior;
- `results/d24_vs_txd24_d16ref.txt`: the same pair, buckets from the d16
  prior;
- `results/topic_buckets.txt`: slot 63 against the d24 transformer by place
  in the piece and repeats (`topic_buckets.py`).

## Results

### 2026-11-16: the smaller imm on d16 (slot 60)

| at 820M characters | params | held-out |
|---|---:|---:|
| **imm h192, word tier 168, learned prior strength** | **1,852,908** | **3.0575** |
| imm h256, word tier 256, learned prior strength (slot 57) | 3,200,444 | 3.0563 |
| 1.86M transformer, prior in the input | 1,859,532 | 3.0987 |

- With 0.36% fewer parameters, imm is 1.34% ahead of the transformer
  (paired). One seed.
- By place: −1.3% at a word's first character, −1.1% inside, −2.4% at its
  last, −6.4% before a phrase, −11.6% before a sentence. By the prior's
  loss: ahead below 3 nats, level at 3–6 (−0.08% ±0.42), ahead at 6 or more
  (−3.6% ±2.5).
- Against h256: level (+0.04%). Cutting imm from 3.2M to 1.85M parameters
  cost nothing.
- Epochs 1 / 5 / 10 / 20 / 30 / 41: 3.272 / 3.155 / 3.108 / 3.075 / 3.061 /
  3.057. Mean prior strength 0.941. 4,126 s.
- The row-cache drop held: the training process's VmRSS peaked at 2.9 GB,
  against slot 57's 9.75 GB.

### 2026-11-16/17: on the 10 GB depth-24 rows (slot 63)

| at 820M characters, 1,852,908 params | prior alone | imm | imm against its prior |
|---|---:|---:|---:|
| **d24, 10 GB rows (slot 63)** | **3.0684** | **2.7757** | −9.54% |
| d16, 200M rows (slot 60) | 3.5320 | 3.0575 | −13.44% |

- The better prior moves imm by −9.2% (paired), 0.0967 nats, against the
  prior alone's 0.1409: imm keeps 69% of the prior's own gain.
- **Against the transformer on the same d24 rows (2.8131): −1.33%**, the
  same margin as on d16. Paired over 249,999 targets: −0.01335 nats, ±0.07%
  per character, ±0.12% in blocks of 1,000. imm is lower on 54.0% of
  characters. One seed each.
- The boundaries still carry it: −6.9% ±0.8 before a phrase, −11.4% ±1.4
  before a sentence, −10.6% before an uppercase letter, but −0.7% on
  lowercase letters.
- **imm now trails where the deeper prior is badly wrong:** +1.11% ±0.42
  at 3–6 nats of the d24 prior's loss, +3.7% ±2.5 at 6 or more (11% of
  targets). With the learned prior strength at 0.984, imm leans on the
  counts even where they are wrong.
- Against slot 60, the deeper counts help within words (−9.5% inside,
  −8.6% at a word's first character), and little at the boundaries, where
  the tiers already worked (−2.6% before a phrase, −1.4% before a
  sentence).
- Epochs 1 / 5 / 10 / 20 / 30 / 41: 2.893 / 2.830 / 2.806 / 2.785 / 2.777 /
  2.776. Epoch 1 already beats slot 60's best. It saturates: −0.04% from
  epoch 30 to 41. 3,854 s.

### 2026-11-17: where the lead comes from, by document (CPU)

A held-out piece is 5,000 characters; a word repeats if its lowercase form
already occurred in the same piece.

- The lead grows through the piece: −0.80% ±0.44 in a piece's first 256
  characters, −0.99% at 256–1,000, −1.41% at 1,000–2,500, −1.43% at
  2,500–5,000.
- Repeated words carry 44% of it (33% of targets, −1.74% ±0.12); new words
  −0.62% ±0.09. Capitalised repeats, mostly names: −12.4% on 2.1% of
  targets, 20% of the lead.
- By distance back: −1.15% at 1–16 words; −2.92% at 16–64, where imm's
  64-record memory reaches and the transformer's 256 characters run out;
  −1.36% at 64–256 and −0.88% beyond.
- Where imm trails (d24 prior ≥ 3 nats): new words +1.46% ±0.48; repeats
  are level. The deficit is surprise, not recall.
- @air tested the user's idea, a measure of how concentrated the unusual
  text is, in one causal form: the prior's recent surprisal within the piece. It sorts
  stretches by difficulty but barely predicts the prior's misses (AUC
  0.556, against 0.716 for the local entropy).

### 2026-11-19: caveat on the transformer comparisons

Later work found @air's transformers at dropout 0.2 held back on PG-19
(slots 101 and 105; the `no-prior-pair-1.8m` entry). For the prior-in-input
form on this split the evidence is weaker (training 1.100 against 1.131
held out), but untested. So the 1.33–1.34% leads here are against an
untuned baseline.

## What we learned

- At equal size and equal prior, imm led the transformer by about 1.3% on
  the 20M split, on both priors. This model (slot 63, 2.7757) became the
  base for the knockouts (`memory-knockouts`), the topic tiers
  (`topic-tiers`) and the memory-window runs (`memory-window`).
- The deeper prior helps inside words; the tiers help at boundaries; the
  memory helps on repeats 16–64 words back. imm's remaining weakness is
  surprise where the prior is confidently wrong.
