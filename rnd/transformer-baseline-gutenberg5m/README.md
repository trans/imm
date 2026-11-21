---
title:    A plain character transformer on Gutenberg 5M
kind:     experiment
status:   concluded
outcome:  reference
question: >-
  How does imm compare with a plain character transformer trained and
  scored on the same split, at nanoGPT's size and at imm's size, and why
  does imm train so much slower?
answer: >-
  nanoGPT's character config (10.7M parameters) scored 3.0212 in 6.4 min,
  14% better than imm's best then (3.4359, about 65 min). At imm's size
  (474k) the transformer scored 3.7527 after 5,000 steps and 3.2694 after
  50,000 (7.1 min), 5% better than imm. imm trained at about 3,600
  characters a second against 1.9M; its one-stream, segment-at-a-time
  shape cost 600x, not the GRU. Later, @air's 1.83M transformer scored
  3.2239 at dropout 0 and 3.0542 at dropout 0.2.
opened:   2026-11-09
updated:  2026-11-10
code: >-
  main 43057d9 (scripts/transformer_baseline.py). @air's 1.83M runs used the
  same script; the commit they ran isn't recorded.
data: >-
  Gutenberg 5M split 4bfd5a43d446644a, imm's vocabulary, every held-out
  character after the first scored once with up to 256 characters of
  context. No count prior in the transformer.
slots:    none (before the slot system)
tags:     [baseline, transformer, throughput, gutenberg]
related:  [batched-streams, residual-width, depth-evidence-input, transformer-prior-pg19]
---

# A plain character transformer on Gutenberg 5M

## Question

imm's best on Gutenberg 5M was 3.4359 after 3 epochs (`depth-evidence-input`).
Is that good? The comparison is a plain character transformer trained on the
same training text and scored the way imm is scored.

## Design

`scripts/transformer_baseline.py` reimplements nanoGPT's Shakespeare
character config (6 layers, 384 wide, context 256, dropout 0.2, 5,000 steps)
so that the scoring matches imm: every held-out character after the first,
with up to 256 characters of context, reported at the last step. Batch 64 ×
256, AdamW, lr 1e-3 with a cosine to 1e-4, 100 warm-up steps, weight decay
0.1, seed 1337. RTX 4070 Laptop GPU, bf16.

Arms: nanoGPT's config; an imm-sized one (4 layers × 96, 4 heads, dropout
0) at 5,000 and 50,000 steps; and @air's 4 × 192 (1.83M) at dropout 0 and
0.2, 50,000 steps.

A second part times training throughput on random data to find where imm's
time goes.

## How to rerun

The script is in git; the commands weren't kept, but each run file in `runs/`
records its arguments, given here. Commit `43057d9` on `main`; proposed tag
`rnd/transformer-baseline-gutenberg5m/nanogpt-char` (the same commit serves
every arm).

```sh
cfg=configs/gutenberg5m_crossfit20_ps100_e1.json
# 10.7M, nanoGPT's character config: the script's defaults
python scripts/transformer_baseline.py --config $cfg --output runs/transformer_baseline_gutenberg5m_nanogpt_char.json
# 474k, 5,000 and 50,000 steps
python scripts/transformer_baseline.py --config $cfg --output runs/transformer_baseline_gutenberg5m_l4w96_d0.json \
  --layers 4 --heads 4 --width 96 --dropout 0
python scripts/transformer_baseline.py --config $cfg --output runs/transformer_baseline_gutenberg5m_l4w96_d0_i50k.json \
  --layers 4 --heads 4 --width 96 --dropout 0 --iters 50000 --eval-every 5000
# @air's 1.83M, dropout 0 and 0.2
python scripts/transformer_baseline.py --config $cfg --output runs/transformer_baseline_gutenberg5m_l4w192_d0_i50k.json \
  --layers 4 --heads 4 --width 192 --dropout 0 --iters 50000 --eval-every 5000
python scripts/transformer_baseline.py --config $cfg --output runs/transformer_baseline_gutenberg5m_l4w192_d02_i50k.json \
  --layers 4 --heads 4 --width 192 --dropout 0.2 --iters 50000 --eval-every 2500
```

The throughput measurements used a timing script that wasn't kept.

## Results

### 2026-11-09: three transformer runs

| model | params | steps | chars seen | time | held-out PPL | bits/char |
|---|---:|---:|---:|---:|---:|---:|
| nanoGPT char config (6 × 384, dropout 0.2) | 10.7M | 5,000 | 82M | 6.4 min | **3.0212** | 1.595 |
| imm-sized (4 × 96, 4 heads, dropout 0) | 474k | 5,000 | 82M | 42 s | 3.7527 | 1.908 |
| imm-sized, 10x longer | 474k | 50,000 | 820M | 7.1 min | **3.2694** | 1.709 |
| imm, best then (scale 1.0, L2 0.05, 3 epochs) | 441k neural, plus counts | | 14M | ~65 min | 3.4359 | 1.781 |

- The large one beats imm by 14% (0.19 bits/char) in a tenth of the time.
  It passed 3.4359 at step 2,000 (2.5 min).
- At imm's size, trained 7 minutes, the transformer is 5% better than imm.
  It passes imm at about step 16,000 (about 260M characters, 55 passes).
- imm learns more per character, the transformer far more per second: at
  82M characters the small transformer is at 3.75, and imm reaches 3.44 on
  14M.
- Both transformers see more context (256 characters) than imm's GRU gets
  gradients through (32 segments of up to 16 characters).

### 2026-11-09: why imm trains slowly

Training throughput, forward + backward + AdamW, bf16, random data,
vocabulary 100:

| model, shape | params | chars/s |
|---|---:|---:|
| transformer 4 × 96, batch 64 × 256 | 477k | 2.07M |
| GRU hidden 280, batch 64 × 256 | 528k | 2.09M |
| GRU hidden 64, batch 64 × 256 | 38k | 12.6M |
| GRU hidden 64, imm's shape: 1 stream, a call per 16 characters, 32 per step | 38k | 21k |
| the same, without a `.item()` per segment | 38k | 23k |
| imm itself, Gutenberg config | 441k | ~3.6k |

- The GRU isn't the slow part: at equal parameters it matches the
  transformer. imm's shape (a batch of one, 16-character calls) costs 600x,
  and imm's per-segment Python work another 6x.
- Segments can't be fused into one GRU call in this config, because the
  carried state is taken after the cross-attention. Independent streams
  can be batched. That became `batched-streams`.

### 2026-11-10: @air's 1.83M transformer

| model | params | best held-out | last held-out |
|---|---:|---:|---:|
| 4 × 192, dropout 0 | 1,832,832 | 3.2239 (step 10k) | 4.1457 (step 50k) |
| 4 × 192, dropout 0.2 | 1,832,832 | **3.0542** (step 47.5k) | 3.0594 (step 50k) |

- At dropout 0 it memorises 5M characters: training loss 0.81 nats at the
  end against 1.42 held out.
- At dropout 0.2 it doesn't overfit (training loss about 1.17 nats with
  dropout on, 1.118 held out) and was still improving. From 474k to 1.83M
  the transformer gains 6.4%.
- @air's reading: imm's wide residuals were unregularised more than out of
  capacity. imm then had no dropout and weight decay 0.01, against the
  transformer's 0.2 and 0.1. That led to imm's `--dropout`
  (`residual-width`).

## What we learned

- This is the reference imm was measured against on Gutenberg 5M. Every
  later Gutenberg comparison uses these numbers.
- imm's training speed, not its quality per character, was the first
  limit. Batching streams came next.
- The transformer's dropout mattered a great deal at 1.83M on 5M
  characters (3.2239 against 3.0542). On PG-19 it later went the other way:
  at 100M characters dropout 0.2 held the 1.84M transformer back (slot
  105; the `no-prior-pair-1.8m` entry). Each corpus needs the
  regularisation checked on both sides (`lessons.md`).
