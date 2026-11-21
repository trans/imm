---
title:    imm on the 20M PG-19 split, and attention-block depth
kind:     experiment
status:   concluded
outcome:  mixed
question: >-
  On a split four times larger, where does imm stand against the
  transformers with the same prior, and is a second or third attention
  block worth more than width?
answer: >-
  The base h256 run scored 3.1517, 5.1% better than on the 4.75M split
  and 1.7% behind the 1.84M transformer with the prior in its input
  (3.0987). A second block is worth 0.40% at h256 (3.1392); 2 × h192
  (1.64M) is level with 1 × h256 (3.1466); 3 × h128 (1.01M) is 0.48%
  behind (3.1668). On the parallel path two blocks give 3.1360 and a third
  only 0.03% more (3.1351). Depth stops paying at two blocks.
opened:   2026-11-12
updated:  2026-11-14
code: >-
  base run ba91341 (branch pg19-20m); depth runs 57f1033 (branch
  prior-dropout); parallel depth runs 787ee25 (branch parallel-memory).
  Branches deleted; the commits are on main.
data: >-
  PG-19 20M split, data/pg19/imm-split-20m (mobius e3622a7; 800 training
  pieces, the same 50 held-out pieces as the 4.75M split, hash
  095cbd13…). Mobius's pg19split20m_gated_d16_pg200m prior (alone 1.8205
  bits/char, PPL 3.532) with compact pg19split20m_kn depth evidence.
slots:    [3, 7, 31, 34]
tags:     [pg19, depth, attention-blocks, width, transformer-comparison]
related:  [pg19-4.75m-split, prior-dropout, parallel-memory, memory-window, transformer-prior-pg19, word-tier]
---

# imm on the 20M PG-19 split, and attention-block depth

## Question

On the 4.75M split the model overfit (`pg19-4.75m-split`). Mobius's 20M
split has four times the training text and the same held-out text. Where
does imm stand there, and how should it grow: wider, or deeper in attention
blocks? The user asked to try depth first (2026-11-12).

## Design

- **Base run** (`ba91341`): h256 (1,949,357 trained parameters), 64
  records, 64 word segments per update, `--carry-bound norm --carry-scale
  32` (`carry-bound`), dropout 0.2, 20 epochs (about as many characters as
  80 on 4.75M) of cosine 1e-3 → 1e-4, seed 1337. No prior dropout.
- **Depth** (`57f1033`, `--attention-layers`), everything else the base:
  - 2 × h192 (1,637,807 parameters, 16% fewer than the base);
  - 2 × h256 (2,739,631), so depth alone;
  - 3 × h128 (1,013,041, half the base).
- **Depth on the parallel path** (`787ee25`, `parallel-memory`): two and
  three blocks at h256.
- After each run, the take-the-prior-away test on its best checkpoint
  (`prior-dropout`).

## How to rerun

Scripts here, as they ran; paths are the original machine's. Comments that
quoted the user or @air were paraphrased; every command is as it ran.

| run | script | commit | slot | proposed tag |
|---|---|---|---|---|
| base 1 × h256 | `pg19_20m_h256.sh` | `ba91341` | none | `rnd/blocks-20m/base-h256` |
| 2 × h192 | `pg19_20m_slot.sh deep` | `57f1033` | 3 | `rnd/blocks-20m/slot3` |
| 2 × h256, then 3 × h128 | `slot7_wait_and_run.sh` (runs `pg19_20m_slot.sh pair`) | `57f1033` | 7 | `rnd/blocks-20m/slot7` |
| parallel, two blocks | `par_push_slot.sh 31` (second run) | `787ee25` | 31 | `rnd/blocks-20m/slot31` |
| parallel, three blocks | `tx_l3_slot.sh 34` (second part) | `787ee25` | 34 | `rnd/blocks-20m/slot34` |

`count_params.py` counts trained parameters for a width and block count;
`par_wide_check.py` checks that the parallel path equals the loop with a
memory window wider than an update. `par_push_slot.sh`'s first run (128
records) is in `memory-window`; `tx_l3_slot.sh`'s first part (the
transformer at imm's budget) is in `transformer-prior-pg19`. The config is
`configs/pg19split20m_crossfit20_ps100_e1.json` (`870c256`).

## Results

### 2026-11-12: the base run (`ba91341`)

- **Best and last 3.1517 (1.6561 bits/char) at epoch 20,** still improving
  (3.1537 at 19). First characters of a word 3.451, later ones 3.092.
  Training NLL 1.1117 nats: a held-out gap of about 0.05 bits/char, not
  overfitting. 1 h 56 min, peak RSS 8.3 GB.
- From 4.75M to 20M: 3.3220 → 3.1517, 5.1%.
- Against the 1.84M transformers on this split with the books prior: 1.7%
  behind the prior-in-input form (3.0987), 1.2% ahead of logits-only
  (3.1904), 11.7% ahead of the plain transformer (3.5689).

### 2026-11-12/13: depth (`57f1033`, slots 3 and 7)

| scored as | 3 × h128 (1.01M) | 2 × h192 (1.64M) | 1 × h256 (1.95M) | 2 × h256 (2.74M) |
|---|---:|---:|---:|---:|
| in full | 3.1668 | 3.1466 | 3.1517 | **3.1392** |
| bits/char | 1.6630 | 1.6538 | 1.6561 | 1.6504 |
| first character of a word / later ones | 3.461 / 3.108 | 3.434 / 3.089 | 3.451 / 3.092 | 3.415 / 3.084 |
| own output, prior in the input | 63.90 | 61.62 | 62.86 | 59.15 |
| own output, no prior | 92.24 | 92.97 | 98.06 | 93.11 |
| fresh layer, prior in the input | 4.72 | 4.43 | 4.29 | 4.24 |
| fresh layer, no prior | 10.32 | 9.58 | 9.42 | 9.45 |
| train / held-out gap, bits/char | 0.037 | 0.046 | 0.052 | 0.065 |

- **The second block is worth 0.40%** at h256 (−0.0057 bits/char). It led
  at every epoch from 2 on, by 0.011–0.015. One seed (1337) for all; the
  seed spread was 0.5–0.6% on Gutenberg 5M.
- 2 × h192 is level with 1 × h256 (0.16% better) with 16% fewer
  parameters, ahead at every epoch from 5 on.
- Width on top of depth adds little: 2 × h256 over 2 × h192 is 0.24% for
  67% more parameters.
- 3 × h128 is 0.48% behind 1 × h256 with half the parameters. No 1 × h128
  run, so depth and width aren't separated at h128.
- Most of the depth gain is at the first characters of words.
- The prior's role didn't change: taken away, every network depends on it
  as much as the base does.
- Every run was still improving at epoch 20.

### 2026-11-14: depth on the parallel path (`787ee25`, slots 31 and 34)

| | one block | two blocks | three blocks | loop, two blocks |
|---|---:|---:|---:|---:|
| trained parameters | 1.95M | 2.74M | 3.53M | 2.74M |
| held-out, best (epoch 20) | 3.1518 | 3.1360 | **3.1351** | 3.1392 |
| epoch 1 / 4 / 8 / 12 / 16 | 3.3404 / 3.2326 / 3.1889 / 3.1681 / 3.1572 | 3.3313 / 3.2246 / 3.1757 / 3.1574 / 3.1417 | 3.3315 / 3.2167 / 3.1699 / 3.1519 / 3.1398 | 3.3357 / 3.2304 / 3.1839 / 3.1617 / 3.1456 |
| training NLL, epoch 20 | 1.1069 | 1.0956 | 1.0885 | 1.0988 |
| first character / later | 3.440 / 3.094 | 3.425 / 3.078 | 3.414 / 3.079 | 3.415 / 3.084 |
| take-away: own, prior in input / none | 61.41 / 99.45 | 59.60 / 95.21 | 57.41 / 95.11 | 59.15 / 93.11 |
| model time, characters/s | 276k | 174k | 126k | |
| wall time, 20 epochs | 2,779 s | 3,202 s | 3,940 s | 8,347 s |

- Two parallel blocks keep their gain without the feedback, 2.6x faster:
  3.1360 against the loop's 3.1392, 0.50% over one parallel block. This
  was imm's best on the 20M split: 1.2% behind the prior-in-input
  transformer, 1.7% ahead of logits-only.
- **The third block is worth 0.03%,** inside any seed spread. It fits the
  training text 0.65% better without generalising better, for 23% more
  wall time.
- Both MLP gates in the two-block models went negative, as in the loop's.

## What we learned

- Depth helps up to two blocks; width helps little at the best epoch.
  Neither changes how much the model leans on the prior.
- At 400M training characters, imm 1 × h256 is level with the 477k
  transformer given the prior, at 4x its parameters
  (`transformer-prior-pg19`). The next lever tried was a word-level tier
  (`word-tier`), not more blocks.
- The user's plan at this point (2026-11-13): push the GRU as far as it
  goes before swapping it for a transformer.
