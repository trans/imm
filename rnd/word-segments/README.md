---
title:    Word-boundary segments
kind:     experiment
status:   concluded
outcome:  positive
question: >-
  imm wrote a memory record at the end of each segment, and a segment ran
  until its text occurred at most once in the training text (the unique
  rule, capped at 16 characters). Do segments that end at word boundaries
  work better?
answer: >-
  Yes. On Tiny canonical, word segments at 16 records beat unique at 8 on
  three seeds: mean 3.6599 against 3.6782 (0.50% better; 0.48%, 0.21% and
  0.80% per seed). On Gutenberg 5M with mobius's depth-16 evidence at
  hidden 256, word segments took the best epoch from 3.1767 to 3.1316
  (1.42% better), and led at every epoch by 0.8–1.9%.
opened:   2026-11-09
updated:  2026-11-09
code: >-
  --segment-rule: c08d617, fixed in fda114d (branch segment-rules); the
  Gutenberg run at 37321f1 (branch word-segments-depth-evidence, which
  merges segment-rules into external-evidence). Branches deleted; the
  commits are on main.
data: >-
  Tiny Shakespeare canonical 4fa9aec1db6b3aea (imm's depth-8 prior,
  cross-fitted, strength 1.0, no prior input; prior alone 3.8595).
  Gutenberg 5M 4bfd5a43d446644a with --prior-input both and mobius's KN
  depth-16 evidence (gutenberg5m_kn).
slots:    none (before the slot system)
tags:     [segments, words, memory-records, tiny, gutenberg]
related:  [depth-evidence-input, residual-width, batched-streams]
---

# Word-boundary segments

## Question

The user's idea (2026-11-09, via @air): end segments at word boundaries.
Under the unique rule a segment averaged 10.8 characters on Gutenberg 5M
and 9.3 on Tiny. @air's point: that length grows with the corpus, so at
1 TB most segments would hit the cap and become fixed 16-character chunks. A
word rule doesn't depend on corpus size. Word segments are about twice as
many, so twice the GRU steps and half the reach per record.

## Design

`--segment-rule` (`c08d617`):

- `unique` (the default, unchanged);
- `word`: each segment is a word and the whitespace after it (`\S+\s*`),
  split at `--max-depth`;
- `hybrid`: the unique rule's length, extended to the next word boundary.

In `c08d617` the cap cut 28% of hybrid segments mid-word; since `fda114d` an
extension that passes the cap ends at the last word start within it (99.93%
end on a word). Mean training-segment length on Tiny: unique 9.31, word
5.50, hybrid 11.73 characters.

Arms:

- **Tiny grid:** the 3.6769 baseline's settings (16 streams, lr 3e-3, scale
  1.0, L2 0.05, 10 epochs, hidden 32, seed 1337, no prior input). unique at 8
  records, word at 8 and 16, hybrid at 8. Then seeds 1 and 2 for unique 8
  and word 16.
- **Gutenberg:** the mobius-16 run of `depth-evidence-input` (hidden 256, lr
  1e-3, 64 streams, 12 epochs, `--prior-input both`) with only
  `--segment-rule word` changed, 16 records.

## How to rerun

- Tiny grid: `segment_grid.sh` here, from a worktree at `fda114d`. Proposed
  tag `rnd/word-segments/tiny-grid`.
- Tiny seeds: the first half of `../depth-evidence-input/seeds_then_depth.sh`,
  same commit and tag.
- Gutenberg: no script was kept. It is `seeds_then_depth.sh`'s mobius16
  command with `--segment-rule word --max-memory 16`, run id
  `gutenberg5m_crossfit20_ps100_rs100_l2005_s64_lr1e-3_h256_e12_both_mobius16_segword_m16`,
  at `37321f1`. Proposed tag `rnd/word-segments/gutenberg-h256`.

Paths in the scripts are the original machine's.

## Results

### 2026-11-09: Tiny canonical grid (`fda114d`)

| rule | records | held-out, epoch 10 | best (epoch) | train | segment length | attention distance | top weight | time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| unique | 8 | 3.6767 | 3.6767 (10) | 3.7624 | 9.31 | 30.6 | 0.35 | 4.6 min |
| word | 8 | 3.6678 | 3.6640 (8) | 3.7409 | 5.50 | 22.8 | 0.48 | 7.0 min |
| word | 16 | **3.6590** | 3.6590 (10) | 3.7437 | 5.50 | 33.0 | 0.35 | 7.0 min |
| hybrid | 8 | 3.6712 | 3.6702 (8) | 3.7492 | 11.73 | 28.8 | 0.52 | 3.4 min |

- The unique rerun reproduces the baseline (3.6767 against 3.6769).
- Every word-boundary rule beats unique: word at 16 by 0.48%, word at 8 by
  0.24% with less reach, hybrid by 0.15% in three quarters of the time.
- Word segments cost 1.5x the time; the number of records costs nothing
  measurable.

### 2026-11-09: three seeds, word at 16 against unique at 8

| seed | unique 8 | word 16 | change |
|---|---:|---:|---:|
| 1337 | 3.6768 | 3.6591 | −0.48% |
| 1 | 3.6740 | 3.6662 | −0.21% |
| 2 | 3.6838 | 3.6544 | −0.80% |
| mean | 3.6782 | **3.6599** | −0.50% |

Each rule's seed spread is about 0.15% (sd 0.005).

### 2026-11-09: Gutenberg 5M, hidden 256, depth 16 (`37321f1`)

| epoch | unique | word | change |
|---:|---:|---:|---:|
| 1 | 3.3687 | 3.3407 | −0.83% |
| 3 | 3.2550 | 3.1979 | −1.75% |
| 5 | 3.1999 | 3.1386 | −1.92% |
| 7 | 3.1767 | **3.1316** | −1.42% |
| 9 | 3.1881 | 3.1423 | −1.44% |
| 12 | 3.2043 | 3.1714 | −1.03% |

- 3.1316 at epoch 7 was imm's best on Gutenberg then: 4.2% ahead of the
  474k transformer, 3.7% behind nanoGPT's char config (3.0212).
- With less reach: word segments average 5.60 characters (unique 10.78), so
  16 records reach about half as far. Mean attention distance 36 characters
  against 65; the top record's weight 0.66 against 0.33. Word at 32 records
  (matched reach) was not tried.
- 1.6x the time: 30.5 min for 12 epochs against 19.3 (848k training
  segments against 441k). One seed, but the lead holds at every epoch.

## What we learned

- Word segments are better, and the gain grows with the data (0.50% on
  Tiny with no prior input, 1.4% on Gutenberg with depth-16 evidence). The
  two gains, word segments and deep evidence, stack.
- Every later imm run uses `--segment-rule word`. The word records became
  the unit of imm's memory and, later, of its word tier (`word-tier`).
