---
title:    A phrase-level GRU tier above the word tier
kind:     experiment
status:   concluded
outcome:  negative
question: >-
  Does a third GRU, over the word tier, that steps once a phrase (at
  ,;:.!?) help beyond the word tier?
answer: >-
  No. It scored 3.1326 against the word tier's 3.1274 (+0.17%). It helps
  the first word of a phrase (−0.47%) and of a sentence (−1.50%), but costs
  0.2–0.3% elsewhere. With own step on both, the phrase model scored
  3.1140 against 3.1155 for the word tier alone (−0.05% ±0.06): nothing
  beyond own step. A CPU analysis found the word tier's gain doesn't follow
  the prior's entropy, and that entropy rises mark word ends.
opened:   2026-11-15
updated:  2026-11-15
code: >-
  --phrase-tier and --char-loss-output at 573cea5 (branch phrase-tier);
  the own-step arm at 55b8128 (branch tier-own-step). Branches deleted;
  the commits are on main. scripts/tier_position_losses.py (573cea5)
  splits the losses by place.
data: >-
  PG-19 20M split, data/pg19/imm-split-20m (250,000 held-out characters).
  Mobius's pg19split20m_gated_d16_pg200m prior (alone 1.8205 bits/char)
  with compact pg19split20m_kn depth evidence; the entropy analysis reads
  that prior's held-out feature rows.
slots:    [40, 50]
tags:     [hierarchy, phrase-tier, gru, pg19, entropy]
related:  [word-tier, topic-tiers, blocks-20m]
---

# A phrase-level GRU tier above the word tier

## Question

The user asked (2026-11-15) for a phrase level next, or failing that a
sentence or punctuation level, expecting the same kind of gain as for
words, though less pronounced. Does a third GRU over the word tier help?

## Design

- **`--phrase-tier WIDTH`** (`573cea5`): a phrase ends at a segment whose
  last character, skipping whitespace and closing quotes and brackets, is
  one of `--phrase-marks` (default `,;:.!?`). The phrase GRU reads the word
  GRU's state after each phrase-ending segment, its state carried. Each
  character gets the phrase state after the phrases that ended before its
  segment, projected and gated (from 0), added with the word tier's.
- **Run:** slot 36's model (`word-tier`: two blocks + word tier 256) plus
  `--phrase-tier 256`, 20 epochs, seed 1337.
- **Per-character losses** (`--char-loss-output`) for four models (two
  blocks; + word tier; + phrase tier; word tier at 41 epochs), split by
  place in the word, phrase and sentence.
- **CPU analysis** (no slot): whether words and phrases fit within the
  prior's depth cap, and how the tiers' gains relate to the prior's
  entropy along a word.
- **Own step** (slot 50, `55b8128`): the phrase arm with `--tier-own-step`.

## How to rerun

Scripts here, as they ran; paths are the original machine's. One comment in
`phrase_slot.sh` quoted the user; it was paraphrased. Every command is as it
ran.

| run | script | commit | slot | proposed tag |
|---|---|---|---|---|
| phrase tier, then four models' character losses | `phrase_slot.sh 40` | `573cea5` | 40 | `rnd/phrase-tier/slot40` |
| paired breakdowns | `paired_buckets.py` | `573cea5` | | |
| depth cap, entropy along a word | `depth_cap.py`, `depth_cap2.py`, `entropy_tiers.py` | `573cea5` | none | `rnd/phrase-tier/entropy` |
| phrase tier with own step | `../word-tier/own_slot.sh 50` (run 2) | `55b8128` | 50 | `rnd/word-tier/slot50` |

Results here: `results/tier_positions.txt` (the four models split by
place, from `scripts/tier_position_losses.py`) and
`results/own_positions_phrase.txt` (own step against none, from
`../word-tier/paired_buckets2.py`).

## Results

### 2026-11-15: slot 40

| | two blocks | + word tier | **+ word + phrase tier** |
|---|---:|---:|---:|
| trained parameters | 2.74M | 3.20M | 3.66M |
| held-out, best (epoch 20) | 3.1360 | **3.1274** | 3.1326 |
| epoch 4 / 8 / 12 / 16 / 19 | | 3.2220 / 3.1739 / 3.1523 / 3.1349 / 3.1282 | 3.2173 / 3.1682 / 3.1476 / 3.1366 / 3.1331 |
| training NLL, epoch 20 | 1.0956 | 1.0900 | 1.0906 |
| tanh of the tier gates | | word 0.076 | word 0.066, phrase 0.039 |
| wall time, 20 epochs | | 2,256 s | 2,449 s |

The phrase tier gains nothing overall: 0.17% behind the word tier alone,
about the GPU's run-to-run spread (0.15%), one seed. It led through epoch 12
by 0.15% and fell behind from epoch 16. Phrases run 6.42 words on the
held-out text, sentences 14.63.

Where each tier helps, as the paired change in perplexity, ± the standard
error over characters ("not last" means a word's positions except its
last):

| bucket | chars | word tier vs two blocks | phrase tier vs word tier |
|---|---:|---:|---:|
| all | 249,999 | −0.28% ±0.06 | +0.17% ±0.06 |
| at a word's first character | 43,854 | −0.49% ±0.12 | +0.22% ±0.13 |
| inside a word | 162,344 | −0.13% ±0.07 | +0.11% ±0.07 |
| at a word's last (predicting the next word's first) | 43,854 | −0.61% ±0.14 | +0.32% ±0.14 |
| first word of a phrase, not last | 29,015 | +0.05% ±0.23 | **−0.47% ±0.24** |
| later words of a phrase, not last | 177,130 | −0.25% ±0.06 | +0.23% ±0.06 |
| first word of a sentence, not last | 13,427 | **+0.79% ±0.42** | **−1.50% ±0.43** |
| 4th+ word of a sentence, not last | 168,438 | −0.29% ±0.07 | +0.28% ±0.06 |
| a phrase's last position (predicting the next phrase's first) | 6,832 | −1.20% ±0.54 | +0.11% ±0.49 |
| a sentence's last position (predicting the next sentence's first) | 2,998 | **−2.72% ±0.89** | +1.40% ±0.81 |

- The phrase tier shows the word tier's pattern one level up: its gain is
  on the first word of a phrase, and most on the first word of a sentence.
  Per position that is more pronounced than the word tier's word-start
  gain, not less as expected, but it covers few characters (12% and 5% of
  the text), and elsewhere the model is 0.2–0.3% worse.
- **The phrase state arrives a word late:** a tier's state from segment k
  reaches only segment k + 1. So the position that predicts a new phrase's
  first character can't use the phrase just finished. That led to
  `--tier-own-step` (`word-tier`).

### 2026-11-15: the depth cap and the prior's entropy (CPU)

- Words longer than the cap: 171 of 43,670 at 16 (0.39%, 1.49% of
  characters), 34 at 24. Phrases and sentences aren't capped, since the
  tier steps once a phrase, but most are longer than the prior's context:
  phrases average 36.6 characters (73% longer than 16), sentences 83.4
  (83%).
- The prior's normalised entropy averages 0.152 at a word's first position,
  0.093 inside it and 0.298 at its last. Cutting where its rise along the
  text is in the top 1/U of positions, at a mean unit of 6.0 characters,
  puts 57.2% of the cuts on a word's last position and cuts 54.0% of word
  ends: a noisier word rule.
- The word tier's gain doesn't follow the prior's uncertainty: within each
  place in a word it is flat, or smaller where the prior is least sure.
  The place in the word explains the large gain at a word's last position,
  not the entropy. That corrected a reading of @air's.

### 2026-11-15: with own step (slot 50)

| | word, own step | word + phrase, own step |
|---|---:|---:|
| held-out, best (epoch 20) | 3.1155 | 3.1140 |

Paired, phrase own against word own: −0.05% ±0.06 overall; −0.43% ±0.13 at
a word's first character; −0.56% ±0.44 on a sentence's first word; −0.70%
±0.48 before a new phrase. Before a new sentence +0.03% ±0.80: the
sentence-end gain comes from the word tier alone.

## What we learned

- A phrase tier adds nothing beyond the word tier with own step. The next
  levels tried, sentence and paragraph tiers with longer backpropagation
  through the tier, are in `topic-tiers`.
- Entropy rises are not a better unit than words.
