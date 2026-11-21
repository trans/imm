---
title:    Weight updates every two characters
kind:     experiment
status:   concluded
outcome:  negative
question: >-
  If a character GRU's weights are updated every 2 characters, so the error
  reaches back only 2 steps, how much worse does it predict than with a
  256-character window?
answer: >-
  Stage 1: updated every 2 characters, the tiny GRU scores 7.473 against
  6.012 with a 256-character window (+24.3%), between counting models that
  see 2 and 3 characters back (8.03, 6.11). Stage 2: level 1 on top,
  updated on its own two steps, makes it worse: 8.165, +9.3% against the
  2-character GRU alone and +14.6% against one of the same size (7.123),
  though its gate opens to 0.68. It is worse on the training text too.
opened:   2026-11-21
updated:  2026-11-21
code:     standalone, this directory (tiny_tbptt.py); no branch
data:     PG-19 100M split (rnd/README.md), its first 2M training characters
slots:    none (CPU, about 1 GB)
tags:     [gru, truncation, training]
related:  [gru-tree]
---

# Weight updates every two characters

## Question

imm trains with truncated backpropagation: the state carries from update to
update, but the error from a prediction reaches back only to the start of its
update. In the `gru-tree` discussion the user asked for the simplest case: one
stream, a plain character GRU, its weights updated every other character. How
bad is it?

The state still carries the whole way, so the GRU reads the same text either
way. What changes is what it can learn: with a 2-character window, nothing
ever rewards keeping something in the state for more than 2 steps.

## Design

- **Model:** a tiny character GRU: embedding 64, GRU 128, a linear output
  layer; 93,024 parameters. No dropout.
- **Text:** one stream reads the first 2,000,000 characters of the 100M
  split's training text (its first 80 pieces of 25,000), once. The state is
  carried and detached between updates and reset to zeros at each piece's
  start; no update crosses a piece's start.
- **Updates:** every K characters, the mean loss over those K predictions,
  Adam at a constant learning rate, gradient norm clipped at 1.0.
  - **K = 2** (the question), at learning rates 1e-4, 3e-4 and 1e-3: it makes
    a million updates, so a smaller rate may suit it.
  - **K = 256** (the control), at 3e-4, 1e-3 and 3e-3.
  - Each window is read at its best learning rate.
- **Score:** the 50 held-out pieces, each from a zero state, every 250,000
  training characters: perplexity per character over characters 2 to 5,000
  of each piece (249,950 predictions). The model is tiny and sees 2M
  characters, so the numbers aren't comparable to imm's runs, only to each
  other.
- **Reference:** `ngram_ref.py` scores counting models that see only the
  last 0 to 5 characters, trained on the same 2M characters and scored the
  same way (interpolated absolute discounting).

### Stage 2: level 1 on top

The user's next step: add level 1 over the 2-character GRU, updated on its
own two steps (`tiny_levels.py`).
- **Level 1,** a GRU 128 wide, steps at every second character, reading the
  character GRU's state there.
- **Its updates:** every 2 of its own steps, so every 4 characters, at
  every other character update, from those 4 predictions. Its error reaches
  back through its 2 steps.
- **It can't reach into the character GRU:** those weights change between
  its two steps, so it reads the character state detached.
- **Read:** each prediction adds level 1's state after its latest step,
  projected 128 → 128 and times tanh of a gate that starts at 0, to the
  character GRU's state in front of the output layer, as `gru-tree` does.
- **Size:** 208,609 parameters, against the 2-character GRU's 93,024. The
  control is a character GRU of the same size (hidden 215, 208,125),
  updated every 2 characters.
- Each at learning rates 3e-5 and 1e-4, the two best for the 2-character
  GRU in stage 1.

## How to rerun

From the repo, on CPU: `sh rnd/two-char-updates/run.sh` runs the six arms in
turn and writes `results/k<K>_lr<LR>.json` (the curve, the final score).
`tiny_tbptt.py --help` gives the single-arm flags. Stage 1 added a
seventh arm afterwards (window 2 at 3e-5, as its best rate was the lowest
tried); `run.sh` lists it. `python3 ngram_ref.py > results/ngram_ref.txt`
gives the counting references. Stage 2: `sh rnd/two-char-updates/run2.sh`.

## Results

### 2026-11-21: stage 1, windows 2 and 256

`run.sh`, CPU, 16:30–17:48 EST, then the 3e-5 arm 17:49–18:11. Held-out
perplexity per character after 2,000,000 training characters:

| window | updates | 3e-5 | 1e-4 | 3e-4 | 1e-3 | 3e-3 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1,000,000 | **7.473** | 7.506 | 8.212 | 10.562 | |
| 256 | 7,840 | | | 6.699 | **6.012** | 6.430 |

- **The 2-character window costs +24.3%** at each window's best rate
  (7.473 against 6.012).
- **It sees about 2–3 characters back.** Counting models that see only the
  last 0–5 characters (`results/ngram_ref.txt`) score 23.91, 12.81, 8.03,
  6.11, 5.47 and 5.50. The 2-character GRU falls between 2 characters back
  (8.03) and 3 (6.11).
- **It stops learning sooner.** At 1e-4 it reached 7.47 at 1.25M characters
  and 7.51 at 2M; at 3e-5 it was still falling (7.54 → 7.47), as was the
  256 window (6.07 → 6.01).
- **Both GRUs are undertrained:** counting the last 4 characters (5.47)
  beats them. The result gives the cost's direction and rough size, not
  either window's ceiling.
- Every arm bumps up at 1.5M characters: the one stream reads a stretch of
  text unlike the held-out pieces.
- Time: 1,240–1,320 s for each 2-character arm, ~275 s for each 256 arm.

### 2026-11-21: stage 2, level 1 on top

`run2.sh`, CPU, the four arms side by side, 18:12–18:55 EST. Held-out
perplexity per character after 2,000,000 training characters:

| model | parameters | 3e-5 | 1e-4 |
|---|---:|---:|---:|
| character GRU 128 + level 1, each on its own two steps | 208,609 | **8.165** | 8.712 |
| character GRU 215, every 2 characters (same size) | 208,125 | **7.123** | 7.263 |
| character GRU 128, every 2 characters (stage 1) | 93,024 | **7.473** | 7.506 |
| character GRU 128, every 256 characters (stage 1, at 1e-3) | 93,024 | 6.012 | |

- **Level 1 makes it worse:** +9.3% against the 2-character GRU it sits
  on, +14.6% against a character GRU of the same size.
- **It is used:** its gate, tanh(g), rose steadily: 0.26, 0.36, 0.44, 0.51,
  0.57, 0.62, 0.65, 0.68 at each 250,000 characters (3e-5).
- **It learns worse, not just generalises worse.** The loss on the
  training text as it was read, over the last 250,000 characters: 7.17
  with level 1 against 6.78 without (3e-5), and 6.71 against 5.76 for the
  same-size GRU (1e-4).
- Time: ~2,535 s for each level-1 arm, ~2,200 s for each same-size control,
  four at once.

**Reading it** (the lead agent's, untested): the two levels learn on
different clocks. The character level changes its weights and the output
layer every 2 characters, to fit predictions that include level 1's read;
level 1 changes every 4, to fit a character level that has since changed
once. Level 1 also gets only a detached copy of the character state, so
all it can add is what it carries across its own two steps. A test: the
same model with both levels updated together every 4 characters, against
a character GRU updated every 4.

