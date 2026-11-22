---
title:    Level 1 makes the prediction
kind:     experiment
status:   concluded
outcome:  negative
question: >-
  If level 1, a GRU stepping once every two characters, makes the
  prediction itself through a small character decoder for each pair, with
  nothing passing between pairs except through it, does it predict as well
  as a plain character GRU of the same size, updated every 4 characters and
  every 256?
answer: >-
  No. At the same size (93,106 against 93,024) the pair model scores 7.975
  against 6.523 updated every 4 characters (+22.3%), and 7.261 against
  6.012 every 256 (+20.8%). It is worse on the training text too. The
  long window helps it a little more (−9.0% against −7.8%), from a much
  worse start. Most of the loss is in each pair's second character (+25.7%,
  against +16.1% for the first), which points at the narrow decoder.
opened:   2026-11-21
updated:  2026-11-21
code:     standalone, this directory (tiny_pairs.py); no branch
data:     PG-19 100M split (rnd/README.md), its first 2M training characters
slots:    none (CPU, ~0.7 GB an arm, GPU hidden)
tags:     [gru, levels, training]
related:  [two-char-updates, gru-tree]
---

# Level 1 makes the prediction

## Question

In `two-char-updates`, level 1 sat on top of a character GRU and read its
state. That level harmed the GRU in every form tried: on its own clock, on
one clock, detached or attached. It never had anything the character GRU
lacked. The character GRU carried the whole state and sat on the shortest
path to the prediction, so nothing made the two divide the work.

The user's next step: let level 1 make the prediction instead. A level that
steps once every two characters predicts the next two. Here it does that
through a small decoder that spells out the pair, as MEGABYTE does with its
patches. Two things change:
- **The work must be divided.** The decoder knows only level 1's state and
  the characters already in the current pair. Anything older reaches the
  prediction only through level 1.
- **Its reach is cheaper.** Level 1's error reaches back through its own
  steps, and each step covers two characters. The same text takes half the
  sequential steps, and the pairs decode in parallel.

## Design

- **Model (`tiny_pairs.py`, 93,106 parameters):**
  - **Input:** an embedding of 64, read in pairs counted from the piece's
    start (characters 2j and 2j+1).
  - **Level 1:** a GRU 96 wide. It steps once a pair, after both of the
    pair's characters, reading their two embeddings side by side.
  - **The decoder:** a GRU 42 wide that starts from a projection of level
    1's state. Its first step, from a learned start input, predicts the
    pair's first character. Its second step, reading that character,
    predicts the second. It starts afresh for every pair.
  - **Output:** a linear layer 42 → 96.
- **The control:** the 128-wide character GRU alone (93,024 parameters),
  with results already in hand at the same windows:
  - every 4 characters (`two-char-updates`, stages 3–4): 7.136, **6.523**,
    6.816 at 3e-5, 1e-4, 3e-4;
  - every 256 (`two-char-updates`, stage 1): 6.699, **6.012**, 6.430 at
    3e-4, 1e-3, 3e-3.
- **Arms:** window 4 at 3e-5, 1e-4 and 3e-4, and window 256 at 3e-4, 1e-3
  and 3e-3, the control's rates. If a best falls at an edge, the next rate
  past it runs too.
- **Training and scoring:** as in `two-char-updates`. One stream reads the
  first 2,000,000 training characters once. The level-1 state is carried,
  detached at each update and reset to zeros at a piece's start. Adam at
  a constant rate, gradient norm clipped at 1.0.
  - The 50 held-out pieces are scored each from a zero state: perplexity
    per character over characters 2 to 5,000 of each.
- **Reading it:**
  - **At window 4,** both reach 4 characters. The test is whether the
    bottleneck costs: everything older than the current pair must pass
    through level 1.
  - **At window 256,** both reach 256 characters. Level 1 gets there in
    128 steps against the GRU's 256. The test is whether a level whose
    steps cover two characters learns more from a long window.
- **Checked before running:**
  - **No leak:** changing character p never changes the predictions of
    character p or earlier, and always changes the next one.
  - **Windowed equals whole:** training's windowed forward, with the
    carried state, gives the same predictions as scoring the whole piece,
    within 1.2e-7 at window 4 and exactly at 256.
- **Machine:** four lanes at once, ~0.7 GB each. A guard stops them if
  the machine's available memory falls under 1.9 GB. The GPU is hidden,
  so no CUDA context opens.

### Added after the first results: where the loss sits

The pair model lost by about 21% at both windows. Is the loss in the first
character of each pair, predicted from level 1's state alone, or in the
second, which also reads the first? Both best 256-window arms run again
(pair model and character GRU, 1e-3, `run_by_place.sh`). Their final scores
are also given over even and odd places: a pair's first and second
character. The rerun should reproduce each final score.

## How to rerun

From the repo, on CPU: `sh rnd/pair-level/run.sh`. It writes
`results/w<window>_lr<rate>.json` (the curve, the final score) and a log
for each arm. `tiny_pairs.py --help` gives the single-arm flags. The by-place reruns:
`sh rnd/pair-level/run_by_place.sh`.

## Results

### 2026-11-21: windows 4 and 256

`run.sh`, CPU, four lanes, 23:01–23:20 EST. Held-out perplexity per
character after 2,000,000 training characters:

| window | model | parameters | rates | scores | best |
|---:|---|---:|---|---|---:|
| 4 | pair model | 93,106 | 3e-5, 1e-4, 3e-4 | 9.022, 7.975, 8.001 | **7.975** |
| 4 | character GRU 128 | 93,024 | the same | 7.136, 6.523, 6.816 | **6.523** |
| 256 | pair model | 93,106 | 3e-4, 1e-3, 3e-3 | 8.196, 7.261, 7.655 | **7.261** |
| 256 | character GRU 128 | 93,024 | the same | 6.699, 6.012, 6.430 | **6.012** |

- **The pair model loses by about a fifth at both windows:** +22.3% at
  4, +20.8% at 256. Each best rate is bracketed, the GRU's included.
- **It learns worse, not just generalises worse:** the loss on the
  training text as it was read, over the last 250,000 characters, 7.17
  against 5.61 (window 4) and 6.76 against 5.38 (window 256).
- **The cheaper reach pays only a little:** the long window improves the
  pair model by 9.0% and the GRU by 7.8%.
- **Where it's worse, against counting:** at window 4 it scores about
  what counting 2 characters back does (8.03, `two-char-updates`).
- Time: ~1,110 s for a window-4 arm, ~180–195 s for a window-256 arm.

### 2026-11-21: where the loss sits

`run_by_place.sh`, 23:22–23:27 EST. The two best 256-window arms again.
They reproduce their scores (7.261, and 6.012 as in `two-char-updates`
stage 1). The final held-out scores by place:

| place in the pair | pair model | character GRU 128 | gap |
|---|---:|---:|---:|
| first (from level 1's state alone) | 6.926 | 5.966 | +16.1% |
| second (also reads the first) | 7.613 | 6.059 | +25.7% |

- **The GRU scores both places alike** (5.97, 6.06): pairs mean nothing
  to it.
- **The pair model loses most on the second character,** the one with
  the most local help: it reads the first directly. The decoder, 42 wide,
  is the weakest spot. It gets level 1's state only through one 96 → 42
  projection.
- **The first character is +16% worse too,** so level 1 is short as well,
  or that same narrow path is.

## What we learned, or what's open

- Making level 1 the only way to the prediction, at the same size, costs
  about a fifth. A hierarchy that forces the work to be divided does
  divide it, but this one pays for it in every prediction.
- Open, none run:
  - a wider decoder and a narrower level 1 at the same size;
  - level 1's state fed into every decoder step, as MEGABYTE adds its
    global output to each local input, not just used to start it;
  - comparing at the same compute rather than the same parameters (the
    pair model does about 60% of the GRU's arithmetic a character: ~52k
    multiply-adds against ~86k).
