---
title:    Level 1 makes the prediction
kind:     experiment
status:   running
outcome:  n/a
question: >-
  If level 1, a GRU stepping once every two characters, makes the
  prediction itself through a small character decoder for each pair, with
  nothing passing between pairs except through it, does it predict as well
  as a plain character GRU of the same size, updated every 4 characters and
  every 256?
answer: >-
  Not yet run.
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

None yet.
