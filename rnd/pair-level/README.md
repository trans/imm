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
  Widening the decoder helps most (6.961 at window 256), but at the GRU
  128's compute the pair model still loses: 6.753 against 6.012 (+12.3%),
  and +16.0% against a GRU of its own size (5.820).
opened:   2026-11-21
updated:  2026-11-22
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

### Added after the by-place scores: the decoder, and equal compute

The user approved the three follow-ups.
- **The screen, at window 256** (`run_variants.sh`): four shapes, each at
  the GRU 128's size, at 3e-4, 1e-3 and 3e-3. Against the first pair
  model (u96/l42, 7.261) and the GRU (6.012):

  | shape | level 1 | decoder | feed | parameters | multiply-adds a character |
  |---|---:|---:|:---:|---:|---:|
  | wider decoder | 80 | 64 | | 92,992 | 58,240 |
  | wider still | 66 | 80 | | 93,192 | 64,086 |
  | feed | 90 | 42 | yes | 92,990 | 51,588 |
  | both | 75 | 64 | yes | 93,261 | 58,358 |

  - "Feed" (`--feed`) adds a projection of level 1's state to the
    decoder's input at both steps, as MEGABYTE adds its global output to
    each local input. Without it, level 1's state only starts the decoder.
  - Each shape passes the leak and windowed-equals-whole checks. The
    first pair model starts from the same weights as before.
- **Equal compute:** the best shape, scaled up to the GRU 128's ~86k
  multiply-adds a character (the first pair model did ~52k), at windows
  256 and 4, at the same rates. It is larger than the GRU in parameters,
  so it is read against both: the GRU of equal compute (128) and of
  equal size.

### Added after the screen: the equal-compute arms

The screen's best shape was the widest decoder, level 1 66 and decoder 80
(6.961; level 1 75 and decoder 64 with feed was within noise, 6.971).
Scaled up in the same ratio to the GRU 128's compute (`run_compute.sh`):

| model | width | parameters | multiply-adds a character |
|---|---|---:|---:|
| pair model | level 1 80, decoder 98 | 122,266 | 85,916 |
| character GRU, equal compute | 128 | 93,024 | 86,016 |
| character GRU, equal size | 154 | 122,664 | 115,500 |

- The pair model and the GRU 154 run at windows 256 (3e-4, 1e-3, 3e-3)
  and 4 (3e-5, 1e-4, 3e-4). The GRU 128's runs are in hand.
- Three lanes at once (another project's slot shares the machine), the
  short arms first.

## How to rerun

From the repo, on CPU: `sh rnd/pair-level/run.sh`. It writes
`results/w<window>_lr<rate>.json` (the curve, the final score) and a log
for each arm. `tiny_pairs.py --help` gives the single-arm flags. The by-place reruns:
`sh rnd/pair-level/run_by_place.sh`. The shapes: `sh rnd/pair-level/run_variants.sh`. Equal compute:
`sh rnd/pair-level/run_compute.sh`.

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

### 2026-11-21: the decoder screen, window 256

`run_variants.sh`, CPU, two lanes, 23:38–23:52 EST. Every shape is the
GRU 128's size and best at 1e-3, bracketed by 3e-4 and 3e-3. Held-out
perplexity per character after 2,000,000 training characters, and by
place in the pair at the best rate:

| shape | level 1 | decoder | feed | 3e-4 | 1e-3 | 3e-3 | first | second |
|---|---:|---:|:---:|---:|---:|---:|---:|---:|
| first pair model | 96 | 42 | | 8.196 | **7.261** | 7.655 | 6.926 | 7.613 |
| wider decoder | 80 | 64 | | 7.963 | **6.998** | 7.315 | 6.737 | 7.268 |
| wider still | 66 | 80 | | 7.856 | **6.961** | 7.164 | 6.857 | 7.067 |
| feed | 90 | 42 | yes | 8.139 | **7.105** | 7.690 | 6.828 | 7.393 |
| both | 75 | 64 | yes | 7.846 | **6.971** | 7.433 | 6.776 | 7.171 |
| character GRU 128 | | | | 6.699 | **6.012** | 6.430 | 5.966 | 6.059 |

- **Widening the decoder helps most:** 7.261 → 6.998 → 6.961. It also
  closes most of the gap between a pair's first and second character.
- **Feeding level 1's state into every step helps a little:** 7.261 →
  7.105 with the narrow decoder, 6.998 → 6.971 with the wider one.
- **The best two are tied within noise** (6.961, 6.971). Both are still
  about 16% behind the GRU.
- Time: ~131–146 s an arm.

### 2026-11-21/22: equal compute

`run_compute.sh`, CPU, three lanes, 23:53–00:38 EST. The widest shape
scaled to the GRU 128's compute (level 1 80, decoder 98), against the GRU
of equal compute (128) and of equal size (154). Each best rate is
bracketed:

| window | model | parameters | multiply-adds a character | rates | scores | best |
|---:|---|---:|---:|---|---|---:|
| 256 | pair model | 122,266 | 85,916 | 3e-4, 1e-3, 3e-3 | 7.532, 6.753, 6.988 | **6.753** |
| 256 | GRU 128 | 93,024 | 86,016 | the same | 6.699, 6.012, 6.430 | **6.012** |
| 256 | GRU 154 | 122,664 | 115,500 | the same | 6.530, 5.820, 6.356 | **5.820** |
| 4 | pair model | 122,266 | 85,916 | 3e-5, 1e-4, 3e-4 | 8.340, 7.405, 7.538 | **7.405** |
| 4 | GRU 128 | 93,024 | 86,016 | the same | 7.136, 6.523, 6.816 | **6.523** |
| 4 | GRU 154 | 122,664 | 115,500 | the same | 6.955, 6.422, 6.794 | **6.422** |

- **At equal compute the pair model still loses:** +12.3% at window 256
  and +13.5% at 4, against the GRU 128. Against the GRU of its own size,
  +16.0% and +15.3%.
- **It gains from size and window about as the GRU does:**
  - From 93k to 122k parameters: −3.0% (6.961 → 6.753), against −3.2%
    for the GRU (128 → 154).
  - From window 4 to 256: −8.8%, against −9.4% for the GRU 154.
  - So no sign that its cheaper reach pays.
- **It learns worse, not just generalises worse:** the loss on the
  training text over the last 250,000 characters, 6.16 against 5.19 for
  the GRU 154 (window 256, 1e-3).
- **Its one advantage is speed:** at window 256 an arm took ~146 s,
  against ~364 s for the GRU 154 in the same lanes. It has half the
  sequential steps, and the pairs decode in parallel. At window 4, Python
  overhead evens them out (~1,045 s against ~1,100 s).
- The two places in a pair are now close for the pair model (6.619 and
  6.889 at window 256), as for the GRU (5.770 and 5.870).

## What we learned, or what's open

- Making level 1 the only way to the prediction costs every prediction
  something, at equal size and at equal compute. A hierarchy that forces
  the work to be divided does divide it, but this one pays for it.
- The decoder's width mattered most: from 7.261 to 6.961 at the same size.
  At the GRU's compute the gap is +12.3% (window 256).
- Its cheaper reach didn't show: a long window helps it no more than it
  helps the GRU.
- Its speed did: about 2.5 times faster an arm at window 256 than a GRU of
  its size, here on CPU. At equal wall-clock time it could be bigger. That
  comparison isn't run.
