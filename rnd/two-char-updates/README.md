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
  Stages 3–4: with both levels updated together every 4 characters,
  level 1 still harms the GRU under it (6.866 against 6.523, +5.2%), and
  attaching it so its error trains that GRU changes almost nothing (6.953).
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

### Stage 3: both levels on one clock

Stage 2's level 1 hurt. Was it the two clocks, or how level 1 is attached?
The user asked for the test (`tiny_levels.py --char-window 4`):
- **The model:** stage 2's (208,609 parameters), with both levels updated
  together every 4 characters, from those 4 predictions. The character
  GRU's error reaches back 4 steps, level 1's through its 2 steps.
- **Only the clocks change:** level 1 still reads the character state
  detached, and its read enters as before.
- **The control:** a character GRU 215 (208,125) updated every 4
  characters (`tiny_tbptt.py --window 4 --hidden 215`). The character
  level now reaches back 4 steps, not 2, so stage 3 is read against its
  own control, not against stage 2.
- **Learning rates:** 3e-5 and 1e-4, with the next rate past either if a
  best falls at the edge.
- **Reading it:**
  - If level 1 beats the control here, the separate clocks were the
    trouble.
  - If it still loses, the trouble is how level 1 is attached: its
    detached read, or a read added in front of the output layer.
- **Memory:** another project's 5 GB job runs alongside, so the arms run
  two at a time, each about 0.9 GB (mostly PyTorch itself).
  - Held-out pieces are scored 2 at a time instead of 10. That changes
    only memory, not the numbers.
  - A guard stops the run if the machine's available memory falls under
    1.9 GB.

### Stage 4: level 1 attached

The user asked whether level 1 is held back by what it reads.
- **What it reads:** every second character, the character state, which
  with these short windows is a blend of about the last 2–3 characters.
- **Why that's the attachment:** detached, the character level alone
  decides what's in that state, for its own next prediction.
- **The test (the user's choice, `tiny_levels.py --char-window 4
  --attach`):** stage 3's model, with level 1 reading the character state
  undetached. Level 1's error then also trains the character GRU, back to
  the update's start (4 characters), so level 1 can shape its own input.
  - Checked on one window with the gate open: detached, level 1 sends no
    gradient into the character GRU; attached, it does.
- **Against what:** stage 3's control (GRU 215 every 4) and its detached
  model, at the same rates (3e-5, 1e-4).
- **Next, if pairs show any gain** (the user's plan): level 1 stepping
  every 3rd character, for comparison. Pairs cut across words, so later
  steps have to build on them, but they can pick up morpheme-sized pieces.
- **When:** right after stage 3, two arms at ~0.9 GB with the same memory
  guard. There isn't room to run it beside stage 3.

### Added after stages 3–4: the small GRU every 4 characters

Stage 2 showed level 1 *harms* by comparing it with the 128-wide character
GRU it sits on, alone. Stages 3–4 had only the same-size control, which
can't tell harm from a level that is merely useless (the wider GRU may
just use the parameters better). So the 128-wide GRU alone, updated every
4 characters (`tiny_tbptt.py --window 4`, 93,024 parameters), runs at 3e-5,
1e-4 and 3e-4 (`run3_h128.sh`).
- If the levels model beats it, level 1 adds something, just less than
  widening the GRU does.
- If it matches it, level 1 is useless.
- If it loses to it, level 1 harms, as in stage 2.

## How to rerun

From the repo, on CPU: `sh rnd/two-char-updates/run.sh` runs the six arms in
turn and writes `results/k<K>_lr<LR>.json` (the curve, the final score).
`tiny_tbptt.py --help` gives the single-arm flags. Stage 1 added a
seventh arm afterwards (window 2 at 3e-5, as its best rate was the lowest
tried); `run.sh` lists it. `python3 ngram_ref.py > results/ngram_ref.txt`
gives the counting references. Stage 2: `sh rnd/two-char-updates/run2.sh`.
Stage 3: `sh rnd/two-char-updates/run3.sh`; `tiny_levels.py`'s defaults
still run stage 2. Stage 4: `sh rnd/two-char-updates/run4.sh`. Both at
3e-4: `sh rnd/two-char-updates/run3-4_lr3e-4.sh`. The small GRU every 4:
`sh rnd/two-char-updates/run3_h128.sh`. The distance table:
`python3 pair_info.py > results/pair_info.txt`.

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


### 2026-11-21: stages 3 and 4, one clock, detached and attached

CPU, 20:19–22:40 EST: `run3.sh`, `run4.sh`, `run3_h128.sh` and
`run3-4_lr3e-4.sh`. Every arm updates every 4 characters. Held-out
perplexity per character after 2,000,000 training characters; every
model's best rate is 1e-4, bracketed on both sides:

| model | parameters | 3e-5 | 1e-4 | 3e-4 |
|---|---:|---:|---:|---:|
| character GRU 128 alone | 93,024 | 7.136 | **6.523** | 6.816 |
| the same + level 1, detached (stage 3) | 208,609 | 7.261 | **6.866** | 7.307 |
| the same + level 1, attached (stage 4) | 208,609 | 7.272 | **6.953** | 7.212 |
| character GRU 215 alone (same size) | 208,125 | 6.647 | **6.179** | 6.726 |

- **Level 1 harms the GRU it sits on, with one clock too:** +5.2%
  detached and +6.6% attached, against that GRU alone. Against the
  same-size GRU, +11.1% and +12.5%.
- **Neither the clocks nor the detach is the cause:**
  - With separate clocks (stage 2) the harm was +9.3%. At a 2-character
    window, though, so the size isn't comparable.
  - Attaching changes almost nothing: the two curves and gates nearly
    overlap (6.953 against 6.866, +1.3%). Level 1's path sends the
    character GRU little gradient.
- **It learns worse, not just generalises worse:** the loss on the
  training text as it was read, over the last 250,000 characters, 5.73
  detached and 5.78 attached, against 5.61 for the GRU alone (1e-4).
- **Its gate opens less with one clock,** and less the faster it learns:
  tanh(g) ends at 0.42 (3e-5), 0.28 (1e-4) and 0.15 (3e-4), against 0.68
  with separate clocks (stage 2, 3e-5).
- **The longer window pays:** the same-size GRU scores 6.179 updated
  every 4 characters against 7.123 every 2 (−13.3%).
- Time, 2–5 arms at once: ~1,620–1,780 s for a level-1 arm (~2,540 s at
  3e-4, five at once), ~1,205–1,260 s for the same-size GRU, ~1,450 s for
  the 128 GRU (five at once).
- Each process opened a CUDA context through torch's Adam step, which
  held another project's slot; the scripts now hide the GPU (`bd646b2`).

**Reading it** (the lead agent's, untested): in these setups level 1
never has anything the character GRU lacks. It reads the character state,
and with one clock its error reaches no further in characters than the
character GRU's own. So the best it could do is nothing; that it harms
suggests the extra path makes learning worse. Triplets, the user's next
step if pairs showed any gain, are not run. Ideas raised in the
discussion, none run:
- Level 1 making the prediction itself, with a small character decoder
  for each pair and no character state carried between pairs (as
  MEGABYTE does), so the levels must divide the work.
- Keeping the tree's nodes so a query can search them, instead of
  reading each level's latest state.

### 2026-11-21: reference, one character d back

From the discussion of levels that each look only at the character d
back. `pair_info.py` (`results/pair_info.txt`) measures how much the
single character d back tells about the next one, on the same first 2M
training characters. Each figure is the perplexity per character from
knowing only that one character; knowing nothing gives 23.08.

| d | 1 | 2 | 3 | 4 | 8 | 16 | 32 | 128 | 1,024 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| perplexity | 12.41 | 17.90 | 20.59 | 21.76 | 22.74 | 22.93 | 23.00 | 23.02 | 23.05 |

- **Close by, characters count together.** Even if the four nearest
  characters' effects simply added up (they needn't), they'd give
  0.62 + 0.25 + 0.11 + 0.06 = 1.05 nats. Counting the four together gives
  1.44 (23.08 → 5.47 held-out, `ngram_ref.txt`). That's a rough
  comparison, not a bound.
- **Far back, one character says almost nothing:** from 32 characters on,
  it improves on chance by about 0.1%. What lasts that far lives in
  combinations of characters (a name, a word), which no single character
  carries.
