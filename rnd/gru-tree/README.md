---
title:    A GRU-only binary tree
kind:     experiment
status:   concluded
outcome:  negative
question: >-
  Without attention or memory records, can a stack of GRUs, each level
  stepping half as often as the one below, carry the text's longer reach
  well enough to match imm's attention models at the same size?
answer: >-
  Not as built. At epoch 3 the tree scores 4.016 against 3.664 for a plain
  character GRU of the same size (+9.61% ±0.13 per character), and only its
  two lowest levels open their gates (−0.27 and +0.15); levels 3–10 stay
  within ±0.01 of zero. Its loss is spelling (inside words +11.6%), which
  its narrower character GRU (192 against 498) explains, and nothing late
  in a piece improves.
opened:   2026-11-21
updated:  2026-11-21
code:     branch gru-tree, 8818093 (from attention-only 15d1a41); flags --tree-levels N --tree-width W
data:     PG-19 100M split (rnd/README.md)
slots:    147 (tag rnd/gru-tree/slot147 at 8818093)
tags:     [hierarchy, gru, no-attention]
related:  [attention-only-words, architecture-1.59m, simplified-no-words]
---

# A GRU-only binary tree

## Question

imm's word level (a word-stepping GRU and word records) is worth about 0.5%
at 1.59M (the `simplified-no-words` entry). Word records gave no gain in the
attention-only model either (the `attention-only-words` entry). The user chose
to step back to simpler designs. The first is a hierarchy of GRUs alone, with
fixed spans instead of words.

## Design

- **Level 0** is the character GRU, as in imm.
- **Level k** (1 … N) steps at the characters p where (p + 1) mod 2^k = 0,
  with p counted from the start of the piece. Its input is level k−1's
  output at the same character, so each level reads pairs of the one below:
  level 1 every 2 characters, level 10 every 1,024.
- **Carry and reset:** every level's state is carried from update to update,
  and starts fresh when a stream begins a new piece.
- **Reading down:** each character's prediction reads, from every level, the
  state after that level's latest step at or before the character (so
  causal). Each level's state goes through a projection and a gate that
  starts at zero, and the sum is added to the character GRU's state before
  the output layer.
- **None of the rest:** no attention blocks, no MLP blocks, no word or
  character records, no word tier.
- **Size:** near 1.59M trained parameters, to compare with:
  - slot 127, the full imm: 3.0552;
  - slot 130, the character and word GRUs without attention: 3.3473;
  - the 1.84M transformer: 3.1862.

The user's design is the tree of GRUs. The step rule, the carry and the
read-down are the lead agent's choices, open to change.

**A known limit:** a training update gives each stream about 370 characters
(64 words). Levels that step every 256 characters or more step at most once
or twice within an update. Under truncated backpropagation, their own
recurrence gets little gradient.

## How to rerun

**The first screen** (approved 2026-11-21): `tree_slot.sh <slot>`, from a
worktree at `8818093`, tag `rnd/gru-tree/slot<N>` once it runs.
- It runs `probe_tree.py` (each arm's `--max-update-chars` cap and speed).
- Then two arms, each stopped after epoch 3 of the 8-epoch schedule:
  - **T:** the tree, h192, 10 levels of width 132;
  - **G:** the plain character GRU at the same size, h498, no tree: the
    control.

The model's flags, on top of the standard run's (`base` in the
`half-lr-10.8m` entry's `lr10_slot.sh`, without `--tier-own-step` and
`--char-memory`, which the tree refuses):

```text
--hidden-size 192 --embedding-size 192 --attention-layers 0 --no-word-records
--tree-levels 10 --tree-width 132
```

Scripts here:
- `size_tree.py` and `size_plain.py` count the trained parameters of tree
  and plain-GRU configs on CPU (`results/sizes.txt`).
- `steps_per_update.py` counts how often each level steps in a training
  update (`results/steps_per_update.txt`).
- `env.sh` and `rss.py` wrap them in the machine's limits.

## Results

### 2026-11-21: built, not yet run

`8818093` on `gru-tree`, built by a subagent of the lead session.

**Choices the build made:**
- **Start state:** under `--documents`, each level starts a piece from a
  learned state, as the character GRU does. Without `--documents` the levels
  start from zeros.
- **Schedule:** the place of an update's first character (mod 2^N) is
  carried with the state, so where an update is cut doesn't change the
  schedule. It is reset at a new piece. N is capped at 24, so the count stays
  exact in float32.
- **Inputs and reads:** level k reads level k−1's GRU output. The reads down
  come from the levels' GRU states.
- **One GRU call per level per update,** of ceil(T / 2^k) steps.
- **GRU only:** `--attention-layers 0 --no-word-records`, with no character
  memory. Without `--tree-levels`, that is the plain character GRU.

**Tests:** `tests/test_gru_tree.py`, 17 tests.
- It matches a slow step-by-step reference at 7 levels.
- One, two or three updates give the same result, with cuts mid-pair and
  mid-block.
- It is causal; with the gates at zero it equals the plain character GRU;
  a new piece starts every level fresh.
- It also runs under bf16 and scores held-out pieces.
- Seven deliberate breaks (off-by-one steps, reading a level's next step,
  no carry, no reset, and others) were each caught.
- The full suite passes: 268 tests, one skipped.

**Sizes** (trained parameters; slot 127 has 1,588,157):

| config | trained parameters |
|---|---:|
| H=E=192, 10 levels, width 132 (slot 127's character GRU) | 1,593,298 |
| H=E=256, 10 levels, width 115 | 1,591,421 |
| H=E=192, 8 levels, width 149 | 1,583,109 |
| plain character GRU, H=E=496 (no tree) | 1,574,896 |
| plain character GRU, H=E=500 (no tree) | 1,599,596 |

At H=192 the tree holds 1.33M of the 1.59M parameters; the character GRU
holds 0.22M.

**Steps per training update,** measured on the first 10M training
characters:
- A stream reads 364 characters per update on average (1st–99th
  percentile 307–464).
- Levels 1–7 step at least twice in nearly every update.
- Level 8 steps once in 59% of updates.
- Level 9 steps at most once, and level 10 not at all in 65% of updates.
- So levels 9 and 10 never get gradient through their own recurrence.

### 2026-11-21: the first screen, slot 147

`tree_slot.sh 147` at `8818093`, granted 20:11 UTC, ended 20:39 UTC. The
probe found both arms fit at the standard cap (64 × 576 characters):
- T: 72 ms an update, peak 1.3 GiB;
- G: 40 ms, peak 1.9 GiB.

Held-out perplexity per character, each arm stopped after epoch 3 of 8:

| arm | trained | e1 | e2 | e3 | train loss, end of e3 |
|---|---:|---:|---:|---:|---:|
| T: tree, h192, 10 levels × 132 | 1,593,298 | 4.260 | 4.088 | **4.016** | 1.3048 |
| G: plain character GRU, h498 | 1,587,222 | 3.870 | 3.736 | **3.664** | 1.2062 |

At epoch 3, for reference:

| model | e3 |
|---|---:|
| slot 127, the full imm | 3.233 |
| slot 130, depth records | 3.178 |
| slot 142, attention alone | 3.423 |
| slot 130, the character and word GRUs with their MLP blocks, no attention (1,067,069 trained) | 3.532 |

**T against G** (`results/tree_vs_gru.txt`, T's change): +9.61% ±0.13.
- The loss is spelling: inside words +11.64%, lowercase letters +13.25%
  (98% of the gap), new words +13.81%.
- Small gains where G is weakest: uppercase −4.82%, and characters the
  prior finds hardest (its loss ≥ 3 nats) −2.8 to −7.6%.
- No reach: the gap is +7.46% in a piece's first 256 characters and
  +9.65% in its last 2,500; repeats more than 256 words back +11.45%.

**The tree's gates** (tanh of each level's gate, level 1 first):

```text
epoch 1: -0.2736  0.1312  0.0014  0.0003  0.0000 -0.0014  0.0002 -0.0011 -0.0005  0.0019
epoch 3: -0.2669  0.1482 -0.0098  0.0011  0.0024  0.0025 -0.0027  0.0004 -0.0004 -0.0011
```

Only levels 1 and 2 are read. Levels 3–7 step 3 to 45 times in every
training update, so the ~364-character updates don't explain their closed
gates.

**Reading it** (the lead agent's, untested):
- The model spent 1.33M of its 1.59M parameters on levels it doesn't
  read, so T is in effect a 192-wide character GRU with two small levels,
  and it loses to a 498-wide one at spelling.
- **The levels have only a linear way in.** Each level's read is added to
  the character GRU's state just before the output layer, which is
  linear. The character GRU never reads the levels, and nothing
  nonlinear combines them with it. Slot 130's GRU model, which had MLP
  blocks between its state and the output, scored 3.532 with fewer
  parameters than either arm here.
- One seed; a screen.

