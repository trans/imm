---
title:    A GRU-only binary tree
kind:     experiment
status:   running
outcome:  n/a
question: >-
  Without attention or memory records, can a stack of GRUs, each level
  stepping half as often as the one below, carry the text's longer reach
  well enough to match imm's attention models at the same size?
answer:   not yet run
opened:   2026-11-21
updated:  2026-11-21
code:     branch gru-tree, 8818093 (from attention-only 15d1a41); flags --tree-levels N --tree-width W
data:     PG-19 100M split (rnd/README.md)
slots:    none yet
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
