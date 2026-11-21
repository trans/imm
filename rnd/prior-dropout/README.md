---
title:    Taking the prior away, and prior dropout
kind:     experiment
status:   concluded
outcome:  negative
question: >-
  How much has imm's residual learned of the text itself? With the prior
  removed, what does it score, and does training with prior dropout make
  it learn the text as @air's transformers did?
answer: >-
  Little of the text itself. Without the prior, a fresh output layer on
  imm's frozen states scores 10.26 (books prior) and 9.24 (own prior) on
  the 4.75M split: better than character frequencies (23.9), but far
  from a plain transformer's
  3.74. The residual learns what the prior gets wrong. Prior dropout 0.5
  on the 20M split makes its own head score 3.6177 with no prior, 1.4%
  behind the plain 1.84M transformer, but costs 4.3% with the prior
  (3.2865 against 3.1517). Not adopted.
opened:   2026-11-12
updated:  2026-11-12
code: >-
  --prior-ablation-output at 733f101 (branch prior-ablation);
  --prior-dropout at 57f1033 (branch prior-dropout). Branches deleted; the
  commits are on main.
data: >-
  PG-19 4.75M split (data/pg19/imm-split) for the take-away test, on the
  h256 runs of pg19-4.75m-split (mobius's pg19split_gated_d16_pg200m prior,
  and imm's own depth-8 prior). PG-19 20M split
  (data/pg19/imm-split-20m) with pg19split20m_gated_d16_pg200m for prior
  dropout. Character frequencies score 23.9 on the held-out text.
slots:    one, imm's first slot from @air (its number was not recorded)
tags:     [prior, ablation, prior-dropout, pg19]
related:  [pg19-4.75m-split, blocks-20m, transformer-prior-pg19, learned-prior-strength]
---

# Taking the prior away, and prior dropout

## Question

The user asked (2026-11-11, via @air): if the prior were taken away, how
would the model perform, and how much has the residual GRU, or the
transformer, learned? @air's transformers with the prior learned only what
the prior gets wrong, unless trained with prior dropout. Is imm the same?

## Design

- **Take-the-prior-away test** (`--eval-only --prior-ablation-output`,
  `733f101`), on a trained checkpoint:
  - scored in full;
  - the residual's own output, with the prior still in its input;
  - its own output with no prior anywhere (the log-probs and depth
    evidence removed from the input too);
  - a fresh output layer (Linear(256, 96), started from the trained head)
    trained 3,000 steps of 4,096 on the frozen final states of 1,008,441
    training characters, Adam 3e-3 → 1e-4, as @air's `probe.py`; with the
    prior in the input and without.
- **Prior dropout** (`--prior-dropout 0.5`, `57f1033`): each training
  stream loses the prior from its input and output with probability 0.5
  per update, and is scored on its own head alone, with no gate, state
  features or L2 leash. Held-out scoring always has the prior. The 20M base
  run (`blocks-20m`) otherwise.

## How to rerun

Scripts here, as they ran; paths are the original machine's.

- The 4.75M test: `prior_ablation.sh`, on the CPU, from a worktree at
  `733f101`. Proposed tag `rnd/prior-dropout/ablation-4.75m`.
- The 20M base run's test: the first part of `pg19_20m_queue.sh`, on the
  CPU, at `57f1033`. The rest of that script (prior dropout on the GPU) was
  superseded when GPU time moved to slots.
- Prior dropout: `../blocks-20m/pg19_20m_slot.sh pd`, at `57f1033`, in
  imm's first slot from @air. Proposed tag `rnd/prior-dropout/pd05`.

## Results

### 2026-11-12: the take-away test, 4.75M split

| scored as | h256, books prior | h256, imm's own prior | 1.84M transformer, books prior in input (20M split, @air) |
|---|---:|---:|---:|
| in full | 3.3220 | 3.7059 | 3.0989 |
| own output, prior still in the input | 67.48 | 44.64 | |
| own output, no prior | 101.12 | 95.62 | |
| fresh layer, prior in the input | 5.37 | 4.49 | 3.56 |
| fresh layer, no prior | 10.26 | 9.24 | 16.0 |
| same size, trained with no prior | (transformer on 4.75M: 3.7364) | | 3.5689 |

- imm is like @air's transformers without prior dropout: it learns what
  the prior gets wrong, not the text. Without the prior its state is worth
  9–10, far from the plain transformer's 3.74.
- The stronger prior leaves less in the state (10.26 against 9.24).
- The transformer's state carries the input prior's prediction better
  (3.56 against 5.37), while imm's carries more without it (10.3 against
  16.0). Different splits, so a rough comparison.

### 2026-11-12: the 20M base run, and prior dropout 0.5

| scored as | prior dropout 0.5 | base run (no prior dropout) |
|---|---:|---:|
| in full | 3.2865 (1.7166 bits/char) | 3.1517 |
| own head, no prior anywhere (the path dropped streams train) | **3.6177** | 98.06 (own output) |
| fresh layer, no prior | 3.6623 | 9.42 |
| fresh layer, prior in the input | 4.67 | 4.29 |

The base run, own output with the prior in its input: 62.86.

- **Prior dropout costs 4.3% with the prior in** (+0.061 bits/char), twice
  @air's 0.03 at 1.84M. The gap held at 0.12–0.14 PPL from epoch 6 on.
- Without any prior, imm's own head scores 3.6177, 1.4% behind the plain
  1.84M transformer (3.5689). @air's 1.84M logits-only network with prior
  dropout was 0.029 bits/char ahead of plain.
- For imm the own head is the better no-prior measure: the fresh layer
  starts from it and drifts up in training.
- 6–7 minutes an epoch, about 2 hours in all.

## What we learned

- With a prior in its output, imm's residual becomes a corrector of the
  prior. That is fine for the score, but it means the residual alone has
  learned little English.
- Prior dropout fixes that at a 4.3% cost to the real score, so it was not
  adopted. This agrees with `research-lines.md`, which lists prior dropout
  under "Don't retry" from the earlier rnn-agpt work ("trains the wrong
  problem").
- The question of a model that learns the text itself came back later:
  the learned prior strength (`learned-prior-strength`), the no-prior row
  (slot 58), and finally the user's choice to drop the prior altogether
  (slot 101, the `no-prior-pair-1.8m` entry).
