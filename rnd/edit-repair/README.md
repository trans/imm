---
title:    Edit-neighbour repair of noisy contexts
kind:     experiment
status:   parked
outcome:  mixed
question: >-
  When the input context is corrupted, can the count prior recover by
  borrowing from nearby real contexts (edit neighbours), and does that
  smoothing cost anything on clean text?
answer: >-
  A gated fuzzy-node repair helps a little, and only on corrupted
  contexts: substitutions at p=0.20 go from 4.464 to 4.377, at p=0.05 from
  3.978 to 3.971. A repair trained at p=0.5 and scored at p=0.05 is worse
  (3.978 to 3.999). The full edit-kernel results are invalid: that gate
  never trained (a bug fixed in 15efc86).
opened:   2026-06-15
updated:  2026-11-07
code: >-
  not in git as it ran (agpt-ultra, June 2026). Ported into imm's first
  commit 15efc86 on main, which also fixes the edit-kernel gradient:
  scripts/run_edit_ablation_prior.py, imm/edit_kernel_prior.py.
data: >-
  Tiny Shakespeare canonical split 4fa9aec1db6b3aea, 10,000 held-out
  positions. imm's depth-8 count prior (exact prior on clean input:
  3.831 over these positions). No neural residual.
slots:    none (before the slot system)
tags:     [prior, noise, edit-distance, repair, bug]
related:  [typo-noise, skip-horizons]
---

# Edit-neighbour repair of noisy contexts

## Question

An exact count lookup fails when the context is misspelled or off the
corpus's paths: the longer contexts vanish. Could the prior find the
nearest real context instead, weighting candidates by edit cost and count?
And should all low-mass paths borrow from their higher-mass neighbours, or
only corrupted ones?

## Design

- **Corruption:** characters of the held-out context are changed with
  probability p: substitutions only, or all edit operations (substitute,
  delete, insert, transpose).
- **Modes** of `scripts/run_edit_ablation_prior.py`:
  - `fuzzy-node`: a gated mix of the exact node and its edit neighbours;
  - `correction`: a correction path beside the exact prior;
  - `full`: the full edit kernel inside the backoff gate.
- Each is scored against the exact prior on the same corrupted text.
- The prior alone, with no residual, on the 10,000 positions the script
  scores by default.

The reasoning, and the revised design (a separate repair path gated by
evidence of corruption), is in `notes/noise-repair.md`.

## How to rerun

Not in git as it ran; no configs or run scripts were kept. The script is in
imm's first commit, `15efc86` on `main`. Proposed tag
`rnd/edit-repair/fuzzy-node`, on `15efc86`. The run files in `runs/` were
`runs/edit_ablation_tiny_canonical_d8_*.json`; each records its own
arguments. The flags that vary are `--mode`, `--corrupt-prob` and
`--corrupt-ops`, with `--exact-theta-json` the depth-8 theta and the
canonical split's three text files.

## Results

### 2026-06-15

| corruption, mode | exact | with repair |
|---|---:|---:|
| substitutions p=0.05, fuzzy-node | 3.978 | 3.971 |
| substitutions p=0.20, fuzzy-node | 4.464 | 4.377 |
| substitutions, trained at p=0.5, scored at p=0.05, fuzzy-node | 3.978 | 3.999 |
| all edit ops p=0.05, correction | 3.993 | 3.995 |
| all edit ops p=0.20, correction | 4.550 | 4.518 |

The full edit-kernel runs scored 5.61 on clean input and 5.75 on corrupted
input, against 3.83 exact.

### 2026-11-07: correction, the full kernel never trained

`EditKernelPrior.prob_and_grad` built each depth's gradient but never kept
it, so it always returned zeros. In `runs/edit_kernel_tiny_canonical_d8_smoke.json`
the loss is flat across epochs; in `runs/edit_ablation_tiny_canonical_d8_p05_smoke.json`
theta is still at its initial values. So the 5.61 and 5.75 describe an
untrained gate. The fix is in `15efc86`, with a finite-difference test and
a "training moves theta" test. The `fuzzy-node` and `correction` modes
compute their gradients directly and did train, so the table above stands.
The `full` runs have not been rerun.

## What we learned, and what's open

- Edit neighbours help only where the context is corrupted, and then by
  about 2% at p=0.20 (4.464 to 4.377). A repair trained on heavy noise hurts
  on light noise.
- **The notes disagree on clean text.** `notes/noise-repair.md` says broad
  neighbour smoothing raised clean-text perplexity. `research-lines.md`
  (later) says that claim is well argued (low mass is not noise) but that no
  valid clean-input measurement supports it, since the clean figure is the
  untrained full kernel's.
- Parked since 2026-06-16, when the work turned to the exact SQLite count
  tree (`research-lines.md`). The typo study of 2026-11-10 (`typo-noise`)
  came back to the same weakness from the model side.
