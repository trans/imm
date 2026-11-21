---
title:    The training-prior leak, and prior strength 1.0 against 0.75
kind:     experiment
status:   concluded
outcome:  positive
question: >-
  The residual trained on rows whose count prior had counted the very
  character being predicted. Does removing that leak change held-out
  perplexity, and does it change which prior strength is best?
answer: >-
  The leak is real but small: cross-fitting the training prior moves
  held-out perplexity by 0.5% on Tiny canonical and 0.33% on Gutenberg 5M
  at strength 1.0. The bigger finding is that strength 1.0 beats 0.75,
  leaky or not, by about 6% on Tiny (3.7822 against 4.0144) and 2.9% on
  Gutenberg (3.4596 against 3.5619). The June baseline's tiny gain over
  the prior came from softening the prior, not from the leak.
opened:   2026-11-08
updated:  2026-11-08
code: >-
  main: the fix is 5a49012; Tiny runs at eeecae3, Gutenberg runs at
  fb9bfab (both on main). The June 2026 baseline ran in agpt-ultra, not in
  git; its rerun (the leaky 0.75 cell) is part of the Gutenberg runs.
data: >-
  Tiny Shakespeare canonical split 4fa9aec1db6b3aea (depth-8 prior alone
  3.8595); Gutenberg 5M split 4bfd5a43d446644a (depth-8 prior alone
  3.5716). imm's own depth-8 gated count prior on both.
slots:    none (before the slot system)
tags:     [prior, leak, prior-strength, baseline, bug]
related:  [depth-evidence-input, transformer-baseline-gutenberg5m]
---

# The training-prior leak, and prior strength 1.0 against 0.75

## Question

imm predicts with a count prior plus a neural residual. The count prior was
counted from the first 95% of the training text, and the residual then
trained on all of it. So for 95% of its training positions the prior's
counts included the character being predicted. @air found this on
2026-11-08. The residual trained against a prior far more reliable than the
one it meets on held-out text. Did that explain the June result, a residual
that barely beat the prior?

## Design

- **The June baseline** (agpt-ultra, 2026-06-14), Gutenberg 5M: prior
  strength 0.75, residual logit scale 0.25, residual L2 0.05, one epoch.
- **The fix** (`5a49012`): `--count-prior-train-folds 20
  --count-prior-fold-chunk <held-out chunk size>`. Each training row's prior
  comes from counts that leave out its own passage (20 folds). Held-out rows
  are unchanged: their prior never saw held-out text.
- **The test:** a 2 × 2 on each corpus, leaky or cross-fitted training
  prior, at prior strength 1.0 or 0.75. One epoch, seed 1337 (the trainer's
  default).
  - Tiny canonical: the tail-split residual config
    (`tiny_shakespeare_full_baseline_e1.json`) moved to the canonical split
    and its theta. Folds of 5,576 characters.
  - Gutenberg 5M: the original config (scale 0.25, L2 0.05). Folds of
    25,000 characters. Leaky 0.75 is the original config, rerun.
- `scripts/prior_row_ppl.py` scores the prior alone on three regions of
  rows, to show the leak and its removal.

## How to rerun

The script and configs are in git, so nothing was copied here.

- Tiny: commit `eeecae3` on `main`, proposed tag
  `rnd/training-prior-leak/tiny`. Run `scripts/rerun_prior_leak.sh` (at
  that commit it runs both corpora in turn; stop it after Tiny).
- Gutenberg: commit `fb9bfab` on `main`, proposed tag
  `rnd/training-prior-leak/gutenberg`. Run
  `scripts/rerun_prior_leak.sh gutenberg`.
- The configs are `configs/tiny_canonical_{leaky,crossfit20}_ps{075,100}_e1.json`
  and `configs/gutenberg5m_{leaky_ps100,crossfit20_ps075,crossfit20_ps100}_e1.json`,
  plus the original `configs/gutenberg5m_priorstrength075_residual025_l2_005_e1.json`.
- The Gutenberg runs need `runs/gutenberg_count_gate_d8_entropy_suffix.json`
  (the gate's theta). The prior cache is rebuilt if missing.
- The June baseline itself: `python scripts/run_segment_memory_model.py
  --config configs/gutenberg5m_priorstrength075_residual025_l2_005_e1.json`.

## Results

### 2026-06-14: the June baseline (agpt-ultra)

Gutenberg 5M, held-out perplexity per character:

| system | PPL |
|---|---:|
| Witten-Bell, depth 8 | 4.604 |
| learned count gate, depth 8, prior alone | 3.5716 |
| backoff oracle (picks the depth using the target; an upper bound) | 2.9465 |
| residual, epoch 0 (prior softened to 0.75) | 3.7228 |
| residual, epoch 1 | 3.5627 |

The residual's gain over the prior was 0.009 PPL. A 200k-character training
slice (same held-out text) went from 5.521 to 5.273 over 5 epochs.
Reproduced in imm on 2026-11-08: 3.5646 at epoch 1, against 3.5627.

### 2026-11-08: Tiny canonical (`eeecae3`)

The prior's own PPL by row region:

| training prior | training rows the held-out prior counts | training rows held back from it | held-out |
|---|---:|---:|---:|
| leaky | 2.390 | 4.803 | 3.8595 |
| cross-fitted | 3.937 | 4.109 | 3.8595 |

Held-out PPL:

| | strength 1.0 | strength 0.75 |
|---|---:|---:|
| epoch 0, either prior | 3.8595 | 4.1507 |
| leaky, epoch 1 | 3.8000 | 4.0343 |
| cross-fitted, epoch 1 | **3.7822** | 4.0144 |

- The leak is gone: cross-fitted training rows score 3.94, against 3.86
  held-out; leaky ones scored 2.39.
- The fix helps by 0.5% at both strengths. At strength 1.0 the residual's
  gain over the prior grows from 0.060 to 0.077 PPL. One seed.
- Strength 1.0 beats 0.75 by about 6%, leaky or not. Softening the prior
  costs 7.5% before training, and one epoch doesn't win it back.
- Attention is more selective at 1.0: mean top weight 0.51–0.54, against
  0.43–0.47 at 0.75.
- Building the cross-fitted prior took 95 s, against 11 s for one model.

### 2026-11-08: Gutenberg 5M (`fb9bfab`)

All four runs took 1 h 35 min at nice 19. Peak RSS was 2.6 GB while the
cross-fitted cache was built, 1.6 GB otherwise.

The prior's own PPL by row region:

| training prior | training rows the held-out prior counts | training rows held back from it | held-out |
|---|---:|---:|---:|
| leaky | 2.465 | 3.602 | 3.5716 |
| cross-fitted | 3.583 | 3.515 | 3.5716 |

Held-out PPL:

| | strength 1.0 | strength 0.75 |
|---|---:|---:|
| epoch 0, either prior | 3.5716 | 3.7228 |
| leaky, epoch 1 | 3.4710 | 3.5646 (original: 3.5627) |
| cross-fitted, epoch 1 | **3.4596** | 3.5619 |

- Fixing the leak changes little: 0.33% at 1.0, 0.08% at 0.75.
- Strength 1.0 beats 0.75 by 2.9%, leaky or not. The residual's gain over
  the prior is 0.112 PPL at 1.0 (3.5716 to 3.4596), against 0.009 at 0.75.
- Attention went diffuse only at 1.0 with the leaky prior: top weight
  0.238, entropy 2.29. The other three runs had top weight 0.65–0.78 and
  entropy 0.91–1.25 (cross-fitted 1.0: 0.660 and 1.19). Tiny showed no such
  effect (top weight 0.43–0.54 in all four).
- One seed. The original and repro runs, the same config, had top weights
  0.778 and 0.646.

## What we learned

- Cross-fit the training prior. Every later run does
  (`--count-prior-train-folds 20`).
- Keep the prior at full strength (1.0). The June conclusion that "the
  prior can starve the neural model" doesn't hold here. What starved it on
  Gutenberg was the leak, which kept attention nearly uniform at
  strength 1.0, and the softened prior, which cost more than the residual
  won back.
- **The notes disagree here.** `research-lines.md` still lists "full prior
  authority with a tiny residual: attention stays almost uniform and the
  residual barely learns" under "Don't retry", from the agpt-ultra handoff
  doc. This study, which is later, says that the diffuse attention came
  from the leak and that full strength is better once the prior is
  cross-fitted. The later result should win; that line in research-lines
  predates it.
