---
title:    The residual's leash, and showing it the prior's per-depth evidence
kind:     experiment
status:   concluded
outcome:  positive
question: >-
  With the leak fixed, is the residual held back by its leash (logit scale
  and L2), its width, or what it is shown of the prior? Does showing it
  each context length's counts help?
answer: >-
  The leash and the width are not the limit: scale 1.0 gains 0.24% on
  Tiny, 4x the width nothing. Per-depth evidence is: on Tiny,
  --prior-input both scores 3.7341 / 3.7350 / 3.7353 over three seeds
  against 3.7734 / 3.7768 / 3.7838 with no input (1.04–1.28% better). On
  Gutenberg at hidden 256 the best epochs go from 3.2712 (no input) to
  3.1970 (imm's depth 8) and 3.1767 (mobius's depth 16), 2.9% better.
opened:   2026-11-08
updated:  2026-11-09
code: >-
  main: leash grid ecbeb12; log-probs input 853c2b8; depth evidence and
  seeds 6778d54 and dc89419; Gutenberg runs 1a39851 (branch
  external-evidence, deleted; the commit is on main).
data: >-
  Tiny Shakespeare canonical 4fa9aec1db6b3aea (prior alone 3.8595) and
  Gutenberg 5M 4bfd5a43d446644a (prior alone 3.5716). imm's depth-8 gated
  prior, cross-fitted, strength 1.0; on Gutenberg also mobius's KN depth
  evidence rows (gutenberg5m_kn, depths 1–16).
slots:    none (before the slot system)
tags:     [prior-input, depth-evidence, leash, seeds, gutenberg, tiny]
related:  [training-prior-leak, residual-width, word-segments, mobius-prior-rows]
---

# The residual's leash, and showing it the prior's per-depth evidence

## Question

After the leak fix (`training-prior-leak`), the residual gained 0.077 PPL
on Tiny and 0.112 on Gutenberg over the prior alone. What limits it? Three
candidates: its leash (the logit scale α and the L2 penalty that keep its
logits small), its width, or what it is shown of the prior.

## Design

All with a cross-fitted prior (20 folds) at strength 1.0, seed 1337 unless
stated.

1. **Leash grid** (Tiny, 1 epoch): residual scale 0.25, 0.5, 1.0 × L2 0.05,
   0.01 (`scripts/tune_residual_leash.sh tiny`). Gutenberg at scale 1.0,
   L2 0.05, 3 epochs, constant lr.
2. **What the residual sees** (Tiny, scale 1.0, L2 0.05, 1 epoch):
   - `--prior-input log-probs`: the prior's final log-probabilities;
   - `--prior-input depth-evidence`: each context length's counts (the
     backoff's parts);
   - `--prior-input both`;
   - `both` without the gate's summary features;
   - no input at hidden and embedding 128 instead of 32.
   Seeds 1 and 2 for none and `both`.
3. **Gutenberg at hidden 256** (lr 1e-3, 64 streams, 12 epochs,
   `--prior-input both`), with three sources of evidence: imm's own depth
   8, a copy of mobius's first 8 depths, and mobius's 16 depths
   (`--prior-evidence-rows`). "None" is epochs 1–12 of the no-input run in
   `residual-width`.

## How to rerun

- Leash grid: commit `ecbeb12` on `main`, proposed tag
  `rnd/depth-evidence-input/leash`. `scripts/tune_residual_leash.sh tiny`;
  Gutenberg: `scripts/tune_residual_leash.sh gutenberg 1.0 0.05 3`. (The
  script's comment says seed 0; the trainer's default, which it uses, is
  1337.)
- The Tiny input runs and seeds: no script was kept. They are
  `configs/tiny_canonical_crossfit20_ps100_e1.json` with
  `--prior-residual-scale 1.0` and `--prior-input
  {log-probs,depth-evidence,both}`, plus `--seed 1` or `2`; the no-summary
  arm adds `--prior-residual-gate none --prior-residual-state-features ""`;
  the wide arm `--hidden-size 128 --embedding-size 128`. log-probs at
  `853c2b8` (proposed tag `rnd/depth-evidence-input/log-probs`), the rest at
  `6778d54` and `dc89419` (proposed tag `rnd/depth-evidence-input/tiny`, on
  `dc89419`, which lets the seeds share one prior cache).
- Gutenberg hidden 256: `seeds_then_depth.sh` here, second half, run from a
  worktree at `1a39851` (proposed tag
  `rnd/depth-evidence-input/gutenberg-h256`). The first half is the word
  segments' seed runs (`word-segments`). The script's mobius path is the
  original machine's.

## Results

### 2026-11-08: the leash, Tiny canonical, 1 epoch

Prior alone 3.8595.

| residual scale | L2 0.05 | L2 0.01 | L2 0.05, `--prior-input log-probs` |
|---|---:|---:|---:|
| 0.25 | 3.7822 | 3.8203 | 3.7714 |
| 0.5 | 3.7772 | 3.7779 | |
| 1.0 | 3.7734 | **3.7731** | **3.7706** |

Gutenberg, scale 1.0, L2 0.05: 3.4654, 3.4434, **3.4359** at epochs 1–3.
At scale 0.25, epoch 1: 3.4596. Prior alone 3.5716.

- Scale 1.0 gains 0.24% on Tiny and nothing on Gutenberg at epoch 1. The
  residual's weight α rises to about 1.0, but it has little more to say.
- More epochs help, with shrinking returns: −0.6% from epoch 1 to 2, −0.2%
  from 2 to 3.
- The 0.25 / 0.01 cell is 1% off its neighbours, three times the seed
  spread below. Unexplained.

### 2026-11-09: what the residual sees, Tiny canonical, 1 epoch

| residual input | seed 1337 | seed 1 | seed 2 |
|---|---:|---:|---:|
| none | 3.7734 | 3.7768 | 3.7838 |
| none, hidden and embedding 128 (from 32) | 3.7733 | | |
| `--prior-input log-probs` | 3.7706 | | |
| `--prior-input depth-evidence` | 3.7497 | | |
| `--prior-input both` | **3.7341** | **3.7350** | **3.7353** |
| `both`, no summaries | 3.7398 | | |

- Seed spread with no input is 0.3%; with `both`, 0.03%.
- `both` beats no input by 1.04–1.28% on the same seeds. The residual's
  gain over the prior grows from 0.076–0.086 to 0.124–0.125 PPL.
- The final prediction alone adds 0.07%; on top of the evidence it adds
  0.4%. To fix a blend, the residual needs both the result and its parts.
- Removing the summaries costs 0.15%, but that run also fixed the trust
  gate at 1.0, so the cost may be the gate's. Not separated.
- The evidence for Tiny took 27 s to build, 1.1 GB on disk (float16), and
  made training about 10% slower.

### 2026-11-09: Gutenberg, hidden 256, 12 epochs (`1a39851`)

| epoch | none | imm d8 | mobius 1–8 | mobius 16 |
|---:|---:|---:|---:|---:|
| 1 | 3.4441 | 3.4267 | 3.4224 | **3.3687** |
| 3 | 3.3804 | 3.3355 | 3.3262 | **3.2550** |
| 5 | 3.3258 | 3.2483 | 3.2482 | **3.1999** |
| 7 | 3.2856 | 3.2054 | 3.2156 | **3.1767** |
| 9 | 3.2732 | 3.2140 | 3.1870 | 3.1881 |
| 11 | 3.2779 | 3.1970 | 3.1954 | 3.1878 |
| 12 | 3.2856 | 3.2016 | 3.2085 | 3.2043 |
| best (epoch) | 3.2712 (10) | 3.1970 (11) | 3.1870 (9) | **3.1767 (7)** |
| time to best | 9.8 min | 15.8 min | 10.2 min | 11.7 min |
| peak RSS | 2.8 GB | 6.8 GB | 6.6 GB | 8.3 GB |

- Evidence helps the wide residual more than the narrow one: 2.3% (imm d8)
  to 2.9% (mobius 16) better than no input, against 1.0–1.3% at hidden 32
  on Tiny. 3.1767 was then imm's best on Gutenberg, 2.8% better than the
  474k transformer (3.2694; `transformer-baseline-gutenberg5m`).
- imm's and mobius's depths 1–8 are the same evidence: the bests differ by
  0.3%, about one seed's spread.
- Depths 9–16 help mostly early: mobius 16 leads mobius 1–8 by 1.6% at
  epoch 1, 2.1% at 3, 1.2% at 7, and 0.3% at the bests. It overfits sooner
  (best at epoch 7; training NLL 1.139 against 1.157 there).
- The top record's attention weight is 0.33 with depth 16, against 0.59 for
  both depth-8 runs and 0.70 for none.
- Depth 16 makes epochs about 1.5 min instead of about 1 min.

## What we learned

- Showing the residual the prior's per-depth evidence is the first input
  that mattered. Every later imm run uses `--prior-input both`.
- Deeper evidence (mobius's rows) buys mainly speed on 5M characters, since
  the model overfits there; on a larger corpus the early lead is the better
  guide. This pointed the work toward mobius's priors and larger corpora
  (`mobius-prior-rows`, `pg19-4.75m-split`).
