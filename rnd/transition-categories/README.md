---
title:    Transition categories as residual features
kind:     experiment
status:   inconclusive
outcome:  mixed
question: >-
  Contexts that lead to the same set of continuations form a category
  ("the king of" leads to england, egypt, france …). Does giving the
  residual a low-rank code for the context's successor distribution help?
answer: >-
  It helped on a 20k smoke (11.763 to 11.647 at epoch 2) and hurt on the
  full Tiny tail split after one epoch (4.97976 to 5.10791), with less
  selective attention (top weight 0.392 to 0.252). The category run never
  went past epoch 1, so the full-split comparison is one epoch only.
opened:   2026-06-15
updated:  2026-06-15
code: >-
  not in git as it ran (agpt-ultra, June 2026). The code and configs were
  ported into imm's first commit, 15efc86 on main
  (--transition-category-features).
data: >-
  Smoke: Tiny Shakespeare, first 20,000 training characters, 2,000 held
  out. Full: Tiny tail split 2b7ded401e96b610 (the last 5% held out,
  55,769 characters; depth-8 prior alone 5.160). imm's depth-8 count prior,
  leaky (before the 2026-11-08 fix), at strength 0.75.
slots:    none (before the slot system)
tags:     [prior-features, categories, tiny]
related:  [skip-horizons, training-prior-leak]
---

# Transition categories as residual features

## Question

The user's idea: a shared prefix that selects a set of continuations
defines a category. The category is the shape of the successor set, not the
prefix's text. The first probe asks whether the residual gains from knowing
which successor-set category the current context belongs to.

## Design

- Sample training rows of the count prior's successor distributions, centre
  them, and fit an 8-dimensional SVD basis.
- Project every training and held-out row onto it and append the 8
  coordinates to the prior's feature rows (columns 11–18 after the 11
  `entropy_delta,suffix_stats` features).
- Feed those columns to the residual-state MLP
  (`prior_residual_state_features` 0,3,11–18).
- Baseline: the same config without the category columns. Same slice and
  seed. Prior strength 0.75, residual scale 0.25, L2 0.05, hidden 32, 8
  records.

The design note is `docs/2026-06-15-transition-category-experiment.md`.

## How to rerun

Not in git as it ran. The configs are in imm's first commit, `15efc86` on
`main`; proposed tag `rnd/transition-categories/tail-e1`, on `15efc86` (the
ported code).

```sh
# smoke, 20k slice, 2 epochs
python scripts/run_segment_memory_model.py --config configs/tiny_shakespeare_transition_category_e2.json
python scripts/run_segment_memory_model.py --config configs/tiny_shakespeare_train_smoke_e2.json
# full tail split, 1 epoch
python scripts/run_segment_memory_model.py --config configs/tiny_shakespeare_full_transition_category_e1.json
python scripts/run_segment_memory_model.py --config configs/tiny_shakespeare_full_baseline_e1.json
```

These ran before the training-prior leak was fixed (`5a49012`). Today's
trainer cross-fits only when asked (`--count-prior-train-folds`), so the
configs still run as they did.

## Results

### 2026-06-15: 20k smoke, 2,000 held-out characters

| run | epoch 0 PPL | epoch 1 | epoch 2 |
|---|---:|---:|---:|
| baseline prior features | 12.13746 | 12.04468 | 11.76312 |
| transition-category features | 12.13746 | 11.98388 | 11.64748 |

### 2026-06-15: full tail split (1,059,625 training, 55,769 held out)

| run | epoch 0 PPL | epoch 1 | attention top weight | residual alpha |
|---|---:|---:|---:|---:|
| baseline prior features | 5.35733 | 4.97976 | 0.39214 | 0.35534 |
| transition-category features | 5.35733 | 5.10791 | 0.25226 | 0.37036 |

The baseline was later resumed to epoch 3 (4.888 at epoch 2, 4.855 at
epoch 3). The category run was not.

## What we learned, and what's open

- This SVD injection is not enough. On the full split it made attention
  less selective, which suggests the coordinates gave the residual a broad
  correction path without better retrieval.
- The comparison is one epoch, one seed. The user had asked (2026-06-15) to
  judge runs by their trajectory over several epochs, and this run wasn't
  extended, so `research-lines.md` lists it as inconclusive.
- The user also ruled (2026-06-15) that category structure belongs outside
  the prior's gate, as a separate signal for the residual. This probe
  respects that.
- Not tried: reverse-prefix categories as token-side embeddings, or
  categories for choosing memory records.
