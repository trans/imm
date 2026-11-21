---
title:    The transformer baselines on PG-19, with and without the prior
kind:     experiment
status:   concluded
outcome:  reference
question: >-
  What does a character transformer score on mobius's PG-19 splits, with
  the books prior on its logits or also in its input, and at imm's
  training budget?
answer: >-
  On the 20M split at 819M training characters: 1.84M plain 3.5689, prior
  on the logits 3.1904, prior also in the input 3.0987 (the bar imm was
  judged against); 10.8M plain 3.0386, where the prior helps only with
  prior dropout (2.9326). On the 4.75M split the 1.84M with the prior in
  its input scored 3.2887. At imm's 400M budget the 477k scored 3.1565 and
  the 1.84M 3.1578. On the d24 rows, 2.8131. The runs at dropout 0.2 were
  later found held back (no-prior form certainly, prior-in-input form
  untested).
opened:   2026-11-11
updated:  2026-11-19
code: >-
  Mostly @air's code, not imm's: air's experiments/prior_scaling
  (prior_transformer.py, probe.py; air 7c46036, bef9142, aae6754, 1f6785c,
  fbd853e), which imports imm's scripts/transformer_baseline.py and
  imm/data.py (last changed in imm at 43057d9). imm ran only the two
  400M-budget runs (slots 13 and 34).
data: >-
  PG-19 4.75M split (data/pg19/imm-split) and 20M split
  (data/pg19/imm-split-20m), the same 250,000 held-out characters. Priors:
  mobius's pg19split_gated_d16_pg200m / pg19split20m_gated_d16_pg200m
  (alone 3.532 on 20M) and pg19split20m_gated_d24_pg10g (alone 3.068).
slots:    [13, 34]
tags:     [baseline, transformer, prior, prior-dropout, pg19]
related:  [pg19-4.75m-split, blocks-20m, prior-dropout, learned-prior-strength, imm-1.85m]
---

# The transformer baselines on PG-19, with and without the prior

## Question

imm's comparisons on PG-19 need a transformer trained on the same text,
given the same prior. @air ran these with imm's transformer, adding the
count prior behind a learned gate on the output ("logits") and, optionally,
a zero-initialised linear layer of the prior's log-probs and gate features
into the embedding ("+ input"), as imm's `--prior-input log-probs` does.
This entry collects those numbers, since every imm entry from
`pg19-4.75m-split` on is measured against them.

## Design

- 4 × 192 (1.84M), 4 heads, dropout 0.2, block 256, batch 64, 50,000 steps
  (819M characters), cosine 1e-3 → 1e-4, weight decay 0.1, seed 1337. Other
  sizes: 2 × 64 (121k), 4 × 96 (477k), 6 × 384 (10.8M).
- **Prior dropout** (@air): each training sequence loses the prior from
  input and output with probability 0.5; held-out always has it.
- **Taken away:** a fresh output layer trained on the frozen final hidden
  state (@air's `probe.py`).
- **At imm's budget:** imm's 20M runs train 20 epochs, 400M characters, so
  imm reran @air's prior-in-input configs with `--iters 24414`, the cosine
  ending there.

## How to rerun

@air's runs are in air's repository (`experiments/prior_scaling/runs/`);
they have no imm tag. The two runs imm made:

| run | script | slot | proposed tag |
|---|---|---|---|
| 477k, prior in the input, 400M | `tx_budget.sh 13` (here) | 13 | `rnd/transformer-prior-pg19/slot13` |
| 1.84M, prior in the input, 400M | `../blocks-20m/tx_l3_slot.sh 34` (first part) | 34 | `rnd/transformer-prior-pg19/slot34` |

Both run air's `prior_transformer.py` (air `7c46036`) from air's checkout,
which imports imm's `transformer_baseline.py`; that file was unchanged
since imm `43057d9`, so both tags would go on `43057d9`. One comment in
`tx_budget.sh` quoted the user; it was paraphrased. Their outputs are here:
`results/s477k_prior_in_i24414.json`, `results/s1.84m_prior_in_i24414.json`
(the run's arguments, score curve and take-away scores).

## Results

### 2026-11-11: plain transformers on the PG-19 splits (@air)

| model, trained on | params | best | last | bits/char (best) |
|---|---:|---:|---:|---:|
| 1.84M, 4.75M | 1.84M | 3.7364 (step 47,500) | 3.7404 | 1.9017 |
| 1.84M, 20M | 1.84M | 3.5689 (step 50,000, still improving) | 3.5689 | 1.8355 |

The transformer gains 4.6% from 4x the training text.

### 2026-11-11/12: the prior inside a transformer, 20M split (@air)

| size | no prior | logits | + input | + input, prior dropout 0.5 | logits, prior dropout 0.5 |
|---|---:|---:|---:|---:|---:|
| 121k (2 × 64) | 4.3752 | 3.2483 | 3.2183 | | |
| 477k (4 × 96) | 3.7235 | 3.2184 | 3.1282 | | |
| 1.84M (4 × 192) | 3.5689 | 3.1904 | 3.0987 | 3.1744 | 3.2539 |
| 10.8M (6 × 384) | 3.0386 | 3.0397 | 3.0667 | 2.9326 | 2.9890 |

- The bar for imm on the 20M split is 3.0987, the 1.84M with the prior in
  its input. The input path adds about 0.04 bits/char up to 1.84M.
- At 10.8M the prior doesn't help without prior dropout; the input version
  overfits sooner. With prior dropout it is a gain (2.9326, still falling).
- Taken away, without prior dropout the networks score 5.9–17.6, against
  character frequencies' 23.9: they learn what the prior gets wrong. With
  prior dropout they're at plain level.

### 2026-11-12: the 1.84M with the books prior on the 4.75M split (@air)

1,859,532 parameters, 1,838,784 of them the transformer; one seed.

| model | best (step) | last |
|---|---:|---:|
| plain transformer | 3.7364 | 3.7404 |
| + prior in the input | **3.2887** (30k) | 3.3186 |
| prior on the logits only | 3.3540 (27.5k) | 3.3859 |

Both overfit this split, peaking at 27.5k–30k of 50k steps.

### 2026-11-13/14: at imm's budget, 400M characters (slots 13 and 34)

| model | params | at 819M (@air) | at 400M |
|---|---:|---:|---:|
| 4 × 96, prior in the input, dropout 0 | 487k | 3.1282 | **3.1565** (1.6583 bits/char) |
| 4 × 192, prior in the input, dropout 0.2 | 1.86M | 3.0987 | **3.1578** |

- Halving the training costs the 477k 0.9% and the 1.84M 1.9%. At 400M
  the bigger transformer is no better than the small one.
- The 477k rerun reproduces the original's start: 3.5832 at step 0 in
  both, 3.3338 against 3.3347 at step 2,500. 321 s; the 1.84M 665 s.
- Taken away, the 477k's own logits score 28.6 with the prior in its
  input, 107.8 with none.

### 2026-11-16: on the d24 rows (@air, slot 64)

`d24_s1.84m_prior_in` (air `fbd853e`): 2.8131. Rescored per character on
imm's side, 2.8130 (`imm-1.85m`).

### 2026-11-19: correction, the dropout handicap

Slots 101 and 105 found that @air's transformers at dropout 0.2 are held
back on PG-19: their training loss sits above held-out (the
`no-prior-pair-1.8m` entry). On this split, `s1.84m_none` trains at 1.313
against 1.272 held out, so its 3.5689 is not a tuned baseline. The
prior-in-input form (`s1.84m_prior_in`, 1.100 against 1.131) is less
clearly held back, but untested. The 10.8M `s10.8m_none` trains at 1.072
against 1.111. The 477k at 400M above ran at dropout 0.

## What we learned

- With a prior, a transformer learns what the prior gets wrong unless
  trained with prior dropout. imm's residual behaves the same way
  (`prior-dropout`).
- Any comparison of imm against these baselines carries the dropout caveat
  until the transformer's regularisation is swept on both sides.
