---
title:    Where imm loses to the transformer, and a learned prior strength
kind:     experiment
status:   concluded
outcome:  positive
question: >-
  imm at 820M characters was level with the 1.86M transformer given the
  prior. Where is the remaining gap, and does letting the model scale the
  prior down per position, as the transformer can, close it?
answer: >-
  The gap is +0.07% made of two large opposite parts: imm is 1.5% ahead
  where the prior is roughly right and 1.5% behind where it is confidently
  wrong (true character under 5%; 12.8% of positions). imm's output fixed
  the prior's strength at 1. With --prior-strength-gate (g = softplus(...)
  per position) imm scored 3.0616 at 400M (−1.73%; the confidently wrong
  positions carry 105% of the gain) and 3.0563 at 820M, 1.37% ahead of the
  transformer (3.0987). With no prior at all it scored 3.3916; the 4.9%
  lead that gave over the no-prior transformer was later found to rest on
  that transformer's dropout.
opened:   2026-11-16
updated:  2026-11-19
code: >-
  --prior-strength-gate at 91939ef (branch prior-strength-gate, deleted;
  the commit is on main). The bottleneck analysis used slot 52's model
  (55b8128) and @air's transformer checkpoint.
data: >-
  PG-19 20M split, data/pg19/imm-split-20m (250,000 held-out characters).
  Mobius's pg19split20m_gated_d16_pg200m prior (alone PPL 3.5320, 1.8205
  bits/char) with compact pg19split20m_kn evidence. Run C uses no prior,
  no rows and no evidence.
slots:    [56, 57, 58]
tags:     [prior, prior-strength, gate, bottleneck, transformer-comparison, pg19]
related:  [word-tier, prior-dropout, imm-1.85m, memory-knockouts]
---

# Where imm loses to the transformer, and a learned prior strength

## Question

After slot 52, imm (3.1011) was 0.08% behind the 1.86M transformer with the
prior in its input (3.0987) at the same training. The user asked (via @air,
2026-11-16) where imm's bottleneck is and what it needs to improve.

## Design

- **Diagnosis (CPU, no slot):** imm's slot-52 model against @air's
  transformer, paired per held-out character. The transformer was rescored
  on the CPU (`score_transformer_chars.py`) exactly as its own scoring does
  (windows of 256, stride 128; 3.0989 again), into imm's per-character
  format; then `scripts/rnd/compare_tx.py` split the difference by the
  prior's loss on the true character, place in the word, entropy and depth.
- **Why, from the code:** imm's output was `1.0 × prior_log_probs + α ×
  own_logits`, with an L2 penalty on the residual. It could only outweigh a
  wrong prior with a large, penalised residual. The transformer's output is
  `own + g × prior` with g = softplus(a linear map of [h, features]),
  learned per position and unpenalised.
- **Runs** (slot 50's model: two blocks, word tier 256, own step; 20
  epochs, seed 1337):
  - **A:** the leash off (`--prior-residual-l2 0`);
  - **B:** `--prior-strength-gate`: g = softplus(w · [the head's input
    state, features] + b), w = 0 and g = 1 at the start, leash as before;
  - **B at 820M:** 41 epochs;
  - **C:** no prior at all (`--count-prior none`, no rows, no evidence, no
    prior input), 20 epochs.

## How to rerun

Scripts here, as they ran; paths are the original machine's. Comments that
quoted the user were paraphrased; every command is as it ran.

| run | script | commit | slot | proposed tag |
|---|---|---|---|---|
| diagnosis | `score_transformer_chars.py`, then `scripts/rnd/compare_tx.py` | `55b8128` | none | `rnd/learned-prior-strength/bottleneck` |
| A and B, 20 epochs | `gate_slot.sh 56` | `91939ef` | 56 | `rnd/learned-prior-strength/slot56` |
| B, 41 epochs | `pg41_slot.sh 57` | `91939ef` | 57 | `rnd/learned-prior-strength/slot57` |
| C, no prior | `noprior_slot.sh 58` | `91939ef` | 58 | `rnd/learned-prior-strength/slot58` |

Paired comparisons here, from `compare_tx.py` (the version now in
`scripts/rnd/`; "imm" is A and "tx" is B in each):

- `results/compare_tx_own41.txt`: slot 52 against the transformer;
- `results/gate_vs_own_pg19s20m_h256_par_l2_w256_own_pg_e20.txt`: B against
  slot 50;
- `results/gate_vs_own_pg19s20m_h256_par_l2_w256_own_l2off_e20.txt`: A
  against slot 50;
- `results/gate_vs_tx.txt`, `results/pg41_vs_tx.txt`: B at 400M and 820M
  against the transformer;
- `results/pg41_vs_own41.txt`: B at 820M against slot 52;
- `results/noprior_vs_gate.txt`, `results/noprior_vs_own.txt`: C against B
  and slot 50.

## Results

### 2026-11-16: where the gap is (CPU)

| the prior's loss on the true character | share | prior alone | imm | transformer | imm − transformer, nats/char |
|---|---:|---:|---:|---:|---:|
| < 0.5 nats | 45.7% | 0.135 | 0.108 | 0.110 | −0.0008 |
| 0.5–3 | 41.5% | 1.564 | 1.411 | 1.445 | −0.0138 |
| 3–6 | 11.4% | 3.932 | 3.575 | 3.498 | **+0.0088** |
| ≥ 6 | 1.4% | 7.436 | 6.419 | 5.940 | **+0.0066** |

- Where the prior is roughly right (under 3 nats, 87%), imm is ahead by
  0.0147 nats a character (1.5%). Where it is confidently wrong (12.8%),
  imm is behind by 0.0154 (1.5%).
- This one effect explains every other gap: each place in the word shows
  imm ahead where the prior is right and behind where it is wrong. imm's
  apparent weakness at word ends, phrase ends and on punctuation is the
  confident-prior problem. The prior's entropy and depth explain little.
- The training text limits too: from epoch 20 to 41, training NLL fell
  1.0837 → 1.0644 while held-out moved 1.1374 → 1.1317.

### 2026-11-16: A and B at 400M (slot 56)

| | own step (slot 50) | A: leash off | **B: learned prior strength** |
|---|---:|---:|---:|
| held-out, best (epoch 20) | 3.1155 | 3.1067 | **3.0616** |
| change | | −0.28% | **−1.73%** |
| epoch 1 / 4 / 8 / 12 / 16 / 20 | 3.305 / 3.201 / 3.155 / 3.134 / 3.121 / 3.116 | 3.303 / 3.203 / 3.148 / 3.124 / 3.110 / 3.107 | 3.259 / 3.158 / 3.097 / 3.078 / 3.065 / 3.062 |
| mean α (residual gate) | 0.256 | 0.399 | 0.231 |
| mean g (prior strength), held-out | 1 | 1 | 0.933 (0.950 at epoch 1) |
| take-away: own, prior in input / none | 57.39 / 106.47 | 42.30 / 136.22 | 42.49 / 60.48 |

B against slot 50, by the prior's loss on the true character:

| the prior's loss | share | B vs own step |
|---|---:|---:|
| < 0.1 nats | 24.4% | −0.03% ±0.03 |
| 0.1–0.5 | 21.4% | −0.13% ±0.08 |
| 0.5–1.5 | 21.0% | +0.25% ±0.13 |
| 1.5–3 | 20.5% | +0.32% ±0.18 |
| **3–6** | 11.4% | **−8.08% ±0.32** |
| **≥ 6** | 1.4% | **−46.88% ±1.94** (perplexity 662 → 352) |

- The confidently wrong 12.8% of positions carry 105% of the gain.
- By what is predicted: uppercase −14.2%, punctuation −8.0%, lowercase
  −1.3%, spaces +0.5%. Before a new sentence −14.0%, before a new phrase
  −10.6%.
- B at 400M beats the transformer at 820M by 1.20%, and is ahead of it in
  every place in the word.
- With a gate on the prior, the network learns more of the prediction
  itself: with no prior at all its own output scores 60.5, against 106.5.
- A helps a little and spreads its gain; without the leash the residual
  leans on the prior's input more (136 with the prior removed).

### 2026-11-16: B at 820M (slot 57)

| at 820M characters | params | held-out |
|---|---:|---:|
| **imm, word tier + own step + learned prior strength** | 3.20M | **3.0563** |
| 1.86M transformer, prior in the input | 1.86M | 3.0987 |
| imm, word tier + own step (slot 52) | 3.20M | 3.1011 |

- 1.37% ahead of the transformer (paired), 1.44% ahead of slot 52.
- imm saturates: doubling the training gained only 0.17%. The train/held-out
  gap grew from 0.054 to 0.069; the 20M split is the limit.
- The mean prior strength is 0.940 on held-out text. 4,603 s.

### 2026-11-16: C, no prior (slot 58)

| at 400M characters | held-out |
|---|---:|
| the prior alone (mobius's d16 rows) | 3.5320 |
| **imm, no prior (C)** | **3.3916** |
| imm with the prior (slot 50) | 3.1155 (−8.1% from C) |
| imm with the learned prior strength (B) | 3.0616 (−9.7% from C) |

@air's transformers at 820M on the same split: 1.86M, no prior 3.5678;
with the prior 3.0987. 10.8M, no prior 3.0386; with the prior 3.0667.

- Without a prior, imm at 400M beat the 1.86M transformer at 820M by 4.9%
  (3.3916 against 3.5678).
- The 10.8M transformer without a prior (3.0386) was 0.6% ahead of imm's
  best with the prior (3.0563).
- C's train/held-out gap is 0.024, against 0.054 with the prior: without
  the prior, imm memorises less.
- C against B by the prior's loss: C is worse below 3 nats and better where
  the prior is confidently wrong (−11.7% at 3–6 nats, −16.9% at 6 or more).
  These bins are chosen by the outcome, so this is not reachable headroom,
  but it says B still pays 12–17% there against a model that never saw the
  prior.
- 2,047 s.

### 2026-11-19: correction to the no-prior comparison

Later work found that @air's transformers at dropout 0.2 were held back on
these corpora: their training loss sat above held-out (slots 101 and 105;
the `no-prior-pair-1.8m` entry). @air's `s1.84m_none` on this split trains
at 1.313 against 1.272 held out. So the 4.9% lead of C over the no-prior
1.86M transformer rests on the same handicap and doesn't stand. The
prior-in-input transformer (`s1.84m_prior_in`, 1.100 against 1.131) is
less clearly held back, but that is untested, so B's 1.37% lead is also
open to a retuned baseline (`lessons.md`, "Tune the baseline's
regularization before claiming a win").

## What we learned

- imm couldn't turn a confidently wrong prior down. A learned prior
  strength fixed it and stayed in imm's model from here on.
- `research-lines.md` still lists "learned or scheduled prior authority" as
  open and not started; this is it, built and positive. The note is stale.
