---
title:    Skip-horizon evidence from the count tree
kind:     experiment
status:   inconclusive
outcome:  mixed
question: >-
  Does empirical evidence from contexts ending h characters back (h = 1 … W)
  add to the count prior, and in what form should the model take it?
answer: >-
  A little, on noisy small evals. On the 50k Tiny slice at lr 1e-4 the
  baseline scored 10.083 at epoch 2; profile features at W=8 10.058,
  one aggregate skip distribution 10.075 (10.079 with a learned gate), and
  separate learned horizon channels 10.056 at gate bias -4. More trust
  hurt: 10.094 at bias -3, 10.294 at -2. The gaps are within what a
  2,000-character eval can show.
opened:   2026-06-17
updated:  2026-06-18
code: >-
  not in git as it ran (agpt-ultra, June 2026). The code was ported into
  imm's first commit, 15efc86 on main: imm/skip_horizon_features.py and the
  --skip-horizon-* flags, with the Crystal builders in bin/.
data: >-
  Tiny Shakespeare slices: the first 50,000 (or 20,000) training
  characters (--max-train-chars), 2,000 held-out characters, no named
  split. imm's own depth-8 count prior fit on the slice.
slots:    none (before the slot system)
tags:     [prior, skip-horizon, empirical-experts, tiny]
related:  [transition-categories, edit-repair]
---

# Skip-horizon evidence from the count tree

## Question

The count prior answers "given the context ending at t−1, what is the next
character?". A skip horizon asks the same of the context ending at t−h, for
h = 1 … W. Each horizon is an empirical distribution over the character at
t, indexed by relative position. Does this evidence help, and how should the
model combine it with the count prior?

## Design

Four forms, each against the same baseline (imm's residual on the count
prior, no skip evidence). All on Tiny slices, 2 epochs, seed 1337.

1. **Profile features:** scalar summaries of the horizons (availability,
   best horizon, sharpness, entropy, max probability, log mass) appended to
   the prior's gate features.
2. **Aggregate distribution:** one skip distribution, the horizons pooled,
   blended into the prior with a fixed weight (0.25) or a learned scalar
   gate.
3. **Learned horizon channels:** the horizons kept separate
   (`skip_log_probs [T, W, V]`), mixed with the count prior by a softmax over
   [count, h1 … hW] from the prior's features, with a strength mask.
4. **Gate-bias sweep** on (3): the initial bias on the horizon channels,
   −5 to −2.

The theory behind it, and the next design it points to (a calibrated
mixture of empirical experts), is in `notes/skip-horizon-prior.md`.

## How to rerun

Not in git as it ran; no configs were kept for these runs
(`lessons.md`, "Skip-horizon runs have no configs"). The code is in imm's
first commit, `15efc86` on `main`. Proposed tag:
`rnd/skip-horizons/learned-w8`, on `15efc86` (the ported code, not the
exact agpt-ultra code that ran).

The best run, as recorded in `measurements.md`:

```sh
python scripts/run_segment_memory_model.py \
  --config configs/tiny_shakespeare_train_smoke_e2.json \
  --run-id tiny_shakespeare_50k_lr1e4_skip_learned_horizon_w8_e2 \
  --max-train-chars 50000 --lr 0.0001 \
  --count-prior-cache runs/tiny_shakespeare_50k_count_prior.pt \
  --skip-horizon-prior learned-horizon-distribution --skip-horizon-max 8 \
  --skip-horizon-prior-gate-bias -4 \
  --skip-horizon-cache runs/tiny_shakespeare_50k_skip_distribution_w8_cache \
  --output runs/tiny_shakespeare_50k_lr1e4_skip_learned_horizon_w8_e2.csv
```

The baseline is the same command without the `--skip-horizon-*` flags. The
horizon cache is built by the Crystal programs in `bin/`, so Crystal is
needed. The other runs' commands can be rebuilt from the arguments stored in
each checkpoint in `runs/`.

## Results

### 2026-06-17: 20k slice

| run | initial PPL | epoch 2 PPL |
|---|---:|---:|
| baseline | 12.137 | 11.763 |
| profile W=8 | 12.137 | 11.710 |
| aggregate distribution W=8, weight 0.25 | 11.988 | 11.604 |

The aggregate distribution lowers the PPL before any training, so the skip
distribution itself moved the frozen prior the right way.

### 2026-06-17: 50k slice, lr 1e-4

At lr 3e-4 the baseline, W=1 and W=8 runs all diverged at epoch 2 (held-out
PPL near 1e14), so these use 1e-4.

| run | epoch 2 PPL |
|---|---:|
| baseline | 10.083 |
| profile W=1 | 10.057 |
| profile W=8 | 10.058 |
| aggregate distribution W=8, fixed weight 0.25 | 10.075 |
| aggregate distribution W=8, learned scalar gate | 10.079 |
| learned horizon channels W=8 | 10.056 |

### 2026-06-17: gate-bias sweep (learned horizon channels, W=8)

| gate bias | initial PPL | epoch 1 | epoch 2 | mean skip weight, epoch 2 |
|---:|---:|---:|---:|---:|
| −5 | 10.197 | 10.139 | 10.060 | 0.0082 |
| −4 | 10.196 | 10.137 | 10.056 | 0.0223 |
| −3 | 10.226 | 10.176 | 10.094 | 0.0600 |
| −2 | 10.360 | 10.358 | 10.294 | 0.1514 |

## What we learned, and what's open

- Profile W=1 and W=8 gave the same gain, so the scalar summaries carry
  reliability, not the horizons' votes.
- Pooling the horizons into one distribution loses which horizon voted. It
  was weaker than separate channels at 50k. `research-lines.md` lists this
  under "Don't retry".
- Too much trust in the horizons adds noise (bias −3 and above). The best
  point gave them about 2% of the weight.
- **Small evals are noisy.** These evaluate 2,000 characters with one seed.
  Gaps of 0.02–0.05 PPL are hints, not results (`lessons.md`). The best gain
  here is 0.027 PPL (0.27%).
- `research-lines.md` lists the empirical-expert mixture as "Open; this is
  the next thread", as of 2026-06-18. It was not taken up after imm moved to
  larger corpora, so this entry marks it inconclusive. The proposed next
  design, per-source features and a calibrator over experts, was never
  built.
