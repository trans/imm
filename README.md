# IMM

IMM is a prior-guided language-memory modeling project.

The current research question:

```text
How should explicit a priori sequence knowledge be provided to a neural model
without letting the prior dominate or starving learned retrieval?
```

This project split out of `agpt-ultra`. The strongest current system is no
longer AGPT tree training. It is a learned count/backoff prior plus a neural
residual memory model:

```text
final_logits = prior_strength * count_prior_log_probs + residual_logits
```

Where:

- `count_prior_log_probs` comes from a learned gated count/backoff prior over
  prefix/suffix statistics.
- `prior_strength` controls how much authority the prior gets in the final
  logit decision.
- `residual_logits` come from a local GRU plus gated cross-attention over
  segment memory records.
- The residual is conditioned on compact prior features such as reliability and
  depth.

## Current Best Result

Full Gutenberg 5M split, one residual epoch, seed 0
(`configs/gutenberg5m_crossfit20_ps100_e1.json`):

```text
full prior-only PPL:        3.5716
after 1 residual epoch:     3.4596
```

The residual adds 0.112 PPL of heldout signal beyond a strong full-corpus
count prior. It needs the training prior cross-fitted (below) and the prior at
full strength.

The key diagnostic:

```text
mean_attn_entropy:       1.188
mean_attn_top_weight:    0.660
mean_attn_char_distance: 39.5
mean_residual_alpha:     0.328
```

## Important Lesson

The training prior must not have seen the text it is predicting.

The count prior is counted from the training text, and the residual trains on
that same text. So at most training positions the prior had already counted
the character it was asked to predict: on Gutenberg its training-row PPL was
2.47, against 3.57 heldout. The residual learned against a prior far more
reliable than the one it meets at test time.

The fix is cross-fitting (`--count-prior-train-folds 20 --count-prior-fold-chunk
<heldout chunk size>`). The training text is cut into chunks the size of a
heldout chunk and dealt into 20 folds. Each training row's prior is counted
from the other 19 folds, so it covers 95% of the text, as the heldout prior
does. Heldout rows are unchanged.

Heldout PPL after one epoch, seed 0 (`scripts/rerun_prior_leak.sh`), same
residual settings within each corpus:

```text
                     Gutenberg 5M           Tiny Shakespeare (canonical)
prior strength       1.0       0.75         1.0       0.75
prior only           3.5716    3.7228       3.8595    4.1507
leaky prior          3.4710    3.5646       3.8000    4.0343
cross-fitted prior   3.4596    3.5619       3.7822    4.0144
```

What this changed:

- An earlier lesson, carried over from agpt-ultra, was that "the prior can
  starve the neural model": at full strength attention stayed nearly uniform
  and the residual barely learned, so the prior was softened to 0.75. Those
  runs used the leaky prior.
- With the leak, attention did go diffuse at strength 1.0 on Gutenberg (top
  weight 0.238, entropy 2.29). Cross-fitted, it stays selective (0.660,
  1.19).
- Softening was the wrong fix. At the same residual settings, strength 1.0
  beats 0.75 by 2.9% on Gutenberg and 6% on Tiny, leaky or not. The leak
  itself moves heldout PPL only 0.1-0.5%.

The open question is still how much authority the prior should get, but at
these settings full strength wins. Learned or scheduled prior authority
(Next Milestones) remains untested.

## Ported Reference Code

The initial IMM scaffold carries over the working reference implementation:

```text
imm/count_gate.py
imm/vectorized_count_prior.py
imm/data.py
imm/eval.py
imm/run_ids.py
scripts/run_segment_memory_model.py
tests/test_count_gate_packed.py
```

This is intentionally a starting point, not a final architecture.

## Quick Validation

Fast local validation uses Tiny Shakespeare and a small mmap count-prior cache:

```sh
python -m unittest discover -s tests
python scripts/run_segment_memory_model.py --config configs/tiny_shakespeare_smoke.json
python scripts/run_segment_memory_model.py --config configs/tiny_shakespeare_train_smoke_e2.json
python scripts/run_segment_memory_model.py --config configs/tiny_shakespeare_transition_category_e2.json
```

Longer reproduction commands are captured as JSON configs:

```text
configs/tiny_shakespeare_smoke.json
configs/tiny_shakespeare_train_smoke_e2.json
configs/tiny_shakespeare_transition_category_e2.json
configs/tiny_shakespeare_full_baseline_e1.json
configs/tiny_shakespeare_full_transition_category_e1.json
configs/gutenberg5m_priorstrength075_residual025_l2_005_e1.json
configs/gutenberg5m_priorstrength075_residual025_l2_005_e2_resume.json
configs/gutenberg5m_crossfit20_ps100_e1.json
```

`scripts/rerun_prior_leak.sh [tiny|gutenberg]` runs the leak comparison above:
leaky or cross-fitted training prior, at strength 1.0 or 0.75
(`configs/tiny_canonical_*`, `configs/gutenberg5m_{leaky,crossfit20}_*`, and
the original `gutenberg5m_priorstrength075_*` config as leaky 0.75).

## Transition Categories

The first IMM-specific experiment appends fixed SVD features from count-prior
successor distributions. These are distributional category features: contexts
with similar continuation sets receive similar low-rank vectors. On the 20k/2k
Tiny Shakespeare two-epoch smoke, this improved validation PPL from `11.7631`
to `11.6475`. On the full Tiny Shakespeare 95/5 split for one epoch, the same
feature path underperformed the baseline: `5.1079` vs `4.9798` PPL. The signal
is therefore real enough to study, but not yet robust.

## Next Milestones

1. Diagnose why transition-category features help the 20k/2k smoke but hurt
   the full Tiny Shakespeare split.
2. Expand config coverage for new experiment sweeps.
3. Optimize throughput; the full Gutenberg residual epoch currently takes
   about one hour on CPU.
4. Test capacity scaling:
   - hidden/embedding 128
   - memory 32
   - two gated cross-attention layers
5. Replace fixed `prior_strength` with learned or scheduled prior authority.
6. Add bucketed diagnostics by prior entropy, reliability, depth, category, and
   surprise.

## Vocabulary

Use IMM vocabulary going forward:

```text
prior provider
count prior
prior strength
residual language-memory model
memory records
retrieval branch
prior-conditioned residual
```

Avoid AGPT/Fisher terminology except in historical notes.
