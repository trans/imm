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

Full Gutenberg 5M split:

```text
full prior-only PPL:        3.57155
softened prior epoch 0:     3.72280
after 1 residual epoch:     3.56268
```

This is modest, but important: the residual/LMA branch added heldout signal
beyond a strong full-corpus count prior.

The key diagnostic:

```text
mean_attn_entropy:       0.912
mean_attn_top_weight:    0.778
mean_attn_char_distance: 42.8
mean_residual_alpha:     0.358
```

Earlier runs with full prior authority had nearly uniform attention. Moderate
prior softening made retrieval selective.

## Important Lesson

The prior can starve the neural model.

Observed pattern:

```text
prior_strength = 1.0, tiny residual:
  attention nearly uniform
  residual barely learns

prior_strength = 0.5, large residual:
  attention wakes up
  run destabilizes

prior_strength = 0.75, moderate residual:
  attention becomes selective
  residual improves heldout PPL
```

The core problem is therefore not simply "add a better prior." It is pricing
prior authority so the neural model receives enough gradient to learn what the
prior cannot express.

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
```

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
2. Reproduce the full Gutenberg result from this project directory.
3. Expand config coverage for new experiment sweeps.
4. Optimize throughput; the full Gutenberg residual epoch currently takes
   about one hour on CPU.
5. Test capacity scaling:
   - hidden/embedding 128
   - memory 32
   - two gated cross-attention layers
6. Replace fixed `prior_strength` with learned or scheduled prior authority.
7. Add bucketed diagnostics by prior entropy, reliability, depth, category, and
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
