# 2026-06-14 AGPT-Ultra To IMM Handoff

## Why This Became IMM

The original line of work started as AGPT/trie training. The current best result
does not depend on AGPT-style model reconciliation or Fisher-node optimization.
It depends on a different idea:

```text
Use explicit corpus statistics as an a priori distribution, then train a neural
residual to correct that prior where history or representation can help.
```

That is broad enough to be its own project. A prior could come from:

- a learned count/backoff trie
- Kneser-Ney or another n-gram smoother
- a suffix/prefix statistic table
- a retrieval index
- a grammar or symbolic rule system
- domain-specific constraints

The neural model should learn what the prior cannot cheaply provide.

## Current Math

At each token position:

```text
prior_logits_t = log P_prior(. | context_t)
features_t     = phi_prior(context_t)
residual_t     = f_theta(local_state_t, memory_t, features_t)

final_logits_t = s * prior_logits_t + residual_t
```

Where:

- `s = prior_strength`
- `f_theta` is currently a GRU plus gated cross-attention memory model
- the residual is additionally gated by prior features
- an L2 penalty is applied to the effective residual logits

The working conservative/full transfer recipe is:

```text
prior_strength: 0.75
residual_scale: 0.25
residual_L2:    0.05
memory_record:  written
rope_positions: char
max_memory:     16
hidden/embed:   64/64
attention:      4 heads, 1 layer
```

## Learned Count Prior

The count prior is a learned gated backoff model. For context node `c`:

```text
q(root) = p_root
q(c)    = w(c) * p_mle(. | c) + (1 - w(c)) * q(suffix(c))
w(c)    = sigmoid(theta . features(c))
```

The strongest current feature set uses:

```text
reliability
kl_gain
entropy_norm
depth_norm
entropy_delta
suffix_mass_norm
suffix_reliability
suffix_entropy_norm
suffix_kl_gain
suffix_entropy_delta
bias
```

The Gutenberg theta file used by the latest run:

```text
runs/gutenberg_count_gate_d8_entropy_suffix.json
```

The full prior cache:

```text
runs/count_prior_gutenberg5m_d8_entropy_suffix_fp16_mmap_numpy.pt
```

## Key Results

### 200k Diagnostic

With full prior authority and tiny residual:

```text
epoch 0: 5.61342
epoch 5: 5.60930
attention entropy: ~2.75-2.76
log(16 memory records): 2.77
```

This showed the memory branch was close to uniform and barely learning.

With moderated prior authority:

```text
prior_strength: 0.75
residual_scale: 0.25
residual_L2:    0.05

epoch 0: 5.52120
epoch 1: 5.41850
epoch 2: 5.38517
epoch 3: 5.30000
epoch 4: 5.28429
epoch 5: 5.27333
```

Attention became selective:

```text
epoch 5 entropy:    1.529
epoch 5 top weight: 0.583
```

### Full Gutenberg

Full Gutenberg 5M split:

```text
train chars: 4,750,000
heldout chars: 250,000
train segments: 440,787
updates: 13,775
```

Result:

```text
full prior-only PPL:        3.57155
softened prior epoch 0:     3.72280
after 1 residual epoch:     3.56268
```

Attention:

```text
mean_attn_entropy:       0.912
mean_attn_top_weight:    0.778
mean_attn_char_distance: 42.8
mean_residual_alpha:     0.358
```

Runtime:

```text
train_sec: 3597.75
tok/sec:   1320.3
eval_sec:  65.77
peak RSS:  ~8.6 GB
```

## Interpretation

The full prior was strong enough to starve the residual when used at full
authority. Softening it gave the neural memory branch useful gradient.

The key result is not that `0.75` is optimal. The key result is:

```text
prior authority is a learnable/control problem.
```

If the prior is too strong, the neural branch never learns retrieval. If it is
too weak, the residual destabilizes. There is a productive middle region.

## What To Build Next

Immediate:

1. Make the latest full Gutenberg run reproducible from this project.
2. Add config files for the best known commands.
3. Add bucketed diagnostics:
   - prior entropy
   - reliability
   - depth
   - surprise
   - residual gain/loss
4. Continue the full Gutenberg run for more epochs if compute allows.

Architecture:

1. Test hidden/embed `128/128`.
2. Test `max_memory=32`.
3. Test two gated cross-attention layers.
4. Add learned/scheduled `prior_strength`.
5. Add direct retrieval/utility objectives only after bucket diagnostics exist.

Engineering:

1. Reduce Python overhead in the residual trainer.
2. Move command-line experiment recipes into versioned config files.
3. Keep prior caches out of git unless explicitly packaged.
4. Do not carry old AGPT run artifacts into IMM.

## Files Ported From AGPT-Ultra

Core:

```text
agpt_ultra/count_gate.py              -> imm/count_gate.py
agpt_ultra/vectorized_count_prior.py  -> imm/vectorized_count_prior.py
agpt_ultra/data.py                    -> imm/data.py
agpt_ultra/eval.py                    -> imm/eval.py
agpt_ultra/run_ids.py                 -> imm/run_ids.py
scripts/run_segment_memory_model.py   -> scripts/run_segment_memory_model.py
tests/test_count_gate_packed.py       -> tests/test_count_gate_packed.py
```

Historical notes:

```text
prior-residual-project-reference.md
```

That existing file is useful background from the parallel AGPT project, but the
latest `agpt-ultra` implementation and results supersede its reported numbers.
