# Skip-Horizon Priors and Empirical Expert Mixtures

## 1. Motivation

The original count-prior path asks a local question:

```text
given a context ending at t-1, what is P(x_t)?
```

The skip-horizon idea asks a broader family of questions:

```text
given a context ending at t-h, what is P(x_t)?
```

for horizons:

```text
h = 1..W
```

So ordinary next-token prediction is `h=1`, and longer-range empirical evidence
is `h>1`.

For a tree depth `D` and horizon `W`, each source is:

```text
context ending at t-h, up to depth D -> empirical distribution over x_t
```

This is not attention in the transformer sense. It is a bank of empirical
conditional distributions indexed by relative position.

## 2. First Profile Experiment

The first implementation compressed all horizon evidence into scalar profile
features:

```text
skip_available_frac
skip_best_h_norm
skip_best_sharpness
skip_best_entropy_norm
skip_best_max_p
skip_best_log_mass_norm
skip_mean_sharpness
skip_late_best_sharpness
skip_local_sharpness
```

These were appended to `prior_gate_features`, so the model could use them in the
residual gate and residual-state MLP.

This helped slightly, but W=1 and W=8 gave nearly identical improvements.

50k Tiny Shakespeare, `lr=1e-4`:

| run | epoch 2 PPL |
|-----|------------:|
| baseline | 10.083 |
| W=1 profile | 10.057 |
| W=8 profile | 10.058 |

Interpretation: the scalar profile features contain useful reliability
information, but they throw away the actual token vote. They tell the model
that a horizon source is sharp or high-mass, but not what token it predicts.

## 3. Aggregate Distribution Experiment

The next implementation compiled a real skip token distribution:

```text
P_skip(x_t) = aggregate over h=1..W of P_h(x_t)
```

and blended it with the count prior:

```text
P_prior = (1 - alpha_t) P_count + alpha_t P_skip
```

The manual blend used:

```text
alpha_t = skip_horizon_prior_weight * skip_strength_t
```

On 20k Tiny Shakespeare, this was the first result that clearly showed actual
token information was entering the model:

| run | initial PPL | epoch 2 PPL |
|-----|------------:|------------:|
| baseline | 12.137 | 11.763 |
| profile W=8 | 12.137 | 11.710 |
| aggregate distribution W=8, weight 0.25 | 11.988 | 11.604 |

The initial PPL improvement matters because it happens before training can learn
anything. The skip distribution itself changed the frozen prior in a useful
direction.

On 50k Tiny Shakespeare with stable `lr=1e-4`, however, aggregate distribution
was weaker:

| run | epoch 2 PPL |
|-----|------------:|
| baseline | 10.083 |
| profile W=8 | 10.058 |
| aggregate distribution W=8, manual weight 0.25 | 10.075 |
| aggregate distribution W=8, learned scalar gate | 10.079 |

Interpretation: aggregating over horizons is still too lossy. It preserves the
token vote, but erases which horizon produced that vote.

## 4. Horizon-Channel Experiment

The current best implementation keeps horizons separate:

```text
skip_log_probs: [T, W, V]
skip_strength:  [T, W]
```

The model learns a channel mixture over:

```text
[count, h1, h2, ..., hW]
```

For each row:

```text
weights_t = softmax(g_theta(features_t) + strength_mask_t)
P_final   = weights_count P_count + sum_h weights_h P_h
```

The strength mask pushes missing or weak horizon sources down without removing
the model's ability to learn channel weights.

50k Tiny Shakespeare, `lr=1e-4`, W=8:

| run | epoch 2 PPL |
|-----|------------:|
| baseline | 10.083 |
| profile W=8 | 10.058 |
| aggregate distribution, manual | 10.075 |
| aggregate distribution, learned scalar gate | 10.079 |
| learned horizon distribution | 10.056 |

The margin is small, but the ordering is meaningful:

```text
aggregate distribution < profile ~= learned horizon
```

The horizon-channel model is at least competitive with profile features while
using actual token distributions. This supports the claim that horizon evidence
is real but not yet optimally harnessed.

## 5. Gate-Bias Sweep

The learned horizon gate was initialized conservatively. A small sweep showed a
clear trust/noise tradeoff.

50k Tiny Shakespeare, `lr=1e-4`, W=8:

| gate bias | initial PPL | epoch 1 PPL | epoch 2 PPL | mean skip alpha, epoch 2 |
|----------:|------------:|------------:|------------:|-------------------------:|
| -5 | 10.197 | 10.139 | 10.060 | 0.0082 |
| -4 | 10.196 | 10.137 | 10.056 | 0.0223 |
| -3 | 10.226 | 10.176 | 10.094 | 0.0600 |
| -2 | 10.360 | 10.358 | 10.294 | 0.1514 |

This is useful evidence. Too little horizon influence underuses the signal. Too
much horizon influence clearly injects noise. The best point so far is a small
but nonzero horizon contribution.

That pattern is what we would expect if the skip-horizon channels contain real
conditional signal, but only under specific contexts.

## 6. Theoretical Framing

The cleaner model is an empirical expert mixture.

Each evidence source `s` provides:

```text
P_s(x_t | evidence_s(history))
```

Sources include:

```text
count prior / local context
backoff contexts
skip horizon h=1..W
possibly reverse/suffix evidence
neural residual distribution
```

The model should learn:

```text
w_s(t) = calibrator_theta(source_stats_s(t), model_state_t)
```

and combine experts:

```text
P(x_t | history) = sum_s w_s(t) P_s(x_t)
```

The core problem is not producing empirical conditional distributions. The tree
can produce them. The core problem is calibration:

```text
when should a sparse, sharp empirical source be trusted?
when should it be ignored?
which horizon is relevant?
which backoff depth is relevant?
```

## 7. Probability Mixture vs. Product of Experts

The current implementation uses probability-space mixture:

```text
P = sum_s w_s P_s
```

This is safe: a bad expert can be assigned low weight, and a missing expert can
be masked out.

An alternative is a product-of-experts or log-linear form:

```text
log P(x) = const + sum_s beta_s log P_s(x)
```

This can sharpen agreement between sources, but it is more dangerous because a
bad sparse expert can suppress the correct token. Given the gate-bias sweep,
the current evidence argues for probability mixtures first.

A hybrid may be useful later:

```text
mixture over expert families, product inside calibrated compatible families
```

but the immediate path should stay with mixture.

## 8. What Is Missing

The current horizon-channel gate is still under-featured. It only sees the
existing count-prior feature row. It does not receive explicit per-horizon
features such as:

```text
horizon index h / W
horizon strength
horizon mass
horizon entropy / sharpness
horizon target_count
agreement with count prior
agreement with other horizons
KL(P_h || P_count)
max probability of P_h
```

Without those, the gate has limited information for deciding why `h=4` should
be trusted in one row but ignored in another.

## 9. Next Architecture

The next serious IMM architecture should treat the tree as a bank of empirical
conditional experts.

A useful first target:

```text
experts:
  count prior
  horizon h=1..W

per-expert distribution:
  P_s over vocab

per-expert features:
  mass
  entropy
  sharpness
  source index
  source depth/horizon
  agreement statistics

calibrator:
  small MLP over per-source features
  optional conditioning on neural state

combination:
  softmax weights over experts
  probability-space mixture
```

This would replace the current scalar profile graft and the current minimally
informed horizon gate with a principled empirical-expert mixture.

## 10. Working Claim

The experiments support this claim:

```text
Long-range empirical horizon distributions contain predictive signal,
but the signal is sparse and conditional. It must be calibrated per source.
```

The observed degradation at higher gate bias is not a failure of the horizon
idea. It is evidence that unconditional horizon trust is wrong.

The model needs to learn:

```text
which empirical expert to trust, when, and how much
```

That is likely the central IMM modeling problem.
