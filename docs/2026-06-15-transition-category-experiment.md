# 2026-06-15 Transition Category Experiment

## Idea

Shared transition structure induces categories.

```text
"the king of" -> {england, egypt, france, ...}
```

The category is not the literal prefix. It is the shape of the successor set.
The reverse direction is also meaningful: common suffixes or continuations can
categorize the set of prefixes that lead to them.

## First Implementation

The first experiment uses the count-prior successor distributions already
computed by `scripts/run_segment_memory_model.py`.

1. Sample train prior probability rows.
2. Center them by their mean successor distribution.
3. Fit a low-rank SVD basis over token-successor axes.
4. Project every train/eval prior row into that basis.
5. Append the projection to the count-prior feature rows.
6. Feed the category feature columns to the residual-state MLP.

This is enabled with:

```text
transition_category_features = svd
transition_category_dim = 8
```

The category feature columns start after the count-prior gate features. With the
current `entropy_delta,suffix_stats` feature set, the base feature count is 11,
so an eight-dimensional category vector occupies columns 11-18.

## Tiny Shakespeare Smoke Result

Config:

```sh
python scripts/run_segment_memory_model.py --config configs/tiny_shakespeare_transition_category_e2.json
```

Baseline:

```sh
python scripts/run_segment_memory_model.py --config configs/tiny_shakespeare_train_smoke_e2.json
```

Same slice and seed:

```text
train chars: 20000
eval chars:  2000
epochs:      2
```

Result:

| run | epoch 0 PPL | epoch 1 PPL | epoch 2 PPL |
|-----|------------:|------------:|------------:|
| baseline prior features | 12.13746 | 12.04468 | 11.76312 |
| transition-category features | 12.13746 | 11.98388 | 11.64748 |

Interpretation: on this tiny smoke, the category features gave a small positive
signal without changing the count prior, memory budget, or residual recipe.

## Full Tiny Shakespeare Result

Config:

```sh
python scripts/run_segment_memory_model.py --config configs/tiny_shakespeare_full_transition_category_e1.json
```

Baseline:

```sh
python scripts/run_segment_memory_model.py --config configs/tiny_shakespeare_full_baseline_e1.json
```

Same full 95/5 tail split:

```text
train chars: 1,059,625
eval chars:     55,769
epochs:              1
```

Result:

| run | epoch 0 PPL | epoch 1 PPL | attn top | residual alpha |
|-----|------------:|------------:|---------:|---------------:|
| baseline prior features | 5.35733 | 4.97976 | 0.39214 | 0.35534 |
| transition-category features | 5.35733 | 5.10791 | 0.25226 | 0.37036 |

Interpretation: the same first-probe category features hurt on the full Tiny
split after one epoch. Attention was less selective (`top_weight` fell from
`0.392` to `0.252`) even though residual alpha increased slightly. That suggests
the appended category coordinates may be giving the residual branch a broad
correction path without improving retrieval geometry.

## Caveats

This is only a first probe. It does not yet expose reverse-prefix categories as
explicit token-side embeddings, and it does not yet use category structure for
memory-record selection. It tests whether the residual benefits from knowing
which successor-set category the current context belongs to. The full Tiny
result says this specific SVD feature injection is not enough.
