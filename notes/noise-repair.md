# Edit-Neighborhoods, Noise Repair, and the Limits of Trie Smoothing

## 1. Background

AGPT’s prefix-trie formulation gives a clean structural advantage: identical corpus prefixes are represented once, and all suffix-gradient contributions beneath that prefix can be aggregated at the shared node. This makes the trie an exact empirical structure over observed prefixes.

In that setting, each trie node represents a real corpus context:

```text
prefix/context → node → empirical successor distribution
```

The model can use node-level information such as:

```text
mass
depth
branching factor
successor entropy
rank/share among siblings
matched depth
node reliability
position/phase features
```

This has already proven useful: when the model is given exact trie positions and geometric node features, it can learn to exploit the empirical structure of the corpus.

However, this leaves open a different question:

```text
What happens when the input context is noisy, corrupted, misspelled,
off-manifold, or simply absent from the trie?
```

That motivated the edit-neighborhood idea.

---

## 2. The Edit-Neighborhood Proposal

The initial proposal was to replace brittle exact lookup:

```python
stats = exact_node(ctx[-d:])
if missing:
    skip_or_backoff()
```

with a soft neighborhood lookup:

```python
candidates = exact_node + edit_neighbor_nodes_at_depth_d

posterior(u | x) ∝ exp(-lambda * edit_cost(x, u)) * count(u)^alpha

p_d(next) = Σ_u posterior(u | x) * p(next | u)
```

Then the existing depth gate could still interpolate as usual:

```python
q_d = w_d * p_d + (1 - w_d) * q_{d-1}
```

The intuition was that edit distance provides an approximate metric over trie nodes:

```text
nearby string contexts → nearby empirical paths
```

So if the exact depth-d node is missing or weak, the model can borrow from nearby high-support nodes.

This reframes edit distance as a lateral geometry over the trie:

```text
vertical relation:    parent/child prefix structure
lateral relation:     nearby contexts at similar depth
```

---

## 3. What the Experiment Showed

The experiment gave a very useful result.

Training low-mass paths to gain contributions from close higher-mass neighbors did help in specific cases where noise was added to the input.

But it also raised average perplexity on normal clean evaluation.

That means the edit-neighborhood mechanism is not simply a better general-purpose smoothing prior.

Instead, it behaves like a conditional repair mechanism.

The observed pattern is:

```text
clean input:
  exact path is already the right statistic
  neighbor smoothing blurs specificity
  average PPL rises

noisy input:
  exact path may be missing, weak, or wrong
  nearest real path is useful
  local PPL improves
```

So edit-neighbor smoothing helps when the observed input is suspected to be corrupted or off-manifold. It hurts when applied broadly to clean corpus contexts.

---

## 4. Why General Smoothing Hurts

The mistake is to equate low mass with noise.

Low mass does not mean wrong.
Low mass only means low empirical support.

A low-mass path may represent:

```text
a rare name
a specific quote
a poetic phrase
a one-off construction
a domain-specific term
a genuinely informative rare context
```

If such a path is forced to borrow from a nearby high-mass path, the model loses specificity.

For example, a rare context may be rare precisely because it carries unusual predictive information. Smoothing it toward a common neighbor converts a sharp empirical signal into a generic one.

So the training-time smoothing rule:

```text
rare path → borrow from close high-mass paths
```

is too blunt.

The better rule is:

```text
corrupted or off-manifold path → infer nearest real path
```

Those are different operations.

---

## 5. Revised Interpretation

The edit-distance mechanism should not be treated as part of the base AGPT gradient tree.

It is better understood as an inference-time or conditionally activated repair module.

Base AGPT learns:

```text
p(next | clean observed context)
```

The repair module estimates:

```text
p(next | nearest plausible real context)
```

Then a repair gate decides whether the repaired distribution should influence the final prediction.

A useful decomposition is:

```python
p_base = p(next | exact_or_backoff_path)

p_repair = Σ_u posterior(u | noisy_context) * p(next | u)

p_final = (1 - r(x)) * p_base + r(x) * p_repair
```

where:

```text
r(x) ≈ probability that the observed context is corrupted, noisy, or off-manifold
```

This `r(x)` is not the same as the ordinary depth gate.

The depth gate asks:

```text
Is this depth-d evidence reliable?
```

The repair gate asks:

```text
Should I trust the observed context at all,
or should I snap it toward a nearby real path?
```

---

## 6. Clean Model vs. Repair Model

The architecture should distinguish two roles.

### 6.1 Base model

The base model handles clean language modeling:

```text
exact trie path
vertical backoff path
node geometry
mass/entropy/reliability features
ordinary sequence context
```

Its job is to model the actual empirical distribution of the corpus.

It should not be forced to blur rare paths into common neighbors.

### 6.2 Repair model

The repair model handles noisy or corrupted contexts:

```text
misspellings
character substitutions
tokenization glitches
OCR-like errors
low-confidence input
off-manifold prefixes
```

Its job is to infer:

```text
What real observed path did this noisy input probably intend?
```

This is a nearest-real-path problem, not a normal language-modeling problem.

---

## 7. The Repair Gate

A repair gate should determine when the edit-neighborhood distribution is allowed to affect prediction.

Useful gate features include:

```text
exact_mass
matched_depth
backoff_distance
exact_successor_entropy
nearest_neighbor_mass
nearest_neighbor_edit_cost
neighbor_candidate_count
neighbor_posterior_entropy
top_neighbor_share
KL(p_base || p_repair)
JS(p_base, p_repair)
input_noise_score
```

The divergence features are especially important.

If `p_base` and `p_repair` agree, using repair is relatively harmless.

If they disagree strongly, the model should only trust repair when there is strong evidence that the input is corrupted.

This avoids the average-PPL penalty observed when smoothing is applied unconditionally.

---

## 8. Training the Repair Module

The repair mechanism should probably be trained with synthetic corruption, not ordinary clean corpus likelihood alone.

A clean context-target pair:

```text
x → y
```

can be corrupted into:

```text
x_tilde → y
```

Then the model sees `x_tilde`, but the target remains the continuation from the original clean context.

Examples of corruption:

```text
character substitution
character deletion
character insertion
transposition
token dropout
token replacement
spacing/punctuation noise
```

The repair module learns to recover predictive information from nearby real paths.

A useful objective:

```python
L = CE(y, p_final)
  + beta * r(clean_x)
  + gamma * repair_gate_supervision
```

Where repair usefulness can be labeled by comparing losses:

```python
repair_is_useful =
    CE(y, p_repair) < CE(y, p_base) - margin
```

So the repair gate learns:

```text
activate repair when repair improves prediction
stay quiet on clean input
```

This turns edit-distance lookup into a conditional denoising operation.

---

## 9. Why This Should Stay Separate from the AGPT Gradient Tree

AGPT’s prefix-trie factorization is clean because it preserves exact empirical structure:

```text
same prefix → same node → aggregated suffix gradient
```

Edit-neighbor smoothing changes the statistical object being estimated.

It says:

```text
nearby but non-identical prefixes should partially share predictive mass
```

That may be useful under noise, but it is not generally true for clean data.

Therefore, the safest design is:

```text
exact trie = base empirical geometry
edit-neighborhood = optional repair geometry
```

The base training graph should remain exact.

The lateral edit-neighbor graph should initially be read-only and conditionally invoked.

Only after the repair mechanism proves useful should one consider pushing gradients through lateral edges.

---

## 10. Practical Architecture

A practical implementation could use three components:

```text
1. Base predictor
2. Repair predictor
3. Repair gate
```

### 10.1 Base predictor

```python
logits_base = BaseModel(
    token_context,
    exact_node_features,
    backoff_features,
    geometric_features
)
```

### 10.2 Repair predictor

```python
neighbors = retrieve_edit_neighbors(context)

p_repair = mixture(
    p_next_given_neighbor,
    weights = softmax(
        -lambda * edit_cost
        + alpha * log_mass
        + reliability_score
    )
)
```

### 10.3 Repair gate

```python
r = sigmoid(RepairGate(
    exact_mass,
    matched_depth,
    neighbor_entropy,
    nearest_edit_cost,
    KL_base_repair,
    input_noise_score
))
```

### 10.4 Final prediction

Either probability-space interpolation:

```python
p_final = (1 - r) * p_base + r * p_repair
```

or logit-space residual correction:

```python
logits_final = logits_base + r * delta_logits_repair
```

The logit-residual version may be safer if the base model should remain dominant.

---

## 11. Main Lesson

The experiment clarified the role of edit distance.

It is not a universal smoothing principle for the trie.

It is a conditional inference mechanism for noisy contexts.

The key distinction is:

```text
low mass ≠ noise
rare path ≠ corrupted path
nearby high-mass path ≠ better path
```

Edit-neighbor recovery should activate only when there is evidence that the observed input is not a trustworthy clean path.

So the model should separate:

```text
clean corpus geometry
```

from:

```text
nearest-real-path repair
```

The base AGPT representation should continue to model the exact empirical structure of observed prefixes.

The edit-neighborhood mechanism should sit beside it as a denoising conditioner, repair gate, or inference-time correction path.

---

## 12. Summary

The revised design is:

```text
Do not train all low-mass paths to borrow from high-mass edit neighbors.

Instead:

  1. Keep the exact trie as the base empirical geometry.
  2. Use node features and backoff features for clean prediction.
  3. Build edit-neighbor retrieval as a separate repair mechanism.
  4. Train the repair mechanism on corrupted inputs.
  5. Gate repair based on evidence of corruption or off-manifold input.
  6. Penalize repair activation on clean inputs.
```

The central conclusion:

```text
Edit distance is useful for nearest-real-path inference,
not as a blanket training-time smoothing prior.
```

This preserves the strength of the trie while adding robustness where the exact path is unreliable.

