from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from imm.count_gate import CountModel, EPS, select_positions


def _softmax3(values: tuple[float, float, float]) -> tuple[float, float, float]:
    high = max(values)
    exp0 = math.exp(values[0] - high)
    exp1 = math.exp(values[1] - high)
    exp2 = math.exp(values[2] - high)
    total = exp0 + exp1 + exp2
    return exp0 / total, exp1 / total, exp2 / total


def _entropy_norm_from_probs(probs: list[float], vocab_size: int) -> float:
    entropy = 0.0
    for prob in probs:
        if prob > 0.0:
            entropy -= prob * math.log(max(prob, EPS))
    return entropy / math.log(vocab_size)


def _kl_div(left: list[float], right: list[float]) -> float:
    total = 0.0
    for p, q in zip(left, right):
        if p > 0.0:
            total += p * math.log(max(p, EPS) / max(q, EPS))
    return total


@dataclass(frozen=True)
class EditKernelRows:
    probs: tuple[float, ...]
    total_mass: float
    candidate_count: int
    min_distance: float
    mean_distance: float


@dataclass(frozen=True)
class FuzzyNodeRows:
    probs: tuple[float, ...]
    total_mass: float
    candidate_count: int
    top_share: float
    exact_share: float
    posterior_entropy: float
    min_distance: float


class EditKernelPrior:
    """Learned mixture of backoff, exact-context, and edit-neighborhood evidence."""

    def __init__(
        self,
        model: CountModel,
        radius: int = 1,
        edit_lambda: float = 1.0,
        candidate_cap: int = 256,
        allow_transpose: bool = True,
    ) -> None:
        if radius != 1:
            raise ValueError("only edit radius 1 is implemented")
        self.model = model
        self.radius = radius
        self.edit_lambda = edit_lambda
        self.candidate_cap = candidate_cap
        self.allow_transpose = allow_transpose
        self._candidate_cache: dict[bytes, tuple[bytes, ...]] = {}
        self._edit_cache: dict[bytes, EditKernelRows | None] = {}
        self._feature_cache: dict[tuple[bytes, tuple[float, ...]], tuple[float, ...]] = {}

    def feature_names(self) -> list[str]:
        return [
            "exact_mass_norm",
            "exact_reliability",
            "exact_entropy_norm",
            "edit_mass_norm",
            "edit_candidate_count_norm",
            "edit_entropy_norm",
            "edit_min_distance",
            "edit_mean_distance",
            "edit_kl_vs_backoff",
            "edit_kl_vs_exact",
            "depth_norm",
            "bias",
        ]

    def parameter_names(self) -> list[str]:
        names = self.feature_names()
        return [f"{source}:{name}" for source in ("backoff", "exact", "edit") for name in names]

    def initial_theta(self, exact_bias: float = 1.0, backoff_bias: float = 0.0, edit_bias: float = -2.0) -> list[float]:
        names = self.feature_names()
        theta = [0.0] * (3 * len(names))
        bias_idx = names.index("bias")
        theta[bias_idx] = backoff_bias
        theta[len(names) + bias_idx] = exact_bias
        theta[2 * len(names) + bias_idx] = edit_bias
        return theta

    def clear_runtime_caches(self) -> None:
        self._candidate_cache.clear()
        self._edit_cache.clear()
        self._feature_cache.clear()
        self.model.clear_runtime_caches()

    def _candidate_contexts(self, ctx: bytes) -> tuple[bytes, ...]:
        cached = self._candidate_cache.get(ctx)
        if cached is not None:
            return cached
        depth = len(ctx)
        candidates: set[bytes] = set()
        vocab = range(self.model.vocab_size)

        for pos in range(depth):
            prefix = ctx[:pos]
            suffix = ctx[pos + 1 :]
            original = ctx[pos]
            for token in vocab:
                if token != original:
                    candidates.add(prefix + bytes([token]) + suffix)

        if depth > 1:
            for pos in range(depth):
                candidates.add(ctx[:pos] + ctx[pos + 1 :])

        if depth < self.model.depth:
            for pos in range(depth + 1):
                prefix = ctx[:pos]
                suffix = ctx[pos:]
                for token in vocab:
                    candidates.add(prefix + bytes([token]) + suffix)

        if self.allow_transpose and depth > 1:
            for pos in range(depth - 1):
                if ctx[pos] != ctx[pos + 1]:
                    candidates.add(ctx[:pos] + bytes([ctx[pos + 1], ctx[pos]]) + ctx[pos + 2 :])

        candidates.discard(ctx)
        ordered = tuple(sorted(candidates))
        self._candidate_cache[ctx] = ordered
        return ordered

    def edit_rows(self, ctx: bytes) -> EditKernelRows | None:
        if len(ctx) > self.model.depth:
            ctx = ctx[-self.model.depth :]
        cached = self._edit_cache.get(ctx)
        if cached is not None or ctx in self._edit_cache:
            return cached
        rows: list[tuple[float, int, bytes, object]] = []
        for candidate in self._candidate_contexts(ctx):
            if not candidate:
                continue
            stats = self.model.stats(candidate)
            if stats is None or stats.total <= 0:
                continue
            distance = 1.0
            weight = math.exp(-self.edit_lambda * distance) * stats.total
            rows.append((weight, int(stats.total), candidate, stats))

        if self.candidate_cap > 0 and len(rows) > self.candidate_cap:
            rows.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
            rows = rows[: self.candidate_cap]

        total_mass = sum(row[0] for row in rows)
        if total_mass <= 0.0:
            self._edit_cache[ctx] = None
            return None

        probs = [0.0] * self.model.vocab_size
        weighted_distance = 0.0
        min_distance = float("inf")
        for weight, _total, _candidate, stats in rows:
            scale = weight / (total_mass * stats.total)
            for token, count in stats.counts.items():
                probs[token] += scale * count
            weighted_distance += weight
            min_distance = min(min_distance, 1.0)
        norm = sum(probs)
        if norm <= 0.0:
            self._edit_cache[ctx] = None
            return None
        result = EditKernelRows(
            probs=tuple(max(prob / norm, EPS) for prob in probs),
            total_mass=total_mass,
            candidate_count=len(rows),
            min_distance=min_distance,
            mean_distance=weighted_distance / total_mass,
        )
        self._edit_cache[ctx] = result
        return result

    def exact_distribution(self, ctx: bytes, backoff: list[float]) -> list[float]:
        stats = self.model.stats(ctx)
        if stats is None or stats.total <= 0:
            return backoff
        probs = [0.0] * self.model.vocab_size
        scale = 1.0 / stats.total
        for token, count in stats.counts.items():
            probs[token] = scale * count
        return probs

    def features(self, ctx: bytes, backoff_probs: list[float]) -> tuple[float, ...]:
        cache_key = (ctx, tuple(round(prob, 12) for prob in backoff_probs))
        cached = self._feature_cache.get(cache_key)
        if cached is not None:
            return cached
        stats = self.model.stats(ctx)
        exact_mass_norm = math.log1p(stats.total) / math.log1p(len(self.model.tokens)) if stats is not None else 0.0
        exact_reliability = stats.total / (stats.total + stats.types) if stats is not None and stats.total > 0 else 0.0
        exact_entropy = self.model.entropy_norm(ctx) if stats is not None and stats.total > 0 else None
        exact_entropy_norm = exact_entropy if exact_entropy is not None else 0.0
        exact_probs = self.exact_distribution(ctx, backoff_probs)

        edit = self.edit_rows(ctx)
        if edit is None:
            edit_probs = backoff_probs
            edit_mass_norm = 0.0
            edit_candidate_count_norm = 0.0
            edit_entropy_norm = 0.0
            edit_min_distance = 0.0
            edit_mean_distance = 0.0
            edit_kl_vs_backoff = 0.0
            edit_kl_vs_exact = 0.0
        else:
            edit_probs = list(edit.probs)
            edit_mass_norm = math.log1p(edit.total_mass) / math.log1p(len(self.model.tokens))
            edit_candidate_count_norm = math.log1p(edit.candidate_count) / math.log1p(max(self.candidate_cap, 1))
            edit_entropy_norm = _entropy_norm_from_probs(edit_probs, self.model.vocab_size)
            edit_min_distance = edit.min_distance / max(self.radius, 1)
            edit_mean_distance = edit.mean_distance / max(self.radius, 1)
            edit_kl_vs_backoff = _kl_div(edit_probs, backoff_probs)
            edit_kl_vs_exact = _kl_div(edit_probs, exact_probs)

        result = (
            exact_mass_norm,
            exact_reliability,
            exact_entropy_norm,
            edit_mass_norm,
            edit_candidate_count_norm,
            edit_entropy_norm,
            edit_min_distance,
            edit_mean_distance,
            edit_kl_vs_backoff,
            edit_kl_vs_exact,
            len(ctx) / self.model.depth,
            1.0,
        )
        self._feature_cache[cache_key] = result
        return result

    def _mixture_weights(self, theta: list[float], feats: tuple[float, ...]) -> tuple[float, float, float]:
        width = len(feats)
        logits = []
        for source in range(3):
            offset = source * width
            logits.append(sum(theta[offset + j] * feats[j] for j in range(width)))
        return _softmax3((logits[0], logits[1], logits[2]))

    def prob_and_grad(self, ctx: bytes, target: int, theta: list[float]) -> tuple[float, list[float], list[dict[str, float]]]:
        max_depth = min(self.model.depth, len(ctx))
        q = self.model.unigram_probs[target]
        probs = self.model.unigram_probs[:]
        grad = [0.0] * len(theta)
        diagnostics: list[dict[str, float]] = []

        for depth in range(1, max_depth + 1):
            sub = ctx[-depth:]
            backoff_probs = probs
            feats = self.features(sub, backoff_probs)
            width = len(feats)
            exact_probs = self.exact_distribution(sub, backoff_probs)
            edit = self.edit_rows(sub)
            edit_probs = list(edit.probs) if edit is not None else backoff_probs
            sources = (q, exact_probs[target], edit_probs[target])
            weights = self._mixture_weights(theta, feats)
            new_q = weights[0] * sources[0] + weights[1] * sources[1] + weights[2] * sources[2]

            # d(new_q)/d(theta): the backoff source is q, so it carries q's
            # gradient. So does any source that fell back to the backoff
            # distribution (no exact counts, no edit neighbours): its target
            # probability is then q too. The features are treated as
            # constants (edit_kl_vs_backoff also depends on theta through the
            # backoff distribution; that path isn't differentiated).
            carry = weights[0]
            if exact_probs is backoff_probs:
                carry += weights[1]
            if edit is None:
                carry += weights[2]
            new_grad = [carry * value for value in grad]
            for source_idx, source_prob in enumerate(sources):
                scale = weights[source_idx] * (source_prob - new_q)
                offset = source_idx * width
                for j, feat in enumerate(feats):
                    new_grad[offset + j] += scale * feat
            grad = new_grad

            probs = [
                weights[0] * backoff_probs[token] + weights[1] * exact_probs[token] + weights[2] * edit_probs[token]
                for token in range(self.model.vocab_size)
            ]
            z = sum(probs)
            if z > 0.0:
                probs = [max(prob / z, EPS) for prob in probs]
            q = max(new_q, EPS)
            diagnostics.append(
                {
                    "backoff_weight": weights[0],
                    "exact_weight": weights[1],
                    "edit_weight": weights[2],
                    "edit_candidate_count": float(edit.candidate_count if edit is not None else 0),
                }
            )

        return q, grad, diagnostics

    def prob(self, ctx: bytes, target: int, theta: list[float]) -> float:
        q, _grad, _diagnostics = self.prob_and_grad(ctx, target, theta)
        return q


class FuzzyNodeLocator:
    """Approximate same-depth node lookup for missing or weak context paths."""

    def __init__(
        self,
        model: CountModel,
        edit_lambda: float = 1.0,
        mass_alpha: float = 1.0,
        candidate_cap: int = 256,
        allow_transpose: bool = True,
    ) -> None:
        self.model = model
        self.edit_lambda = edit_lambda
        self.mass_alpha = mass_alpha
        self.candidate_cap = candidate_cap
        self.allow_transpose = allow_transpose
        self._candidate_cache: dict[bytes, tuple[tuple[bytes, float], ...]] = {}
        self._rows_cache: dict[bytes, FuzzyNodeRows | None] = {}

    def _candidate_contexts(self, ctx: bytes) -> tuple[tuple[bytes, float], ...]:
        cached = self._candidate_cache.get(ctx)
        if cached is not None:
            return cached
        depth = len(ctx)
        candidates: dict[bytes, float] = {ctx: 0.0}
        vocab = range(self.model.vocab_size)
        for pos in range(depth):
            prefix = ctx[:pos]
            suffix = ctx[pos + 1 :]
            original = ctx[pos]
            for token in vocab:
                if token != original:
                    candidates[prefix + bytes([token]) + suffix] = 1.0
        if self.allow_transpose and depth > 1:
            for pos in range(depth - 1):
                if ctx[pos] != ctx[pos + 1]:
                    candidates[ctx[:pos] + bytes([ctx[pos + 1], ctx[pos]]) + ctx[pos + 2 :]] = 1.0
        ordered = tuple(sorted(candidates.items(), key=lambda item: (item[1], item[0])))
        self._candidate_cache[ctx] = ordered
        return ordered

    def rows(self, ctx: bytes) -> FuzzyNodeRows | None:
        if len(ctx) > self.model.depth:
            ctx = ctx[-self.model.depth :]
        cached = self._rows_cache.get(ctx)
        if cached is not None or ctx in self._rows_cache:
            return cached
        rows: list[tuple[float, float, bytes, object]] = []
        for candidate, distance in self._candidate_contexts(ctx):
            if not candidate:
                continue
            stats = self.model.stats(candidate)
            if stats is None or stats.total <= 0:
                continue
            weight = math.exp(-self.edit_lambda * distance) * (stats.total**self.mass_alpha)
            rows.append((weight, distance, candidate, stats))

        if self.candidate_cap > 0 and len(rows) > self.candidate_cap:
            rows.sort(key=lambda row: (row[0], -row[1], row[2]), reverse=True)
            rows = rows[: self.candidate_cap]
        total_mass = sum(row[0] for row in rows)
        if total_mass <= 0.0:
            self._rows_cache[ctx] = None
            return None

        probs = [0.0] * self.model.vocab_size
        top_weight = 0.0
        exact_weight = 0.0
        entropy = 0.0
        min_distance = float("inf")
        for weight, distance, candidate, stats in rows:
            posterior = weight / total_mass
            top_weight = max(top_weight, posterior)
            if candidate == ctx:
                exact_weight = posterior
            entropy -= posterior * math.log(max(posterior, EPS))
            min_distance = min(min_distance, distance)
            scale = posterior / stats.total
            for token, count in stats.counts.items():
                probs[token] += scale * count
        norm = sum(probs)
        if norm <= 0.0:
            self._rows_cache[ctx] = None
            return None
        result = FuzzyNodeRows(
            probs=tuple(max(prob / norm, EPS) for prob in probs),
            total_mass=total_mass,
            candidate_count=len(rows),
            top_share=top_weight,
            exact_share=exact_weight,
            posterior_entropy=entropy / math.log(max(len(rows), 2)),
            min_distance=min_distance,
        )
        self._rows_cache[ctx] = result
        return result


class FuzzyNodeCorrectionPrior:
    """Learned correction that uses fuzzy node lookup only when it beats base backoff."""

    def __init__(self, model: CountModel, exact_theta: list[float], locator: FuzzyNodeLocator) -> None:
        self.model = model
        self.exact_theta = exact_theta
        self.locator = locator

    def feature_names(self) -> list[str]:
        return [
            "exact_present",
            "exact_mass_norm",
            "exact_reliability",
            "base_entropy_norm",
            "fuzzy_mass_norm",
            "fuzzy_candidate_count_norm",
            "fuzzy_top_share",
            "fuzzy_exact_share",
            "fuzzy_posterior_entropy",
            "fuzzy_successor_entropy",
            "fuzzy_kl_vs_base",
            "fuzzy_min_distance",
            "bias",
        ]

    def parameter_names(self) -> list[str]:
        names = self.feature_names()
        return [f"{source}:{name}" for source in ("base", "fuzzy") for name in names]

    def initial_theta(self, fuzzy_bias: float = -4.0) -> list[float]:
        names = self.feature_names()
        theta = [0.0] * (2 * len(names))
        theta[len(names) + names.index("bias")] = fuzzy_bias
        return theta

    def features(self, ctx: bytes, base_probs: list[float]) -> tuple[float, ...]:
        if len(ctx) > self.model.depth:
            ctx = ctx[-self.model.depth :]
        stats = self.model.stats(ctx)
        fuzzy = self.locator.rows(ctx)
        fuzzy_probs = list(fuzzy.probs) if fuzzy is not None else base_probs
        return (
            1.0 if stats is not None and stats.total > 0 else 0.0,
            math.log1p(stats.total) / math.log1p(len(self.model.tokens)) if stats is not None else 0.0,
            stats.total / (stats.total + stats.types) if stats is not None and stats.total > 0 else 0.0,
            _entropy_norm_from_probs(base_probs, self.model.vocab_size),
            math.log1p(fuzzy.total_mass) / math.log1p(len(self.model.tokens)) if fuzzy is not None else 0.0,
            math.log1p(fuzzy.candidate_count) / math.log1p(max(self.locator.candidate_cap, 1)) if fuzzy is not None else 0.0,
            fuzzy.top_share if fuzzy is not None else 0.0,
            fuzzy.exact_share if fuzzy is not None else 0.0,
            fuzzy.posterior_entropy if fuzzy is not None else 0.0,
            _entropy_norm_from_probs(fuzzy_probs, self.model.vocab_size) if fuzzy is not None else 0.0,
            _kl_div(fuzzy_probs, base_probs) if fuzzy is not None else 0.0,
            fuzzy.min_distance if fuzzy is not None else 0.0,
            1.0,
        )

    def _weights(self, theta: list[float], feats: tuple[float, ...]) -> tuple[float, float]:
        width = len(feats)
        base_logit = sum(theta[j] * feats[j] for j in range(width))
        fuzzy_logit = sum(theta[width + j] * feats[j] for j in range(width))
        high = max(base_logit, fuzzy_logit)
        base_exp = math.exp(base_logit - high)
        fuzzy_exp = math.exp(fuzzy_logit - high)
        total = base_exp + fuzzy_exp
        return base_exp / total, fuzzy_exp / total

    def prob_and_grad(self, ctx: bytes, target: int, theta: list[float]) -> tuple[float, list[float], dict[str, float]]:
        if len(ctx) > self.model.depth:
            ctx = ctx[-self.model.depth :]
        base_probs = self.model.gated_distribution(ctx, self.exact_theta)
        fuzzy = self.locator.rows(ctx)
        fuzzy_probs = list(fuzzy.probs) if fuzzy is not None else base_probs
        feats = self.features(ctx, base_probs)
        base_weight, fuzzy_weight = self._weights(theta, feats)
        base_prob = base_probs[target]
        fuzzy_prob = fuzzy_probs[target]
        q = max(base_weight * base_prob + fuzzy_weight * fuzzy_prob, EPS)
        width = len(feats)
        grad = [0.0] * len(theta)
        grad_base_logit = base_weight * (base_prob - q)
        grad_fuzzy_logit = fuzzy_weight * (fuzzy_prob - q)
        for j, feat in enumerate(feats):
            grad[j] = grad_base_logit * feat
            grad[width + j] = grad_fuzzy_logit * feat
        return q, grad, {
            "base_weight": base_weight,
            "fuzzy_weight": fuzzy_weight,
            "candidate_count": float(fuzzy.candidate_count if fuzzy is not None else 0),
            "exact_present": 1.0 if self.model.stats(ctx) is not None else 0.0,
        }

    def prob(self, ctx: bytes, target: int, theta: list[float]) -> float:
        q, _grad, _diag = self.prob_and_grad(ctx, target, theta)
        return q


def train_fuzzy_node_correction_prior(
    prior: FuzzyNodeCorrectionPrior,
    valid_tokens: bytes,
    depth: int,
    positions: list[int],
    epochs: int,
    lr: float,
    context_fn=None,
) -> tuple[list[float], list[dict[str, float]]]:
    theta = prior.initial_theta()
    m = [0.0] * len(theta)
    v = [0.0] * len(theta)
    beta1 = 0.9
    beta2 = 0.999
    history: list[dict[str, float]] = []
    step = 0
    for epoch in range(1, epochs + 1):
        random.Random(epoch).shuffle(positions)
        grad_sum = [0.0] * len(theta)
        total_loss = 0.0
        base_weight = 0.0
        fuzzy_weight = 0.0
        candidate_count = 0.0
        exact_present = 0.0
        for pos in positions:
            ctx = valid_tokens[max(0, pos - depth) : pos]
            if context_fn is not None:
                ctx = context_fn(ctx, pos, epoch)
            target = valid_tokens[pos]
            q, grad_q, diag = prior.prob_and_grad(ctx, target, theta)
            total_loss -= math.log(max(q, EPS))
            for j, grad in enumerate(grad_q):
                grad_sum[j] += -grad / max(q, EPS)
            base_weight += diag["base_weight"]
            fuzzy_weight += diag["fuzzy_weight"]
            candidate_count += diag["candidate_count"]
            exact_present += diag["exact_present"]
        scale = 1.0 / max(len(positions), 1)
        step += 1
        for j in range(len(theta)):
            grad = grad_sum[j] * scale
            m[j] = beta1 * m[j] + (1.0 - beta1) * grad
            v[j] = beta2 * v[j] + (1.0 - beta2) * grad * grad
            m_hat = m[j] / (1.0 - beta1**step)
            v_hat = v[j] / (1.0 - beta2**step)
            theta[j] -= lr * m_hat / (math.sqrt(v_hat) + 1.0e-8)
        loss = total_loss * scale
        history.append(
            {
                "epoch": epoch,
                "loss": loss,
                "ppl": math.exp(loss),
                "avg_base_weight": base_weight * scale,
                "avg_fuzzy_weight": fuzzy_weight * scale,
                "avg_candidate_count": candidate_count * scale,
                "exact_present_rate": exact_present * scale,
            }
        )
    return theta, history

    def distribution(self, ctx: bytes, theta: list[float]) -> list[float]:
        probs = self.model.unigram_probs[:]
        for depth in range(1, min(self.model.depth, len(ctx)) + 1):
            sub = ctx[-depth:]
            feats = self.features(sub, probs)
            weights = self._mixture_weights(theta, feats)
            exact_probs = self.exact_distribution(sub, probs)
            edit = self.edit_rows(sub)
            edit_probs = list(edit.probs) if edit is not None else probs
            probs = [
                weights[0] * probs[token] + weights[1] * exact_probs[token] + weights[2] * edit_probs[token]
                for token in range(self.model.vocab_size)
            ]
            z = sum(probs)
            if z > 0.0:
                probs = [max(prob / z, EPS) for prob in probs]
        return probs


class EditKernelCorrectionPrior:
    """Baseline-preserving edit kernel: learned blend of exact prior and edit-neighborhood prior."""

    def __init__(self, model: CountModel, exact_theta: list[float], edit_kernel: EditKernelPrior) -> None:
        self.model = model
        self.exact_theta = exact_theta
        self.edit_kernel = edit_kernel

    def feature_names(self) -> list[str]:
        return [
            "edit_mass_norm",
            "edit_candidate_count_norm",
            "edit_entropy_norm",
            "edit_kl_vs_base",
            "base_entropy_norm",
            "depth_norm",
            "bias",
        ]

    def parameter_names(self) -> list[str]:
        names = self.feature_names()
        return [f"{source}:{name}" for source in ("base", "edit") for name in names]

    def initial_theta(self, edit_bias: float = -4.0) -> list[float]:
        names = self.feature_names()
        theta = [0.0] * (2 * len(names))
        theta[len(names) + names.index("bias")] = edit_bias
        return theta

    def features(self, ctx: bytes, base_probs: list[float]) -> tuple[float, ...]:
        edit = self.edit_kernel.edit_rows(ctx)
        if edit is None:
            edit_probs = base_probs
            edit_mass_norm = 0.0
            edit_candidate_count_norm = 0.0
            edit_entropy_norm = 0.0
            edit_kl_vs_base = 0.0
        else:
            edit_probs = list(edit.probs)
            edit_mass_norm = math.log1p(edit.total_mass) / math.log1p(len(self.model.tokens))
            edit_candidate_count_norm = math.log1p(edit.candidate_count) / math.log1p(max(self.edit_kernel.candidate_cap, 1))
            edit_entropy_norm = _entropy_norm_from_probs(edit_probs, self.model.vocab_size)
            edit_kl_vs_base = _kl_div(edit_probs, base_probs)
        return (
            edit_mass_norm,
            edit_candidate_count_norm,
            edit_entropy_norm,
            edit_kl_vs_base,
            _entropy_norm_from_probs(base_probs, self.model.vocab_size),
            min(len(ctx), self.model.depth) / self.model.depth,
            1.0,
        )

    def _weights(self, theta: list[float], feats: tuple[float, ...]) -> tuple[float, float]:
        width = len(feats)
        base_logit = sum(theta[j] * feats[j] for j in range(width))
        edit_logit = sum(theta[width + j] * feats[j] for j in range(width))
        high = max(base_logit, edit_logit)
        base_exp = math.exp(base_logit - high)
        edit_exp = math.exp(edit_logit - high)
        total = base_exp + edit_exp
        return base_exp / total, edit_exp / total

    def prob_and_grad(self, ctx: bytes, target: int, theta: list[float]) -> tuple[float, list[float], dict[str, float]]:
        base_probs = self.model.gated_distribution(ctx, self.exact_theta)
        edit = self.edit_kernel.edit_rows(ctx)
        edit_probs = list(edit.probs) if edit is not None else base_probs
        feats = self.features(ctx, base_probs)
        base_weight, edit_weight = self._weights(theta, feats)
        base_prob = base_probs[target]
        edit_prob = edit_probs[target]
        q = max(base_weight * base_prob + edit_weight * edit_prob, EPS)
        width = len(feats)
        grad = [0.0] * len(theta)
        grad_base_logit = base_weight * (base_prob - q)
        grad_edit_logit = edit_weight * (edit_prob - q)
        for j, feat in enumerate(feats):
            grad[j] = grad_base_logit * feat
            grad[width + j] = grad_edit_logit * feat
        return q, grad, {
            "base_weight": base_weight,
            "edit_weight": edit_weight,
            "edit_candidate_count": float(edit.candidate_count if edit is not None else 0),
        }

    def prob(self, ctx: bytes, target: int, theta: list[float]) -> float:
        q, _grad, _diag = self.prob_and_grad(ctx, target, theta)
        return q


def train_edit_correction_prior(
    prior: EditKernelCorrectionPrior,
    valid_tokens: bytes,
    depth: int,
    positions: list[int],
    epochs: int,
    lr: float,
    context_fn=None,
) -> tuple[list[float], list[dict[str, float]]]:
    theta = prior.initial_theta()
    m = [0.0] * len(theta)
    v = [0.0] * len(theta)
    beta1 = 0.9
    beta2 = 0.999
    history: list[dict[str, float]] = []
    step = 0
    for epoch in range(1, epochs + 1):
        random.Random(epoch).shuffle(positions)
        grad_sum = [0.0] * len(theta)
        total_loss = 0.0
        base_weight = 0.0
        edit_weight = 0.0
        edit_candidates = 0.0
        for pos in positions:
            ctx = valid_tokens[max(0, pos - depth) : pos]
            if context_fn is not None:
                ctx = context_fn(ctx, pos, epoch)
            target = valid_tokens[pos]
            q, grad_q, diag = prior.prob_and_grad(ctx, target, theta)
            total_loss -= math.log(max(q, EPS))
            for j, grad in enumerate(grad_q):
                grad_sum[j] += -grad / max(q, EPS)
            base_weight += diag["base_weight"]
            edit_weight += diag["edit_weight"]
            edit_candidates += diag["edit_candidate_count"]
        scale = 1.0 / max(len(positions), 1)
        step += 1
        for j in range(len(theta)):
            grad = grad_sum[j] * scale
            m[j] = beta1 * m[j] + (1.0 - beta1) * grad
            v[j] = beta2 * v[j] + (1.0 - beta2) * grad * grad
            m_hat = m[j] / (1.0 - beta1**step)
            v_hat = v[j] / (1.0 - beta2**step)
            theta[j] -= lr * m_hat / (math.sqrt(v_hat) + 1.0e-8)
        loss = total_loss * scale
        history.append(
            {
                "epoch": epoch,
                "loss": loss,
                "ppl": math.exp(loss),
                "avg_base_weight": base_weight * scale,
                "avg_edit_weight": edit_weight * scale,
                "avg_edit_candidates": edit_candidates * scale,
            }
        )
    return theta, history


def train_edit_kernel_prior(
    prior: EditKernelPrior,
    valid_tokens: bytes,
    depth: int,
    positions: list[int],
    epochs: int,
    lr: float,
    context_fn=None,
) -> tuple[list[float], list[dict[str, float]]]:
    theta = prior.initial_theta()
    m = [0.0] * len(theta)
    v = [0.0] * len(theta)
    beta1 = 0.9
    beta2 = 0.999
    history: list[dict[str, float]] = []
    step = 0

    for epoch in range(1, epochs + 1):
        random.Random(epoch).shuffle(positions)
        grad_sum = [0.0] * len(theta)
        total_loss = 0.0
        backoff_weight = 0.0
        exact_weight = 0.0
        edit_weight = 0.0
        edit_candidates = 0.0
        diag_count = 0

        for pos in positions:
            ctx = valid_tokens[max(0, pos - depth) : pos]
            if context_fn is not None:
                ctx = context_fn(ctx, pos, epoch)
            target = valid_tokens[pos]
            q, grad_q, diagnostics = prior.prob_and_grad(ctx, target, theta)
            total_loss -= math.log(max(q, EPS))
            for j, grad in enumerate(grad_q):
                grad_sum[j] += -grad / max(q, EPS)
            for row in diagnostics:
                backoff_weight += row["backoff_weight"]
                exact_weight += row["exact_weight"]
                edit_weight += row["edit_weight"]
                edit_candidates += row["edit_candidate_count"]
                diag_count += 1

        scale = 1.0 / max(len(positions), 1)
        step += 1
        for j in range(len(theta)):
            grad = grad_sum[j] * scale
            m[j] = beta1 * m[j] + (1.0 - beta1) * grad
            v[j] = beta2 * v[j] + (1.0 - beta2) * grad * grad
            m_hat = m[j] / (1.0 - beta1**step)
            v_hat = v[j] / (1.0 - beta2**step)
            theta[j] -= lr * m_hat / (math.sqrt(v_hat) + 1.0e-8)

        loss = total_loss * scale
        history.append(
            {
                "epoch": epoch,
                "loss": loss,
                "ppl": math.exp(loss),
                "avg_backoff_weight": backoff_weight / max(diag_count, 1),
                "avg_exact_weight": exact_weight / max(diag_count, 1),
                "avg_edit_weight": edit_weight / max(diag_count, 1),
                "avg_edit_candidates": edit_candidates / max(diag_count, 1),
            }
        )

    return theta, history


def score_prior(
    prob_fn,
    tokens: bytes,
    positions: Iterable[int],
    depth: int,
    context_fn=None,
) -> tuple[float, float, int]:
    total_loss = 0.0
    count = 0
    for pos in positions:
        ctx = tokens[max(0, pos - depth) : pos]
        if context_fn is not None:
            ctx = context_fn(ctx, pos, 0)
        target = tokens[pos]
        total_loss -= math.log(max(prob_fn(ctx, target), EPS))
        count += 1
    loss = total_loss / max(count, 1)
    return loss, math.exp(loss), count


def sampled_positions(tokens: bytes, max_positions: int, depth: int, seed: int) -> list[int]:
    return select_positions(len(tokens), max_positions, max(1, depth), seed)


def corrupt_context_once(
    ctx: bytes,
    vocab_size: int,
    probability: float,
    seed: int,
    max_depth: int,
    allow_transpose: bool = True,
    operations: tuple[str, ...] | None = None,
) -> bytes:
    if not ctx or probability <= 0.0:
        return ctx
    rng = random.Random(seed)
    if rng.random() >= probability:
        return ctx
    allowed = list(operations) if operations is not None else ["substitute", "delete", "insert", "transpose"]
    possible = ["substitute"]
    if "delete" in allowed and len(ctx) > 1:
        possible.append("delete")
    if "insert" in allowed and len(ctx) < max_depth:
        possible.append("insert")
    if "transpose" in allowed and allow_transpose and len(ctx) > 1:
        possible.append("transpose")
    op = rng.choice([name for name in possible if name in allowed] or ["substitute"])
    if op == "substitute":
        pos = rng.randrange(len(ctx))
        token = rng.randrange(vocab_size)
        if token == ctx[pos]:
            token = (token + 1) % vocab_size
        return ctx[:pos] + bytes([token]) + ctx[pos + 1 :]
    if op == "delete":
        pos = rng.randrange(len(ctx))
        return ctx[:pos] + ctx[pos + 1 :]
    if op == "insert":
        pos = rng.randrange(len(ctx) + 1)
        token = rng.randrange(vocab_size)
        return ctx[:pos] + bytes([token]) + ctx[pos:]
    pos = rng.randrange(len(ctx) - 1)
    return ctx[:pos] + bytes([ctx[pos + 1], ctx[pos]]) + ctx[pos + 2 :]
