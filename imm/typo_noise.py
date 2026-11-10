"""Typo noise: substitute characters of a text, each replacement drawn from a
unigram distribution over the vocabulary (Xie et al. 2017, "Data Noising as
Smoothing in Neural Network Language Models"), never the character it
replaces.

Substitutions only, so the noisy text keeps every position: prior rows and
segment indices computed for the clean text still line up with it."""

from __future__ import annotations

import numpy as np


def unigram_probs(ids, vocab_size: int) -> np.ndarray:
    counts = np.bincount(np.asarray(ids, dtype=np.int64), minlength=vocab_size).astype(np.float64)
    if counts.sum() <= 0:
        raise ValueError("unigram_probs needs at least one id")
    return counts / counts.sum()


def substitute(ids, rate: float, probs: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """A copy of ids with about rate of its positions replaced. Returns the
    copy and the number of positions changed. A replacement is drawn from
    probs given that it differs from the original: drawn again while it
    equals it."""
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {rate}")
    original = np.asarray(ids, dtype=np.int64)
    out = original.copy()
    if rate == 0.0 or out.size == 0:
        return out, 0
    if np.count_nonzero(probs) < 2:
        raise ValueError("substitute needs at least two characters with nonzero probability")
    chosen = np.flatnonzero(rng.random(out.size) < rate)
    pending = chosen
    for _ in range(1000):
        if pending.size == 0:
            break
        out[pending] = rng.choice(probs.size, size=pending.size, p=probs)
        pending = pending[out[pending] == original[pending]]
    else:
        raise RuntimeError("substitute could not draw replacements that differ from the original")
    return out, int(chosen.size)
