"""Cut text into variable-length segments: by how often each stretch occurs
in the training text (the unique rule), at word boundaries (the word rule),
or by the unique rule extended to the next word boundary (hybrid).

A segment starting at position p runs to the first length L at which
text[p:p+L] occurs at most `unique_threshold` times in the training text
(or not at all). It stops sooner at `max_depth` characters, and it never
covers a text's last character, which is only ever a target.

This replaces a trie of every training substring up to `max_depth`
characters, which took ~255 bytes a node and ~6-9 nodes a character (about
8 GB for 4.75M characters). Here the substrings of each length are numbered
with one `np.unique` per length. A position drops out once its substring is
rare. Its longer substrings are rarer still, and no substring that is still
common has it as a prefix, so dropping it changes no remaining count.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def unique_segments(
    train_ids: Sequence[int],
    eval_ids: Sequence[int],
    max_depth: int,
    unique_threshold: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Segments of the training and evaluation texts, both cut by counts in
    the training text alone. Each is a list of (start, end) pairs."""
    train = np.asarray(train_ids, dtype=np.int64)
    held = np.asarray(eval_ids, dtype=np.int64)
    train_lengths, eval_lengths = segment_lengths([train, held], max_depth, unique_threshold)
    return walk(train_lengths), walk(eval_lengths)


def segment_lengths(texts: list[np.ndarray], max_depth: int, unique_threshold: int) -> list[np.ndarray]:
    """For each text, the length of a segment starting at each position.
    Counts come from texts[0]; a substring is counted only where it lies
    wholly inside it. Entries at a text's last position are meaningless."""
    sizes = [len(t) for t in texts]
    x = np.concatenate(texts) if sum(sizes) else np.zeros(0, dtype=np.int64)
    n = len(x)
    ends = np.repeat(np.cumsum(sizes), sizes)  # the end of each position's own text
    cap = ends - 1 - np.arange(n)  # a segment may not cover the last character
    lengths = np.minimum(cap, max_depth)  # if nothing ever becomes rare
    rare_at_most = max(unique_threshold, 0)
    base = int(x.max()) + 1 if n else 1

    pos = np.arange(n)  # positions whose substring is still common
    code = np.zeros(n, dtype=np.int64)  # dense number of text[p:p+L-1]
    for depth in range(1, max_depth + 1):
        fits = pos + depth <= ends[pos]
        pos, code = pos[fits], code[fits]
        if not len(pos):
            break
        keys = code * base + x[pos + depth - 1]
        _, code = np.unique(keys, return_inverse=True)
        counts = np.bincount(code[pos < sizes[0]], minlength=int(code.max()) + 1)
        rare = counts[code] <= rare_at_most
        hit = pos[rare]
        lengths[hit] = np.minimum(depth, cap[hit])
        pos, code = pos[~rare], code[~rare]

    lengths = np.maximum(lengths, 1)
    bounds = np.cumsum([0, *sizes])
    return [lengths[a:b] for a, b in zip(bounds[:-1], bounds[1:])]


SEGMENT_RULES = ("unique", "word", "hybrid")


def rule_segments(
    rule: str,
    train_ids: Sequence[int],
    eval_ids: Sequence[int],
    max_depth: int,
    unique_threshold: int,
    space_ids: Sequence[int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Segments of the training and evaluation texts under rule:
    - unique: unique_segments, counted in the training text;
    - word: each segment is a word and the whitespace after it (the matches
      of the regex \\S+\\s*, with punctuation part of the word), split at
      max_depth;
    - hybrid: the unique length, extended to the next word boundary.
    All are capped at max_depth and never cover a text's last character.
    space_ids are the IDs of whitespace characters."""
    if rule not in SEGMENT_RULES:
        raise ValueError(f"unknown segment rule: {rule}")
    if rule == "unique":
        return unique_segments(train_ids, eval_ids, max_depth, unique_threshold)
    texts = [np.asarray(train_ids, dtype=np.int64), np.asarray(eval_ids, dtype=np.int64)]
    if rule == "word":
        lengths = [word_lengths(t, space_ids, max_depth, np.ones(len(t), dtype=np.int64)) for t in texts]
    else:
        unique = segment_lengths(texts, max_depth, unique_threshold)
        lengths = [word_lengths(t, space_ids, max_depth, u) for t, u in zip(texts, unique)]
    return walk(lengths[0]), walk(lengths[1])


def word_starts(text: np.ndarray, space_ids: Sequence[int]) -> np.ndarray:
    """Positions where a word begins: a non-space whose predecessor is a
    space (or that starts the text)."""
    space = np.isin(text, np.asarray(list(space_ids), dtype=np.int64))
    start = ~space
    start[1:] &= space[:-1]
    return np.flatnonzero(start)


def word_lengths(
    text: np.ndarray, space_ids: Sequence[int], max_depth: int, at_least: np.ndarray
) -> np.ndarray:
    """For each position p, the length of a segment that starts at p, runs at
    least at_least[p] characters, and then ends where the next word begins
    (or at the text's end), capped at max_depth and short of the last
    character. With at_least all ones, that is the word rule."""
    n = len(text)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    pos = np.arange(n)
    starts = word_starts(text, space_ids)
    reach = pos + np.maximum(at_least, 1)  # the segment ends at or after here
    i = np.searchsorted(starts, reach, side="left")
    end = np.where(i < len(starts), starts[np.minimum(i, len(starts) - 1)], n)
    cap = n - 1 - pos  # a segment may not cover the last character
    return np.maximum(np.minimum(np.minimum(end - pos, max_depth), cap), 1)


def walk(lengths: np.ndarray) -> list[tuple[int, int]]:
    """Segments from position 0, each starting where the last one ended."""
    steps = lengths.tolist()
    limit = len(steps) - 1
    segments = []
    pos = 0
    while pos < limit:
        end = pos + steps[pos]
        segments.append((pos, end))
        pos = end
    return segments
