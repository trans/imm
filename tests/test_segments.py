"""Segmentation by training-text counts (imm/segments.py) against the trie it
replaced, copied here unchanged as the reference."""

import unittest

import numpy as np

from imm.segments import segment_lengths, unique_segments


class PrefixCountTrie:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.counts: list[int] = [0]
        self.children: list[dict[int, int]] = [{}]

    def insert_suffixes(self, ids: list[int], max_depth: int) -> None:
        limit = len(ids)
        for start in range(limit):
            node = 0
            self.counts[node] += 1
            end = min(limit, start + max_depth)
            for pos in range(start, end):
                token = ids[pos]
                child = self.children[node].get(token)
                if child is None:
                    child = len(self.counts)
                    self.children[node][token] = child
                    self.counts.append(0)
                    self.children.append({})
                node = child
                self.counts[node] += 1


def segment_ids(
    ids: list[int],
    trie: PrefixCountTrie,
    max_depth: int,
    unique_threshold: int,
) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    pos = 0
    input_limit = len(ids) - 1
    while pos < input_limit:
        start = pos
        node = 0
        depth = 0
        while pos < input_limit and depth < max_depth:
            token = ids[pos]
            child = trie.children[node].get(token)
            pos += 1
            depth += 1
            if child is None:
                break
            node = child
            if trie.counts[node] <= unique_threshold:
                break
        if pos == start:
            pos += 1
        segments.append((start, pos))
    return segments


def by_trie(train, held, max_depth, threshold):
    trie = PrefixCountTrie(0)
    trie.insert_suffixes(train, max_depth)
    return segment_ids(train, trie, max_depth, threshold), segment_ids(held, trie, max_depth, threshold)


def repetitive_text(n: int, alphabet: int, seed: int) -> list[int]:
    # Pieces of a few short phrases, so substrings repeat at many lengths,
    # with some random characters between.
    rng = np.random.default_rng(seed)
    phrases = [rng.integers(0, alphabet, size=rng.integers(3, 12)).tolist() for _ in range(5)]
    out: list[int] = []
    while len(out) < n:
        if rng.random() < 0.7:
            p = phrases[rng.integers(len(phrases))]
            out.extend(p[: rng.integers(1, len(p) + 1)])
        else:
            out.append(int(rng.integers(0, alphabet)))
    return out[:n]


class UniqueSegmentsTest(unittest.TestCase):
    def check(self, train, held, max_depth, threshold):
        want = by_trie(train, held, max_depth, threshold)
        got = unique_segments(train, held, max_depth, threshold)
        self.assertEqual(got[0], want[0], f"train, depth {max_depth}, threshold {threshold}")
        self.assertEqual(got[1], want[1], f"eval, depth {max_depth}, threshold {threshold}")

    def test_matches_the_trie(self) -> None:
        cases = 0
        for seed in range(6):
            alphabet = (2, 3, 5, 8, 20, 65)[seed]
            train = repetitive_text(400, alphabet, seed)
            held = repetitive_text(150, alphabet, seed + 100)
            held[10:13] = [alphabet, alphabet + 1, 0]  # characters the training text lacks
            for max_depth in (1, 2, 3, 5, 8, 16):
                for threshold in (-1, 0, 1, 2, 5):
                    self.check(train, held, max_depth, threshold)
                    cases += 1
        self.assertEqual(cases, 180)

    def test_uniform_random_text(self) -> None:
        rng = np.random.default_rng(7)
        train = rng.integers(0, 4, size=600).tolist()
        held = rng.integers(0, 4, size=200).tolist()
        for max_depth in (4, 16):
            self.check(train, held, max_depth, 1)

    def test_short_and_empty_texts(self) -> None:
        for train, held in [([], []), ([3], []), ([1, 2], [2]), ([1, 1, 1], [1, 1]), ([0, 1, 0, 1], [])]:
            for max_depth in (0, 1, 4):
                self.check(train, held, max_depth, 1)

    def test_lengths_stop_before_the_last_character(self) -> None:
        train = [0] * 20
        (lengths,) = segment_lengths([np.asarray(train)], 16, 1)
        # Every substring of a run of zeros is common, so each segment runs to
        # max_depth or to the character before the last.
        self.assertEqual(lengths[:3].tolist(), [16, 16, 16])
        self.assertEqual(lengths[10:19].tolist(), [9, 8, 7, 6, 5, 4, 3, 2, 1])


if __name__ == "__main__":
    unittest.main()
