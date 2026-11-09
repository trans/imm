"""--segment-rule: word and hybrid segments (imm/segments.py, rule_segments)."""

import re
import unittest

import numpy as np

from imm.data import CharVocab
from imm.segments import rule_segments, segment_lengths, unique_segments

TRAIN = "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.\n" * 3
HELD = "  You are all resolved rather to die than to famish?\nAll:\nResolved. resolved.\n"
VOCAB = CharVocab.from_text(TRAIN + HELD + "Y?R")
SPACES = [i for i, ch in enumerate(VOCAB.chars) if ch.isspace()]


def by_regex(text: str, max_depth: int) -> list[tuple[int, int]]:
    """Each \\S+\\s* match (or leading whitespace), split every max_depth
    characters, never covering the last character."""
    matches = list(re.finditer(r"\S+\s*", text))
    cuts = sorted({0, len(text) - 1, matches[0].start()} | {m.end() for m in matches})
    segments = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        b = min(b, len(text) - 1)
        while a < b:
            segments.append((a, min(a + max_depth, b)))
            a = min(a + max_depth, b)
    return segments


class SegmentRuleTest(unittest.TestCase):
    def test_word_rule_matches_the_regex(self) -> None:
        for max_depth in (16, 4):
            train, held = rule_segments(
                "word", VOCAB.encode(TRAIN), VOCAB.encode(HELD), max_depth, 1, SPACES
            )
            self.assertEqual(train, by_regex(TRAIN, max_depth), max_depth)
            self.assertEqual(held, by_regex(HELD, max_depth), max_depth)

    def test_unique_rule_is_unchanged(self) -> None:
        train, held = VOCAB.encode(TRAIN), VOCAB.encode(HELD)
        self.assertEqual(rule_segments("unique", train, held, 16, 1, SPACES), unique_segments(train, held, 16, 1))

    def test_hybrid_extends_unique_segments_to_a_word_boundary(self) -> None:
        train, held = VOCAB.encode(TRAIN), VOCAB.encode(HELD)
        max_depth = 16
        for ids, segments in zip((train, held), rule_segments("hybrid", train, held, max_depth, 1, SPACES)):
            text = VOCAB.decode(ids)
            unique = segment_lengths([np.asarray(train), np.asarray(held)], max_depth, 1)[0 if ids is train else 1]
            self.assertEqual(segments[0][0], 0)
            self.assertEqual(segments[-1][1], len(ids) - 1)
            for (a, b), (c, _) in zip(segments, segments[1:]):
                self.assertEqual(b, c)  # contiguous
            word_start = [i for i in range(1, len(text)) if text[i - 1].isspace() and not text[i].isspace()]
            for a, b in segments:
                length = b - a
                self.assertLessEqual(length, max_depth)
                if b == len(ids) - 1:
                    continue  # the text's end
                inside = [w for w in word_start if a < w <= a + max_depth]
                if inside:
                    # ends where a word begins: a space, then a non-space
                    self.assertIn(b, word_start, (a, b, text[a:b + 1]))
                    nxt = next(w for w in word_start if w >= a + int(unique[a]))
                    # at the first word start past the unique length, or the
                    # last one within the cap if that is too far
                    self.assertEqual(b, nxt if nxt - a <= max_depth else inside[-1], (a, b, text[a:b + 1]))
                else:
                    self.assertEqual(length, max_depth)  # one word longer than the cap

    def test_unknown_rule(self) -> None:
        with self.assertRaises(ValueError):
            rule_segments("sentence", [0, 1], [0, 1], 16, 1, SPACES)


if __name__ == "__main__":
    unittest.main()
