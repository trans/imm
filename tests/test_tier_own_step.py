"""--tier-own-step: a segment's last character gets the word state its own
segment makes (and a phrase's last character the phrase state after it),
not the one before. Only those characters change; nothing before a changed
character changes, so it stays causal; an update split in two gives what the
whole update gives; it needs the word tier."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_phrase_tier import CLASSES, PARALLEL, P, W, ends_by_hand, run  # noqa: E402
from test_train_streams import V, corpus, grads, make_model, ragged_segments, runner  # noqa: E402

TIERS = dict(word_tier=W, phrase_tier=P, phrase_classes=CLASSES, **PARALLEL)


def states(model, ids, rows, part):
    """The final state at every position, streams in order, positions in order."""
    sink = []
    model.state_sink = sink
    try:
        with torch.no_grad():
            run(model, ids, rows, part, None, None)
    finally:
        model.state_sink = None
    return torch.cat([s for s, _ in sink]).float()


def flat_mask(part, pick):
    """[N] bool over the positions as states() lays them out: pick(stream,
    k, position, segment) for each."""
    out = []
    for b, segs in enumerate(part):
        for k, (a, e) in enumerate(segs):
            out.extend(pick(b, k, q, (a, e)) for q in range(a, e))
    return torch.tensor(out)


class TierOwnStepTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ids, self.rows = corpus(n=160)
        self.segs = runner.split_streams(ragged_segments(len(self.ids) - 1), 3)
        self.part = [s[:8] for s in self.segs]

    def test_only_a_segments_last_character_changes(self) -> None:
        # The same weights, own step off and on: every state is the same but
        # at a segment's last character, and those differ.
        for layers in (1, 2):
            with self.subTest(layers=layers):
                off = make_model(layers=layers, word_tier=W, **PARALLEL)
                on = make_model(layers=layers, word_tier=W, tier_own_step=True, **PARALLEL)
                for (name, a), (_, b) in zip(off.named_parameters(), on.named_parameters()):
                    self.assertTrue(torch.equal(a, b), name)
                off.eval()
                on.eval()
                a, b = states(off, self.ids, self.rows, self.part), states(on, self.ids, self.rows, self.part)
                last = flat_mask(self.part, lambda b_, k, q, seg: q == seg[1] - 1)
                self.assertEqual(len(last), len(a))
                torch.testing.assert_close(a[~last], b[~last], rtol=0, atol=0)
                differs = (a[last] - b[last]).abs().amax(dim=-1) > 1e-6
                self.assertTrue(bool(differs.all()), f"{int((~differs).sum())} last characters unchanged")

    def test_only_a_phrases_last_character_changes_with_the_phrase_tier(self) -> None:
        # Both tiers own-step; the phrase tier's alone turned off: only the
        # last characters of the segments that end a phrase differ.
        model = make_model(layers=2, tier_own_step=True, **TIERS)
        model.eval()
        a = states(model, self.ids, self.rows, self.part)
        model.phrase_tier.own_step = False
        b = states(model, self.ids, self.rows, self.part)
        ends = ends_by_hand(self.ids, self.part, CLASSES)
        self.assertTrue(any(any(r) for r in ends))
        mark = flat_mask(self.part, lambda b_, k, q, seg: ends[b_][k] and q == seg[1] - 1)
        torch.testing.assert_close(a[~mark], b[~mark], rtol=0, atol=0)
        self.assertTrue(bool(((a[mark] - b[mark]).abs().amax(dim=-1) > 1e-6).all()))

    def test_still_causal(self) -> None:
        # Change one input character, at a segment's first or last character:
        # no state before it changes. The last characters of segments that
        # end a phrase are tried too. A planted leak (every character of a
        # segment given its own segment's word or phrase state) fails this.
        model = make_model(layers=2, tier_own_step=True, **TIERS)
        model.eval()
        ends = ends_by_hand(self.ids, self.part, CLASSES)
        tries = [(0, 3, "last"), (0, 3, "first"), (0, 5, "last")]
        phrase = [(b, k, "last") for b, segs in enumerate(self.part) for k, (a, e) in enumerate(segs) if ends[b][k] and e - a >= 2]
        self.assertTrue(phrase, "no phrase-ending segment of two characters")
        for b, k, at in tries + phrase:
            seg_start, seg_end = self.part[b][k]
            q = seg_end - 1 if at == "last" else seg_start
            with self.subTest(stream=b, segment=k, at=at):
                got = []
                for flip in (False, True):
                    ids = self.ids.clone()
                    if flip:
                        ids[q] = (ids[q] + 1) % V
                    got.append(states(model, ids, self.rows, self.part))
                before, after = got
                n = int(flat_mask(self.part, lambda b_, k_, q_, seg: (b_, q_) < (b, q)).sum())  # positions before q
                self.assertTrue(torch.equal(before[:n], after[:n]))
                self.assertFalse(torch.equal(before[n], after[n]))

    def test_split_update_is_the_whole_update(self) -> None:
        # The carried states don't depend on own step, but check it: two
        # updates of 5 segments are one of 10 (records never slide out).
        model = make_model(layers=2, tier_own_step=True, **TIERS)
        model.train()
        got = []
        for pieces in ((slice(0, 10),), (slice(0, 5), slice(5, 10))):
            model.zero_grad()
            memory, hidden, total = None, None, 0.0
            for piece in pieces:
                part = [s[piece] for s in self.segs]
                loss, _, memory, hidden, _ = run(model, self.ids, self.rows, part, memory, hidden, max_memory=32)
                loss.backward()
                total += float(loss.detach())
                memory, hidden = (memory[0].detach(), memory[1]), hidden.detach()
            got.append((total, grads(model), memory, hidden))
        (la, ga, ma, ha), (lb, _, mb, hb) = got
        self.assertAlmostEqual(la, lb, places=4)
        torch.testing.assert_close(ma[0], mb[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(ha, hb, rtol=1e-5, atol=1e-5)
        for name in ("word_tier.proj.weight", "phrase_tier.proj.weight", "phrase_tier.gate"):
            self.assertGreater(float(ga[name].abs().sum()), 0.0, name)

    def test_needs_the_word_tier(self) -> None:
        with self.assertRaisesRegex(ValueError, "give --word-tier too"):
            make_model(tier_own_step=True, **PARALLEL)


if __name__ == "__main__":
    unittest.main()
