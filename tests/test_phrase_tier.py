"""--phrase-tier: a third GRU, over the word tier, that steps once a phrase
(at each segment whose last character, skipping whitespace and closers, is a
mark), its state carried after the word GRU's. At gate 0 the model is the
word-tier model; an update split in two gives what the whole update gives;
a character never sees its own phrase's state; phrase ends are found as the
text says; a call with no phrase end carries the phrase state unchanged."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_train_streams import H, V, corpus, grads, make_model, ragged_segments, runner  # noqa: E402

from imm.segments import PHRASE_MARK, PHRASE_ORDINARY, PHRASE_SKIPPED, phrase_classes, phrase_end_segments, rule_segments  # noqa: E402

CPU = torch.device("cpu")
PARALLEL = dict(carry_bound="gru", memory_state="gru")
W = 5  # the word tier's width
P = 3  # the phrase tier's, unlike both, so a mix-up shows
SPACE, MARK, CLOSER = 0, 1, 2
CLASSES = [PHRASE_SKIPPED, PHRASE_MARK, PHRASE_SKIPPED] + [PHRASE_ORDINARY] * (V - 3)
TIERS = dict(word_tier=W, phrase_tier=P, phrase_classes=CLASSES, **PARALLEL)


def run(model, ids, rows, stream_segments, memory, hidden, max_memory=4, **kw):
    spans = [(segs[0][0], segs[-1][1]) for segs in stream_segments]
    given = {k: runner.prior_rows_for_spans(v, spans, CPU) for k, v in rows.items()}
    return model.segment_loss_streams(
        ids, stream_segments, memory, max_memory, carried_hidden=hidden, carry_hidden=True, **given, **kw
    )


def layout(stream_segments):
    """positions and valid [B, K, L], as segment_loss_streams lays them out."""
    starts = np.array([[a for a, _ in segs] for segs in stream_segments], dtype=np.int64)
    ends = np.array([[b for _, b in segs] for segs in stream_segments], dtype=np.int64)
    offsets = np.arange(int((ends - starts).max()))
    return starts[:, :, None] + offsets, offsets[None, None, :] < (ends - starts)[:, :, None]


def ends_by_hand(ids, stream_segments, classes):
    out = []
    for segs in stream_segments:
        row = []
        for a, b in segs:
            kinds = [classes[int(i)] for i in ids[a:b] if classes[int(i)] != PHRASE_SKIPPED]
            row.append(bool(kinds) and kinds[-1] == PHRASE_MARK)
        out.append(row)
    return out


class PhraseTierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ids, self.rows = corpus(n=160)
        self.segs = runner.split_streams(ragged_segments(len(self.ids) - 1), 3)
        marked = ends_by_hand(self.ids, [s[:15] for s in self.segs], CLASSES)
        # the random text must have phrase ends, and segments that aren't
        self.assertTrue(any(any(r) for r in marked) and not all(all(r) for r in marked))

    def test_gate_zero_is_the_word_tier_model(self) -> None:
        for layers in (1, 2):
            with self.subTest(layers=layers):
                base = make_model(layers=layers, word_tier=W, **PARALLEL)
                tier = make_model(layers=layers, **TIERS)
                with torch.no_grad():
                    tier.phrase_tier.gate.zero_()
                shared = dict(base.named_parameters())
                for name, p in tier.named_parameters():
                    if name in shared:
                        self.assertTrue(torch.equal(p, shared[name]), name)
                got = []
                for model in (base, tier):
                    model.train()
                    model.zero_grad()
                    memory, hidden, losses = None, None, []
                    for u in range(3):
                        part = [s[u * 5 : (u + 1) * 5] for s in self.segs]
                        loss, _, memory, hidden, _ = run(model, self.ids, self.rows, part, memory, hidden)
                        loss.backward()
                        losses.append(loss.detach())
                        memory, hidden = (memory[0].detach(), memory[1]), hidden.detach()
                    got.append((torch.stack(losses), grads(model), memory, hidden))
                (la, ga, ma, ha), (lb, gb, mb, hb) = got
                torch.testing.assert_close(la, lb, rtol=1e-6, atol=1e-6)
                for name in ga:
                    torch.testing.assert_close(ga[name], gb[name], rtol=1e-5, atol=1e-6, msg=name)
                torch.testing.assert_close(ma[0], mb[0], rtol=1e-6, atol=1e-6)
                self.assertEqual(hb.shape, (1, 3, H + W + P))
                torch.testing.assert_close(ha, hb[..., : H + W], rtol=1e-6, atol=1e-6)
                for name in ("phrase_tier.rnn.weight_ih_l0", "phrase_tier.proj.weight"):
                    self.assertEqual(float(gb[name].abs().sum()), 0.0, name)
                self.assertGreater(float(gb["phrase_tier.gate"].abs().sum()), 0.0)

    def test_split_update_is_the_whole_update(self) -> None:
        # Records never slide out (max_memory 32): two updates of 5 segments,
        # the state carried, are one of 10 (forward results, phrase state
        # included), and the phrase tier gets a gradient.
        model = make_model(layers=2, **TIERS)
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
        (la, ga, ma, ha), (lb, gb, mb, hb) = got
        self.assertAlmostEqual(la, lb, places=4)
        torch.testing.assert_close(ma[0], mb[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(ha, hb, rtol=1e-5, atol=1e-5)
        for name in ("phrase_tier.rnn.weight_ih_l0", "phrase_tier.rnn.weight_hh_l0", "phrase_tier.proj.weight", "phrase_tier.gate"):
            self.assertGreater(float(ga[name].abs().sum()), 0.0, name)

    def test_a_character_never_sees_its_own_phrase(self) -> None:
        # Make segment k end a phrase, or stop it ending one, by its last
        # character: every state before that character is unchanged. Were a
        # segment's characters given the phrase state its own end makes,
        # those earlier in the segment would change.
        model = make_model(layers=2, **TIERS)
        model.eval()
        part = [s[:8] for s in self.segs]
        start = part[0][0][0]
        tried = 0
        for k in range(1, 7):
            seg_start, seg_end = part[0][k]
            if seg_end - seg_start < 2:
                continue
            q = seg_end - 1
            states = []
            for flip in (False, True):
                ids = self.ids.clone()
                if flip:
                    ids[q] = 3 if int(ids[q]) == MARK else MARK
                sink = []
                model.state_sink = sink
                try:
                    with torch.no_grad():
                        run(model, ids, self.rows, part, None, None)
                finally:
                    model.state_sink = None
                states.append(torch.cat([s for s, _ in sink]).float())
            before, after = states
            n = q - start
            with self.subTest(segment=k):
                self.assertTrue(torch.equal(before[:n], after[:n]))
                self.assertFalse(torch.equal(before[n], after[n]))
            tried += 1
        self.assertGreater(tried, 1)

    def test_phrase_state_reaches_later_segments(self) -> None:
        model = make_model(layers=1, **TIERS)
        model.eval()
        part = [s[:8] for s in self.segs]
        with torch.no_grad():
            on = float(run(model, self.ids, self.rows, part, None, None)[0])
            gate = model.phrase_tier.gate.clone()
            model.phrase_tier.gate.zero_()
            off = float(run(model, self.ids, self.rows, part, None, None)[0])
            model.phrase_tier.gate.copy_(gate)
        self.assertNotAlmostEqual(on, off, places=4)

    def test_no_phrase_end_carries_the_state(self) -> None:
        model = make_model(layers=1, **TIERS)
        model.eval()
        ids = torch.where(self.ids == MARK, torch.full_like(self.ids, 3), self.ids)
        part = [s[:6] for s in self.segs]
        hidden = torch.randn(1, 3, H + W + P)
        with torch.no_grad():
            out = run(model, ids, self.rows, part, None, hidden)[3]
        self.assertTrue(torch.equal(out[..., H + W :], hidden[..., H + W :]))
        self.assertFalse(torch.equal(out[..., : H + W], hidden[..., : H + W]))

    def test_phrase_ends_in_text(self) -> None:
        text = 'He said, "go." Then: (so) it ends; or (does it?) Not\nquite! x'
        chars = sorted(set(text))
        ids = np.array([chars.index(c) for c in text], dtype=np.int64)
        space_ids = [i for i, c in enumerate(chars) if c.isspace()]
        segs, _ = rule_segments("word", ids, ids[:2], 16, 1, space_ids)
        words = [text[a:b] for a, b in segs]
        self.assertEqual(words[:3], ["He ", "said, ", '"go." '])
        for marks, want in (
            (",;:.!?", {"said, ", '"go." ', "Then: ", "ends; ", "it?) ", "quite! "}),
            (".!?", {'"go." ', "it?) ", "quite! "}),
        ):
            with self.subTest(marks=marks):
                classes = phrase_classes(chars, marks)
                fake = SimpleNamespace(phrase_class=torch.tensor(classes))
                positions, valid = layout([segs])
                got = runner.SegmentMemoryLM.phrase_ends(fake, torch.from_numpy(ids), positions, valid)
                self.assertEqual({w for w, e in zip(words, got[0].tolist()) if e}, want)
                self.assertEqual(got[0].tolist(), ends_by_hand(ids, [segs], classes)[0])
                self.assertEqual(phrase_end_segments(ids, segs, classes).tolist(), got[0].tolist())

    def test_phrase_ends_on_random_text(self) -> None:
        # The model's per-update phrase_ends and the whole-text
        # phrase_end_segments agree, on the ragged random segments.
        fake = SimpleNamespace(phrase_class=torch.tensor(CLASSES))
        part = [s[:15] for s in self.segs]
        positions, valid = layout(part)
        got = runner.SegmentMemoryLM.phrase_ends(fake, self.ids, positions, valid)
        flat = [seg for segs in part for seg in segs]
        want = phrase_end_segments(self.ids.numpy(), flat, CLASSES).reshape(len(part), -1)
        self.assertEqual(got.tolist(), want.tolist())
        self.assertEqual(got.tolist(), ends_by_hand(self.ids, part, CLASSES))

    def test_refusals(self) -> None:
        with self.assertRaisesRegex(ValueError, "give --word-tier too"):
            make_model(phrase_tier=P, phrase_classes=CLASSES, **PARALLEL)
        with self.assertRaisesRegex(ValueError, "a phrase class for each"):
            make_model(word_tier=W, phrase_tier=P, **PARALLEL)
        with self.assertRaisesRegex(ValueError, "a phrase class for each"):
            make_model(word_tier=W, phrase_tier=P, phrase_classes=CLASSES[:-1], **PARALLEL)
        with self.assertRaisesRegex(ValueError, "phrase_tier must be"):
            make_model(word_tier=W, phrase_tier=-1, **PARALLEL)
        for marks in ("", ", ", ".'"):
            with self.subTest(marks=marks), self.assertRaises(ValueError):
                phrase_classes(list(" a.,'"), marks)

    def test_loss_sink_keeps_each_scored_character(self) -> None:
        model = make_model(layers=1, **TIERS)
        model.eval()
        part = [s[:6] for s in self.segs]
        sink = []
        model.loss_sink = sink
        try:
            with torch.no_grad():
                loss, scored, *_ = run(model, self.ids, self.rows, part, None, None, collect_stats=True)
        finally:
            model.loss_sink = None
        positions = torch.cat([p for p, _ in sink])
        losses = torch.cat([x for _, x in sink])
        self.assertEqual(len(positions), scored)
        want = sorted(i for segs in part for a, b in segs for i in range(a, b))
        self.assertEqual(sorted(positions.tolist()), want)
        self.assertAlmostEqual(float(losses.sum()), float(loss), places=3)


if __name__ == "__main__":
    unittest.main()
