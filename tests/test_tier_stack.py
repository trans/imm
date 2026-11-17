"""--sentence-tier and --paragraph-tier (BoundaryTier): a stack over the word
tier. At bptt 0 with the end state alone a boundary tier is the phrase tier;
--tier-bptt changes no forward value, and its gradient reaches the tier's
own earlier steps but nothing below; --tier-input end+centroid reads each
unit's mean of the states below, carried across updates; an update split in
two gives the whole update's values; it stays causal; a piece reset under
--documents reads as a fresh stream."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_phrase_tier import CLASSES, PARALLEL, W, run  # noqa: E402
from test_tier_own_step import flat_mask, states  # noqa: E402
from test_train_streams import H, V, corpus, grads, make_model, ragged_segments, runner  # noqa: E402

NEWLINE = 0  # the "space" id of the toy vocabulary doubles as a newline: two in a segment end a paragraph
S_W, P_W = 4, 3  # sentence and paragraph widths, unlike the word tier's and each other's
STACK = dict(word_tier=W, sentence_tier=S_W, paragraph_tier=P_W, sentence_classes=CLASSES, newline_id=NEWLINE,
             tier_own_step=True, **PARALLEL)


class BoundaryTierTest(unittest.TestCase):
    def test_at_bptt_0_the_end_state_tier_is_the_phrase_tier(self) -> None:
        torch.manual_seed(0)
        B, K, T = 3, 6, 20
        phrase = runner.PhraseTier(W, P_W, H, own_step=True)
        tier = runner.BoundaryTier(W, P_W, H, own_step=True)
        tier.load_state_dict(phrase.state_dict())
        with torch.no_grad():
            tier.gate.fill_(0.7)
            phrase.gate.fill_(0.7)
        after = torch.randn(B, K, W)
        ends = torch.rand(B, K) > 0.5
        carry = torch.randn(1, B, P_W)
        segment_of = torch.sort(torch.randint(0, K, (B, T)), dim=1).values
        at_last = torch.rand(B, T) > 0.7
        want_add, want_out = phrase(after, ends, carry, segment_of, lambda x: x, at_last)
        got_add, got_out, seg_after = tier(after, ends, carry, segment_of, lambda x: x, at_last)
        torch.testing.assert_close(got_add, want_add)
        torch.testing.assert_close(got_out, want_out)
        self.assertEqual(seg_after.shape, (B, K, P_W))

    def test_centroid_is_each_units_mean(self) -> None:
        torch.manual_seed(1)
        B, K, T = 2, 7, 10
        tier = runner.BoundaryTier(W, P_W, H, centroid=True)
        seen = []
        tier.rnn.register_forward_pre_hook(lambda m, args: seen.append(args[0].detach().clone()))
        after = torch.randn(B, K, W)
        ends = torch.zeros(B, K, dtype=torch.bool)
        ends[0, [1, 4]] = True
        ends[1, [0, 6]] = True
        carry = torch.zeros(1, B, tier.carry_width())
        carry[0, :, P_W : P_W + W] = torch.ones(W) * 3.0  # a unit begun before: sum 3, count 2
        carry[0, :, P_W + W] = 2.0
        segment_of = torch.zeros(B, T, dtype=torch.long)
        _, out, _ = tier(after, ends, carry, segment_of, lambda x: x)
        seq = seen[0]
        for b in range(B):
            done, start, run_sum, run_n = 0, 0, torch.full((W,), 3.0), 2.0
            for k in range(K):
                run_sum, run_n = run_sum + after[b, k], run_n + 1
                if ends[b, k]:
                    torch.testing.assert_close(seq[b, done, :W], after[b, k])
                    torch.testing.assert_close(seq[b, done, W:], run_sum / run_n)
                    done, run_sum, run_n = done + 1, torch.zeros(W), 0.0
            # what's carried out: the sum and count since the last end
            torch.testing.assert_close(out[0, b, P_W : P_W + W], run_sum)
            self.assertAlmostEqual(float(out[0, b, P_W + W].detach()), run_n)


class StackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ids, self.rows = corpus(n=240)
        self.segs = runner.split_streams(ragged_segments(len(self.ids) - 1), 3)

    def losses(self, model, pieces, max_memory=32):
        """The loss of each update, eval mode, state carried."""
        out, memory, hidden = [], None, None
        with torch.no_grad():
            for piece in pieces:
                part = [s[piece] for s in self.segs]
                loss, _, memory, hidden, _ = run(model, self.ids, self.rows, part, memory, hidden, max_memory=max_memory)
                out.append(float(loss))
        return out, memory, hidden

    def test_the_tiers_have_boundaries_to_step_at(self) -> None:
        model = make_model(layers=2, **STACK)
        part = [s[:12] for s in self.segs]
        starts = np.array([[a for a, _ in segs] for segs in part])
        lengths = np.array([[e - a for a, e in segs] for segs in part])
        offsets = np.arange(lengths.max())
        positions = starts[:, :, None] + offsets
        valid = offsets[None, None, :] < lengths[:, :, None]
        sentence, paragraph = model.boundary_ends(self.ids, positions, valid)
        self.assertTrue(bool(paragraph.any()) and bool(sentence.any()))
        self.assertTrue(bool((sentence | ~paragraph).all()), "a paragraph's end ends its sentence")
        for b, segs in enumerate(part):
            for k, (a, e) in enumerate(segs):
                self.assertEqual(bool(paragraph[b, k]), int((self.ids[a:e] == NEWLINE).sum()) >= 2)

    def test_bptt_changes_no_forward_value(self) -> None:
        for centroid in (False, True):
            with self.subTest(centroid=centroid):
                pieces = [slice(4 * u, 4 * u + 4) for u in range(5)]
                a = make_model(layers=2, tier_centroid=centroid, **STACK)
                b = make_model(layers=2, tier_centroid=centroid, tier_bptt=3, **STACK)
                b.load_state_dict(a.state_dict())
                a.eval()
                b.eval()
                la, _, _ = self.losses(a, pieces)
                lb, _, _ = self.losses(b, pieces)
                for x, y in zip(la, lb):
                    self.assertAlmostEqual(x, y, places=4)

    def test_bptt_gradient_reaches_the_tiers_own_earlier_steps_only(self) -> None:
        got = {}
        for bptt in (0, 3):
            model = make_model(layers=2, tier_bptt=bptt, tier_centroid=True, **STACK)
            model.train()
            memory, hidden = None, None
            for u in range(3):
                model.zero_grad()
                part = [s[4 * u : 4 * u + 4] for s in self.segs]
                loss, _, memory, hidden, _ = run(model, self.ids, self.rows, part, memory, hidden, max_memory=32)
                loss.backward()
                memory, hidden = (memory[0].detach(), memory[1]), hidden.detach()
            got[bptt] = grads(model)  # the last update's
        a, b = got[0], got[3]
        for tier in ("sentence_tier.rnn", "paragraph_tier.rnn"):
            # (weight_hh alone may agree: a buffered step from a zero state gives it nothing)
            names = [n for n in a if n.startswith(tier)]
            differs = [n for n in names if not torch.allclose(a[n], b[n], rtol=1e-4, atol=1e-6)]
            self.assertTrue(differs, f"{tier}: no gradient from earlier steps")
        for name in a:
            if not name.startswith(("sentence_tier", "paragraph_tier")):
                torch.testing.assert_close(a[name], b[name], rtol=1e-4, atol=1e-6, msg=name)

    def test_split_update_is_the_whole_update(self) -> None:
        for kw in (dict(), dict(tier_bptt=3, tier_centroid=True)):
            with self.subTest(**kw):
                model = make_model(layers=2, **kw, **STACK)
                model.eval()
                whole, mw, hw = self.losses(model, [slice(0, 12)])
                split, ms, hs = self.losses(model, [slice(0, 5), slice(5, 9), slice(9, 12)])
                self.assertAlmostEqual(whole[0], sum(split), places=4)
                torch.testing.assert_close(mw[0], ms[0], rtol=1e-5, atol=1e-5)
                width = model.hidden_size + model.word_tier_size + model.sentence_tier.width
                torch.testing.assert_close(hw[..., :width], hs[..., :width], rtol=1e-5, atol=1e-5)

    def test_still_causal(self) -> None:
        for top_down in (False, True):
            with self.subTest(top_down=top_down):
                self.check_causal(make_model(layers=2, tier_bptt=2, tier_centroid=True, top_down=top_down, **STACK))

    def check_causal(self, model) -> None:
        model.eval()
        part = [s[:10] for s in self.segs]
        for b, k in ((0, 3), (1, 5), (2, 8)):
            seg_start, seg_end = part[b][k]
            q = seg_end - 1
            got = []
            for flip in (False, True):
                ids = self.ids.clone()
                if flip:
                    ids[q] = (ids[q] + 1) % V
                got.append(states(model, ids, self.rows, part))
            before, after = got
            n = int(flat_mask(part, lambda b_, k_, q_, seg: (b_, q_) < (b, q)).sum())
            self.assertTrue(torch.equal(before[:n], after[:n]), f"stream {b} segment {k}")

    def test_top_down_at_zero_is_the_model_without(self) -> None:
        pieces = [slice(4 * u, 4 * u + 4) for u in range(4)]
        a = make_model(layers=2, tier_bptt=2, **STACK)
        b = make_model(layers=2, tier_bptt=2, top_down=True, **STACK)
        missing, unexpected = b.load_state_dict(a.state_dict(), strict=False)
        self.assertEqual(sorted(missing), ["top_down_char.bias", "top_down_char.weight", "top_down_word.bias", "top_down_word.weight"])
        with torch.no_grad():
            for layer in (b.top_down_char, b.top_down_word):
                layer.weight.zero_()
                layer.bias.zero_()
        a.eval()
        b.eval()
        for x, y in zip(self.losses(a, pieces)[0], self.losses(b, pieces)[0]):
            self.assertAlmostEqual(x, y, places=5)

    def test_top_down_reads_the_paragraph_state(self) -> None:
        model = make_model(layers=2, tier_bptt=2, top_down=True, **STACK)
        model.eval()
        part = [s[:6] for s in self.segs]
        hidden = model.start_hidden(3, torch.device("cpu"))
        width = model.carried_width() - model.paragraph_tier.carry_width()
        moved = hidden.clone()
        moved[..., width : width + model.paragraph_tier.width] += 0.5  # the paragraph state alone
        got = []
        for h in (hidden, moved):
            sink = []
            model.state_sink = sink
            with torch.no_grad():
                run(model, self.ids, self.rows, part, None, h)
            model.state_sink = None
            got.append(torch.cat([s for s, _ in sink]))
        self.assertGreater(float((got[0][0] - got[1][0]).abs().max()), 1e-4, "the first character didn't read it")

    def test_top_down_trains_the_paragraph_tier(self) -> None:
        # The paragraph tier's read-out off (gate 0): its GRU learns only
        # through top-down, from the buffered steps' rerun.
        model = make_model(layers=2, tier_bptt=3, top_down=True, **STACK)
        with torch.no_grad():
            model.paragraph_tier.gate.zero_()
        model.train()
        memory, hidden = None, None
        for u in range(3):
            model.zero_grad()
            part = [s[4 * u : 4 * u + 4] for s in self.segs]
            loss, _, memory, hidden, _ = run(model, self.ids, self.rows, part, memory, hidden, max_memory=32)
            loss.backward()
            memory, hidden = (memory[0].detach(), memory[1]), hidden.detach()
        g = grads(model)
        self.assertGreater(float(g["paragraph_tier.rnn.weight_ih_l0"].abs().sum()), 0.0)
        self.assertGreater(float(g["top_down_char.weight"].abs().sum()), 0.0)
        self.assertGreater(float(g["top_down_word.weight"].abs().sum()), 0.0)

    def test_refusals(self) -> None:
        with self.assertRaisesRegex(ValueError, "give --paragraph-tier too"):
            make_model(word_tier=W, sentence_tier=S_W, sentence_classes=CLASSES, top_down=True, **PARALLEL)
        with self.assertRaisesRegex(ValueError, "give --word-tier too"):
            make_model(sentence_tier=S_W, sentence_classes=CLASSES, **PARALLEL)
        with self.assertRaisesRegex(ValueError, "give --sentence-tier too"):
            make_model(word_tier=W, paragraph_tier=P_W, newline_id=NEWLINE, **PARALLEL)
        with self.assertRaisesRegex(ValueError, "replaces --phrase-tier"):
            make_model(word_tier=W, phrase_tier=P_W, phrase_classes=CLASSES, sentence_tier=S_W, sentence_classes=CLASSES, **PARALLEL)
        with self.assertRaisesRegex(ValueError, "sentence and paragraph tiers"):
            make_model(word_tier=W, tier_bptt=2, **PARALLEL)


class SwapTest(unittest.TestCase):
    def setUp(self) -> None:
        from imm.documents import cut_segments, piece_windows
        from test_documents import STARTS
        self.ids, self.rows = corpus(n=400)
        self.segments = cut_segments(ragged_segments(399), STARTS)
        self.pieces = piece_windows(self.segments, STARTS)

    def score(self, model, swap):
        nll, _, _, _ = runner.evaluate(
            model, self.ids, self.segments, 4, carry_hidden=True, feedback_state="gru", token_feedback="none",
            chunk=5, pieces=self.pieces, swap_block=swap, **self.rows,
        )
        return nll

    def test_the_swap_moves_only_its_tier(self) -> None:
        model = make_model(layers=2, documents=True, tier_bptt=2, **STACK)
        model.eval()
        base = self.score(model, None)
        for tier, block in (("sentence", 2), ("paragraph", 3)):
            self.assertEqual(runner.swap_block(model, tier), block)
            self.assertNotAlmostEqual(self.score(model, block), base, places=4, msg=tier)
            # with the tier's read-out off, its state reaches nothing (no top-down): no change
            module = getattr(model, f"{tier}_tier")
            saved = module.gate.detach().clone()
            with torch.no_grad():
                module.gate.zero_()
                if tier == "sentence":
                    model.paragraph_tier.gate.zero_()  # the paragraph tier reads the sentence states
            off = self.score(model, None)
            self.assertAlmostEqual(self.score(model, block), off, places=6, msg=tier)
            with torch.no_grad():
                module.gate.copy_(saved)
            model = make_model(layers=2, documents=True, tier_bptt=2, **STACK)
            model.eval()


class StackDocumentsTest(unittest.TestCase):
    def test_a_reset_stream_reads_its_piece_as_a_fresh_stream(self) -> None:
        from test_documents import STARTS, call, losses_by_position
        from imm.documents import aligned_streams, cut_segments
        ids, rows = corpus(n=400)
        segments = cut_segments(ragged_segments(399), STARTS)
        model = make_model(layers=2, documents=True, tier_bptt=2, tier_centroid=True, **STACK)
        with torch.no_grad():
            model.start_state.normal_(0.0, 1.0)
        model.eval()
        streams, resets, _ = aligned_streams(segments, STARTS, 2, 4)
        b, u = next((b, u) for b in range(2) for u in range(1, resets.shape[1]) if resets[b, u])

        def run_streams():
            memory, hidden = None, None
            for v in range(resets.shape[1]):
                if resets[:, v].any():
                    hidden, memory = model.reset_streams(hidden, memory, resets[:, v])
                part = [segs[4 * v : 4 * v + 4] for segs in streams]
                _, _, memory, hidden, _ = call(model, ids, rows, part, memory, hidden, collect_stats=True)

        got = losses_by_position(model, run_streams)
        end = next((v for v in range(u + 1, resets.shape[1]) if resets[b, v]), resets.shape[1])
        piece = streams[b][4 * u : 4 * end]

        def run_alone():
            memory, hidden = None, model.start_hidden(1, torch.device("cpu"))
            for k in range(0, len(piece), 4):
                _, _, memory, hidden, _ = call(model, ids, rows, [piece[k : k + 4]], memory, hidden, collect_stats=True)

        want = losses_by_position(model, run_alone)
        for p, loss in want.items():
            self.assertAlmostEqual(got[p], loss, places=5, msg=f"position {p}")


if __name__ == "__main__":
    unittest.main()
