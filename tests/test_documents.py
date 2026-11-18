"""--documents: each piece read from its start. The pieces' segments are cut
so none spans two; training streams put every piece start at an update's
start; a stream reset there reads the piece exactly as a fresh stream would
(its carried state the start state, the records before it gone); the
held-out score, one stream per piece, is the sum of each piece read alone."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_phrase_tier import PARALLEL, W  # noqa: E402
from test_train_streams import corpus, make_model, ragged_segments, runner  # noqa: E402

from imm.documents import aligned_streams, cut_segments, first_positions, piece_starts, piece_windows, text_prefix  # noqa: E402

CPU = torch.device("cpu")
STARTS = np.array([0, 97, 230, 301])  # four pieces of a 400-character text


def call(model, ids, rows, stream_segments, memory, hidden, **kw):
    spans = [(segs[0][0], segs[-1][1]) for segs in stream_segments]
    return model.segment_loss_streams(
        ids, stream_segments, memory, 4, carried_hidden=hidden, carry_hidden=True,
        **{k: runner.prior_rows_for_spans(v, spans, CPU) for k, v in rows.items()}, **kw,
    )


def losses_by_position(model, fn):
    """{position: loss} of what fn() scores on the stats path."""
    sink = []
    model.loss_sink = sink
    try:
        with torch.no_grad():
            fn()
    finally:
        model.loss_sink = None
    positions = torch.cat([p for p, _ in sink]).tolist()
    losses = torch.cat([x for _, x in sink]).tolist()
    return dict(zip(positions, losses))


def documents_model(**kw):
    model = make_model(layers=2, documents=True, **kw, **PARALLEL)
    with torch.no_grad():
        model.start_state.normal_(0.0, 1.0)  # a start state that isn't zeros, so using it shows
    model.eval()
    return model


class HelpersTest(unittest.TestCase):
    def setUp(self) -> None:
        self.segments = cut_segments(ragged_segments(399), STARTS)

    def test_cut_segments(self) -> None:
        segs = self.segments
        self.assertEqual(segs[0][0], 0)
        self.assertEqual(segs[-1][1], 399)
        self.assertTrue(all(a[1] == b[0] and a[0] < a[1] for a, b in zip(segs, segs[1:])))
        seg_starts = {a for a, _ in segs}
        for s in STARTS[1:]:
            self.assertIn(s - 1, seg_starts)  # the stray position: its own segment
            self.assertIn(s, seg_starts)  # the piece's own text starts one
            self.assertIn((s - 1, s), segs)
        # nothing else is cut: every new segment start is one of the cuts
        before = {a for a, _ in ragged_segments(399)}
        self.assertLessEqual(seg_starts - before, {int(p) for s in STARTS[1:] for p in (s - 1, s)})

    def test_piece_windows(self) -> None:
        windows = piece_windows(self.segments, STARTS)
        self.assertEqual(len(windows), 4)
        self.assertEqual(windows[0][0], 0)
        self.assertEqual(windows[-1][1], len(self.segments))
        for (lo, hi), first in zip(windows, first_positions(STARTS)):
            self.assertEqual(self.segments[lo][0], first)
        self.assertTrue(all(a[1] == b[0] for a, b in zip(windows, windows[1:])))
        raw = ragged_segments(399)
        a, _ = next(s for s in raw if s[1] - s[0] >= 3)
        with self.assertRaisesRegex(ValueError, "cut the segments"):
            piece_windows(raw, np.array([0, a + 2]))  # its first position, a + 1, is inside a segment

    def test_aligned_streams(self) -> None:
        streams, resets, dropped = aligned_streams(self.segments, STARTS, 2, 4)
        windows = piece_windows(self.segments, STARTS)
        firsts = {self.segments[lo] for lo, _ in windows}
        per = len(streams[0])
        self.assertEqual(per % 4, 0)
        self.assertEqual(len(streams[1]), per)
        self.assertEqual(resets.shape, (2, per // 4))
        for b, segs in enumerate(streams):
            for k, seg in enumerate(segs):
                if seg in firsts:
                    self.assertEqual(k % 4, 0, "a piece begins mid-update")
                    self.assertTrue(resets[b, k // 4])
            for u in range(per // 4):
                self.assertEqual(bool(resets[b, u]), segs[4 * u] in firsts)
            # within an update, segments are contiguous: one span of rows
            for u in range(per // 4):
                part = segs[4 * u : 4 * u + 4]
                self.assertTrue(all(a[1] == b_[0] for a, b_ in zip(part, part[1:])))
        self.assertTrue(resets[0, 0])
        used = sum(e - s for segs in streams for s, e in segs)
        self.assertEqual(used + dropped, 399)

    def test_text_prefix(self) -> None:
        segs, starts = text_prefix(self.segments, STARTS, 250)
        self.assertEqual(segs, self.segments[: len(segs)])  # a prefix of the segments
        self.assertLessEqual(segs[-1][1], 250)
        self.assertGreater(self.segments[len(segs)][1], 250)  # the next segment ends past it
        self.assertEqual(starts.tolist(), [0, 97, 230])  # 301 begins past it
        streams, resets, dropped = aligned_streams(segs, starts, 2, 4)
        used = sum(e - s for ss in streams for s, e in ss)
        self.assertEqual(used + dropped, segs[-1][1])
        self.assertTrue(all(e <= 250 for ss in streams for _, e in ss))
        whole, all_starts = text_prefix(self.segments, STARTS, 10**6)
        self.assertEqual(whole, self.segments)
        self.assertEqual(all_starts.tolist(), STARTS.tolist())
        with self.assertRaisesRegex(ValueError, "chars >= 1"):
            text_prefix(self.segments, STARTS, 0)
        if self.segments[0][1] > 1:
            with self.assertRaisesRegex(ValueError, "no segment"):
                text_prefix(self.segments, STARTS, self.segments[0][1] - 1)

    def test_piece_starts(self) -> None:
        m = {
            "mode": "pieces", "train_size": 100,
            "train_records": [{"start": 50, "end": 100}, {"start": 0, "end": 50}],
            "chunk_records": [{"start": 100, "end": 110}, {"start": 110, "end": 120}],
        }
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "manifest.json"
            path.write_text(json.dumps(m))
            train, held = piece_starts(path, 100, 20)
            self.assertEqual(train.tolist(), [0, 50])
            self.assertEqual(held.tolist(), [0, 10])
            train, held = piece_starts(path, 40, 15)  # shortened texts keep the pieces that start in them
            self.assertEqual(train.tolist(), [0])
            self.assertEqual(held.tolist(), [0, 10])
            m["train_records"][0]["start"] = 60
            path.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, "tile"):
                piece_starts(path, 100, 20)


class ResetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ids, self.rows = corpus(n=400)
        self.segments = cut_segments(ragged_segments(399), STARTS)

    def test_a_reset_stream_reads_its_piece_as_a_fresh_stream(self) -> None:
        for tiers in ({}, dict(word_tier=W, tier_own_step=True)):
            with self.subTest(**tiers):
                model = documents_model(**tiers)
                streams, resets, _ = aligned_streams(self.segments, STARTS, 2, 4)
                # some stream resets after its first update, with records and state to forget
                b, u = next((b, u) for b in range(2) for u in range(1, resets.shape[1]) if resets[b, u])

                def run_streams():
                    memory, hidden = None, None
                    for v in range(resets.shape[1]):
                        if resets[:, v].any():
                            hidden, memory = model.reset_streams(hidden, memory, resets[:, v])
                        part = [segs[4 * v : 4 * v + 4] for segs in streams]
                        _, _, memory, hidden, _ = call(model, self.ids, self.rows, part, memory, hidden, collect_stats=True)

                got = losses_by_position(model, run_streams)
                # the same piece, read alone from its start
                end = next((v for v in range(u + 1, resets.shape[1]) if resets[b, v]), resets.shape[1])
                piece = streams[b][4 * u : 4 * end]

                def run_alone():
                    memory, hidden = None, model.start_hidden(1, CPU)
                    for k in range(0, len(piece), 4):
                        _, _, memory, hidden, _ = call(
                            model, self.ids, self.rows, [piece[k : k + 4]], memory, hidden, collect_stats=True
                        )

                want = losses_by_position(model, run_alone)
                self.assertGreater(len(want), 20)
                for p, loss in want.items():
                    self.assertAlmostEqual(got[p], loss, places=5, msg=f"position {p}")

    def test_the_old_records_mattered(self) -> None:
        # Without the reset the same characters score differently: the check
        # above isn't vacuous.
        model = documents_model(word_tier=W, tier_own_step=True)
        streams, resets, _ = aligned_streams(self.segments, STARTS, 2, 4)

        def run(reset_on):
            def fn():
                memory, hidden = None, None
                for v in range(resets.shape[1]):
                    if reset_on and resets[:, v].any():
                        hidden, memory = model.reset_streams(hidden, memory, resets[:, v])
                    part = [segs[4 * v : 4 * v + 4] for segs in streams]
                    _, _, memory, hidden, _ = call(model, self.ids, self.rows, part, memory, hidden, collect_stats=True)
            return losses_by_position(model, fn)

        a, b = run(True), run(False)
        self.assertGreater(max(abs(a[p] - b[p]) for p in a), 1e-3)

    def test_start_state_gets_a_gradient(self) -> None:
        model = documents_model(word_tier=W, tier_own_step=True)
        model.train()
        streams, resets, _ = aligned_streams(self.segments, STARTS, 2, 4)
        hidden, memory = model.reset_streams(None, None, resets[:, 0])
        loss, _, _, _, _ = call(model, self.ids, self.rows, [segs[:4] for segs in streams], memory, hidden)
        loss.backward()
        g = model.start_state.grad
        self.assertIsNotNone(g)
        self.assertGreater(float(g[: model.hidden_size].abs().max()), 0.0)
        self.assertGreater(float(g[model.hidden_size :].abs().max()), 0.0)


class EvaluateDocumentsTest(unittest.TestCase):
    def test_one_stream_per_piece_is_each_piece_alone(self) -> None:
        ids, rows = corpus(n=400)
        segments = cut_segments(ragged_segments(399), STARTS)
        pieces = piece_windows(segments, STARTS)
        model = documents_model(word_tier=W, tier_own_step=True)
        nll, _, _, stats = runner.evaluate(
            model, ids, segments, 4, carry_hidden=True, feedback_state="gru", token_feedback="none",
            chunk=5, pieces=pieces, **rows,
        )
        self.assertEqual(stats["token_count"], 399)
        total = 0.0
        with torch.no_grad():
            for lo, hi in pieces:  # each piece in one call, from the start state
                loss, tokens, _, _, _ = call(model, ids, rows, [segments[lo:hi]], None, model.start_hidden(1, CPU))
                total += float(loss)
        self.assertAlmostEqual(nll, total / 399, places=5)


if __name__ == "__main__":
    unittest.main()
