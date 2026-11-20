"""--max-update-chars: an update whose streams x longest span passes the cap
runs as groups of streams (stream_groups), each group's forward and
backward in turn (grouped_update). The streams don't interact, so this
gives the whole update's loss, gradients, memory and carried state, update
after update; the groups are contiguous, cover every stream once, and each
fits the cap when it can."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_train_streams import corpus, grads, make_model, ragged_segments, runner  # noqa: E402

NO_PRIOR = dict(prior_input="none", prior_residual_gate="none", prior_residual_state_features=[])
PARALLEL = dict(carry_bound="gru", memory_state="gru")


class StreamGroupsTest(unittest.TestCase):
    def test_fits_or_off_is_one_group(self) -> None:
        spans = [(0, 10), (10, 30), (30, 35)]
        self.assertEqual(runner.stream_groups(spans, 0), [slice(0, 3)])
        self.assertEqual(runner.stream_groups(spans, 60), [slice(0, 3)])

    def test_groups_cover_the_streams_and_fit(self) -> None:
        g = torch.Generator().manual_seed(0)
        lengths = torch.randint(1, 50, (64,), generator=g).tolist()
        lengths[17] = 120
        starts = [sum(lengths[:i]) for i in range(64)]
        spans = [(s, s + n) for s, n in zip(starts, lengths)]
        for cap in (64 * 120 - 1, 4000, 1500, 500):
            groups = runner.stream_groups(spans, cap)
            self.assertGreater(len(groups), 1)
            self.assertEqual(groups[0].start, 0)
            self.assertEqual(groups[-1].stop, 64)
            for a, b in zip(groups, groups[1:]):
                self.assertEqual(a.stop, b.start)
            for grp in groups:
                self.assertLessEqual((grp.stop - grp.start) * max(lengths[grp]), cap)
            # the fewest equal groups: one fewer doesn't fit
            bounds = np.linspace(0, 64, len(groups)).round().astype(int)
            fewer = [slice(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:])]
            self.assertTrue(any((grp.stop - grp.start) * max(lengths[grp]) > cap for grp in fewer))

    def test_a_stream_past_the_cap_runs_alone(self) -> None:
        spans = [(0, 10), (10, 200), (200, 210)]
        self.assertEqual(runner.stream_groups(spans, 100), [slice(0, 1), slice(1, 2), slice(2, 3)])


class GroupedUpdateTest(unittest.TestCase):
    def model(self, **extra):
        model = make_model(layers=2, word_tier=5, char_memory=6, **NO_PRIOR, **PARALLEL, **extra)
        with torch.no_grad():  # the gates start at 0: open them, so the blocks count
            for block in model.gated_xattn_blocks:
                block.gate_attn.fill_(0.5)
                block.gate_mlp.fill_(0.5)
        return model

    def updates(self, grouped: bool, streams=5, updates=3, per_update=6, resets=None, **extra):
        ids, _ = corpus(n=600)
        stream_segments = runner.split_streams(ragged_segments(len(ids) - 1, longest=7), streams)
        model = self.model(**extra)
        memory = hidden = None
        out = []
        for u in range(updates):
            batch = [segs[u * per_update : (u + 1) * per_update] for segs in stream_segments]
            model.zero_grad(set_to_none=True)
            if resets is not None:  # --documents, as the training loop does it: the start state, a parameter
                hidden, memory = model.reset_streams(hidden, memory, np.array(resets[u]))
            if grouped:
                spans = [(s[0][0], s[-1][1]) for s in batch]
                groups = runner.stream_groups(spans, streams * max(e - s for s, e in spans) // 3)
                self.assertGreater(len(groups), 1)
                loss, tokens, memory, hidden = runner.grouped_update(
                    model, ids, batch, groups, memory, hidden, 4, carry_hidden=True
                )
            else:
                loss, tokens, memory, hidden, _ = model.segment_loss_streams(
                    ids, batch, memory, 4, carried_hidden=hidden, carry_hidden=True
                )
                (loss / tokens).backward()
                memory = tuple(m.detach() for m in memory)
                hidden = hidden.detach()
            out.append((float(loss.detach()), tokens, grads(model), memory, hidden))
        return out

    def check(self, **extra) -> None:
        whole, split = self.updates(False, **extra), self.updates(True, **extra)
        if extra.get("resets"):
            self.assertIn("start_state", whole[0][2])
        for (l1, t1, g1, m1, h1), (l2, t2, g2, m2, h2) in zip(whole, split):
            self.assertEqual(t1, t2)
            self.assertAlmostEqual(l1, l2, places=4)
            self.assertEqual(g1.keys(), g2.keys())
            for name in g1:
                torch.testing.assert_close(g1[name], g2[name], rtol=1e-4, atol=1e-6, msg=name)
            self.assertEqual(len(m1), len(m2))
            for a, b in zip(m1, m2):
                self.assertEqual(a.shape, b.shape)
                torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)
            torch.testing.assert_close(h1, h2, rtol=1e-4, atol=1e-6)

    def test_grouped_equals_whole(self) -> None:
        self.check()

    def test_grouped_equals_whole_with_documents(self) -> None:
        self.check(documents=True)

    def test_grouped_equals_whole_across_piece_starts(self) -> None:
        # Streams that begin a piece take the learned start state, one
        # graph shared by every group: it must get its gradient once.
        resets = [[True] * 5, [True, False, True, False, False], [False, True, False, False, True]]
        self.check(documents=True, resets=resets)

    def test_grouped_under_bf16_is_finite_and_float32(self) -> None:
        ids, _ = corpus(n=600)
        stream_segments = runner.split_streams(ragged_segments(len(ids) - 1, longest=7), 4)
        model = self.model()
        model.autocast_dtype = torch.bfloat16
        batch = [segs[:6] for segs in stream_segments]
        loss, tokens, memory, hidden = runner.grouped_update(
            model, ids, batch, [slice(0, 2), slice(2, 4)], None, None, 4, carry_hidden=True
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(hidden.dtype, torch.float32)
        self.assertEqual(hidden.shape[1], 4)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_cli_refuses_prior_rows(self) -> None:
        source = Path(runner.__file__).read_text()
        self.assertIn('raise ValueError("--max-update-chars splits batched updates without prior rows', source)


if __name__ == "__main__":
    unittest.main()
