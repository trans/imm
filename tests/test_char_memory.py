"""--char-memory N: records at character resolution beside the word records,
in the same attention blocks. Character u's record is written from the GRU's
state before u and u's embedding; character t sees those of u <= t within
its last N characters, in its piece; the last N are carried from update to
update. A character never sees its future; each sees exactly its window;
an update split in two gives what the whole update gives; a piece forgets
the one before; the other paths refuse it."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_train_streams import H, V, corpus, grads, make_model, ragged_segments, runner  # noqa: E402

CPU = torch.device("cpu")
PARALLEL = dict(carry_bound="gru", memory_state="gru")
N = 6  # the character window, shorter than an update so the carry matters


def run(model, ids, rows, stream_segments, memory, hidden, max_memory=4):
    spans = [(segs[0][0], segs[-1][1]) for segs in stream_segments]
    given = {k: runner.prior_rows_for_spans(v, spans, CPU) for k, v in rows.items()}
    return model.segment_loss_streams(
        ids, stream_segments, memory, max_memory, carried_hidden=hidden, carry_hidden=True, **given
    )


def states_of(model, ids, rows, part, memory=None, hidden=None):
    sink = []
    model.state_sink = sink
    try:
        with torch.no_grad():
            out = run(model, ids, rows, part, memory, hidden)
    finally:
        model.state_sink = None
    return torch.cat([s for s, _ in sink]).float(), out


class CharMemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ids, self.rows = corpus(n=160)
        self.segs = runner.split_streams(ragged_segments(len(self.ids) - 1), 3)

    def test_shared_parameters_start_the_same_and_the_records_matter(self) -> None:
        base = make_model(layers=2, word_tier=5, **PARALLEL)
        char = make_model(layers=2, word_tier=5, char_memory=N, **PARALLEL)
        shared = dict(base.named_parameters())
        names = [name for name, _ in char.named_parameters()]
        self.assertTrue(any(name.startswith("char_writer.") for name in names))
        for name, p in char.named_parameters():
            if name in shared:
                self.assertTrue(torch.equal(p, shared[name]), name)
            else:
                self.assertTrue(name.startswith("char_writer."), name)
        part = [s[:8] for s in self.segs]
        base.eval(), char.eval()
        with torch.no_grad():
            a = float(run(base, self.ids, self.rows, part, None, None)[0])
            b = float(run(char, self.ids, self.rows, part, None, None)[0])
        self.assertNotAlmostEqual(a, b, places=4)

    def test_a_character_never_sees_its_future(self) -> None:
        # Change stream 0's input at one character, in the first update and
        # in the second (the records carried): every state before it is
        # unchanged, its own is not.
        model = make_model(layers=2, word_tier=5, char_memory=N, **PARALLEL)
        model.eval()
        first, second = [s[:6] for s in self.segs], [s[6:12] for s in self.segs]
        start = first[0][0][0]
        for update, k in ((0, 2), (0, 5), (1, 0), (1, 3)):
            seg_start, seg_end = (first if update == 0 else second)[0][k]
            for q in (seg_start, seg_end - 1):
                with self.subTest(update=update, segment=k, at=q):
                    got = []
                    for flip in (False, True):
                        ids = self.ids.clone()
                        if flip:
                            ids[q] = (ids[q] + 1) % V
                        s0, (_, _, memory, hidden, _) = states_of(model, ids, self.rows, first)
                        if update == 0:
                            got.append(s0[: first[0][-1][1] - start])
                        else:
                            s1, _ = states_of(model, ids, self.rows, second, memory, hidden)
                            got.append(s1[: second[0][-1][1] - second[0][0][0]])
                    before, after = got
                    n = q - (start if update == 0 else second[0][0][0])
                    self.assertTrue(torch.equal(before[:n], after[:n]))
                    self.assertFalse(torch.equal(before[n], after[n]))

    def test_each_character_sees_exactly_its_window(self) -> None:
        # The blocks get every record with a mask: for each real query, the
        # character records it sees are those at positions p - N + 1 .. p of
        # its own stream, in the second update too (carried ones included).
        model = make_model(layers=1, char_memory=N, **PARALLEL)
        model.eval()
        block = model.gated_xattn_blocks[0]
        seen = []
        original = block.forward

        def capture(hidden, memory, **kw):
            seen.append((kw["query_positions"].clone(), kw["memory_positions"].clone(), kw["memory_mask"].clone()))
            return original(hidden, memory, **kw)

        block.forward = capture
        memory = hidden = None
        for u in range(2):
            part = [s[u * 5 : (u + 1) * 5] for s in self.segs]
            word_held = 0 if memory is None else memory[0].shape[1]
            with torch.no_grad():
                _, _, memory, hidden, _ = run(model, self.ids, self.rows, part, memory, hidden)
            q, mp, mask = seen[-1]
            words = word_held + len(part[0])  # word records come first: held, then this update's
            char_pos, char_mask = mp[:, words:], mask[:, :, words:]
            for b, segs in enumerate(part):
                for t, p in enumerate(range(segs[0][0], segs[-1][1])):
                    self.assertEqual(int(q[b, t]), p)
                    got = sorted(int(x) for x in char_pos[b][char_mask[b, t]])
                    first = max(p - N + 1, self.segs[b][0][0])
                    self.assertEqual(got, list(range(first, p + 1)), (u, b, p))
            # the carried records are each stream's last N positions, in order
            for b, segs in enumerate(part):
                end = segs[-1][1]
                want = [p if p >= self.segs[b][0][0] else -1 for p in range(end - N, end)]
                self.assertEqual(memory[3][b].tolist(), want)
        block.forward = original

    def test_a_record_is_the_state_before_and_the_character(self) -> None:
        # Character u's record is char_writer([GRU state before u, u's
        # embedding]): the carried ones, from the GRU run by hand.
        model = make_model(layers=1, char_memory=N, prior_input="none", **PARALLEL)
        model.eval()
        part = [s[:5] for s in self.segs]
        with torch.no_grad():
            _, _, memory, _, _ = run(model, self.ids, self.rows, part, None, None)
            for b, segs in enumerate(part):
                lo, hi = segs[0][0], segs[-1][1]
                emb = model.embed(self.ids[lo:hi])
                states, _ = model.rnn(emb[None], torch.zeros(1, 1, H))
                before = torch.cat([torch.zeros(1, H), states[0, :-1]])
                want = model.char_writer(torch.cat([before, emb], dim=-1))[-N:]
                self.assertEqual(memory[3][b].tolist(), list(range(hi - N, hi)))
                torch.testing.assert_close(memory[2][b], want, rtol=1e-5, atol=1e-6)

    def test_split_update_is_the_whole_update(self) -> None:
        # Two updates of 5 segments, records and state carried, are one of
        # 10: the same total loss, word records and character records.
        model = make_model(layers=2, word_tier=5, char_memory=N, **PARALLEL)
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
                memory, hidden = tuple(m.detach() for m in memory), hidden.detach()
            got.append((total, grads(model), memory, hidden))
        (la, ga, ma, ha), (lb, gb, mb, hb) = got
        self.assertAlmostEqual(la, lb, places=4)
        self.assertEqual(len(ma), 4)
        for x, y in zip(ma, mb):
            if x.dtype.is_floating_point:
                torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-5)
            else:
                self.assertTrue(torch.equal(x, y))
        torch.testing.assert_close(ha, hb, rtol=1e-5, atol=1e-5)
        for name in ("char_writer.0.weight", "char_writer.1.weight"):
            self.assertGreater(float(ga[name].abs().sum()), 0.0, name)

    def test_a_piece_forgets_the_one_before(self) -> None:
        model = make_model(layers=1, char_memory=N, documents=True, **PARALLEL)
        model.eval()
        part = [s[:5] for s in self.segs]
        with torch.no_grad():
            _, _, memory, hidden, _ = run(model, self.ids, self.rows, part, None, None)
        self.assertTrue((memory[3] >= 0).all())
        hidden, memory = model.reset_streams(hidden, memory, np.array([True, False, True]))
        self.assertTrue((memory[3][[0, 2]] == -1).all())
        self.assertTrue((memory[3][1] >= 0).all())
        self.assertTrue((memory[1][[0, 2]] == -1).all())
        # and a stream that forgot gives what it gives starting fresh
        nxt = [s[5:10] for s in self.segs]
        with torch.no_grad():
            s_carried, _ = states_of(model, self.ids, self.rows, nxt, memory, hidden)
            fresh_hidden, _ = model.reset_streams(None, None, np.array([True, True, True]))
            s_fresh, _ = states_of(model, self.ids, self.rows, nxt, None, fresh_hidden)
        n0 = nxt[0][-1][1] - nxt[0][0][0]
        torch.testing.assert_close(s_carried[:n0], s_fresh[:n0], rtol=1e-5, atol=1e-6)

    def test_other_paths_refuse_it(self) -> None:
        model = make_model(layers=1, char_memory=N)  # carry_bound none: not the parallel path
        part = [s[:3] for s in self.segs]
        with self.assertRaises(ValueError):
            run(model, self.ids, self.rows, part, None, None)
        with self.assertRaises(ValueError):
            make_model(layers=1, char_memory=-1, **PARALLEL)


if __name__ == "__main__":
    unittest.main()
