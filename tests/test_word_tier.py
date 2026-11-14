"""--word-tier: a second GRU that steps once a segment, over the character
GRU's state at each segment's end, its state carried beside the character
GRU's. At gate 0 the model is the one without it; an update split in two
gives what the whole update gives (the word state is carried); a character
never sees its own word's state; the other paths refuse it."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_train_streams import H, V, corpus, grads, make_model, ragged_segments, runner  # noqa: E402

CPU = torch.device("cpu")
PARALLEL = dict(carry_bound="gru", memory_state="gru")
W = 5  # the word tier's width, unlike H so a mix-up of the two shows


def run(model, ids, rows, stream_segments, memory, hidden, max_memory=4):
    spans = [(segs[0][0], segs[-1][1]) for segs in stream_segments]
    given = {k: runner.prior_rows_for_spans(v, spans, CPU) for k, v in rows.items()}
    return model.segment_loss_streams(
        ids, stream_segments, memory, max_memory, carried_hidden=hidden, carry_hidden=True, **given
    )


class WordTierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ids, self.rows = corpus(n=160)
        self.segs = runner.split_streams(ragged_segments(len(self.ids) - 1), 3)

    def test_gate_zero_is_the_model_without_it(self) -> None:
        # Train mode, three updates with the state carried and detached: the
        # same losses, gradients, records and character state; the tier's
        # own gradients are zero but its gate's isn't.
        for layers in (1, 2):
            with self.subTest(layers=layers):
                base = make_model(layers=layers, **PARALLEL)
                tier = make_model(layers=layers, word_tier=W, **PARALLEL)
                with torch.no_grad():
                    tier.word_tier.gate.zero_()
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
                self.assertEqual(hb.shape, (1, 3, H + W))
                torch.testing.assert_close(ha, hb[..., :H], rtol=1e-6, atol=1e-6)
                for name in ("word_tier.rnn.weight_ih_l0", "word_tier.proj.weight"):
                    self.assertEqual(float(gb[name].abs().sum()), 0.0, name)
                self.assertGreater(float(gb["word_tier.gate"].abs().sum()), 0.0)

    def test_split_update_is_the_whole_update(self) -> None:
        # Records never slide out (max_memory 32), so two updates of 5
        # segments, the state carried, are one update of 10: the same total
        # loss, gradients, records and carried state, word part included.
        model = make_model(layers=2, word_tier=W, **PARALLEL)
        model.train()
        self.assertNotEqual(float(model.word_tier.gate.detach().abs().sum()), 0.0)
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
        # Split, the carried state is a constant to the second half, as in
        # training, so the gradients differ: compare the forward results,
        # and check the tier gets a gradient at all.
        torch.testing.assert_close(ma[0], mb[0], rtol=1e-5, atol=1e-5)
        self.assertTrue(torch.equal(ma[1], mb[1]))
        torch.testing.assert_close(ha, hb, rtol=1e-5, atol=1e-5)
        for name in ("word_tier.rnn.weight_ih_l0", "word_tier.rnn.weight_hh_l0", "word_tier.proj.weight", "word_tier.gate"):
            self.assertGreater(float(ga[name].abs().sum()), 0.0, name)

    def test_a_character_never_sees_its_own_word(self) -> None:
        # Change stream 0's input at one character; every state before it,
        # in the same segment too, is unchanged. Were a segment's characters
        # given their own word's state (read after its last character),
        # those earlier in the segment would change.
        model = make_model(layers=2, word_tier=W, **PARALLEL)
        model.eval()
        part = [s[:8] for s in self.segs]
        start = part[0][0][0]
        for k, at in ((3, "last"), (3, "first"), (5, "last")):
            seg_start, seg_end = part[0][k]
            q = seg_end - 1 if at == "last" else seg_start
            with self.subTest(segment=k, at=at):
                states = []
                for flip in (False, True):
                    ids = self.ids.clone()
                    if flip:
                        ids[q] = (ids[q] + 1) % V
                    sink = []
                    model.state_sink = sink
                    try:
                        with torch.no_grad():
                            run(model, ids, self.rows, part, None, None)
                    finally:
                        model.state_sink = None
                    states.append(torch.cat([s for s, _ in sink]).float())
                before, after = states
                n = q - start  # stream 0's positions come first, in order
                self.assertTrue(torch.equal(before[:n], after[:n]))
                self.assertFalse(torch.equal(before[n], after[n]))
                # and the next segment's characters do see the changed word
                next_start = part[0][k + 1][0] - start
                self.assertFalse(torch.equal(before[next_start], after[next_start]))

    def test_word_state_reaches_the_next_segment(self) -> None:
        # The tier changes the scores: its gate at a random value against 0.
        model = make_model(layers=1, word_tier=W, **PARALLEL)
        model.eval()
        part = [s[:6] for s in self.segs]
        with torch.no_grad():
            on = float(run(model, self.ids, self.rows, part, None, None)[0])
            gate = model.word_tier.gate.clone()
            model.word_tier.gate.zero_()
            off = float(run(model, self.ids, self.rows, part, None, None)[0])
            model.word_tier.gate.copy_(gate)
        self.assertNotAlmostEqual(on, off, places=4)

    def test_other_paths_refuse_it(self) -> None:
        loop = make_model(word_tier=W)  # memory_state terminal: the per-segment loop
        self.assertFalse(loop.parallel_memory(True))
        part = [s[:5] for s in self.segs]
        with self.assertRaisesRegex(ValueError, "parallel memory path"):
            run(loop, self.ids, self.rows, part, None, None)
        single = make_model(word_tier=W, **PARALLEL)
        with self.assertRaisesRegex(ValueError, "batched parallel path only"):
            single.segment_loss(self.ids, self.segs[0][:5], [], 4, carry_hidden=True)
        with self.assertRaisesRegex(ValueError, "word_tier must be"):
            make_model(word_tier=-1)


if __name__ == "__main__":
    unittest.main()
