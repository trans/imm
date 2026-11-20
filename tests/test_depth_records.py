"""--depth-records: block i > 1 reads records written, by the same writers,
from block i-1's output instead of the GRU's state. A word record is the
writer's map of the state at the word's last character; a character
record, of the state before the character and its embedding (the carried
one before an update's first). Block 1 reads the GRU's, as without it.
No new parameters; one block is the model without it; a character never
sees its future; an update split in two is the whole update; a split
update (--max-update-chars) gives the whole one's gradients; a piece
forgets the one before; the other paths refuse it."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_char_memory import PARALLEL, run, states_of  # noqa: E402
from test_train_streams import V, corpus, grads, make_model, ragged_segments, runner  # noqa: E402

N = 6
TIERS = dict(word_tier=5, tier_own_step=True)


def model_of(depth, layers=2, **extra):
    model = make_model(layers=layers, char_memory=N, depth_records=depth, **PARALLEL, **extra)
    with torch.no_grad():  # open the gates, so every block's attention counts
        for block in model.gated_xattn_blocks:
            block.gate_attn.fill_(0.7)
            block.gate_mlp.fill_(0.5)
    return model


class DepthRecordsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ids, self.rows = corpus(n=160)
        self.segs = runner.split_streams(ragged_segments(len(self.ids) - 1), 3)

    def test_same_parameters_and_the_records_matter(self) -> None:
        for layers in (1, 2, 3):
            with self.subTest(layers=layers):
                base, depth = model_of(False, layers, **TIERS), model_of(True, layers, **TIERS)
                a, b = dict(base.named_parameters()), dict(depth.named_parameters())
                self.assertEqual(a.keys(), b.keys())
                for name in a:
                    self.assertTrue(torch.equal(a[name], b[name]), name)
                part = [s[:8] for s in self.segs]
                base.eval(), depth.eval()
                sa, _ = states_of(base, self.ids, self.rows, part)
                sb, _ = states_of(depth, self.ids, self.rows, part)
                if layers == 1:  # nothing to deepen: the same model
                    torch.testing.assert_close(sa, sb, rtol=0, atol=0)
                else:
                    self.assertGreater(float((sa - sb).abs().max()), 1e-3)

    def test_block_two_reads_records_from_block_one(self) -> None:
        # Capture what each block is given: block 2's records are the writers'
        # maps of block 1's output (word: at the word's last character;
        # character: the state before it, then its embedding), in block 1's
        # layout; block 1's are the GRU's, as without --depth-records.
        model = model_of(True, prior_input="none")
        model.eval()
        seen = []
        originals = [block.forward for block in model.gated_xattn_blocks]
        for i, block in enumerate(model.gated_xattn_blocks):

            def capture(hidden, memory, _i=i, _f=originals[i], **kw):
                out = _f(hidden, memory, **kw)
                seen.append((_i, hidden.clone(), memory.clone(), out.clone()))
                return out

            block.forward = capture
        part = [s[:5] for s in self.segs]
        try:
            with torch.no_grad():
                _, _, memory, _, _ = run(model, self.ids, self.rows, part, None, None, max_memory=32)
        finally:
            for block, f in zip(model.gated_xattn_blocks, originals):
                block.forward = f
        (_, in1, mem1, out1), (_, in2, mem2, _) = seen
        self.assertEqual(mem1.shape, mem2.shape)
        K = len(part[0])
        with torch.no_grad():
            for b, segs in enumerate(part):
                lo = segs[0][0]
                last = torch.tensor([e - 1 - lo for _, e in segs])
                word_want = model.write_memory(out1[b, last])
                torch.testing.assert_close(mem2[b, :K], word_want, rtol=1e-5, atol=1e-6)
                n = segs[-1][1] - lo
                emb = model.embed(self.ids[lo : lo + n])
                before = torch.cat([torch.zeros(1, out1.shape[-1]), out1[b, : n - 1]])  # no carry: zeros
                char_want = model.char_writer(torch.cat([before, emb], dim=-1))
                torch.testing.assert_close(mem2[b, K : K + n], char_want, rtol=1e-5, atol=1e-6)
                # block 1's word records are the GRU's, written as without it
                self.assertFalse(torch.allclose(mem1[b, :K], mem2[b, :K]))
        # the carried records are each block's side by side
        H = out1.shape[-1]
        self.assertEqual(memory[0].shape[-1], 2 * H)
        self.assertEqual(memory[2].shape[-1], 2 * H)
        torch.testing.assert_close(memory[0][:, :, H:], mem2[:, :K], rtol=1e-5, atol=1e-6)

    def test_a_character_never_sees_its_future(self) -> None:
        model = model_of(True, **TIERS)
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

    def test_split_update_is_the_whole_update(self) -> None:
        # Updates of 4, 3 and 3 segments, records and state carried, are one
        # of 10: the same loss, gradients, records and carried state (the
        # carried state before each block's next first character included).
        for layers in (2, 3):
            with self.subTest(layers=layers):
                model = model_of(True, layers, **TIERS)
                model.train()
                got = []
                for pieces in ((slice(0, 10),), (slice(0, 4), slice(4, 7), slice(7, 10))):
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
                self.assertEqual(ha.shape[-1], model.carried_width())
                self.assertEqual(model.depth_carry_width(), (layers - 1) * model.hidden_size)
                for x, y in zip(ma, mb):
                    if x.dtype.is_floating_point:
                        torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-5)
                    else:
                        self.assertTrue(torch.equal(x, y))
                torch.testing.assert_close(ha, hb, rtol=1e-5, atol=1e-5)
                # the writers learn from every block
                for name in ("char_writer.0.weight", "memory_writer.1.weight"):
                    self.assertGreater(float(ga[name].abs().sum()), 0.0, name)

    def test_grouped_update_is_the_whole_update(self) -> None:
        # --max-update-chars: the streams in groups, each group's backward in
        # turn, with documents and piece starts: the whole update's loss,
        # gradients, records and carried state.
        ids, _ = corpus(n=600)
        segs = runner.split_streams(ragged_segments(len(ids) - 1, longest=7), 5)
        resets = [[True] * 5, [True, False, True, False, False]]
        out = []
        for grouped in (False, True):
            model = model_of(True, documents=True, prior_input="none", prior_residual_gate="none",
                             prior_residual_state_features=[], **TIERS)
            memory = hidden = None
            got = []
            for u in range(2):
                batch = [s[u * 4 : (u + 1) * 4] for s in segs]
                model.zero_grad(set_to_none=True)
                hidden, memory = model.reset_streams(hidden, memory, np.array(resets[u]))
                if grouped:
                    loss, tokens, memory, hidden = runner.grouped_update(
                        model, ids, batch, [slice(0, 2), slice(2, 5)], memory, hidden, 4, carry_hidden=True
                    )
                else:
                    loss, tokens, memory, hidden, _ = model.segment_loss_streams(
                        ids, batch, memory, 4, carried_hidden=hidden, carry_hidden=True
                    )
                    (loss / tokens).backward()
                    memory, hidden = tuple(m.detach() for m in memory), hidden.detach()
                got.append((float(loss.detach()), grads(model), memory, hidden))
            out.append(got)
        for (l1, g1, m1, h1), (l2, g2, m2, h2) in zip(*out):
            self.assertAlmostEqual(l1, l2, places=4)
            self.assertIn("start_state", g1)
            for name in g1:
                torch.testing.assert_close(g1[name], g2[name], rtol=1e-4, atol=1e-6, msg=name)
            for a, b in zip(m1, m2):
                torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)
            torch.testing.assert_close(h1, h2, rtol=1e-4, atol=1e-6)

    def test_a_piece_forgets_the_one_before(self) -> None:
        model = model_of(True, documents=True)
        model.eval()
        part = [s[:5] for s in self.segs]
        with torch.no_grad():
            _, _, memory, hidden, _ = run(model, self.ids, self.rows, part, None, None)
            hidden, memory = model.reset_streams(hidden, memory, np.array([True, False, True]))
            # the carried state before block 2's next first character starts at zeros
            depth = hidden[0, :, -model.depth_carry_width() :]
            self.assertTrue(torch.equal(depth[[0, 2]], torch.zeros_like(depth[[0, 2]])))
            self.assertFalse(torch.equal(depth[1], torch.zeros_like(depth[1])))
            nxt = [s[5:10] for s in self.segs]
            s_carried, _ = states_of(model, self.ids, self.rows, nxt, memory, hidden)
            fresh_hidden, _ = model.reset_streams(None, None, np.array([True, True, True]))
            s_fresh, _ = states_of(model, self.ids, self.rows, nxt, None, fresh_hidden)
        n0 = nxt[0][-1][1] - nxt[0][0][0]
        torch.testing.assert_close(s_carried[:n0], s_fresh[:n0], rtol=1e-5, atol=1e-6)

    def test_refused_elsewhere(self) -> None:
        with self.assertRaisesRegex(ValueError, "--depth-records"):
            make_model(layers=2, depth_records=True, memory_attention=False, **PARALLEL)
        with self.assertRaisesRegex(ValueError, "--depth-records"):
            make_model(layers=2, depth_records=True, mixing="none", **PARALLEL)
        model = make_model(layers=2, depth_records=True)  # not the parallel path
        model.eval()
        with self.assertRaisesRegex(ValueError, "--depth-records"):
            with torch.no_grad():
                run(model, self.ids, self.rows, [s[:4] for s in self.segs], None, None)

    def test_cli_flag(self) -> None:
        source = Path(runner.__file__).read_text()
        self.assertIn('"--depth-records"', source)
        self.assertIn("depth_records=args.depth_records", source)


if __name__ == "__main__":
    unittest.main()
