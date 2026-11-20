"""--no-word-records: the attention blocks read the character records only.
It is the model whose word records no query sees (max_memory 0), update
after update, with and without --depth-records; no gradient reaches the
word-record writer or its LayerNorm, which drop out of the trained
parameters; an update split in two is the whole update; nothing is carried
for words; it works with --char-kv; without character records, or off the
parallel path, it is refused."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_char_memory import PARALLEL, run  # noqa: E402
from test_train_streams import corpus, grads, make_model, ragged_segments, runner  # noqa: E402

N = 6
TIERS = dict(word_tier=5, tier_own_step=True)
UNUSED = ("memory_writer.", "memory_interface_norm.")


def model_of(word_records, layers=2, **extra):
    model = make_model(layers=layers, char_memory=N, word_records=word_records, **PARALLEL, **extra)
    with torch.no_grad():  # open the gates, so every block's attention counts
        for block in model.gated_xattn_blocks:
            block.gate_attn.fill_(0.7)
            block.gate_mlp.fill_(0.5)
    return model


def states(model, ids, rows, part, memory, hidden, max_memory):
    sink = []
    model.state_sink = sink
    try:
        with torch.no_grad():
            out = run(model, ids, rows, part, memory, hidden, max_memory=max_memory)
    finally:
        model.state_sink = None
    return torch.cat([s for s, _ in sink]).float(), out


class NoWordRecordsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ids, self.rows = corpus(n=160)
        self.segs = runner.split_streams(ragged_segments(len(self.ids) - 1), 3)

    def test_same_as_word_records_unseen(self) -> None:
        for extra in ({}, TIERS, dict(depth_records=True, **TIERS), dict(depth_records=True, layers=3)):
            with self.subTest(**extra):
                extra = dict(extra)
                layers = extra.pop("layers", 2)
                base, off = model_of(True, layers, **extra), model_of(False, layers, **extra)
                for (name, a), (_, b) in zip(base.named_parameters(), off.named_parameters()):
                    self.assertTrue(torch.equal(a, b), name)
                base.eval(), off.eval()
                mem = {"seen": (None, None), "unseen": (None, None), "off": (None, None)}
                for u in range(3):
                    part = [s[u * 4 : (u + 1) * 4] for s in self.segs]
                    seen, out_seen = states(base, self.ids, self.rows, part, *mem["seen"], 32)
                    unseen, out_unseen = states(base, self.ids, self.rows, part, *mem["unseen"], 0)
                    got, out_off = states(off, self.ids, self.rows, part, *mem["off"], 32)
                    torch.testing.assert_close(got, unseen, rtol=1e-5, atol=1e-6)
                    self.assertGreater(float((seen - got).abs().max()), 1e-3)  # the word records mattered
                    mem = {k: (o[2], o[3]) for k, o in (("seen", out_seen), ("unseen", out_unseen), ("off", out_off))}
                    # nothing carried for words; the character records as with them unseen
                    records, positions = out_off[2][0], out_off[2][1]
                    self.assertEqual(records.shape[1], 0)
                    self.assertEqual(positions.shape[1], 0)
                    self.assertEqual(records.shape[-1], base.record_width())
                    for a, b in zip(out_off[2][2:], out_unseen[2][2:]):
                        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(out_off[3], out_unseen[3], rtol=1e-5, atol=1e-6)

    def test_no_gradient_to_the_word_writer(self) -> None:
        # Of the parameters the model with word records trains, the word
        # writer (and its LayerNorm, where there is one) go untrained.
        def untrained(model):
            model.train()
            model.zero_grad()
            memory, hidden = None, None
            for u in range(2):
                part = [s[u * 4 : (u + 1) * 4] for s in self.segs]
                loss, _, memory, hidden, _ = run(model, self.ids, self.rows, part, memory, hidden, max_memory=32)
                loss.backward()
                memory, hidden = tuple(m.detach() for m in memory), hidden.detach()
            return {n for n, p in model.named_parameters() if p.grad is None}

        for depth in (False, True):
            with self.subTest(depth=depth):
                base, off = model_of(True, depth_records=depth, **TIERS), model_of(False, depth_records=depth, **TIERS)
                lost = untrained(off) - untrained(base)
                self.assertTrue(any(n.startswith("memory_writer.") for n in lost), lost)
                self.assertTrue(all(n.startswith(UNUSED) for n in lost), lost)
                for name in ("char_writer.0.weight", "gated_xattn_blocks.0.k_proj.weight", "gated_xattn_blocks.1.v_proj.weight"):
                    g = dict(off.named_parameters())[name].grad
                    self.assertGreater(float(g.abs().sum()), 0.0, name)

    def test_split_update_is_the_whole_update(self) -> None:
        for extra in ({}, dict(depth_records=True), dict(char_kv=True), dict(depth_records=True, char_kv=True)):
            with self.subTest(**extra):
                model = model_of(False, **extra, **TIERS)
                if extra.get("char_kv"):  # make_model redraws all-zero parameters: copy again, as construction does
                    with torch.no_grad():
                        for block in model.gated_xattn_blocks:
                            block.add_char_kv()
                model.train()
                got = []
                # (the pieces' gradients differ: the carry is detached between them)
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
                for x, y in zip(ma, mb):
                    if x.dtype.is_floating_point:
                        torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-5)
                    else:
                        self.assertTrue(torch.equal(x, y))
                torch.testing.assert_close(ha, hb, rtol=1e-5, atol=1e-5)

    def test_refused_elsewhere(self) -> None:
        with self.assertRaisesRegex(ValueError, "--no-word-records"):
            make_model(layers=2, word_records=False, **PARALLEL)
        with self.assertRaisesRegex(ValueError, "--no-word-records"):
            make_model(layers=2, word_records=False, char_memory=N, memory_attention=False, **PARALLEL)
        with self.assertRaisesRegex(ValueError, "--no-word-records"):
            make_model(layers=2, word_records=False, char_memory=N, mixing="none", **PARALLEL)
        model = make_model(layers=2, word_records=False, char_memory=N)  # not the parallel path
        model.eval()
        with self.assertRaisesRegex(ValueError, "--no-word-records|--char-memory"):
            with torch.no_grad():
                run(model, self.ids, self.rows, [s[:4] for s in self.segs], None, None)

    def test_cli_flag(self) -> None:
        source = Path(runner.__file__).read_text()
        self.assertIn('"--no-word-records"', source)
        self.assertIn("word_records=not args.no_word_records", source)


if __name__ == "__main__":
    unittest.main()
