"""--bf16: each update runs under torch.autocast in bfloat16 (model.
autocast_dtype). The GRUs stay float32 inside it; the loss, the memory and
the carried state come back float32; the loss and the gradients stay close
to float32's; off, nothing changes; the unbatched path refuses it. On the
CPU, whose autocast also covers linear layers and attention."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_train_streams import corpus, grads, make_model, ragged_segments, runner  # noqa: E402

CPU = torch.device("cpu")
PARALLEL = dict(carry_bound="gru", memory_state="gru")


def run(model, ids, rows, stream_segments, memory=None, hidden=None, max_memory=4):
    spans = [(segs[0][0], segs[-1][1]) for segs in stream_segments]
    given = {k: runner.prior_rows_for_spans(v, spans, CPU) for k, v in rows.items()}
    return model.segment_loss_streams(
        ids, stream_segments, memory, max_memory, carried_hidden=hidden, carry_hidden=True, **given
    )


class Bf16Test(unittest.TestCase):
    def setUp(self) -> None:
        self.ids, self.rows = corpus(n=160)
        self.segs = runner.split_streams(ragged_segments(len(self.ids) - 1), 3)

    def model(self):
        model = make_model(layers=2, word_tier=5, char_memory=6, **PARALLEL)
        with torch.no_grad():  # the gates start at 0: open them, so the blocks count
            for block in model.gated_xattn_blocks:
                block.gate_attn.fill_(0.5)
                block.gate_mlp.fill_(0.5)
        return model

    def test_off_is_float32_unchanged(self) -> None:
        model = self.model()
        part = [s[:8] for s in self.segs]
        self.assertIsNone(model.autocast_dtype)
        a = run(model, self.ids, self.rows, part)
        b = model._segment_loss_streams(
            self.ids, part, None, 4, carried_hidden=None, carry_hidden=True,
            **{k: runner.prior_rows_for_spans(v, [(s[0][0], s[-1][1]) for s in part], CPU) for k, v in self.rows.items()},
        )
        self.assertTrue(torch.equal(a[0], b[0]))
        self.assertTrue(torch.equal(a[3], b[3]))

    def test_bf16_runs_the_blocks_low_and_the_grus_float32(self) -> None:
        model = self.model()
        seen = {}

        def hook(name):
            def f(mod, inp, out):
                seen.setdefault(name, set()).add((out[0] if isinstance(out, tuple) else out).dtype)
            return f

        for name, mod in model.named_modules():
            if isinstance(mod, (torch.nn.GRU, torch.nn.Linear)):
                mod.register_forward_hook(hook((type(mod).__name__, name)))
        model.autocast_dtype = torch.bfloat16
        part = [s[:8] for s in self.segs]
        loss, _, memory, hidden, _ = run(model, self.ids, self.rows, part)
        grus = {k: v for k, v in seen.items() if k[0] == "GRU"}
        linears = {k: v for k, v in seen.items() if k[0] == "Linear"}
        self.assertEqual(len(grus), 2)  # the character GRU and the word tier's
        self.assertTrue(all(v == {torch.float32} for v in grus.values()), grus)
        self.assertTrue(any(torch.bfloat16 in v for v in linears.values()), linears)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(hidden.dtype, torch.float32)
        for m in memory:
            self.assertIn(m.dtype, (torch.float32, torch.int64))

    def test_bf16_loss_and_gradients_near_float32(self) -> None:
        part = [s[:8] for s in self.segs]
        out = {}
        for dt in (None, torch.bfloat16):
            model = self.model()
            model.autocast_dtype = dt
            loss, tokens, _, _, _ = run(model, self.ids, self.rows, part)
            (loss / tokens).backward()
            out[dt] = (float(loss.detach() / tokens), grads(model))
        (l32, g32), (l16, g16) = out[None], out[torch.bfloat16]
        self.assertLess(abs(l16 / l32 - 1), 0.01)
        flat32 = torch.cat([g.flatten() for g in g32.values()])
        flat16 = torch.cat([g16[k].flatten() for k in g32])
        self.assertTrue(torch.isfinite(flat16).all())
        cos = torch.nn.functional.cosine_similarity(flat32, flat16, dim=0)
        self.assertGreater(float(cos), 0.99)

    def test_carry_across_updates_under_bf16(self) -> None:
        model = self.model()
        model.autocast_dtype = torch.bfloat16
        first, second = [s[:8] for s in self.segs], [s[8:16] for s in self.segs]
        _, _, memory, hidden, _ = run(model, self.ids, self.rows, first)
        loss, _, memory, hidden, _ = run(model, self.ids, self.rows, second, memory, hidden)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(hidden.dtype, torch.float32)

    def test_the_gru_stays_float32_under_autocast(self) -> None:
        # On CUDA, autocast would run nn.GRU in float16. On the CPU the
        # model's GRUs get float32 inputs anyway, so test the class itself
        # on a low-precision input.
        torch.manual_seed(0)
        gru = runner.GRU(4, 3, batch_first=True)
        x = torch.randn(2, 5, 4)
        want, want_h = gru(x)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out, h = gru(x.bfloat16(), torch.zeros(1, 2, 3, dtype=torch.bfloat16))
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(h.dtype, torch.float32)
        torch.testing.assert_close(out, gru(x.bfloat16().float())[0])
        self.assertLess(float((out - want).abs().max()), 0.02)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA only")
    def test_the_model_grus_stay_float32_on_cuda(self) -> None:
        model = self.model().cuda()
        model.autocast_dtype = torch.bfloat16
        seen = []
        for mod in model.modules():
            if isinstance(mod, torch.nn.GRU):
                mod.register_forward_hook(lambda m, i, o: seen.append(o[0].dtype))
        part = [s[:8] for s in self.segs]
        ids = self.ids.cuda()
        rows = {k: v.cuda() if torch.is_tensor(v) else v for k, v in self.rows.items()}
        spans = [(segs[0][0], segs[-1][1]) for segs in part]
        given = {k: runner.prior_rows_for_spans(v, spans, ids.device) for k, v in rows.items()}
        model.segment_loss_streams(ids, part, None, 4, carried_hidden=None, carry_hidden=True, **given)
        self.assertTrue(seen)
        self.assertTrue(all(d == torch.float32 for d in seen), seen)

    def test_cli_refuses_the_unbatched_path(self) -> None:
        source = Path(runner.__file__).read_text()
        self.assertIn('raise ValueError("--bf16 runs on the batched path only', source)


if __name__ == "__main__":
    unittest.main()
