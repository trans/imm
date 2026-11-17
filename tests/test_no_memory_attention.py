"""--no-memory-attention: the gated blocks see no records, so the model is
the one whose attention gates are zero, MLPs kept; no gradient reaches the
record writer or the attention; it runs on the parallel path only."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_phrase_tier import PARALLEL, W, run  # noqa: E402
from test_tier_own_step import states  # noqa: E402
from test_train_streams import corpus, make_model, ragged_segments, runner  # noqa: E402

ATTENTION = ("ln_q", "ln_kv", "q_proj", "k_proj", "v_proj", "out_proj", "gate_attn")


def is_attention(name):
    return name.startswith("memory_writer.") or (
        name.startswith("gated_xattn_blocks.") and name.split(".")[2] in ATTENTION
    )


class NoMemoryAttentionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ids, self.rows = corpus(n=160)
        self.segs = runner.split_streams(ragged_segments(len(self.ids) - 1), 3)
        self.part = [s[:8] for s in self.segs]

    def test_same_as_attention_gates_at_zero(self) -> None:
        for tiers in ({}, dict(word_tier=W, tier_own_step=True)):
            with self.subTest(**tiers):
                on = make_model(layers=2, **tiers, **PARALLEL)
                off = make_model(layers=2, memory_attention=False, **tiers, **PARALLEL)
                for (name, a), (_, b) in zip(on.named_parameters(), off.named_parameters()):
                    self.assertTrue(torch.equal(a, b), name)
                on.eval()
                off.eval()
                full = states(on, self.ids, self.rows, self.part)
                got = states(off, self.ids, self.rows, self.part)
                self.assertGreater(float((full - got).abs().max()), 1e-3)  # the memory mattered
                with torch.no_grad():
                    for block in on.gated_xattn_blocks:
                        block.gate_attn.zero_()
                torch.testing.assert_close(got, states(on, self.ids, self.rows, self.part), rtol=0, atol=0)

    def test_no_gradient_to_the_writer_or_the_attention(self) -> None:
        model = make_model(layers=2, memory_attention=False, word_tier=W, tier_own_step=True, **PARALLEL)
        model.train()
        model.zero_grad()
        memory, hidden = None, None
        for u in range(2):
            part = [s[u * 4 : (u + 1) * 4] for s in self.segs]
            loss, _, memory, hidden, _ = run(model, self.ids, self.rows, part, memory, hidden)
            loss.backward()
            memory, hidden = (memory[0].detach(), memory[1]), hidden.detach()
        seen = set()
        for name, p in model.named_parameters():
            g = p.grad
            if is_attention(name):
                seen.add(name.split(".")[0])
                self.assertTrue(g is None or float(g.abs().max()) == 0.0, name)
            elif name.startswith("gated_xattn_blocks.") and ".mlp." in name:
                self.assertIsNotNone(g, name)
                self.assertGreater(float(g.abs().max()), 0.0, name)
        self.assertEqual(seen, {"memory_writer", "gated_xattn_blocks"})

    def test_parallel_path_only(self) -> None:
        with self.assertRaisesRegex(ValueError, "--mixing gated-xattn"):
            make_model(memory_attention=False, mixing="none", **PARALLEL)
        model = make_model(memory_attention=False)  # not the parallel path's carry bound and memory state
        model.eval()
        with self.assertRaisesRegex(ValueError, "--no-memory-attention"):
            with torch.no_grad():
                run(model, self.ids, self.rows, self.part, None, None)


if __name__ == "__main__":
    unittest.main()
