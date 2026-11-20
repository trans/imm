"""RoPE in the gated cross-attention blocks scores by the gap between a
query and a key, so where a stream sits in the text must not matter.
Positions are character offsets into the training text, up to 1e8; float32
can't hold position x frequency there to within a radian, and bfloat16
(--bf16) holds integers exactly only to 256. The blocks count positions
from each stream's last query and keep the angles in float32: the output
is the same at offset 0 and at offset 7e7, in float32 and under bf16
autocast, and so is the model's loss on a stream placed far into a text."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_train_streams import make_model, ragged_segments, runner  # noqa: E402

FAR = 70_000_001  # past 2^26: float32 spaces integers 8 apart here
NO_PRIOR = dict(prior_input="none", prior_residual_gate="none", prior_residual_state_features=[])


def block_and_inputs(seed=0, streams=3, queries=11, records=17, hidden=32, heads=4):
    torch.manual_seed(seed)
    block = runner.GatedCrossAttentionBlock(hidden, heads)
    with torch.no_grad():
        block.gate_attn.fill_(1.0)
        block.gate_mlp.fill_(0.5)
        block.q_proj.weight.mul_(3)  # sharper attention, so positions matter
        block.k_proj.weight.mul_(3)
    x = torch.randn(streams, queries, hidden)
    memory = torch.randn(streams, records, hidden)
    q = torch.arange(queries)[None, :] + torch.tensor([[40], [300], [7]])  # per-stream positions
    m = torch.arange(records)[None, :] - 5 + torch.tensor([[40], [300], [7]])
    mask = (m[:, None, :] <= q[:, :, None])
    return block, x, memory, q, m, mask


class BlockRopeTest(unittest.TestCase):
    def out(self, block, x, memory, q, m, mask, offset):
        return block(x, memory, query_positions=q + offset, memory_positions=m + offset, memory_mask=mask)

    def test_float32_is_the_same_far_into_the_text(self) -> None:
        block, x, memory, q, m, mask = block_and_inputs()
        near = self.out(block, x, memory, q, m, mask, 0)
        far = self.out(block, x, memory, q, m, mask, FAR)
        torch.testing.assert_close(far, near, rtol=1e-5, atol=1e-5)

    def test_bf16_keeps_positions(self) -> None:
        block, x, memory, q, m, mask = block_and_inputs()
        with torch.no_grad():
            want = self.out(block, x, memory, q, m, mask, 0)
            # what positions add: the output with every gap taken as 0
            signal = float((self.out(block, x, memory, 0 * q, 0 * m, mask, 0) - want).abs().max())
            for offset in (0, 1_000, FAR):
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    got = self.out(block, x, memory, q, m, mask, offset).float()
                # bf16 rounding of the activations only, not of the positions
                self.assertLess(float((got - want).abs().max()), 0.15 * signal, offset)

    def test_bf16_keeps_long_gaps(self) -> None:
        # Character records reach 1,024 back, word records further: gaps
        # past 256, which bf16 can't hold exactly.
        block, x, memory, q, _, _ = block_and_inputs()
        m = q[:, -1:] - torch.linspace(1_900, 0, memory.shape[1]).round().long()[None, :]
        mask = m[:, None, :] <= q[:, :, None]
        with torch.no_grad():
            want = self.out(block, x, memory, q, m, mask, 0)
            signal = float((self.out(block, x, memory, 0 * q, 0 * m, mask, 0) - want).abs().max())
            for offset in (0, FAR):
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    got = self.out(block, x, memory, q, m, mask, offset).float()
                self.assertLess(float((got - want).abs().max()), 0.15 * signal, offset)

    def test_unmasked_path_too(self) -> None:
        block, x, memory, q, m, _ = block_and_inputs(streams=1)
        near = block(x, memory, query_positions=q[0], memory_positions=m[0])
        far = block(x, memory, query_positions=q[0] + FAR, memory_positions=m[0] + FAR)
        torch.testing.assert_close(far, near, rtol=1e-5, atol=1e-5)


class ModelRopeTest(unittest.TestCase):
    def test_a_stream_far_into_the_text_scores_the_same(self) -> None:
        torch.manual_seed(3)
        n = 400
        small = torch.randint(0, 7, (n + 1,))
        ids = torch.empty(FAR + n + 1, dtype=torch.long)  # untouched pages cost nothing
        ids[FAR:] = small
        segs = runner.split_streams(ragged_segments(n, longest=7), 3)
        model = make_model(layers=2, word_tier=5, char_memory=6, carry_bound="gru", memory_state="gru", **NO_PRIOR)
        with torch.no_grad():
            for block in model.gated_xattn_blocks:
                block.gate_attn.fill_(1.0)
                block.gate_mlp.fill_(0.5)
        losses = []
        for text, shift in ((small, 0), (ids, FAR)):
            batch = [[(s + shift, e + shift) for s, e in st[:12]] for st in segs]
            loss, tokens, _, _, _ = model.segment_loss_streams(text, batch, None, 4, carried_hidden=None, carry_hidden=True)
            losses.append(float(loss.detach()) / tokens)
        self.assertAlmostEqual(losses[0], losses[1], places=5)


if __name__ == "__main__":
    unittest.main()
