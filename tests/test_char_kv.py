"""--char-kv: in each attention block, the character records get their own
LayerNorm and key and value projections; the word records keep the shared
ones. They start as copies, so the model starts as the one without it. A
block's output is masked attention over the word records' keys and values
from one map and the character records' from the other; both maps learn;
it works with --depth-records; an update split in two is the whole update;
without character records it is refused."""

import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_char_memory import PARALLEL, run, states_of  # noqa: E402
from test_train_streams import corpus, grads, make_model, ragged_segments, runner  # noqa: E402

N = 6
TIERS = dict(word_tier=5, tier_own_step=True)
CHAR_KV = ("ln_kv_char", "k_proj_char", "v_proj_char")


def model_of(char_kv, layers=2, **extra):
    model = make_model(layers=layers, char_memory=N, char_kv=char_kv, **PARALLEL, **extra)
    with torch.no_grad():
        for block in model.gated_xattn_blocks:
            block.gate_attn.fill_(0.7)
            block.gate_mlp.fill_(0.5)
            if char_kv:
                # make_model draws every all-zero parameter (the biases) on
                # its own, the copies' too: copy again, as construction does
                block.add_char_kv()
    return model


def by_hand(block, hidden, memory, split, q_pos, m_pos, mask):
    """The block's attention and MLP, written out: word keys and values from
    ln_kv, k_proj, v_proj; character ones from the _char maps."""
    B, T, H = hidden.shape
    heads, d = block.heads, block.head_dim
    q = block.q_proj(block.ln_q(hidden)).view(B, T, heads, d).transpose(1, 2)
    w, c = block.ln_kv(memory[:, :split]), block.ln_kv_char(memory[:, split:])
    k = torch.cat([block.k_proj(w), block.k_proj_char(c)], dim=1).view(B, -1, heads, d).transpose(1, 2)
    v = torch.cat([block.v_proj(w), block.v_proj_char(c)], dim=1).view(B, -1, heads, d).transpose(1, 2)
    origin = q_pos.amax(dim=-1, keepdim=True)
    q = block.apply_rope(q, q_pos - origin, True)
    k = block.apply_rope(k, m_pos - origin, True)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d)
    scores = scores.masked_fill(~mask[:, None], float("-inf"))
    out = (torch.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(B, T, H)
    hidden = hidden + torch.tanh(block.gate_attn) * block.out_proj(out)
    up, act, down = block.mlp
    return hidden + torch.tanh(block.gate_mlp) * down(act(up(block.ln_mlp(hidden))))


class CharKVTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ids, self.rows = corpus(n=160)
        self.segs = runner.split_streams(ragged_segments(len(self.ids) - 1), 3)

    def test_construction_copies_the_shared_maps(self) -> None:
        model = make_model(layers=2, char_memory=N, char_kv=True, **PARALLEL)
        for block in model.gated_xattn_blocks:
            for name in CHAR_KV:
                mine, shared = getattr(block, name), getattr(block, name.replace("_char", ""))
                self.assertIsNot(mine, shared)
                self.assertTrue(torch.equal(mine.weight, shared.weight), name)  # biases: make_model redraws zeros
                self.assertIsNot(mine.weight, shared.weight)

    def test_starts_as_the_shared_model(self) -> None:
        for depth in (False, True):
            with self.subTest(depth=depth):
                base, own = model_of(False, depth_records=depth, **TIERS), model_of(True, depth_records=depth, **TIERS)
                # make_model's draws for all-zero parameters run in parameter
                # order, so the new maps shift every later draw: give own the
                # base's parameters, then copy the maps, as construction does
                own.load_state_dict(base.state_dict(), strict=False)
                for block in own.gated_xattn_blocks:
                    block.add_char_kv()
                a, b = dict(base.named_parameters()), dict(own.named_parameters())
                for name, p in b.items():
                    if name.split(".")[-2] in CHAR_KV:
                        shared = name.replace("_char", "")
                        self.assertTrue(torch.equal(p, a[shared]), name)
                    else:
                        self.assertTrue(torch.equal(p, a[name]), name)
                extra = sum(p.numel() for n, p in b.items() if n not in a)
                H = base.hidden_size
                self.assertEqual(extra, 2 * (2 * (H * H + H) + 2 * H))  # 2 blocks: two maps and a LayerNorm each
                part = [s[:8] for s in self.segs]
                base.eval(), own.eval()
                sa, _ = states_of(base, self.ids, self.rows, part)
                sb, _ = states_of(own, self.ids, self.rows, part)
                torch.testing.assert_close(sa, sb, rtol=0, atol=0)

    def test_the_block_keys_each_kind_by_its_own_map(self) -> None:
        # Part the maps, then compare a block's output in the model with the
        # same attention written out by hand.
        model = model_of(True, prior_input="none")
        model.eval()
        torch.manual_seed(5)
        with torch.no_grad():
            for block in model.gated_xattn_blocks:
                for name in CHAR_KV:
                    for p in getattr(block, name).parameters():
                        p.add_(0.3 * torch.randn_like(p))
        block = model.gated_xattn_blocks[0]
        seen = []
        original = block.forward

        def capture(hidden, memory, **kw):
            out = original(hidden, memory, **kw)
            seen.append((hidden, memory, kw, out))
            return out

        block.forward = capture
        try:
            with torch.no_grad():
                run(model, self.ids, self.rows, [s[:5] for s in self.segs], None, None)
        finally:
            block.forward = original
        hidden, memory, kw, out = seen[0]
        split = kw["kind_split"]
        self.assertEqual(split, 5)  # this update's 5 word records, nothing held
        mask = kw["memory_mask"]
        with torch.no_grad():
            want = by_hand(block, hidden, memory, split, kw["query_positions"], kw["memory_positions"], mask)
        sees = mask.any(dim=-1)  # a query that sees nothing gets no attention output: compare those that see
        torch.testing.assert_close(out[sees], want[sees], rtol=1e-4, atol=1e-5)
        # and the parted maps matter
        with torch.no_grad():
            for name in CHAR_KV:
                getattr(block, name).load_state_dict(getattr(block, name.replace("_char", "")).state_dict())
            shared = by_hand(block, hidden, memory, split, kw["query_positions"], kw["memory_positions"], mask)
        self.assertGreater(float((shared[sees] - want[sees]).abs().max()), 1e-3)

    def test_both_maps_learn_and_split_update_is_whole(self) -> None:
        for depth in (False, True):
            with self.subTest(depth=depth):
                model = model_of(True, depth_records=depth, **TIERS)
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
                for x, y in zip(ma, mb):
                    if x.dtype.is_floating_point:
                        torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-5)
                    else:
                        self.assertTrue(torch.equal(x, y))
                torch.testing.assert_close(ha, hb, rtol=1e-5, atol=1e-5)
                for i in range(2):
                    for name in ("k_proj", "k_proj_char", "v_proj", "v_proj_char"):
                        key = f"gated_xattn_blocks.{i}.{name}.weight"
                        self.assertGreater(float(ga[key].abs().sum()), 0.0, key)

    def test_refused_without_character_records(self) -> None:
        with self.assertRaisesRegex(ValueError, "--char-kv"):
            make_model(layers=2, char_kv=True, **PARALLEL)
        with self.assertRaisesRegex(ValueError, "--char-kv"):
            make_model(layers=2, char_kv=True, char_memory=N, memory_attention=False, **PARALLEL)

    def test_cli_flag(self) -> None:
        source = Path(runner.__file__).read_text()
        self.assertIn('"--char-kv"', source)
        self.assertIn("char_kv=args.char_kv", source)


if __name__ == "__main__":
    unittest.main()
