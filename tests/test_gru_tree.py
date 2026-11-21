"""--tree-levels N: GRUs alone, a binary tree of them over the character GRU.
Level k steps at the characters p where (p + 1) % 2^k == 0 (p from the
piece's or the stream's start), over level k - 1's state there; each
character gets each level's state after its latest step at or before it,
projected, times tanh(gate_k), added to its GRU state. No blocks, no records.

The model gives what a slow loop of GRUCells applying that rule gives; an
update cut anywhere (mid-pair, mid-block) gives what the whole update gives;
a character never sees a later one; at gate 0 it is the character GRU alone;
a stream that begins a piece starts every level fresh; it runs under bf16 and
on the held-out paths; it trains only what it uses; other flags are refused."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_train_streams import V, grads, make_model, ragged_segments, runner  # noqa: E402

from imm.documents import aligned_streams, cut_segments, piece_windows  # noqa: E402

CPU = torch.device("cpu")
# The parallel path with no blocks and no records, and no prior.
ALONE = dict(
    carry_bound="gru", memory_state="gru", word_records=False, prior_input="none", prior_residual_gate="none",
    prior_residual_state_features=[],
)
W = 5  # the levels' width, unlike H so a mix-up shows


def tree_model(levels=4, width=W, **extra):
    return make_model(layers=0, tree_levels=levels, tree_width=width, **{**ALONE, **extra})


def text(n, seed=1):
    return torch.randint(0, V, (n + 1,), generator=torch.Generator().manual_seed(seed))


def run(model, ids, part, memory, hidden, **kw):
    return model.segment_loss_streams(ids, part, memory, 4, carried_hidden=hidden, carry_hidden=True, **kw)


def head_outputs(model, fn):
    """fn()'s result and the logits the head gave in it, call by call."""
    seen = []
    hook = model.gru_head.register_forward_hook(lambda _m, _i, out: seen.append(out.detach().clone()))
    try:
        result = fn()
    finally:
        hook.remove()
    return seen, result


def feed(model, ids, segs, cuts, hidden=None, backward=False):
    """Run each stream's segments cuts[i]..cuts[i + 1] - 1 as update i, the
    state carried (and detached, as in training). Returns each stream's
    logits over all its characters, the summed loss and the last carry."""
    per_stream = [[] for _ in segs]
    memory, total = None, 0.0
    for a, b in zip(cuts, cuts[1:]):
        part = [s[a:b] for s in segs]
        seen, (loss, _, memory, hidden, _) = head_outputs(model, lambda: run(model, ids, part, memory, hidden))
        if backward:
            loss.backward()
        total += float(loss.detach())
        memory, hidden = tuple(m.detach() for m in memory), hidden.detach()
        sizes = [sum(e - s for s, e in p) for p in part]
        for b_, logits in enumerate(seen[0].split(sizes)):
            per_stream[b_].append(logits)
    return [torch.cat(x) for x in per_stream], total, hidden


def cut_places(segs, cuts):
    """Each cut's character count from its stream's start, for every stream."""
    return [sum(e - s for s, e in segs_b[:c]) for segs_b in segs for c in cuts[1:-1]]


def cell_like(gru):
    cell = nn.GRUCell(gru.input_size, gru.hidden_size)
    with torch.no_grad():
        for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            getattr(cell, name).copy_(getattr(gru, name + "_l0"))
    return cell


@torch.no_grad()
def slow_logits(model, ids, positions, start=None):
    """One stream's logits, a character at a time: the character GRU, then
    each level k that steps at p ((p + 1) % 2^k == 0, p from 0) over level
    k - 1's state as it now is, then the head over the GRU's state plus each
    level's current state through its projection and gate. start: the
    carried vector at p = 0 (zeros if None)."""
    tree = model.tree
    char, cells = cell_like(model.rnn), [cell_like(r) for r in tree.rnns]
    h = torch.zeros(1, model.hidden_size)
    levels = [torch.zeros(1, tree.width) for _ in cells]
    if start is not None:
        h = start[:, : model.hidden_size]
        levels = list(start[:, model.hidden_size : model.hidden_size + tree.levels * tree.width].split(tree.width, dim=-1))
    out = []
    for p, q in enumerate(positions):
        h = char(model.embed(ids[q : q + 1]), h)
        for k in range(1, tree.levels + 1):
            if (p + 1) % 2**k == 0:
                below = h if k == 1 else levels[k - 2]
                levels[k - 1] = cells[k - 1](below, levels[k - 1])
        state = h + sum(torch.tanh(tree.gates[j]) * tree.projs[j](levels[j]) for j in range(tree.levels))
        out.append(model.gru_head(state))
    return torch.cat(out)


def stream_positions(segs_b):
    return [q for s, e in segs_b for q in range(s, e)]


class GRUTreeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ids = text(600)
        self.segs = runner.split_streams(ragged_segments(len(self.ids) - 1), 3)

    def test_matches_a_slow_loop_of_the_rule(self) -> None:
        # Seven levels over ~200 characters a stream, in updates of 9
        # segments (~30 characters): the top levels go updates without a
        # step and read their carried state; every level steps somewhere.
        count = len(self.segs[0])
        cuts = list(range(0, count, 9)) + [count]
        model = tree_model(levels=7)
        model.eval()
        with torch.no_grad():
            got, _, _ = feed(model, self.ids, self.segs, cuts)
        for b, segs_b in enumerate(self.segs):
            positions = stream_positions(segs_b)
            self.assertGreater(len(positions), 2**7)
            torch.testing.assert_close(got[b], slow_logits(model, self.ids, positions), rtol=1e-5, atol=1e-5, msg=f"stream {b}")
        places = cut_places(self.segs, cuts)
        self.assertTrue(any(p % 2 for p in places))  # some updates end mid-pair

    def test_matches_the_slow_loop_from_a_carried_state(self) -> None:
        # Not from zeros: a random carried state, at place 0.
        model = tree_model(levels=3)
        model.eval()
        start = torch.randn(1, 3, model.carried_width()) * 0.5
        start[..., -1] = 0.0  # the schedule: place 0
        part = [s[:20] for s in self.segs]
        with torch.no_grad():
            seen, _ = head_outputs(model, lambda: run(model, self.ids, part, None, start))
        sizes = [sum(e - s for s, e in p) for p in part]
        for b, logits in enumerate(seen[0].split(sizes)):
            want = slow_logits(model, self.ids, stream_positions(part[b]), start=start[:, b])
            torch.testing.assert_close(logits, want, rtol=1e-5, atol=1e-5)

    def test_split_update_is_the_whole_update(self) -> None:
        # Four levels (a top block of 16 characters); one update of 40
        # segments, or two, or three, cut mid-pair and mid-block: the same
        # logits, loss and carried state (every level's and the schedule).
        model = tree_model(levels=4)
        model.train()
        got = {}
        for cuts in ((0, 40), (0, 13, 40), (0, 7, 22, 40)):
            model.zero_grad()
            got[cuts] = feed(model, self.ids, self.segs, list(cuts), backward=True) + (grads(model),)
        places = cut_places(self.segs, [0, 13, 7, 22, 40])
        self.assertTrue(any(p % 2 for p in places), places)  # mid-pair
        self.assertTrue(any(p % 2 == 0 and p % 16 for p in places), places)  # mid-block, not mid-pair
        (la, ta, ha, ga) = got[(0, 40)]
        for cuts in ((0, 13, 40), (0, 7, 22, 40)):
            with self.subTest(cuts=cuts):
                lb, tb, hb, _ = got[cuts]
                for x, y in zip(la, lb):
                    torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-5)
                self.assertLess(abs(ta - tb), 1e-5 * abs(ta))
                torch.testing.assert_close(ha, hb, rtol=1e-5, atol=1e-5)
                self.assertTrue(torch.equal(ha[..., -1], hb[..., -1]))  # the schedule, exactly
        # (split, the carry is detached between updates, so the gradients
        # differ; that every level learns at all:)
        for k in range(4):
            for name in (f"tree.rnns.{k}.weight_ih_l0", f"tree.rnns.{k}.weight_hh_l0", f"tree.projs.{k}.weight"):
                self.assertGreater(float(ga[name].abs().sum()), 0.0, name)
        self.assertTrue(bool((ga["tree.gates"] != 0).all()))
        self.assertEqual(ha.shape, (1, 3, model.hidden_size + 4 * W + 1))
        lengths = [sum(e - s for s, e in segs_b[:40]) for segs_b in self.segs]
        self.assertEqual(ha[0, :, -1].tolist(), [float(n % 16) for n in lengths])

    def test_a_character_never_sees_a_later_one(self) -> None:
        # Change stream 1's input at one character: no logit before it
        # changes, its own does. Tried at a top-level step, just before one,
        # and at a character no level steps at.
        model = tree_model(levels=4)
        model.eval()
        part = [s[:40] for s in self.segs]
        positions = stream_positions(part[1])
        for p in (15, 14, 31, 16, 30):
            with self.subTest(place=p, steps=[k for k in range(1, 5) if (p + 1) % 2**k == 0]):
                q = positions[p]
                out = []
                for flip in (False, True):
                    ids = self.ids.clone()
                    if flip:
                        ids[q] = (ids[q] + 1) % V
                    with torch.no_grad():
                        seen, _ = head_outputs(model, lambda: run(model, ids, part, None, None))
                    out.append(seen[0])
                before, after = out
                n = sum(e - s for s, e in part[0]) + p  # stream 0 comes first
                self.assertTrue(torch.equal(before[:n], after[:n]))
                self.assertFalse(torch.equal(before[n], after[n]))

    def test_gate_zero_is_the_character_gru_alone(self) -> None:
        # The tree at gate 0 against a model of no blocks and no records,
        # the same weights: three updates in train mode, the same logits,
        # losses and character state; every shared gradient the same, the
        # tree's zero but its gates'.
        plain = make_model(layers=0, **ALONE)
        tree = tree_model(levels=3)
        missing, unexpected = tree.load_state_dict(plain.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(missing and all(name.startswith("tree.") for name in missing), missing)
        with torch.no_grad():
            tree.tree.gates.zero_()
        cuts = [0, 9, 20, 33]
        out = {}
        for name, model in (("plain", plain), ("tree", tree)):
            model.train()
            model.zero_grad()
            out[name] = feed(model, self.ids, self.segs, cuts, backward=True) + (grads(model),)
        (la, ta, ha, ga), (lb, tb, hb, gb) = out["plain"], out["tree"]
        for x, y in zip(la, lb):
            self.assertTrue(torch.equal(x, y))
        self.assertEqual(ta, tb)
        self.assertTrue(torch.equal(ha, hb[..., : plain.hidden_size]))
        for name in ga:
            torch.testing.assert_close(ga[name], gb[name], rtol=1e-6, atol=0, msg=name)
        tree_grads = {name: g for name, g in gb.items() if name.startswith("tree.")}
        self.assertTrue(tree_grads)
        for name, g in tree_grads.items():
            if name != "tree.gates":
                self.assertEqual(float(g.abs().sum()), 0.0, name)
        self.assertTrue(bool((gb["tree.gates"] != 0).all()))

    def test_trains_only_what_it_uses(self) -> None:
        # The embedding, the character GRU, the tree and the head (and with
        # --documents the start state); nothing else gets a gradient.
        for documents in (False, True):
            with self.subTest(documents=documents):
                model = tree_model(levels=3, documents=documents)
                model.train()
                model.zero_grad(set_to_none=True)
                hidden = model.start_hidden(3, CPU) if documents else None
                loss, tokens, _, _, _ = run(model, self.ids, [s[:12] for s in self.segs], None, hidden)
                (loss / tokens).backward()
                named = dict(model.named_parameters())
                trained = {name for name, p in named.items() if p.grad is not None}
                allowed = ("embed.", "rnn.", "tree.", "gru_head.") + (("start_state",) if documents else ())
                self.assertEqual({n for n in trained if not n.startswith(allowed)}, set())
                for name in named:
                    if name.startswith(allowed):
                        self.assertIn(name, trained)
                        self.assertGreater(float(named[name].grad.abs().sum()), 0.0, name)
                self.assertFalse(model.gated_xattn_blocks)

    def test_both_reads_agree(self) -> None:
        # With dropout off, each level's state is projected once and then
        # read; with it, each character's read is gathered, dropped and
        # projected. Without the drop the two are the same.
        model = tree_model(levels=5)
        torch.manual_seed(3)
        states = torch.randn(3, 37, model.hidden_size)
        lengths = torch.tensor([37, 30, 12])
        carry = torch.randn(1, 3, 5 * W)
        phase = torch.tensor([0.0, 7.0, 29.0]).view(1, 3, 1)
        with torch.no_grad():
            a = model.tree(states, lengths, carry, phase, None)
            b = model.tree(states, lengths, carry, phase, lambda x: x)
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(a[2][0, :, 0], torch.tensor([5.0, 5.0, 9.0]))  # (phase + length) mod 2^5

    def test_dropout_reaches_the_read_in_training(self) -> None:
        model = tree_model(levels=3, dropout=0.5)
        seen = []
        model.tree.projs[0].register_forward_hook(lambda _m, inp, _o: seen.append(inp[0].shape))
        model.train()
        run(model, self.ids, [s[:6] for s in self.segs], None, None)
        model.eval()
        with torch.no_grad():
            run(model, self.ids, [s[:6] for s in self.segs], None, None)
        span = max(sum(e - s for s, e in seg[:6]) for seg in self.segs)
        self.assertEqual(seen[0][1], span)  # training: a read for every character
        self.assertEqual(seen[1][1], (span + 1) // 2 + 1)  # eval: each state once

    def test_grouped_update_is_the_whole_update(self) -> None:
        # --max-update-chars: streams in groups give the whole update.
        model = tree_model(levels=4)
        model.train()
        part = [s[:15] for s in self.segs]
        hidden = torch.zeros(1, 3, model.carried_width())
        hidden[..., -1] = torch.tensor([3.0, 0.0, 9.0])
        model.zero_grad()
        loss, tokens, _, whole, _ = run(model, self.ids, part, None, hidden)
        (loss / tokens).backward()
        ga = grads(model)
        model.zero_grad()
        loss_g, tokens_g, _, grouped = runner.grouped_update(
            model, self.ids, part, [slice(0, 1), slice(1, 3)], None, hidden, 4, carry_hidden=True
        )
        self.assertEqual(tokens, tokens_g)
        self.assertLess(abs(float(loss.detach()) - float(loss_g)), 1e-5 * float(loss.detach()))
        torch.testing.assert_close(whole.detach(), grouped, rtol=1e-6, atol=1e-6)
        for name, g in grads(model).items():
            torch.testing.assert_close(g, ga[name], rtol=1e-4, atol=1e-6, msg=name)

    def test_bf16(self) -> None:
        # Under autocast the tree's GRUs run float32, the projections low;
        # the loss stays near float32's and the carried state, the schedule
        # exactly, comes back float32 past bfloat16's 256.
        ids = text(2400)
        segs = runner.split_streams(ragged_segments(len(ids) - 1), 3)
        count = len(segs[0])
        cuts = list(range(0, count, 50)) + [count]
        out = {}
        for dtype in (None, torch.bfloat16):
            model = tree_model(levels=10)
            model.autocast_dtype = dtype
            dtypes = {}
            for name, mod in model.tree.named_modules():
                if isinstance(mod, (nn.GRU, nn.Linear)):
                    mod.register_forward_hook(
                        lambda _m, _i, o, name=name: dtypes.setdefault(name, set()).add((o[0] if isinstance(o, tuple) else o).dtype)
                    )
            model.eval()
            with torch.no_grad():
                logits, total, hidden = feed(model, ids, segs, cuts)
            out[dtype] = (total, hidden, dtypes)
        (l32, h32, _), (l16, h16, d16) = out[None], out[torch.bfloat16]
        self.assertEqual(h16.dtype, torch.float32)
        self.assertLess(abs(l16 / l32 - 1), 0.01)
        self.assertTrue(torch.equal(h16[..., -1], h32[..., -1]))
        self.assertGreater(float(h32[..., -1].max()), 256)
        self.assertTrue(all(v == {torch.float32} for k, v in d16.items() if k.startswith("rnns.")), d16)
        self.assertTrue(all(v == {torch.bfloat16} for k, v in d16.items() if k.startswith("projs.")), d16)
        model.autocast_dtype = torch.bfloat16
        model.train()
        loss, tokens, _, _, _ = run(model, ids, [s[:20] for s in segs], None, None)
        (loss / tokens).backward()
        self.assertTrue(all(torch.isfinite(g).all() for g in grads(model).values()))

    def test_refusals(self) -> None:
        refusals = (
            ("--tree-levels is GRUs only", dict(layers=1)),
            ("--tree-levels is GRUs only", dict(mixing="none", layers=0)),
            ("--tree-levels reads no records", dict(word_records=True)),
            ("--no-word-records|--tree-levels reads no records", dict(char_memory=4)),
            ("--tree-levels takes no tiers", dict(word_tier=5)),
            ("--tree-levels takes no tiers", dict(word_tier=5, tier_own_step=True)),
            ("--tree-levels steps over the character GRU", dict(rnn_core="tanh")),
            ("at most 24", dict(tree_levels=25)),
            ("tree_levels and tree_width must be >= 0", dict(tree_levels=-1)),
            ("tree_levels and tree_width must be >= 0", dict(tree_width=-2)),
            ("--tree-width is the tree's", dict(tree_levels=0, tree_width=8)),
        )
        for message, options in refusals:
            with self.subTest(**options):
                options = dict(options)
                layers = options.pop("layers", 0)
                with self.assertRaisesRegex(ValueError, message):
                    make_model(layers=layers, **{**ALONE, "tree_levels": 3, "tree_width": W, **options})
        # --no-word-records without the tree: no blocks and no records are
        # fine; blocks without records to read still aren't
        make_model(layers=0, **ALONE)
        with self.assertRaisesRegex(ValueError, "--no-word-records"):
            make_model(layers=1, **ALONE)
        part = [s[:4] for s in self.segs]
        loop = tree_model(memory_state="terminal")  # not the parallel path
        with self.assertRaisesRegex(ValueError, "--tree-levels needs the parallel memory path"):
            run(loop, self.ids, part, None, None)
        with self.assertRaisesRegex(ValueError, "--tree-levels runs on the batched parallel path only"):
            tree_model().segment_loss(self.ids, self.segs[0][:4], [], 4, carry_hidden=True)

    def test_cli_flags(self) -> None:
        from unittest.mock import patch

        argv = ["run_segment_memory_model.py", "--input", "x.txt", "--tree-levels", "10", "--tree-width", "96"]
        with patch.object(sys, "argv", argv):
            args = runner.parse_args()
        self.assertEqual((args.tree_levels, args.tree_width), (10, 96))
        with patch.object(sys, "argv", argv[:3]):
            args = runner.parse_args()
        self.assertEqual((args.tree_levels, args.tree_width), (0, 0))
        source = Path(runner.__file__).read_text()
        self.assertIn("tree_levels=args.tree_levels", source)
        self.assertIn("tree_width=args.tree_width", source)
        self.assertEqual(tree_model(width=0).tree.width, 8)  # 0: the hidden size


class PieceTest(unittest.TestCase):
    """--documents: a stream that begins a piece starts every level from the
    start state and its schedule from place 0."""

    STARTS = np.array([0, 97, 230, 301])

    def setUp(self) -> None:
        self.ids = text(400)
        self.segments = cut_segments(ragged_segments(399), self.STARTS)

    def model(self):
        model = tree_model(levels=4, documents=True)
        with torch.no_grad():
            model.start_state.normal_(0.0, 1.0)
        model.eval()
        return model

    def losses(self, model, fn):
        sink = []
        model.loss_sink = sink
        try:
            with torch.no_grad():
                fn()
        finally:
            model.loss_sink = None
        return dict(zip(torch.cat([p for p, _ in sink]).tolist(), torch.cat([x for _, x in sink]).tolist()))

    def streams(self, model, reset):
        """Two aligned streams, 4 segments an update; reset(hidden, memory,
        marks) at each update where some stream begins a piece."""
        streams, resets, _ = aligned_streams(self.segments, self.STARTS, 2, 4)

        def fn():
            memory, hidden = None, None
            for v in range(resets.shape[1]):
                if resets[:, v].any():
                    hidden, memory = reset(hidden, memory, resets[:, v])
                part = [segs[4 * v : 4 * v + 4] for segs in streams]
                _, _, memory, hidden, _ = run(model, self.ids, part, memory, hidden, collect_stats=True)

        return streams, resets, self.losses(model, fn)

    def test_a_reset_stream_reads_its_piece_as_a_fresh_stream(self) -> None:
        model = self.model()
        streams, resets, got = self.streams(model, model.reset_streams)
        b, u = next((b, u) for b in range(2) for u in range(1, resets.shape[1]) if resets[b, u])
        end = next((v for v in range(u + 1, resets.shape[1]) if resets[b, v]), resets.shape[1])
        piece = streams[b][4 * u : 4 * end]
        self.assertGreater(sum(e - s for s, e in streams[b][: 4 * u]) % 16, 0)  # the stream had a block open

        def alone():
            memory, hidden = None, model.start_hidden(1, CPU)
            for k in range(0, len(piece), 4):
                _, _, memory, hidden, _ = run(model, self.ids, [piece[k : k + 4]], memory, hidden, collect_stats=True)

        want = self.losses(model, alone)
        self.assertGreater(len(want), 32)
        for p, loss in want.items():
            self.assertAlmostEqual(got[p], loss, places=5, msg=f"position {p}")
        # and the reset puts every level at the start state, the schedule at 0
        H, n = model.hidden_size, model.tree.levels * model.tree.width
        with torch.no_grad():
            hidden, _ = model.reset_streams(torch.randn(1, 2, model.carried_width()), None, np.array([True, False]))
            self.assertTrue(torch.equal(hidden[0, 0, H : H + n], torch.tanh(model.start_state[H : H + n])))
        self.assertEqual(float(hidden[0, 0, -1]), 0.0)

    def test_the_tree_reset_matters(self) -> None:
        # Resetting the character GRU alone, the tree's levels and schedule
        # carried on from the piece before, scores the piece differently.
        model = self.model()
        H = model.hidden_size

        def char_only(hidden, memory, marks):
            fresh, memory = model.reset_streams(hidden, memory, marks)
            if hidden is None:
                return fresh, memory
            return torch.cat([fresh[..., :H], hidden[..., H:]], dim=-1), memory

        _, _, full = self.streams(model, model.reset_streams)
        _, _, partial = self.streams(model, char_only)
        self.assertGreater(max(abs(full[p] - partial[p]) for p in full), 1e-3)

    def test_start_state_learns_every_level(self) -> None:
        model = self.model()
        model.train()
        streams, resets, _ = aligned_streams(self.segments, self.STARTS, 2, 4)
        hidden, memory = model.reset_streams(None, None, resets[:, 0])
        loss, _, _, _, _ = run(model, self.ids, [segs[:12] for segs in streams], memory, hidden)
        loss.backward()
        g = model.start_state.grad
        H = model.hidden_size
        self.assertEqual(g.shape, (H + 4 * W,))  # the schedule's block has no learned start
        for k in range(4):
            self.assertGreater(float(g[H + k * W : H + (k + 1) * W].abs().max()), 0.0, f"level {k + 1}")

    def test_held_out_pieces_are_each_piece_alone(self) -> None:
        # evaluate with pieces (--documents' held-out score, and what
        # --char-loss-output keeps): chunks of 5 segments, cut mid-pair,
        # give each piece read in one call from the start state.
        model = self.model()
        pieces = piece_windows(self.segments, self.STARTS)
        sink = []
        model.loss_sink = sink
        nll, _, _, stats = runner.evaluate(
            model, self.ids, self.segments, 4, carry_hidden=True, feedback_state="gru", token_feedback="none",
            chunk=5, pieces=pieces,
        )
        model.loss_sink = None
        self.assertEqual(stats["token_count"], 399)
        self.assertAlmostEqual(float(torch.cat([x for _, x in sink]).sum()) / 399, nll, places=5)
        total = 0.0
        with torch.no_grad():
            for lo, hi in pieces:
                loss, _, _, _, _ = run(model, self.ids, [self.segments[lo:hi]], None, model.start_hidden(1, CPU))
                total += float(loss)
        self.assertAlmostEqual(nll, total / 399, places=5)


class CommandLineTest(unittest.TestCase):
    def test_trains_and_scores_from_the_command_line(self) -> None:
        # A tiny --documents run, trained under --bf16, then scored in
        # float32 with --char-loss-output.
        script = Path(runner.__file__)
        rng = np.random.default_rng(0)
        words = ["the", "tree", "of", "a", "grus", "steps", "once", "per", "pair"]
        body = " ".join(rng.choice(words, size=900))
        train, held = body[:3000], body[3000:3800]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "train.txt").write_text(train)
            (d / "held.txt").write_text(held)
            (d / "all.txt").write_text(train + held)
            manifest = {
                "mode": "pieces", "train_size": len(train),
                "train_records": [{"start": 0, "end": 1700}, {"start": 1700, "end": len(train)}],
                "chunk_records": [{"start": len(train), "end": len(train) + 500}, {"start": len(train) + 500, "end": len(train) + len(held)}],
            }
            (d / "manifest.json").write_text(json.dumps(manifest))
            common = [
                sys.executable, "-B", str(script), "--input", str(d / "all.txt"), "--train-input", str(d / "train.txt"),
                "--eval-input", str(d / "held.txt"), "--vocab-input", str(d / "all.txt"), "--mixing", "gated-xattn",
                "--carry-hidden", "--rope-positions", "char", "--carry-bound", "gru", "--memory-state", "gru",
                "--count-prior", "none", "--segment-rule", "word", "--train-streams", "4", "--eval-streams", "4",
                "--segments-per-update", "16", "--hidden-size", "16", "--embedding-size", "8",
                "--attention-layers", "0", "--no-word-records", "--tree-levels", "6", "--tree-width", "12",
                "--documents", str(d / "manifest.json"), "--run-id", "tree",
            ]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
            trained = subprocess.run(
                common + ["--epochs", "1", "--bf16", "--output", str(d / "tree.csv")],
                capture_output=True, text=True, env=env, cwd=d,
            )
            self.assertEqual(trained.returncode, 0, trained.stderr[-3000:])
            self.assertIn("tree_levels=6 tree_width=12", trained.stdout)
            self.assertIn("memory_path=parallel", trained.stdout)
            self.assertIn("epoch=1 tree_gates=", trained.stdout)
            checkpoint = d / "tree_tree.best.pt"
            self.assertTrue(checkpoint.exists(), sorted(p.name for p in d.iterdir()))
            scored = subprocess.run(
                common + ["--eval-only", "--resume-checkpoint", str(checkpoint), "--char-loss-output", str(d / "cl.npz"),
                          "--output", str(d / "cl.csv")],
                capture_output=True, text=True, env=env, cwd=d,
            )
            self.assertEqual(scored.returncode, 0, scored.stderr[-3000:])
            self.assertIn("char_losses heldout_ppl=", scored.stdout)
            losses = np.load(d / "cl.npz")
            self.assertEqual(len(losses["positions"]), len(held) - 1)


if __name__ == "__main__":
    unittest.main()
