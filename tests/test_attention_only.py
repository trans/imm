"""Attention alone, and word records for it.

--rnn-core none: no character GRU; the first block's input is each
character's embedding (projected when the widths differ), and with
--depth-records each block reads character records rebuilt from the block
below, as a transformer's layers do. --word-pool mean: a word record is
written from the mean of the block's input over the word's characters.
--words-beyond-chars: a character sees a word's record only once the word's
last character is --char-memory or more back. Each: an update split in
pieces is the whole update; the records are what they say, checked by hand;
refused where it can't run. --stop-after-epoch is a CLI flag."""

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_char_memory import PARALLEL, run  # noqa: E402
from test_train_streams import corpus, grads, make_model, ragged_segments, runner  # noqa: E402

N = 6
ALONE = dict(rnn_core="none", depth_records=True, char_memory=N)


def model_of(layers=2, **extra):
    model = make_model(layers=layers, **{**ALONE, **extra}, **PARALLEL)
    with torch.no_grad():  # open the gates, so every block's attention counts
        for block in model.gated_xattn_blocks:
            block.gate_attn.fill_(0.7)
            block.gate_mlp.fill_(0.5)
    return model


def capture(model, fn):
    """Run fn with each block's forward recorded: [(block, hidden, memory, kw, out)]."""
    seen, originals = [], []
    for i, block in enumerate(model.gated_xattn_blocks):
        original = block.forward
        originals.append((block, original))

        def wrapped(hidden, memory, i=i, original=original, **kw):
            out = original(hidden, memory, **kw)
            seen.append((i, hidden, memory, kw, out))
            return out

        block.forward = wrapped
    try:
        result = fn()
    finally:
        for block, original in originals:
            block.forward = original
    return seen, result


class AttentionOnlyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ids, self.rows = corpus(n=160)
        self.segs = runner.split_streams(ragged_segments(len(self.ids) - 1), 3)

    def test_no_gru_and_the_first_block_reads_embeddings(self) -> None:
        for width in (None, 6):
            with self.subTest(embedding_size=width):
                extra = {} if width is None else dict(embedding_size=width)
                model = model_of(word_records=False, prior_input="none", **extra)  # the prior input adds to the embedding
                names = [n for n, _ in model.named_parameters()]
                self.assertFalse([n for n in names if n.startswith(("rnn.", "cell."))])
                self.assertIsInstance(model.core_in, nn.Identity if width is None else nn.Linear)
                model.eval()
                made = []
                core_in = model.core_in
                hook = core_in.register_forward_hook(lambda _m, _args, out: made.append(out))
                try:
                    with torch.no_grad():
                        seen, _ = capture(model, lambda: run(model, self.ids, self.rows, [s[:5] for s in self.segs], None, None))
                finally:
                    hook.remove()
                first = [h for i, h, *_ in seen if i == 0][0]
                self.assertTrue(torch.equal(first, made[0]))
                # every real row is a character's embedding, through core_in
                table = core_in(model.embed.weight.detach())  # [V, H]
                lengths = [sum(e - s for s, e in segs[:5]) for segs in self.segs]
                for b, n in enumerate(lengths):
                    rows = first[b, :n]
                    gap = (rows[:, None, :] - table[None, :, :]).abs().amax(dim=-1).amin(dim=-1)
                    torch.testing.assert_close(gap, torch.zeros_like(gap), rtol=0, atol=1e-6)

    def test_split_update_is_the_whole_update(self) -> None:
        cases = (
            dict(word_records=False),
            dict(word_records=False, layers=3),
            dict(word_pool="mean"),
            dict(word_pool="mean", words_beyond_chars=True),
            dict(words_beyond_chars=True, layers=3),
        )
        for extra in cases:
            with self.subTest(**extra):
                extra = dict(extra)
                model = model_of(extra.pop("layers", 2), **extra)
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
                self.assertLess(abs(la - lb), 1e-6 * abs(la))  # a sum over characters, in float32
                for x, y in zip(ma, mb):
                    if x.dtype.is_floating_point:
                        torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-5)
                    else:
                        self.assertTrue(torch.equal(x, y))
                torch.testing.assert_close(ha, hb, rtol=1e-5, atol=1e-5)
                # every block's attention and the character writer learn
                for name in ("char_writer.0.weight", "gated_xattn_blocks.0.k_proj.weight", "gated_xattn_blocks.1.v_proj.weight"):
                    self.assertGreater(float(ga[name].abs().sum()), 0.0, name)
                if extra.get("word_records", True):
                    writer = [n for n in ga if n.startswith("memory_writer.") and n.endswith("weight")]
                    self.assertTrue(writer)
                    for name in writer:
                        self.assertGreater(float(ga[name].abs().sum()), 0.0, name)

    def test_a_word_record_is_the_mean_over_its_characters(self) -> None:
        for alone in (True, False):
            with self.subTest(rnn_none=alone):
                extra = {} if alone else dict(rnn_core="gru")
                model = model_of(word_pool="mean", **extra)
                model.eval()
                part = [s[:5] for s in self.segs]
                with torch.no_grad():
                    seen, _ = capture(model, lambda: run(model, self.ids, self.rows, part, None, None))
                    for i, hidden, memory, kw, _ in seen:
                        for b, segs in enumerate(part):
                            start = 0
                            for k, (s, e) in enumerate(segs):
                                end = start + e - s
                                # the record sits at the word's last character
                                self.assertEqual(int(kw["memory_positions"][b, k]), int(kw["query_positions"][b, end - 1]))
                                want = model.memory_interface_norm(model.write_memory(hidden[b, start:end].mean(dim=0)))
                                torch.testing.assert_close(memory[b, k], want, rtol=1e-5, atol=1e-5, msg=f"block {i} {b} {k}")
                                start = end
                    self.assertEqual(sorted({i for i, *_ in seen}), [0, 1])

    def test_words_beyond_chars_is_the_mask_and_nothing_else(self) -> None:
        for extra in (dict(), dict(rnn_core="gru")):
            with self.subTest(**extra):
                near, far = model_of(**extra), model_of(words_beyond_chars=True, **extra)
                for (name, a), (_, b) in zip(near.named_parameters(), far.named_parameters()):
                    self.assertTrue(torch.equal(a, b), name)
                near.eval(), far.eval()
                carried = {"near": (None, None), "far": (None, None)}
                dropped = kept = 0
                for u in range(3):
                    part = [s[u * 4 : (u + 1) * 4] for s in self.segs]
                    masks = {}
                    for key, model in (("near", near), ("far", far)):
                        with torch.no_grad():
                            seen, out = capture(model, lambda: run(model, self.ids, self.rows, part, *carried[key], max_memory=32))
                        held = 0 if carried[key][0] is None else carried[key][0][0].shape[1]
                        words = held + 4
                        kw = [k for i, _, _, k, _ in seen if i == 0][0]
                        masks[key] = (kw["memory_mask"], kw["query_positions"], kw["memory_positions"], words)
                        carried[key] = (out[2], out[3])
                    (a, q, m, w), (b, _, _, w2) = masks["near"], masks["far"]
                    self.assertEqual(w, w2)
                    reach = (q[:, :, None] - m[:, None, :w]) >= N
                    self.assertTrue(torch.equal(b[..., :w], a[..., :w] & reach))
                    self.assertTrue(torch.equal(b[..., w:], a[..., w:]))  # the character records as before
                    dropped += int((a[..., :w] & ~reach).sum())
                    kept += int(b[..., :w].sum())
                self.assertGreater(dropped, 0)
                self.assertGreater(kept, 0)

    def test_refused_where_it_cannot_run(self) -> None:
        refusals = (
            ("--rnn-core none", dict(rnn_core="none", depth_records=True)),  # no character records
            ("--rnn-core none", dict(rnn_core="none", char_memory=N)),  # no depth records
            ("--rnn-core none", dict(rnn_core="none", char_memory=N, depth_records=True, layers=1)),
            ("--rnn-core none", dict(ALONE, word_tier=5, tier_own_step=True)),
            ("--rnn-core none|--depth-records", dict(ALONE, memory_attention=False)),
            ("--word-pool", dict(word_pool="mean", word_records=False, char_memory=N)),
            ("--word-pool", dict(word_pool="mean", memory_attention=False)),
            ("--words-beyond-chars", dict(words_beyond_chars=True)),
            ("--words-beyond-chars", dict(words_beyond_chars=True, char_memory=N, word_records=False)),
        )
        for message, options in refusals:
            with self.subTest(**options):
                options = dict(options)
                layers = options.pop("layers", 2)
                with self.assertRaisesRegex(ValueError, message):
                    make_model(layers=layers, **options, **PARALLEL)
        for options in (ALONE, dict(word_pool="mean"), dict(words_beyond_chars=True, char_memory=N)):
            with self.subTest(off_the_parallel_path=options):
                model = make_model(layers=2, **options)  # not the parallel path
                model.eval()
                with self.assertRaisesRegex(ValueError, "--rnn-core none|--word-pool|--words-beyond-chars|--char-memory"):
                    with torch.no_grad():
                        run(model, self.ids, self.rows, [s[:4] for s in self.segs], None, None)

    def test_cli_flags(self) -> None:
        source = Path(runner.__file__).read_text()
        self.assertIn('choices=["gru", "tanh", "none"]', source)
        for flag in ('"--word-pool"', '"--words-beyond-chars"', '"--stop-after-epoch"'):
            self.assertIn(flag, source)
        self.assertIn("word_pool=args.word_pool", source)
        self.assertIn("words_beyond_chars=args.words_beyond_chars", source)
        self.assertIn("if args.stop_after_epoch and epoch >= args.stop_after_epoch:", source)


if __name__ == "__main__":
    unittest.main()
