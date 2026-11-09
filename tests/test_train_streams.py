"""--train-streams: B streams batched through one GRU call per segment index
give what B separate single-stream runs of segment_loss give (losses,
gradients, memory, carried state)."""

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

V = 7
D = 3  # evidence depths; each row is D x (V + 1)
H = 8


def load_runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_segment_memory_model.py"
    spec = importlib.util.spec_from_file_location("run_segment_memory_model", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner_module()

# The Gutenberg config's settings, then variations the batched path also covers.
BASE = dict(
    rope_positions="char", memory_state="terminal", memory_record="written", use_rope=True, input_rope=False,
    prior_input="both", prior_residual_gate="context", prior_residual_state_features=[0, 1], prior_residual_l2=0.05,
)
VARIANTS = {
    "gutenberg": {},
    "segment rope, raw records, fused memory": dict(rope_positions="segment", memory_record="raw", memory_state="fused"),
    "input rope, no prior input, no gate": dict(input_rope=True, prior_input="none", prior_residual_gate="none",
                                                prior_residual_state_features=[]),
}


def make_model(seed=0, layers=1, **overrides):
    o = {**BASE, **overrides}
    torch.manual_seed(seed)
    model = runner.SegmentMemoryLM(
        V, H, H,
        use_rope=o["use_rope"], rope_positions=o["rope_positions"], memory_state=o["memory_state"],
        memory_record=o["memory_record"], rnn_core="gru", attn_interface_norm="none", mixing="gated-xattn",
        attention_layers=layers, attention_heads=2, input_rope=o["input_rope"], feedback_gate_bias=-1.0,
        feedback_delta_cap=1.0, prior_strength=1.0, prior_residual_scale=1.0, prior_residual_l2=o["prior_residual_l2"],
        prior_residual_gate=o["prior_residual_gate"], prior_residual_gate_max=2.0, prior_residual_gate_features=3,
        prior_residual_state_features=o["prior_residual_state_features"], skip_prior_gate="none",
        skip_prior_gate_max=0.5, skip_prior_gate_bias=0.0, skip_prior_channels=0, record_aux_weight=0.0,
        terminal_record_aux_weight=0.0, retrieval_aux_weight=0.0, utility_aux_weight=0.0, utility_temperature=1.0,
        prior_input=o["prior_input"], prior_evidence_width=D * (V + 1) if o["prior_input"] in ("depth-evidence", "both") else 0,
    )
    # Random heads and gates, not the zeroed start, so every path carries signal.
    with torch.no_grad():
        for p in model.parameters():
            if p.abs().sum() == 0:
                p.normal_(0.0, 0.3)
    return model


def corpus(n=120, seed=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, V, (n + 1,), generator=g)
    rows = {
        "prior_log_probs": torch.log_softmax(torch.randn(n, V, generator=g), dim=-1),
        "prior_gate_features": torch.randn(n, 3, generator=g),
        "prior_evidence": torch.rand(n, D, V + 1, generator=g),
    }
    return ids, rows


def ragged_segments(n, seed=2, longest=6):
    """Contiguous segments of 1..longest characters covering 0..n."""
    g = torch.Generator().manual_seed(seed)
    out, start = [], 0
    while start < n:
        end = min(n, start + int(torch.randint(1, longest + 1, (1,), generator=g)))
        out.append((start, end))
        start = end
    return out


def single(model, ids, rows, segments, memory, hidden, feedback_state):
    offset, end = segments[0][0], segments[-1][1]
    return model.segment_loss(
        ids, segments, memory, 4, carried_hidden=hidden, carry_hidden=True, feedback_state=feedback_state,
        prior_offset=offset, **{k: v[offset:end] for k, v in rows.items()},
    )


def batched(model, ids, rows, stream_segments, memory, hidden, feedback_state):
    spans = [(segs[0][0], segs[-1][1]) for segs in stream_segments]
    return model.segment_loss_streams(
        ids, stream_segments, memory, 4, carried_hidden=hidden, carry_hidden=True, feedback_state=feedback_state,
        **{k: runner.prior_rows_for_spans(v, spans, torch.device("cpu")) for k, v in rows.items()},
    )


def grads(model):
    return {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}


class TrainStreamsTest(unittest.TestCase):
    def run_both(self, streams, updates=3, per_update=5, feedback_state="gru", layers=1, **variant):
        """Train-mode losses and gradients over `updates` consecutive updates,
        state carried and detached between them as the training loop does."""
        ids, rows = corpus()
        stream_segments = runner.split_streams(ragged_segments(len(ids) - 1), streams)
        model = make_model(layers=layers, **variant)
        model.train()

        model.zero_grad()
        want_losses, want_tokens = [], 0
        for b, segs in enumerate(stream_segments):
            memory, hidden = [], None
            for u in range(updates):
                part = segs[u * per_update : (u + 1) * per_update]
                loss, tokens, memory, hidden, _ = single(model, ids, rows, part, memory, hidden, feedback_state)
                loss.backward()
                want_losses.append(loss.detach())
                want_tokens += tokens
                memory = [(state.detach(), position) for state, position in memory]
                hidden = hidden.detach()
            if b == 0:
                want_memory0, want_hidden0 = memory, hidden
        want_grads = grads(model)

        model.zero_grad()
        got_losses, got_tokens, memory, hidden = [], 0, None, None
        for u in range(updates):
            part = [segs[u * per_update : (u + 1) * per_update] for segs in stream_segments]
            loss, tokens, memory, hidden, _ = batched(model, ids, rows, part, memory, hidden, feedback_state)
            loss.backward()
            got_losses.append(loss.detach())
            got_tokens += tokens
            memory = (memory[0].detach(), memory[1])
            hidden = hidden.detach()
        got_grads = grads(model)

        torch.testing.assert_close(torch.stack(got_losses).sum(), torch.stack(want_losses).sum(), rtol=1e-5, atol=1e-5)
        self.assertEqual(got_tokens, want_tokens)
        self.assertEqual(set(got_grads), set(want_grads))
        for name in want_grads:
            torch.testing.assert_close(got_grads[name], want_grads[name], rtol=1e-4, atol=1e-5, msg=name)
        # stream 0's memory and carried state, as its own run left them
        torch.testing.assert_close(memory[0][0], torch.cat([s for s, _ in want_memory0]), rtol=1e-5, atol=1e-5)
        self.assertEqual(memory[1][0].tolist(), [p for _, p in want_memory0])
        torch.testing.assert_close(hidden[:, 0], want_hidden0[:, 0], rtol=1e-5, atol=1e-5)
        return got_losses

    def test_one_stream_is_segment_loss(self) -> None:
        for name, variant in VARIANTS.items():
            with self.subTest(name):
                self.run_both(1, **variant)

    def test_streams_are_separate_runs(self) -> None:
        for name, variant in VARIANTS.items():
            for streams in (2, 3):
                with self.subTest(name, streams=streams):
                    self.run_both(streams, **variant)

    def test_feedback_states_and_two_layers(self) -> None:
        for feedback_state in ("context", "gated"):
            with self.subTest(feedback_state):
                self.run_both(3, feedback_state=feedback_state, layers=2)

    def test_memory_longer_than_max_memory(self) -> None:
        # 3 updates of 5 segments against max_memory 4: memory rolls over
        self.run_both(2, updates=3, per_update=5)

    def test_eval_mode_has_no_l2_term(self) -> None:
        ids, rows = corpus()
        segs = runner.split_streams(ragged_segments(len(ids) - 1), 2)
        part = [s[:5] for s in segs]
        model = make_model()
        model.train()
        train_loss = batched(model, ids, rows, part, None, None, "gru")[0]
        model.eval()
        eval_loss = batched(model, ids, rows, part, None, None, "gru")[0]
        self.assertGreater(float(train_loss.detach()), float(eval_loss.detach()))

    def test_refuses_what_it_does_not_cover(self) -> None:
        model = make_model()
        model.mixing = "late"
        with self.assertRaisesRegex(ValueError, "--mixing late"):
            model.check_streams_supported("none")
        model = make_model()
        with self.assertRaisesRegex(ValueError, "--token-feedback gated"):
            model.check_streams_supported("gated")
        model.retrieval_aux_weight = 0.1
        with self.assertRaisesRegex(ValueError, "--retrieval-aux-weight"):
            model.check_streams_supported("none")

    def test_uneven_stream_lengths_are_refused(self) -> None:
        ids, rows = corpus()
        model = make_model()
        with self.assertRaisesRegex(ValueError, "same, nonzero number"):
            batched(model, ids, rows, [[(0, 3), (3, 5)], [(60, 62)]], None, None, "gru")


class SplitStreamsTest(unittest.TestCase):
    def test_equal_contiguous_runs(self) -> None:
        segments = [(i, i + 1) for i in range(10)]
        self.assertEqual(runner.split_streams(segments, 1), [segments])
        self.assertEqual(runner.split_streams(segments, 3), [segments[0:3], segments[3:6], segments[6:9]])
        with self.assertRaises(ValueError):
            runner.split_streams(segments, 11)

    def test_prior_rows_for_spans(self) -> None:
        rows = torch.arange(20.0).view(10, 2)
        got = runner.prior_rows_for_spans(rows, [(1, 3), (6, 7)], torch.device("cpu"))
        torch.testing.assert_close(got, rows[[1, 2, 6]])
        mm = rows.numpy()
        torch.testing.assert_close(runner.prior_rows_for_spans(mm, [(1, 3), (6, 7)], torch.device("cpu")), rows[[1, 2, 6]])
        self.assertIsNone(runner.prior_rows_for_spans(None, [(0, 1)], torch.device("cpu")))


if __name__ == "__main__":
    unittest.main()
