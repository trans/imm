"""The take-the-prior-away test: prior_ablation "output" scores the residual
logits alone, "all" also leaves the prior out of the input, "none" changes
nothing; the state sink collects the final state and target of every scored
character, and the probe's fresh layer starts as the trained head."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_train_streams import corpus, make_model, ragged_segments, runner  # noqa: E402

STREAMS = 3


class PriorAblationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ids, self.rows = corpus()
        self.segments = ragged_segments(len(self.ids) - 1)
        self.stream_segments = runner.split_streams(self.segments, STREAMS)
        spans = [(segs[0][0], segs[-1][1]) for segs in self.stream_segments]
        self.span_rows = {k: runner.prior_rows_for_spans(v, spans, torch.device("cpu")) for k, v in self.rows.items()}

    def run_streams(self, model, rows=None, score_mask=None):
        with torch.no_grad():
            return model.segment_loss_streams(
                self.ids, self.stream_segments, None, 4, carry_hidden=True, score_mask=score_mask,
                collect_stats=score_mask is not None, **(self.span_rows if rows is None else rows),
            )

    def residual_loss(self, model, rows):
        """The residual logits' loss, computed outside the model: its output
        less prior_strength times the prior's log-probs."""
        sink = []
        model.state_sink = sink
        self.run_streams(model, rows)
        model.state_sink = None
        states, targets = sink[0]
        flat = rows["prior_log_probs"]  # span rows are already in (stream, segment, offset) order
        with torch.no_grad():
            logits = model.gru_head(states.float())
            _, residual, _, _ = model.apply_prior_residual(logits, flat, rows["prior_gate_features"])
            return float(F.cross_entropy(residual, targets, reduction="sum"))

    def test_none_is_the_model_as_trained(self) -> None:
        model = make_model().eval()
        want = float(self.run_streams(model)[0])
        model.prior_ablation = "none"
        self.assertEqual(float(self.run_streams(model)[0]), want)

    def test_output_scores_the_residual_logits_alone(self) -> None:
        model = make_model().eval()
        full = float(self.run_streams(model)[0])
        model.prior_ablation = "output"
        got = float(self.run_streams(model)[0])
        model.prior_ablation = "none"
        self.assertNotAlmostEqual(got, full, places=3)
        self.assertAlmostEqual(got, self.residual_loss(model, self.span_rows), places=3)

    def test_all_leaves_the_prior_out_of_the_input(self) -> None:
        model = make_model().eval()
        model.prior_ablation = "all"
        got = float(self.run_streams(model)[0])
        # The same as "output" with the input rows zeroed out of the input
        # projections: the projections have biases and a LayerNorm, so
        # instead remove the projections themselves.
        bare = make_model(prior_input="none").eval()
        own = {k: v for k, v in model.state_dict().items() if not k.startswith(("prior_input_proj", "prior_evidence_proj"))}
        bare.load_state_dict(own)
        bare.prior_ablation = "output"
        self.assertAlmostEqual(got, float(self.run_streams(bare)[0]), places=4)
        model.prior_ablation = "output"
        self.assertNotAlmostEqual(got, float(self.run_streams(model)[0]), places=3)

    def test_unknown_mode_is_refused(self) -> None:
        model = make_model().eval()
        model.prior_ablation = "input"
        with self.assertRaises(ValueError):
            self.run_streams(model)

    def test_sink_holds_each_scored_character_once(self) -> None:
        model = make_model().eval()
        count = len(self.stream_segments[0])
        mask = np.zeros((STREAMS, count), dtype=bool)
        mask[:, count // 2 :] = True
        sink = []
        model.state_sink = sink
        loss, _, _, _, stats = self.run_streams(model, score_mask=mask)
        model.state_sink = None
        states, targets = sink[0]
        self.assertEqual(len(states), int(stats["token_count"]))
        self.assertEqual(states.dtype, torch.float16)
        want = sorted(
            int(self.ids[p + 1])
            for segs in self.stream_segments
            for start, end in segs[count // 2 :]
            for p in range(start, end)
        )
        self.assertEqual(sorted(targets.tolist()), want)

    def test_probe_starts_as_the_trained_head_and_learns(self) -> None:
        g = torch.Generator().manual_seed(0)
        head = torch.nn.Linear(4, 3)
        weights = torch.randn(3, 4, generator=g) * 2
        states = torch.randn(4000, 4, generator=g)
        targets = torch.distributions.Categorical(logits=states @ weights.T).sample()
        result = runner.train_probe(head, states[:3000], targets[:3000], states[3000:], targets[3000:],
                                    steps=500, batch=256, lr=3e-2, seed=1)
        start = math.exp(runner.probe_nll(head, states[3000:], targets[3000:]))
        self.assertAlmostEqual(result["trained_head_alone_ppl"], start, places=5)
        self.assertLess(result["heldout_ppl"], start)
        self.assertEqual(result["curve"][0]["step"], 0)
        self.assertEqual(result["train_states"], 3000)

    def test_train_states_skip_the_warm_up_chunk(self) -> None:
        model = make_model().eval()
        states, targets = runner.collect_train_states(
            model, self.ids, self.segments, STREAMS, 4, 10**6, 4, True, "gru", "none",
            self.rows["prior_log_probs"], self.rows["prior_gate_features"], self.rows["prior_evidence"],
        )
        per_stream = len(self.stream_segments[0])
        want = sum(end - start for segs in self.stream_segments for start, end in segs[4:per_stream])
        self.assertEqual(len(states), want)
        self.assertEqual(len(targets), want)
        self.assertIsNone(model.state_sink)


if __name__ == "__main__":
    unittest.main()
