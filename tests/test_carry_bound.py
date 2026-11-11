"""--carry-bound: what a stream carries into its next segment's GRU. "none"
is the state after the attention block, as before; "tanh" squashes it into
the GRU's range; "gru" carries the GRU's own last state. The single-stream
and batched paths agree, and the parameters don't change."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_train_streams import corpus, make_model, ragged_segments, runner  # noqa: E402

STREAMS = 3


class CarryBoundTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ids, self.rows = corpus()
        self.segments = ragged_segments(len(self.ids) - 1)

    def streams(self, model, carried=None):
        stream_segments = runner.split_streams(self.segments, STREAMS)
        spans = [(segs[0][0], segs[-1][1]) for segs in stream_segments]
        loss, _, _, hidden, _ = model.segment_loss_streams(
            self.ids, stream_segments, None, 4, carried_hidden=carried, carry_hidden=True,
            **{k: runner.prior_rows_for_spans(v, spans, torch.device("cpu")) for k, v in self.rows.items()},
        )
        return loss, hidden

    def single(self, model, segments):
        loss, _, _, hidden, _ = model.segment_loss(
            self.ids, segments, [], 4, carry_hidden=True, prior_offset=segments[0][0],
            **{k: v[segments[0][0] : segments[-1][1]] for k, v in self.rows.items()},
        )
        return loss, hidden

    def test_parameters_unchanged(self) -> None:
        names = list(make_model().state_dict())
        for bound in ("tanh", "norm", "gru"):
            self.assertEqual(list(make_model(carry_bound=bound).state_dict()), names)

    def test_none_is_the_old_carry(self) -> None:
        with torch.no_grad():
            a = self.streams(make_model().eval())
            b = self.streams(make_model(carry_bound="none").eval())
        self.assertEqual(float(a[0]), float(b[0]))
        self.assertTrue(torch.equal(a[1], b[1]))

    def test_tanh_and_gru_keep_the_carried_state_in_the_grus_range(self) -> None:
        """With the attention block's addition blown up, as in a runaway
        stream, the old carry leaves the GRU's range over a few chunks; tanh
        and gru stay inside it (gru by induction: each GRU step mixes the
        previous state with a tanh)."""

        def carried_max(model, chunks=4):
            hidden, worst = None, 0.0
            with torch.no_grad():
                for _ in range(chunks):
                    _, hidden = self.streams(model, hidden)
                    worst = max(worst, float(hidden.abs().max()))
            return worst

        base = make_model().eval()
        with torch.no_grad():
            for block in base.gated_xattn_blocks:
                block.gate_mlp.fill_(3.0)
                block.mlp[2].weight.mul_(20.0)
        self.assertGreater(carried_max(base), 1.0)
        for bound in ("tanh", "gru"):
            model = make_model(carry_bound=bound).eval()
            model.load_state_dict(base.state_dict())
            self.assertLessEqual(carried_max(model), 1.0, bound)
        capped = make_model(carry_bound="norm", carry_scale=2.0).eval()
        capped.load_state_dict(base.state_dict())
        with torch.no_grad():
            hidden = None
            for _ in range(4):
                _, hidden = self.streams(capped, hidden)
                self.assertLessEqual(float(hidden.norm(dim=-1).max()), 2.0 + 1e-5)

    def test_scaled_tanh_and_norm_cap(self) -> None:
        """tanh with scale c bounds every value by c and leaves small ones
        nearly as they were; norm caps the length at c and leaves shorter
        states alone."""
        x = torch.tensor([[0.1, -0.2, 40.0, -7.0]])
        small = torch.tensor([[0.3, -0.4, 0.0, 0.0]])  # length 0.5
        tanh3 = make_model(carry_bound="tanh", carry_scale=3.0)
        y = tanh3.bound_carry(x, x)
        self.assertLessEqual(float(y.abs().max()), 3.0)
        self.assertAlmostEqual(float(y[0, 0]), 0.1, places=3)
        cap = make_model(carry_bound="norm", carry_scale=5.0)
        self.assertAlmostEqual(float(cap.bound_carry(x, x).norm()), 5.0, places=4)
        self.assertTrue(torch.allclose(cap.bound_carry(x, x) / 5.0, x / x.norm()))
        self.assertTrue(torch.equal(cap.bound_carry(small, small), small))
        self.assertTrue(torch.equal(make_model(carry_bound="tanh").bound_carry(x, x), torch.tanh(x)))

    def test_single_and_batched_paths_agree(self) -> None:
        for bound in ("tanh", "norm", "gru"):
            model = make_model(carry_bound=bound, carry_scale=3.0 if bound == "tanh" else 2.0).eval()
            stream_segments = runner.split_streams(self.segments, STREAMS)
            with torch.no_grad():
                batched_loss, batched_hidden = self.streams(model)
                total = 0.0
                for b, segs in enumerate(stream_segments):
                    loss, hidden = self.single(model, segs)
                    total += float(loss)
                    self.assertTrue(torch.allclose(hidden.reshape(-1), batched_hidden[0, b], atol=1e-5), bound)
            self.assertAlmostEqual(total, float(batched_loss), places=3)


if __name__ == "__main__":
    unittest.main()
