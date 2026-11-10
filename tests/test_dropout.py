"""--dropout: off at 0 (the same model and losses, no random numbers drawn),
off in eval mode, random in training mode on both the single-stream and the
batched path, and the parameter names don't change, so old checkpoints load."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_train_streams import corpus, make_model, ragged_segments, runner  # noqa: E402

STREAMS = 3


class DropoutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ids, self.rows = corpus()
        self.segments = ragged_segments(len(self.ids) - 1)

    def loss_single(self, model):
        loss, *_ = model.segment_loss(
            self.ids, self.segments, [], 4, carry_hidden=True, prior_offset=self.segments[0][0], **self.rows
        )
        return loss

    def loss_streams(self, model):
        stream_segments = runner.split_streams(self.segments, STREAMS)
        spans = [(segs[0][0], segs[-1][1]) for segs in stream_segments]
        loss, *_ = model.segment_loss_streams(
            self.ids, stream_segments, None, 4, carry_hidden=True,
            **{k: runner.prior_rows_for_spans(v, spans, torch.device("cpu")) for k, v in self.rows.items()},
        )
        return loss

    def paired(self, p):
        """A model without dropout and one with it, the same weights."""
        plain, dropped = make_model(), make_model(dropout=p)
        dropped.load_state_dict(plain.state_dict())
        return plain, dropped

    def test_parameter_names_and_init_unchanged(self) -> None:
        plain, dropped = make_model(), make_model(dropout=0.3)
        self.assertEqual(list(plain.state_dict()), list(dropped.state_dict()))
        for (name, a), b in zip(plain.state_dict().items(), dropped.state_dict().values()):
            self.assertTrue(torch.equal(a, b), name)

    def test_zero_draws_nothing_and_changes_nothing(self) -> None:
        for loss in (self.loss_single, self.loss_streams):
            plain, zero = self.paired(0.0)
            plain.train(), zero.train()
            with torch.no_grad():
                torch.manual_seed(5)
                want = loss(plain)
                after_plain = torch.rand(1)
                torch.manual_seed(5)
                got = loss(zero)
            self.assertEqual(float(got), float(want))
            self.assertTrue(torch.equal(torch.rand(1), after_plain))  # the RNG is where it was

    def test_eval_mode_is_off(self) -> None:
        for loss in (self.loss_single, self.loss_streams):
            plain, dropped = self.paired(0.5)
            plain.eval(), dropped.eval()
            with torch.no_grad():
                self.assertEqual(float(loss(dropped)), float(loss(plain)))

    def test_training_mode_is_random_and_seeded(self) -> None:
        for loss in (self.loss_single, self.loss_streams):
            plain, dropped = self.paired(0.5)
            plain.train(), dropped.train()
            with torch.no_grad():
                torch.manual_seed(1)
                a = float(loss(dropped))
                b = float(loss(dropped))
                torch.manual_seed(1)
                again = float(loss(dropped))
                self.assertNotAlmostEqual(a, float(loss(plain)), places=4)
                self.assertNotAlmostEqual(a, b, places=4)
                self.assertEqual(a, again)

    def test_gradients_flow(self) -> None:
        _, dropped = self.paired(0.2)
        dropped.train()
        self.loss_streams(dropped).backward()
        self.assertTrue(all(p.grad is not None for p in dropped.gru_head.parameters()))

    def test_refusals(self) -> None:
        for bad in (-0.1, 1.0):
            with self.assertRaises(ValueError):
                make_model(dropout=bad)


if __name__ == "__main__":
    unittest.main()
