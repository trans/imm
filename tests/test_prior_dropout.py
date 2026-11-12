"""--prior-dropout: in training each stream loses the prior, from its input
and its output, and is scored on its own head; at 0, or in eval mode,
nothing changes and no random number is drawn."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_train_streams import corpus, make_model, ragged_segments, runner  # noqa: E402

STREAMS = 4
CPU = torch.device("cpu")


class PriorDropoutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ids, self.rows = corpus(n=160)
        self.segments = ragged_segments(len(self.ids) - 1)
        self.stream_segments = runner.split_streams(self.segments, STREAMS)

    def loss(self, model, streams=None, rows=True):
        streams = self.stream_segments if streams is None else streams
        spans = [(segs[0][0], segs[-1][1]) for segs in streams]
        given = {k: runner.prior_rows_for_spans(v, spans, CPU) for k, v in self.rows.items()} if rows else {}
        loss, *_ = model.segment_loss_streams(self.ids, streams, None, 4, carry_hidden=True, **given)
        return loss

    def paired(self, p, **overrides):
        plain, dropped = make_model(**overrides), make_model(prior_dropout=p, **overrides)
        dropped.load_state_dict(plain.state_dict())
        return plain, dropped

    def test_zero_draws_nothing_and_changes_nothing(self) -> None:
        plain, zero = self.paired(0.0)
        plain.train(), zero.train()
        with torch.no_grad():
            torch.manual_seed(3)
            want = float(self.loss(plain))
            after = torch.rand(1)
            torch.manual_seed(3)
            self.assertEqual(float(self.loss(zero)), want)
            self.assertTrue(torch.equal(torch.rand(1), after))

    def test_eval_mode_always_has_the_prior(self) -> None:
        plain, dropped = self.paired(0.7)
        plain.eval(), dropped.eval()
        with torch.no_grad():
            torch.manual_seed(3)
            want = float(self.loss(plain))
            after = torch.rand(1)
            torch.manual_seed(3)
            self.assertEqual(float(self.loss(dropped)), want)
            self.assertTrue(torch.equal(torch.rand(1), after))

    def test_each_stream_is_scored_with_the_prior_or_on_its_own_head(self) -> None:
        """The loss is the sum over streams of the full model's loss (L2 leash
        included) for streams that keep the prior, and, for streams that
        lose it, the loss of the same weights with no prior anywhere: no
        prior input, no gate, no state features, the head alone."""
        p = 0.5
        plain, dropped = self.paired(p)
        bare = make_model(prior_input="none", prior_residual_gate="none", prior_residual_state_features=[])
        own = {k: v for k, v in plain.state_dict().items() if k in bare.state_dict()}
        bare.load_state_dict(own)
        for model in (plain, dropped, bare):
            model.train()
        seen = set()
        with torch.no_grad():
            for seed in range(6):
                torch.manual_seed(seed)
                kept = (torch.rand(STREAMS) >= p).tolist()
                seen.add(tuple(kept))
                want = 0.0
                for b, segs in enumerate(self.stream_segments):
                    if kept[b]:
                        want += float(self.loss(plain, [segs]))
                    else:
                        want += float(self.loss(bare, [segs], rows=False))
                torch.manual_seed(seed)
                self.assertAlmostEqual(float(self.loss(dropped)), want, places=3)
        self.assertTrue(any(any(k) and not all(k) for k in seen))  # some draws mix the two

    def test_gradients_flow_to_the_head_from_dropped_streams(self) -> None:
        _, dropped = self.paired(0.999)
        dropped.train()
        torch.manual_seed(0)
        self.loss(dropped).backward()
        self.assertTrue(all(p.grad is not None for p in dropped.gru_head.parameters()))
        for p in dropped.prior_input_proj.parameters():
            self.assertTrue(p.grad is None or torch.count_nonzero(p.grad) == 0)

    def test_refusals(self) -> None:
        for bad in (-0.1, 1.0):
            with self.assertRaises(ValueError):
                make_model(prior_dropout=bad)


if __name__ == "__main__":
    unittest.main()
