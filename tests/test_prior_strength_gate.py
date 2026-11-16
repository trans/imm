"""--prior-strength-gate: the prior's strength learned per position,
g = softplus(w . [state, features] + b). At its start (w = 0, g = 1) the
model is the one without it, losses and gradients alike, and the gate gets a
gradient; g near 0 leaves the residual alone; the other paths refuse it."""

import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_train_streams import corpus, grads, make_model, ragged_segments, runner  # noqa: E402
from test_word_tier import PARALLEL, W, run  # noqa: E402

MODEL = dict(layers=2, word_tier=W, tier_own_step=True, **PARALLEL)


def gated_twin(base):
    """A model with the gate, every shared weight copied from base, the gate
    at its start."""
    model = make_model(prior_strength_gate=True, **MODEL)
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("prior_strength_gate.") for k in missing), (missing, unexpected)
    with torch.no_grad():
        model.prior_strength_gate.weight.zero_()
        model.prior_strength_gate.bias.fill_(math.log(math.e - 1.0))
    return model


class PriorStrengthGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ids, self.rows = corpus(n=160)
        self.segs = runner.split_streams(ragged_segments(len(self.ids) - 1), 3)
        self.part = [s[:8] for s in self.segs]

    def test_start_is_the_model_without_it(self) -> None:
        base = make_model(**MODEL)
        gated = gated_twin(base)
        got = []
        for model in (base, gated):
            model.train()
            model.zero_grad()
            torch.manual_seed(5)  # the same dropout draws, were there any
            loss = run(model, self.ids, self.rows, self.part, None, None)[0]
            loss.backward()
            got.append((float(loss.detach()), grads(model)))
        (la, ga), (lb, gb) = got
        self.assertAlmostEqual(la, lb, places=5)
        for name in ga:
            torch.testing.assert_close(ga[name], gb[name], rtol=1e-5, atol=1e-6, msg=name)
        torch.testing.assert_close(gated.last_prior_strength, torch.ones_like(gated.last_prior_strength), rtol=0, atol=1e-6)
        self.assertGreater(float(gb["prior_strength_gate.weight"].abs().sum()), 0.0)
        self.assertGreater(float(gb["prior_strength_gate.bias"].abs().sum()), 0.0)

    def test_g_near_zero_leaves_the_residual(self) -> None:
        # g ~ 0: the scores are those of the same model with prior strength 0.
        base = make_model(**MODEL)
        gated = gated_twin(base)
        with torch.no_grad():
            gated.prior_strength_gate.bias.fill_(-40.0)
        base.prior_strength = 0.0
        for model in (base, gated):
            model.eval()
        with torch.no_grad():
            a = float(run(base, self.ids, self.rows, self.part, None, None)[0])
            b = float(run(gated, self.ids, self.rows, self.part, None, None)[0])
        self.assertAlmostEqual(a, b, places=4)
        self.assertLess(float(gated.last_prior_strength.max()), 1e-12)

    def test_g_depends_on_state_and_features(self) -> None:
        gated = gated_twin(make_model(**MODEL))
        with torch.no_grad():
            gated.prior_strength_gate.weight.normal_(0.0, 0.5)
        gated.eval()
        with torch.no_grad():
            run(gated, self.ids, self.rows, self.part, None, None)
        g = gated.last_prior_strength
        self.assertGreater(float(g.std()), 1e-3)
        self.assertTrue(bool((g > 0).all()))

    def test_both_batched_paths_take_it_the_single_stream_path_refuses(self) -> None:
        loop = make_model(prior_strength_gate=True)  # memory_state terminal: the batched per-segment loop
        loop.eval()
        with torch.no_grad():
            run(loop, self.ids, self.rows, [s[:5] for s in self.segs], None, None)
        self.assertEqual(loop.last_prior_strength.shape[-1], 1)
        with self.assertRaisesRegex(ValueError, "batched path only"):
            loop.segment_loss(self.ids, self.segs[0][:5], [], 4, carry_hidden=True)


if __name__ == "__main__":
    unittest.main()
