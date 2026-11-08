import random
import unittest

from imm.count_gate import CountModel
from imm.edit_kernel_prior import FuzzyNodeCorrectionPrior, FuzzyNodeLocator, EditKernelPrior, train_fuzzy_node_correction_prior, train_edit_kernel_prior


class EditKernelPriorTest(unittest.TestCase):
    def test_edit_neighborhood_finds_nearby_context_successors(self) -> None:
        text = b"the king of england. the king of egypt. " * 4
        model = CountModel(text, 256, 8, [])
        model.build()
        prior = EditKernelPrior(model, candidate_cap=512)

        edit = prior.edit_rows(b"the xing")

        self.assertIsNotNone(edit)
        assert edit is not None
        probs = list(edit.probs)
        self.assertAlmostEqual(sum(probs), 1.0, places=8)
        self.assertGreater(edit.candidate_count, 0)
        self.assertGreater(probs[ord(" ")], 0.0)

    def test_training_returns_valid_probability(self) -> None:
        text = b"abc abd abc abe abc abd " * 8
        model = CountModel(text, 256, 4, [])
        model.build()
        prior = EditKernelPrior(model, candidate_cap=128)
        positions = list(range(4, min(len(text), 40)))

        theta, history = train_edit_kernel_prior(prior, text, 4, positions, epochs=2, lr=0.01)
        prob = prior.prob(b"abc", ord(" "), theta)

        self.assertEqual(len(theta), len(prior.parameter_names()))
        self.assertEqual(len(history), 2)
        self.assertGreater(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_gradient_matches_finite_differences(self) -> None:
        # The analytic gradient treats the features as constants, so compare
        # it with central differences taken with the features frozen at theta0.
        text = b"the king of england. the king of egypt. the ring of fire. " * 6
        model = CountModel(text, 256, 6, [])
        model.build()
        prior = EditKernelPrior(model, candidate_cap=256)
        rng = random.Random(0)
        theta0 = [rng.uniform(-0.5, 0.5) for _ in prior.parameter_names()]
        live_features = prior.features
        frozen: dict[bytes, tuple[float, ...]] = {}

        def frozen_features(sub: bytes, backoff: list[float]) -> tuple[float, ...]:
            if sub not in frozen:
                frozen[sub] = live_features(sub, backoff)
            return frozen[sub]

        prior.features = frozen_features  # type: ignore[method-assign]
        # Unseen context (exact falls back), edit neighbours, and plain exact hits
        for ctx, target in [(b"the xing", ord(" ")), (b"zzq of e", ord("n")), (b"king of", ord(" "))]:
            frozen.clear()
            _q, grad, _diag = prior.prob_and_grad(ctx, target, theta0)
            self.assertGreater(max(abs(g) for g in grad), 1e-3)
            h = 1e-6
            for j in range(len(theta0)):
                plus = theta0[:]
                plus[j] += h
                minus = theta0[:]
                minus[j] -= h
                fd = (prior.prob(ctx, target, plus) - prior.prob(ctx, target, minus)) / (2 * h)
                self.assertAlmostEqual(grad[j], fd, delta=1e-7, msg=f"{ctx!r} theta[{j}]")

    def test_training_moves_theta(self) -> None:
        text = b"abc abd abc abe abc abd " * 8
        model = CountModel(text, 256, 4, [])
        model.build()
        prior = EditKernelPrior(model, candidate_cap=128)
        positions = list(range(4, min(len(text), 40)))

        theta, _history = train_edit_kernel_prior(prior, text, 4, positions, epochs=2, lr=0.01)

        self.assertNotEqual(theta, prior.initial_theta())

    def test_fuzzy_node_locator_prefers_high_mass_neighbor(self) -> None:
        text = (b"the king " * 30) + (b"the ring " * 2)
        model = CountModel(text, 256, 8, [])
        model.build()
        locator = FuzzyNodeLocator(model, candidate_cap=128)

        rows = locator.rows(b"the xing")

        self.assertIsNotNone(rows)
        assert rows is not None
        self.assertAlmostEqual(sum(rows.probs), 1.0, places=8)
        self.assertGreater(rows.top_share, 0.5)
        self.assertGreater(rows.probs[ord(" ")], 0.5)

    def test_fuzzy_node_correction_training_returns_valid_probability(self) -> None:
        text = b"abc abd abc abe abc abd " * 8
        model = CountModel(text, 256, 4, [])
        model.build()
        exact_theta = [0.0] * len(model.feature_names())
        prior = FuzzyNodeCorrectionPrior(model, exact_theta, FuzzyNodeLocator(model, candidate_cap=128))
        positions = list(range(4, min(len(text), 40)))

        theta, history = train_fuzzy_node_correction_prior(prior, text, 4, positions, epochs=2, lr=0.01)
        prob = prior.prob(b"abx", ord(" "), theta)

        self.assertEqual(len(theta), len(prior.parameter_names()))
        self.assertEqual(len(history), 2)
        self.assertGreater(prob, 0.0)
        self.assertLessEqual(prob, 1.0)


if __name__ == "__main__":
    unittest.main()
