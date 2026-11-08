"""Cross-fitted training priors (lessons.md, "The residual trains against a
prior that has seen the answer"). All cases are a few hundred characters."""

import math
import unittest
from collections import Counter

import numpy as np

from imm.count_gate import CountModel, PackedCountModel, expand_extra_features, pack_context
from imm.vectorized_count_prior import (
    build_depth_prior_tables,
    compile_cross_fit_rows_numpy_mmap,
    compile_prior_rows_numpy,
    cross_fit_count_model,
    cross_fit_folds,
)

FEATURES = expand_extra_features("entropy_delta,suffix_stats")


def random_text(n: int, alphabet: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return rng.integers(0, alphabet, size=n).tolist()


def theta_for(model: PackedCountModel, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.3, size=len(model.feature_names())).tolist()


def table_counts(table) -> dict[int, Counter]:
    out = {}
    for i, key in enumerate(table.keys.tolist()):
        a, b = int(table.offsets[i]), int(table.offsets[i + 1])
        out[key] = Counter(dict(zip(table.tokens[a:b].tolist(), table.counts[a:b].tolist())))
    return out


class SegmentedCountsTest(unittest.TestCase):
    def test_no_ngram_crosses_a_segment_join(self) -> None:
        tokens = random_text(300, 4, seed=0)
        starts = [0, 100, 170]
        depth = 4
        model = PackedCountModel(bytes(tokens), 8, depth, FEATURES, segment_starts=starts)
        model.build()
        segments = [tokens[s:e] for s, e in zip(starts, [*starts[1:], len(tokens)])]
        for d in range(depth + 1):
            prefix, suffix = {}, {}
            for seg in segments:
                for i in range(d, len(seg)):  # the d tokens before seg[i]
                    prefix.setdefault(pack_context(bytes(seg[i - d : i])), Counter())[seg[i]] += 1
                for i in range(len(seg) - max(d, 1)):  # the d tokens after seg[i]
                    suffix.setdefault(pack_context(bytes(seg[i + 1 : i + 1 + d])), Counter())[seg[i]] += 1
            self.assertEqual(table_counts(model.tables[d]), prefix, f"prefix depth {d}")
            self.assertEqual(table_counts(model.suffix_tables[d]), suffix, f"suffix depth {d}")

    def test_one_segment_is_the_plain_model(self) -> None:
        tokens = bytes(random_text(200, 5, seed=1))
        plain = PackedCountModel(tokens, 8, 4, FEATURES)
        one = PackedCountModel(tokens, 8, 4, FEATURES, segment_starts=[0])
        plain.build()
        one.build()
        for a, b in zip(plain.tables + plain.suffix_tables, one.tables + one.suffix_tables):
            self.assertEqual(table_counts(a), table_counts(b))


class CrossFitRowsTest(unittest.TestCase):
    def rows(self, ids, folds, chunk, theta, tmp):
        log_rows, feature_rows = compile_cross_fit_rows_numpy_mmap(
            ids, 8, 4, FEATURES, theta, folds, chunk, f"{tmp}/log.dat", f"{tmp}/feat.dat", log_dtype=np.float32
        )
        return np.asarray(log_rows), np.asarray(feature_rows)

    def test_two_folds_match_the_python_model_on_the_other_half(self) -> None:
        # Two chunks, two folds: each half's rows come from counts of the other
        # half alone, which the original pure-Python CountModel can build.
        import tempfile

        ids = random_text(400, 4, seed=2)
        probe = PackedCountModel(bytes(ids), 8, 4, FEATURES)
        theta = theta_for(probe, seed=3)
        with tempfile.TemporaryDirectory() as tmp:
            log_rows, _ = self.rows(ids, 2, 200, theta, tmp)
        text = bytes(ids)
        halves = {0: text[200:], 1: text[:200]}  # the counts each half's rows may use
        references = {}
        for fold, counted in halves.items():
            references[fold] = CountModel(counted, 8, 4, FEATURES)
            references[fold].build()
        for r in range(len(ids) - 1):
            target = r + 1
            fold = 0 if target < 200 else 1
            ctx = text[max(0, target - 4) : target]
            want = [math.log(p) for p in references[fold].gated_distribution(ctx, theta)]
            np.testing.assert_allclose(log_rows[r], want, atol=1e-5, err_msg=f"row {r}")

    def test_each_row_comes_from_its_own_folds_model(self) -> None:
        import tempfile

        ids = random_text(500, 5, seed=4)
        folds, chunk = 3, 60
        probe = PackedCountModel(bytes(ids), 8, 4, FEATURES)
        theta = theta_for(probe, seed=5)
        with tempfile.TemporaryDirectory() as tmp:
            log_rows, feature_rows = self.rows(ids, folds, chunk, theta, tmp)
        arr = np.asarray(ids, dtype=np.uint8)
        for fold, held in enumerate(cross_fit_folds(len(ids), folds, chunk)):
            model = cross_fit_count_model(arr, held, 8, 4, FEATURES)
            want_log, want_feat = compile_prior_rows_numpy(ids, model, build_depth_prior_tables(model, theta))
            for start, end in held:
                for r in range(max(0, start - 1), min(len(ids) - 1, end - 1)):
                    np.testing.assert_allclose(log_rows[r], want_log[r], atol=1e-6, err_msg=f"row {r}")
                    np.testing.assert_allclose(feature_rows[r], want_feat[r], atol=1e-6)

    def test_a_string_seen_once_no_longer_predicts_itself(self) -> None:
        # The leak in one case: "5675" occurs once, in chunk 1. The old single
        # model counted it, so it predicts its own last character well. The
        # cross-fitted prior has never seen 5, 6 or 7.
        import tempfile

        ids = random_text(400, 4, seed=6)
        ids[150:154] = [5, 6, 7, 5]
        theta = theta_for(PackedCountModel(bytes(ids), 8, 4, FEATURES), seed=7)
        leaky = PackedCountModel(bytes(ids), 8, 4, FEATURES)
        leaky.build()
        leaky_log, _ = compile_prior_rows_numpy(ids, leaky, build_depth_prior_tables(leaky, theta))
        with tempfile.TemporaryDirectory() as tmp:
            fit_log, _ = self.rows(ids, 4, 100, theta, tmp)
        row = 152  # predicts ids[153] = 5 after 5, 6, 7
        self.assertGreater(math.exp(leaky_log[row, 5]), 0.05)
        self.assertLess(math.exp(fit_log[row, 5]), 1e-6)


if __name__ == "__main__":
    unittest.main()
