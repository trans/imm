"""Per-depth evidence: what each context length's counts say about a row,
before blending (imm/vectorized_count_prior.py, depth_evidence_chunk)."""

import math
import tempfile
import unittest
from collections import Counter

import numpy as np

from imm.count_gate import PackedCountModel, expand_extra_features
from imm.vectorized_count_prior import (
    compile_cross_fit_depth_evidence_mmap,
    compile_depth_evidence_mmap,
    cross_fit_count_model,
    cross_fit_folds,
    depth_evidence_chunk,
)

FEATURES = expand_extra_features("entropy_delta,suffix_stats")
V, DEPTH = 8, 4


def random_text(n: int, alphabet: int, seed: int) -> list[int]:
    return np.random.default_rng(seed).integers(0, alphabet, size=n).tolist()


def by_hand(counted: list[int], query: list[int], r: int, d: int) -> tuple[list[float], float]:
    """The next-token counts after query[r-d+1 .. r] in counted, as a
    distribution and log(1 + total)."""
    if r - d + 1 < 0:
        return [0.0] * V, 0.0
    context = query[r - d + 1 : r + 1]
    counts = Counter(counted[i] for i in range(d, len(counted)) if counted[i - d : i] == context)
    total = sum(counts.values())
    if total == 0:
        return [0.0] * V, 0.0
    return [counts[t] / total for t in range(V)], math.log1p(total)


class DepthEvidenceTest(unittest.TestCase):
    def test_matches_counts_by_hand(self) -> None:
        text = random_text(300, 4, seed=0)
        model = PackedCountModel(bytes(text), V, DEPTH, FEATURES)
        model.build()
        ev = depth_evidence_chunk(np.asarray(text, dtype=np.uint8), model, 0, len(text) - 1)
        self.assertEqual(ev.shape, (299, DEPTH, V + 1))
        for r in range(299):
            for d in range(1, DEPTH + 1):
                dist, log_total = by_hand(text, text, r, d)
                np.testing.assert_allclose(ev[r, d - 1, :V], dist, atol=1e-6, err_msg=f"row {r} depth {d}")
                self.assertAlmostEqual(float(ev[r, d - 1, V]), log_total, places=5)

    def test_unseen_contexts_and_a_middle_chunk(self) -> None:
        # Held-out-style rows: counts from one text, contexts from another,
        # some never seen; and a chunk that doesn't start at row 0.
        counted = random_text(300, 3, seed=1)
        query = random_text(120, 5, seed=2)
        model = PackedCountModel(bytes(counted), V, DEPTH, FEATURES)
        model.build()
        ev = depth_evidence_chunk(np.asarray(query, dtype=np.uint8), model, 40, 119)
        seen_zero = 0
        for r in range(40, 119):
            for d in range(1, DEPTH + 1):
                dist, log_total = by_hand(counted, query, r, d)
                seen_zero += log_total == 0.0
                np.testing.assert_allclose(ev[r - 40, d - 1, :V], dist, atol=1e-6)
                self.assertAlmostEqual(float(ev[r - 40, d - 1, V]), log_total, places=5)
        self.assertGreater(seen_zero, 20)  # the case is exercised

    def test_mmap_matches_the_chunks(self) -> None:
        text = random_text(250, 4, seed=3)
        model = PackedCountModel(bytes(text), V, DEPTH, FEATURES)
        model.build()
        with tempfile.TemporaryDirectory() as tmp:
            rows = np.asarray(compile_depth_evidence_mmap(text, model, f"{tmp}/ev.dat", block_size=37))
        want = depth_evidence_chunk(np.asarray(text, dtype=np.uint8), model, 0, 249)
        np.testing.assert_allclose(rows, want, atol=2e-3, rtol=1e-3)  # float16 storage

    def test_cross_fit_rows_come_from_their_folds_model(self) -> None:
        ids = random_text(500, 5, seed=4)
        folds, chunk = 3, 60
        arr = np.asarray(ids, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            rows = np.asarray(
                compile_cross_fit_depth_evidence_mmap(ids, V, DEPTH, FEATURES, folds, chunk, f"{tmp}/ev.dat", block_size=23)
            )
        covered = 0
        for held in cross_fit_folds(len(ids), folds, chunk):
            model = cross_fit_count_model(arr, held, V, DEPTH, FEATURES)
            for start, end in held:
                a, b = max(0, start - 1), min(len(ids) - 1, end - 1)
                np.testing.assert_allclose(rows[a:b], depth_evidence_chunk(arr, model, a, b), atol=2e-3, rtol=1e-3)
                covered += b - a
        self.assertEqual(covered, len(ids) - 1)

    def test_a_string_seen_once_has_no_evidence_in_its_own_row(self) -> None:
        # "567" occurs once, in chunk 1. Counted from the whole text, row 152
        # (context ending 5, 6, 7) has evidence at depth 3; cross-fitted it has none.
        ids = random_text(400, 4, seed=6)
        ids[150:154] = [5, 6, 7, 5]
        arr = np.asarray(ids, dtype=np.uint8)
        whole = PackedCountModel(bytes(ids), V, DEPTH, FEATURES)
        whole.build()
        leaky = depth_evidence_chunk(arr, whole, 152, 153)[0]
        self.assertAlmostEqual(float(leaky[2, V]), math.log1p(1), places=5)
        self.assertEqual(float(leaky[2, 5]), 1.0)
        with tempfile.TemporaryDirectory() as tmp:
            fit = np.asarray(compile_cross_fit_depth_evidence_mmap(ids, V, DEPTH, FEATURES, 4, 100, f"{tmp}/ev.dat"))
        self.assertEqual(float(fit[152, 2, V]), 0.0)
        self.assertEqual(float(fit[152, 2, :V].sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
