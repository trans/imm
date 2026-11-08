import math
import sqlite3
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from imm.skip_horizon_features import (
    FEATURE_NAMES,
    compile_skip_horizon_channel_rows,
    compile_skip_horizon_distribution_rows,
    compile_skip_horizon_features,
)


def u32_blob(values: list[int]) -> bytes:
    return struct.pack("<" + "I" * len(values), *values)


def u16_blob(values: list[int]) -> bytes:
    return struct.pack("<" + "H" * len(values), *values)


def f32_blob(values: list[float]) -> bytes:
    return struct.pack("<" + "f" * len(values), *values)


class SkipHorizonFeatureTest(unittest.TestCase):
    def build_db(self, path: Path) -> None:
        conn = sqlite3.connect(path)
        try:
            conn.executescript(
                """
                CREATE TABLE nodes (
                  id INTEGER NOT NULL PRIMARY KEY,
                  direction INTEGER NOT NULL,
                  parent_id INTEGER NOT NULL,
                  token INTEGER NOT NULL,
                  depth INTEGER NOT NULL,
                  key BLOB NOT NULL,
                  mass INTEGER NOT NULL,
                  UNIQUE(direction, key)
                ) WITHOUT ROWID;
                CREATE TABLE horizon_arrays (
                  node_id INTEGER NOT NULL PRIMARY KEY,
                  max_horizon INTEGER NOT NULL,
                  total_mass BLOB NOT NULL,
                  entropy BLOB NOT NULL,
                  max_target BLOB NOT NULL,
                  max_mass BLOB NOT NULL,
                  target_count BLOB NOT NULL
                ) WITHOUT ROWID;
                CREATE TABLE horizon_counts (
                  node_id INTEGER NOT NULL,
                  horizon INTEGER NOT NULL,
                  target INTEGER NOT NULL,
                  mass INTEGER NOT NULL,
                  PRIMARY KEY (node_id, horizon, target)
                ) WITHOUT ROWID;
                """
            )
            nodes = [
                (1, 0, 0, 1, 1, b"\x01", 5),
                (2, 0, 1, 2, 2, b"\x01\x02", 4),
                (3, 0, 0, 2, 1, b"\x02", 5),
                (4, 0, 3, 3, 2, b"\x02\x03", 2),
            ]
            conn.executemany(
                "INSERT INTO nodes(id, direction, parent_id, token, depth, key, mass) VALUES (?, ?, ?, ?, ?, ?, ?)",
                nodes,
            )
            ln2 = math.log(2.0)
            conn.execute(
                """
                INSERT INTO horizon_arrays(node_id, max_horizon, total_mass, entropy, max_target, max_mass, target_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    2,
                    2,
                    u32_blob([4, 4]),
                    f32_blob([0.0, ln2]),
                    b"\x03\x04",
                    u32_blob([4, 2]),
                    u16_blob([1, 2]),
                ),
            )
            conn.execute(
                """
                INSERT INTO horizon_arrays(node_id, max_horizon, total_mass, entropy, max_target, max_mass, target_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    4,
                    2,
                    u32_blob([2, 0]),
                    f32_blob([ln2, 0.0]),
                    b"\x04\x00",
                    u32_blob([1, 0]),
                    u16_blob([2, 0]),
                ),
            )
            conn.executemany(
                "INSERT INTO horizon_counts(node_id, horizon, target, mass) VALUES (?, ?, ?, ?)",
                [
                    (2, 1, 3, 4),
                    (2, 2, 4, 4),
                    (4, 1, 4, 1),
                    (4, 1, 1, 1),
                ],
            )
            conn.commit()
        finally:
            conn.close()

    def test_compile_profile_features_without_target_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "skip.sqlite"
            self.build_db(db_path)

            rows = compile_skip_horizon_features(
                [1, 2, 3, 4],
                db_path,
                depth=2,
                horizon=2,
                vocab_size=5,
                train_length=10,
            )

        self.assertEqual(rows.shape, (3, len(FEATURE_NAMES)))
        self.assertTrue(np.allclose(rows[0], 0.0))
        self.assertAlmostEqual(float(rows[1, FEATURE_NAMES.index("skip_available_frac")]), 0.5)
        self.assertAlmostEqual(float(rows[1, FEATURE_NAMES.index("skip_best_h_norm")]), 0.5)
        self.assertAlmostEqual(float(rows[1, FEATURE_NAMES.index("skip_best_sharpness")]), 1.0)
        self.assertAlmostEqual(float(rows[1, FEATURE_NAMES.index("skip_best_max_p")]), 1.0)
        self.assertAlmostEqual(float(rows[1, FEATURE_NAMES.index("skip_local_sharpness")]), 1.0)
        self.assertAlmostEqual(float(rows[2, FEATURE_NAMES.index("skip_available_frac")]), 1.0)
        self.assertGreater(float(rows[2, FEATURE_NAMES.index("skip_late_best_sharpness")]), 0.0)

    def test_compile_distribution_rows_aligns_future_horizon_votes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "skip.sqlite"
            self.build_db(db_path)

            log_rows, strength = compile_skip_horizon_distribution_rows(
                [1, 2, 3, 4],
                db_path,
                depth=2,
                horizon=2,
                vocab_size=5,
                train_length=10,
            )

        probs = np.exp(log_rows)
        self.assertEqual(log_rows.shape, (3, 5))
        self.assertEqual(strength.shape, (3, 1))
        self.assertTrue(np.allclose(probs[0], 0.2))
        self.assertGreater(float(strength[1, 0]), 0.0)
        self.assertAlmostEqual(float(probs[1, 3]), 1.0, places=5)
        self.assertGreater(float(strength[2, 0]), 0.0)
        self.assertEqual(int(probs[2].argmax()), 4)
        self.assertGreater(float(probs[2, 4]), 0.8)

    def test_compile_channel_rows_keeps_horizons_separate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "skip.sqlite"
            self.build_db(db_path)

            log_rows, strength = compile_skip_horizon_channel_rows(
                [1, 2, 3, 4],
                db_path,
                depth=2,
                horizon=2,
                vocab_size=5,
                train_length=10,
            )

        probs = np.exp(log_rows)
        self.assertEqual(log_rows.shape, (3, 2, 5))
        self.assertEqual(strength.shape, (3, 2))
        self.assertTrue(np.allclose(probs[0, 0], 0.2))
        self.assertAlmostEqual(float(probs[1, 0, 3]), 1.0, places=5)
        self.assertGreater(float(strength[1, 0]), 0.0)
        self.assertAlmostEqual(float(probs[2, 1, 4]), 1.0, places=5)
        self.assertGreater(float(strength[2, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
