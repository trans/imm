"""--lr-schedule cosine: scheduled_lr's shape (warm-up, cosine, the floor), and
that it is transformer_baseline.py's schedule."""

import math
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_train_streams import runner  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class ScheduledLrTest(unittest.TestCase):
    LR, MIN = 1e-3, 1e-4

    def lr(self, update, total=1000, warmup=0):
        return runner.scheduled_lr(update, total, self.LR, self.MIN, warmup)

    def test_starts_at_lr_falls_to_the_floor(self) -> None:
        self.assertAlmostEqual(self.lr(0), self.LR, places=15)
        self.assertAlmostEqual(self.lr(500), (self.LR + self.MIN) / 2, places=12)
        self.assertAlmostEqual(self.lr(999), self.MIN, delta=1e-8)  # the last update, just above the floor
        self.assertEqual(self.lr(1000), self.MIN)
        self.assertEqual(self.lr(5000), self.MIN)  # past the end it stays at the floor

    def test_never_rises_after_warmup(self) -> None:
        values = [self.lr(u, warmup=50) for u in range(1000)]
        rising, falling = values[:50], values[50:]
        self.assertTrue(all(a < b for a, b in zip(rising, rising[1:])))
        self.assertTrue(all(a >= b for a, b in zip(falling, falling[1:])))
        self.assertAlmostEqual(rising[0], self.LR / 51)
        self.assertAlmostEqual(falling[0], self.LR, places=15)

    def test_is_transformer_baselines_schedule(self) -> None:
        # transformer_baseline.py defines its lr_at inside main(); its body is
        # read from the source and run here, so a change on either side shows.
        source = (ROOT / "scripts" / "transformer_baseline.py").read_text()
        body = re.search(r"    def lr_at\(step: int\) -> float:\n((?:        .*\n)+)", source)
        self.assertIsNotNone(body, "transformer_baseline.py's lr_at not found")
        namespace = {"math": math}
        exec("def lr_at(step):\n" + "".join(line[4:] + "\n" for line in body.group(1).splitlines()), namespace)
        for warmup in (0, 100):
            args = type("A", (), {"lr": self.LR, "min_lr": self.MIN, "warmup": warmup, "iters": 1000})
            namespace["args"] = args
            for update in (0, 1, 50, 99, 100, 101, 400, 999):
                self.assertAlmostEqual(namespace["lr_at"](update), self.lr(update, warmup=warmup), places=15)


if __name__ == "__main__":
    unittest.main()
