"""The transformer baseline scores every held-out target once, with the
context it says (scripts/transformer_baseline.py)."""

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


def load(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tb = load("transformer_baseline")


class ScoreWindowsTest(unittest.TestCase):
    def test_every_target_once_with_enough_context(self) -> None:
        for n in (2, 5, 17, 33, 100, 257):
            for block in (4, 8, 16):
                targets = []
                for s, first, end in tb.score_windows(n, block):
                    self.assertLessEqual(end - 1 - s, block)  # the window fits the model
                    for j in range(first, end):
                        self.assertGreaterEqual(j - s, min(j, block // 2 + 1))  # context length
                        targets.append(j)
                self.assertEqual(targets, list(range(1, n)), f"n={n} block={block}")


class HeldoutNllTest(unittest.TestCase):
    def test_matches_scoring_each_target_alone(self) -> None:
        torch.manual_seed(0)
        model = tb.CharGPT(vocab=7, block=8, layers=2, heads=2, width=16, dropout=0.0)
        ids = torch.randint(0, 7, (45,))
        total, count = tb.heldout_nll(model, ids, batch=3, device=torch.device("cpu"), dtype=None)
        want = 0.0
        for s, first, end in tb.score_windows(len(ids), 8):
            for j in range(first, end):  # target j from context ids[s:j], run on its own
                logits = model(ids[s:j].unsqueeze(0))[0, -1]
                want += float(F.cross_entropy(logits.unsqueeze(0), ids[j : j + 1]))
        self.assertEqual(count, 44)
        self.assertAlmostEqual(total, want, places=3)

    def test_one_window_is_the_whole_sequence_loss(self) -> None:
        torch.manual_seed(1)
        model = tb.CharGPT(vocab=5, block=32, layers=1, heads=1, width=8, dropout=0.0)
        ids = torch.randint(0, 5, (20,))
        total, count = tb.heldout_nll(model, ids, batch=4, device=torch.device("cpu"), dtype=None)
        with torch.no_grad():
            want = F.cross_entropy(model(ids[:-1].unsqueeze(0))[0], ids[1:], reduction="sum").item()
        self.assertEqual(count, 19)
        self.assertTrue(math.isclose(total, want, rel_tol=1e-5))


if __name__ == "__main__":
    unittest.main()
