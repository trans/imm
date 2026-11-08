import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


def load_runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_segment_memory_model.py"
    spec = importlib.util.spec_from_file_location("run_segment_memory_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SegmentMemoryConfigTest(unittest.TestCase):
    def test_config_defaults_and_cli_overrides(self) -> None:
        runner = load_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "input": "data/input.txt",
                        "output": "runs/from_config.csv",
                        "epochs": 0,
                        "carry_hidden": True,
                        "count_prior_cache": "runs/cache.pt",
                        "skip_horizon_features": "profile",
                        "skip_horizon_max": 4,
                    }
                ),
                encoding="utf-8",
            )
            argv = [
                "run_segment_memory_model.py",
                "--config",
                str(config_path),
                "--epochs",
                "3",
                "--input",
                "data/override.txt",
            ]
            with patch.object(sys, "argv", argv):
                args = runner.parse_args()

        self.assertEqual(args.input, Path("data/override.txt"))
        self.assertEqual(args.output, Path("runs/from_config.csv"))
        self.assertEqual(args.epochs, 3)
        self.assertTrue(args.carry_hidden)
        self.assertEqual(args.count_prior_cache, Path("runs/cache.pt"))
        self.assertEqual(args.skip_horizon_features, "profile")
        self.assertEqual(args.skip_horizon_max, 4)

    def test_transition_category_projection_appends_features(self) -> None:
        runner = load_runner_module()
        log_probs = torch.log(
            torch.tensor(
                [
                    [0.7, 0.2, 0.1],
                    [0.65, 0.25, 0.1],
                    [0.1, 0.2, 0.7],
                    [0.15, 0.2, 0.65],
                ],
                dtype=torch.float32,
            )
        )
        features = torch.ones((4, 2), dtype=torch.float32)

        projection = runner.fit_transition_category_projection(log_probs, dim=2, sample_rows=4, seed=1)
        appended = runner.append_transition_category_features(
            features,
            log_probs,
            projection,
            torch.device("cpu"),
        )

        self.assertEqual(appended.shape, (4, 4))
        self.assertTrue(torch.allclose(appended[:, :2], features))
        self.assertGreater(float(appended[:, 2:].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
