"""The count-prior cache matches when the rows it holds would be the same."""

import importlib.util
import sys
import unittest
from pathlib import Path


def load_runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_segment_memory_model.py"
    spec = importlib.util.spec_from_file_location("run_segment_memory_model", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


matches = load_runner_module().count_prior_cache_matches
BASE = {"train_digest": "a", "depth": 8, "seed": 1337, "theta_json_digest": None}


class CacheMetaTest(unittest.TestCase):
    def test_the_seed_matters_when_the_gate_is_fitted(self) -> None:
        self.assertTrue(matches({"meta": dict(BASE)}, dict(BASE)))
        self.assertFalse(matches({"meta": dict(BASE)}, {**BASE, "seed": 1}))

    def test_the_seed_is_ignored_when_theta_comes_from_a_file(self) -> None:
        stored = {**BASE, "theta_json_digest": "f00"}
        self.assertTrue(matches({"meta": stored}, {**stored, "seed": 2}))
        self.assertFalse(matches({"meta": stored}, {**stored, "seed": 2, "depth": 7}))
        self.assertFalse(matches({"meta": stored}, {**stored, "theta_json_digest": "ba2"}))

    def test_no_meta_never_matches(self) -> None:
        self.assertFalse(matches({}, dict(BASE)))


if __name__ == "__main__":
    unittest.main()
