"""--prior-rows: prior rows another program wrote replace imm's count prior;
scripts/combine_prior_rows.py writes such a prefix from mobius's rows, imm's
cache, or a mixture."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from imm.data import CharVocab

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_train_streams import runner  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["reliability", "kl_gain", "entropy_norm", "depth_norm", "bias"]


class PriorRowsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.vocab = CharVocab.from_text("abcde ")
        rng = np.random.default_rng(0)
        self.train = [int(i) for i in rng.integers(0, self.vocab.size, 300)]
        self.eval = [int(i) for i in rng.integers(0, self.vocab.size, 80)]
        self.rng = rng

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def log_rows(self, n):
        return torch.log_softmax(torch.from_numpy(self.rng.standard_normal((n, self.vocab.size))), dim=-1).numpy()

    def write_prefix(self, name, features=True, **overrides):
        """A mobius_prior_rows_v2 prefix: log-probs (and gate features) for this text."""
        meta = {
            "schema": "mobius_prior_rows_v2",
            "builder": name,
            "vocab_chars": list(self.vocab.chars),
            "train_digest": runner.ids_digest(self.train),
            "eval_digest": runner.ids_digest(self.eval),
            "log_probs": {},
        }
        arrays = {}
        for split, ids in (("train", self.train), ("eval", self.eval)):
            rows = self.log_rows(len(ids) - 1).astype(np.float16)
            rows.tofile(self.dir / f"{name}.{split}_log_probs.dat")
            meta["log_probs"][split] = {"path": f"{name}.{split}_log_probs.dat", "shape": list(rows.shape), "dtype": "float16"}
            arrays[split] = rows
        if features:
            meta["gate_features"] = {"feature_names": NAMES}
            for split, ids in (("train", self.train), ("eval", self.eval)):
                rows = self.rng.random((len(ids) - 1, len(NAMES))).astype(np.float32)
                rows.tofile(self.dir / f"{name}.{split}_gate_features.dat")
                meta["gate_features"][split] = {"path": f"{name}.{split}_gate_features.dat", "shape": list(rows.shape), "dtype": "float32"}
                arrays[f"{split}_features"] = rows
        meta.update(overrides)
        (self.dir / f"{name}.json").write_text(json.dumps(meta))
        return self.dir / name, arrays

    def load(self, prefix):
        return runner.load_prior_rows(prefix, self.train, self.eval, self.vocab, torch.device("cpu"))

    def test_loads_log_probs_and_features(self) -> None:
        prefix, arrays = self.write_prefix("a")
        train, eval_, train_f, eval_f, names, meta = self.load(prefix)
        np.testing.assert_array_equal(np.asarray(train), arrays["train"])
        np.testing.assert_array_equal(eval_.numpy(), arrays["eval"])
        np.testing.assert_array_equal(np.asarray(train_f), arrays["train_features"])
        np.testing.assert_array_equal(eval_f.numpy(), arrays["eval_features"])
        self.assertEqual(names, NAMES)
        self.assertEqual(meta["builder"], "a")

    def test_features_are_optional(self) -> None:
        prefix, _ = self.write_prefix("b", features=False)
        *_, train_f, eval_f, names, _ = self.load(prefix)
        self.assertIsNone(train_f)
        self.assertIsNone(eval_f)
        self.assertIsNone(names)

    def test_refuses_other_text_vocab_or_shape(self) -> None:
        for name, overrides in [
            ("c", {"train_digest": "0" * 64}),
            ("d", {"eval_digest": "0" * 64}),
            ("e", {"vocab_chars": ["x"]}),
        ]:
            prefix, _ = self.write_prefix(name, **overrides)
            with self.assertRaises(ValueError, msg=name):
                self.load(prefix)
        prefix, _ = self.write_prefix("f")
        meta = json.loads((self.dir / "f.json").read_text())
        meta["log_probs"]["train"]["shape"] = [len(self.train) - 2, self.vocab.size]
        (self.dir / "f.json").write_text(json.dumps(meta))
        with self.assertRaises(ValueError):
            self.load(prefix)

    def test_feature_columns_by_name(self) -> None:
        self.assertEqual(runner.parse_feature_indices("0,3"), [0, 3])
        self.assertEqual(runner.parse_feature_indices("reliability,depth_norm", NAMES), [0, 3])
        self.assertEqual(runner.parse_feature_indices(""), [])
        with self.assertRaises(ValueError):
            runner.parse_feature_indices("reliability")
        with self.assertRaises(ValueError):
            runner.parse_feature_indices("nonsense", NAMES)

    def combine(self, *argv):
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "combine_prior_rows.py"), *argv],
            capture_output=True, text=True, cwd=self.dir,
        )

    def test_combine_mixes_in_probability_space(self) -> None:
        a, arrays_a = self.write_prefix("a")
        b, arrays_b = self.write_prefix("b", features=False)
        out = self.dir / "mix"
        done = self.combine("--source", f"a={a}", "--source", f"b={b}", "--log-probs", "a:0.3", "--log-probs", "b:0.7",
                            "--gate-features", "a", "--out", str(out))
        self.assertEqual(done.returncode, 0, done.stderr)
        train, eval_, train_f, _, names, meta = self.load(out)
        for split, got in (("train", np.asarray(train)), ("eval", eval_.numpy())):
            want = np.log(0.3 * np.exp(arrays_a[split].astype(np.float64)) + 0.7 * np.exp(arrays_b[split].astype(np.float64)))
            np.testing.assert_allclose(got.astype(np.float64), want, atol=0.01)
        np.testing.assert_array_equal(np.asarray(train_f), arrays_a["train_features"])
        self.assertEqual(names, NAMES)
        self.assertEqual([s["weight"] for s in meta["log_prob_sources"]], [0.3, 0.7])

    def test_combine_references_one_source(self) -> None:
        a, arrays_a = self.write_prefix("a", features=False)
        g, arrays_g = self.write_prefix("g")
        done = self.combine("--source", f"a={a}", "--source", f"g={g}", "--log-probs", "a", "--gate-features", "g",
                            "--out", str(self.dir / "one"))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertFalse((self.dir / "one.train_log_probs.dat").exists())  # referenced, not copied
        train, _, train_f, *_ = self.load(self.dir / "one")
        np.testing.assert_array_equal(np.asarray(train), arrays_a["train"])
        np.testing.assert_array_equal(np.asarray(train_f), arrays_g["train_features"])

    def test_combine_reads_imm_cache(self) -> None:
        prefix, arrays = self.write_prefix("c")
        cache = {
            "meta": {
                "vocab_chars": list(self.vocab.chars), "train_digest": runner.ids_digest(self.train),
                "eval_digest": runner.ids_digest(self.eval), "depth": 8, "extra_features": "", "train_folds": 5,
                "fold_chunk": 50,
            },
            "format": "mmap", "log_prob_dtype": "float16", "feature_dtype": "float32", "feature_count": 5,
            **{f"{s}_log_probs_path": str(self.dir / f"c.{s}_log_probs.dat") for s in ("train", "eval")},
            **{f"{s}_gate_features_path": str(self.dir / f"c.{s}_gate_features.dat") for s in ("train", "eval")},
            **{f"{s}_log_probs_shape": arrays[s].shape for s in ("train", "eval")},
            **{f"{s}_gate_features_shape": arrays[f"{s}_features"].shape for s in ("train", "eval")},
        }
        torch.save(cache, self.dir / "cache.pt")
        done = self.combine("--source", f"imm={self.dir / 'cache.pt'}", "--log-probs", "imm", "--gate-features", "imm",
                            "--out", str(self.dir / "fromcache"))
        self.assertEqual(done.returncode, 0, done.stderr)
        train, _, train_f, _, names, _ = self.load(self.dir / "fromcache")
        np.testing.assert_array_equal(np.asarray(train), arrays["train"])
        np.testing.assert_array_equal(np.asarray(train_f), arrays["train_features"])
        self.assertEqual(names, NAMES)

    def test_combine_refusals(self) -> None:
        a, _ = self.write_prefix("a")
        other, _ = self.write_prefix("o", train_digest="0" * 64)
        for argv in (
            ["--source", f"a={a}", "--log-probs", "a:0.5", "--out", str(self.dir / "x")],
            ["--source", f"a={a}", "--source", f"o={other}", "--log-probs", "a", "--out", str(self.dir / "x")],
            ["--source", f"a={a}", "--log-probs", "z", "--out", str(self.dir / "x")],
        ):
            self.assertNotEqual(self.combine(*argv).returncode, 0, argv)


if __name__ == "__main__":
    unittest.main()
