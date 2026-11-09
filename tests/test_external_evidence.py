"""--prior-evidence-rows: depth evidence another program wrote (mobius, say),
loaded in place of imm's own, at whatever depth the files hold."""

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from imm.count_gate import PackedCountModel, expand_extra_features
from imm.data import CharVocab
from imm.vectorized_count_prior import depth_evidence_chunk


def load_runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_segment_memory_model.py"
    spec = importlib.util.spec_from_file_location("run_segment_memory_model", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner_module()

TEXT = "the cat sat on the mat. the dog sat on the log. " * 6
DEPTH = 5  # not the runner's default --count-prior-depth (8)


def digest(ids: list[int]) -> str:
    return hashlib.sha256(bytes(ids)).hexdigest()


class ExternalEvidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.vocab = CharVocab.from_text(TEXT)
        ids = self.vocab.encode(TEXT)
        self.train_ids, self.eval_ids = ids[:200], ids[200:]
        model = PackedCountModel(
            bytes(self.train_ids), self.vocab.size, DEPTH, expand_extra_features("entropy_delta,suffix_stats")
        )
        model.build()
        self.expected = {
            "train": depth_evidence_chunk(np.asarray(self.train_ids, dtype=np.uint8), model, 0, len(self.train_ids) - 1),
            "eval": depth_evidence_chunk(np.asarray(self.eval_ids, dtype=np.uint8), model, 0, len(self.eval_ids) - 1),
        }
        self.prefix = self.dir / "ext"
        for split in ("train", "eval"):
            self.expected[split].astype(np.float16).tofile(self.dir / f"ext.{split}_depth_evidence.dat")
        self.meta = {
            "train_digest": digest(self.train_ids),
            "eval_digest": digest(self.eval_ids),
            "vocab_chars": list(self.vocab.chars),
            "builder": "test",
        }
        self.write_meta()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_meta(self) -> None:
        (self.dir / "ext.json").write_text(json.dumps(self.meta))

    def load(self):
        return runner.load_external_depth_evidence(
            self.prefix, self.train_ids, self.eval_ids, self.vocab, torch.device("cpu")
        )

    def test_reads_rows_at_their_own_depth(self) -> None:
        train, eval_rows = self.load()
        width = self.vocab.size + 1
        self.assertEqual(train.shape, (len(self.train_ids) - 1, DEPTH, width))
        self.assertEqual(tuple(eval_rows.shape), (len(self.eval_ids) - 1, DEPTH, width))
        np.testing.assert_array_equal(np.asarray(train), self.expected["train"].astype(np.float16))
        np.testing.assert_array_equal(eval_rows.numpy(), self.expected["eval"].astype(np.float16))

    def test_paths_and_shapes_named_in_the_json(self) -> None:
        (self.dir / "ext.train_depth_evidence.dat").rename(self.dir / "elsewhere.dat")
        self.meta["train"] = {"path": "elsewhere.dat", "shape": [len(self.train_ids) - 1, DEPTH, self.vocab.size + 1]}
        self.write_meta()
        train, _ = self.load()
        self.assertEqual(train.shape[1], DEPTH)
        self.meta["train"]["shape"] = [len(self.train_ids) - 1, DEPTH + 1, self.vocab.size + 1]
        self.write_meta()
        with self.assertRaisesRegex(ValueError, "shape"):
            self.load()

    def test_refuses_rows_for_other_text(self) -> None:
        self.meta["eval_digest"] = digest(self.eval_ids[:-1])
        self.write_meta()
        with self.assertRaisesRegex(ValueError, "eval_digest"):
            self.load()

    def test_refuses_another_vocabulary(self) -> None:
        self.meta["vocab_chars"] = list(self.vocab.chars)[::-1]
        self.write_meta()
        with self.assertRaisesRegex(ValueError, "vocab_chars"):
            self.load()

    def test_refuses_a_file_of_the_wrong_size(self) -> None:
        with open(self.dir / "ext.eval_depth_evidence.dat", "ab") as handle:
            handle.write(b"\0\0")
        with self.assertRaisesRegex(ValueError, "bytes"):
            self.load()


if __name__ == "__main__":
    unittest.main()
