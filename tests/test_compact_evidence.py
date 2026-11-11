"""imm/compact_evidence.py: depth evidence stored as a shallow table plus
sparse deep rows, read back as the same dense float16 bytes."""

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from imm.compact_evidence import convert_split, load_compact, main, part_path, verify_split
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
DEPTH = 5


def digest(ids: list[int]) -> str:
    return hashlib.sha256(bytes(ids)).hexdigest()


def same_bytes(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(np.asarray(a).view(np.uint16), np.asarray(b).view(np.uint16))


class CompactEvidenceTest(unittest.TestCase):
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
        self.dense = {
            "train": depth_evidence_chunk(np.asarray(self.train_ids, dtype=np.uint8), model, 0, len(self.train_ids) - 1)
            .astype(np.float16),
            "eval": depth_evidence_chunk(np.asarray(self.eval_ids, dtype=np.uint8), model, 0, len(self.eval_ids) - 1)
            .astype(np.float16),
        }
        self.dense_prefix = self.dir / "dense"
        meta = {
            "train_digest": digest(self.train_ids),
            "eval_digest": digest(self.eval_ids),
            "vocab_chars": list(self.vocab.chars),
            "builder": "test",
            "depth_evidence": {},
        }
        for split in ("train", "eval"):
            self.dense[split].tofile(self.dir / f"dense.{split}_depth_evidence.dat")
            meta["depth_evidence"][split] = {
                "path": f"dense.{split}_depth_evidence.dat",
                "shape": list(self.dense[split].shape),
            }
        (self.dir / "dense.json").write_text(json.dumps(meta))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def convert(self, shallow: int, block: int = 37) -> Path:
        out = self.dir / f"compact{shallow}"
        self.assertEqual(main(["convert", str(self.dense_prefix), str(out), "--shallow", str(shallow), "--block", str(block)]), 0)
        return out

    def test_slices_rebuild_the_dense_bytes_at_every_shallow_depth(self) -> None:
        for shallow in (0, 1, 3, DEPTH):
            out = self.convert(shallow)
            compact = load_compact(out, json.loads(out.with_name(out.name + ".json").read_text()))
            for split in ("train", "eval"):
                dense = self.dense[split]
                rows = compact[split]
                self.assertEqual(rows.shape, dense.shape)
                for a, b in ((0, len(dense)), (0, 1), (5, 6), (17, 90), (len(dense) - 3, len(dense)), (10, 10)):
                    self.assertTrue(same_bytes(rows[a:b], dense[a:b]), (shallow, split, a, b))
                self.assertEqual(verify_split(dense, rows, block=13), 0)

    def test_the_shallow_table_holds_each_distinct_block_once(self) -> None:
        out = self.convert(2)
        section = json.loads(out.with_name(out.name + ".json").read_text())["compact_evidence"]
        distinct = {self.dense["train"][r, :2].tobytes() for r in range(len(self.dense["train"]))}
        self.assertEqual(section["train"]["entries"], len(distinct))
        self.assertLess(section["train"]["entries"], len(self.dense["train"]))
        deep_nonzero = int(np.count_nonzero(self.dense["train"][:, 2:].view(np.uint16)))
        self.assertEqual(section["train"]["nnz"], deep_nonzero)

    def test_arbitrary_values_round_trip_including_negative_zero(self) -> None:
        rng = np.random.default_rng(0)
        x = np.zeros((300, 4, 7), dtype=np.float16)
        mask = rng.random(x.shape) < 0.2
        x[mask] = rng.standard_normal(int(mask.sum())).astype(np.float16)
        x[3, 2, 1] = np.float16(-0.0)
        x[4, 0, 0] = np.float16(-0.0)
        for shallow in (0, 2):
            prefix = self.dir / f"rand{shallow}"
            info = convert_split(x, prefix, "train", shallow, block=64)
            section = {"schema": "depth_evidence_compact_v1", "depth": 4, "width": 7, "shallow": shallow, "train": info, "eval": info}
            for part in ("shallow_index", "shallow_table", "ptr", "pos", "val"):
                src = part_path(prefix, "train", part)
                if src.exists():
                    part_path(prefix, "eval", part).write_bytes(src.read_bytes())
            rows = load_compact(prefix, {"compact_evidence": section})["train"]
            self.assertTrue(same_bytes(rows[0:300], x))
            self.assertTrue(same_bytes(rows[3:5], x[3:5]))

    def test_the_runner_loads_compact_evidence_as_the_dense_rows(self) -> None:
        out = self.convert(3)
        device = torch.device("cpu")
        train_dense, eval_dense = runner.load_external_depth_evidence(
            self.dense_prefix, self.train_ids, self.eval_ids, self.vocab, device
        )
        train_compact, eval_compact = runner.load_external_depth_evidence(out, self.train_ids, self.eval_ids, self.vocab, device)
        self.assertEqual(train_compact.shape, train_dense.shape)
        spans = [(0, 7), (40, 41), (100, len(self.train_ids) - 1)]
        a = runner.prior_rows_for_spans(train_dense, spans, device)
        b = runner.prior_rows_for_spans(train_compact, spans, device)
        self.assertTrue(torch.equal(a.view(torch.int16), b.view(torch.int16)))
        self.assertTrue(torch.equal(eval_dense.view(torch.int16), eval_compact.view(torch.int16)))

    def test_the_runner_refuses_compact_rows_for_other_text(self) -> None:
        out = self.convert(1)
        meta_path = out.with_name(out.name + ".json")
        meta = json.loads(meta_path.read_text())
        meta["train_digest"] = "0" * 64
        meta_path.write_text(json.dumps(meta))
        with self.assertRaises(ValueError):
            runner.load_external_depth_evidence(out, self.train_ids, self.eval_ids, self.vocab, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
