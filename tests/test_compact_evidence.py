"""imm/compact_evidence.py: depth evidence stored as a shallow table plus
sparse deep rows, read back as the same dense float16 bytes."""

import hashlib
import os
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

# The GPU belongs to whoever holds the slot: GPU cases only on request.
CUDA_TESTS = os.environ.get("IMM_CUDA_TESTS") == "1"


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

    def test_take_spans_is_the_slices_one_after_another(self) -> None:
        """take_spans (the reader's few-call gather) gives the same bytes as
        concatenating slices, at every shallow depth, with empty spans,
        spans out of order and a reused out buffer."""
        span_sets = [
            [(0, 37), (37, 90), (120, 199)],
            [(150, 151), (3, 3), (10, 60), (0, 2), (198, 199)],
            [(5, 5)],
            [(0, 199)],
        ]
        for shallow in (0, 1, 3, DEPTH):
            out_prefix = self.convert(shallow)
            rows = load_compact(out_prefix, json.loads(out_prefix.with_name(out_prefix.name + ".json").read_text()))["train"]
            for spans in span_sets:
                want = np.concatenate([rows[a:b] for a, b in spans])
                self.assertTrue(same_bytes(rows.take_spans(spans), want), (shallow, spans))
                buf = np.full(want.shape, np.float16(7.0))
                self.assertIs(rows.take_spans(spans, out=buf), buf)
                self.assertTrue(same_bytes(buf, want), (shallow, spans, "out"))
            with self.assertRaises(IndexError):
                rows.take_spans([(0, len(rows) + 1)])
            with self.assertRaises(ValueError):
                rows.take_spans([(0, 4)], out=np.zeros((3, DEPTH, rows.width), np.float16))

    def test_compact_spans_expand_to_take_spans(self) -> None:
        """take_spans_compact then expand (the reader's compact read, rebuilt
        on the device) gives take_spans' bytes, at every shallow depth, with
        empty spans, spans out of order and rows without deep nonzeros;
        through the prefetcher too, which hands it to rows_to_device."""
        from imm.row_prefetch import RowPrefetcher, rows_to_device

        cpu = torch.device("cpu")
        span_sets = [
            [(0, 37), (37, 90), (120, 199)],
            [(150, 151), (3, 3), (10, 60), (0, 2), (198, 199)],
            [(5, 5)],
            [(0, 199)],
        ]
        devices = [cpu] + ([torch.device("cuda")] if torch.cuda.is_available() and CUDA_TESTS else [])
        for shallow in (0, 1, 3, DEPTH):
            out_prefix = self.convert(shallow)
            rows = load_compact(out_prefix, json.loads(out_prefix.with_name(out_prefix.name + ".json").read_text()))["train"]
            for spans in span_sets:
                want = rows.take_spans(spans)
                for device in devices:
                    got = rows.take_spans_compact(spans).expand(device)
                    self.assertEqual(got.dtype, torch.float16)
                    self.assertTrue(same_bytes(got.cpu().numpy(), want), (shallow, spans, str(device)))
            prefetcher = RowPrefetcher({"evidence": rows}, gather="vector", compact=True)
            for spans in span_sets:
                got = rows_to_device(prefetcher.submit(spans).result()["evidence"], cpu)
                self.assertTrue(same_bytes(got.numpy(), rows.take_spans(spans)), (shallow, spans, "prefetcher"))
            prefetcher.close()
            with self.assertRaises(IndexError):
                rows.take_spans_compact([(0, len(rows) + 1)])
        with self.assertRaises(ValueError):
            RowPrefetcher({"evidence": rows}, gather="spans", compact=True)

    def test_compact_rows_advise_any_range_and_keep_their_bytes(self) -> None:
        from imm.row_prefetch import RowPrefetcher, rows_to_device

        for shallow in (0, 3):
            out_prefix = self.convert(shallow)
            rows = load_compact(out_prefix, json.loads(out_prefix.with_name(out_prefix.name + ".json").read_text()))["train"]
            want = rows[0 : len(rows)]
            for a, b in ((0, len(rows)), (len(rows) - 1, len(rows) + 100), (-5, 3), (10, 10), (50, 20)):
                rows.advise_rows(a, b)
            self.assertTrue(same_bytes(rows[0 : len(rows)], want))
            spans = [(0, 37), (37, 90), (120, 199)]
            prefetcher = RowPrefetcher({"evidence": rows}, gather="vector", compact=True, readahead_rows=16)
            got = rows_to_device(prefetcher.submit(spans).result()["evidence"], torch.device("cpu"))
            self.assertTrue(same_bytes(got.numpy(), rows.take_spans(spans)))
            prefetcher.close()

    def test_compact_rows_drop_their_pages_and_keep_their_bytes(self) -> None:
        # What --drop-evidence-cache does to compact evidence: the pages of
        # each part under the rows read leave the page cache (where the file
        # system lets them), the partial last page stays, and no byte
        # changes. Until 2026-11-16 the prefetcher passed compact evidence
        # over, and the flag did nothing.
        from imm.row_prefetch import RowPrefetcher, rows_to_device

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from page_cache import evicts, resident, settle

        can_see = evicts(self.dir)
        for shallow in (0, 3):
            with self.subTest(shallow=shallow):
                out_prefix = self.convert(shallow)
                rows = load_compact(out_prefix, json.loads(out_prefix.with_name(out_prefix.name + ".json").read_text()))["train"]
                want = rows[0 : len(rows)]
                for a, b in ((0, 37), (len(rows) - 1, len(rows) + 100), (-5, 3), (10, 10), (50, 20)):
                    rows.drop_rows(a, b)
                self.assertTrue(same_bytes(rows[0 : len(rows)], want))
                spans = [(0, 37), (37, 90), (120, 199)]
                prefetcher = RowPrefetcher({"evidence": rows}, gather="vector", compact=True, readahead_rows=16,
                                           drop_cache=("evidence",))
                self.assertEqual(list(prefetcher.droppers), ["evidence"])
                got = rows_to_device(prefetcher.submit(spans).result()["evidence"], torch.device("cpu"))
                self.assertTrue(same_bytes(got.numpy(), rows.take_spans(spans)))
                prefetcher.close()
                if not can_see:
                    continue
                parts = {name: Path(getattr(rows, name).filename)
                         for name in ("ptr", "pos", "val", "index") if isinstance(getattr(rows, name), np.memmap)}
                self.assertTrue({"ptr", "pos", "val"} <= set(parts))
                for path in parts.values():
                    settle(path)
                    self.assertTrue(resident(path).all())
                self.assertEqual(set(rows.part_ranges(0, len(rows))), set(parts))
                rows.drop_rows(0, len(rows))  # every part to its end: every page goes
                for name, path in parts.items():
                    self.assertFalse(resident(path).any(), name)
                self.assertTrue(same_bytes(rows[0 : len(rows)], want))

    def test_compact_spans_keep_arbitrary_values_and_negative_zero(self) -> None:
        rng = np.random.default_rng(1)
        x = np.zeros((300, 4, 7), dtype=np.float16)
        mask = rng.random(x.shape) < 0.2
        x[mask] = rng.standard_normal(int(mask.sum())).astype(np.float16)
        x[3, 2, 1] = np.float16(-0.0)
        x[40:60] = 0  # rows with no nonzeros at all
        for shallow in (0, 2):
            prefix = self.dir / f"crand{shallow}"
            info = convert_split(x, prefix, "train", shallow, block=64)
            section = {"schema": "depth_evidence_compact_v1", "depth": 4, "width": 7, "shallow": shallow, "train": info, "eval": info}
            for part in ("shallow_index", "shallow_table", "ptr", "pos", "val"):
                src = part_path(prefix, "train", part)
                if src.exists():
                    part_path(prefix, "eval", part).write_bytes(src.read_bytes())
            rows = load_compact(prefix, {"compact_evidence": section})["train"]
            for spans in ([(0, 300)], [(41, 59)], [(250, 300), (0, 45), (3, 4)]):
                got = rows.take_spans_compact(spans).expand(torch.device("cpu"))
                self.assertTrue(same_bytes(got.numpy(), np.concatenate([x[a:b] for a, b in spans])), (shallow, spans))

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
