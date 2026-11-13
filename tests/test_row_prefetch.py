"""RowPrefetcher: the same rows as a direct read, from memmaps and tensors,
in submission order; dropping a memmap's pages from the cache changes no
values."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from imm.row_prefetch import PageDropper, RowPrefetcher, gather_span_rows, rows_to_device, take_span_rows

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_train_streams import runner  # noqa: E402


class RowPrefetchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        rng = np.random.default_rng(0)
        self.data = rng.standard_normal((5000, 3, 7)).astype(np.float16)
        path = Path(self.tmp.name) / "evidence.dat"
        self.data.tofile(path)
        self.mm = np.memmap(path, dtype=np.float16, mode="r", shape=self.data.shape)
        self.tensor = torch.from_numpy(rng.standard_normal((5000, 4)).astype(np.float32))
        self.spans = [[(0, 37), (1200, 1290), (4000, 4013)], [(37, 90), (1290, 1500), (4013, 4999)]]

    def tearDown(self) -> None:
        del self.mm
        self.tmp.cleanup()

    def test_same_rows_as_a_direct_read(self) -> None:
        cpu = torch.device("cpu")
        for drop in ((), ("evidence",)):
            prefetcher = RowPrefetcher({"evidence": self.mm, "features": self.tensor, "none": None}, drop_cache=drop)
            futures = [prefetcher.submit(spans) for spans in self.spans]
            for spans, future in zip(self.spans, futures):
                got = future.result()
                self.assertIsNone(got["none"])
                for name, rows in (("evidence", self.mm), ("features", self.tensor)):
                    want = runner.prior_rows_for_spans(rows, spans, cpu)
                    self.assertTrue(torch.equal(rows_to_device(got[name], cpu), want), name)
            prefetcher.close()

    def test_vector_gather_and_several_workers_give_the_same_rows(self) -> None:
        cpu = torch.device("cpu")
        spans_list = self.spans + [[(10, 10), (4999, 5000), (0, 3)], [(2000, 2000)]]
        sources = {"evidence": self.mm, "array": self.data, "features": self.tensor, "none": None}
        for workers in (1, 3):
            prefetcher = RowPrefetcher(sources, workers=workers, gather="vector")
            futures = [prefetcher.submit(spans) for spans in spans_list]
            for spans, future in zip(spans_list, futures):
                got = future.result()
                self.assertIsNone(got["none"])
                for name, rows in (("evidence", self.mm), ("array", self.data), ("features", self.tensor)):
                    want = runner.prior_rows_for_spans(rows, spans, cpu)
                    have = rows_to_device(got[name], cpu)
                    self.assertEqual(have.dtype, want.dtype, name)
                    self.assertTrue(torch.equal(have, want), (workers, name, spans))
            prefetcher.close()

    def test_take_span_rows_matches_gather_span_rows(self) -> None:
        for rows in (self.mm, self.data, self.tensor, None):
            for spans in self.spans:
                a, b = gather_span_rows(rows, spans), take_span_rows(rows, spans)
                if rows is None:
                    self.assertIsNone(b)
                    continue
                a = a if isinstance(a, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(a))
                b = b if isinstance(b, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(b))
                self.assertTrue(torch.equal(a, b))

    def test_refusals(self) -> None:
        with self.assertRaises(ValueError):
            RowPrefetcher({}, gather="other")
        with self.assertRaises(ValueError):
            RowPrefetcher({}, gather="spans", pin=True)
        with self.assertRaises(ValueError):
            RowPrefetcher({}, workers=0)

    def test_dropping_pages_keeps_values(self) -> None:
        dropper = PageDropper(self.mm)
        dropper.drop([(0, 5000)])
        np.testing.assert_array_equal(np.asarray(self.mm), self.data)
        dropper.close()
        self.assertIs(gather_span_rows(None, [(0, 1)]), None)

    def test_dropper_refuses_a_non_memmap(self) -> None:
        with self.assertRaises(ValueError):
            PageDropper(self.data)


if __name__ == "__main__":
    unittest.main()
