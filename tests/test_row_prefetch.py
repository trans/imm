"""RowPrefetcher: the same rows as a direct read, from memmaps and tensors,
in submission order; dropping a memmap's pages from the cache changes no
values."""

import mmap
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from imm.row_prefetch import PageDropper, ReadAhead, RowPrefetcher, gather_span_rows, rows_to_device, take_span_rows

sys.path.insert(0, str(Path(__file__).resolve().parent))
from page_cache import evicts, resident, settle  # noqa: E402
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

    def test_readahead_asks_each_stream_ahead_in_large_pieces(self) -> None:
        ahead = ReadAhead({"evidence": self.mm, "array": self.data, "none": None}, rows_ahead=10)
        self.assertEqual(len(ahead.targets), 1)  # only the memmap has a file
        asked = []
        for u in range(8):  # two streams, 5 rows an update, the second near the end of the file
            asked.append(ahead.advise([(5 * u, 5 * u + 5), (4980 + 5 * u, 4985 + 5 * u)]))
        # Each stream is asked rows_ahead..2 * rows_ahead beyond its span, then
        # nothing until fewer than rows_ahead are left asked beyond it.
        self.assertEqual(asked[0], [(0, 25), (4980, 5005)])
        self.assertEqual(asked[1], [])
        self.assertEqual(asked[2], [])
        self.assertEqual(asked[3], [(25, 40), (5005, 5020)])
        self.assertEqual(asked[6], [(40, 55), (5020, 5035)])
        self.assertEqual(sum(len(a) for a in asked), 2 * 3)  # updates 0, 3 and 6
        ahead.close()
        with self.assertRaises(ValueError):
            ReadAhead({}, rows_ahead=0)

    def test_readahead_keeps_the_rows(self) -> None:
        cpu = torch.device("cpu")
        sources = {"evidence": self.mm, "features": self.tensor}
        for gather in ("spans", "vector"):
            prefetcher = RowPrefetcher(sources, workers=3, gather=gather, readahead_rows=50)
            futures = [prefetcher.submit(spans) for spans in self.spans]
            for spans, future in zip(self.spans, futures):
                got = future.result()
                for name, rows in (("evidence", self.mm), ("features", self.tensor)):
                    self.assertTrue(torch.equal(rows_to_device(got[name], cpu), runner.prior_rows_for_spans(rows, spans, cpu)))
            prefetcher.close()

    def test_dropper_refuses_a_non_memmap(self) -> None:
        with self.assertRaises(ValueError):
            PageDropper(self.data)

    def test_drop_cache_refuses_rows_with_no_file(self) -> None:
        # Rows in memory have no pages to drop: refused, not passed over (as
        # compact evidence was, before 2026-11-16). Absent rows are fine.
        for name, rows in (("features", self.tensor), ("array", self.data)):
            with self.assertRaisesRegex(ValueError, f"drop_cache: {name} has no file"):
                RowPrefetcher({name: rows}, drop_cache=(name,))
        prefetcher = RowPrefetcher({"evidence": self.mm, "none": None}, drop_cache=("evidence", "none", "missing"))
        self.assertEqual(sorted(prefetcher.droppers), ["evidence"])
        prefetcher.close()

    def test_dropped_pages_leave_the_cache(self) -> None:
        # One stream's spans, read one after another as training reads them
        # (about 70 KB each, smaller than the folios a mapped read makes):
        # every page 2 MB or more behind the last span read leaves the cache,
        # none past it does, and no value changes. Dropping only each span's
        # own pages, as before 2026-11-16, left most of them on this laptop.
        if not evicts(Path(self.tmp.name)):
            self.skipTest("this file system keeps pages: nothing to see")
        rows_bytes = 192  # a row of 96 float16 log-probs
        data = np.random.default_rng(1).standard_normal((44000, rows_bytes // 2)).astype(np.float16)
        path = Path(self.tmp.name) / "log_probs.dat"
        data.tofile(path)
        mm = np.memmap(path, dtype=np.float16, mode="r", shape=data.shape)
        settle(path)
        self.assertTrue(resident(path).all())
        prefetcher = RowPrefetcher({"rows": mm}, gather="vector", drop_cache=("rows",))
        last = 30000
        for a in range(0, last, 360):
            spans = [(a, min(a + 360, last))]
            got = prefetcher.submit(spans).result()["rows"]
            np.testing.assert_array_equal(np.asarray(got), data[spans[0][0] : spans[0][1]])
        prefetcher.close()
        cached = resident(path)
        b = last * rows_bytes
        behind = (b - (2 << 20)) // mmap.PAGESIZE
        self.assertFalse(cached[:behind].any(), f"{int(cached[:behind].sum())} of {behind} pages behind stayed")
        self.assertTrue(cached[-(-b // mmap.PAGESIZE) :].all())
        np.testing.assert_array_equal(np.asarray(mm), data)
        dropper = PageDropper(mm)
        dropper.drop([(0, len(data))])  # to the file's end: every page goes
        self.assertFalse(resident(path).any())
        dropper.close()
        del mm

if __name__ == "__main__":
    unittest.main()
