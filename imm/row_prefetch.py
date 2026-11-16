"""Prior rows for the next update, read while the current one runs.

Training reads, for every update, a span of prior rows per stream: log-probs,
gate features, and depth evidence. The depth-16 evidence is a 10 GB memmap,
larger than the page cache can keep beside everything else, so every epoch
reads it from disk again. On a machine short of memory that read also pushes
other processes into swap, and it stalls the update that waits for it.

RowPrefetcher gathers the next update's spans in a background thread (numpy
copies and page faults run without the GIL), so the read overlaps the GPU
work and the Python of the current update. The rows are the same bytes as a
direct read: results don't change.

With drop_cache, pages of a memmap, or of compact evidence's parts, are
dropped from the page cache once their rows are read, so a file read
straight through each epoch doesn't crowd out other processes' memory. It
only makes sense for files too big to stay cached. A source named there that
has no file behind it (rows in memory) is refused, not passed over. Until
2026-11-16 compact evidence was passed over, and a drop covered only its
span's own pages, which in folios larger than a span mostly stayed
(drop_pages).

Faster, the same bytes (2026-11-13): gather="vector" decodes compact
evidence for all spans in a few numpy calls instead of a decode per span,
and writes every source straight into its output buffer; workers > 1 lets several updates' reads run at once (the training
loop keeps that many submitted ahead); pin=True gathers into pinned host
memory, so the copy to the GPU can be non-blocking and the training thread
doesn't wait for the GPU to drain before it.

Faster again, the same bytes (2026-11-14): with compact=True, compact depth
evidence is read compact (take_spans_compact: about 7 MB an update at 64
streams, against 72 MB of dense rows) and rows_to_device rebuilds the dense
rows on the device. The host no longer zeroes and scatters 62 GB of dense
rows an epoch.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import torch

from imm.compact_evidence import CompactSpans, drop_pages, ragged_arange



def gather_span_rows(rows, spans: list[tuple[int, int]]):
    """Rows start..end - 1 of each span, the spans one after another, where the
    rows live (a numpy array for a memmap, a tensor for a tensor)."""
    if rows is None:
        return None
    if isinstance(rows, torch.Tensor):
        return torch.cat([rows[start:end] for start, end in spans])
    return np.concatenate([np.asarray(rows[start:end]) for start, end in spans])


def take_span_rows(rows, spans: list[tuple[int, int]], pin: bool = False, compact: bool = False):
    """gather_span_rows's rows: a numpy array or memmap by block copies of
    its spans, compact evidence through its take_spans (a few numpy calls
    whatever the number of spans), a tensor by index_select. With pin, the
    rows land straight in a pinned torch tensor (the same bytes), so the
    copy to the GPU can be non-blocking. With compact, compact evidence
    comes back as CompactSpans, which rows_to_device expands."""
    if rows is None:
        return None
    if isinstance(rows, torch.Tensor):
        idx = ragged_arange(*span_bounds(spans))
        return rows.index_select(0, torch.from_numpy(idx))
    if compact and hasattr(rows, "take_spans_compact"):
        return rows.take_spans_compact(spans, pin=pin)
    if hasattr(rows, "take_spans"):
        shape = (sum(b - a for a, b in spans),) + tuple(rows.shape[1:])
        if not pin:
            return rows.take_spans(spans)
        out = torch.empty(shape, dtype=torch.float16, pin_memory=True)
        rows.take_spans(spans, out=out.numpy())
        return out
    blocks = [rows[a:b] for a, b in spans]  # views of a memmap: the copy below reads them
    if not pin:
        return np.concatenate(blocks)
    n = sum(b - a for a, b in spans)
    out = torch.empty((n,) + tuple(rows.shape[1:]), dtype=TORCH_DTYPES[np.dtype(rows.dtype)], pin_memory=True)
    np.concatenate(blocks, out=out.numpy())
    return out


def span_bounds(spans: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    return (np.array([a for a, _ in spans], dtype=np.int64), np.array([b for _, b in spans], dtype=np.int64))


TORCH_DTYPES = {np.dtype(np.float16): torch.float16, np.dtype(np.float32): torch.float32,
                np.dtype(np.float64): torch.float64, np.dtype(np.int64): torch.int64}


def rows_to_device(rows, device: torch.device, non_blocking: bool = False) -> torch.Tensor | None:
    if rows is None:
        return None
    if isinstance(rows, CompactSpans):
        return rows.expand(device, non_blocking=non_blocking and rows.counts.is_pinned())
    if isinstance(rows, torch.Tensor):
        return rows.to(device, non_blocking=non_blocking and rows.is_pinned())
    return torch.from_numpy(rows).to(device)


class PageDropper:
    """Drops the pages under rows start..end - 1 of a memmap from the page
    cache, reaching back of start too (drop_pages): unmapped from this
    process (madvise), then evicted (fadvise)."""

    def __init__(self, rows: np.memmap) -> None:
        if not isinstance(rows, np.memmap) or rows.offset != 0 or getattr(rows, "_mmap", None) is None:
            raise ValueError("PageDropper needs a memmap of a whole file (offset 0)")
        self.mm = rows._mmap
        self.row_bytes = int(rows.strides[0])
        self.size = len(self.mm)
        self.fd = os.open(rows.filename, os.O_RDONLY)

    def drop(self, spans: list[tuple[int, int]]) -> None:
        for start, end in spans:
            drop_pages(self.mm, self.fd, start * self.row_bytes, end * self.row_bytes)

    def close(self) -> None:
        os.close(self.fd)


class CompactPageDropper:
    """PageDropper for compact evidence: drops the pages of each of its
    parts (ptr, shallow index, pos, val) under the rows read."""

    def __init__(self, rows) -> None:
        self.rows = rows

    def drop(self, spans: list[tuple[int, int]]) -> None:
        for start, end in spans:
            self.rows.drop_rows(start, end)

    def close(self) -> None:
        pass


def page_dropper(name: str, rows):
    """The dropper for a source: refused if it has no file to drop."""
    if isinstance(rows, np.memmap):
        return PageDropper(rows)
    if hasattr(rows, "drop_rows"):
        return CompactPageDropper(rows)
    raise ValueError(f"drop_cache: {name} has no file behind it ({type(rows).__name__}), so no pages to drop")


class ReadAhead:
    """Asks the kernel to read each stream's next rows ahead, in large reads.

    A training update reads one span per stream, the streams spread through
    each file; the kernel's own readahead sees one file read at 64 places and
    reads a few KB per page fault (measured 2026-11-14 on the 20M split:
    7,000 reads a second of 10 KB, 73 MB/s, the disk 58% busy, about the
    rate the one-block model needed from it). advise(spans) asks for each
    span's next rows_ahead to 2 * rows_ahead rows whenever fewer than
    rows_ahead are asked for beyond it (span i is stream i), with
    posix_fadvise WILLNEED, which queues large reads and returns; the page
    faults then find the rows cached. The rows are the same."""

    def __init__(self, sources: dict, rows_ahead: int) -> None:
        if rows_ahead < 1:
            raise ValueError("rows_ahead must be at least 1")
        self.rows_ahead = rows_ahead
        self.targets = []
        for rows in sources.values():
            if isinstance(rows, np.memmap) and rows.offset == 0 and rows.filename:
                self.targets.append(("memmap", rows, os.open(rows.filename, os.O_RDONLY)))
            elif hasattr(rows, "advise_rows"):
                self.targets.append(("compact", rows, None))
        self.until: dict[int, int] = {}
        self.lock = threading.Lock()

    def advise(self, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """The row ranges asked for (for tests)."""
        asked = []
        with self.lock:
            for i, (start, end) in enumerate(spans):
                until = self.until.get(i, start)
                if until < end + self.rows_ahead:
                    lo, hi = max(until, start), end + 2 * self.rows_ahead
                    self.until[i] = hi
                    asked.append((lo, hi))
        for lo, hi in asked:
            for kind, rows, fd in self.targets:
                if kind == "memmap":
                    row_bytes = int(rows.strides[0])
                    a, b = lo * row_bytes, min(hi, rows.shape[0]) * row_bytes
                    if b > a:
                        os.posix_fadvise(fd, a, b - a, os.POSIX_FADV_WILLNEED)
                else:
                    rows.advise_rows(lo, hi)
        return asked

    def close(self) -> None:
        for kind, _, fd in self.targets:
            if fd is not None:
                os.close(fd)


class RowPrefetcher:
    """submit(spans) starts reading every source's rows for those spans in the
    background; the Future's result is {name: rows on the host}. With one
    worker, one read at a time, in submission order; with more, that many
    reads at once (each Future still holds its own spans' rows). gather is
    "spans" (gather_span_rows, a slice per span) or "vector"
    (take_span_rows); pin puts memmap rows in pinned memory (vector only);
    compact reads compact evidence compact, for rows_to_device to expand
    (vector only); readahead_rows > 0 asks the kernel for each stream's
    next rows ahead (ReadAhead)."""

    def __init__(
        self,
        sources: dict,
        drop_cache: tuple[str, ...] = (),
        workers: int = 1,
        gather: str = "spans",
        pin: bool = False,
        compact: bool = False,
        readahead_rows: int = 0,
    ) -> None:
        if gather not in ("spans", "vector"):
            raise ValueError(f"unknown gather: {gather}")
        if pin and gather != "vector":
            raise ValueError("pin needs gather='vector'")
        if compact and gather != "vector":
            raise ValueError("compact needs gather='vector'")
        if workers < 1:
            raise ValueError("workers must be at least 1")
        self.sources = sources
        self.gather = gather
        self.pin = pin
        self.compact = compact
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rows")
        self.readahead = ReadAhead(sources, readahead_rows) if readahead_rows > 0 else None
        self.droppers = {name: page_dropper(name, sources[name]) for name in drop_cache if sources.get(name) is not None}

    def _read(self, spans: list[tuple[int, int]]) -> dict:
        if self.readahead is not None:
            self.readahead.advise(spans)
        out = {}
        for name, rows in self.sources.items():
            if self.gather == "vector":
                out[name] = take_span_rows(rows, spans, pin=self.pin, compact=self.compact)
            else:
                out[name] = gather_span_rows(rows, spans)
            if name in self.droppers:
                self.droppers[name].drop(spans)
        return out

    def submit(self, spans: list[tuple[int, int]]) -> Future:
        return self.pool.submit(self._read, spans)

    def close(self) -> None:
        self.pool.shutdown(wait=True)
        if self.readahead is not None:
            self.readahead.close()
        for dropper in self.droppers.values():
            dropper.close()
