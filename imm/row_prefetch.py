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

With drop_cache, pages of a memmap are dropped from the page cache once their
rows are read, so a file read straight through each epoch doesn't crowd out
other processes' memory. It only makes sense for files too big to stay cached.

Faster, the same bytes (2026-11-13): gather="vector" decodes compact
evidence for all spans in a few numpy calls instead of a decode per span,
and writes every source straight into its output buffer; workers > 1 lets several updates' reads run at once (the training
loop keeps that many submitted ahead); pin=True gathers into pinned host
memory, so the copy to the GPU can be non-blocking and the training thread
doesn't wait for the GPU to drain before it.
"""

from __future__ import annotations

import mmap
import os
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import torch

from imm.compact_evidence import ragged_arange

PAGE = mmap.PAGESIZE


def gather_span_rows(rows, spans: list[tuple[int, int]]):
    """Rows start..end - 1 of each span, the spans one after another, where the
    rows live (a numpy array for a memmap, a tensor for a tensor)."""
    if rows is None:
        return None
    if isinstance(rows, torch.Tensor):
        return torch.cat([rows[start:end] for start, end in spans])
    return np.concatenate([np.asarray(rows[start:end]) for start, end in spans])


def take_span_rows(rows, spans: list[tuple[int, int]], pin: bool = False):
    """gather_span_rows's rows: a numpy array or memmap by block copies of
    its spans, compact evidence through its take_spans (a few numpy calls
    whatever the number of spans), a tensor by index_select. With pin, the
    rows land straight in a pinned torch tensor (the same bytes), so the
    copy to the GPU can be non-blocking."""
    if rows is None:
        return None
    if isinstance(rows, torch.Tensor):
        idx = ragged_arange(*span_bounds(spans))
        return rows.index_select(0, torch.from_numpy(idx))
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
    if isinstance(rows, torch.Tensor):
        return rows.to(device, non_blocking=non_blocking and rows.is_pinned())
    return torch.from_numpy(rows).to(device)


class PageDropper:
    """Drops the whole pages under rows start..end - 1 of a memmap from the page
    cache: unmapped from this process (madvise), then evicted (fadvise)."""

    def __init__(self, rows: np.memmap) -> None:
        if not isinstance(rows, np.memmap) or rows.offset != 0 or getattr(rows, "_mmap", None) is None:
            raise ValueError("PageDropper needs a memmap of a whole file (offset 0)")
        self.mm = rows._mmap
        self.row_bytes = int(rows.strides[0])
        self.size = len(self.mm)
        self.fd = os.open(rows.filename, os.O_RDONLY)

    def drop(self, spans: list[tuple[int, int]]) -> None:
        for start, end in spans:
            a = (start * self.row_bytes) // PAGE * PAGE
            b = min(self.size, (end * self.row_bytes) // PAGE * PAGE)  # the partial last page stays
            if b > a:
                self.mm.madvise(mmap.MADV_DONTNEED, a, b - a)
                os.posix_fadvise(self.fd, a, b - a, os.POSIX_FADV_DONTNEED)

    def close(self) -> None:
        os.close(self.fd)


class RowPrefetcher:
    """submit(spans) starts reading every source's rows for those spans in the
    background; the Future's result is {name: rows on the host}. With one
    worker, one read at a time, in submission order; with more, that many
    reads at once (each Future still holds its own spans' rows). gather is
    "spans" (gather_span_rows, a slice per span) or "vector"
    (take_span_rows); pin puts memmap rows in pinned memory (vector only)."""

    def __init__(
        self, sources: dict, drop_cache: tuple[str, ...] = (), workers: int = 1, gather: str = "spans", pin: bool = False
    ) -> None:
        if gather not in ("spans", "vector"):
            raise ValueError(f"unknown gather: {gather}")
        if pin and gather != "vector":
            raise ValueError("pin needs gather='vector'")
        if workers < 1:
            raise ValueError("workers must be at least 1")
        self.sources = sources
        self.gather = gather
        self.pin = pin
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rows")
        self.droppers = {
            name: PageDropper(sources[name]) for name in drop_cache if isinstance(sources.get(name), np.memmap)
        }

    def _read(self, spans: list[tuple[int, int]]) -> dict:
        out = {}
        for name, rows in self.sources.items():
            if self.gather == "vector":
                out[name] = take_span_rows(rows, spans, pin=self.pin)
            else:
                out[name] = gather_span_rows(rows, spans)
            if name in self.droppers:
                self.droppers[name].drop(spans)
        return out

    def submit(self, spans: list[tuple[int, int]]) -> Future:
        return self.pool.submit(self._read, spans)

    def close(self) -> None:
        self.pool.shutdown(wait=True)
        for dropper in self.droppers.values():
            dropper.close()
