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
"""

from __future__ import annotations

import mmap
import os
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import torch

PAGE = mmap.PAGESIZE


def gather_span_rows(rows, spans: list[tuple[int, int]]):
    """Rows start..end - 1 of each span, the spans one after another, where the
    rows live (a numpy array for a memmap, a tensor for a tensor)."""
    if rows is None:
        return None
    if isinstance(rows, torch.Tensor):
        return torch.cat([rows[start:end] for start, end in spans])
    return np.concatenate([np.asarray(rows[start:end]) for start, end in spans])


def rows_to_device(rows, device: torch.device) -> torch.Tensor | None:
    if rows is None:
        return None
    if isinstance(rows, torch.Tensor):
        return rows.to(device)
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
    background; the Future's result is {name: rows on the host}. One read at a
    time, in submission order."""

    def __init__(self, sources: dict, drop_cache: tuple[str, ...] = ()) -> None:
        self.sources = sources
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rows")
        self.droppers = {
            name: PageDropper(sources[name]) for name in drop_cache if isinstance(sources.get(name), np.memmap)
        }

    def _read(self, spans: list[tuple[int, int]]) -> dict:
        out = {}
        for name, rows in self.sources.items():
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
