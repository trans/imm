"""Which pages of a file are in the page cache (mincore), for tests that
check pages are really dropped. A file system that keeps every page
(tmpfs) can't show it: evicts() says whether this one can."""

import ctypes
import mmap
import os
from pathlib import Path

import numpy as np

_libc = ctypes.CDLL(None, use_errno=True)
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]


def resident(path: Path) -> np.ndarray:
    """[pages] bool: each page of the file in the page cache now."""
    size = os.path.getsize(path)
    n = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
    if n == 0:
        return np.zeros(0, dtype=bool)
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = _libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if addr in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_errno(), "mmap failed")
        try:
            vec = (ctypes.c_ubyte * n)()
            if _libc.mincore(addr, size, vec) != 0:
                raise OSError(ctypes.get_errno(), "mincore failed")
            return (np.frombuffer(vec, dtype=np.uint8) & 1).astype(bool)
        finally:
            _libc.munmap(addr, size)
    finally:
        os.close(fd)


def settle(path: Path) -> None:
    """Write the file's pages out (fsync), so dropping them can evict them,
    then fault every page in through a read-only mapping, the way rows are
    read: the cache then holds the file in the folios a training read makes."""
    with open(path, "rb+") as f:
        os.fsync(f.fileno())
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        if os.path.getsize(path):
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                for offset in range(0, len(mm), mmap.PAGESIZE):
                    mm[offset]
            finally:
                mm.close()


def evicts(directory: Path) -> bool:
    """Whether this file system drops a clean file's pages on fadvise."""
    probe = Path(directory) / "evicts.probe"
    probe.write_bytes(b"\0" * (4 * mmap.PAGESIZE))
    try:
        settle(probe)
        fd = os.open(probe, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        return not resident(probe).any()
    finally:
        probe.unlink()
