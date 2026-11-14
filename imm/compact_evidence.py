"""Depth evidence in a compact layout, read back as the same dense rows.

Dense depth evidence (depth_evidence_chunk's layout: rows x depth x
(vocab + 1) float16) is mostly zeros. On mobius's PG-19 evidence a row holds
about 155 nonzero values out of 1,536, and depths 1-3 hold about 107 of
them. At depth 16 dense rows cost 3,104 bytes a character, 62 GB at 20M
characters. This layout stores the same values in two parts:

- shallow: depths 1..S of a row depend only on its last S characters and its
  cross-fitting fold, so few distinct blocks occur. Each distinct block is
  stored once in a table, and each row keeps its block's index.
- deep: depths S+1..D are stored sparse, as compressed rows: an offset per
  row, then each nonzero's position in the row's deep part and its value.

Reading a slice rebuilds the dense float16 rows bit for bit, so training on
either layout sees the same numbers. The table is found from the data, not
from the counts, so edge cases (rows near a text's start, or a join between
pieces) are kept exactly.

Files, for split in (train, eval), next to PREFIX.json, little-endian:

    PREFIX.<split>_ev_shallow_index.dat  uint32 [rows]            (S > 0)
    PREFIX.<split>_ev_shallow_table.dat  float16 [entries, S, W]  (S > 0)
    PREFIX.<split>_ev_ptr.dat            int64 [rows + 1]: row r's nonzeros are ptr[r]..ptr[r+1]-1
    PREFIX.<split>_ev_pos.dat            uint16 [nnz]: (depth - S - 1) * W + column, increasing within a row
    PREFIX.<split>_ev_val.dat            float16 [nnz]

PREFIX.json carries the dense json's digests and vocabulary plus
"compact_evidence": {"schema": "depth_evidence_compact_v1", "depth": D,
"width": W, "shallow": S, "train": {"rows", "entries", "nnz"}, "eval": {...}}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCHEMA = "depth_evidence_compact_v1"
PARTS = ("shallow_index", "shallow_table", "ptr", "pos", "val")


def part_path(prefix: Path, split: str, part: str) -> Path:
    return prefix.parent / f"{prefix.name}.{split}_ev_{part}.dat"


def _hash_weights(width: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20261111)
    return (
        rng.integers(1, 2**63, size=width, dtype=np.uint64) | np.uint64(1),
        rng.integers(1, 2**63, size=width, dtype=np.uint64) | np.uint64(1),
    )


def convert_split(dense: np.ndarray, prefix: Path, split: str, shallow: int, block: int = 50_000) -> dict:
    """Write dense rows (rows, D, W) float16 in the compact layout; returns
    {"rows", "entries", "nnz"}. Reads dense a block at a time."""
    rows, depth, width = dense.shape
    if dense.dtype != np.float16:
        raise ValueError(f"dense evidence must be float16, got {dense.dtype}")
    if not 0 <= shallow <= depth:
        raise ValueError(f"shallow {shallow} outside 0..{depth}")
    if (depth - shallow) * width > np.iinfo(np.uint16).max + 1:
        raise ValueError(f"deep part {(depth - shallow) * width} wide doesn't fit uint16 positions")
    w1, w2 = _hash_weights(shallow * width)
    seen: dict[tuple[int, int], int] = {}
    nnz = 0
    files = {part: open(part_path(prefix, split, part), "wb") for part in PARTS if shallow or not part.startswith("shallow")}
    try:
        np.zeros(1, dtype=np.int64).tofile(files["ptr"])
        for a in range(0, rows, block):
            b = min(rows, a + block)
            x = np.ascontiguousarray(dense[a:b])
            n = b - a
            if shallow:
                s = x[:, :shallow].reshape(n, -1)
                u = s.view(np.uint16).astype(np.uint64)
                keys = np.stack([(u * w1).sum(axis=1), (u * w2).sum(axis=1)], axis=1)  # wraps mod 2**64
                unique, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
                entry = np.empty(len(unique), dtype=np.uint32)
                for j, key in enumerate(map(tuple, unique.tolist())):
                    e = seen.get(key)
                    if e is None:
                        e = len(seen)
                        seen[key] = e
                        files["shallow_table"].write(s[first[j]].tobytes())
                    entry[j] = e
                entry[inverse.ravel()].tofile(files["shallow_index"])
            deep = x[:, shallow:].reshape(n, -1)
            r, c = np.nonzero(deep.view(np.uint16))  # by bits, so -0.0 is kept; row-major: by row, then position
            (nnz + np.cumsum(np.bincount(r, minlength=n))).astype(np.int64).tofile(files["ptr"])
            c.astype(np.uint16).tofile(files["pos"])
            deep[r, c].astype(np.float16).tofile(files["val"])
            nnz += len(r)
    finally:
        for f in files.values():
            f.close()
    return {"rows": rows, "entries": len(seen), "nnz": nnz}


class CompactEvidenceRows:
    """One split's compact evidence, sliced like the dense memmap: rows[a:b]
    is a float16 array (b - a, depth, width), the same bytes as the dense
    rows. The shallow table is read into memory; the rest stays mapped."""

    def __init__(self, prefix: Path, split: str, section: dict) -> None:
        self.depth = int(section["depth"])
        self.width = int(section["width"])
        self.shallow = int(section["shallow"])
        info = section[split]
        self.rows = int(info["rows"])
        self.shape = (self.rows, self.depth, self.width)
        self.dtype = np.dtype(np.float16)
        self.ptr = np.memmap(part_path(prefix, split, "ptr"), dtype=np.int64, mode="r", shape=(self.rows + 1,))
        nnz = int(info["nnz"])
        if int(self.ptr[-1]) != nnz:
            raise ValueError(f"{prefix} {split}: ptr ends at {int(self.ptr[-1])}, json says {nnz} nonzeros")
        self.pos = np.memmap(part_path(prefix, split, "pos"), dtype=np.uint16, mode="r", shape=(nnz,)) if nnz else np.zeros(0, np.uint16)
        self.val = np.memmap(part_path(prefix, split, "val"), dtype=np.float16, mode="r", shape=(nnz,)) if nnz else np.zeros(0, np.float16)
        self.index = None
        self.table = None
        if self.shallow:
            entries = int(info["entries"])
            self.index = np.memmap(part_path(prefix, split, "shallow_index"), dtype=np.uint32, mode="r", shape=(self.rows,))
            self.table = np.fromfile(part_path(prefix, split, "shallow_table"), dtype=np.float16).reshape(
                entries, self.shallow, self.width
            )
        for path in [part_path(prefix, split, p) for p in PARTS if self.shallow or not p.startswith("shallow")]:
            if not path.exists():
                raise FileNotFoundError(path)

    def __len__(self) -> int:
        return self.rows

    def __getitem__(self, key) -> np.ndarray:
        if not isinstance(key, slice):
            raise TypeError("compact evidence is read by slices of rows")
        start, stop, step = key.indices(self.rows)
        if step != 1:
            raise ValueError("compact evidence slices need step 1")
        n = max(0, stop - start)
        out = np.zeros((n, self.depth, self.width), dtype=np.float16)
        if n == 0:
            return out
        if self.shallow:
            out[:, : self.shallow] = self.table[np.asarray(self.index[start:stop])]
        ptr = np.asarray(self.ptr[start : stop + 1])
        p0, p1 = int(ptr[0]), int(ptr[-1])
        if p1 > p0:
            local = np.repeat(np.arange(n), np.diff(ptr))
            flat = out.reshape(n, self.depth * self.width)
            flat[local, np.asarray(self.pos[p0:p1]).astype(np.int64) + self.shallow * self.width] = self.val[p0:p1]
        return out

    def take_spans(self, spans: list[tuple[int, int]], out: np.ndarray | None = None) -> np.ndarray:
        """Rows start..end - 1 of each span, the spans one after another: the
        same bytes as concatenating self[start:end] over the spans, in a
        few numpy calls whatever the number of spans. Written into out
        (float16, (rows, depth, width)) if given."""
        starts = np.array([a for a, _ in spans], dtype=np.int64)
        ends = np.array([b for _, b in spans], dtype=np.int64)
        if ((starts < 0) | (ends > self.rows) | (ends < starts)).any():
            raise IndexError("compact evidence spans must lie within the rows")
        idx = ragged_arange(starts, ends)
        n = len(idx)
        shape = (n, self.depth, self.width)
        if out is None:
            out = np.zeros(shape, dtype=np.float16)
        else:
            if out.shape != shape or out.dtype != np.float16 or not out.flags.c_contiguous:
                raise ValueError(f"out must be C-contiguous float16 {shape}, got {out.dtype} {out.shape}")
            out.fill(0)
        if n == 0:
            return out
        if self.shallow:
            out[:, : self.shallow] = np.take(self.table, np.take(self.index, idx), axis=0)
        nz_starts = np.take(self.ptr, starts)
        nz_ends = np.take(self.ptr, ends)
        if int((nz_ends - nz_starts).sum()):
            # A span's nonzeros are one contiguous run of pos and val: block
            # copies, not a per-element gather.
            pos = np.concatenate([self.pos[a:b] for a, b in zip(nz_starts.tolist(), nz_ends.tolist())])
            val = np.concatenate([self.val[a:b] for a, b in zip(nz_starts.tolist(), nz_ends.tolist())])
            counts = np.take(self.ptr, idx + 1) - np.take(self.ptr, idx)
            local = np.repeat(np.arange(n, dtype=np.int64), counts)
            out.reshape(-1)[local * (self.depth * self.width) + pos.astype(np.int64) + self.shallow * self.width] = val
        return out

    def take_spans_compact(self, spans: list[tuple[int, int]], pin: bool = False) -> "CompactSpans":
        """take_spans' rows, still compact: each row's shallow-table index,
        each row's count of deep nonzeros, and the nonzeros' positions and
        values, spans one after another, as torch tensors (pinned with pin).
        CompactSpans.expand rebuilds take_spans' bytes where the tensors are,
        so the dense rows (depth x width per row, mostly zeros) are written
        and copied on the GPU, not on the host."""
        starts = np.array([a for a, _ in spans], dtype=np.int64)
        ends = np.array([b for _, b in spans], dtype=np.int64)
        if ((starts < 0) | (ends > self.rows) | (ends < starts)).any():
            raise IndexError("compact evidence spans must lie within the rows")
        idx = ragged_arange(starts, ends)
        n = len(idx)
        index = None
        if self.shallow:
            index = torch.empty(n, dtype=torch.int64, pin_memory=pin)
            index.numpy()[:] = np.take(self.index, idx)
        counts = torch.empty(n, dtype=torch.int64, pin_memory=pin)
        if n:
            np.subtract(np.take(self.ptr, idx + 1), np.take(self.ptr, idx), out=counts.numpy())
        nz_starts = np.take(self.ptr, starts).tolist()
        nz_ends = np.take(self.ptr, ends).tolist()
        nnz = int(sum(b - a for a, b in zip(nz_starts, nz_ends)))
        # Positions are below depth * width (at most 16 * 97 here), so int16
        # holds them; torch has few uint16 operations.
        if self.depth * self.width > np.iinfo(np.int16).max:
            raise ValueError("compact positions must fit int16 for take_spans_compact")
        pos = torch.empty(nnz, dtype=torch.int16, pin_memory=pin)
        val = torch.empty(nnz, dtype=torch.float16, pin_memory=pin)
        if nnz:
            runs = [(a, b) for a, b in zip(nz_starts, nz_ends) if b > a]
            np.concatenate([self.pos[a:b] for a, b in runs], out=pos.numpy().view(np.uint16))
            np.concatenate([self.val[a:b] for a, b in runs], out=val.numpy())
        return CompactSpans(self, n, index, counts, pos, val)

    def device_table(self, device: torch.device) -> torch.Tensor | None:
        """The shallow table as a tensor on device, made once per device."""
        if self.table is None:
            return None
        cache = self.__dict__.setdefault("_device_tables", {})
        key = str(device)
        if key not in cache:
            cache[key] = torch.from_numpy(self.table).to(device)
        return cache[key]

    def dense(self) -> np.ndarray:
        return self[0 : self.rows]


class CompactSpans:
    """One read of compact evidence rows (take_spans_compact): n rows, their
    shallow-table indices, their deep nonzero counts, the nonzeros' positions
    and values."""

    def __init__(self, source: CompactEvidenceRows, n: int, index, counts, pos, val) -> None:
        self.source = source
        self.n = n
        self.index = index
        self.counts = counts
        self.pos = pos
        self.val = val

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.n, self.source.depth, self.source.width)

    def expand(self, device: torch.device, non_blocking: bool = False) -> torch.Tensor:
        """The dense float16 rows (n, depth, width) on device: the bytes
        take_spans gives for the same spans. Positions within a row are
        distinct, so the scatter has one value per place."""
        src = self.source
        width, shallow = src.width, src.shallow
        out = torch.zeros(self.shape, dtype=torch.float16, device=device)
        if self.n == 0:
            return out
        if shallow:
            index = self.index.to(device, non_blocking=non_blocking)
            out[:, :shallow] = src.device_table(device).index_select(0, index)
        nnz = int(self.pos.shape[0])
        if nnz:
            counts = self.counts.to(device, non_blocking=non_blocking)
            local = torch.repeat_interleave(torch.arange(self.n, device=device), counts, output_size=nnz)
            flat = local * (src.depth * width) + self.pos.to(device, non_blocking=non_blocking).long() + shallow * width
            out.view(-1)[flat] = self.val.to(device, non_blocking=non_blocking)
        return out


def ragged_arange(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """The integers starts[i]..ends[i] - 1 for each i, one run after another
    (int64)."""
    lengths = ends - starts
    total = int(lengths.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    run_start = np.cumsum(lengths) - lengths  # where each run begins in the output
    return np.repeat(starts - run_start, lengths) + np.arange(total, dtype=np.int64)


def load_compact(prefix: Path, meta: dict) -> dict[str, CompactEvidenceRows]:
    section = meta.get("compact_evidence")
    if not isinstance(section, dict) or section.get("schema") != SCHEMA:
        raise ValueError(f"{prefix}: no {SCHEMA} section in its json")
    return {split: CompactEvidenceRows(prefix, split, section) for split in ("train", "eval")}


def verify_split(dense: np.ndarray, compact: CompactEvidenceRows, block: int = 50_000) -> int:
    """Rows where the compact layout doesn't rebuild the dense bytes exactly
    (0 when it does), over every row."""
    if tuple(dense.shape) != compact.shape:
        raise ValueError(f"shapes differ: dense {tuple(dense.shape)}, compact {compact.shape}")
    bad = 0
    for a in range(0, compact.rows, block):
        b = min(compact.rows, a + block)
        x = np.asarray(dense[a:b]).view(np.uint16)
        y = compact[a:b].view(np.uint16)
        bad += int((x != y).reshape(b - a, -1).any(axis=1).sum())
    return bad


def dense_memmap(prefix: Path, meta: dict, split: str, width: int) -> np.memmap:
    """The dense split named by the json's depth_evidence.<split> (path and
    shape, as mobius writes them)."""
    section = meta.get("depth_evidence") if isinstance(meta.get("depth_evidence"), dict) else {}
    entry = section.get(split) or {}
    if entry.get("shape") is None:
        raise ValueError(f"{prefix}: the json gives no depth_evidence.{split}.shape")
    rows, depth, w = (int(v) for v in entry["shape"])
    if w != width:
        raise ValueError(f"{prefix}: {split} rows are {w} wide, the vocabulary makes {width}")
    named = entry.get("path")
    path = prefix.parent / named if named else prefix.parent / f"{prefix.name}.{split}_depth_evidence.dat"
    return np.memmap(path, dtype=np.float16, mode="r", shape=(rows, depth, w))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    conv = sub.add_parser("convert", help="write a dense evidence prefix in the compact layout")
    conv.add_argument("dense_prefix", type=Path, help="PREFIX of PREFIX.json and PREFIX.<split>_depth_evidence.dat")
    conv.add_argument("out_prefix", type=Path)
    conv.add_argument("--shallow", type=int, default=3, help="depths 1..S kept as a table of distinct blocks")
    conv.add_argument("--block", type=int, default=50_000)
    conv.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True, help="check every row after")
    args = parser.parse_args(argv)

    meta = json.loads(args.dense_prefix.with_name(args.dense_prefix.name + ".json").read_text())
    width = len(meta["vocab_chars"]) + 1
    out_meta = {k: v for k, v in meta.items() if k not in ("depth_evidence", "log_probs", "gate_features")}
    section: dict = {"schema": SCHEMA, "shallow": args.shallow, "width": width, "source": str(args.dense_prefix)}
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for split in ("train", "eval"):
        dense = dense_memmap(args.dense_prefix, meta, split, width)
        section["depth"] = int(dense.shape[1])
        t0 = time.perf_counter()
        info = convert_split(dense, args.out_prefix, split, args.shallow, args.block)
        section[split] = info
        sizes = sum(
            part_path(args.out_prefix, split, p).stat().st_size
            for p in PARTS
            if args.shallow or not p.startswith("shallow")
        )
        print(
            f"{split}: rows={info['rows']} entries={info['entries']} nnz={info['nnz']} "
            f"nnz_per_row={info['nnz'] / max(1, info['rows']):.1f} bytes={sizes} "
            f"bytes_per_row={sizes / max(1, info['rows']):.0f} dense_bytes={dense.nbytes} "
            f"ratio={dense.nbytes / max(1, sizes):.2f} sec={time.perf_counter() - t0:.0f}",
            flush=True,
        )
    out_meta["compact_evidence"] = section
    args.out_prefix.with_name(args.out_prefix.name + ".json").write_text(json.dumps(out_meta, indent=2) + "\n")
    if args.verify:
        compact = load_compact(args.out_prefix, out_meta)
        for split in ("train", "eval"):
            t0 = time.perf_counter()
            bad = verify_split(dense_memmap(args.dense_prefix, meta, split, width), compact[split], args.block)
            print(f"verify {split}: rows_differing={bad} sec={time.perf_counter() - t0:.0f}", flush=True)
            if bad:
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
