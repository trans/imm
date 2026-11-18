"""Documents (--documents): the training and held-out texts are runs of
pieces, each from another book, with nothing between them. With documents
on, a piece is read from its start: every carried state resets there, and
the memory forgets the piece before.

Positions are prediction positions: position p reads character p and
predicts character p + 1. A piece starting at character s > 0 therefore
begins at position s - 1, which predicts its first character from the
previous piece's last one; a piece at the text's start begins at position 0.
That one stray character gets a segment of its own, and the piece's own
text starts a new segment, so no segment spans two pieces."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

Segment = tuple[int, int]


def piece_starts(manifest: Path | str, train_len: int, eval_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Each piece's first character within its own text, from a pieces
    manifest (mobius scripts/pg19-split.cr): its train_records, then its
    chunk_records, which start at train_size. The pieces must tile each
    text from its start; pieces past a text's end (a shortened text) are
    left out."""
    m = json.loads(Path(manifest).read_text())
    if m.get("mode") != "pieces":
        raise ValueError(f"{manifest}: --documents needs a pieces manifest, got mode {m.get('mode')!r}")
    out = []
    for key, base, length in (("train_records", 0, train_len), ("chunk_records", int(m["train_size"]), eval_len)):
        spans = sorted((int(r["start"]) - base, int(r["end"]) - base) for r in m[key])
        if not spans or spans[0][0] != 0 or any(a[1] != b[0] for a, b in zip(spans, spans[1:])):
            raise ValueError(f"{manifest}: {key} don't tile the text from its start")
        if spans[-1][1] < length:
            raise ValueError(f"{manifest}: {key} end at {spans[-1][1]}, before the text's {length} characters")
        out.append(np.array([a for a, _ in spans if a < length], dtype=np.int64))
    return out[0], out[1]


def first_positions(starts: np.ndarray) -> np.ndarray:
    """The position each piece begins at: start - 1, or 0 at the text's start."""
    return np.maximum(np.asarray(starts, dtype=np.int64) - 1, 0)


def cut_segments(segments: list[Segment], starts: np.ndarray) -> list[Segment]:
    """segments, cut at each piece's first position and at its first own
    character, so that both start a segment."""
    cuts = sorted({int(p) for s in starts if s > 0 for p in (s - 1, s)})
    out: list[Segment] = []
    i = 0
    for a, b in segments:
        while i < len(cuts) and cuts[i] <= a:
            i += 1
        at = a
        while i < len(cuts) and cuts[i] < b:
            out.append((at, cuts[i]))
            at = cuts[i]
            i += 1
        out.append((at, b))
    return out


def piece_windows(segments: list[Segment], starts: np.ndarray) -> list[tuple[int, int]]:
    """For each piece, the segment indices lo..hi - 1 whose positions are its
    own: from the segment at its first position to the next piece's. The
    segments must be cut (cut_segments) and in order."""
    seg_starts = np.array([a for a, _ in segments], dtype=np.int64)
    first = first_positions(starts)
    lo = np.searchsorted(seg_starts, first)
    if (lo >= len(segments)).any() or (seg_starts[np.minimum(lo, len(segments) - 1)] != first).any():
        raise ValueError("piece_windows: a piece's first position doesn't start a segment (cut the segments first)")
    hi = np.append(lo[1:], len(segments))
    return [(int(a), int(b)) for a, b in zip(lo, hi) if b > a]


def text_prefix(segments: list[Segment], starts: np.ndarray, chars: int) -> tuple[list[Segment], np.ndarray]:
    """The training text's first `chars` characters, for a data-size test
    that keeps the whole text (and the prior rows made for it) loaded: the
    segments that end by then, and the pieces that begin before the last of
    them ends. The last piece kept is cut short at a segment boundary. The
    segments must be cut (cut_segments) and in order."""
    if chars < 1:
        raise ValueError("text_prefix needs chars >= 1")
    kept = [s for s in segments if s[1] <= chars]
    if not kept:
        raise ValueError(f"no segment ends within the first {chars} characters")
    starts = np.asarray(starts, dtype=np.int64)
    return kept, starts[first_positions(starts) < kept[-1][1]]


def aligned_streams(
    segments: list[Segment], starts: np.ndarray, streams: int, multiple: int
) -> tuple[list[list[Segment]], np.ndarray, int]:
    """Training streams that reset only between updates: each piece's
    segments, its last ones dropped to make a multiple of `multiple` (the
    segments of an update), one piece after another, then cut into `streams`
    contiguous runs, each a multiple of `multiple` long. So in every stream
    each piece begins at an update's first segment.

    Returns the streams; resets [streams, updates] bool, whether stream b's
    update u begins a piece (a stream that begins mid-piece doesn't); and the
    number of characters dropped."""
    if streams < 1 or multiple < 1:
        raise ValueError("aligned_streams needs streams >= 1 and multiple >= 1")
    kept: list[Segment] = []
    begins: list[int] = []
    for lo, hi in piece_windows(segments, starts):
        n = (hi - lo) // multiple * multiple
        if n:
            begins.append(len(kept))
            kept.extend(segments[lo : lo + n])
    per = len(kept) // streams // multiple * multiple
    if per == 0:
        raise ValueError(f"{len(kept)} aligned segments can't make {streams} streams of {multiple}-segment updates")
    begin = np.zeros(len(kept), dtype=bool)
    begin[begins] = True
    out = [kept[b * per : (b + 1) * per] for b in range(streams)]
    resets = np.stack([begin[b * per : (b + 1) * per : multiple] for b in range(streams)])
    used = sum(e - s for segs in out for s, e in segs)
    return out, resets, sum(e - s for s, e in segments) - used
