"""The longest run of 64 word segments in the 100M training text (within
5M-character chunks), with 2,000 characters before it, written to
s10/worst_window.txt for the probe's worst-case update."""
import sys
import numpy as np
sys.path.insert(0, ".")
from imm.segments import walk, word_lengths
best = (0, None)
with open("data/pg19/imm-split-100m/train_corpus.txt", encoding="utf-8") as f:
    while True:
        text = f.read(5_000_000)
        if not text:
            break
        ids = np.where(np.array([c.isspace() for c in text]), 0, 1).astype(np.int64)
        segs = walk(word_lengths(ids, [0], 16, np.ones(len(ids), dtype=np.int64)))
        starts = np.array([s for s, _ in segs]); ends = np.array([e for _, e in segs])
        span = ends[63:] - starts[: len(segs) - 63]
        i = int(span.argmax())
        if span[i] > best[0]:
            lo = max(0, int(starts[i]) - 2000)
            best = (int(span[i]), text[lo : int(ends[i + 63]) + 1])
print(f"longest 64-segment window: {best[0]} characters")
open(sys.argv[1], "w", encoding="utf-8").write(best[1])
