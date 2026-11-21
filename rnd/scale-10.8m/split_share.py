"""How often an update of 64 streams x 64 word segments has a stream longer
than L characters (so --max-update-chars 64 x L splits it): the 100M
training text's aligned 64-segment windows, in 5M-character chunks (under
1 GB), as window_lengths.py, at L from 384 to 768."""
import sys
import numpy as np
sys.path.insert(0, ".")
from imm.segments import walk, word_lengths
f = open("data/pg19/imm-split-100m/train_corpus.txt", encoding="utf-8")
sums, carry = [], np.zeros(0, dtype=np.int64)
while True:
    text = f.read(5_000_000)
    if not text:
        break
    ids = np.where(np.array([c.isspace() for c in text]), 0, 1).astype(np.int64)
    segs = walk(word_lengths(ids, [0], 16, np.ones(len(ids), dtype=np.int64)))
    lens = np.concatenate([carry, np.fromiter((e - b for b, e in segs), dtype=np.int64, count=len(segs))])
    n = len(lens) // 64 * 64
    sums.append(lens[:n].reshape(-1, 64).sum(1))
    carry = lens[n:]
s = np.concatenate(sums)
for L in (384, 448, 512, 576, 640, 704, 768):
    p = float((s > L).mean())
    print(f"L={L}: windows over {p:.4%}; updates split ~{1 - (1 - p) ** 64:.1%}")
