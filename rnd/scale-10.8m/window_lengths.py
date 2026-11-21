"""Characters in each run of 64 word segments across the 100M training
text, in 5M-character chunks (so under 1 GB): how long an update's longest
stream can get. The probe's 13 updates had a longest stream of ~500."""
import sys
import numpy as np
sys.path.insert(0, ".")
from imm.segments import walk, word_lengths
path = "data/pg19/imm-split-100m/train_corpus.txt"
V = 96
f = open(path, encoding="utf-8")
sums, carry = [], np.zeros(0, dtype=np.int64)
while True:
    text = f.read(5_000_000)
    if not text:
        break
    sp = np.array([c.isspace() for c in text])
    ids = np.where(sp, 0, 1).astype(np.int64)
    segs = walk(word_lengths(ids, [0], 16, np.ones(len(ids), dtype=np.int64)))
    lens = np.fromiter((e - b for b, e in segs), dtype=np.int64, count=len(segs))
    lens = np.concatenate([carry, lens])
    n = len(lens) // 64 * 64
    sums.append(lens[:n].reshape(-1, 64).sum(1))
    carry = lens[n:]
s = np.concatenate(sums)
q = np.percentile(s, [50, 99, 99.9, 99.99, 100])
print(f"windows={len(s):,} median={q[0]:.0f} p99={q[1]:.0f} p99.9={q[2]:.0f} p99.99={q[3]:.0f} max={q[4]:.0f}")
for x in (500, 600, 700, 800, 900, 1000):
    k = int((s > x).sum())
    print(f"  > {x}: {k:,} windows ({k / len(s):.4%}); per update of 64 streams, P(longest > {x}) ~ {1 - (1 - k / len(s)) ** 64:.1%}")
