"""How many characters a stream reads an update under the standard run
(--segment-rule word, --max-depth 16, --train-streams 64,
--segments-per-update 64, --documents), and how often each tree level
steps in one, on the first CHARS characters of the 100m split's training
text (its pieces that start there). Run from the imm-tree worktree:
    sh $S/tree/env.sh python3 -B $S/tree/steps_per_update.py [CHARS]"""
import sys

import numpy as np

sys.path.insert(0, ".")
from imm.data import CharVocab, read_text  # noqa: E402
from imm.documents import aligned_streams, cut_segments, piece_starts  # noqa: E402
from imm.segments import walk, word_lengths  # noqa: E402

CHARS = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000
root = "data/pg19/imm-split-100m/"
vocab = CharVocab.from_text(read_text(root + "vocab.txt"))
with open(root + "train_corpus.txt", encoding="utf-8") as f:
    text = f.read(CHARS)
ids = np.asarray(vocab.encode(text), dtype=np.int64)
space = [i for i, ch in enumerate(vocab.chars) if ch.isspace()]
segments = walk(word_lengths(ids, space, 16, np.ones(len(ids), dtype=np.int64)))
starts, _ = piece_starts(root + "manifest.json", len(ids), 1)
segments = cut_segments(segments, starts)
lengths = np.array([e - s for s, e in segments])
print(f"{len(ids):,} characters, {len(starts)} pieces, {len(segments):,} segments, mean {lengths.mean():.3f} characters")
streams, resets, dropped = aligned_streams(segments, starts, 64, 64)
updates = resets.shape[1]
per = np.array([[sum(e - s for s, e in segs[64 * u : 64 * u + 64]) for u in range(updates)] for segs in streams])
longest = per.max(axis=0)
print(
    f"{updates} updates of 64 streams: a stream's characters an update mean {per.mean():.1f}, "
    f"sd {per.std():.1f}, 1st-99th percentile {np.percentile(per, 1):.0f}-{np.percentile(per, 99):.0f}; "
    f"an update's longest stream mean {longest.mean():.0f}, max {longest.max()}; "
    f"{per.sum(axis=0).mean():,.0f} characters an update in all"
)
for k in range(1, 11):
    size = 2**k
    counts = []
    for b in range(64):
        phase = 0
        for u in range(updates):
            if resets[b, u]:
                phase = 0
            n = (phase + per[b, u]) // size - phase // size
            counts.append(n)
            phase += per[b, u]
    counts = np.array(counts)
    shares = ", ".join(f"{j}: {np.mean(counts == j):.0%}" for j in range(0, 4) if np.mean(counts == j) > 0)
    print(f"level {k:2d} (every {size:4d}): steps an update mean {counts.mean():6.2f}; fewer than 2 in {np.mean(counts < 2):4.0%} ({shares}{', more' if counts.max() > 3 else ''})")
