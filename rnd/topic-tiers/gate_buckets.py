"""The sentence tier's per-position factor (gate_map.py) by bucket: where the
prediction is made, what is predicted, the word's place in its sentence, the
target's place in its 5,000-character piece. A factor of 1 is the tier's add
unchanged (the start); 0 shuts it; 2 doubles it.

    python gate_buckets.py IMM_REPO GATE_MAP.npz CHAR_LOSSES.npz
"""
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from imm.segments import rule_segments, phrase_classes, phrase_end_segments

G = np.load(sys.argv[2]); A = np.load(sys.argv[3])
ids = A["text"].astype(np.int64); vocab = [str(c) for c in A["vocab"]]
n = len(ids) - 1
f = np.full(n, np.nan); keep = G["positions"] < n
f[G["positions"][keep]] = G["factors"][keep]
pos = np.arange(n); tgt = ids[1:]
sp = [i for i, c in enumerate(vocab) if c.isspace()]
segs, _ = rule_segments("word", ids, ids[:2], 16, 1, sp)
starts = np.array([s for s, _ in segs]); stops = np.array([e for _, e in segs])
seg = np.searchsorted(starts, pos, side="right") - 1
off = pos - starts[seg]; last = off == stops[seg] - starts[seg] - 1; first = off == 0
pe = phrase_end_segments(ids, segs, phrase_classes(vocab, ",;:.!?"))[seg]
se_seg = phrase_end_segments(ids, segs, phrase_classes(vocab, ".!?"))
se = se_seg[seg]
# the word's place in its sentence: segments since the last sentence end
since = np.zeros(len(segs), np.int64); c = 0
for k in range(len(segs)):
    since[k] = c; c = 0 if se_seg[k] else c + 1
wis = since[seg]
tc = np.array([vocab[t] for t in tgt])
place = (pos + 1) % 5000

print(f"{'':52} {'targets':>8} {'mean factor':>12} {'sd':>6} {'< 0.5':>6} {'> 1.5':>6}")
def row(name, m):
    m = m & np.isfinite(f)
    if m.sum() < 50: return
    x = f[m]
    print(f"{name:52} {m.sum():8d} {x.mean():12.3f} {x.std():6.3f} {100*(x < 0.5).mean():5.1f}% {100*(x > 1.5).mean():5.1f}%")
row("all", np.ones(n, bool))
print("-- place in the word (where the prediction is made)")
row("at a word's first character", first & ~last)
row("inside a word", ~first & ~last)
row("at a word's last (predicting the next word's first)", last)
row("  ... not ending a phrase", last & ~pe)
row("  ... ending a phrase", last & pe)
row("  ... ending a sentence", last & se)
print("-- what is predicted")
row("a lowercase letter", np.char.islower(tc))
row("an uppercase letter", np.char.isupper(tc))
row("a space or newline", np.char.isspace(tc))
row("punctuation or digit", ~np.char.isalpha(tc) & ~np.char.isspace(tc))
print("-- the word's place in its sentence")
for lo, hi in [(0, 1), (1, 4), (4, 12), (12, 10**9)]:
    row(f"word {lo + 1}" + (f"-{hi}" if hi - lo > 1 and hi < 10**9 else "+" if hi >= 10**9 else ""), (wis >= lo) & (wis < hi))
print("-- the target's place in its piece")
for lo, hi in [(0, 256), (256, 1000), (1000, 2500), (2500, 5000)]:
    row(f"chars {lo}-{hi}", (place >= lo) & (place < hi))
