"""imm (slot 63) against the 1.86M transformer, both on the d24 rows, paired
per held-out target, by (a) the target's place in its 5,000-character
held-out piece and (b) whether the target's word already occurred earlier in
the same piece. losses[p] is the loss of target p + 1."""
import math, re, sys
import numpy as np
A = np.load(sys.argv[1]); B = np.load(sys.argv[2])
la, lb = A["losses"].astype(np.float64), B["losses"].astype(np.float64)
vocab = [str(c) for c in A["vocab"]]
text = "".join(vocab[i] for i in A["text"].astype(np.int64))
import os
LP = np.memmap(os.environ.get("PRIOR_ROWS", "/home/trans/Projects/air/mobius/runs/imm/pg19split20m_gated_d24_pg10g") + ".eval_log_probs.dat", dtype=np.float16, mode="r").reshape(-1, 96)
ids = A["text"].astype(np.int64); n = len(la)
prior = -LP[np.arange(n), ids[1:]].astype(np.float64)
d = la - lb
t = np.arange(1, n + 1)               # target index of each loss
piece_pos = t % 5000
# word of each target char; repeat = same lowercase word earlier in the piece
word_id = np.full(n + 1, -1); repeat = np.zeros(n + 1, bool); first_char = np.zeros(n + 1, bool)
seen, cur_piece = set(), -1
for m in re.finditer(r"[A-Za-z0-9']+", text):
    s, e = m.span(); w = m.group().lower().strip("'")
    if s // 5000 != cur_piece: seen, cur_piece = set(), s // 5000
    repeat[s:e] = w in seen; word_id[s:e] = 1; first_char[s] = True
    seen.add(w)
in_word = word_id[t] >= 0; rep = repeat[t]
cap = np.array([c.isupper() for c in text])[t]
def row(name, m):
    if m.sum() < 100: return
    se = d[m].std(ddof=1) / math.sqrt(m.sum())
    print(f"{name:52} {m.sum():7d} {100*m.mean():5.1f}%  imm {math.exp(la[m].mean()):8.4f}  tx {math.exp(lb[m].mean()):8.4f}  {100*np.expm1(d[m].mean()):+6.2f}% ±{100*se:.2f}  share {d[m].sum()/d.sum()*100:+6.1f}%")
row("all", np.ones(n, bool))
print("-- the target's place in its 5,000-character held-out piece")
for lo, hi in [(0, 256), (256, 1000), (1000, 2500), (2500, 5000)]:
    row(f"chars {lo}-{hi} of the piece", (piece_pos >= lo) & (piece_pos < hi))
print("-- the target's word, earlier in the same piece")
row("a word's character, the word new in this piece", in_word & ~rep)
row("a word's character, the word seen before in this piece", in_word & rep)
row("  ... and capitalized", in_word & rep & cap)
row("not in a word (space, punctuation)", ~in_word)
print("-- where imm trails: the d24 prior's loss >= 3 nats")
hi = prior >= 3
row("prior >= 3, word new in this piece", hi & in_word & ~rep)
row("prior >= 3, word seen before in this piece", hi & in_word & rep)
row("prior >= 3, not in a word", hi & ~in_word)
row("prior >= 3, chars 0-1000 of the piece", hi & (piece_pos < 1000))
row("prior >= 3, chars 1000-5000 of the piece", hi & (piece_pos >= 1000))
print("-- a repeated word: words since its last occurrence in this piece")
dist = np.full(n + 1, -1)
last, cur_piece, k = {}, -1, 0
for m in re.finditer(r"[A-Za-z0-9']+", text):
    s, e = m.span(); w = m.group().lower().strip("'")
    if s // 5000 != cur_piece: last, cur_piece = {}, s // 5000
    if w in last: dist[s:e] = k - last[w]
    last[w] = k; k += 1
dt = dist[t]
for lo, hi in [(1, 16), (16, 64), (64, 256), (256, 10**9)]:
    row(f"repeat, {lo}-{hi if hi < 10**9 else 'more'} words back", in_word & (dt >= lo) & (dt < hi))
    row(f"  ... capitalized", in_word & (dt >= lo) & (dt < hi) & cap)
