"""Where imm's remaining loss is, and where it trails the transformer.

    python bottleneck.py IMM.npz TX.npz PRIOR_ROWS

Per held-out target (losses[p] scores text[p + 1]), paired: imm against the
transformer (256-character window, exact characters) and the d24 prior
alone. Targets in a word ([A-Za-z0-9']+) are split by the letter's place in
the word (k = 0 is its first letter) and by whether the word (lowercased)
occurred before in the same 5,000-character piece, and how many characters
back its last occurrence started: within the transformer's window (< 256,
where it can copy the letters exactly) or beyond it.

Columns: targets; share of imm's total loss (nats); imm, transformer and
prior perplexity; imm's change against the transformer (exp of the mean
paired difference, minus 1) ± its standard error over targets.
"""
import math, re, sys
import numpy as np

A = np.load(sys.argv[1]); B = np.load(sys.argv[2]); R = sys.argv[3]
la, lb = A["losses"].astype(np.float64), B["losses"].astype(np.float64)
text = A["text"].astype(np.int64); vocab = [str(c) for c in A["vocab"]]
n = len(la); pos = np.arange(n); tgt = text[1:]
LP = np.memmap(R + ".eval_log_probs.dat", dtype=np.float16, mode="r").reshape(-1, len(vocab))
lp = -LP[pos, tgt].astype(np.float64)
s = "".join(vocab[i] for i in text)

T = n + 1
k = np.full(T, -1); wlen = np.zeros(T, int); back = np.full(T, -1)  # back: characters since the word's last start, -1 new
seen, piece = {}, -1
for m in re.finditer(r"[A-Za-z0-9']+", s):
    a, b = m.span(); w = m.group().lower()
    if a // 5000 != piece: seen, piece = {}, a // 5000
    k[a:b] = np.arange(b - a); wlen[a:b] = b - a
    if w in seen: back[a:b] = a - seen[w]
    seen[w] = a
t = pos + 1
K, L, BK = k[t], wlen[t], back[t]
total = la.sum(); d = la - lb


def row(name, m):
    if m.sum() < 200:
        return
    dm = d[m]
    se = dm.std(ddof=1) / math.sqrt(m.sum())
    print(f"{name:46} {m.sum():7d} {100 * la[m].sum() / total:6.1f}%  {math.exp(la[m].mean()):7.3f} {math.exp(lb[m].mean()):7.3f} "
          f"{math.exp(lp[m].mean()):7.3f}   {100 * math.expm1(dm.mean()):+6.2f}% ±{100 * se:.2f}")


print(f"{'':46} {'targets':>7} {'share':>7}  {'imm':>7} {'tx':>7} {'prior':>7}   imm vs tx")
row("all", np.ones(n, bool))
inw = K >= 0
row("not in a word (space, punctuation)", ~inw)
print("-- the letter's place in its word")
for kk, label in [(0, "first letter (which word comes next)"), (1, "second letter"), (2, "third letter"), (3, "fourth letter on")]:
    row(label, (K == kk) if kk < 3 else (K >= 3))
groups = [("new in the piece", BK < 0), ("repeat, < 256 chars back (in tx's window)", (BK >= 0) & (BK < 256)),
          ("repeat, 256-1,000 back", (BK >= 256) & (BK < 1000)), ("repeat, 1,000+ back", BK >= 1000)]
for gname, g in groups:
    print(f"-- {gname}")
    row("  all its letters", inw & g)
    row("  first letter", inw & g & (K == 0))
    row("  second letter", inw & g & (K == 1))
    row("  third letter on", inw & g & (K >= 2))
    row("  third letter on, words of 6+ letters", inw & g & (K >= 2) & (L >= 6))
    row("  third letter on, prior >= 3 nats", inw & g & (K >= 2) & (lp >= 3))
print("-- the letters only copying can give: third on, prior >= 3 nats")
for gname, g in groups:
    row(f"  {gname}", inw & g & (K >= 2) & (lp >= 3))
