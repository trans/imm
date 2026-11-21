"""Slot 63's knockouts, paired per held-out target against the full model:
word GRU off, memory attention off, both off. For each bucket, the change in
perplexity (exp of the mean paired difference in nats, minus 1) ± the
standard error over targets, and the interaction: both off minus the two
single knockouts. Negative means overlap (losing both costs less than the
sum); positive means each needs the other. losses[p] is target p + 1's.

    python ko_buckets.py IMM_REPO KO_DIR [PREFIX]   (PREFIX mean_: the mean ablations)
"""
import math, re, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from imm.segments import rule_segments, phrase_classes, phrase_end_segments

K = sys.argv[2]; P = sys.argv[3] if len(sys.argv) > 3 else ""
import os
ARMS = [a for a in ("full", "noword", "nomem", "neither") if os.path.exists(f"{K}/char_losses_{P}{a}.npz")]
arms = {a: np.load(f"{K}/char_losses_{P}{a}.npz") for a in ARMS}  # without "neither": no both-off or overlap column
L = {a: z["losses"].astype(np.float64) for a, z in arms.items()}
A = arms["full"]
ids = A["text"].astype(np.int64); vocab = [str(c) for c in A["vocab"]]
text = "".join(vocab[i] for i in ids)
n = len(L["full"]); pos = np.arange(n); tgt = ids[1:]
OFF = [a for a in ("noword", "nomem", "neither") if a in L]
D = {a: L[a] - L["full"] for a in OFF}
inter = D["neither"] - D["noword"] - D["nomem"] if "neither" in D else None

sp = [i for i, c in enumerate(vocab) if c.isspace()]
segs, _ = rule_segments("word", ids, ids[:2], 16, 1, sp)
starts = np.array([s for s, _ in segs]); stops = np.array([e for _, e in segs])
seg = np.searchsorted(starts, pos, side="right") - 1
off = pos - starts[seg]; last = off == stops[seg] - starts[seg] - 1; first = off == 0
pe = phrase_end_segments(ids, segs, phrase_classes(vocab, ",;:.!?"))[seg]
se = phrase_end_segments(ids, segs, phrase_classes(vocab, ".!?"))[seg]
tc = np.array([vocab[t] for t in tgt])

# a target's word, and words since that word last occurred in the same 5,000-character piece
t = pos + 1
dist = np.full(n + 1, -1); in_word = np.zeros(n + 1, bool)
lastseen, piece, k = {}, -1, 0
for m in re.finditer(r"[A-Za-z0-9']+", text):
    s, e = m.span(); w = m.group().lower().strip("'")
    if s // 5000 != piece: lastseen, piece = {}, s // 5000
    in_word[s:e] = True
    if w in lastseen: dist[s:e] = k - lastseen[w]
    lastseen[w] = k; k += 1
dt, iw = dist[t], in_word[t]
cap = np.char.isupper(tc)

def pct(x, m):
    return 100 * np.expm1(x[m].mean()), 100 * x[m].std(ddof=1) / math.sqrt(m.sum())

NAMES = {"noword": "word GRU off", "nomem": "memory off", "neither": "both off"}
print(f"{'':50} {'targets':>8} {'full':>7}   " + "   ".join(f"{NAMES[a]:>14}" for a in OFF) + (f"   {'overlap':>8}" if inter is not None else ""))
def row(name, m):
    if m.sum() < 100: return
    cells = "   ".join("{:+6.2f}% ±{:.2f}".format(*pct(D[a], m)) for a in OFF)
    tail = f"   {100*np.expm1(inter[m].mean()):+7.2f}%" if inter is not None else ""
    print(f"{name:50} {m.sum():8d} {math.exp(L['full'][m].mean()):7.4f}   {cells}{tail}")

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
row("an uppercase letter", cap)
row("a space or newline", np.char.isspace(tc))
row("punctuation or digit", ~np.char.isalpha(tc) & ~np.char.isspace(tc))
print("-- the target's word, earlier in the same 5,000-character piece")
row("a new word in this piece", iw & (dt < 0))
for lo, hi in [(1, 16), (16, 64), (64, 256), (256, 10**9)]:
    label = f"{lo}-{hi}" if hi < 10**9 else f"{lo} or more"
    row(f"a repeat, {label} words back", iw & (dt >= lo) & (dt < hi))
row("  ... its capital letter, 16-64 words back", iw & (dt >= 16) & (dt < 64) & cap)
row("  ... its capital letter, 64 or more back", iw & (dt >= 64) & cap)
