"""imm against the transformer, paired per held-out character, by bucket."""
import math, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from imm.segments import rule_segments, phrase_classes, phrase_end_segments
import os
R = os.environ.get("PRIOR_ROWS", "/home/trans/Projects/air/mobius/runs/imm/pg19split20m_gated_d16_pg200m")
F = np.fromfile(R + ".eval_gate_features.dat", dtype=np.float32).reshape(-1, 11)
LP = np.memmap(R + ".eval_log_probs.dat", dtype=np.float16, mode="r").reshape(-1, 96)
A = np.load(sys.argv[2]); B = np.load(sys.argv[3])  # A = imm, B = transformer
la, lb = A["losses"].astype(np.float64), B["losses"].astype(np.float64)
text = A["text"].astype(np.int64); vocab = [str(c) for c in A["vocab"]]; n = len(la)
pos = np.arange(n); tgt = text[1:]
prior_loss = -LP[pos, tgt].astype(np.float64)
H, depth, rel = F[:, 2], F[:, 3], F[:, 0]
sp = [i for i, c in enumerate(vocab) if c.isspace()]
segs, _ = rule_segments("word", text, text[:2], 16, 1, sp)
starts = np.array([s for s, _ in segs]); stops = np.array([e for _, e in segs])
seg = np.searchsorted(starts, pos, side="right") - 1
off = pos - starts[seg]; last = off == stops[seg] - starts[seg] - 1; first = off == 0
pe = phrase_end_segments(text, segs, phrase_classes(vocab, ",;:.!?"))[seg]
se = phrase_end_segments(text, segs, phrase_classes(vocab, ".!?"))[seg]
tc = np.array([vocab[t] for t in tgt])
d = la - lb
def row(name, m):
    m = m & np.isfinite(d)
    if m.sum() < 100: return
    pa, pb = math.exp(la[m].mean()), math.exp(lb[m].mean())
    se_ = d[m].std(ddof=1) / math.sqrt(m.sum())
    share = d[m].sum() / d.sum() * 100
    print(f"{name:58} {m.sum():7d} {100*m.mean():5.1f}%  imm {pa:8.4f}  tx {pb:8.4f}  {100*np.expm1(d[m].mean()):+6.2f}% ±{100*se_:.2f}  share of gap {share:+6.1f}%")
print(f"imm {math.exp(la.mean()):.4f}  tx {math.exp(lb.mean()):.4f}  gap {100*np.expm1(d.mean()):+.3f}% (nats {d.mean():+.5f})\n")
print("-- place in the word (where the prediction is made)")
row("at a word's first character", first & ~last)
row("inside a word", ~first & ~last)
row("at a word's last (predicting the next word's first)", last)
row("  ... ending a phrase", last & pe)
row("  ... ending a sentence", last & se)
print("-- what is predicted")
row("a lowercase letter", np.char.islower(tc))
row("an uppercase letter", np.char.isupper(tc))
row("a space or newline", np.char.isspace(tc))
row("punctuation or digit", ~np.char.isalpha(tc) & ~np.char.isspace(tc))
print("-- the prior's own loss on the target (nats)")
for lo, hi in ((0, 0.1), (0.1, 0.5), (0.5, 1.5), (1.5, 3), (3, 6), (6, 99)):
    row(f"prior loss in [{lo}, {hi})", (prior_loss >= lo) & (prior_loss < hi))
print("-- the prior's entropy (entropy_norm)")
q = np.quantile(H[H > 0], [0.25, 0.5, 0.75])
row("entropy 0", H == 0)
for lo, hi in ((0, q[0]), (q[0], q[1]), (q[1], q[2]), (q[2], 9)):
    row(f"entropy in ({lo:.3f}, {hi:.3f}]", (H > lo) & (H <= hi))
print("-- the prior's matched depth (depth_norm)")
for lo, hi in ((-1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.99), (0.99, 9)):
    row(f"depth_norm in ({lo}, {hi}]", (depth > lo) & (depth <= hi))
print("-- the transformer's context in its window (targets with < 256 back)")
row("first 256 characters of the text", pos < 255)
