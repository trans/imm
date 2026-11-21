"""Where the prior's entropy is high or rises along the held-out text, against
word boundaries and against where the word tier's gain falls."""
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
from imm.segments import rule_segments
R = "/home/trans/Projects/air/mobius/runs/imm/pg19split20m_gated_d16_pg200m"
F = np.fromfile(R + ".eval_gate_features.dat", dtype=np.float32).reshape(-1, 11)
names = ['reliability','kl_gain','entropy_norm','depth_norm','entropy_delta','suffix_mass_norm','suffix_reliability','suffix_entropy_norm','suffix_kl_gain','suffix_entropy_delta','bias']
H = F[:, 2].astype(np.float64); depth = F[:, 3]
runs = {k: np.load(v) for k, v in (a.split("=") for a in sys.argv[2:])}
base = next(iter(runs.values()))
text = base["text"].astype(np.int64); vocab = [str(c) for c in base["vocab"]]; pos = base["positions"]
print("text", len(text), "features", len(F), "scored", len(pos), pos.min(), pos.max())
sp = [i for i, c in enumerate(vocab) if c.isspace()]
segs, _ = rule_segments("word", text, text[:2], 16, 1, sp)
starts = np.array([a for a, _ in segs]); stops = np.array([b for _, b in segs])
seg = np.searchsorted(starts, np.arange(len(F)), side="right") - 1
off = np.arange(len(F)) - starts[seg]; last = off == (stops[seg] - starts[seg] - 1)
where = np.where(last, 2, np.where(off == 0, 0, 1))
rise = np.concatenate([[0.0], np.diff(H)])
for w, tag in enumerate(("word's first char", "inside", "word's last char")):
    m = where == w
    print(f"{tag:18} share {100*m.mean():5.1f}%  entropy_norm mean {H[m].mean():.3f}  rise mean {rise[m].mean():+.3f}  depth_norm {depth[m].mean():.3f}")
# cut rules: entropy level above q, or rise above q, set to a word-like mean unit
for label, sig in (("level", H), ("rise", rise)):
    for target in (5.69, 3.0, 9.0):
        q = np.quantile(sig, 1 - 1 / target)
        cut = sig > q
        prec = (cut & last).sum() / cut.sum(); rec = (cut & last).sum() / last.sum()
        near = cut & (last | np.roll(last, 1) | np.roll(last, -1))
        print(f"cut on {label:5} mean unit {1/cut.mean():5.2f}: {100*prec:4.1f}% of cuts at a word's last char ({100*near.sum()/cut.sum():4.1f}% within 1), {100*rec:4.1f}% of word ends cut")
# word-tier gain by entropy decile, split by place in word
if len(runs) >= 2:
    (la, a), (lb, b) = list(runs.items())[:2]
    assert np.array_equal(a["positions"], b["positions"])
    d = b["losses"].astype(np.float64) - a["losses"].astype(np.float64)
    Hp = H[pos]; wp = where[pos]
    qs = np.concatenate([[0.0], np.quantile(Hp[Hp > 0], np.linspace(0, 1, 5))])
    print(f"\n{lb} vs {la}: change in perplexity by the prior's entropy: 0, then quartiles of the rest")
    for w, tag in ((None, "all"), (0, "first"), (1, "inside"), (2, "last")):
        cells = []
        for i in range(5):
            m = ((Hp == 0) if i == 0 else (Hp > qs[i]) & (Hp <= qs[i + 1]) if i > 1 else (Hp > 0) & (Hp <= qs[2])) & (True if w is None else (wp == w))
            if m.sum() < 200: cells.append("     -        "); continue
            se = d[m].std() / np.sqrt(m.sum())
            cells.append(f"{100*np.expm1(d[m].mean()):+5.2f}±{100*se:4.2f} n{m.sum()//1000:>3}k")
        print(f"  {tag:7}", " | ".join(cells))
    print("  quintile edges", np.round(qs, 3))
