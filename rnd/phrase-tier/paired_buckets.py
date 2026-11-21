"""Paired differences per bucket, phrase vs word (and word vs base), with
the standard error of the mean per-character loss difference."""
import math, sys
import numpy as np
sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
from tier_position_losses import words_since, load
from imm.segments import phrase_classes, phrase_end_segments, rule_segments
R = "/home/trans/Projects/air/imm/runs/char_losses_%s_best.npz"
runs = {k: load(R % v) for k, v in dict(base="pg19s20m_h256_par_l2_e20", word="pg19s20m_h256_par_l2_w256_e20",
        phrase="pg19s20m_h256_par_l2_w256_p256_e20", word41="pg19s20m_h256_par_l2_w256_e41").items()}
text = runs["base"]["text"].astype(np.int64); vocab = [str(c) for c in runs["base"]["vocab"]]; pos = runs["base"]["positions"]
segs, _ = rule_segments("word", text, text[:2], 16, 1, [i for i, c in enumerate(vocab) if c.isspace()])
starts = np.array([a for a, _ in segs]); stops = np.array([b for _, b in segs])
seg = np.searchsorted(starts, pos, side="right") - 1
at_end = pos - starts[seg] == stops[seg] - starts[seg] - 1
first = pos == starts[seg]
pe = phrase_end_segments(text, segs, phrase_classes(vocab, ",;:.!?")); se = phrase_end_segments(text, segs, phrase_classes(vocab, ".!?"))
ip, isn = words_since(pe)[seg], words_since(se)[seg]
B = {
 "all": np.ones(len(pos), bool),
 "word first char": first, "inside word": ~first & ~at_end, "word last char (predicts next word's 1st)": at_end,
 "phrase 1st word, not last char": (ip == 0) & ~at_end,
 "phrase 2nd+ word, not last char": (ip > 0) & ~at_end,
 "sentence 1st word, not last char": (isn == 0) & ~at_end,
 "sentence 2nd-3rd word, not last char": (isn >= 1) & (isn <= 2) & ~at_end,
 "sentence 4th+ word, not last char": (isn >= 3) & ~at_end,
 "phrase-ending word, last char (predicts next phrase's 1st)": pe[seg] & at_end,
 "sentence-ending word, last char (predicts next sentence's 1st)": se[seg] & at_end,
 "word last char, not ending a phrase": ~pe[seg] & at_end,
}
def row(a, b, m):
    d = runs[a]["losses"][m].astype(np.float64) - runs[b]["losses"][m].astype(np.float64)
    return 100 * (math.exp(d.mean()) - 1), 100 * d.std(ddof=1) / math.sqrt(m.sum())
print(f"{'':64}{'chars':>7}  {'phrase vs word':>18}  {'word vs base':>18}")
for name, m in B.items():
    a = row("phrase", "word", m); b = row("word", "base", m)
    print(f"{name:64}{int(m.sum()):7d}  {a[0]:+7.2f}% ±{a[1]:.2f}  {b[0]:+7.2f}% ±{b[1]:.2f}")
