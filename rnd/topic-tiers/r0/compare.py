"""R0 against slot 63's usual score, paired per held-out target, by the
target's place in its 5,000-character piece. A: the usual 64 streams, each
warmed up on 256 segments (which can come from the piece before); B: one
stream per piece from its start. losses[p] is target p + 1's.

    python compare.py A.npz B.npz
"""
import math, sys
import numpy as np
a, b = (np.load(f) for f in sys.argv[1:3])
assert (a["positions"] == b["positions"]).all()
la, lb = a["losses"].astype(np.float64), b["losses"].astype(np.float64)
t = a["positions"] + 1
place = t % 5000
d = lb - la
print(f"{'target place in its piece':28} {'targets':>8} {'usual':>7} {'by piece':>8} {'change':>16}")
for lo, hi in [(0, 5000), (0, 16), (16, 64), (64, 256), (256, 1000), (1000, 2500), (2500, 5000)]:
    m = (place >= lo) & (place < hi)
    se = 100 * d[m].std(ddof=1) / math.sqrt(m.sum())
    print(f"{'all' if (lo, hi) == (0, 5000) else f'{lo}-{hi}':28} {m.sum():8d} {math.exp(la[m].mean()):7.4f} {math.exp(lb[m].mean()):8.4f} "
          f"{100 * np.expm1(d[m].mean()):+8.2f}% ±{se:.2f}")
