"""How much the single character d back tells about the next character, on
the first 2M training characters: the mutual information between the two
(nats, less what a shuffled pairing gives by chance), and the perplexity per
character a model would get knowing only that one character. Pairs that
cross a piece's start are kept (under 5% at d = 1,024).

  python3 pair_info.py > results/pair_info.txt"""

import math
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data/pg19/imm-split-100m"
TRAIN_CHARS = 2_000_000

vocab = (DATA / "vocab.txt").read_text()
index = {c: i for i, c in enumerate(vocab)}
with open(DATA / "train_corpus.txt") as f:
    x = np.array([index[c] for c in f.read(TRAIN_CHARS)], dtype=np.int64)
V = len(vocab)
p = np.bincount(x, minlength=V) / len(x)
H = -(p[p > 0] * np.log(p[p > 0])).sum()
rng = np.random.default_rng(0)


def mi(a, b):
    joint = np.bincount(a * V + b, minlength=V * V).reshape(V, V) / len(a)
    outer = np.outer(joint.sum(1), joint.sum(0))
    m = joint > 0
    return float((joint[m] * np.log(joint[m] / outer[m])).sum())


print(f"unigram perplexity {math.exp(H):.2f}")
print("d\tinfo_nats\tperplexity_from_that_char")
for d in [1, 2, 3, 4, 5, 6, 8, 12, 16, 32, 64, 128, 256, 512, 1024]:
    a, b = x[:-d], x[d:]
    info = mi(a, b) - mi(rng.permutation(a), b)
    print(f"{d}\t{info:.4f}\t{math.exp(H - info):.2f}")
