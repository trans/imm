"""Reference points for the tiny GRUs: counting models that see only the last
n-1 characters, trained on the same first 2M training characters and scored
the same way (characters 2..5,000 of each held-out piece, context kept inside
the piece). Interpolated absolute discounting, d = 0.75, over an add-one
unigram. Prints perplexity per character for n = 1..6.

  python3 ngram_ref.py > results/ngram_ref.txt"""

import json
import math
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data/pg19/imm-split-100m"
TRAIN_CHARS, MAX_N, D = 2_000_000, 6, 0.75

vocab = (DATA / "vocab.txt").read_text()
with open(DATA / "train_corpus.txt") as f:
    train = f.read(TRAIN_CHARS)
held = (DATA / "heldout_corpus.txt").read_text()
pieces = [held[i : i + 5000] for i in range(0, len(held), 5000)]

gram = defaultdict(int)  # context + next -> count
total = defaultdict(int)  # context -> count
types = defaultdict(int)  # context -> distinct nexts
for k in range(MAX_N):
    for i in range(k, len(train)):
        key = train[i - k : i + 1]
        if gram[key] == 0:
            types[key[:-1]] += 1
        gram[key] += 1
        total[key[:-1]] += 1


def prob(context: str, c: str, n: int) -> float:
    p = (gram.get(c, 0) + 1) / (len(train) + len(vocab))  # order 1, add-one
    for k in range(1, n):
        if k > len(context):
            break
        ctx = context[-k:]
        t = total.get(ctx, 0)
        if t == 0:
            break
        p = max(gram.get(ctx + c, 0) - D, 0) / t + D * types[ctx] / t * p
    return p


rows = []
for n in range(1, MAX_N + 1):
    nats, count = 0.0, 0
    for piece in pieces:
        for j in range(1, len(piece)):
            nats -= math.log(prob(piece[max(0, j - n + 1) : j], piece[j], n))
            count += 1
    rows.append({"n": n, "context_chars": n - 1, "heldout_ppl": round(math.exp(nats / count), 4), "predictions": count})
    print(json.dumps(rows[-1]), flush=True)
