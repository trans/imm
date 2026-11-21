import sys, numpy as np, collections
sys.path.insert(0, sys.argv[1])
from imm.segments import word_starts
with np.load(sys.argv[2]) as z:
    text = z["text"].astype(np.int64); vocab = [str(c) for c in z["vocab"]]
sp = [i for i, c in enumerate(vocab) if c.isspace()]
starts = word_starts(text, sp); ends = np.append(starts[1:], len(text))
s = "".join(vocab[i] for i in text)
long = [(a, b) for a, b in zip(starts, ends) if b - a > 16]
spc = np.isin(text, sp)
ws = sum(int(spc[a:b].sum()) for a, b in long); tot = sum(b - a for a, b in long)
print(f"{len(long)} long words, {tot} chars, of which whitespace {ws} ({100*ws/tot:.0f}%)")
core = [s[a:b].rstrip() for a, b in long]
print("non-space part longer than 16:", sum(len(c) > 16 for c in core))
for c in [c for c in core if len(c) > 16][:15]: print(repr(c))
