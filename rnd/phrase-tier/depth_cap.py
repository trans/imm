import sys, numpy as np
sys.path.insert(0, sys.argv[1])
from imm.segments import rule_segments, phrase_classes, phrase_end_segments, word_starts
with np.load(sys.argv[2]) as z:
    text = z["text"].astype(np.int64); vocab = [str(c) for c in z["vocab"]]
sp = [i for i, c in enumerate(vocab) if c.isspace()]
starts = word_starts(text, sp)
nat = np.diff(np.append(starts, len(text)))  # natural \S+\s* lengths
for d in (16, 24):
    segs, _ = rule_segments("word", text, text[:2], d, 1, sp)
    L = np.array([b - a for a, b in segs])
    print(f"depth {d}: {len(segs)} segments; words longer than {d}: {(nat > d).sum()} of {len(nat)} ({100*(nat>d).mean():.3f}%), chars in them {100*nat[nat>d].sum()/nat.sum():.3f}%; segments at the cap {(L==d).sum()}")
for marks in (",;:.!?", ".!?"):
    segs, _ = rule_segments("word", text, text[:2], 16, 1, sp)
    ends = phrase_end_segments(text, segs, phrase_classes(vocab, marks))
    stops = np.array([b for _, b in segs])[ends]
    span = np.diff(np.concatenate([[0], stops]))
    print(f"marks {marks!r}: units {len(span)}, mean {span.mean():.1f} chars, median {np.median(span):.0f}, >16: {100*(span>16).mean():.1f}%, >24: {100*(span>24).mean():.1f}%")
