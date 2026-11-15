"""Where in a word, phrase and sentence each model's held-out losses fall.

Reads the .npz files run_segment_memory_model.py writes with
--char-loss-output (each held-out character's loss and the held-out text),
cuts the text into word segments as the word rule does, and splits the
losses:

- by where the prediction is made in a word: at its first character (of the
  second), inside it, or at its last (of the next word's first character,
  where the whitespace after a word predicts what follows);
- by the word's place in its phrase (words since the last segment ending in
  one of --phrase-marks) and in its sentence (since the last ending in one
  of --sentence-marks), each split again at the word's last position.

The first file is the baseline: every other is also shown as its change in
perplexity from it, bucket by bucket.

    python scripts/tier_position_losses.py base=a.npz word=b.npz phrase=c.npz [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imm.segments import phrase_classes, phrase_end_segments, rule_segments  # noqa: E402

PLACE_BUCKETS = ((0, 0, "1st"), (1, 1, "2nd"), (2, 2, "3rd"), (3, 5, "4th-6th"), (6, 10**9, "7th+"))


def words_since(ends: np.ndarray) -> np.ndarray:
    """For each segment, how many segments since the last that ended a unit
    (0: the first of its unit). The first segment counts from the text's
    start."""
    k = np.arange(len(ends))
    last_end = np.where(ends, k, -1)
    prev = np.concatenate([[-1], np.maximum.accumulate(last_end)[:-1]])
    return k - prev - 1


def load(path: Path) -> dict:
    with np.load(path) as z:
        return {name: z[name] for name in ("positions", "losses", "text", "vocab")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", metavar="LABEL=NPZ")
    parser.add_argument("--max-depth", type=int, default=16, help="the runs' --max-depth")
    parser.add_argument("--phrase-marks", default=",;:.!?")
    parser.add_argument("--sentence-marks", default=".!?")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    runs = []
    for item in args.runs:
        label, _, path = item.partition("=")
        if not path:
            parser.error(f"{item!r}: give LABEL=NPZ")
        runs.append((label, load(Path(path))))
    text, vocab = runs[0][1]["text"].astype(np.int64), [str(c) for c in runs[0][1]["vocab"]]
    for label, run in runs[1:]:
        if not np.array_equal(run["text"], runs[0][1]["text"]) or list(run["vocab"]) != vocab:
            raise SystemExit(f"{label}: a different held-out text or vocabulary from {runs[0][0]}")
    positions = runs[0][1]["positions"]
    for label, run in runs:
        if not np.array_equal(run["positions"], positions):
            raise SystemExit(f"{label}: scored different positions from {runs[0][0]}")

    space_ids = [i for i, c in enumerate(vocab) if c.isspace()]
    segments, _ = rule_segments("word", text, text[:2], args.max_depth, 1, space_ids)
    starts = np.array([a for a, _ in segments])
    stops = np.array([b for _, b in segments])
    seg = np.searchsorted(starts, positions, side="right") - 1
    if (seg < 0).any() or (positions >= stops[seg]).any():
        raise SystemExit("scored positions outside the word segments: is --max-depth the runs'?")
    offset = positions - starts[seg]
    length = stops[seg] - starts[seg]
    at_end = offset == length - 1
    where = np.where(at_end, "last", np.where(offset == 0, "first", "inside"))

    phrase_ends = phrase_end_segments(text, segments, phrase_classes(vocab, args.phrase_marks))
    sentence_ends = phrase_end_segments(text, segments, phrase_classes(vocab, args.sentence_marks))
    in_phrase = words_since(phrase_ends)[seg]
    in_sentence = words_since(sentence_ends)[seg]

    groups: dict[str, np.ndarray] = {"all": np.ones(len(positions), dtype=bool)}
    for name, tag in (("first", "at its first character"), ("inside", "inside it"), ("last", "at its last character")):
        groups[f"word: {tag}"] = where == name
    for unit, place in (("phrase", in_phrase), ("sentence", in_sentence)):
        for lo, hi, tag in PLACE_BUCKETS:
            in_bucket = (place >= lo) & (place <= hi)
            groups[f"{unit}: {tag} word, not its last character"] = in_bucket & ~at_end
            groups[f"{unit}: {tag} word, its last character"] = in_bucket & at_end
    groups["phrase: last word of a phrase, its last character"] = phrase_ends[seg] & at_end
    groups["sentence: last word of a sentence, its last character"] = sentence_ends[seg] & at_end

    table = {}
    for name, mask in groups.items():
        row = {"chars": int(mask.sum())}
        for label, run in runs:
            row[label] = math.exp(float(run["losses"][mask].astype(np.float64).mean())) if mask.any() else None
        table[name] = row
    result = {
        "runs": {label: str(path) for label, path in (item.partition("=")[::2] for item in args.runs)},
        "segments": len(segments),
        "words_per_phrase": len(segments) / max(int(phrase_ends.sum()), 1),
        "words_per_sentence": len(segments) / max(int(sentence_ends.sum()), 1),
        "phrase_marks": args.phrase_marks,
        "sentence_marks": args.sentence_marks,
        "ppl": table,
    }

    base = runs[0][0]
    labels = [label for label, _ in runs]
    print(
        f"{len(positions)} characters, {len(segments)} words; {result['words_per_phrase']:.2f} words a phrase "
        f"({args.phrase_marks!r}), {result['words_per_sentence']:.2f} a sentence ({args.sentence_marks!r})"
    )
    head = f"{'':52} {'chars':>7} " + " ".join(f"{label:>9}" for label in labels)
    head += "".join(f" {label + ' vs ' + base:>14}" for label in labels[1:])
    print(head)
    for name, row in table.items():
        cells = " ".join(f"{row[label]:9.4f}" if row[label] is not None else f"{'-':>9}" for label in labels)
        deltas = "".join(
            f" {100.0 * (row[label] / row[base] - 1.0):+13.2f}%" if row[label] and row[base] else f" {'-':>14}"
            for label in labels[1:]
        )
        print(f"{name:52} {row['chars']:7d} {cells}{deltas}")
    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
