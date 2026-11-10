"""Write a typo'd copy of a text: substitutions at a given rate, replacements
drawn from the unigram distribution of a reference text (the training text),
as --input-noise draws them during training. Beside it, a json with the
rate, the seed, how many characters changed, and both texts' digests.

    python scripts/make_typo_text.py --input data/.splits/<split>/heldout_corpus.txt \\
        --reference data/.splits/<split>/train_corpus.txt --vocab-input data/gutenberg_5m.txt \\
        --rate 0.02 --seed 1 --output runs/typos/heldout_typo2.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imm.data import CharVocab, read_text  # noqa: E402
from imm.typo_noise import substitute, unigram_probs  # noqa: E402


def digest(ids: list[int]) -> str:
    return hashlib.sha256(bytes(ids)).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="text whose character frequencies the replacements follow")
    parser.add_argument("--vocab-input", type=Path, required=True)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    vocab = CharVocab.from_text(read_text(args.vocab_input))
    ids = vocab.encode(read_text(args.input))
    probs = unigram_probs(vocab.encode(read_text(args.reference)), vocab.size)
    noisy, changed = substitute(ids, args.rate, probs, np.random.default_rng(args.seed))
    noisy_ids = [int(i) for i in noisy]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(vocab.chars[i] for i in noisy_ids), encoding="utf-8")
    meta = {
        "input": str(args.input),
        "reference": str(args.reference),
        "rate": args.rate,
        "seed": args.seed,
        "chars": len(ids),
        "chars_changed": changed,
        "input_digest": digest(ids),
        "output_digest": digest(noisy_ids),
        "noise": "substitutions, unigram replacements from the reference text, never the original character",
    }
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
