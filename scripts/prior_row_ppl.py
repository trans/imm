"""Prior-only perplexity of a count-prior cache's rows, per region, to check
the training rows' prior for the leak (lessons.md, "The residual trains
against a prior that has seen the answer").

    python scripts/prior_row_ppl.py --config configs/<run>.json

Regions: the training rows the held-out prior's counts cover (the first
1 - valid_ratio), the training rows held back from those counts, and the
held-out rows. Without cross-fitting the first region is far lower than the
others: those rows' prior counted the character they predict. Row r
predicts character r + 1. Needs the cache the config names, mmap format.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imm.data import CharVocab  # noqa: E402


def region_ppl(rows: np.memmap, ids: list[int], lo: int, hi: int, block: int = 500_000) -> float:
    total, count = 0.0, 0
    for a in range(lo, hi, block):
        b = min(hi, a + block)
        chunk = np.asarray(rows[a:b], dtype=np.float32)
        total += float(chunk[np.arange(b - a), np.asarray(ids[a + 1 : b + 1])].sum())
        count += b - a
    return math.exp(-total / count)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    cfg = json.loads(p.parse_args().config.read_text())
    vocab = CharVocab.from_text(Path(cfg["vocab_input"]).read_text())
    train = vocab.encode(Path(cfg["train_input"]).read_text()[: cfg.get("max_train_chars")])
    held = vocab.encode(Path(cfg["eval_input"]).read_text()[: cfg.get("eval_max_chars")])
    cache = torch.load(cfg["count_prior_cache"], weights_only=False)
    load = lambda which: np.memmap(  # noqa: E731
        cache[f"{which}_log_probs_path"], dtype=cache["log_prob_dtype"], mode="r", shape=tuple(cache[f"{which}_log_probs_shape"])
    )
    train_rows, held_rows = load("train"), load("eval")
    ratio = cfg.get("count_prior_valid_ratio", 0.05)
    split = int(len(train) * (1.0 - ratio))
    folds = cache["meta"].get("train_folds", 0)
    print(f"cache {cfg['count_prior_cache']} (train folds: {folds or 'none'})")
    print(f"  training rows counted by the held-out prior: {region_ppl(train_rows, train, 0, split - 1):.4f}")
    print(f"  training rows held back from it:             {region_ppl(train_rows, train, split - 1, train_rows.shape[0]):.4f}")
    print(f"  held-out rows:                               {region_ppl(held_rows, held, 0, held_rows.shape[0]):.4f}")


if __name__ == "__main__":
    main()
