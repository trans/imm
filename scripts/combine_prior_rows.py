"""Write a prior-rows prefix for --prior-rows from one or more sources:
another program's rows (a PREFIX with PREFIX.json, mobius_prior_rows_v2) or
imm's own count-prior cache (the .pt of an mmap cache).

    # mobius's unit-context log-probs with imm's gate features
    python scripts/combine_prior_rows.py --source units=../mobius/runs/imm/gutenberg5m_units \\
        --source imm=runs/count_prior_gutenberg5m_d8_entropy_suffix_fp16_mmap_numpy_crossfit20.pt \\
        --log-probs units --gate-features imm --out runs/prior_rows/units_immgate

    # a 50/50 mixture of the two, in probability space
    python scripts/combine_prior_rows.py --source units=... --source imm=... \\
        --log-probs units:0.5 --log-probs imm:0.5 --gate-features imm --out runs/prior_rows/mix50_immgate

One log-prob source is referenced, not copied. A mixture is written:
log(sum_i w_i exp(l_i)) row by row, float16, the weights summing to 1. Gate
features are always referenced. The sources must be for the same text and
vocabulary (their digests and vocab_chars agree), and so is the output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imm.count_gate import expand_extra_features  # noqa: E402

BLOCK = 200_000


def entry(path: Path, shape, dtype) -> dict:
    return {"path": str(path.resolve()), "shape": [int(n) for n in shape], "dtype": np.dtype(dtype).name}


def read_source(spec: str) -> tuple[str, dict]:
    """NAME=PATH: a prefix with PATH.json, or imm's cache .pt. Returns the name
    and a json-like dict whose paths are absolute."""
    name, _, raw = spec.partition("=")
    if not name or not raw:
        raise ValueError(f"--source wants NAME=PATH, got {spec!r}")
    path = Path(raw)
    if path.suffix == ".pt":
        cache = torch.load(path, map_location="cpu", weights_only=False)
        if cache.get("format") != "mmap":
            raise ValueError(f"{path}: only mmap caches have row files to reference")
        meta = cache["meta"]
        root = Path.cwd()  # the cache's paths are relative to where the trainer ran
        names = ["reliability", "kl_gain", "entropy_norm", "depth_norm", *expand_extra_features(meta["extra_features"]), "bias"]
        if len(names) != cache["feature_count"]:
            raise ValueError(f"{path}: {cache['feature_count']} feature columns, but the meta names {len(names)}")
        return name, {
            "builder": f"imm count prior (depth {meta['depth']}, {meta['extra_features']}, folds {meta.get('train_folds')})",
            "vocab_chars": list(meta["vocab_chars"]),
            "train_digest": meta["train_digest"],
            "eval_digest": meta["eval_digest"],
            "log_probs": {
                split: entry(root / cache[f"{split}_log_probs_path"], cache[f"{split}_log_probs_shape"], cache["log_prob_dtype"])
                for split in ("train", "eval")
            },
            "gate_features": {
                "feature_names": names,
                "layout": "row r: imm's count-gate features of the deepest context with counts ending at r; zeros and bias 1 when none",
                **{
                    split: entry(root / cache[f"{split}_gate_features_path"], cache[f"{split}_gate_features_shape"], cache["feature_dtype"])
                    for split in ("train", "eval")
                },
            },
            "cross_fit": {"folds": meta.get("train_folds"), "chunk": meta.get("fold_chunk")},
        }
    meta_path = path.parent / f"{path.name}.json"
    meta = json.loads(meta_path.read_text())
    for section in ("log_probs", "gate_features", "depth_evidence"):
        if isinstance(meta.get(section), dict):
            for split in ("train", "eval"):
                if isinstance(meta[section].get(split), dict):
                    meta[section][split] = {**meta[section][split], "path": str((path.parent / meta[section][split]["path"]).resolve())}
    return name, meta


def mix(sources: list[tuple[dict, float]], split: str, out_path: Path) -> dict:
    entries = [meta["log_probs"][split] for meta, _ in sources]
    shape = tuple(entries[0]["shape"])
    if any(tuple(e["shape"]) != shape for e in entries):
        raise ValueError(f"{split} log-prob shapes differ: {[e['shape'] for e in entries]}")
    inputs = [np.memmap(e["path"], dtype=np.dtype(e["dtype"]), mode="r", shape=shape) for e in entries]
    log_w = np.log(np.array([w for _, w in sources], dtype=np.float32))[:, None, None]
    out = np.memmap(out_path, dtype=np.float16, mode="w+", shape=shape)
    for a in range(0, shape[0], BLOCK):
        b = min(shape[0], a + BLOCK)
        stacked = np.stack([np.asarray(x[a:b], dtype=np.float32) for x in inputs]) + log_w
        top = stacked.max(axis=0)
        out[a:b] = (top + np.log(np.exp(stacked - top).sum(axis=0))).astype(np.float16)
    out.flush()
    return entry(out_path, shape, np.float16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", action="append", required=True, help="NAME=PATH, a rows prefix or imm's cache .pt")
    parser.add_argument("--log-probs", action="append", required=True, help="NAME or NAME:WEIGHT; weights sum to 1")
    parser.add_argument("--gate-features", default=None, help="NAME of the source whose gate features to use")
    parser.add_argument("--out", type=Path, required=True, help="output PREFIX (PREFIX.json, PREFIX.*.dat)")
    args = parser.parse_args()

    sources = dict(read_source(spec) for spec in args.source)
    first = next(iter(sources.values()))
    for name, meta in sources.items():
        for key in ("vocab_chars", "train_digest", "eval_digest"):
            if meta.get(key) != first.get(key):
                raise ValueError(f"source {name}'s {key} differs from the others'")
    chosen = []
    for spec in args.log_probs:
        name, _, weight = spec.partition(":")
        if name not in sources:
            raise ValueError(f"--log-probs {name}: no such --source")
        chosen.append((name, float(weight) if weight else 1.0))
    if abs(sum(w for _, w in chosen) - 1.0) > 1e-6 or any(w <= 0 for _, w in chosen):
        raise ValueError(f"log-prob weights must be positive and sum to 1, got {chosen}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if len(chosen) == 1:
        log_probs = dict(sources[chosen[0][0]]["log_probs"])
        builder = sources[chosen[0][0]].get("builder")
    else:
        log_probs = {
            split: mix([(sources[n], w) for n, w in chosen], split, args.out.parent / f"{args.out.name}.{split}_log_probs.dat")
            for split in ("train", "eval")
        }
        builder = "mixture: " + " + ".join(f"{w:g} x {sources[n].get('builder')}" for n, w in chosen)
    log_probs["layout"] = "row r predicts ids[r + 1] from ids[0..r]; natural-log probabilities over the vocabulary"
    meta = {
        "schema": "mobius_prior_rows_v2",
        "builder": builder,
        "combined_by": "imm scripts/combine_prior_rows.py",
        "log_prob_sources": [{"source": n, "weight": w, "from": sources[n].get("builder")} for n, w in chosen],
        "vocab_chars": first["vocab_chars"],
        "train_digest": first["train_digest"],
        "eval_digest": first["eval_digest"],
        "log_probs": log_probs,
        "cross_fit": {n: sources[n].get("cross_fit") for n, _ in chosen},
    }
    if args.gate_features is not None:
        if not isinstance(sources.get(args.gate_features, {}).get("gate_features"), dict):
            raise ValueError(f"--gate-features {args.gate_features}: that source has no gate features")
        meta["gate_features"] = {**sources[args.gate_features]["gate_features"], "source": args.gate_features}
    out_json = args.out.parent / f"{args.out.name}.json"
    out_json.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out_json}: log-probs {[(n, w) for n, w in chosen]}, gate features {args.gate_features}")


if __name__ == "__main__":
    main()
