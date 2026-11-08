#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imm.count_gate import CountModel, MarisaCountModel, PackedCountModel, build_vocab, encode, expand_extra_features
from imm.edit_kernel_prior import EditKernelPrior, sampled_positions, score_prior, train_edit_kernel_prior


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a learned edit-kernel count prior.")
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--heldout-input", type=Path, required=True)
    parser.add_argument("--vocab-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--impl", choices=["python", "packed", "marisa"], default="marisa")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--max-fit-positions", type=int, default=10000)
    parser.add_argument("--max-eval-positions", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--extra-features", default="entropy_delta,suffix_stats")
    parser.add_argument("--edit-radius", type=int, default=1)
    parser.add_argument("--edit-lambda", type=float, default=1.0)
    parser.add_argument("--candidate-cap", type=int, default=256)
    parser.add_argument("--no-transpose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vocab_text = args.vocab_input.read_text(encoding="utf-8")
    train_text = args.train_input.read_text(encoding="utf-8")
    heldout_text = args.heldout_input.read_text(encoding="utf-8")
    chars, stoi = build_vocab(vocab_text)
    train_ids = encode(train_text, stoi)
    heldout_ids = encode(heldout_text, stoi)

    split = int(len(train_ids) * (1.0 - args.valid_ratio))
    split = max(args.depth + 1, min(split, len(train_ids) - args.depth - 1))
    count_train = train_ids[:split]
    valid = train_ids[split:]

    model_cls = {
        "python": CountModel,
        "packed": PackedCountModel,
        "marisa": MarisaCountModel,
    }[args.impl]
    t0 = time.perf_counter()
    model = model_cls(count_train, len(chars), args.depth, expand_extra_features(args.extra_features))
    model.build()
    build_sec = time.perf_counter() - t0

    prior = EditKernelPrior(
        model,
        radius=args.edit_radius,
        edit_lambda=args.edit_lambda,
        candidate_cap=args.candidate_cap,
        allow_transpose=not args.no_transpose,
    )
    fit_positions = sampled_positions(valid, args.max_fit_positions, args.depth, args.seed)
    t_fit = time.perf_counter()
    theta, history = train_edit_kernel_prior(prior, valid, args.depth, fit_positions, args.epochs, args.lr)
    fit_sec = time.perf_counter() - t_fit

    eval_positions = sampled_positions(heldout_ids, args.max_eval_positions, args.depth, args.seed)
    t_eval = time.perf_counter()
    heldout_loss, heldout_ppl, heldout_count = score_prior(
        lambda ctx, target: prior.prob(ctx, target, theta),
        heldout_ids,
        eval_positions,
        args.depth,
    )
    eval_sec = time.perf_counter() - t_eval
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    payload = {
        "config": vars(args) | {
            "train_input": str(args.train_input),
            "heldout_input": str(args.heldout_input),
            "vocab_input": str(args.vocab_input),
            "output": str(args.output),
        },
        "corpus": {
            "vocab_size": len(chars),
            "train_total_chars": len(train_ids),
            "count_train_chars": len(count_train),
            "valid_chars": len(valid),
            "heldout_chars": len(heldout_ids),
        },
        "feature_names": prior.feature_names(),
        "parameter_names": prior.parameter_names(),
        "theta": dict(zip(prior.parameter_names(), theta)),
        "history": history,
        "heldout": {
            "loss_nats": heldout_loss,
            "ppl": heldout_ppl,
            "positions": heldout_count,
        },
        "wall_seconds": {
            "build": build_sec,
            "fit": fit_sec,
            "eval": eval_sec,
            "total": build_sec + fit_sec + eval_sec,
        },
        "peak_rss_mb": rss_mb,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"edit_kernel impl={args.impl} depth={args.depth} fit_ppl={history[-1]['ppl'] if history else math.nan:.3f} "
        f"heldout_ppl={heldout_ppl:.6f} positions={heldout_count} "
        f"build_sec={build_sec:.2f} fit_sec={fit_sec:.2f} eval_sec={eval_sec:.2f} "
        f"peak_rss_mb={rss_mb:.1f} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
