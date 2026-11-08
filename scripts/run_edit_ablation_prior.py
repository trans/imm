#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imm.count_gate import CountModel, MarisaCountModel, PackedCountModel, build_vocab, encode, expand_extra_features
from imm.edit_kernel_prior import (
    EditKernelCorrectionPrior,
    EditKernelPrior,
    FuzzyNodeCorrectionPrior,
    FuzzyNodeLocator,
    corrupt_context_once,
    sampled_positions,
    score_prior,
    train_fuzzy_node_correction_prior,
    train_edit_correction_prior,
    train_edit_kernel_prior,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare exact and edit-kernel priors on context-corrupted heldout.")
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--heldout-input", type=Path, required=True)
    parser.add_argument("--vocab-input", type=Path, required=True)
    parser.add_argument("--exact-theta-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--impl", choices=["python", "packed", "marisa"], default="marisa")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--max-fit-positions", type=int, default=2000)
    parser.add_argument("--max-eval-positions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--extra-features", default="entropy_delta,suffix_stats")
    parser.add_argument("--corrupt-prob", type=float, default=0.05)
    parser.add_argument(
        "--train-corrupt-prob",
        type=float,
        default=None,
        help="Corruption probability for gate training. Defaults to --corrupt-prob.",
    )
    parser.add_argument("--edit-lambda", type=float, default=1.0)
    parser.add_argument("--mass-alpha", type=float, default=1.0)
    parser.add_argument("--candidate-cap", type=int, default=256)
    parser.add_argument("--mode", choices=["correction", "full", "fuzzy-node"], default="correction")
    parser.add_argument("--corrupt-ops", default="substitute,delete,insert,transpose")
    return parser.parse_args()


def load_exact_theta(path: Path, model: CountModel) -> list[float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    feature_names = model.feature_names()
    if payload.get("feature_names") != feature_names:
        raise ValueError(f"theta feature mismatch: expected {feature_names}, got {payload.get('feature_names')}")
    return [float(payload["theta"][name]) for name in feature_names]


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

    exact_theta = load_exact_theta(args.exact_theta_json, model)
    edit_kernel = EditKernelPrior(model, edit_lambda=args.edit_lambda, candidate_cap=args.candidate_cap)
    if args.mode == "fuzzy-node":
        locator = FuzzyNodeLocator(
            model,
            edit_lambda=args.edit_lambda,
            mass_alpha=args.mass_alpha,
            candidate_cap=args.candidate_cap,
        )
        prior = FuzzyNodeCorrectionPrior(model, exact_theta, locator)
    elif args.mode == "correction":
        prior = EditKernelCorrectionPrior(model, exact_theta, edit_kernel)
    else:
        prior = edit_kernel

    train_corrupt_prob = args.corrupt_prob if args.train_corrupt_prob is None else args.train_corrupt_prob

    def corrupt_with_probability(ctx: bytes, pos: int, epoch: int, probability: float) -> bytes:
        return corrupt_context_once(
            ctx,
            len(chars),
            probability,
            seed=args.seed + 1000003 * epoch + pos,
            max_depth=args.depth,
            operations=tuple(part.strip() for part in args.corrupt_ops.split(",") if part.strip()),
        )

    def corrupt_train(ctx: bytes, pos: int, epoch: int) -> bytes:
        return corrupt_with_probability(ctx, pos, epoch, train_corrupt_prob)

    def corrupt_eval(ctx: bytes, pos: int, epoch: int) -> bytes:
        return corrupt_with_probability(ctx, pos, epoch, args.corrupt_prob)

    fit_positions = sampled_positions(valid, args.max_fit_positions, args.depth, args.seed)
    eval_positions = sampled_positions(heldout_ids, args.max_eval_positions, args.depth, args.seed)

    t_fit = time.perf_counter()
    if args.mode == "fuzzy-node":
        edit_theta, history = train_fuzzy_node_correction_prior(
            prior,
            valid,
            args.depth,
            fit_positions,
            args.epochs,
            args.lr,
            context_fn=corrupt_train,
        )
    elif args.mode == "correction":
        edit_theta, history = train_edit_correction_prior(
            prior,
            valid,
            args.depth,
            fit_positions,
            args.epochs,
            args.lr,
            context_fn=corrupt_train,
        )
    else:
        edit_theta, history = train_edit_kernel_prior(
            prior,
            valid,
            args.depth,
            fit_positions,
            args.epochs,
            args.lr,
            context_fn=corrupt_train,
        )
    fit_sec = time.perf_counter() - t_fit

    t_eval = time.perf_counter()
    exact_clean = score_prior(lambda ctx, target: model.gated_prob(ctx, target, exact_theta)[0], heldout_ids, eval_positions, args.depth)
    exact_corrupt = score_prior(
        lambda ctx, target: model.gated_prob(ctx, target, exact_theta)[0],
        heldout_ids,
        eval_positions,
        args.depth,
        context_fn=corrupt_eval,
    )
    edit_corrupt = score_prior(
        lambda ctx, target: prior.prob(ctx, target, edit_theta),
        heldout_ids,
        eval_positions,
        args.depth,
        context_fn=corrupt_eval,
    )
    eval_sec = time.perf_counter() - t_eval

    payload = {
        "config": vars(args) | {
            "train_input": str(args.train_input),
            "heldout_input": str(args.heldout_input),
            "vocab_input": str(args.vocab_input),
            "exact_theta_json": str(args.exact_theta_json),
            "output": str(args.output),
            "resolved_train_corrupt_prob": train_corrupt_prob,
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
        "theta": dict(zip(prior.parameter_names(), edit_theta)),
        "history": history,
        "scores": {
            "exact_clean": {"loss_nats": exact_clean[0], "ppl": exact_clean[1], "positions": exact_clean[2]},
            "exact_corrupt": {"loss_nats": exact_corrupt[0], "ppl": exact_corrupt[1], "positions": exact_corrupt[2]},
            f"{args.mode}_corrupt": {"loss_nats": edit_corrupt[0], "ppl": edit_corrupt[1], "positions": edit_corrupt[2]},
        },
        "wall_seconds": {
            "build": build_sec,
            "fit": fit_sec,
            "eval": eval_sec,
            "total": build_sec + fit_sec + eval_sec,
        },
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"edit_ablation mode={args.mode} impl={args.impl} depth={args.depth} "
        f"train_corrupt_prob={train_corrupt_prob} eval_corrupt_prob={args.corrupt_prob} "
        f"exact_clean_ppl={exact_clean[1]:.6f} exact_corrupt_ppl={exact_corrupt[1]:.6f} "
        f"{args.mode}_corrupt_ppl={edit_corrupt[1]:.6f} "
        f"fit_ppl={history[-1]['ppl'] if history else float('nan'):.3f} "
        f"build_sec={build_sec:.2f} fit_sec={fit_sec:.2f} eval_sec={eval_sec:.2f} "
        f"peak_rss_mb={payload['peak_rss_mb']:.1f} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
