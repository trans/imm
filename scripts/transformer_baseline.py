"""A plain character transformer as a baseline for imm, on imm's split and
vocabulary, scored the way imm is scored.

The model and training follow nanoGPT's Shakespeare character config
(github.com/karpathy/nanoGPT, config/train_shakespeare_char.py): 6 layers,
6 heads, 384 wide, context 256, dropout 0.2, no biases, tied embeddings,
AdamW (lr 1e-3, betas 0.9/0.99, weight decay 0.1 on matrices), 100 warmup
steps then cosine to 1e-4 at 5,000 steps, batch 64, gradient clip 1.0. This
is a reimplementation, not nanoGPT's code.

Scoring: every character of the held-out text after the first, in order,
with up to `block` characters of context (windows overlap by half, and each
character is scored once, from the window where it has the most context).
Perplexity is per character, natural log, as imm's val_ppl. Reported at the
last step, not the best one: the held-out text is the only test set.

    python scripts/transformer_baseline.py --config configs/<imm config>.json --output runs/<name>.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imm.data import CharVocab, read_text  # noqa: E402


class Block(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(width, bias=False)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.ln2 = nn.LayerNorm(width, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width, bias=False),
            nn.GELU(),
            nn.Linear(4 * width, width, bias=False),
            nn.Dropout(dropout),
        )
        self.dropout = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, w = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(w, dim=2)
        q, k, v = (z.view(b, t, self.heads, w // self.heads).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.resid_drop(self.proj(y.transpose(1, 2).reshape(b, t, w)))
        return x + self.mlp(self.ln2(x))


class CharGPT(nn.Module):
    def __init__(self, vocab: int, block: int, layers: int, heads: int, width: int, dropout: float) -> None:
        super().__init__()
        self.block = block
        self.tok = nn.Embedding(vocab, width)
        self.pos = nn.Embedding(block, width)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(width, heads, dropout) for _ in range(layers))
        self.ln = nn.LayerNorm(width, bias=False)
        self.head = nn.Linear(width, vocab, bias=False)
        self.head.weight = self.tok.weight  # tied
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                std = 0.02 / math.sqrt(2 * layers) if name.endswith(("proj.weight", "mlp.2.weight")) else 0.02
                nn.init.normal_(p, 0.0, std)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        t = ids.shape[1]
        x = self.drop(self.tok(ids) + self.pos(torch.arange(t, device=ids.device)))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x))


def score_windows(n: int, block: int) -> list[tuple[int, int, int]]:
    """(window start, first scored target, end) for scoring targets 1..n-1 of
    a text of n tokens: window [s, s + block) predicts tokens s+1 .. s+block,
    and only targets from `first` to `end` - 1 count, so each target counts
    once, with at least block // 2 tokens of context where the text allows."""
    if n < 2:
        return []
    out, stride, s = [], max(1, block // 2), 0
    first = 1
    while True:
        end = min(n, s + block + 1)
        out.append((s, first, end))
        if end == n:
            return out
        s += stride
        first = end


@torch.no_grad()
def heldout_nll(model: CharGPT, ids: torch.Tensor, batch: int, device: torch.device, dtype) -> tuple[float, int]:
    model.eval()
    total, count = 0.0, 0
    wins = score_windows(len(ids), model.block)
    # All windows but possibly the last have the same length: run those in batches.
    by_length: dict[int, list[tuple[int, int, int]]] = {}
    for w in wins:
        by_length.setdefault(w[2] - 1 - w[0], []).append(w)
    for length, group in by_length.items():
        for i in range(0, len(group), batch):
            part = group[i : i + batch]
            x = torch.stack([ids[s : s + length] for s, _, _ in part]).to(device)
            y = torch.stack([ids[s + 1 : s + 1 + length] for s, _, _ in part]).to(device)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                logits = model(x).float()
            nll = F.cross_entropy(logits.transpose(1, 2), y, reduction="none")  # (windows, length)
            for row, (s, first, end) in enumerate(part):
                total += float(nll[row, first - s - 1 :].sum())
                count += end - first
    model.train()
    return total, count


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, required=True, help="imm config: its train_input, eval_input, vocab_input, eval_max_chars")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--block", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--iters", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=1000, help="score the held-out text this often, for the curve only")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    cfg = json.loads(args.config.read_text())
    root = Path(__file__).resolve().parents[1]
    vocab = CharVocab.from_text(read_text(root / cfg["vocab_input"]))
    train = torch.tensor(vocab.encode(read_text(root / cfg["train_input"])), dtype=torch.long)
    held_text = read_text(root / cfg["eval_input"])
    if cfg.get("eval_max_chars"):
        held_text = held_text[: cfg["eval_max_chars"]]
    held = torch.tensor(vocab.encode(held_text), dtype=torch.long)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    model = CharGPT(vocab.size, args.block, args.layers, args.heads, args.width, args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.99),
    )

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / (args.warmup + 1)
        frac = min(1.0, (step - args.warmup) / max(1, args.iters - args.warmup))
        return args.min_lr + 0.5 * (1.0 + math.cos(math.pi * frac)) * (args.lr - args.min_lr)

    print(f"params={n_params} train_chars={len(train)} heldout_chars={len(held)} device={device} dtype={dtype}", flush=True)
    gen = torch.Generator().manual_seed(args.seed)
    curve, started = [], time.perf_counter()
    for step in range(args.iters + 1):
        if step % args.eval_every == 0 or step == args.iters:
            total, count = heldout_nll(model, held, args.batch, device, dtype)
            curve.append({"step": step, "heldout_ppl": math.exp(total / count), "sec": time.perf_counter() - started})
            print(f"step={step} heldout_ppl={curve[-1]['heldout_ppl']:.4f} scored={count} sec={curve[-1]['sec']:.0f}", flush=True)
        if step == args.iters:
            break
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        starts = torch.randint(0, len(train) - args.block - 1, (args.batch,), generator=gen)
        x = torch.stack([train[s : s + args.block] for s in starts]).to(device)
        y = torch.stack([train[s + 1 : s + 1 + args.block] for s in starts]).to(device)
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            loss = F.cross_entropy(model(x).float().view(-1, vocab.size), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            print(f"step={step} train_loss={loss.item():.4f} lr={lr_at(step):.2e}", flush=True)

    result = {
        "model": "char transformer (nanoGPT Shakespeare char config, reimplemented)",
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "params": n_params,
        "split": {"train_input": cfg["train_input"], "eval_input": cfg["eval_input"], "eval_chars": len(held)},
        "heldout_ppl": curve[-1]["heldout_ppl"],
        "heldout_bits_per_char": math.log2(curve[-1]["heldout_ppl"]),
        "protocol": "every held-out character after the first, up to block characters of context; last step",
        "curve": curve,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"result: heldout_ppl={result['heldout_ppl']:.4f} -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
