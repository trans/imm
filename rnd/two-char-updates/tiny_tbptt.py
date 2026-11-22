"""A tiny character GRU read by one stream, its weights updated every K
characters (truncated backpropagation through K steps). The question is how
much the window K costs; see README.md.

  python3 tiny_tbptt.py --window 2 --lr 1e-3 --train-chars 2000000 --out run.json

One stream reads the training text from its start. The state is carried
from update to update (detached) and reset to zeros at each piece's start;
no update crosses a piece's start. Every --eval-every characters, and at
the end, it scores the 50 held-out pieces, each from a zero state:
perplexity per character, exp of the mean loss in nats over characters
2..5,000 of each piece. CPU, float32."""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/pg19/imm-split-100m"


class TinyGRU(nn.Module):
    def __init__(self, vocab: int, embed: int, hidden: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, embed)
        self.gru = nn.GRU(embed, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, ids, h):
        out, h = self.gru(self.embed(ids), h)
        return self.head(out), h


def encode(text: str, vocab: str) -> torch.Tensor:
    index = {c: i for i, c in enumerate(vocab)}
    return torch.tensor([index[c] for c in text], dtype=torch.long)


@torch.no_grad()
def heldout_ppl(model, pieces, group: int = 10) -> float:
    model.eval()
    nats, n = 0.0, 0
    for g in range(0, pieces.shape[0], group):  # 10 pieces at a time, to keep memory low
        part = pieces[g : g + group]
        logits, _ = model(part[:, :-1], None)
        y = part[:, 1:].reshape(-1)
        nats += nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y, reduction="sum").item()
        n += y.numel()
    model.train()
    return math.exp(nats / n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, required=True, help="characters an update (K)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-chars", type=int, default=2_000_000)
    ap.add_argument("--eval-every", type=int, default=250_000)
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-group", type=int, default=10, help="held-out pieces scored at once (memory only)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    vocab = (DATA / "vocab.txt").read_text()
    manifest = json.loads((DATA / "manifest.json").read_text())
    starts = [r["start"] for r in manifest["train_records"] if r["start"] < args.train_chars]
    with open(DATA / "train_corpus.txt") as f:
        train = encode(f.read(args.train_chars), vocab)
    held = encode((DATA / "heldout_corpus.txt").read_text(), vocab).view(50, 5000)

    model = TinyGRU(len(vocab), args.embed, args.hidden)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ends = starts[1:] + [args.train_chars]

    log = {"args": {k: str(v) for k, v in vars(args).items()}, "params": params, "curve": []}
    print(f"window={args.window} lr={args.lr} params={params} pieces={len(starts)}", flush=True)
    log["curve"].append({"chars": 0, "heldout_ppl": heldout_ppl(model, held, args.eval_group), "sec": 0.0})
    t0, seen, updates, next_eval = time.time(), 0, 0, args.eval_every
    train_nats, train_n = 0.0, 0
    for start, end in zip(starts, ends):
        h = None  # a new piece: zeros
        # predict characters start+1 .. end-1, K of them an update
        for a in range(start, end - 1, args.window):
            b = min(a + args.window, end - 1)
            x = train[a:b].unsqueeze(0)
            y = train[a + 1 : b + 1]
            logits, h = model(x, h)
            loss = nn.functional.cross_entropy(logits[0], y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            h = h.detach()
            updates += 1
            seen += b - a
            train_nats += loss.item() * (b - a)
            train_n += b - a
            if seen >= next_eval:
                row = {"chars": seen, "updates": updates, "heldout_ppl": heldout_ppl(model, held, args.eval_group),
                       "train_ppl": math.exp(train_nats / train_n), "sec": round(time.time() - t0, 1)}
                log["curve"].append(row)
                print(json.dumps(row), flush=True)
                train_nats, train_n = 0.0, 0
                next_eval += args.eval_every
    final = {"chars": seen, "updates": updates, "heldout_ppl": heldout_ppl(model, held, args.eval_group), "sec": round(time.time() - t0, 1)}
    if log["curve"][-1]["chars"] != seen:
        log["curve"].append(final)
    log["final"] = final
    print("final", json.dumps(final), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(log, indent=1) + "\n")


if __name__ == "__main__":
    main()
