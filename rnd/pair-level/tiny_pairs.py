"""Level 1 makes the prediction: a GRU over pairs of characters, with a small
character decoder that spells out each pair. See README.md.

  python3 tiny_pairs.py --window 4 --lr 1e-4 --train-chars 2000000 --out run.json

- The text is read in pairs, counted from each piece's start: pair j is
  characters 2j and 2j+1.
- The upper GRU steps once a pair, after both of its characters, reading
  their two embeddings side by side. Its state after pair j is all the
  model knows of the text up to character 2j+1.
- The decoder predicts pair j+1 from that state alone. A small GRU starts
  from a projection of it:
  - its first step (a learned start input) predicts character 2j+2;
  - its second step, reading that character, predicts character 2j+3.
  It starts afresh for every pair, so nothing passes from pair to pair
  except through the upper GRU.
- Updates every --window characters (even), from those predictions. The
  upper state is carried, detached at each update, and reset to zeros at a
  piece's start; its error reaches back --window / 2 of its steps.
- Training text and held-out scoring as in rnd/two-char-updates: one
  stream, the first --train-chars training characters once, Adam at a
  constant rate, gradient norm clipped at 1.0. The 50 held-out pieces,
  each from a zero state, every --eval-every characters: perplexity per
  character over characters 2..5,000 of each piece. CPU."""

import argparse
import json
import math
import os
import time
from pathlib import Path

# CPU only. Hide the GPU: on CPU, Adam's step still checks for CUDA-graph
# capture, which opens a CUDA context, and the slot watcher counts any GPU
# process as a heavy job.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/pg19/imm-split-100m"


class PairGRU(nn.Module):
    def __init__(self, vocab: int, embed: int, upper: int, local: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, embed)
        self.upper = nn.GRU(2 * embed, upper, batch_first=True)
        self.init = nn.Linear(upper, local)
        self.start = nn.Parameter(torch.zeros(embed))
        self.local = nn.GRU(embed, local, batch_first=True)
        self.head = nn.Linear(local, vocab)
        self.upper_width = upper

    def pair_inputs(self, pairs):  # [B, P, 2] -> [B, P, 2E]
        e = self.embed(pairs)
        return e.reshape(pairs.shape[0], pairs.shape[1], -1)

    def decode(self, before, pairs):
        """before: [B, P, U], the upper state before each pair; pairs: [B, P, 2],
        the pair's characters. Returns logits [B, 2P, V] for those characters."""
        B, P, _ = before.shape
        h0 = torch.tanh(self.init(before)).reshape(1, B * P, -1)
        first = self.start.expand(B * P, 1, -1)
        second = self.embed(pairs[:, :, 0]).reshape(B * P, 1, -1)
        out, _ = self.local(torch.cat([first, second], dim=1), h0)  # [B*P, 2, L]
        return self.head(out).reshape(B, 2 * P, -1)

    def forward(self, pairs, u0):
        """pairs: [B, P, 2]; u0: [1, B, U], the state before the first pair.
        Returns logits for all 2P characters and the state after the last pair."""
        after, last = self.upper(self.pair_inputs(pairs), u0)  # [B, P, U]
        before = torch.cat([u0.transpose(0, 1), after[:, :-1]], dim=1)
        return self.decode(before, pairs), last


def encode(text: str, vocab: str) -> torch.Tensor:
    index = {c: i for i, c in enumerate(vocab)}
    return torch.tensor([index[c] for c in text], dtype=torch.long)


@torch.no_grad()
def heldout_ppl(model, pieces, group: int = 2, by_place: bool = False):
    """Perplexity per character over characters 1..4,999 (0-based) of each piece.
    With by_place, also over the first and the second character of each pair
    (even and odd places), as (all, first, second)."""
    model.eval()
    nats, n = torch.zeros(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)
    for g in range(0, pieces.shape[0], group):
        part = pieces[g : g + group]  # [B, 5000]
        u0 = torch.zeros(1, part.shape[0], model.upper_width)
        logits, _ = model(part.view(part.shape[0], -1, 2), u0)
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), part.reshape(-1), reduction="none").view(part.shape)
        loss = loss[:, 1:].double()  # character 0 of each piece is not scored
        place = torch.arange(1, part.shape[1]) % 2  # 0: a pair's first character
        for k in (0, 1):
            nats[k] += loss[:, place == k].sum()
            n[k] += loss[:, place == k].numel()
    model.train()
    total = math.exp(nats.sum().item() / n.sum().item())
    if by_place:
        return total, math.exp(nats[0].item() / n[0].item()), math.exp(nats[1].item() / n[1].item())
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, required=True, help="characters an update (even)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--train-chars", type=int, default=2_000_000)
    ap.add_argument("--eval-every", type=int, default=250_000)
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--upper", type=int, default=96, help="the upper GRU's width")
    ap.add_argument("--local", type=int, default=42, help="the decoder GRU's width")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-group", type=int, default=2, help="held-out pieces scored at once (memory only)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    assert args.window >= 2 and args.window % 2 == 0, "--window must be even"
    per = args.window // 2  # pairs an update

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    vocab = (DATA / "vocab.txt").read_text()
    manifest = json.loads((DATA / "manifest.json").read_text())
    starts = [r["start"] for r in manifest["train_records"] if r["start"] < args.train_chars]
    with open(DATA / "train_corpus.txt") as f:
        train = encode(f.read(args.train_chars), vocab)
    held = encode((DATA / "heldout_corpus.txt").read_text(), vocab).view(50, 5000)
    ends = starts[1:] + [args.train_chars]
    assert all((e - s) % 2 == 0 for s, e in zip(starts, ends)), "pieces must have an even length"

    model = PairGRU(len(vocab), args.embed, args.upper, args.local)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    log = {"args": {k: str(v) for k, v in vars(args).items()}, "params": params, "curve": []}
    print(f"pairs: window={args.window} upper={args.upper} local={args.local} lr={args.lr} params={params} pieces={len(starts)}", flush=True)
    log["curve"].append({"chars": 0, "heldout_ppl": heldout_ppl(model, held, args.eval_group), "sec": 0.0})
    t0, seen, updates, next_eval = time.time(), 0, 0, args.eval_every
    train_nats, train_n = 0.0, 0
    for start, end in zip(starts, ends):
        u = torch.zeros(1, 1, args.upper)  # a new piece: zeros
        n_pairs = (end - start) // 2
        for m in range(0, n_pairs, per):
            q = min(m + per, n_pairs)
            seg = train[start + 2 * m : start + 2 * q]
            logits, last = model(seg.view(1, -1, 2), u)
            skip = 1 if m == 0 else 0  # the piece's first character is not predicted
            loss = nn.functional.cross_entropy(logits[0, skip:], seg[skip:])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            u = last.detach()
            updates += 1
            k = seg.numel() - skip
            seen += k
            train_nats += loss.item() * k
            train_n += k
            if seen >= next_eval:
                row = {"chars": seen, "updates": updates, "heldout_ppl": heldout_ppl(model, held, args.eval_group),
                       "train_ppl": math.exp(train_nats / train_n), "sec": round(time.time() - t0, 1)}
                log["curve"].append(row)
                print(json.dumps(row), flush=True)
                train_nats, train_n = 0.0, 0
                next_eval += args.eval_every
    total, first, second = heldout_ppl(model, held, args.eval_group, by_place=True)
    final = {"chars": seen, "updates": updates, "heldout_ppl": total, "heldout_ppl_first": first,
             "heldout_ppl_second": second, "sec": round(time.time() - t0, 1)}
    if log["curve"][-1]["chars"] != seen:
        log["curve"].append(final)
    log["final"] = final
    print("final", json.dumps(final), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(log, indent=1) + "\n")


if __name__ == "__main__":
    main()
