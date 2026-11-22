"""The tiny character GRU of tiny_tbptt.py with level 1 on top. By default
each level is updated on its own two steps (README.md, stage 2); with
--char-window 4 both are updated together every 4 characters (stage 3).

  python3 tiny_levels.py --lr 1e-4 --train-chars 2000000 --out run.json
  python3 tiny_levels.py --char-window 4 --lr 1e-4 --train-chars 2000000 --out run.json

- The character GRU steps at every character. Its weights (with the
  embedding and the output layer) are updated every --char-window
  characters (default 2), from those predictions; its error reaches back
  that many steps.
- Level 1, a GRU, steps at every odd place p (counted from the piece's start,
  from 0), reading the character GRU's state there, detached. Its weights
  (with its read's projection and gate) are updated every --level-window
  characters (default 4: every 2 of its own steps, at every other character
  update), from those predictions; its error reaches back through its steps
  in that window. It can't reach into the character GRU: with the defaults,
  those weights change between its two steps (stage 3 keeps the detach, so
  only the clocks change).
- Each prediction at p adds level 1's state after its latest step at or
  before p, projected and times tanh(gate) (gate starts at 0), to the
  character GRU's state, in front of the output layer (as gru-tree does).
- Both states are carried, detached at each level's update, and reset to
  zeros at a piece's start. Held-out scoring as in tiny_tbptt.py. CPU."""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/pg19/imm-split-100m"


class TinyLevels(nn.Module):
    def __init__(self, vocab: int, embed: int, hidden: int, width: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, embed)
        self.gru = nn.GRU(embed, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab)
        self.level1 = nn.GRU(hidden, width, batch_first=True)
        self.proj = nn.Linear(width, hidden)
        self.gate = nn.Parameter(torch.zeros(1))
        self.width = width

    def char_params(self):
        return list(self.embed.parameters()) + list(self.gru.parameters()) + list(self.head.parameters())

    def level1_params(self):
        return list(self.level1.parameters()) + list(self.proj.parameters()) + [self.gate]

    def logits(self, char_states, reads):
        return self.head(char_states + torch.tanh(self.gate) * self.proj(reads))


def encode(text: str, vocab: str) -> torch.Tensor:
    index = {c: i for i, c in enumerate(vocab)}
    return torch.tensor([index[c] for c in text], dtype=torch.long)


@torch.no_grad()
def heldout_ppl(model, pieces, group: int = 10) -> float:
    model.eval()
    nats, n = 0.0, 0
    for g in range(0, pieces.shape[0], group):
        part = pieces[g : g + group]
        x, y = part[:, :-1], part[:, 1:]
        c, _ = model.gru(model.embed(x), None)  # [B, T, H], p = 0 .. T-1
        s, _ = model.level1(c[:, 1::2], None)  # its steps at p = 1, 3, ...
        after = torch.cat([torch.zeros_like(s[:, :1]), s], dim=1)  # after j steps
        index = (torch.arange(x.shape[1]) + 1) // 2  # steps at or before p
        logits = model.logits(c, after[:, index])
        nats += nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum").item()
        n += y.numel()
    model.train()
    return math.exp(nats / n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--train-chars", type=int, default=2_000_000)
    ap.add_argument("--eval-every", type=int, default=250_000)
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--width", type=int, default=128, help="level 1's width")
    ap.add_argument("--char-window", type=int, default=2, help="characters a character-level update")
    ap.add_argument("--level-window", type=int, default=4, help="characters a level-1 update (a multiple of --char-window)")
    ap.add_argument("--eval-group", type=int, default=10, help="held-out pieces scored at once (memory only)")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    kc, kl = args.char_window, args.level_window
    assert kc >= 1 and kl % kc == 0, "--level-window must be a multiple of --char-window"

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    vocab = (DATA / "vocab.txt").read_text()
    manifest = json.loads((DATA / "manifest.json").read_text())
    starts = [r["start"] for r in manifest["train_records"] if r["start"] < args.train_chars]
    with open(DATA / "train_corpus.txt") as f:
        train = encode(f.read(args.train_chars), vocab)
    held = encode((DATA / "heldout_corpus.txt").read_text(), vocab).view(50, 5000)

    model = TinyLevels(len(vocab), args.embed, args.hidden, args.width)
    params = sum(p.numel() for p in model.parameters())
    opt_c = torch.optim.Adam(model.char_params(), lr=args.lr)
    opt_1 = torch.optim.Adam(model.level1_params(), lr=args.lr)
    ends = starts[1:] + [args.train_chars]

    def gate() -> float:
        return round(math.tanh(model.gate.item()), 4)

    log = {"args": {k: str(v) for k, v in vars(args).items()}, "params": params, "curve": []}
    print(f"levels: hidden={args.hidden} width={args.width} char_window={kc} level_window={kl} lr={args.lr} params={params} pieces={len(starts)}", flush=True)
    log["curve"].append({"chars": 0, "heldout_ppl": heldout_ppl(model, held, args.eval_group), "gate": gate(), "sec": 0.0})
    t0, seen, char_updates, level_updates, next_eval = time.time(), 0, 0, 0, args.eval_every
    train_nats, train_n = 0.0, 0
    for start, end in zip(starts, ends):
        h = None
        s = torch.zeros(1, 1, args.width)
        for a in range(start, end - 1, kc):  # a character update of kc
            b = min(a + kc, end - 1)
            x = train[a:b].unsqueeze(0)
            out, h = model.gru(model.embed(x), h)
            reads = []
            for j in range(b - a):
                if (a - start + j) % 2 == 1:  # level 1 steps at odd p
                    _, s = model.level1(out[:, j : j + 1].detach(), s)
                reads.append(s[0])
            logits = model.logits(out, torch.stack(reads, dim=1))
            loss = nn.functional.cross_entropy(logits[0], train[a + 1 : b + 1])
            level_end = (a - start + kc) % kl == 0 or b >= end - 1  # level 1's window ends, or the piece
            loss.backward(retain_graph=not level_end)
            nn.utils.clip_grad_norm_(model.char_params(), args.clip)
            opt_c.step()
            opt_c.zero_grad(set_to_none=True)
            char_updates += 1
            if level_end:
                nn.utils.clip_grad_norm_(model.level1_params(), args.clip)
                opt_1.step()
                opt_1.zero_grad(set_to_none=True)
                level_updates += 1
                s = s.detach()
            h = h.detach()
            seen += b - a
            train_nats += loss.item() * (b - a)
            train_n += b - a
            if seen >= next_eval:
                row = {"chars": seen, "char_updates": char_updates, "level_updates": level_updates,
                       "heldout_ppl": heldout_ppl(model, held, args.eval_group), "train_ppl": math.exp(train_nats / train_n),
                       "gate": gate(), "sec": round(time.time() - t0, 1)}
                log["curve"].append(row)
                print(json.dumps(row), flush=True)
                train_nats, train_n = 0.0, 0
                next_eval += args.eval_every
    final = {"chars": seen, "char_updates": char_updates, "level_updates": level_updates,
             "heldout_ppl": heldout_ppl(model, held, args.eval_group), "gate": gate(), "sec": round(time.time() - t0, 1)}
    if log["curve"][-1]["chars"] != seen:
        log["curve"].append(final)
    log["final"] = final
    print("final", json.dumps(final), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(log, indent=1) + "\n")


if __name__ == "__main__":
    main()
