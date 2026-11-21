"""Each held-out character's loss under air's prior transformer, in the
format of imm's --char-loss-output (positions p predicting p + 1, losses,
text, vocab), so imm's per-position tools compare the two. Scores exactly as
prior_transformer.heldout_prior: windows of `block`, stride block // 2, each
target once. Read-only on the checkpoint and air's code.

    python score_transformer_chars.py CKPT OUT.npz [--prior PREFIX]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PS = Path("/home/trans/Projects/air/experiments/prior_scaling")
sys.path.insert(0, str(PS))
import prior_transformer as pt  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("ckpt", type=Path)
p.add_argument("out", type=Path)
p.add_argument("--prior", default=None, help="rows prefix (default: the checkpoint's, under ~/Projects/air)")
p.add_argument("--batch", type=int, default=32)
a = p.parse_args()

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg_args = ck["args"]
prior = a.prior or cfg_args["prior"].replace("/home/trans/air/", "/home/trans/Projects/air/")
cfg = json.loads((PS / "pg19split20m_cfg.json").read_text())
vocab = pt.CharVocab.from_text(pt.read_text(pt.IMM / cfg["vocab_input"]))
held_text = pt.read_text(pt.IMM / cfg["eval_input"])[: cfg["eval_max_chars"]]
held = torch.tensor(vocab.encode(held_text), dtype=torch.long)
rows = pt.PriorRows(prior, "eval", vocab.size, len(held))
rows.mean = np.asarray(ck["gate_feature_mean"], dtype=np.float32)
rows.std = np.asarray(ck["gate_feature_std"], dtype=np.float32)

base = pt.tb.CharGPT(vocab.size, cfg_args["block"], cfg_args["layers"], cfg_args["heads"], cfg_args["width"], 0.0)
model = pt.PriorGPT(base, rows.features, cfg_args["prior_input"])
model.load_state_dict(ck["model"])
model.eval()

n = len(held)
losses = np.full(n - 1, np.nan, dtype=np.float32)  # index p: the loss of target p + 1
own_losses = np.full(n - 1, np.nan, dtype=np.float32)
by_length: dict[int, list] = {}
for w in pt.tb.score_windows(n, model.block):
    by_length.setdefault(w[2] - 1 - w[0], []).append(w)
with torch.no_grad():
    for length, group in by_length.items():
        for i in range(0, len(group), a.batch):
            part = group[i : i + a.batch]
            x = torch.stack([held[s : s + length] for s, _, _ in part])
            y = torch.stack([held[s + 1 : s + 1 + length] for s, _, _ in part])
            lp, gf = rows.window([s for s, _, _ in part], length)
            logits, g, own = model(x, lp, gf)
            nll = F.cross_entropy(logits.float().transpose(1, 2), y, reduction="none")
            nll_own = F.cross_entropy(own.float().transpose(1, 2), y, reduction="none")
            for r, (s, first, end) in enumerate(part):
                k = first - s - 1  # window offset of the first counted target
                losses[first - 1 : end - 1] = nll[r, k:].numpy()
                own_losses[first - 1 : end - 1] = nll_own[r, k:].numpy()
assert not np.isnan(losses).any(), "a target was not scored"
ppl = math.exp(float(losses.astype(np.float64).mean()))
print(f"heldout_ppl={ppl:.4f} (the run's json: {ck.get('heldout_ppl', 'n/a')}); own logits alone {math.exp(float(own_losses.astype(np.float64).mean())):.2f}")
np.savez(a.out, positions=np.arange(n - 1, dtype=np.int64), losses=losses, own_losses=own_losses,
         text=held.numpy().astype(np.int16), vocab=np.array(list(vocab.chars)))
print(f"-> {a.out}")
