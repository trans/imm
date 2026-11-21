"""Each held-out character's loss under a no-prior transformer checkpoint
(prior_transformer.py --prior none --save), in imm's --char-loss-output
format, scored exactly as tb.heldout_nll / heldout_prior: windows of
`block`, stride block // 2, each target once. Read-only on air's code.

    python score_tx_noprior.py CKPT CFG.json OUT.npz
"""
import json, math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

PS = Path("/home/trans/Projects/air/experiments/prior_scaling")
sys.path.insert(0, str(PS))
import prior_transformer as pt  # noqa: E402

ckpt, cfg_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ck = torch.load(ckpt, map_location="cpu", weights_only=False)
a = ck["args"]
if a["prior"] != "none":
    raise SystemExit("score_tx_noprior: the checkpoint was trained with a prior")
cfg = json.loads(cfg_path.read_text())
vocab = pt.CharVocab.from_text(pt.read_text(pt.IMM / cfg["vocab_input"]))
held_text = pt.read_text(pt.IMM / cfg["eval_input"])[: cfg["eval_max_chars"]]
held = torch.tensor(vocab.encode(held_text), dtype=torch.long)
model = pt.tb.CharGPT(vocab.size, a["block"], a["layers"], a["heads"], a["width"], 0.0)
model.load_state_dict(ck["model"])
model.eval()
n = len(held)
losses = np.full(n - 1, np.nan, dtype=np.float32)  # index p: the loss of target p + 1
by_length: dict[int, list] = {}
for w in pt.tb.score_windows(n, model.block):
    by_length.setdefault(w[2] - 1 - w[0], []).append(w)
with torch.no_grad():
    for length, group in by_length.items():
        for i in range(0, len(group), 32):
            part = group[i : i + 32]
            x = torch.stack([held[s : s + length] for s, _, _ in part])
            y = torch.stack([held[s + 1 : s + 1 + length] for s, _, _ in part])
            nll = F.cross_entropy(model(x).float().transpose(1, 2), y, reduction="none")
            for r, (s, first, end) in enumerate(part):
                losses[first - 1 : end - 1] = nll[r, first - s - 1 :].numpy()
assert not np.isnan(losses).any(), "a target was not scored"
print(f"heldout_ppl={math.exp(float(losses.astype(np.float64).mean())):.4f} params={sum(q.numel() for q in model.parameters())}")
np.savez(out, positions=np.arange(n - 1, dtype=np.int64), losses=losses,
         text=held.numpy().astype(np.int16), vocab=np.array(list(vocab.chars)))
print(f"-> {out}")
