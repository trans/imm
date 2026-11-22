"""Training loss against held-out loss, per epoch, for the 10.8M imm with
depth records (slot 137) and the 10.8M transformer (dropout 0), from their
logs. Does imm trail because it fits the training text less well, or
because it generalizes worse?

- imm: the run's CSV, train_nll (the mean over every training character of
  the epoch) and val_nll (held-out, at the epoch's end).
- The transformer: its log's train_loss, one batch (16 x 1,024 characters)
  every 500 steps, averaged over the steps of each eighth of the run; and
  the held-out curve's point nearest that eighth's end.

    python3 fit_gap.py > results/fit_gap_10.8m.txt"""
import csv
import json
import math
import re

S = "/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad"
RUN = "pg19s100m_h448_par_l3_w392_own_np_doc_mm128_do0_cm1024_depth_bf16_e8"
IMM = f"{S}/imm-q/runs/{RUN}_{RUN}.csv"
TX = f"{S}/s10/tx10.8m_np_100m_b1024_do0"

d = json.load(open(f"{TX}.json"))
steps = [(int(m.group(1)), float(m.group(2))) for m in re.finditer(r"step=(\d+) train_loss=([\d.]+)", open(f"{TX}.log").read())]
curve = {r["step"]: r["heldout_ppl"] for r in d["curve"]}
per_epoch = max(curve) / 8
tx = []
for e in range(1, 9):
    lo, hi = (e - 1) * per_epoch, e * per_epoch
    tr = [loss for s, loss in steps if lo <= s < hi]
    near = min(curve, key=lambda s: abs(s - hi))
    tx.append((sum(tr) / len(tr), math.log(curve[near])))
imm = [(float(r["train_nll"]), float(r["val_nll"])) for r in csv.DictReader(open(IMM)) if r["train_nll"]]

print("nats per character; gap = held-out - training")
print("epoch\timm_train\timm_heldout\timm_gap\ttx_train\ttx_heldout\ttx_gap")
for e, ((it, iv), (tt, tv)) in enumerate(zip(imm, tx), 1):
    print(f"{e}\t{it:.4f}\t{iv:.4f}\t{iv - it:+.4f}\t{tt:.4f}\t{tv:.4f}\t{tv - tt:+.4f}")
(it, iv), (tt, tv) = imm[-1], tx[-1]
print(f"epoch 8: imm's held-out excess {iv - tv:+.4f} = training excess {it - tt:+.4f} + gap excess {(iv - it) - (tv - tt):+.4f}")
