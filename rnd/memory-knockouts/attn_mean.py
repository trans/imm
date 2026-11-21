"""Eval-only runs of scripts/run_segment_memory_model.py (unchanged, loaded
from the current directory, a worktree) with the gated block's attention
output (out_proj, before the gate) replaced. One block only.

ABL_MODE:
  collect   output unchanged (the control); accumulate the attention output's
            mean over every real position that sees a record, overall and by
            the character's position in its word, into ABL_FILE (.pt), with
            a summary in ABL_FILE.json
  zero      the output set to 0 (must equal gate_attn = 0)
  global    the output set to its overall mean (from ABL_FILE)
  position  the output set to its mean at the character's position in its
            word (from ABL_FILE)

Usage, from the worktree: ABL_MODE=... ABL_FILE=... python -B attn_mean.py <run_segment_memory_model.py args>
"""

import importlib.util
import json
import os
import sys

import numpy as np
import torch

MODE = os.environ["ABL_MODE"]
FILE = os.environ["ABL_FILE"]
if MODE not in ("collect", "zero", "global", "position"):
    raise SystemExit(f"unknown ABL_MODE {MODE}")
MAXP = 64

root = os.getcwd()
script = os.path.join(root, "scripts", "run_segment_memory_model.py")
sys.path[0:1] = [os.path.join(root, "scripts"), root]
sys.argv = [script] + sys.argv[1:]
spec = importlib.util.spec_from_file_location("run_segment_memory_model", script)
mod = importlib.util.module_from_spec(spec)
sys.modules["run_segment_memory_model"] = mod
spec.loader.exec_module(mod)

ctx: dict = {}
acc: dict = {}
tables: dict = {}
blocks_seen: set = set()
calls = {"segments": 0, "parallel": 0, "replaced": 0}

if MODE in ("global", "position"):
    saved = torch.load(FILE, map_location="cpu")
    overall = saved["sum"] / saved["n"]
    pn = saved["pn"]
    by_pos = saved["psum"] / pn.clamp_min(1)[:, None]
    by_pos[pn == 0] = overall
    tables["global"], tables["position"] = overall, by_pos

LM = mod.SegmentMemoryLM
orig_steps = LM.segment_steps_streams
orig_par = LM.parallel_memory_streams
orig_fwd = mod.GatedCrossAttentionBlock.forward


def steps(self, embeddings, valid, valid_t, *args, **kwargs):
    ctx.update(path="segments", valid_t=valid_t, k=-1)
    try:
        return orig_steps(self, embeddings, valid, valid_t, *args, **kwargs)
    finally:
        ctx.clear()


def par(self, embeddings, valid, lengths, positions, *args, **kwargs):
    longest = valid.shape[2]
    assert (valid == (np.arange(longest)[None, None, :] < lengths[:, :, None])).all()
    stream_lengths = lengths.sum(axis=1)
    span_valid = np.arange(int(stream_lengths.max()))[None, :] < stream_lengths[:, None]
    offset = np.zeros(span_valid.shape, dtype=np.int64)
    offset[span_valid] = np.broadcast_to(np.arange(longest)[None, None, :], valid.shape)[valid]
    device = embeddings.device
    ctx.update(
        path="parallel",
        span_valid=torch.from_numpy(span_valid).to(device),
        offset=torch.from_numpy(offset).to(device),
    )
    try:
        return orig_par(self, embeddings, valid, lengths, positions, *args, **kwargs)
    finally:
        ctx.clear()


def out_hook(module, inputs, out):
    rows, offs = ctx["rows"], ctx["offs"]
    if MODE == "collect":
        o = out.detach().double()[rows]
        p = offs[rows]
        h = ctx["hidden"].detach().double()[rows]
        if not acc:
            acc.update(
                sum=o.new_zeros(o.shape[1]), sq=o.new_zeros(()), n=0, hsq=o.new_zeros(()),
                psum=o.new_zeros((MAXP, o.shape[1])), pn=o.new_zeros(MAXP),
            )
        acc["sum"] += o.sum(0)
        acc["sq"] += (o * o).sum()
        acc["n"] += o.shape[0]
        acc["hsq"] += (h * h).sum()
        acc["psum"].index_add_(0, p, o)
        acc["pn"].index_add_(0, p, torch.ones_like(p, dtype=torch.float64))
        return None
    calls["replaced"] += 1
    if MODE == "zero":
        return torch.zeros_like(out)
    table = tables[MODE].to(device=out.device, dtype=out.dtype)
    if MODE == "global":
        return table.expand_as(out).clone()
    return table[offs.clamp(max=MAXP - 1)]


def fwd(self, hidden, memory, query_positions=None, memory_positions=None, use_rope=True,
        return_diagnostics=False, memory_mask=None):
    blocks_seen.add(id(self))
    if len(blocks_seen) > 1:
        raise RuntimeError("attn_mean.py handles one gated block")
    if not getattr(self, "_abl_hooked", False):
        self.out_proj.register_forward_hook(out_hook)
        self._abl_hooked = True
    b, t, _ = hidden.shape
    path = ctx.get("path")
    if path == "segments":
        ctx["k"] += 1
        assert ctx["k"] < ctx["valid_t"].shape[1]
        rows = ctx["valid_t"][:, ctx["k"], :t]
        offs = torch.arange(t, device=hidden.device).expand(b, t)
    elif path == "parallel":
        rows = ctx["span_valid"] & memory_mask.any(dim=-1)
        offs = ctx["offset"]
    else:
        raise RuntimeError("a gated block ran outside the batched stream paths")
    calls[path] += 1
    ctx["rows"], ctx["offs"], ctx["hidden"] = rows, offs, hidden
    return orig_fwd(self, hidden, memory, query_positions, memory_positions, use_rope,
                    return_diagnostics, memory_mask)


LM.segment_steps_streams = steps
LM.parallel_memory_streams = par
mod.GatedCrossAttentionBlock.forward = fwd

try:
    mod.main()
finally:
    print(f"attn_mean mode={MODE} block_calls={calls}", flush=True)
    if MODE == "collect" and acc:
        saved = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in acc.items()}
        torch.save(saved, FILE)
        n, sq = saved["n"], float(saved["sq"])
        overall = saved["sum"] / n
        pn = saved["pn"]
        by_pos = saved["psum"] / pn.clamp_min(1)[:, None]
        summary = {
            "positions": n,
            "rms_attn_out": (sq / n) ** 0.5,
            "rms_block_input": (float(saved["hsq"]) / n) ** 0.5,
            "norm_overall_mean": float(overall.norm()),
            "share_overall_mean": float(overall.pow(2).sum() * n / sq),
            "share_position_means": float((pn[:, None] * by_pos.pow(2)).sum() / sq),
            "positions_by_offset": [int(x) for x in pn[: int((pn > 0).nonzero().max()) + 1]],
            "norm_mean_by_offset": [round(float(by_pos[i].norm()), 4) for i in range(int((pn > 0).nonzero().max()) + 1)],
        }
        with open(FILE + ".json", "w") as f:
            json.dump(summary, f, indent=1)
        print("attn_mean summary " + json.dumps(summary), flush=True)
