"""Where the sentence tier's per-position gate opens (--tier-position-gate).
Runs run_segment_memory_model.py's main() from the current directory's repo,
with write_char_losses wrapped to collect the factors (position_gate_sink)
while it scores the held-out text, then writes them by position to GATE_MAP
(npz: positions, factors). Position p reads character p and predicts p + 1.

    GATE_MAP=out.npz python gate_map.py <scorer args, with --char-loss-output>
"""
import importlib.util, os, sys
import numpy as np
import torch

REPO = os.getcwd()
spec = importlib.util.spec_from_file_location("rsm", f"{REPO}/scripts/run_segment_memory_model.py")
M = importlib.util.module_from_spec(spec)
sys.modules["rsm"] = M
spec.loader.exec_module(M)
out = os.environ["GATE_MAP"]
orig = M.write_char_losses


def wrapped(model, *a, **k):
    sink = []
    model.position_gate_sink = sink
    try:
        return orig(model, *a, **k)
    finally:
        model.position_gate_sink = None
        if sink:
            pos = torch.cat([p for p, _ in sink]).numpy()
            f = torch.cat([x for _, x in sink]).float().numpy()
            order = np.argsort(pos, kind="stable")
            np.savez(out, positions=pos[order], factors=f[order])
            print(f"gate_map n={len(pos)} mean={f.mean():.4f} -> {out}", flush=True)


M.write_char_losses = wrapped
sys.argv = [f"{REPO}/scripts/run_segment_memory_model.py"] + sys.argv[1:]
M.main()
