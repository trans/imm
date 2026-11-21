# Per-character held-out losses of the spiking epoch-5 weights: runs the
# runner's own --eval-only path with segment_loss_streams wrapped to record,
# for every scored character, its position in the held-out ids and its loss.
import importlib.util, sys, numpy as np, torch
import torch.nn.functional as F
root = sys.argv[1]; out = sys.argv[2]; argv = sys.argv[3:]
spec = importlib.util.spec_from_file_location("rsm", f"{root}/scripts/run_segment_memory_model.py")
rsm = importlib.util.module_from_spec(spec); sys.modules["rsm"] = rsm; spec.loader.exec_module(rsm)
rec_pos, rec_loss, rec_h, rec_start = [], [], [], []
pending = {}
orig_ce = F.cross_entropy
def ce(*a, **k):
    r = orig_ce(*a, **k)
    if k.get("reduction") == "none" and "pos" in pending:
        rec_pos.append(pending.pop("pos")); rec_loss.append(r.detach().float().cpu().numpy())
    return r
F.cross_entropy = ce
orig = rsm.SegmentMemoryLM.segment_loss_streams
def wrapped(self, ids, stream_segments, memory, max_memory, *a, score_mask=None, collect_stats=False, **k):
    if collect_stats and not self.training:
        starts = np.array([[s for s, _ in segs] for segs in stream_segments]); ends = np.array([[e for _, e in segs] for segs in stream_segments])
        lengths = ends - starts; L = int(lengths.max()); off = np.arange(L)
        valid = off[None, None, :] < lengths[:, :, None]
        positions = (starts[:, :, None] + off[None, None, :])[valid]
        keep = np.ones(len(positions), bool) if score_mask is None else np.broadcast_to(score_mask[:, :, None], valid.shape)[valid]
        pending["pos"] = positions[keep]
    result = orig(self, ids, stream_segments, memory, max_memory, *a, score_mask=score_mask, collect_stats=collect_stats, **k)
    if collect_stats and not self.training:
        rec_h.append(result[3].detach().float().norm(dim=-1).reshape(-1).cpu().numpy())
        rec_start.append(np.array([segs[-1][1] for segs in stream_segments]))
    return result
rsm.SegmentMemoryLM.segment_loss_streams = wrapped
sys.argv = ["run_segment_memory_model.py"] + argv
rsm.main()
pos = np.concatenate(rec_pos); loss = np.concatenate(rec_loss)
np.savez(out, pos=pos, loss=loss, hnorm=np.stack(rec_h), hend=np.stack(rec_start))
print("recorded", len(pos), "mean nll", loss.mean(), "bits", loss.mean() / np.log(2))
