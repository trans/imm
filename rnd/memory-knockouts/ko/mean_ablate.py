"""Mean ablation of slot 63's word GRU and memory attention. Runs
run_segment_memory_model.py's main() with two patches:

- the word tier's add (tanh(gate) * proj(word state)) is replaced by its
  average: one average for a segment's last character (which, with own
  step, sees its own word's state), one for the other characters;
- each gated block's attention output (out_proj's output, before the gate;
  zero still where no record is visible) is replaced by its average.

KO_MODE=collect scores the model unchanged and writes the averages over the
held-out run's real positions to KO_MEANS; KO_MODE=word, mem or both applies
them. The averages come from the held-out text: 192 numbers per stream.

    KO_MODE=collect KO_MEANS=means.pt python mean_ablate.py <scorer args>
"""
import importlib.util, os, sys
import torch

IMM = "/home/trans/Projects/air/imm"
spec = importlib.util.spec_from_file_location("rsm", f"{IMM}/scripts/run_segment_memory_model.py")
M = importlib.util.module_from_spec(spec)
sys.modules["rsm"] = M
spec.loader.exec_module(M)

mode = os.environ["KO_MODE"]
assert mode in ("collect", "word", "mem", "both"), mode
means_path = os.environ["KO_MEANS"]
means = None if mode == "collect" else torch.load(means_path, map_location="cpu")
st = {"valid": None, "mask": None}
acc, sq, cnt = {}, {}, {}


def collect(key, x, m):
    """Adds x[m] ([N, H]) to the running sum, sum of squared norms and count."""
    rows = x[m].detach().double()
    if key not in acc:
        acc[key] = torch.zeros(x.shape[-1], dtype=torch.float64, device=x.device)
        sq[key] = 0.0
        cnt[key] = 0
    acc[key] += rows.sum(0)
    sq[key] += float((rows * rows).sum())
    cnt[key] += int(m.sum())


orig_span = M.SegmentMemoryLM.span_gru


def span_gru(self, *a, **k):
    out = orig_span(self, *a, **k)
    st["valid"] = out[1]  # [B, T] bool on the device
    return out


M.SegmentMemoryLM.span_gru = span_gru

orig_word = M.WordTier.forward


def word_forward(self, word_ends, word_in, segment_of, drop, at_last=None):
    add, word_out, word_states = orig_word(self, word_ends, word_in, segment_of, drop, at_last)
    valid = st["valid"]
    if mode == "collect":
        collect("word_last", add, valid & at_last)
        collect("word_other", add, valid & ~at_last)
    elif mode in ("word", "both"):
        last = means["word_last"].to(add.device, add.dtype)
        other = means["word_other"].to(add.device, add.dtype)
        add = torch.where(at_last[..., None], last, other).expand_as(add).contiguous()
    return add, word_out, word_states


M.WordTier.forward = word_forward

orig_block = M.GatedCrossAttentionBlock.forward
block_index = {}


def out_proj_hook(module, inputs, out):
    key = f"attn{module._ko_block}"
    if mode == "collect":
        collect(key, out, st["valid"] & st["mask"].any(dim=-1))
        return None
    if mode in ("mem", "both"):
        return means[key].to(out.device, out.dtype).expand_as(out).contiguous()
    return None


def block_forward(self, *a, **k):
    if id(self) not in block_index:
        block_index[id(self)] = len(block_index)
        self.out_proj._ko_block = block_index[id(self)]
        self.out_proj.register_forward_hook(out_proj_hook)
    if k.get("memory_mask") is None:
        raise RuntimeError("mean_ablate expects the parallel path's memory_mask")
    st["mask"] = k["memory_mask"]
    return orig_block(self, *a, **k)


M.GatedCrossAttentionBlock.forward = block_forward

sys.argv = [f"{IMM}/scripts/run_segment_memory_model.py"] + sys.argv[1:]
M.main()

if mode == "collect":
    out = {}
    for key in acc:
        mean = acc[key] / cnt[key]
        out[key] = mean.float().cpu()
        # How much of the add's energy is its average: |mean|^2 / E|x|^2.
        share = float(mean @ mean) / (sq[key] / cnt[key])
        print(f"mean {key}: n={cnt[key]} |mean|={float(mean.norm()):.4f} "
              f"rms|x|={(sq[key] / cnt[key]) ** 0.5:.4f} mean share of energy={share:.3f}", flush=True)
    torch.save(out, means_path)
print(f"ko_mode={mode} blocks_seen={len(block_index)}", flush=True)
