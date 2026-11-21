"""Small GPU check of --bf16 (model.autocast_dtype): which modules run in
bfloat16, first loss against float32 on the same weights (gates opened to
0.5), and a few training updates. h192, 8 streams x 32 words.
    python -B bf16_check.py   (from the imm-char worktree)"""
import importlib.util, sys
import numpy as np, torch
spec = importlib.util.spec_from_file_location("rsm", "scripts/run_segment_memory_model.py")
rsm = importlib.util.module_from_spec(spec); sys.modules["rsm"] = rsm; spec.loader.exec_module(rsm)
from imm.segments import walk, word_lengths
dev = torch.device("cuda")
V = 96
text = open("data/pg19/imm-split-100m/train_corpus.txt", encoding="utf-8").read(400_000)
chars = sorted(set(text)); index = {c: i % V for i, c in enumerate(chars)}
ids_np = np.fromiter((index[c] for c in text), dtype=np.int64, count=len(text))
space = sorted({i % V for i, c in enumerate(chars) if c.isspace()})
segments = walk(word_lengths(ids_np, space, 16, np.ones(len(ids_np), dtype=np.int64)))
ids = torch.from_numpy(ids_np).to(dev)
torch.manual_seed(0)
model = rsm.SegmentMemoryLM(
    V, 192, 192, use_rope=True, rope_positions="char", memory_state="gru", memory_record="written",
    rnn_core="gru", attn_interface_norm="none", mixing="gated-xattn", attention_layers=2, attention_heads=4,
    input_rope=False, feedback_gate_bias=-6.0, feedback_delta_cap=1.0, carry_bound="gru", carry_scale=1.0,
    prior_strength=1.0, prior_residual_scale=1.0, prior_residual_l2=0.05, prior_residual_gate="none",
    prior_residual_gate_max=2.0, prior_residual_gate_features=0, prior_residual_state_features=[],
    skip_prior_gate="none", skip_prior_gate_max=0.5, skip_prior_gate_bias=0.0, skip_prior_channels=0,
    record_aux_weight=0.0, terminal_record_aux_weight=0.0, retrieval_aux_weight=0.0, utility_aux_weight=0.0,
    utility_temperature=1.0, prior_input="none", prior_evidence_width=0, dropout=0.0,
    word_tier=168, tier_own_step=True, documents=True, char_memory=1024,
).to(dev)
with torch.no_grad():
    for b in model.gated_xattn_blocks:
        b.gate_attn.fill_(0.5); b.gate_mlp.fill_(0.5)
seen = {}
def hook(name):
    def f(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        seen.setdefault(name, set()).add(str(o.dtype).replace("torch.", ""))
    return f
for name, mod in model.named_modules():
    if isinstance(mod, (torch.nn.GRU, torch.nn.Linear, torch.nn.LayerNorm, torch.nn.Embedding)):
        mod.register_forward_hook(hook(f"{type(mod).__name__} {name}"))
streams = rsm.split_streams(segments, 8)
batch = [s[:32] for s in streams]
model.train()
losses = {}
for dt in (None, torch.bfloat16):
    model.autocast_dtype = dt
    seen.clear()
    loss, tokens, memory, hidden, _ = model.segment_loss_streams(ids, batch, None, 128, carried_hidden=None, carry_hidden=True)
    losses[dt] = float(loss / tokens)
    print(f"{'bf16' if dt else 'fp32'}: first loss {losses[dt]:.6f} loss {loss.dtype} hidden {hidden.dtype} "
          f"memory {[str(m.dtype).replace('torch.', '') for m in memory]}")
    if dt:
        for k in sorted(seen):
            print(f"   {k}: {sorted(seen[k])}")
print(f"bf16 against fp32: {100 * (losses[torch.bfloat16] / losses[None] - 1):+.3f}%")
model.autocast_dtype = torch.bfloat16
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
memory = hidden = None
for u in range(6):
    b = [s[u * 32 : (u + 1) * 32] for s in streams]
    opt.zero_grad(set_to_none=True)
    loss, tokens, memory, hidden, _ = model.segment_loss_streams(ids, b, memory, 128, carried_hidden=hidden, carry_hidden=True)
    (loss / tokens).backward()
    g = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    memory = tuple(m.detach() for m in memory); hidden = hidden.detach()
    print(f"update {u}: loss {float(loss / tokens):.4f} grad norm {float(g):.3f} grads "
          f"{sorted({str(p.grad.dtype) for p in model.parameters() if p.grad is not None})}")
