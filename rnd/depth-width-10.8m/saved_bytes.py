"""Bytes autograd saves for backward in one unsplit update, per character,
for imm shapes with and without --depth-records (CPU, fp32, 4 streams).
Tensors are counted once per storage. Run from an imm worktree with
--depth-records.
    python -B saved_bytes.py"""
import importlib.util, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch
spec = importlib.util.spec_from_file_location("rsm", "scripts/run_segment_memory_model.py")
rsm = importlib.util.module_from_spec(spec); sys.modules["rsm"] = rsm; spec.loader.exec_module(rsm)
from imm.segments import walk, word_lengths
torch.set_num_threads(2)
V = 96
text = open("data/pg19/imm-split-100m/train_corpus.txt", encoding="utf-8").read(400_000)
chars = sorted(set(text)); index = {c: i % V for i, c in enumerate(chars)}
ids_np = np.fromiter((index[c] for c in text), dtype=np.int64, count=len(text))
space = sorted({i % V for i, c in enumerate(chars) if c.isspace()})
segs = walk(word_lengths(ids_np, space, 16, np.ones(len(ids_np), dtype=np.int64)))
ids = torch.from_numpy(ids_np)
STREAMS = 4
streams = rsm.split_streams(segs, STREAMS)

def make(h, l, heads, tier, depth):
    torch.manual_seed(0)
    return rsm.SegmentMemoryLM(
        V, h, h, use_rope=True, rope_positions="char", memory_state="gru", memory_record="written",
        rnn_core="gru", attn_interface_norm="none", mixing="gated-xattn", attention_layers=l, attention_heads=heads,
        input_rope=False, feedback_gate_bias=-6.0, feedback_delta_cap=1.0, carry_bound="gru", carry_scale=1.0,
        prior_strength=1.0, prior_residual_scale=1.0, prior_residual_l2=0.05, prior_residual_gate="none",
        prior_residual_gate_max=2.0, prior_residual_gate_features=0, prior_residual_state_features=[],
        skip_prior_gate="none", skip_prior_gate_max=0.5, skip_prior_gate_bias=0.0, skip_prior_channels=0,
        record_aux_weight=0.0, terminal_record_aux_weight=0.0, retrieval_aux_weight=0.0, utility_aux_weight=0.0,
        utility_temperature=1.0, prior_input="none", prior_evidence_width=0, dropout=0.0,
        word_tier=tier, tier_own_step=True, documents=True, char_memory=1024, depth_records=depth)

for h, l, heads, tier in ((448, 3, 8, 392), (344, 6, 8, 301), (344, 3, 8, 301), (448, 6, 8, 392)):
    for depth in (False, True):
        model = make(h, l, heads, tier, depth); model.train()
        memory = carried = None
        with torch.no_grad():  # fill the carried records first: 20 updates
            for u in range(20):
                b = [s[u * 64:(u + 1) * 64] for s in streams]
                _, _, memory, carried, _ = model.segment_loss_streams(ids, b, memory, 128, carried_hidden=carried, carry_hidden=True)
        b = [s[20 * 64:21 * 64] for s in streams]
        n = sum(x[-1][1] - x[0][0] for x in b)
        seen, total = set(), [0]
        def pack(t):
            st = t.untyped_storage()
            key = (st.data_ptr(), st.nbytes())
            if key not in seen:
                seen.add(key); total[0] += st.nbytes()
            return t
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            loss, tokens, *_ = model.segment_loss_streams(ids, b, memory, 128, carried_hidden=carried, carry_hidden=True)
        print(f"h{h} L{l} depth={int(depth)}: {total[0] / 2**20:8.1f} MiB saved for {n} characters, "
              f"{total[0] / n / 1024:6.1f} KiB a character", flush=True)
        del model, memory, carried, loss
