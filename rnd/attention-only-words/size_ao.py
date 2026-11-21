"""Trained parameters of imm configs (one CPU update on random text; those
with a gradient), for sizing the 10.8M pair.
    python -B size_ao.py RECORD H:E:L:HEADS:CHARS:POOL ... (POOL none: --no-word-records; mean: --word-pool mean --words-beyond-chars); --rnn-core none, --depth-records   (from the imm-char worktree)"""
import importlib.util, sys, torch
spec = importlib.util.spec_from_file_location("rsm", "scripts/run_segment_memory_model.py")
rsm = importlib.util.module_from_spec(spec); sys.modules["rsm"] = rsm; spec.loader.exec_module(rsm)
from imm.segments import walk, word_lengths
import numpy as np
torch.set_num_threads(2)
V = 96
rng = np.random.default_rng(0)
ids_np = rng.integers(0, V, 4000)
ids_np[rng.random(4000) < 0.18] = 0
segs = walk(word_lengths(ids_np, [0], 16, np.ones(len(ids_np), dtype=np.int64)))
ids = torch.from_numpy(ids_np)
for spec_ in sys.argv[2:]:
    h, e, l, heads, chars, pool = spec_.split(":"); h, e, l, heads, chars = map(int, (h, e, l, heads, chars)); tier = 0
    torch.manual_seed(0)
    m = rsm.SegmentMemoryLM(
        V, e, h, use_rope=True, rope_positions="char", memory_state="gru", memory_record=sys.argv[1],
        rnn_core="none", attn_interface_norm="none", mixing="gated-xattn", attention_layers=l, attention_heads=heads,
        input_rope=False, feedback_gate_bias=-6.0, feedback_delta_cap=1.0, carry_bound="gru", carry_scale=1.0,
        prior_strength=1.0, prior_residual_scale=1.0, prior_residual_l2=0.05, prior_residual_gate="none",
        prior_residual_gate_max=2.0, prior_residual_gate_features=0, prior_residual_state_features=[],
        skip_prior_gate="none", skip_prior_gate_max=0.5, skip_prior_gate_bias=0.0, skip_prior_channels=0,
        record_aux_weight=0.0, terminal_record_aux_weight=0.0, retrieval_aux_weight=0.0, utility_aux_weight=0.0,
        utility_temperature=1.0, prior_input="none", prior_evidence_width=0, dropout=0.0,
        word_tier=tier, tier_own_step=bool(tier), documents=True, char_memory=chars, word_records=pool != "none", depth_records=True, word_pool="last" if pool == "none" else pool, words_beyond_chars=pool == "mean",
    )
    with torch.no_grad():
        for b in m.gated_xattn_blocks:
            b.gate_attn.fill_(0.5); b.gate_mlp.fill_(0.5)
    st = rsm.split_streams(segs, 4)
    batch = [s[:32] for s in st]
    loss, tokens, *_ = m.segment_loss_streams(ids, batch, None, 600, carried_hidden=None, carry_hidden=True)
    (loss / tokens).backward()
    used = sum(p.numel() for p in m.parameters() if p.grad is not None)
    total = sum(p.numel() for p in m.parameters())
    print(f"h{h} e{e} L{l} heads{heads} chars{chars} words:{pool}: {used:,} used of {total:,}", flush=True)
