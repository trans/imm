"""Trained parameters of --tree-levels configs: one CPU update on random
text, --documents on (so the start state counts, as in size_ao.py); the
parameters that get a gradient. Run from the imm-tree worktree:
    sh $S/tree/env.sh python3 -B $S/tree/size_tree.py H:E:LEVELS:WIDTH ...
(WIDTH 0: the hidden size). Slot 127, the comparison, has 1,588,157."""
import importlib.util
import sys

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("rsm", "scripts/run_segment_memory_model.py")
rsm = importlib.util.module_from_spec(spec)
sys.modules["rsm"] = rsm
spec.loader.exec_module(rsm)
from imm.segments import walk, word_lengths  # noqa: E402

torch.set_num_threads(2)
V = 96  # the 100m split's vocabulary
SLOT_127 = 1_588_157
rng = np.random.default_rng(0)
ids_np = rng.integers(0, V, 6000)
ids_np[rng.random(6000) < 0.18] = 0
segs = walk(word_lengths(ids_np, [0], 16, np.ones(len(ids_np), dtype=np.int64)))
ids = torch.from_numpy(ids_np)
for config in sys.argv[1:]:
    h, e, levels, width = map(int, config.split(":"))
    torch.manual_seed(0)
    m = rsm.SegmentMemoryLM(
        V, e, h, use_rope=True, rope_positions="char", memory_state="gru", memory_record="written",
        rnn_core="gru", attn_interface_norm="none", mixing="gated-xattn", attention_layers=0, attention_heads=4,
        input_rope=False, feedback_gate_bias=-6.0, feedback_delta_cap=1.0, carry_bound="gru", carry_scale=1.0,
        prior_strength=1.0, prior_residual_scale=1.0, prior_residual_l2=0.05, prior_residual_gate="none",
        prior_residual_gate_max=2.0, prior_residual_gate_features=0, prior_residual_state_features=[],
        skip_prior_gate="none", skip_prior_gate_max=0.5, skip_prior_gate_bias=0.0, skip_prior_channels=0,
        record_aux_weight=0.0, terminal_record_aux_weight=0.0, retrieval_aux_weight=0.0, utility_aux_weight=0.0,
        utility_temperature=1.0, prior_input="none", prior_evidence_width=0, dropout=0.0,
        documents=True, word_records=False, tree_levels=levels, tree_width=width,
    )
    with torch.no_grad():
        if m.tree is not None: m.tree.gates.fill_(0.5)  # open, so every level's weights get a gradient
    st = rsm.split_streams(segs, 2)
    batch = [s[:600] for s in st]  # ~2,000 characters a stream: every level up to 10 steps
    loss, tokens, *_ = m.segment_loss_streams(ids, batch, None, 128, carried_hidden=m.start_hidden(2, torch.device("cpu")), carry_hidden=True)
    (loss / tokens).backward()
    used = {n: p.numel() for n, p in m.named_parameters() if p.grad is not None}
    total = sum(p.numel() for p in m.parameters())
    parts = {}
    for n, c in used.items():
        key = n.split(".")[0] if not n.startswith("tree.") else ".".join(n.split(".")[:2])
        parts[key] = parts.get(key, 0) + c
    n_used = sum(used.values())
    print(
        f"h{h} e{e} levels{levels} width{m.tree.width if m.tree is not None else 0}: {n_used:,} trained of {total:,} "
        f"({n_used - SLOT_127:+,} against slot 127's {SLOT_127:,}); "
        + ", ".join(f"{k} {v:,}" for k, v in parts.items()),
        flush=True,
    )
