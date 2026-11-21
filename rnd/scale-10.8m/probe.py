"""In a slot, before the 10.8M pair: one training update of imm with no
prior, 1,024 character records, 64 streams x 64 words, at several sizes,
on the GPU, with real word lengths from the 100M split's first 2M training
characters. Per size: 3 warm-up and 10 timed updates, ms/update and peak
allocated GPU memory. "tf32" also lets fp32 matmuls use TF32 (imm's runs
so far don't). Slot 115's model (h192, 2 blocks) is the yardstick: 219
ms/update in slot 115's probe, 1,013 s an epoch in its run.

    python -B probe.py   (from the imm-char worktree)
"""
import importlib.util
import sys
import time

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("rsm", "scripts/run_segment_memory_model.py")
rsm = importlib.util.module_from_spec(spec)
sys.modules["rsm"] = rsm
spec.loader.exec_module(rsm)
from imm.segments import walk, word_lengths  # noqa: E402

dev = torch.device("cuda")
V = 96
text = open("data/pg19/imm-split-100m/train_corpus.txt", encoding="utf-8").read(2_000_000)
chars = sorted(set(text))
index = {c: i % V for i, c in enumerate(chars)}
ids_np = np.fromiter((index[c] for c in text), dtype=np.int64, count=len(text))
space = sorted({i % V for i, c in enumerate(chars) if c.isspace()})
segments = walk(word_lengths(ids_np, space, 16, np.ones(len(ids_np), dtype=np.int64)))
ids = torch.from_numpy(ids_np).to(dev)


def make(h, l, heads, tier):
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
        word_tier=tier, tier_own_step=True, documents=True, char_memory=1024,
    ).to(dev)


def bench(h, l, heads, tier, tf32, streams=64, k=64, warm=3, timed=10):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    model = make(h, l, heads, tier)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    stream_segments = rsm.split_streams(segments, streams)
    memory = carried = None
    times, chars_done = [], 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for u in range(warm + timed):
        batch = [segs[u * k : (u + 1) * k] for segs in stream_segments]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss, tokens, memory, carried, _ = model.segment_loss_streams(
            ids, batch, memory, 128, carried_hidden=carried, carry_hidden=True
        )
        (loss / tokens).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        memory = tuple(m.detach() for m in memory)
        carried = carried.detach()
        torch.cuda.synchronize()
        if u >= warm:
            times.append(time.perf_counter() - t0)
            chars_done += int(tokens)
    ms = 1000 * float(np.median(times))
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"h{h} L{l} heads{heads} tier{tier} {'tf32' if tf32 else 'fp32'}: {ms:7.1f} ms/update  "
        f"{chars_done / sum(times) / 1e3:6.1f}k chars/s  peak {peak:.2f} GiB allocated  "
        f"epoch ~{ms / 219.0 * 1013 / 60:.0f} min (scaled from slot 115)",
        flush=True,
    )
    del model, opt, memory, carried
    torch.cuda.empty_cache()


for cfg in (
    (192, 2, 4, 168, False),
    (192, 2, 4, 168, True),
    (448, 3, 8, 392, False),
    (448, 3, 8, 392, True),
    (512, 2, 8, 448, False),
    (384, 4, 8, 336, False),
):
    try:
        bench(*cfg)
    except torch.OutOfMemoryError as e:
        print(f"{cfg}: out of GPU memory ({str(e)[:120]})", flush=True)
        torch.cuda.empty_cache()
