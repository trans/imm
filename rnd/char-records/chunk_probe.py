"""In a slot, before the 1,024-character run: one training update of slot
113's model (no prior, 64 streams x 64 words, 128 word records, word tier
168, two blocks) on the GPU, with real word lengths from the 100M split's
first 2M training characters. Per variant (character window N, run length
C; C 0 is slot 113's whole-update mask): 3 warm-up and 10 timed updates,
ms/update and peak allocated GPU memory. And whether runs give the
whole-update mask's loss on the GPU too (first update, same weights,
the gates opened to 0.5).

    python -B chunk_probe.py   (from the imm-char worktree)
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

import os
dev = torch.device(os.environ.get("PROBE_DEVICE", "cuda"))
V = 96
text = open("data/pg19/imm-split-100m/train_corpus.txt", encoding="utf-8").read(2_000_000)
chars = sorted(set(text))
index = {c: i % V for i, c in enumerate(chars)}
ids_np = np.fromiter((index[c] for c in text), dtype=np.int64, count=len(text))
space = sorted({i % V for i, c in enumerate(chars) if c.isspace()})
segments = walk(word_lengths(ids_np, space, 16, np.ones(len(ids_np), dtype=np.int64)))
ids = torch.from_numpy(ids_np).to(dev)
print(f"segments={len(segments)} mean_len={np.mean([e - s for s, e in segments]):.2f}", flush=True)


def make(n, chunk):
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
        word_tier=168, tier_own_step=True, documents=True, char_memory=n, char_chunk=chunk,
    ).to(dev)
    # The gates start at 0, so at first the attention changes nothing: open
    # them, so the first loss tells the reads apart (timing unchanged).
    with torch.no_grad():
        for block in model.gated_xattn_blocks:
            block.gate_attn.fill_(0.5)
            block.gate_mlp.fill_(0.5)
    return model


def bench(n, chunk, streams=64, k=64, warm=3, timed=10):
    model = make(n, chunk)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    stream_segments = rsm.split_streams(segments, streams)
    memory = carried = None
    times, chars_done, first_loss, longest = [], 0, None, 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for u in range(warm + timed):
        batch = [segs[u * k : (u + 1) * k] for segs in stream_segments]
        longest = max(longest, max(s[-1][1] - s[0][0] for s in batch))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss, tokens, memory, carried, _ = model.segment_loss_streams(
            ids, batch, memory, 128, carried_hidden=carried, carry_hidden=True
        )
        if u == 0:
            first_loss = float(loss.detach() / tokens)
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
        f"N={n:5d} C={chunk:4d}: {ms:6.1f} ms/update  {chars_done / sum(times) / 1e3:6.1f}k chars/s  "
        f"peak {peak:.2f} GiB allocated  first loss {first_loss:.6f}  longest stream {longest}",
        flush=True,
    )
    del model, opt, memory, carried
    return first_loss


losses = {}
for n, chunk in ((256, 0), (256, 128), (1024, 128), (1024, 256), (1024, 0)):
    try:
        losses[(n, chunk)] = bench(n, chunk)
    except torch.OutOfMemoryError as e:
        print(f"N={n:5d} C={chunk:4d}: out of GPU memory ({str(e)[:120]})", flush=True)
        torch.cuda.empty_cache()
for n in (256, 1024):
    base = losses.get((n, 0))
    for (m, c), x in losses.items():
        if m == n and c and base is not None:
            print(f"N={n} C={c}: first loss differs from C=0 by {abs(x - base):.2e}")
