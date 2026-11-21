"""In a slot, before imm's 10.8M run: imm with no prior, 1,024 character
records, 64 streams x 64 words, under --bf16 (the GRUs in float32), at
several sizes. Per size: 3 warm-up and 10 timed updates on the 100M
split's first 2M training characters (ms/update, peak allocated GPU
memory), then one update whose every stream is the training text's longest
64-segment window (864 characters, s10/worst_window.txt), carried memory
and all: the worst case's peak. Prints CHOICE: the first of h448 x 3
blocks, h512 x 2, h384 x 4 whose worst-case peak is under 7.0 GiB.

    python -B probe_bf16.py WORST_WINDOW.txt   (from the imm-char worktree)
"""
import importlib.util
import os
import sys
import time

import numpy as np
import torch

spec = importlib.util.spec_from_file_location("rsm", "scripts/run_segment_memory_model.py")
rsm = importlib.util.module_from_spec(spec)
sys.modules["rsm"] = rsm
spec.loader.exec_module(rsm)
from imm.segments import walk, word_lengths  # noqa: E402

LIMIT = 7.0  # GiB allocated, of the 7.7 the GPU gives
dev = torch.device("cuda")
V = 96
main = open("data/pg19/imm-split-100m/train_corpus.txt", encoding="utf-8").read(2_000_000)
worst_text = open(sys.argv[1], encoding="utf-8").read()
text = main + worst_text
chars = sorted(set(text))
index = {c: i % V for i, c in enumerate(chars)}
ids_np = np.fromiter((index[c] for c in text), dtype=np.int64, count=len(text))
space = sorted({i % V for i, c in enumerate(chars) if c.isspace()})
segments = walk(word_lengths(ids_np, space, 16, np.ones(len(ids_np), dtype=np.int64)))
ids = torch.from_numpy(ids_np).to(dev)
normal = [s for s in segments if s[1] <= len(main)]
tail = [s for s in segments if s[0] >= len(main)]
spans = [tail[i + 63][1] - tail[i][0] for i in range(len(tail) - 63)]
w = int(np.argmax(spans))
worst = tail[w : w + 64]
print(f"normal segments={len(normal)}; worst window {spans[w]} characters", flush=True)


def make(h, l, heads, tier):
    torch.manual_seed(0)
    model = rsm.SegmentMemoryLM(
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
    model.autocast_dtype = torch.bfloat16
    return model


def step(model, opt, batch, memory, carried):
    opt.zero_grad(set_to_none=True)
    loss, tokens, memory, carried, _ = model.segment_loss_streams(
        ids, batch, memory, 128, carried_hidden=carried, carry_hidden=True
    )
    (loss / tokens).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return int(tokens), tuple(m.detach() for m in memory), carried.detach()


def bench(h, l, heads, tier, streams=64, k=64, warm=3, timed=10):
    model = make(h, l, heads, tier)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    stream_segments = rsm.split_streams(normal, streams)
    memory = carried = None
    times, chars_done = [], 0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for u in range(warm + timed):
        batch = [segs[u * k : (u + 1) * k] for segs in stream_segments]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        tokens, memory, carried = step(model, opt, batch, memory, carried)
        torch.cuda.synchronize()
        if u >= warm:
            times.append(time.perf_counter() - t0)
            chars_done += tokens
    ms = 1000 * float(np.median(times))
    peak = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    try:
        step(model, opt, [worst] * streams, memory, carried)
        torch.cuda.synchronize()
        worst_peak = torch.cuda.max_memory_allocated() / 2**30
    except torch.OutOfMemoryError:
        worst_peak = float("inf")
    epoch_min = ms / 1000 * 4232 * 1.09 / 60  # slot 115: train_sec 1,013 against 4,232 x 219 ms
    print(
        f"h{h} L{l} heads{heads} tier{tier} bf16: {ms:7.1f} ms/update  {chars_done / sum(times) / 1e3:6.1f}k chars/s  "
        f"peak {peak:.2f} GiB, worst case {worst_peak:.2f} GiB  epoch ~{epoch_min:.0f} min, 8 epochs ~{8 * epoch_min / 60:.1f} h",
        flush=True,
    )
    del model, opt, memory, carried
    torch.cuda.empty_cache()
    return worst_peak


QUICK = bool(os.environ.get("PROBE_QUICK"))  # a functional check: h192 only, 8 streams
results = {}
for cfg in ((192, 2, 4, 168),) if QUICK else ((192, 2, 4, 168), (448, 3, 8, 392), (512, 2, 8, 448), (384, 4, 8, 336)):
    try:
        results[cfg] = bench(*cfg, **(dict(streams=8) if QUICK else {}))
    except torch.OutOfMemoryError as e:
        print(f"{cfg}: out of GPU memory in the normal updates ({str(e)[:100]})", flush=True)
        results[cfg] = float("inf")
        torch.cuda.empty_cache()
for cfg in ((448, 3, 8, 392), (512, 2, 8, 448), (384, 4, 8, 336)):
    if results.get(cfg, float("inf")) < LIMIT:
        h, l, heads, tier = cfg
        print(f"CHOICE h={h} l={l} heads={heads} tier={tier}")
        break
else:
    print("CHOICE none")
