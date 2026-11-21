"""probe_split.py, with the run's training text on the GPU: slot 124's
10.8M run ran out of GPU memory at 7.25 GiB allocated, since the runner
keeps the 100M-character training text there as int64 (0.75 GiB), which
probe_split.py left out. Here a 100M-long int64 tensor stands in for it
through every measurement, the limit is 6.9 GiB in all, and the cap is
tried in finer steps (L from 704 down by 32).

In a slot, before imm's 10.8M run (slot 120 found no size whose
worst-case update fits the GPU in bf16, though all three fit the usual
updates): the same sizes, under --bf16, with --max-update-chars. Per size,
in the order of preference (h448 x 3 blocks, h512 x 2, h384 x 4):

  1. 3 warm-up and 10 timed updates on the 100M split's first 2M training
     characters (ms/update, peak), unsplit, as slot 120's probe;
  2. the cap: for 64 x L, L from 704 down to 384, one unsplit update whose
     64 streams are all a 64-segment window of at most L characters (the
     largest such update the cap lets through), carried memory and all;
     the first L whose peak stays under LIMIT;
  3. at that cap, the worst update (every stream the training text's longest
     64-segment window, 864 characters) run split, through grouped_update;
  4. scoring's worst case: the same 64 windows under no_grad, in float32
     (the final per-character scoring) and in bf16 (the evaluations in
     training; there are 50 held-out pieces, so 50 streams).

Prints CHOICE h= l= heads= tier= cap= for the first size whose cap is found
and whose steps 3 and 4 fit too, else CHOICE none.

    python -B probe_split.py WORST_WINDOW.txt   (from the imm-char worktree)
    PROBE_DEVICE=cpu PROBE_QUICK=1: a functional check on the CPU, h32, 4 streams
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

LIMIT = 6.9  # GiB allocated in all, the run's text included; slot 124's run failed at 7.25
dev = torch.device(os.environ.get("PROBE_DEVICE", "cuda"))
CUDA = dev.type == "cuda"
QUICK = bool(os.environ.get("PROBE_QUICK"))
STREAMS = 4 if QUICK else 64
V = 96
main = open("data/pg19/imm-split-100m/train_corpus.txt", encoding="utf-8").read(200_000 if QUICK else 2_000_000)
worst_text = open(sys.argv[1], encoding="utf-8").read()
text = main + worst_text
chars = sorted(set(text))
index = {c: i % V for i, c in enumerate(chars)}
ids_np = np.fromiter((index[c] for c in text), dtype=np.int64, count=len(text))
space = sorted({i % V for i, c in enumerate(chars) if c.isspace()})
segments = walk(word_lengths(ids_np, space, 16, np.ones(len(ids_np), dtype=np.int64)))
ids = torch.from_numpy(ids_np).to(dev)
# The run's train_ids: torch.tensor(..., dtype=torch.long, device=device), 100M characters.
text_on_gpu = torch.empty(1_000_000 if QUICK else 100_000_000, dtype=torch.long, device=dev)
normal = [s for s in segments if s[1] <= len(main)]
starts = np.array([s for s, _ in segments])
ends = np.array([e for _, e in segments])
spans = ends[63:] - starts[: len(segments) - 63]  # every 64-segment window, sliding
tail_from = len(normal)
w = tail_from + int(np.argmax(spans[tail_from:]))
worst = segments[w : w + 64]
print(f"normal segments={len(normal)}; worst window {spans[w]} characters; devices {dev}", flush=True)


def window_at_most(limit):
    """The 64-segment window with the largest span at most limit."""
    ok = np.where(spans <= limit)[0]
    i = int(ok[np.argmax(spans[ok])])
    return segments[i : i + 64], int(spans[i])


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


def sync():
    if CUDA:
        torch.cuda.synchronize()


def reset_peak():
    if CUDA:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak():
    return torch.cuda.max_memory_allocated() / 2**30 if CUDA else 0.0


def step(model, opt, batch, memory, carried, cap=0):
    opt.zero_grad(set_to_none=True)
    groups = rsm.stream_groups([(s[0][0], s[-1][1]) for s in batch], cap)
    if len(groups) > 1:
        _, tokens, memory, carried = rsm.grouped_update(model, ids, batch, groups, memory, carried, 128, carry_hidden=True)
    else:
        loss, tokens, memory, carried, _ = model.segment_loss_streams(
            ids, batch, memory, 128, carried_hidden=carried, carry_hidden=True
        )
        (loss / tokens).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return int(tokens), tuple(m.detach() for m in memory), carried.detach(), len(groups)


def timed(fn):
    sync()
    t0 = time.perf_counter()
    out = fn()
    sync()
    return out, 1000 * (time.perf_counter() - t0)


def try_peak(fn):
    reset_peak()
    try:
        out, ms = timed(fn)
        return out, ms, peak()
    except torch.OutOfMemoryError:
        reset_peak()
        return None, float("nan"), float("inf")


def probe(h, l, heads, tier, warm=3, timed_updates=10):
    model = make(h, l, heads, tier)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    stream_segments = rsm.split_streams(normal, STREAMS)
    memory = carried = None
    times, chars_done, longest = [], 0, 0
    reset_peak()
    for u in range(warm + timed_updates):
        batch = [segs[u * 64 : (u + 1) * 64] for segs in stream_segments]
        longest = max(longest, max(s[-1][1] - s[0][0] for s in batch))
        (tokens, memory, carried, _), ms = timed(lambda: step(model, opt, batch, memory, carried))
        if u >= warm:
            times.append(ms)
            chars_done += tokens
    ms = float(np.median(times))
    normal_peak = peak()
    name = f"h{h} L{l} heads{heads} tier{tier}"
    print(f"{name}: {ms:7.1f} ms/update, {chars_done / sum(times):6.1f}k chars/s, peak {normal_peak:.2f} GiB "
          f"(longest stream {longest}); 8 epochs ~{8 * ms / 1000 * 4232 * 1.09 / 3600:.1f} h unsplit", flush=True)
    cap = None
    for L in range(704, 383, -32):
        batch_win, span = window_at_most(L)
        out, ms_b, pk = try_peak(lambda: step(model, opt, [batch_win] * STREAMS, memory, carried))
        print(f"  cap 64 x {L}: an unsplit update of {STREAMS} x {span} characters: peak {pk:.2f} GiB, {ms_b:.0f} ms", flush=True)
        if out is not None:
            _, memory, carried, _ = out
        if pk < LIMIT:
            cap = STREAMS * L
            break
    if cap is None:
        print(f"  {name}: no cap fits", flush=True)
        return None
    out, ms_w, pk_w = try_peak(lambda: step(model, opt, [worst] * STREAMS, memory, carried, cap))
    groups = out[3] if out is not None else "?"
    print(f"  the worst update ({STREAMS} x {spans[w]}) split at cap {cap}: {groups} groups, peak {pk_w:.2f} GiB, {ms_w:.0f} ms",
          flush=True)
    if out is not None:
        _, memory, carried, _ = out
    model.eval()
    scoring = {}
    for label, dtype in (("fp32", None), ("bf16", torch.bfloat16)):
        model.autocast_dtype = dtype

        def score():
            with torch.no_grad():
                return model.segment_loss_streams(ids, [worst] * STREAMS, memory, 128, carried_hidden=carried, carry_hidden=True)

        _, ms_s, pk_s = try_peak(score)
        scoring[label] = pk_s
        print(f"  scoring's worst ({STREAMS} x {spans[w]}, no_grad, {label}): peak {pk_s:.2f} GiB, {ms_s:.0f} ms", flush=True)
    model.autocast_dtype = torch.bfloat16
    del model, opt, memory, carried
    reset_peak()
    if pk_w < LIMIT and all(v < LIMIT for v in scoring.values()):
        return cap
    print(f"  {name}: the split worst update or scoring doesn't fit", flush=True)
    return None


configs = ((32, 2, 4, 28),) if QUICK else ((448, 3, 8, 392), (512, 2, 8, 448), (384, 4, 8, 336))
for cfg in configs:
    try:
        cap = probe(*cfg)
    except torch.OutOfMemoryError as e:
        print(f"{cfg}: out of GPU memory in the normal updates ({str(e)[:100]})", flush=True)
        reset_peak()
        cap = None
    if cap is not None:
        h, l, heads, tier = cfg
        print(f"CHOICE h={h} l={l} heads={heads} tier={tier} cap={cap}")
        break
else:
    print("CHOICE none")
