---
title:    The 10.8M pair, imm and the transformer at ~11M parameters
kind:     experiment
status:   concluded
outcome:  negative
question: >-
  At ~1.6M imm with 1,024 character records leads the 1.84M transformer.
  Does the lead hold at ~10.8M parameters, with no prior and the same 798M
  characters seen?
answer: >-
  No. The 10.8M transformer (block 1,024, dropout 0) scores 2.7458. imm at
  10,913,495 parameters (GRU 448, 3 blocks of 8 heads, word tier 392, 1,024
  character records, trained in bf16) scores 2.8188, +2.66% ±0.13 behind.
  On the same code at the small size imm leads by 4.11% ±0.12 (3.0552
  against 3.1862). From the small size to 10.8M imm gains 7.74% ±0.12 (6.9×
  the parameters), the transformer 13.8% (5.6×), so the transformer
  overtakes imm between ~2M and ~11M. Dropout 0.1 costs the 10.8M
  transformer 1.5% (2.7877).
opened:   2026-11-19
updated:  2026-11-21
code: >-
  The transformer: air's experiments/prior_scaling/prior_transformer.py
  (air main 7b73806), slot 117; that slot's imm probe ran at char-records
  7f8238e, whose config the transformer read: tag rnd/scale-10.8m/slot117
  on 7f8238e. imm, branch bf16: slot 120 at 0d208bd (probe only; tag
  rnd/scale-10.8m/slot120), slot 125 at cac43fb (failed at its first split
  update; tag rnd/scale-10.8m/slot125), slot 126 at 00b6a52 (the run; tag
  rnd/scale-10.8m/slot126). Slots 123 and 124 also tried the 10.8M run;
  their tags are under rope-far-positions.
data:     PG-19 100M split (rnd/README.md), configs/pg19split100m_noprior.json
slots:    117, 120, 125, 126 (and 123, 124 in rope-far-positions)
tags:     [no-prior, scale, transformer-baseline, bf16, gpu-memory]
related:  [char-records, rope-far-positions, depth-records-10.8m, half-lr-10.8m]
---

# The 10.8M pair

## Question

At ~1.6M parameters with 1,024 character records, imm led the block-1,024
transformer by 4.46% per character ([char-records](../char-records/)). The
plan was to see next whether its lead on new words and piece starts holds
at scale. The user approved the 10.8M pair (2026-11-19), with dropout tuned
on both sides.

## Design

- **The transformer:** air's `s10.8m_none` shape, 6 layers × 384, 6 heads;
  lr 1e-3 cosine to 1e-4, warmup 100, weight decay 0.1, seed 1337; no
  prior; block 1,024, batch 16; 48,728 iterations (imm's 798M characters);
  bf16. 11,051,904 parameters. Dropout 0.0 and 0.1. Per-character losses
  scored in fp32 on the CPU (`../no-prior-pair-1.8m/score_tx_noprior.py`).
- **imm:** GRU and embedding 448, 3 blocks, 8 heads of 56, word tier 392,
  1,024 character records: 10,913,495 trained parameters (`size.py`). It
  scales the way the transformer's 1.84M → 10.8M does: depth ×1.5, width
  ×2.3. No prior, dropout 0, lr 1e-3 cosine to 1e-4 (the 1.59M's
  settings), 64 streams × 64 words, 8 epochs, scored in fp32.
- **Fitting it on the 8 GB GPU** took six slots:
  - **Slot 117** (2026-11-19 22:43 UTC – 2026-11-20 00:56 UTC,
    `tx10_slot.sh`): `probe.py` found imm at ~10.8M doesn't fit in fp32.
    GRU 448 × 3 blocks and 384 × 4 ran out of memory, with TF32 or
    without. 512 × 2 fit at 6.37 GiB and 637 ms/update, but only on
    typical updates: an update's longest stream passes 600 characters in
    ~9% of updates and reaches 825 (`window_lengths.py`). Then the
    transformer at dropout 0.0 and 0.1.
  - **The user chose bf16 for imm** (2026-11-19), over fp32 at 32 streams
    or at 32 words an update. `--bf16` (`0d208bd`) runs each update under
    autocast in bfloat16 and keeps the GRUs in float32 (rope-far-positions
    has its check).
  - **Slot 120** (`0d208bd`, `imm10_slot.sh`, `probe_bf16.py`), released
    after 21 s: in bf16 every size fits the usual update, but none fits
    the worst (64 streams of 864 characters, `worst_window.txt` from
    `worst_window.py`).
  - **`--max-update-chars N`** (`1edcb56`): past N characters (streams ×
    longest stream) an update runs as the fewest equal groups of streams
    that fit, each group's forward and backward in turn. The streams don't
    interact, so the gradient is the whole update's. How often an update
    splits (`split_share.py`): 1.3% at 64 × 704, 4.0% at 64 × 640, 13.9% at
    64 × 576, 32.6% at 64 × 512.
  - **Slots 123 and 124** chose GRU 448 × 3 at a cap of 40,960 (64 × 640);
    slot 123's bf16 check failed and slot 124's 10.8M run ran out of GPU
    memory in its first updates. Both are in
    [rope-far-positions](../rope-far-positions/). The probe had left out
    the 100M training text the runner keeps on the GPU as int64 (0.75 GiB).
  - **Slot 125** (`cac43fb`, `imm10d_slot.sh`, `probe_split2.py`, which
    holds a 100M-long int64 tensor on the GPU and keeps everything under
    6.9 GiB): cap 36,864 (64 × 576). The run failed in its first split
    update ("backward through the graph a second time"): with
    `--documents`, a stream that begins a piece takes the start state, one
    graph shared by every group. **Fixed in `00b6a52`**: shared inputs
    enter the groups as leaves, and their gradient goes back once.
  - **Slot 126** (`00b6a52`, `imm10e_slot.sh`): the same probe and cap,
    then the 10.8M run, 2026-11-20 04:33–08:37 UTC. 560 of each epoch's
    4,232 updates split in 2.
- The script went on to rerun the 1.59M model on the same code. The user
  moved that rerun behind another project's runs (2026-11-20), so
  `stop_after_10m.sh` stopped it at its start and it ran as slot 127
  ([rope-far-positions](../rope-far-positions/)).

## How to rerun

From a worktree at the slot's tag (rnd/README.md):

| script | slot | what it ran |
|---|---|---|
| `tx10_slot.sh` | 117 | `probe.py` (imm in fp32 at several sizes), then the 10.8M transformer at dropout 0.0 and 0.1, then their per-character losses |
| `imm10_slot.sh` | 120 | `probe_bf16.py` (bf16, usual and worst-case updates); no size fit, so it stopped |
| `imm10d_slot.sh` | 125 | `probe_split2.py`, then the 10.8M run (failed at its first split update) |
| `imm10e_slot.sh` | 126 | `probe_split2.py`, the 10.8M run and its fp32 per-character losses, then the 1.59M rerun (stopped) |
| `stop_after_10m.sh` | 126 | stopped the 1.59M rerun at its start, so slot 126 ended after the 10.8M scoring |

Helpers, run on the CPU: `size.py` (trained parameters of a config),
`window_lengths.py` and `split_share.py` (stream lengths in the training
text), `worst_window.py` (writes `worst_window.txt`, the longest
64-word window and 2,000 characters before it, which the probes and
`../depth-records-10.8m/probe_depth.py` read). The scripts of slots 123
and 124 (`imm10b_slot.sh`, `imm10c_slot.sh`, `probe_split.py`) are in
[rope-far-positions](../rope-far-positions/).

- Pairings: `scripts/rnd/rpair.sh A.npz B.npz OUT.txt`. The bottleneck
  tables came from `../no-prior-pair-1.8m/bottleneck.py`.
- Comments that quoted people word for word are paraphrased in the copied
  scripts. No command line changed.

**Results files** (`results/`): slot logs (`slot_117.log`,
`slot_120.log`, `slot_125.log`, `slot_126.log`, `stop_after_10m.log`),
probes (`probe_117.log`, `probe_bf16_120.log`, `probe_split_125.log`,
`probe_split_126.log`), memory samples (`rss_120.log`, `rss_125.log`,
`rss_126.log`), `split_share.txt`, the transformers' training logs and
result JSON (`tx10.8m_np_100m_b1024_do{0,1}.{log,json}`, `_cl.log`), and
the pairings.

**Checkpoints and per-character losses:**

- imm 10.8M (slot 126), in `runs/`:
  `pg19s100m_h448_par_l3_w392_own_np_doc_mm128_do0_cm1024_rel_bf16_e8`
  (`.log`, and `…_e8_pg19s100m_…_e8.csv`, `.pt`, `.best.pt`) and
  `char_losses_pg19s100m_h448_par_l3_w392_own_np_doc_mm128_do0_cm1024_rel_bf16_e8_best.npz`.
- The failed starts' logs: `runs/…_rel_bf16_e8_oom124.log` (slot 124) and
  `runs/…_rel_bf16_e8_err125.log` (slot 125).
- The 10.8M transformers were written to the session's scratch directory
  (`$S/s10/`) and copied into `runs/` on 2026-11-21, unchanged:
  `runs/tx10.8m_np_100m_b1024_do{0,1}.pt` (with `.log` and `.json`) and
  `runs/char_losses_tx10.8m_np_100m_b1024_do{0,1}.npz`.

## Results

Percentages are A's change in held-out perplexity per character against B
(`rpair.sh`), ± one standard error.

### 2026-11-19: the 10.8M transformer (slot 117)

| no prior, 798M characters | params | held out (fp32 per character) | train loss (last 10 logs) | wall time |
|---|---:|---:|---:|---:|
| **block 1,024, dropout 0.0** | 11,051,904 | **2.7458** (in training 2.7462; best 2.7445 at 47,500) | 0.910 | 3,922 s |
| block 1,024, dropout 0.1 | 11,051,904 | 2.7877 | 0.999 | 4,019 s |
| 1.84M, block 1,024, dropout 0 (slot 107) | 1,986,240 | 3.1862 | 1.114 | ~990 s |

- **Dropout 0 is still the better setting at 10.8M**: 0.1 costs 1.5%,
  even with dropout 0's training loss 0.10 nats under held-out (1.010).
  Held-out was still falling at the end of both runs' cosine.
- **Against the 1.59M imm with 1,024 character records (slot 115), the
  10.8M transformer is 10.86% ±0.14 ahead** (`imm1.59m_cm1024_vs_tx10.8m_do0.txt`,
  `imm1.59m_cm1024_bottleneck_tx10.8m_do0.txt`). It leads in every bucket,
  most on words new in the piece (+14.1%) and repeats 256+ words back
  (+13.1%), least on repeats 64–256 words back (+7.4%) and a piece's first
  256 characters (+2.1%).
- It's below the d24 prior's 3.068 on these characters, by 10.5%.
- 6.7× the 1.84M transformer's time, for 5.6× the parameters.

### 2026-11-20: imm at 10.8M (slot 126)

| no prior, 798M characters, dropout 0 | params | held out (fp32) | train loss | wall time |
|---|---:|---:|---:|---:|
| **imm 10.8M** | 10,913,495 | **2.8188** | 0.939 | 14,642 s |
| transformer 10.8M, block 1,024 | 11,051,904 | **2.7458** | 0.910 | 3,922 s |
| **imm 1.59M, same code, bf16** (slot 127) | 1,588,157 | **3.0552** | 1.069 | 4,950 s |
| imm 1.59M, before the RoPE fix, fp32 (slot 115) | 1,588,157 | 3.0440 | 1.060 | 8,243 s |
| transformer 1.84M, block 1,024 | 1,986,240 | 3.1862 | 1.114 | ~990 s |

- **The transformer leads at 10.8M by 2.66% ±0.13**
  (`imm10.8m_vs_tx10.8m_do0.txt`, `imm10.8m_bottleneck_tx10.8m_do0.txt`).
  At ~1.6–2M imm leads by 4.11% ±0.12 on the same code
  (`imm1.59m_fixed_vs_tx1.84m_b1024_do0.txt`; 4.46% before the fix).
- **From the small size to 10.8M imm gains 7.74% ±0.12** (6.9× the
  parameters; `imm10.8m_vs_imm1.59m_fixed.txt`) **and the transformer
  13.8%** (5.6×). So the transformer overtakes imm between ~2M and ~11M.
- **The change is nearly uniform:** 6–8 points in almost every bucket
  (small, same code: −3.8% on new words, −4.7% on repeated ones, −3.7%
  inside words; 10.8M: +4.3%, +1.7%, +2.3%). No single mechanism stopped
  scaling; the whole model gains less per parameter.
- **imm still leads** in a piece's first 256 characters (−5.9% ±0.90;
  small: −8.6%), after a sentence's end (−8.2% ±1.6), on targets the prior
  finds hardest (≥ 6 nats, −8.8% ±3.6) and on repeats 256–1,000 characters
  back (−1.6% ±0.38; third letter on, −4.0%).
- **It trails most** on punctuation and digits (+6.7%), new words (+4.3%;
  second letter +5.9%), repeats 1,000+ characters back (+4.8%) and 256+
  words back (+4.9%).
- 1.59M → 10.8M imm, paired: −7.74%, between −7.3% and −8.8% in every
  bucket of place in the piece and word new or repeated. Most on
  punctuation and digits (−19.4%) and capitals (−11.4%); least on spaces
  (−2.2%) and on new words the prior finds hard (≥ 3 nats: +0.5%).
- Train against held-out: imm 0.939 against 1.036 nats (gap 0.097; 0.048
  at 1.59M); the transformer 0.910 against 1.010 (0.100), where dropout
  0.1 cost 1.5%. imm's dropout and learning rate are untuned at this size.
- Held-out by epoch: 3.182 / 3.024 / 2.951 / 2.905 / 2.868 / 2.841 / 2.823
  / 2.819, still falling at the cosine's end.
- 3.7× the transformer's wall time, both in bf16.

### 2026-11-21: notes on the record

- **The train losses for the bf16 runs are a different measure from the
  others.** imm's 0.939 (10.8M) and 1.069 (slot 127) are the running
  training loss at step 30,000, from the logs' last `progress` lines. The
  epoch-8 means in the runs' CSVs are 0.9305 and 1.0604. Slot 115's 1.060
  is its epoch-8 mean (1.0596). The transformers' figures are the mean of
  their last 10 logged losses. With the epoch means, imm's train-to-held-out
  gap is 0.106 nats at 10.8M and 0.057 at 1.59M, not 0.097 and 0.048.
- `imm10.8m_vs_imm1.59m_old.txt` (10.8M against slot 115, −7.40%) is
  copied too; the notes cite only the same-code pairing.
- Counted from the 10.8M checkpoint's optimizer state, imm trains
  10,914,335 parameters; `size.py`'s one-update count gives 10,913,495.

## What we learned

- At ~11M parameters, no prior, the transformer is ahead of imm by 2.66%
  per character and about 3.7× faster.
- imm's loss of ground is spread over every bucket, not one mechanism.
  imm keeps its lead at piece starts, sentence ends and the hardest
  targets.
- Two follow-ups at 10.8M: depth records
  ([depth-records-10.8m](../depth-records-10.8m/), −0.65%) and half the
  learning rate ([half-lr-10.8m](../half-lr-10.8m/), no gain).
- **One seed** each. Never tested: imm's dropout at 10.8M (a dropout-0.1
  run was proposed and set aside for the architecture runs) and second
  seeds.
