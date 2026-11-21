---
title:    Character records, 256 and 1,024 characters back
kind:     experiment
status:   concluded
outcome:  positive
question: >-
  imm trailed the transformer on exact copying of recent words. If each
  character also writes a record, read by the same attention blocks as the
  word records, does the gap close, and how far back should the window
  reach?
answer: >-
  Yes. With the last 256 characters' records (slot 113) imm scores 3.1519:
  −1.69% ±0.12 against the block-256 transformer, −1.07% ±0.13 against
  block 1,024, and −4.08% ±0.12 against imm without them (slot 107). With
  1,024 (slot 115) it scores 3.0440: −4.46% ±0.12 against block 1,024,
  −5.05% ±0.14 against block 256 and −3.42% ±0.12 against the 256 window,
  ahead of block 1,024 in every bucket but two, both within noise. It costs
  time: 5,824 s and 8,243 s against slot 107's 3,597 s. Reading the records
  in runs (--char-chunk) gave the same loss but was slower, so it was
  dropped.
opened:   2026-11-19
updated:  2026-11-21
code: >-
  branch char-records. Slot 111 at 2556382 (out of GPU memory; tag
  rnd/char-records/slot111), slot 113 at d4199c6 (tag
  rnd/char-records/slot113), slot 115 at fe7dfb9 with --char-chunk 0 (tag
  rnd/char-records/slot115).
data:     PG-19 100M split (rnd/README.md), configs/pg19split100m_noprior.json
slots:    111, 113, 115
tags:     [no-prior, character-records, copying, attention]
related:  [no-prior-pair-1.8m, scale-10.8m, rope-far-positions, reading-early]
---

# Character records, 256 and 1,024 characters back

## Question

At ~1.8M with no prior and dropout 0, imm trailed the transformer by
2.5–3.1% per character, almost all of it on copying a recent word letter
for letter ([no-prior-pair-1.8m](../no-prior-pair-1.8m/)). imm's word
records see far back, but one record per word can't give its spelling. The
user agreed to try character records (2026-11-19).

## Design

The base is slot 107's imm: m128 (GRU 192, word tier 168, 2 blocks, 128
word records), no prior, dropout 0, 8 epochs on the 100M split, 3.2862.

- **`--char-memory N`** (`2556382`): each character u writes a record from
  the GRU's state before u and u's embedding, so one lookup can copy it.
  Character t sees the records of its last N characters (u ≤ t) in its
  piece.
- The records sit in **the same two attention blocks** as the word
  records; they only lengthen the list those blocks read. It isn't a new
  block (the user asked where it sits). The GRU, word tier and output head
  are unchanged.
- The last N records are carried between updates, detached.
- The read is dense: each character attends to the word records and every
  character record of the update (~1,050), with a mask.
- 1,588,157 trained parameters, against slot 107's 1,513,853.
- **Slot 111** (2026-11-19 12:03–12:05 UTC, `char_slot.sh`, `2556382`)
  ran out of GPU memory in the first update's backward (5.6 GiB allocated
  of the 7.7 GiB GPU). The block kept the full attention weights, and the
  records made them about 5× longer. **`d4199c6`** fuses the masked
  attention (`scaled_dot_product_attention`). That also touches the path
  without character records, by float rounding only.
- **Slot 113** (14:39–16:17 UTC, `char_slot2.sh`, `d4199c6`): N = 256.
- **Reading in runs** (`--char-chunk C`, `fe7dfb9`): each run of C
  queries attends to its own N + C − 1 character records, not all of the
  update's. Same attention, fewer masked-out records.
- **Slot 115**, N = 1,024 at `fe7dfb9`: `char_slot3.sh` first ran
  `chunk_probe.py` and started the run in runs of 128 at 18:57:38 UTC. The
  probe showed the whole-update mask faster, so after about a minute the
  run was stopped and restarted with `--char-chunk 0` (`char_slot4.sh`,
  18:58:53–21:16:38 UTC).

## How to rerun

From a worktree at the slot's tag (rnd/README.md):

| script | slot | what it ran |
|---|---|---|
| `char_slot.sh` | 111 | N = 256 at `2556382`; out of GPU memory |
| `char_slot2.sh` | 113 | `sdpa_probe.py` (does the fused kernel take the mask, and its peak), then N = 256 |
| `char_slot3.sh` | 115, first start | `chunk_probe.py`, then N = 1,024 in runs of 128 (stopped after a minute) |
| `char_slot4.sh` | 115 | N = 1,024 with the whole-update mask (`--char-chunk 0`) |

- Each slot script trains, then scores the best checkpoint's per-character
  held-out losses.
- Pairings: `scripts/rnd/rpair.sh A.npz B.npz OUT.txt`. The bottleneck
  tables came from `../no-prior-pair-1.8m/bottleneck.py`.
- Comments that quoted people word for word are paraphrased in the copied
  scripts. No command line changed.

**Results files** (`results/`): the slot logs (`char_slot_111.log`,
`char_slot_113.log`, `char_slot_115_first.log`, `char_slot_115.log`), the
probe (`chunk_probe_115.log`), the stopped start's training log
(`cm1024_c128_killed.log`), memory samples (`rss_111.log`, `rss_113.log`,
`rss_115.log`) and the pairings.

**Checkpoints and per-character losses**, in `runs/`:

- slot 113: `pg19s100m_h192_par_l2_w168_own_np_doc_mm128_do0_cm256_e8`
  (`.log`, and `…_e8_pg19s100m_…_e8.csv`, `.pt`, `.best.pt`), and
  `char_losses_pg19s100m_h192_par_l2_w168_own_np_doc_mm128_do0_cm256_e8_best.npz`;
- slot 115: the same names with `cm1024`.
- Slot 111's training log had the same run id as slot 113's and was
  overwritten by it; slot 115's first start was overwritten the same way,
  which is why its copy is kept in `results/`.
- The transformers' losses are named in
  [no-prior-pair-1.8m](../no-prior-pair-1.8m/).

## Results

Percentages are A's change in held-out perplexity per character against B
(`rpair.sh`), ± one standard error.

### 2026-11-19: 256 character records (slot 113)

| no prior, 798M characters, dropout 0 | params used | held out | train loss | wall time |
|---|---:|---:|---:|---:|
| imm m128 + character records 256 | 1,588,157 | 3.1519 | 1.0799 | 5,824 s |
| imm m128 (slot 107) | 1,513,853 | 3.2862 | 1.1028 | 3,597 s |
| transformer, block 256 | 1,838,784 | 3.2060 | 1.129 | ~885 s |
| transformer, block 1,024 | 1,986,240 | 3.1862 | 1.114 | ~990 s |

- **Ahead of both transformers: −1.69% ±0.12 against block 256 and
  −1.07% ±0.13 against block 1,024.**
- Against slot 107's imm: −4.08% ±0.12. Ahead from epoch 1: 3.743 / 3.453
  / 3.343 / 3.277 / 3.228 / 3.192 / 3.164 / 3.152 against 3.757 / 3.553 /
  3.466 / 3.406 / 3.361 / 3.326 / 3.297 / 3.286.
- **The copying gap inside 256 characters is closed.** On the third letter
  on of a word repeated < 256 characters back: against block 1,024, from
  +50.2% to −0.5% ±0.36; against block 256, from +20.6% to −20.1%; where
  the prior finds it hard (≥ 3 nats), from +872% to −1.5% against block
  1,024.
- **What's left is copying from 256–1,000 characters back**, outside the
  window and inside block 1,024's: +19.4% ±0.56 on all letters, +37.8% on
  the third letter on. Against block 256, which can't see that far either,
  it's +1.6%.
- Unchanged: new words (−3.8% against block 1,024, +0.9% against block
  256) and repeats 1,000+ back (−3.2% / +1.1%).
- Cost: 1.62× slot 107's time, 6.6× block 256's. Epochs took ~700 s
  against ~450 s.

### 2026-11-19: reading the records in runs (slot 115's probe)

`chunk_probe.py`, slot 113's model on the GPU, 64 streams × 64 words with
the 100M split's real word lengths, 3 warm-up and 10 timed updates:

| window N | read | ms/update | chars/s | peak allocated |
|---:|---|---:|---:|---:|
| 256 | whole-update mask (slot 113's) | 152.1 | 152.1k | 2.15 GiB |
| 256 | runs of 128 | 151.7 | 151.3k | 2.11 GiB |
| 1,024 | runs of 128 | 280.5 | 82.4k | 3.48 GiB |
| 1,024 | runs of 256 | 249.6 | 92.4k | 2.87 GiB |
| 1,024 | **whole-update mask** | **218.9** | **105.5k** | 2.96 GiB |

- The first loss is identical across the reads at each N (difference
  0.00e+00).
- Copying keys and values into the runs costs more than skipping the
  masked-out records saves.
- **GPU memory was never the limit:** 2.15 GiB allocated for slot 113's
  model. The "7.3 GB" peak seen during slot 113 was nvidia-smi's figure,
  the allocator's reserve.
- `--char-chunk` defaults to 0 from `84ddd77` and was dropped in
  `7f8238e`, the branch's tip.

### 2026-11-19: 1,024 character records (slot 115)

| no prior, 798M characters, dropout 0 | params used | held out | train loss | wall time |
|---|---:|---:|---:|---:|
| imm m128 + character records 1,024 | 1,588,157 | **3.0440** | 1.0596 | 8,243 s |
| imm m128 + character records 256 (slot 113) | 1,588,157 | 3.1519 | 1.0799 | 5,824 s |
| transformer, block 256 | 1,838,784 | 3.2060 | 1.129 | ~885 s |
| transformer, block 1,024 | 1,986,240 | 3.1862 | 1.114 | ~990 s |

- **−4.46% ±0.12 against block 1,024, −5.05% ±0.14 against block 256,
  −3.42% ±0.12 against the 256 window.**
- Ahead of slot 113 at every epoch: 3.711 / 3.310 / 3.210 / 3.153 / 3.109
  / 3.077 / 3.054 / 3.044. Still falling at epoch 8 (−0.010).
- **The 256–1,000 copying gap is closed.** Against block 1,024, on a word
  repeated 256–1,000 characters back: all letters from +19.38% ±0.56 to
  −7.73% ±0.36; the third letter on from +37.75% to −7.97% ±0.57; the
  third letter on where the prior finds it hard (≥ 3 nats, 486 targets)
  from +431% to −30.2% ±7.3.
- Against block 1,024, imm is ahead in every bucket but two, both within
  noise: the text's first 256 characters (255 targets, +3.75% ±5.10) and
  the third letter on of 6+-letter words repeated < 256 back (+0.83%
  ±0.73). Its largest leads: before a sentence −12.1%, capitalised words
  seen before in the piece −9.9%, repeats 64–256 words back −7.7%. New
  words −4.10%.
- Against block 256: new words +0.60% ±0.20, repeats 256+ words back
  +0.89% ±0.53.
- Against the 256 window, the gain is all in repeats: words seen before
  −8.41%, repeats 64–256 words back −17.0%; new words −0.29% ±0.18.
- The d24 prior alone scores 3.068 on these characters; imm, with no
  prior, is now below it overall. On new words the prior is still far
  better (3.464 against 4.131).
- Cost: 1.42× slot 113's time, 9.3× block 256's and 8.3× block 1,024's,
  at 92–99k characters/s. Three CUDA allocator retries in the first 11
  minutes, all recovered. Peak RSS 5.26 GB.

### 2026-11-21: a note on the record

The notes say slot 115 restarted with the new default from `84ddd77`. The
slot log shows the restart ran `fe7dfb9` with `--char-chunk 0` passed
explicitly; `84ddd77` was committed 45 s later. The two behave the same.

## What we learned

- Character records close imm's copying gap, and a window of 1,024 puts
  imm ahead of the block-1,024 transformer in nearly every bucket at
  ~1.6M.
- Most of the 1,024 window's gain is on repeats 256–1,000 characters back.
- imm is now 8–9× slower than the transformer per run.
- The next question was whether the lead holds at 10.8M
  ([scale-10.8m](../scale-10.8m/)).
- **One seed** for each run. The transformer's learning rate and shape
  aren't tuned; both sides are at their better dropout.
