---
title:    RoPE far into the text, and training in bf16
kind:     experiment
status:   concluded
outcome:  positive
question: >-
  imm's attention blocks computed their rotary position angles from
  absolute character offsets, in the queries' number format. Did that hurt
  training, and once fixed, does training in bf16 give the same model as
  fp32?
answer: >-
  The fault was real but cost nothing measurable. In fp32 the fastest
  rotary pairs became noise past ~5e7 characters into the training text;
  in bf16 the positions themselves rounded. Fixed in cac43fb. Slot 127, the
  1.59M model on the fixed code trained in bf16, scored 3.0552 against
  slot 115's 3.0440 (fp32, before the fix): +0.37% ±0.10, between −0.7%
  and +1.1% in every bucket, about one seed's spread. bf16 is 1.67× faster.
  The one-epoch checks spread more than rounding alone should (fixed: bf16
  3.643, fp32 3.742).
opened:   2026-11-20
updated:  2026-11-21
code: >-
  branch bf16. Slot 123 at 1edcb56 (the check that found it; tag
  rnd/rope-far-positions/slot123), slot 124 at cac43fb (the fix and its
  check; tag rnd/rope-far-positions/slot124), slot 127 at 00b6a52 (the
  1.59M rerun; tag rnd/rope-far-positions/slot127). The fix's tests are
  tests/test_rope_positions.py at cac43fb.
data:     PG-19 100M split (rnd/README.md), configs/pg19split100m_noprior.json
slots:    123, 124, 127
tags:     [rope, numerics, bf16, bug]
related:  [scale-10.8m, char-records, architecture-1.59m]
---

# RoPE far into the text, and training in bf16

## Question

The 10.8M imm only fits the GPU in bf16 ([scale-10.8m](../scale-10.8m/)).
Before trusting a bf16 run, each 10.8M slot first checked bf16 on slot
115's 1.59M model: one epoch of its 8-epoch schedule, against fp32's 3.711
after epoch 1, with a limit of 0.5% worse.

## Design

- **`--bf16`** (`0d208bd`): each update runs under autocast in bfloat16,
  with the GRUs kept in float32. Autocast would have run cuDNN's GRU in
  float16, whose small gradients underflow without loss scaling, so a
  subclass of `nn.GRU` turns autocast off inside it. On the GPU
  (`bf16_check.py`) the first loss was within 0.002% of fp32's.
- **Slot 123** (2026-11-20 03:39–03:49 UTC, `imm10b_slot.sh`, `1edcb56`):
  the bf16 check scored 3.771 after one epoch against fp32's 3.711, 1.6%
  worse, so the slot stopped. The cause was RoPE (rotary position
  encoding, which turns each query and key by its position):
  - Positions are character offsets into the training text, and the
    blocks computed the angles (position × frequency) in the queries'
    format.
  - **In fp32**, a large position and its product with a frequency are
    rounded. At the highest frequency (1 radian a character) the error is
    ~0.03 rad at 1e6, ~0.5 rad at 1e7 and several radians at 1e8. Past
    ~5e7 the 6–8 fastest of 24 rotary pairs (head width 48) are noise.
    That touches nearly all of a 100M text and most of a 20M one.
  - The held-out text is its own tensor, at positions under 250K, so
    scoring saw intact angles: training and scoring didn't match.
  - **In bf16** the positions themselves rounded: bf16 holds integers
    exactly only up to 256.
- **The fix, `cac43fb`:** each stream's positions are counted from its
  last query (scores depend only on the gaps, so they're the same in exact
  arithmetic), and the angles and the rotation are computed in fp32, then
  cast back. The serial paths' RoPE and the input RoPE get fp32 angles
  too, but still count from 0. `tests/test_rope_positions.py`: a block's
  output and a stream's loss are the same at offset 0 and 7e7. Under bf16,
  with gaps up to 1,900, the error is under 15% of what positions add
  (measured: 9%). Removing either half of the fix fails the tests.
- **Slot 124** (03:56–04:26 UTC, `imm10c_slot.sh`, `cac43fb`): the check
  again, one epoch in fp32, then in bf16. It passed. The slot then ran out
  of GPU memory starting the 10.8M run (see scale-10.8m).
- **Slot 127** (08:38–10:02 UTC, `imm159_slot.sh`, `00b6a52`): slot 115's
  1.59M model on the fixed code, 8 epochs in bf16, scored in fp32, so both
  imm sizes of the 10.8M pair run the same code. The 1.59M model: GRU 192,
  2 blocks of 4 heads, word tier 168, 128 word records and 1,024 character
  records, no prior, dropout 0, 1,588,157 trained parameters.

## How to rerun

From a worktree at the slot's tag (rnd/README.md):

| script | slot | what it ran |
|---|---|---|
| `bf16_check.py` | none | a small GPU check of `--bf16`: which modules run in bfloat16, the first loss against fp32, a few updates |
| `imm10b_slot.sh` | 123 | `probe_split.py` (sizes and the update cap for the 10.8M), then the one-epoch bf16 check (failed) |
| `imm10c_slot.sh` | 124 | `probe_split.py`, the one-epoch check in fp32 and bf16 (passed), then the 10.8M run (out of GPU memory) |
| `imm159_slot.sh` | 127 | the 1.59M model, 8 epochs bf16, then its fp32 per-character losses |

- Pairing: `scripts/rnd/rpair.sh A.npz B.npz OUT.txt`.
- Comments that quoted people word for word are paraphrased in the copied
  scripts. No command line changed.

**Results files** (`results/`): `imm1.59m_fixed_vs_old.txt` (slot 127
against slot 115), slot logs (`slot_123.log`, `slot_124.log`,
`slot_127.log`), probes (`probe_split_123.log`, `probe_split_124.log`) and
memory samples (`rss_123.log`, `rss_124.log`, `rss_127.log`).

**Checkpoints and per-character losses**, in `runs/`:

- slot 123's check: `pg19s100m_h192_par_l2_w168_own_np_doc_mm128_do0_cm1024_bf16_check`
  (`.log`, `…_check_pg19s100m_…_check.csv`, `.pt`, `.best.pt`);
- slot 124's checks: the same model with `_rel_check32` (fp32) and
  `_rel_check16` (bf16); its 10.8M start: `…_h448_…_rel_bf16_e8_oom124.log`;
- slot 127: `pg19s100m_h192_par_l2_w168_own_np_doc_mm128_do0_cm1024_rel_bf16_e8`
  (`.log`, `.csv`, `.pt`, `.best.pt`) and
  `char_losses_pg19s100m_h192_par_l2_w168_own_np_doc_mm128_do0_cm1024_rel_bf16_e8_best.npz`;
- slot 115, the reference: listed in [char-records](../char-records/).

## Results

### 2026-11-20: the one-epoch checks (slots 123 and 124)

Slot 115's 1.59M model after one epoch of its 8-epoch schedule, held-out
perplexity scored in the run's own precision:

| code | fp32 | bf16 | training seconds in epoch 1, fp32 / bf16 |
|---|---:|---:|---:|
| before the fix (slots 115, 123) | 3.711 | 3.771 | 1,013 / 535 |
| **fixed, `cac43fb`** (slot 124) | 3.742 | **3.643** | 1,000 / 600 |

- bf16 passes, but **these spreads are larger than rounding alone should
  give**: fixed bf16 is 2.6% under fixed fp32, and fixed fp32 0.8% over
  the unfixed one. Epoch 1 lands mid-descent at a learning rate of
  ~1e-3, so its value may vary by a percent or more between runs that
  differ only in rounding; GPU training here isn't deterministic either.
- So the 0.5% check is weaker than it was taken to be, and slot 123's
  1.6% may have been partly that. The bf16 position rounding was a real
  fault either way (the tests).
- The fix costs time: bf16 at 1.59M went from 535 to 600 s an epoch, and
  the 10.8M (GRU 448 × 3) from 353.8 to 396.6 ms an update. The angles are
  now fp32, computed per block for queries and keys; caching them once per
  update would win most of that back.

### 2026-11-20: the 1.59M model on the fixed code, in bf16 (slot 127)

| no prior, 798M characters, dropout 0 | params | held out (fp32) | train loss | wall time |
|---|---:|---:|---:|---:|
| **imm 1.59M, fixed, bf16** (slot 127) | 1,588,157 | **3.0552** | 1.069 | 4,950 s |
| imm 1.59M, before the fix, fp32 (slot 115) | 1,588,157 | 3.0440 | 1.060 | 8,243 s |

- **+0.37% ±0.10 against slot 115** (A's change in held-out perplexity per
  character against B, ± one standard error), between −0.7% and +1.1% in
  every bucket (`imm1.59m_fixed_vs_old.txt`). The RoPE fix and bf16
  together change nothing that one seed could show.
- Held-out by epoch: 3.748 / 3.340 / 3.233 / 3.169 / 3.126 / 3.093 / 3.067
  / 3.055.
- bf16 is 1.67× faster here.

### 2026-11-21: a note on the record

Slot 127's train loss of 1.069 is the running training loss at step
30,000 (the log's last `progress` line); the epoch-8 mean in its CSV is
1.0604. Slot 115's 1.060 is its epoch-8 mean.

## What we learned

- Every imm run before `cac43fb` that used RoPE in its attention blocks
  trained with noisy angles for most of its text and was scored with clean
  ones. At 1.59M on the 100M split it cost nothing measurable.
- bf16 with fp32 GRUs trains as well as fp32 within one seed, 1.67× faster.
  It is the setting for the runs after this.
- Slot 127 (3.0552) became the 1.59M base for the architecture runs.
- The one-seed spread, +0.37% here, became the yardstick for "noise" in
  later comparisons.
- **Open:** the spread of epoch 1 across identical runs was never
  measured. No commit through `15d1a41` caches the angles once per update.
