---
title:    Attention alone, and word records for far characters
kind:     experiment
status:   concluded
outcome:  negative
question: >-
  Without the character GRU, can hierarchical attention do the job: at the
  same number of records per character, do word records standing in for
  far characters reach further and score better than characters alone?
answer: >-
  No, on both counts, at epoch 3 of 8. Attention alone (A, 1,024 character
  records, 1,892,264 trained parameters) scores 3.423 against slot 127's
  3.233 and depth records' 3.178 at epoch 3 (+5.72 and +7.43 points): the
  character GRU is worth several percent. Mean word records lose to A's
  characters wherever they start: past 512 characters (B) +0.68 points,
  the same with each letter turned by its place (R) +0.89, from the current
  word (W) +1.35, past 64 characters (S) +6.77. Their one gain, at every
  cutoff, is ~1–2% on repeats more than 256 words back (4.4% of
  characters).
opened:   2026-11-20
updated:  2026-11-21
code: >-
  branch attention-only. Slot 142 at 0ff2938 (A and B; tag
  rnd/attention-only-words/slot142), slot 143 at 15d1a41 (R; tag
  rnd/attention-only-words/slot143), slot 144 at 15d1a41 (W and S; tag
  rnd/attention-only-words/slot144).
data:     >-
  PG-19 100M split (rnd/README.md), configs/pg19split100m_noprior.json;
  epoch-3 screens (stopped after 3 of the 8-epoch schedule)
slots:    142, 143, 144
tags:     [attention-only, word-records, hierarchy, screen, no-prior]
related:  [reading-early, architecture-1.59m, simplified-no-words, gru-tree]
---

# Attention alone, and word records for far characters

## Question

The user asked (2026-11-20, via air) to explore hierarchical attention on
its own, with no GRU. The user's point, paraphrased: more character records
could cover what the word attention sees, but attention's cost grows with
the square of the context; the levels of words, then phrases, sentences
and paragraphs are meant to give longer windows of attention without
paying per character. So the arms are compared at about the same records
per character, the same attention cost, and judged by the reach and loss
that cost buys.

The lead agent's first design had a reach control with 2,048 character
records; the user found it unfair. The revised design's A and B were
approved as epoch-3 screens ([reading-early](../reading-early/)).

## Design

- **The model:** imm with the GRU removed (`--rnn-core none`: each block's
  input is the character's embedding), `--depth-records`, 4 blocks × 192
  with 4 heads of 48, the 1.84M transformer's shape. No prior, dropout 0,
  lr 1e-3 cosine to 1e-4 over 8 epochs, stopped after epoch 3
  (`--stop-after-epoch 3`, which leaves the learning rate where the full
  schedule has it), bf16, the epoch-3 best scored in fp32. Every arm ran at
  an update cap of 36,864 characters (`probe_ao.py`).
- **A:** 1,024 character records, no word records. In effect a transformer
  inside imm's training (carried streams, a full 1,024 characters always
  seen), so a B win wouldn't be imm's machinery.
- **B:** 512 character records, plus up to 600 word records, each the
  mean of that block's input over the word's characters (`--word-pool
  mean`), seen only once the word ends 512 or more characters back
  (`--words-beyond-chars`). The records are raw (`--memory-record raw`: no
  writer MLP), so A and B train the same 1,892,264 parameters. About as
  many keys per character as A (1,112 against 1,024), and a reach of
  ~3,400 characters.
- **R (slot 143):** B with `--word-pool rotated-mean` (`15d1a41`). In B's
  first block a record is a bag of letters: "dog" and "god" get the same
  one. R turns each character's input by its place in the word before the
  mean (pair k by 16^(−k/(W/2−1)) radians a place, W the width), so the
  record keeps the letters' order at every block. No parameters.
- **W and S (slot 144):** B's 512-character cutoff was the lead agent's
  choice (an even split at A's cost, no text covered twice), so B's words
  only ever stood in for far characters. The user asked why the words
  didn't start from the current word. Each arm is B with one change, the
  plain mean:
  - W: every word record seen from the current word back, beside the 512
    characters (no `--words-beyond-chars`): ~1,112 records per character.
  - S: 64 character records, then the last 960 words beyond them, about
    A's cost, reaching past a whole 5,000-character piece.

Slot 142 ran 2026-11-21 00:09–01:45 UTC, slot 143 04:27–05:21 UTC, slot
144 07:21–08:59 UTC.

## How to rerun

From a worktree at the slot's tag (rnd/README.md):

| script | slot | what it ran |
|---|---|---|
| `ao_slot.sh` | 142 | `probe_ao.py` for arms A and B, then A and B, 3 of 8 epochs each, scored in fp32 |
| `rot_slot.sh` | 143 | `probe_ao.py` for arm R, then R |
| `ws_slot.sh` | 144 | `probe_ao.py` for arms W and S, then W and S |
| `probe_ao.py` | all | each arm's update cap under 6.9 GiB, its time a update and its peak |
| `size_ao.py` | none | trained parameters of the arms (one update on the CPU) |

- `probe_ao.py` is the version slot 144 ran, which holds all five arms.
  Slots 142 and 143 ran earlier versions of the same file, edited in place
  and not kept.
- The probe reads `../scale-10.8m/worst_window.txt` (the script's path is
  the original `$S/s10/worst_window.txt`).
- Pairings: `scripts/rnd/rpair.sh A.npz B.npz OUT.txt`.
- Comments that quoted people word for word are paraphrased in the copied
  scripts. No command line changed.

**Results files** (`results/`): the pairings (`wordsB_vs_A_ao1.89m.txt`,
`rot_vs_wordsB_ao1.89m.txt`, `rot_vs_A_ao1.89m.txt`,
`wordsW_vs_B_ao1.89m.txt`, `wordsW_vs_A_ao1.89m.txt`,
`short64_vs_A_ao1.89m.txt`), slot logs (`slot_142.log`, `slot_143.log`,
`slot_144.log`), probes (`probe_ao_142.log`, `probe_ao_143.log`,
`probe_ao_144.log`) and memory samples (`rss_142.log`, `rss_143.log`,
`rss_144.log`).

**Checkpoints and per-character losses**, in `runs/`, each as `<id>.log`,
`<id>_<id>.csv`, `.pt`, `.best.pt` and `char_losses_<id>_best.npz`:

- A: `pg19s100m_ao_h192_l4_cm1024_depth_bf16_e8s3`;
- B: `pg19s100m_ao_h192_l4_cm512_w600mean_beyond_raw_depth_bf16_e8s3`;
- R: `pg19s100m_ao_h192_l4_cm512_w600rot_beyond_raw_depth_bf16_e8s3`;
- W: `pg19s100m_ao_h192_l4_cm512_w600mean_all_raw_depth_bf16_e8s3`;
- S: `pg19s100m_ao_h192_l4_cm64_w960mean_beyond_raw_depth_bf16_e8s3`.

## Results

Points are 100 × the difference in mean held-out loss (nats per
character), from each epoch's score in the logs. Percentages are A's
change in held-out perplexity per character against B (`rpair.sh`, from
the fp32 per-character losses), ± one standard error.

### 2026-11-21: A and B (slot 142)

| run | e1 | e2 | e3 |
|---|---:|---:|---:|
| A | 4.030 | 3.598 | 3.423 |
| B | 4.006 | 3.600 | 3.447 |
| B − A (points) | −0.61 | +0.06 | +0.68 |
| A − slot 127 (1.59M) | +7.25 | +7.43 | +5.72 |
| A − depth records (1.59M) | +8.20 | +9.39 | +7.43 |

- **Without the GRU, imm is 6–7% behind at epoch 3**, with 19% more
  parameters: 3.423 against slot 127's 3.233 and depth records' 3.178.
  Well past the screen's line, so the character GRU is worth several
  percent. A is built much like a transformer, so the GRU is likely what
  puts imm ahead of one.
- **B, words for the far characters, trails A by 0.68 ±0.12 points**, and
  the gap grew each epoch (−0.61, +0.06, +0.68). Under the screen's line,
  but moving against B. Pairing in `wordsB_vs_A_ao1.89m.txt` (B's change):
  - **It loses the band it gave up:** repeats 64–256 words back +10.2%
    ±0.4 (143% of the gap), capitalised there +24.5%; words seen before
    +2.3%; punctuation +12.9%, capitals +7.5%.
  - **Words carry something:** repeats more than 256 words back −1.3% ±0.5
    (beyond A's window), new words −0.5%, spaces −2.0%, repeats 1–16 words
    back −1.5%.
  - So the word records can't stand in for exact characters when a word
    is copied. One record has to give its spelling, and at block 0 the
    mean is a bag of letters. Above block 0 each character's input has
    already seen the word's earlier letters, so the bag explains part of
    it at most.
- 5,000-character pieces leave few targets for B's longer reach (repeats
  past 256 words: 4.4% of characters).
- Speed: A 900 s an epoch, B 973 s (195.7 and 205.7 ms an update in the
  probe).

### 2026-11-21: the rotation (slot 143)

| run | e1 | e2 | e3 |
|---|---:|---:|---:|
| R | 3.992 | 3.661 | 3.454 |
| R − B (points) | −0.35 | +1.68 | +0.21 |
| R − A | −0.96 | +1.74 | +0.89 |

- **No gain: R trails B by 0.21 ±0.11 points**, inside seed noise
  (~0.37). Pairings in `rot_vs_wordsB_ao1.89m.txt` and
  `rot_vs_A_ao1.89m.txt`.
- **The band B lost is unchanged:** repeats 64–256 words back +0.16% ±0.35
  against B, still +10.4% against A.
- Smaller moves against B, about what a second run differs by: the hardest
  characters (the prior's loss ≥ 3 nats) −3.8% for new words and −7.4% for
  words seen before; repeats 1–16 and 16–64 words back +1.9% and +1.5%,
  though those are inside both arms' 512 characters.
- **So the bag of letters wasn't why B loses.** One vector a word, read
  through one attention, doesn't stand in for the word's characters when
  it is copied, whether or not it keeps the letters' order.
- 216.8 ms an update in the probe (B 205.7), peak 4.73 GiB; 3 epochs in
  52 min.

### 2026-11-21: words from the start (slot 144)

| run | e1 | e2 | e3 | e3 − A (points) |
|---|---:|---:|---:|---:|
| A | 4.030 | 3.598 | 3.423 | |
| B | 4.006 | 3.600 | 3.447 | +0.68 |
| W | 3.991 | 3.643 | 3.470 | +1.35 |
| S | 4.064 | 3.788 | 3.663 | +6.77 |

Pairings in `wordsW_vs_B_ao1.89m.txt`, `wordsW_vs_A_ao1.89m.txt` and
`short64_vs_A_ao1.89m.txt`.

- **W is worse than B, by 0.66 ±0.11 points, though it sees all of B's
  records and more.** It trains worse too (train loss 1.191 against
  1.180). It loses copies inside the 512 characters both have: repeats
  16–64 words back +3.1%, 1–16 back +1.4%. It gains on the hardest
  characters (the prior's loss ≥ 3 nats, −4.1 to −4.5%) and punctuation
  (−4.2%). A guess, not tested: the recent words' records share the
  softmax with the characters' records and draw weight from the exact
  letters.
- **S is 7.0 ±0.14% behind A, and the gap grew each epoch** (+0.84, +5.14,
  +6.77). Repeats are 17–27% worse at every distance under 256 words,
  capitals +44%, punctuation +22%. Its only gain is the one every word arm
  shows, repeats more than 256 words back (−1.3%).
- **So, in this attention-only model, mean word records don't pay at any
  cutoff:** past 512 characters 0.68 points behind A, from the start
  1.35, past 64 characters 6.77. What they give, at every cutoff, is ~1–2%
  on repeats more than 256 words back (4.4% of characters).
- Probe: W 205.3 ms an update, peak 4.64 GiB; S 161.2 ms, peak 3.96 GiB. 3
  epochs: W 49 min, S 47 min.

### 2026-11-21: notes on the record

- The notes write the word-arm summary as "−0.68", "−1.35" and "−6.77"
  points against A; the tables and pairings show the word arms behind A by
  those amounts (+0.68, +1.35, +6.77 as arm − A). This record uses the
  tables' sign.
- Counted from A's and B's best checkpoints' optimizer state
  (`../no-prior-pair-1.8m/used_params.py`), both train 1,892,456
  parameters; `size_ao.py`'s one-update count gives 1,892,264.
- The "screen's line" above is the 1-unit line of
  [reading-early](../reading-early/), whose unit is a percent of the
  base's loss, not a point. At these losses (~1.23 nats) that line is
  about 1.2 points, so B's +0.68 is under it either way.

## What we learned

- Attention alone, in imm's training, trails the GRU model by 6–7% at
  epoch 3. The character GRU is likely what puts imm ahead of a
  transformer at small size.
- Mean word records don't replace characters for copying, at any cutoff
  tried, with or without the letters' order. They help only on repeats
  past A's 1,024-character window, a small share of these short pieces.
- **One seed** for every arm. The held-out pieces are 5,000 characters, so
  a longer held-out text could still change the verdict on reach.
- The user then chose to step back to simpler designs; the first is a
  GRU-only binary tree ([gru-tree](../gru-tree/)). Alternating GRU and
  attention layers waits behind it.
- The transformer's own curve at 3/8 of its run (~3.455) isn't a screen
  baseline: its curve's shape differs from imm's (slot 127 leads it by
  ~6% at 3/8 and ~4% at the end).
