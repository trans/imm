---
title:    The no-prior pair at ~1.8M
kind:     experiment
status:   concluded
outcome:  negative
question: >-
  With no prior on either side, the same 100M-character split and the same
  798M characters seen, how does imm at ~1.5M trained parameters compare
  with a transformer of ~1.8–2.0M?
answer: >-
  With both at dropout 0 the transformer leads. imm scores 3.2862 against
  3.2060 for the block-256 transformer and 3.1862 for block 1,024: imm is
  +2.50% ±0.13 and +3.14% ±0.15 per character. The first reading (slot
  101: imm 3.3927, −5.20% ±0.11 against 3.5788) came from the transformer
  being held back by dropout 0.2; at 0 it scores 3.2065. imm's gap is exact
  copying of recent words. It is level or ahead on new words, far repeats
  and sentence starts, with 18–24% fewer parameters, and about 4× slower.
opened:   2026-11-19
updated:  2026-11-21
code: >-
  imm branch topic-tiers 91c3c55 (slots 101 and 107; tags
  rnd/no-prior-pair-1.8m/slot101, rnd/no-prior-pair-1.8m/slot107). The
  transformer is air's experiments/prior_scaling/prior_transformer.py (air
  main 7b73806; the file last changed in f78af95), which imports imm's
  scripts/transformer_baseline.py (43057d9, the same at 91c3c55). Slot 105
  ran transformers only; its tag rnd/no-prior-pair-1.8m/slot105 goes on
  91c3c55, the commit of the config they read.
data:     PG-19 100M split (rnd/README.md), configs/pg19split100m_noprior.json
slots:    101, 105, 107 (104 was queued and replaced by 105 before it ran)
tags:     [no-prior, transformer-baseline, dropout, scale]
related:  [char-records, scale-10.8m, reading-early]
---

# The no-prior pair at ~1.8M

## Question

The no-prior scale test pits imm against a transformer with no count prior
on either side. The user asked (2026-11-18) for one imm run at ~1.85M
parameters and one transformer of the same size, compared before any bigger
runs or second seeds. The reason, the user's (via air): a prior can help
any model, so imm's own model has to be judged without one first.

## Design

Both models train on the 100M split and see the same 798M characters:
8 epochs of imm's 99,795,945 training characters an epoch (the pieces cut
to whole updates).

- **imm "m128"**, `np_slot.sh` and `do0_slot.sh`:
  - a GRU of width 192 over characters, embedding 192;
  - a word tier (a GRU that steps once a word) of width 168, with
    `--tier-own-step`;
  - 2 gated cross-attention blocks reading 128 word records (one record
    per word, written from the GRU's state);
  - `--documents`: each piece is read from its start;
  - 64 streams × 64 words an update, lr 1e-3 cosine to 1e-4, fp32;
  - dropout 0.2 (slot 101) or 0.0 (slot 107);
  - 1,513,853 trained parameters, counted from the optimizer's state with
    `used_params.py` (AdamW keeps state only for parameters that got a
    gradient). The checkpoint holds 5.48M because of mixing modes this
    config never runs.
- **The transformer**, air's `prior_transformer.py --prior none`:
  - 4 layers × 192, 4 heads; block 256 at batch 64, or block 1,024 at
    batch 16 (16,384 characters a step either way);
  - lr 1e-3 cosine to 1e-4, warmup 100, weight decay 0.1, seed 1337,
    48,728 iterations, bf16;
  - 1,838,784 parameters at block 256 and 1,986,240 at block 1,024 (the
    position table grows).
- **Scoring differs a little.** imm scores each held-out piece from a fresh
  state. The transformer scores windows of its block at a stride of half
  the block over the joined pieces, in fp32 on the CPU
  (`score_tx_noprior.py`). Air's numbers are scored the same way.
- **One thing at a time:** slot 105 reran the transformer at dropout 0.0
  and 0.1 (block 256) and at block 1,024 with dropout 0.2. Slot 107 ran
  imm at dropout 0 and the block-1,024 transformer at dropout 0.

## How to rerun

From a worktree at the run's tag (rnd/README.md), with the slot tool and
air's checkout where the scripts expect them:

| script | what it ran |
|---|---|
| `np_slot.sh` | slot 101: imm at dropout 0.2, then the block-256 transformer at 0.2, then both runs' per-character losses |
| `tx_slot2.sh` | slot 105: the transformer at block 1,024 / dropout 0.2, block 256 / 0.0, block 256 / 0.1 |
| `do0_slot.sh` | slot 107: imm at dropout 0, then the block-1,024 transformer at dropout 0 |
| `score_tx_noprior.py` | a transformer checkpoint's per-character held-out losses, in imm's `--char-loss-output` format |
| `used_params.py` | trained parameters from a checkpoint's optimizer state |
| `bottleneck.py` | the `*_bottleneck*.txt` tables: imm, the transformer and the d24 prior by letter place and repeat distance (`python bottleneck.py IMM.npz TX.npz PRIOR_ROWS`) |

- The paired comparisons (`results/*_vs_*.txt`) came from
  `scripts/rnd/rpair.sh A.npz B.npz OUT.txt`; A is the first name in the
  file name.
- The `bottleneck.py` command lines weren't kept. The prior rows it read
  were presumably mobius's d24 rows that `rpair.sh` uses.
- Slot 104's script (`txdo_slot.sh`, dropout 0.0 and 0.1 only) never ran.
  It was replaced by `tx_slot2.sh`, which added the block-1,024 arm, so it
  isn't copied.
- Comments that quoted people word for word are paraphrased in the copied
  scripts. No command line changed.

**Results files** (`results/`): the slot wrappers' logs
(`np_slot_101.log`, `tx_slot_105.log`, `do0_slot_107.log`), memory samples
(`rss_101.log`, `rss_107.log`), the transformers' training logs and result
JSON (`tx1.84m_np_100m*.log`, `.json`, `_cl.log`), and the pairings.

**Checkpoints and per-character losses:**

- imm, in `runs/`:
  - slot 101: `pg19s100m_h192_par_l2_w168_own_np_doc_mm128_e8` (`.log`,
    and `…_e8_pg19s100m_…_e8.csv`, `.pt`, `.best.pt`), and
    `char_losses_pg19s100m_h192_par_l2_w168_own_np_doc_mm128_e8_best.npz`;
  - slot 107: the same names with `_do0_e8`.
- The transformers were written to the session's scratch directory
  (`$S/scale/`) and copied into `runs/` on 2026-11-21, unchanged:
  `runs/tx1.84m_np_100m{,_do0,_do1,_b1024,_b1024_do0}.pt` (with `.log` and
  `.json`) and `runs/char_losses_tx1.84m_np_100m{,_do0,_do1,_b1024,_b1024_do0}.npz`.

## Results

### 2026-11-19: slot 101, both at dropout 0.2

Slot 101 ran 2026-11-19 00:14–01:40 UTC.

| no prior, 798M characters | params used | held out | wall time |
|---|---:|---:|---:|
| **imm m128**, 8 epochs, dropout 0.2 | 1,513,853 | **3.3927** | 4,169 s |
| transformer, block 256, dropout 0.2 | 1,838,784 | 3.5788 | 896 s |

Percentages here and below are A's change in held-out perplexity per
character against B, from `rpair.sh`, ± one standard error.

- **Per character, imm is −5.20% ±0.11 ahead** of the transformer, with 18%
  fewer parameters. It's ahead in every bucket (`np100m_vs_tx.txt`,
  `np100m_bottleneck.txt`):
  - choosing the next word: −7.28% ±0.24 at a word's last letter, −11.0%
    before a phrase, −15.8% before a sentence; −12.6% on capitals, −11.1%
    on punctuation and digits;
  - inside words −4.96% ±0.14; on letters the d24 prior finds hard
    (≥ 3 nats), −16% to −28%. The d24 prior is mobius's count model over
    up to 24 characters of context, built from 10 GB of text; no model
    here uses it, but the pairings bucket targets by its loss;
  - level only where the transformer copies: the third letter on of a
    word of 6+ letters repeated < 256 characters back, −0.16% ±0.96.
- **Per second, the transformer is ahead.** imm took 4.65× as long. imm
  was at 3.797 after epoch 1 (530 s) and 3.626 after epoch 2 (~1,060 s),
  on a schedule set for 8 epochs. imm ran at 186k–210k characters/s in
  fp32; the transformer ran in bf16.
- **Neither overfits.** Training loss (dropout on) is above held-out for
  both: the transformer's 1.340 against 1.275, imm's 1.2336 against 1.2216.
- **Neither is short of data at this size.** The transformer scored 3.5678
  on the 20M split (air, 41 epochs) against 3.5788 here.
- Curves, held-out perplexity:
  - imm, epochs 1 / 2 / 4 / 6 / 8: 3.797 / 3.626 / 3.492 / 3.420 / 3.393;
  - the transformer, steps 7.5k / 22.5k / 47.5k / 48.7k: 4.017 / 3.751 /
    3.582 / 3.579.
- For scale: the d24 count prior alone, built from 10 GB, scores 3.068 on
  the same targets.

### 2026-11-19: slot 105, the transformer's dropout

Slot 105 ran 02:18–03:06 UTC. Same transformer, same characters, one change
at a time:

| transformer, no prior | params | held out | train loss (last 10 logs) | imm m128 (dropout 0.2) against it |
|---|---:|---:|---:|---:|
| block 256, dropout 0.2 (slot 101) | 1,838,784 | 3.5788 | 1.340 | −5.20% ±0.11 |
| block 256, dropout 0.1 | 1,838,784 | 3.3099 | 1.251 | +2.51% |
| **block 256, dropout 0.0** | 1,838,784 | **3.2065** | 1.129 | **+5.82% ±0.13** |
| block 1,024, batch 16, dropout 0.2 | 1,986,240 | 3.6878 | 1.343 | −8.00% |

- **Slot 101's lead was the transformer's dropout.** At dropout 0 it is
  5.8% ahead of imm at 0.2; dropout 0.2 cost it 10.4%. Its training loss
  now sits under held-out (1.129 against 1.165).
- Where it leads (`np100m_vs_tx_do0.txt`, `np100m_bottleneck_do0.txt`):
  copying a recent word (the third letter on of a word repeated < 256
  characters back, +21.9% ±0.61; repeats 1–16 words back +16.1%); inside
  words +6.45%, lowercase +7.15%, punctuation and digits +22%.
- imm still leads before a sentence (−5.1% ±1.5), before a phrase
  (−2.2%), on spaces (−2.2%) and on letters the prior finds hard (≥ 6 nats,
  −6.5%).
- Block 1,024 at dropout 0.2 is worse than block 256 (3.6878 against
  3.5788): a quarter of the sequences a step, same dropout. It isn't a fair
  reading of longer context.
- The held-out figures in the table are the transformer's own in-training
  scores (bf16). Its fp32 per-character scores, which the pairings use, are
  3.2060 (dropout 0) and 3.3097 (0.1).
- The same handicap sits under an older result: air's 1.84M no-prior
  transformer on the 20M split, at dropout 0.2, also trains above
  held-out (1.313 against 1.272), so the earlier "imm 4.9% ahead" (slot 58)
  rests on it too.

### 2026-11-19: slot 107, both at dropout 0

Slot 107 ran 03:18–04:36 UTC. This is the fair first pair.

| no prior, 798M characters, dropout 0 | params used | held out | train loss | wall time |
|---|---:|---:|---:|---:|
| imm m128 | 1,513,853 | 3.2862 | 1.1028 | 3,597 s |
| transformer, block 256 | 1,838,784 | 3.2060 | 1.129 | ~885 s |
| transformer, block 1,024, batch 16 | 1,986,240 | 3.1862 | 1.114 | ~990 s |

- **imm at dropout 0 is 3.14% ±0.11 better than at 0.2**
  (`imm_do0_vs_do2.txt`). Its held-out still falls every epoch: 3.757 /
  3.553 / 3.406 / 3.326 / 3.286 at epochs 1 / 2 / 4 / 6 / 8. Training is
  now below held-out (1.103 against 1.190).
- **Paired, imm is +2.50% ±0.13 against block 256 and +3.14% ±0.15 against
  block 1,024** (`np100m_do0_vs_tx_*.txt`, `np100m_do0_bottleneck_*.txt`).
  Block 1,024 beats block 256 by only 0.62% ±0.13
  (`tx_b1024_vs_b256_do0.txt`).
- **imm's gap is copying recent text exactly.** On the third letter on of
  a word repeated < 256 characters back, imm trails block 256 by 20.6% and
  block 1,024 by 50.2%. On a word repeated 1–16 words back it trails by
  13.5% and 18.3%.
- **Where imm is level or ahead:** new words (+1.0% against block 256,
  −3.6% against block 1,024); repeats 1,000+ characters back (+1.2% and
  −3.1%); a piece's first 256 characters (−1.9% and −7.2%); before a phrase
  (−2.8%) and a sentence (−3.3%) against block 256, though it trails block
  1,024 there (+4.9% and +12.4%).
- So imm's word records see further back than block 256, but too coarsely
  to copy a word's letters. The user put it as a question of resolution:
  imm sees further back, but how precisely?
- imm has 18–24% fewer parameters and is about 4× slower (3,597 s against
  885–990 s).
- The block-1,024 run's in-training score (bf16) was 3.1930; its fp32
  per-character score is 3.1862. At block 256 the two agree to 0.0005.

### 2026-11-21: notes on the record

- The notes on slot 101 say slot 104 would run the transformer at dropout
  0.0 and 0.1. Those runs are slot 105's. Slot 104's wrapper log is empty.
- `np100m_vs_tx_do1.txt` and `np100m_vs_tx_b1024.txt` (imm at 0.2 against
  the dropout-0.1 and the block-1,024 transformers) are copied too; the
  notes give their headlines (+2.51%, −8.00%) but not their file names.

## What we learned

- At ~1.8M with no prior, a transformer at dropout 0 is 2.5–3.1% ahead of
  imm per character, and about 4× faster.
- imm's gap is exact copying from the last few hundred characters. That led
  to character records ([char-records](../char-records/)).
- Dropout 0.2 held back both models here. Dropout 0 is the setting for the
  rest of this line.
- **One seed** for every run. Never tested: second seeds, imm at dropout
  0.1, and the transformer's learning rate or shape.
