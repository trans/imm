---
title:    Three architecture changes at 1.59M
kind:     experiment
status:   concluded
outcome:  positive
question: >-
  Which parts of imm carry its lead at 1.59M? Three changes to slot 127's
  model, one at a time: the GRU hierarchy alone with no attention to
  records; depth records, so each block reads records rebuilt from the
  block before it; and the character records' own key and value maps.
answer: >-
  Depth records help: 3.0109, −1.45% ±0.10 against slot 127's 3.0552, in
  every bucket but one (a piece's first 256 characters, +0.38% ±0.59) and
  at every epoch, with no new parameters; −5.50% ±0.12 against the 1.84M
  transformer. The GRU hierarchy alone loses 9.56% ±0.14 (3.3473), 79% of
  it on repeated words: the attention's worth is copying from the last
  ~1,000 characters. The character records' own key/value maps give only
  −0.34% ±0.10 (3.0449) for 9.3% more parameters, about one seed's spread.
opened:   2026-11-20
updated:  2026-11-21
code: >-
  branch depth-records. Slot 130 at b729f6c (the GRU hierarchy alone, then
  depth records; tag rnd/architecture-1.59m/slot130), slot 131 at 878686e
  (--char-kv; tag rnd/architecture-1.59m/slot131).
data:     PG-19 100M split (rnd/README.md), configs/pg19split100m_noprior.json
slots:    130, 131
tags:     [architecture, depth-records, attention, ablation, no-prior]
related:  [rope-far-positions, depth-records-10.8m, simplified-no-words, attention-only-words, reading-early]
---

# Three architecture changes at 1.59M

## Question

The user asked (2026-11-20) how imm does with the GRU hierarchy alone and
with attention alone, whether the attention is held back by reading GRU
vectors, and whether characters, words and sentences should have their own
embedding spaces.

The lead agent's reading of the code:

- every block reads the same records, written once from the GRU's state,
  so more blocks only deepen the queries (a transformer rebuilds keys and
  values in every layer);
- nothing the attention reads reaches the GRU or later records;
- word and character records share each block's key/value map.

Attention alone would be a weaker transformer, so it wasn't run here (it
was later: [attention-only-words](../attention-only-words/)). The user
judged these runs more telling than the planned 10.8M dropout-0.1 run,
which was set aside.

## Design

The base is slot 127's model ([rope-far-positions](../rope-far-positions/)):
GRU 192, 2 blocks of 4 heads, word tier 168, 128 word records and 1,024
character records, no prior, dropout 0, 8 epochs on the 100M split,
trained in bf16 and scored in fp32: 3.0552, 1,588,157 trained parameters.
Each run changes one thing:

- **The GRU hierarchy alone** (`--no-memory-attention --char-memory 0`,
  from `ace8c00`): no word or character records are attended; the blocks'
  MLPs stay. A third fewer trained parameters: the attention's maps and the
  record writers go.
- **`--depth-records`** (`b729f6c`): block i > 1 reads records written by
  the same writers from block i−1's output. No new parameters. The state
  before an update's first character is carried per block.
- **`--char-kv`** (`878686e`): in each block the character records get
  their own LayerNorm and key and value maps, copies of the word records'
  at the start (+148,992 trained parameters).

Slot 130 ran 2026-11-20 11:21–13:25 UTC; slot 131 13:26–14:55 UTC.

## How to rerun

From a worktree at the slot's tag (rnd/README.md):

| script | slot | what it ran |
|---|---|---|
| `arch_slot.sh` | 130 | the GRU hierarchy alone, then `--depth-records`, each 8 epochs bf16 and scored in fp32 |
| `kv_slot.sh` | 131 | `--char-kv`, 8 epochs bf16, scored in fp32 |
| `size_noattn.py` | none | trained parameters of a config with no attention to records (one update on the CPU) |

- Pairings: `scripts/rnd/rpair.sh A.npz B.npz OUT.txt`.
- Comments that quoted people word for word are paraphrased in the copied
  scripts. No command line changed.

**Results files** (`results/`): the pairings, the slot logs
(`slot_130.log`, `slot_131.log`) and memory samples (`rss_130.log`,
`rss_131.log`).

**Checkpoints and per-character losses**, in `runs/`, each run as
`<id>.log`, `<id>_<id>.csv`, `.pt`, `.best.pt` and
`char_losses_<id>_best.npz`:

- `pg19s100m_h192_par_l2_w168_own_np_doc_mm128_do0_noattn_bf16_e8`;
- `pg19s100m_h192_par_l2_w168_own_np_doc_mm128_do0_cm1024_depth_bf16_e8`;
- `pg19s100m_h192_par_l2_w168_own_np_doc_mm128_do0_cm1024_charkv_bf16_e8`.

## Results

### 2026-11-20: slots 130 and 131

Percentages are A's change in held-out perplexity per character against B
(`rpair.sh`), ± one standard error; B is slot 127 unless named.

| change | trained params | held out (fp32) | vs slot 127 | train loss | wall time |
|---|---:|---:|---:|---:|---:|
| none (slot 127) | 1,588,157 | 3.0552 | | 1.069 | 4,950 s |
| GRU hierarchy alone (`--no-memory-attention --char-memory 0`, MLPs kept) | ~1,067,000 | **3.3473** | **+9.56% ±0.14** | 1.118 | 1,817 s |
| `--depth-records` | 1,588,157 | **3.0109** | **−1.45% ±0.10** | 1.051 | 5,541 s |
| `--char-kv` (slot 131) | ~1,736,000 | **3.0449** | **−0.34% ±0.10** | 1.061 | 5,317 s |

- **Depth records help in every bucket** (`depth_vs_rel1.59m.txt`):
  inside words −1.34%, word starts −1.57%, after a word −1.74%. Most on
  repeated words (−2.62%, 61% of the gain), capitals (−4.4%), punctuation
  and digits (−6.1%) and targets the prior finds hardest (≥ 6 nats,
  −13.4%). Only a piece's first 256 characters don't gain (+0.38% ±0.59).
  - Ahead at every epoch: 3.713 / 3.275 / 3.178 / 3.122 / 3.081 / 3.048 /
    3.022 / 3.011 (slot 127: 3.748 / 3.340 / … / 3.055).
  - 1.30× slot 127's wall time.
  - Against the 1.84M transformer: −5.50% ±0.12 (`depth_vs_tx1.84m.txt`;
    slot 127: −4.11%).
- **The GRU hierarchy alone loses 9.6%** (`noattn_vs_rel1.59m.txt`).
  - 79% of the loss is on repeated words (+24.2%; 1–16 words back +28%,
    16–64 +35%, 64–256 +21%), with capitals +62% and punctuation +28%.
  - Repeats 256+ words back lose nothing (+0.43% ±0.52), nor does a
    piece's first 256 characters (+1.05%). So the attention's worth is
    copying from the last ~1,000 characters, which the records hold.
  - ~3.7 min an epoch against ~10.
- **The character records' own key/value maps barely help:** −0.34% ±0.10
  for 9.3% more parameters, about one seed's spread
  (`charkv_vs_rel1.59m.txt`).
  - Most on repeats 1–16 words back (−1.8%) and capitals (−2.4%); nothing
    on new words (−0.03%) or predicting after a word (+0.21%).
  - It starts faster (epoch 1 3.665 against 3.748), then the lead shrinks:
    3.314 / 3.208 / 3.153 / 3.112 / 3.080 / 3.056 / 3.045.
  - Depth records beat it by 1.13% (`charkv_vs_depth1.59m.txt`) with no
    new parameters. So word and character records sharing a key/value map
    is not the handicap; what the records are written from is.
- One seed each. Slot 127 against slot 115 (the same model on older code)
  was +0.37% ±0.10, so −1.45% is well outside one-seed noise.

### 2026-11-21: notes on the record

- **Trained parameters, exact.** Counted from each best checkpoint's
  optimizer state with `../no-prior-pair-1.8m/used_params.py`: the GRU
  hierarchy alone 1,067,069; `--char-kv` 1,737,149; depth records and slot
  127 1,588,157 each.
- **The train losses** in the table (1.069, 1.118, 1.051, 1.061) are the
  running training loss at step 30,000, from each log's last `progress`
  line. The epoch-8 means in the CSVs are 1.0604, 1.1227, 1.0457 and
  1.0569.

## What we learned

- Rebuilding the records at each block (depth records) is worth 1.45% at
  1.59M at no parameter cost. It was taken to 10.8M
  ([depth-records-10.8m](../depth-records-10.8m/)).
- The attention's value is copying from the recent ~1,000 characters; the
  GRU hierarchy alone can't do it.
- Separate key/value maps for character records aren't worth their
  parameters; not taken further.
- **One seed** each. The planned dropout-0.1 run at 10.8M never ran.
