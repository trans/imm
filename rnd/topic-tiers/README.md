---
title:    "Topic tiers: reading by piece, sentence and paragraph GRUs"
kind:     experiment
status:   concluded
outcome:  negative
question: >-
  Can imm carry what a piece of text is about, through training by piece,
  sentence and paragraph tiers over the word tier, longer backpropagation
  through the tiers, and a top-down read of the paragraph state?
answer: >-
  No arm gained at 20M characters. Scored by piece, slot 63's model gives
  2.7735; trained with resets (R1) 2.7738; the sentence and paragraph stack
  2.7785 (R2) and 2.7746 with centroids (R2c); top-down 2.7867 (+0.43%);
  the sentence tier alone 2.7751 (S1) and with a per-position gate 2.7736
  (S2). Swapping pieces' sentence states costs only 0.20%. A topic prior
  applied to imm's output after the fact gained 1.01%, but the user ruled
  out adding topic knowledge as a prior.
opened:   2026-11-17
updated:  2026-11-18
code: >-
  branch topic-tiers (kept, not merged into main): --documents 4616a6a;
  the stack, --tier-bptt, --tier-input end+centroid, --top-down 9b2a329;
  --swap-tier 07f370d; a sentence tier alone ending at blank lines
  39ae260; --tier-position-gate c732057.
data: >-
  PG-19 20M split, data/pg19/imm-split-20m: 800 training pieces, 50
  held-out pieces of 5,000 characters (249,999 targets). Mobius's
  pg19split20m_gated_d24_pg10g prior (alone 3.068) with pg19split20m_kn
  evidence. Scored one stream per piece (--documents) unless stated.
slots:    [75, 76, 77, 80, 81, 83, 89]
tags:     [hierarchy, topic, sentence-tier, paragraph-tier, documents, pg19]
related:  [imm-1.85m, phrase-tier, word-tier, memory-window, data-context-size, memory-knockouts]
---

# Topic tiers: reading by piece, sentence and paragraph GRUs

## Question

imm's lead over the transformer grew through a piece and sat on repeated
words (`imm-1.85m`). Could a hierarchy above the word tier carry the
piece's topic? The plan (approved by the user via @air, 2026-11-17, with an
"overtrust" arm included) built it in steps, each against slot 63's model
(h192, two blocks, word tier 168 with own step, learned prior strength,
d24 rows, 41 epochs, 2.7757).

## Design

- **R0:** slot 63's checkpoint scored one stream per piece, each piece
  from a zero state (`--documents`). The usual score runs 64 streams that
  carry state across piece boundaries.
- **R1:** slot 63 trained with `--documents` (each piece read from its
  start, a learned start state).
- **Overtrust:** R1 with `--prior-residual-l2 0` (@air's prediction: the
  leash keeps imm trusting the prior where it is wrong).
- **R2:** R1 with a sentence tier (steps at `.!?` or a blank line) and a
  paragraph tier (steps at a blank line), 48 wide each, over the word tier,
  which shrinks to 128; `--tier-bptt 32` carries each tier's last 32
  inputs so its gradient spans more than one update. R2 reads each tier's
  end state; **R2c** the end state and the centroid of its inputs.
- **R3:** R2c with `--top-down`: the character GRU and the word tier read
  the paragraph state as of the update's start.
- **Swap test** (`--swap-tier`, eval only): before every chunk of 64
  segments, one tier's state passes to the next piece's stream. If the
  loss doesn't rise, the tier carries nothing of its piece.
- **S1:** the sentence tier alone (48, end + centroid, `--tier-bptt 32`)
  added to R1, the word tier kept at 168.
- **S2:** S1 with `--tier-position-gate`: each position scales the
  sentence tier's add by f = 2 sigmoid(Linear([its own state ; the tier
  state])), zero-init so f = 1 at the start; 241 parameters.
- **Topic prior after the fact** (@air): slot 63's and S1's full held-out
  distributions (`dist_dump.py`) rescaled by a 300-topic mixture of
  unigrams fitted on 3,000 books of the 10 GB, none of them imm's training
  books, the mix refitted after every word.

## How to rerun

From the `topic-tiers` branch; paths are the original machine's. Comments
that quoted the user were paraphrased; every command is as it ran.

| run | script | commit | slot | proposed tag |
|---|---|---|---|---|
| R0 | `topic_slot.sh`'s `r0` arm, or `r0/run.sh` (see below); `r0/compare.py` pairs it | `9b2a329` | 75 | `rnd/topic-tiers/slot75` |
| R1, overtrust | `topic_slot.sh 75` with arms `r1 ot` | `9b2a329` | 75 | (same) |
| R2, R2c | `topic_slot.sh 76` with arms `r2 r2c` | `9b2a329` | 76 | `rnd/topic-tiers/slot76` |
| R3 | `r3_slot.sh 77` with arm `r3` | `9b2a329` | 77 | `rnd/topic-tiers/slot77` |
| swap test | `swap_slot.sh 80` | `07f370d` | 80 | `rnd/topic-tiers/slot80` |
| S1 | `topic_slot.sh 81 s1` | `39ae260` | 81 | `rnd/topic-tiers/slot81` |
| S2, then its gate map | `topic_slot.sh 83` with arms `s2 s2map` (`gate_map.py`, `gate_buckets.py`) | `c732057` | 83 | `rnd/topic-tiers/slot83` |
| distributions for @air | `dist_slot.sh 89` (runs `dist_dump.py`) | `c732057` | 89 | `rnd/topic-tiers/slot89` |

`topic_slot.sh` survives only in its last form, which also holds the S1,
S2 and S2-map arms; slots 75 and 76 ran an earlier copy with the same R1,
overtrust, R2 and R2c commands. The exact arm strings passed in slots 75
and 76 (and 77, 83) aren't recorded; slot 81's was `s1`. R0 was also written as a stand-alone script,
`r0/run.sh` with `r0/slot.sh`, on an earlier commit (`4616a6a`, which adds
`--documents`); both it and `topic_slot.sh`'s `r0` arm score the same
checkpoint the same way, and the notes don't say which one ran. `r3_slot.sh` is the copy that ran slot 77:
it picks R3's tier input from R2 and R2c's scores. The `m32` and `m128`
arms are in `../memory-window/mem_slot.sh`.

Paired comparisons here (`scripts/rnd/rpair.sh`; "imm" is A, "tx" is B):
`results/r0_vs_usual.txt`; `results/r1/` (R1 and overtrust against R0, R1
and the d24 transformer, and `*_place.txt` by place in the piece);
`results/r2/`, `results/r3/`, `results/s1/`, `results/s2/` (S2 against R1
and S1, and the gate's factor by bucket). The topic-prior analysis is
@air's (air `7b98d3c`, `experiments/topic_prior/`).

## Results

### 2026-11-17: R0, reading by piece (slot 75)

| target's place in its 5,000-character piece | targets | usual | by piece | change |
|---|---:|---:|---:|---:|
| all | 249,999 | 2.7757 | **2.7735** | **−0.08% ±0.02** |
| 0–16 | 799 | 8.8916 | 9.0766 | +2.08% ±3.79 |
| 16–64 | 2,400 | 3.2717 | 3.1158 | −4.77% ±1.27 |
| 64–256 | 9,600 | 2.7985 | 2.7807 | −0.63% ±0.24 |
| 256–1000 | 37,200 | 2.7670 | 2.7663 | −0.03% ±0.05 |

The book before costs a little at a piece's start; past 256 characters,
nothing. R1 onward are scored by piece, so compare them with 2.7735.

### 2026-11-17: R1 and overtrust (slot 75)

| | held-out, by piece | change | against the d24 transformer | residual α | prior strength | last training NLL |
|---|---:|---:|---:|---:|---:|---:|
| R0 (slot 63) | 2.7735 | | | 0.183 | 0.984 | |
| R1 | 2.7738 | +0.01% ±0.06 against R0 | −1.39% ±0.08 | 0.187 | 0.987 | 0.9461 |
| overtrust | 2.7755 | +0.06% ±0.06 against R1 | −1.33% ±0.08 | 0.325 | 0.910 | 0.9397 |

- R1 gains at a piece's start (−4.7% ±0.5 over the first 256 characters)
  and loses +0.28–0.38% ±0.1 past 1,000. Nothing overall.
- Overtrust does what @air predicted where predicted (−2.77% ±0.36 on new
  words the prior misses by 3+ nats) and pays it back elsewhere
  (capitalised repeats +5.7% ±1.1). Without the L2 imm's own share nearly
  doubles and it overfits: the L2 is a regulariser as much as a trust
  setting.

### 2026-11-17: R2, R2c and R3 (slots 76 and 77)

| | held-out, by piece | change | epochs 10 / 20 / 30 / 41 | wall time |
|---|---:|---:|---|---:|
| R1 (word tier 168) | 2.7738 | | 2.805 / 2.784 / 2.778 / 2.774 | 3,853 s |
| R2 (end) | 2.7785 | **+0.17% ±0.06** against R1 | 2.805 / 2.785 / 2.779 / 2.781 | 3,982 s |
| R2c (end + centroid) | 2.7746 | **+0.03% ±0.06** against R1 | 2.801 / 2.782 / 2.779 / 2.775 | 4,051 s |
| R3 (R2c + top-down) | 2.7867 | **+0.43% ±0.06** against R2c | 2.810 / 2.794 / 2.791 / 2.788 | |

- The tiers gain nothing; no arm passed the plan's 0.15% bar for a second
  seed. R2s (only if R2 gained) wasn't run.
- The centroid helps over the end state alone (R2c −0.14% ±0.05 against
  R2).
- R2 also narrowed the word tier (168 → 128), worth about 0.57% at 168
  (`memory-knockouts`), so its loss may be the width, not the stack. The
  user judged that an unfair comparison: it took width from the level with
  the most data for levels that are much sparser, which should matter more
  as the corpus grows. Per epoch of the 20M text: about 3.5M word steps,
  256k sentence steps, 99k paragraph steps.
- Top-down hurts from epoch 10 on, in training too, and where the word
  GRU's worth is (phrase ends +3.0% ±0.5, sentence ends +4.4% ±1.0).

### 2026-11-18: the swap test (slot 80)

| model | swapped | score | change | first 500 chars | 500–1500 | 1500–3000 | 3000–5000 |
|---|---|---:|---:|---:|---:|---:|---:|
| R2c | nothing | 2.7746 | | | | | |
| R2c | sentence | 2.7802 | **+0.20% ±0.02** | +0.04% | +0.14% | +0.21% | +0.26% |
| R2c | paragraph | 2.7768 | +0.08% ±0.02 | +0.02% | +0.03% | +0.12% | +0.08% |
| R3 | nothing | 2.7867 | | | | | |
| R3 | sentence | 2.7886 | +0.07% ±0.02 | +0.02% | +0.03% | +0.10% | +0.08% |
| R3 | paragraph | 2.7893 | +0.09% ±0.03 | +0.03% | −0.04% | +0.17% | +0.13% |

The sentence state carries something about its piece, more the further in,
and older than the attention's reach. But it is small, and it is reliance,
not worth: R2c is level with R1.

### 2026-11-18: S1 and S2 (slots 81 and 83)

| | held-out, by piece | change | train NLL, epoch 41 | wall time |
|---|---:|---:|---:|---:|
| R1 | 2.7738 | | 0.9786 | 3,853 s |
| S1 | 2.7751 | +0.05% ±0.06 against R1; −1.35% against the transformer | 0.9761 | 3,955 s |
| S2 | 2.7736 | −0.05% ±0.06 against S1; −0.01% ±0.05 against R1 | 0.9756 | 4,008 s |

- S1, the sentence tier added fairly, neither gains nor costs at 20M.
  1,918,309 trained parameters.
- S2's gate is used where you'd expect: f averages 0.84; 1.37 at a
  sentence's end, 0.75 at other word ends; 1.05 in a piece's first 256
  characters, 0.80 past 2,500. But where the add is loudest S2 scores
  worse than S1 (sentence ends +2.96% ±0.89).
- Both sentence arms fit the training text about 0.3% better and the
  held-out text no better: at 20M the tier learns something that doesn't
  carry to new books, as the user expected for sparse sentence data.

### 2026-11-18: a topic prior on imm's output, after the fact (@air)

| | alone | with topics | change |
|---|---:|---:|---:|
| slot 63 (by piece) | 1.4717 bits/char | 1.4567 | **−1.01% ±0.10** (48 of 50 pieces) |
| S1 | 1.4725 | 1.4584 | −0.95% ±0.10 (48 of 50) |
| the d24 prior alone | 1.6175 | 1.5938 | −1.45% ±0.12 |

- imm already holds about a third of what the topics carry: in bits it
  gains 63% of what the prior alone gains. S1's sentence tier holds no more
  topic than slot 63.
- It is knowledge from outside the 20M text, worth twice the 128-record
  gain (`memory-window`). It points to data.

### 2026-11-18: the user's decision

imm proposed topic-adapted prior rows (the d24 prior rescaled by the
piece's topic mix). The user declined it as one more add-on to the prior,
like the document-level rows before it, and closed the line: no topic
level and no document rows for imm. imm is a hybrid model, and it should
learn long-range knowledge itself and take on a transformer on its own
merits at scale.

## What we learned

- At 20M characters, sentence and paragraph tiers, longer backpropagation
  through them, a per-position gate and top-down reads add nothing to the
  word-tier model. Whether they would on more data wasn't tested.
- Reading by piece gains at a piece's start and loses a little later;
  net nothing.
- The L2 leash on the residual is a regulariser as much as a trust
  setting.
- Sentence records in the attention (build 4 of the plan) were proposed on
  S2 and not built.
