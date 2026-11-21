# Research and development experiments

Each idea we try gets its own directory here, made before the experiment runs.
Its `README.md` says what the idea is, which code it ran, how to run it again
and what came of it. [`CHRONOLOGY.md`](CHRONOLOGY.md) lists the experiments in
order.

The code an experiment ran stays on its own branch. This directory, on `main`,
holds the record: the README, the run scripts, and small results such as the
paired comparisons. A run can be repeated however far `main` has moved,
because each README names the commit it ran and that commit is tagged
(below).

## An entry

```text
rnd/
  <experiment>/
    README.md        the record (front matter below)
    *.sh, *.py       the scripts that ran it, as they ran
    results/         small results: paired comparisons, tables
```

Checkpoints, per-character loss files and training logs stay in `runs/`, which
git ignores. Commit the explanation, not the bulk output.

Analysis tools that many experiments share live in `scripts/rnd/`, not here,
so every directory in `rnd/` is an experiment. A script that belongs to one
experiment stays in its entry.

Each `README.md` starts with front matter:

```yaml
title:    short name
kind:     experiment | design
status:   planned | running | concluded | parked | inconclusive
outcome:  positive | negative | mixed | reference | n/a
question: the question the experiment answers
answer:   the answer so far, with the headline numbers
opened:   date
updated:  date
code:     branch, the commit each run used, and its tag
data:     the split (see the protocol below)
slots:    the GPU slots it ran in, if any
tags:     [...]
related:  [other experiment directories]
```

The body follows:

- **Question**
- **Design:** what changed against which baseline.
- **How to rerun**
- **Results:** dated sections, appended. An earlier result is never
  rewritten. A correction is a new dated section that says what was wrong.
- **What we learned, or what's open**

An experiment from before this directory existed (2026-11-21) may have lost its
scripts. Its README says so, and gives the flags it is known to have used, so
it can be rebuilt.

## Running an old experiment again

Each run's commit is tagged `rnd/<experiment>/slot<N>`, or
`rnd/<experiment>/<run>` for one that took no slot:

```sh
git worktree add ../imm-rerun rnd/<experiment>/slot<N>
ln -s "$PWD/data" ../imm-rerun/data
ln -s "$PWD/runs" ../imm-rerun/runs
```

Then run the entry's script from that worktree. Scripts are kept as they ran,
so their paths (the scratch directory `S`, the worktree) are the original
machine's: set them before running.

## Evaluation protocol

Experiments since 2026-11-19 use this protocol unless their README says
otherwise. Earlier ones name their own split.

- **Corpus:** PG-19, the 100M split, `data/pg19/imm-split-100m` (built by
  mobius `scripts/pg19-more.cr`).
  - **Training text:** 100,000,000 characters, 1,005 pieces.
  - **Held-out text:** 250,000 characters, 50 pieces of 5,000.
  - **Hashes** (in its `manifest.json`): training `fc55205e…`, held-out
    `095cbd13…`.
  - **Normalisation:** the text is normalised to printable ASCII plus `\n`:
    96 characters.
- **Score:** held-out perplexity per character, which is e raised to the mean
  loss in nats per character. Lower is better. It's `val_ppl` in the run logs,
  scored in fp32. Bits per character are log₂ of it.
- **No prior:** since slot 101, no count prior and no prior input. Earlier
  experiments used mobius's d16 or d24 prior and say so.
- **Training:** 8 epochs of the 100M characters (798M with the pieces cut to
  whole updates), cosine learning rate, 64 streams, one seed unless stated.
- **Paired comparisons:** `scripts/rnd/rpair.sh A.npz B.npz OUT.txt` pairs two
  runs' per-character losses (`--char-loss-output`) over the held-out
  characters.
  - It reports A's change against B as a percentage, exp(mean difference)
    − 1, with ± one standard error of the mean difference.
  - It breaks the change down by place in the word, what is predicted, the
    prior's loss on the target, place in the piece, and new or repeated
    words.
  - The ± treats characters as independent, so it understates the spread
    between runs. The same model on older code differed by 0.37% (slot 127
    against slot 115). Read gaps under about half a percent as noise.
- **Screens:** an architecture idea can be read at epoch 3 of the 8-epoch
  schedule, against the base's own epoch-3 score.
  - Gaps of about 1% or more at epoch 3 kept their sign in eight past pairs.
  - Smaller ones moved by up to 0.5%.
  - Learning rate and dropout show late, so they need full runs.
- **Reference models:** the transformer baseline (`scripts/transformer_baseline.py`):
  - block 1,024, 1,986,240 parameters (called the 1.84M, its size at block
    256): 3.1862;
  - block 1,024, 11,051,904 parameters (the 10.8M): 2.7458.

## Index

[`CHRONOLOGY.md`](CHRONOLOGY.md) lists every experiment, in order, with its
outcome.
