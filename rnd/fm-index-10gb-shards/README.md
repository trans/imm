---
title:    FM-index on 10 GB of books as 25 subject shards
kind:     experiment
status:   planned
outcome:  open
question: >-
  Can the count prior come from 10 GB of books held as 25 FM-indexes, one
  per subject group, with their follower counts added at lookup time? Per
  shard: the build's time, memory and size. Over the 250K held-out text:
  the summed follower histograms of depth 1 to 24, in us a character on one
  core and on four, and how deep each shard's contexts reach. And the sum
  checked exactly against a scan of the whole text. Each shard gets a second
  index on the unreversed text for the predecessor counts the gate's suffix
  features need, timed on one core and checked the same way.
answer: >-
  Not yet run.
opened:   2026-11-25
updated:  2026-11-25
code: >-
  branch fm-index-10gb-shards: fm_shards.cpp, split_shards.py, run.sh.
  sdsl-lite from xxsds at 7696536, header-only, built with g++ 14 -O3
  -march=native -std=c++20 -pthread.
data: >-
  mobius's data/pg19/imm-split-20m-books/books.txt (10,303,343,939 bytes,
  26,337 books) cut by mobius's rnd/books-heldout/groups_k25.books (mobius
  76a1227); queries: data/pg19/imm-split-20m/heldout_corpus.txt (250,000
  characters, sha256 095cbd13...), as in fm-index-1gb.
slots:    []
tags:     [prior, fm-index, scale, timing, shards, mobius]
related:  [fm-index-1gb, mobius-prior-rows]
---

# FM-index on 10 GB of books as 25 subject shards

## Question

[fm-index-1gb](../fm-index-1gb/) found an FM-index fast enough to serve
the count prior at 1 GB: 15.2 µs a held-out character on one core for the
follower histograms of depth 1 to 24, with plain bit vectors. It left 10 GB
open, since one 10 GB piece needs 64-bit suffix-array positions in the
build, about 90 GB of memory, against the second machine's 62 GB.

The user's direction (to mobius, 2026-11-24) is to group the material by
subject and index the groups separately, combined on the fly, rather than
one giant corpus. The user said go to building 10 GB that way on
2026-11-25 (in mobius's session; mobius's mail 20261125-014355Z), and
asked imm to go ahead if the k-means grouping is the right call. It is:
counts add, so the pooled prior is the same whatever the grouping;
mobius's leave-one-group-out test found the groups meaningful (removing a
group cost its own books about 10x what it cost others); and regrouping
costs only a rebuild, about 25 minutes of the second machine.

So this measures:

- what one lookup costs when every one of 25 shards is consulted and the
  counts added per symbol (the plain sum, the pooled prior). mobius expects
  250-350 µs a character on one core, 25 times the 1 GB figure, less where
  a small shard's walk stops early. Consulting only some shards per
  document is the step after, which is what the user's plan is about;
- how deep each shard's contexts reach on the held-out text, which says how
  early the shards that don't fit a document stop;
- that the sum is exact.

mobius found one more need before the build started (mail
20261125-024824Z). Five of its gate's eleven features describe the
characters seen *before* a context: how often it occurs with one, how many
distinct ones, their entropy, and so on (mobius's
`src/mobius/gated_counts.cr`, lines 451-455, read from the level built at
line 933). An index on the reversed text gives followers. Predecessors come
from an index on the text as it stands. So each shard gets both, the two
together a bidirectional index, and the build doubles.

## Design

- **The shards:** one per k-means group of the counted books, 25 in all,
  from mobius's list (one line per book: name, shard, characters).
  The list was checked against the books' own files:
  - 26,337 books, in `books.names` order;
  - every book in a shard 0-24;
  - the shard sizes add to the file's 10,303,343,939 bytes;
  - the largest, shard 0, is 1,263,824,461 characters, under 2^31, so the
    32-bit build (5 bytes a character) applies to each, about 6.3 GB peak.
- **Separators:** every book is preceded and followed by 0x01 (one 0x01
  between two books), a byte outside the 96 characters of the vocabulary.
  So no context runs across a book's edge. 0x01 shows up only as a follower
  at a book's end or a predecessor at its start, and the first book of a
  shard has one before it, as it has in the whole text. mobius's count
  table counts nothing across a book's edge either. (mobius proposed 0x00,
  but sdsl's byte-alphabet build rejects zero bytes: 0 is its own
  end-of-text symbol.) The summed histograms drop 0x01 and 0.
- **The indexes:** as in fm-index-1gb, plain variant:
  `csa_wt<wt_huff<bit_vector>, 1u << 20, 1u << 20>`, two per shard.
  - **rev**, on the reversed shard text. A context's followers are the BWT
    symbols in its interval, and extending it at its end is one
    backward-search step.
  - **fwd**, on the shard text as it stands. A context's predecessors are
    the BWT symbols in its interval, and extending it at its start is one
    backward-search step.
- **The walk:** as in fm-index-1gb. In the rev indexes, from each held-out
  start j, the context H[j..j+k-1] grows one character at a time, k =
  1..24, in each shard until it no longer occurs there. The fwd indexes run
  the same code on the reversed held-out text: from each end position the
  context grows toward its start. By symmetry the cost should match.
- **The passes on one core** (`fm_shards query`):
  - Each shard alone, over all the held-out starts. This gives the shard's
    µs a character (what a process holding only that shard would spend),
    the mean depth its contexts reach, and its counts' total.
  - The sum, best of 2. For each start, every shard's walk, with the counts
    added into one histogram per depth. The summed histograms are then read
    (folded into a checksum) and cleared. The counts' total must equal the
    per-shard pass's.
- **Four cores:** four processes, a quarter of the held-out text each, each
  with all 25 shards loaded (4 × about 8.8 GB). That's the same shape as
  the 1 GB runs. One process per shard would need less memory, but its
  histograms would still have to be merged per start. Its cost can be read
  off the per-shard pass.
- **The check** (`fm_shards check`): fm-index-1gb's 300 contexts (the same
  seed; half of them 8 characters or shorter). For each, the shards' summed
  follower counts are compared with a memmem scan of the whole text in book
  order, separators in (`all_sep.txt`), 0x01 counts included. The fwd
  indexes are checked the same way on predecessors, with their contexts
  drawn from the reversed held-out text.
- **Order:** the rev set is built and measured first, since its summed
  lookup is the open question. Then the fwd set: its builds, its one-core
  pass and its check. Its four-process pass is left out, since four
  processes each holding both sets would need about 70 GB.

## How to rerun

On the second machine, in a work directory holding `books.txt`,
`books.starts` and `books.names` from mobius's
`data/pg19/imm-split-20m-books/`, mobius's `groups_k25.books`, and the
held-out text as `heldout.txt`:

```sh
git clone https://github.com/xxsds/sdsl-lite && git -C sdsl-lite checkout 7696536
g++ -O3 -march=native -std=c++20 -DNDEBUG -pthread -I sdsl-lite/include fm_shards.cpp -o fm_shards
sh run.sh
```

It needs about 50 GB of disk (the shard texts, the whole text with
separators, the 50 indexes) and 36 GB of memory at the four-process step.

## Results

### 2026-11-25: the program checked on 3 small shards (laptop)

These are the first 150 books of the list (53 MB), cut into 3 shards by
their group number mod 3, with both index sets.

- Builds: 0.9-3.6 s each, 6.8-6.9 bits a character.
- Check, rev (followers): 300 contexts, 0 mismatches. 81 of them occur in
  no shard, and 635 of the followers counted are book ends.
- Check, fwd (predecessors): 300 contexts, 0 mismatches. 83 occur in no
  shard, and 139 of the predecessors counted are book starts.
- Query: in each set the summed pass's counts total equals the per-shard
  pass's. The follower totals didn't change when the leading separator was
  added. Both sets reach the same mean depth (10.11 a shard, 11.29
  pooled), as the symmetry says.
  - rev: each shard alone took 11.6 µs a character in all, the sum 18.9.
  - fwd: 12.1 and 20.2.
  - These shards (15 MB of index each) nearly fit in the laptop's cache
    when queried one at a time, but not three in turn. At 8.8 GB nothing
    fits either way.

## What we learned, or what's open

Open until the run on the second machine.
