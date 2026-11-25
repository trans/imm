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
  checked exactly against a scan of the whole text.
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

## Design

- **The shards:** one per k-means group of the counted books, 25 in all,
  from mobius's list (one line per book: name, shard, characters).
  The list was checked against the books' own files:
  - 26,337 books, in `books.names` order;
  - every book in a shard 0-24;
  - the shard sizes add to the file's 10,303,343,939 bytes;
  - the largest, shard 0, is 1,263,824,461 characters, under 2^31, so the
    32-bit build (5 bytes a character) applies to each, about 6.3 GB peak.
- **Separators:** every book is followed by 0x01, a byte outside the 96
  characters of the vocabulary. So no context runs across a book's edge,
  and 0x01 shows up only as a follower at a book's end. mobius's count
  table counts nothing across a book's edge either. (mobius proposed 0x00,
  but sdsl's byte-alphabet build rejects zero bytes: 0 is its own
  end-of-text symbol.) The summed histograms drop 0x01 and 0.
- **The index:** as in fm-index-1gb, plain variant:
  `csa_wt<wt_huff<bit_vector>, 1u << 20, 1u << 20>` on the **reversed**
  shard text. A context's followers are the BWT symbols in its interval.
  Extending it at its end is one backward-search step.
- **The walk:** as in fm-index-1gb, from each held-out start j the context
  H[j..j+k-1] grows one character at a time, k = 1..24, in each shard until
  it no longer occurs there.
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
  seed; half of them 8 characters or shorter). For each, the shards'
  summed follower counts are compared with a memmem scan of the whole text
  in book order, separators in (`all_sep.txt`). Followers 0x01 are compared
  too.

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

It needs about 40 GB of disk (the shard texts, the whole text with
separators, the indexes) and 36 GB of memory at the four-process step.

## Results

### 2026-11-25: the program checked on 3 small shards (laptop)

These are the first 150 books of the list (53 MB), cut into 3 shards by
their group number mod 3.

- Builds: 0.9-2.8 s, 6.8-6.9 bits a character.
- Check: 300 contexts, 0 mismatches. 81 of them occur in no shard, and 635
  of the followers counted are book ends.
- Query: the summed pass's counts total equals the per-shard pass's. Each
  shard alone took 3.2-4.6 µs a character (11.6 in all). The sum took
  19.2 µs. These shards (15 MB of index each) nearly fit in the laptop's
  cache when queried one at a time, but not three in turn. At 8.8 GB
  nothing fits either way.

## What we learned, or what's open

Open until the run on the second machine.
