---
title:    FM-index on 10 GB of books as 25 subject shards (and a 26th)
kind:     experiment
status:   concluded
outcome:  positive
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
  Yes, exactly, and fast enough offline. The 26 shards' summed counts
  matched a scan of the whole 10.4 GB on 300 contexts each way (followers
  and predecessors), 0 mismatches. Each set of 26 indexes built in 21 min
  (largest shard 2.9 min at a 6.0 GB peak), 8.8 GB in RAM. On one core the
  follower histograms cost 270 us a held-out character with each shard
  walked over all the starts, but 690 us with all 26 walked in turn for
  each start: the cache holds one shard's most-used parts, not 26 shards'.
  Two shards in turn cost 1.22x, four 1.48x, eight 1.78x. Taking the starts
  in blocks of 4,096 per shard brings all 26 back to 1.08x (286 us), so
  summing on the fly can keep nearly the one-shard-at-a-time rate. Four
  processes give 232 us a character for the machine. The predecessor set
  costs the same within 3%.
opened:   2026-11-25
updated:  2026-11-25
code: >-
  branch fm-index-10gb-shards, merged into main: fm_shards.cpp,
  split_shards.py, run.sh. The laptop test at 2c9c937 (tag
  rnd/fm-index-10gb-shards/laptop-small); the run on the second machine at
  eaec6d9 (tag rnd/fm-index-10gb-shards/server); fm_shards order at 36d60f9
  (tag rnd/fm-index-10gb-shards/order). sdsl-lite from xxsds at 7696536,
  header-only, built with g++ 14 -O3 -march=native -std=c++20 -pthread.
data: >-
  mobius's data/pg19/imm-split-20m-books/books.txt (10,303,343,939 bytes,
  26,337 books) cut by mobius's rnd/books-heldout/groups_k25.books (mobius
  76a1227), and mobius's data/pg19/shards/extra160.txt (160 books,
  92,780,874 bytes) as a 26th shard; queries: data/pg19/imm-split-20m/heldout_corpus.txt (250,000
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

mobius added a 26th shard the same night (mail 20261125-025437Z): the 160
books of the 20M split's corpus that the books split leaves out (its 50
held-out books and 110 more), as mobius's `data/pg19/shards/extra160.txt`
(92,780,874 bytes). mobius checked that the books split's corpus is the
20M split's minus exactly those books, same order, same bytes. Its size
plus the 25 shards' is 10,396,124,813 bytes, the 20M split's whole.
So shards 0-24 sum to the books split's corpus, and all 26 sum to the
20M split's, whose count prior and runs (1.6175 alone, 1.3825 under the
10.8M) mobius will reproduce first. Shard 25 is left out when the
held-out books are scored.

## Design

- **The 26th shard:** mobius's extra160 books, with their own starts and
  names, cut and built like the others. The timing passes and the checks
  use all 26 shards. The per-shard pass shows shard 25's own cost, so the
  25 subject shards' cost can be read off as well.
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
`data/pg19/imm-split-20m-books/`, mobius's `groups_k25.books`, the three
`extra160` files from mobius's `data/pg19/shards/`, and the held-out text
as `heldout.txt`:

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
- With a 4th shard from the first 10 extra160 books, cut by the extra
  arguments, both checks again gave 0 mismatches (not kept).

### 2026-11-25: 10.4 GB in 26 shards on the second machine

The 26 shards hold 26,497 books, 10,396,124,813 characters (10,396,151,311
with the separators). They range from shard 0 (2,690 books, 1.26 GB) to
shard 23 (115 books, 10 MB); the sizes are in `results/server/split.txt`.
The run started 03:02 UTC and finished 04:08 UTC (`results/server/run.log`).

**Builds** (`build_shard_*.{rev,fwd}.txt`):

| | rev (followers) | fwd (predecessors) |
|---|---:|---:|
| All 26, one after another | 1,267 s (21.1 min) | 1,275 s (21.2 min) |
| Largest, shard 0 | 172 s, 6.03 GB peak | the same |
| Index, all 26 | 8.82 GB (6.79 bits a character) | 8.82 GB |

**Checks** (`check_{rev,fwd}.txt`): 300 contexts each, the shards' summed
counts against a scan of the whole text.

| | Mismatches | In no shard | Book edges among the counts |
|---|---:|---:|---:|
| rev (followers) | 0 | 43 | 113,583 book ends |
| fwd (predecessors) | 0 | 40 | 25,037 book starts |

Each check took about 225 s on 4 threads, at a 19.7 GB peak (the indexes
and the whole text).

**One core** (`query_{rev,fwd}.txt`), per held-out character, depths 1-24:

| | rev (followers) | fwd (predecessors) |
|---|---:|---:|
| Each shard alone, all 26 added up | 270.1 µs | 276.9 µs |
| The same, shards 0-24 only | 263.2 µs | 269.7 µs |
| All 26 per start, summed (best of 2) | 689.6 µs | 717.1 µs |
| Mean depth reached, a shard | 12.19 | 12.19 |
| Mean depth reached, all shards pooled | 16.48 | 16.48 |
| Peak memory | 9.8 GB | 9.8 GB |

- **Shards alone:** shard 0 (1.26 GB) costs 14.0 µs a character, close to
  fm-index-1gb's 15.2 µs at 1 GB. Shard 3 (745 MB) costs 14.2. Cost
  follows the number of followers a histogram has (24.0 against 21.3) as
  well as the text's size. The smallest shards cost 2.8 µs (shard 23,
  depth 6.4) and 4.0 µs (shard 24, depth 8.8).
- **The order:** walking the shards in turn for each start costs 2.55x
  (rev) and 2.59x (fwd) what walking each shard over all the starts does.
  The counts are identical, and the adding up is a small part (see the
  laptop test), so it's the order. The second machine's cache is 12 MB
  (L3) and 2 MB a core (L2). One shard's most-used parts, its shortest
  contexts, stay in it across starts; 26 shards' don't.
- **Load:** the 8.8 GB of a set load in 2.7 s from the page cache.

**Four processes** (`query4_rev_*.txt`, rev only), a quarter of the
held-out text each, all 26 shards in each, summed per start: 900-930 µs a
character each. The slowest took 58.1 s for its 62,500, so the machine
does 232 µs a character, 2.97x one core.

### 2026-11-25: the order of the walks (second machine)

After the run, the user asked whether two shards at a time would still be
too much for the cache. `fm_shards order` (`36d60f9`) times the walks
alone, nothing summed, over the rev set: shards taken g at a time, and
within a group the starts in blocks of B, each shard of the group walking
every start of a block (`results/server/order_rev.txt`). Per held-out
character, one core:

| Order | µs a character | Against one at a time |
|---|---:|---:|
| One shard at a time, all starts | 264.9 | 1.00x |
| 2 shards in turn, per start | 323.2 | 1.22x |
| 4 in turn | 391.3 | 1.48x |
| 8 in turn | 472.6 | 1.78x |
| All 26 in turn | 678.8 | 2.56x |
| All 26, blocks of 16 starts | 517.5 | 1.95x |
| All 26, blocks of 256 | 401.6 | 1.52x |
| All 26, blocks of 4,096 | 285.7 | 1.08x |

- Each doubling of the shards in turn adds about 20%. There's no cliff, so
  two at a time mostly fits.
- All 26 in turn without summing (678.8) is within 2% of the summed pass
  (689.6), so the adding up costs little; the order is the cost.
- The reuse that the cache gives spans thousands of starts: blocks of 256
  recover only half the gap, blocks of 4,096 nearly all of it. The blocked
  order wasn't timed with the summing, which then has to hold a block's
  histograms (about 4,096 starts x 24 depths x 20 symbols).

## What we learned, or what's open

- **The subject partition is exact and cheap to build.** 26 shards, both
  directions, 42 minutes of builds, 17.6 GB of index, and 0 mismatches
  against the whole text.
- **The plain sum over all shards costs about 10x one index.** 270 µs a
  character against the 27-29 µs fm-index-1gb extrapolated for one 10 GB
  piece. Each shard's walk costs nearly what a whole index's does: a shard's
  contexts reach 12.2 deep on average, against 16.5 pooled. That is the
  price mobius expected (250-350 µs), and why consulting only some shards
  per document is the step after.
- **Order matters 2.5x.** Walk shard by shard, or in blocks of thousands of
  positions; never all shards per position. mobius's event-file walks
  should go shard by shard. Its estimate of about 1 µs an interval a shard
  matches the shard-by-shard rate here (0.85 µs a histogram on average,
  0.99 for shard 0).
- **Predecessors cost the same as followers,** within 3%, as the symmetry
  says. Same depths reached.
- **Open:** the blocked order with the summing, timed, if on-the-fly
  summing is wanted.
- **Where the indexes are kept:** the run's `/var/tmp` work directory on
  the second machine is emptied of files unused for 30 days (systemd's
  tmpfiles, `q /var/tmp ... 30d`). So on 2026-11-25, at the user's word,
  the 52 indexes moved to a folder in the user's home directory there, on
  the mirrored ZFS pool. The folder holds `shards/`, `pairs26/` and
  `pairs25/` (links), `sha256.txt` (checked after the copy) and a README.
  They take 17.6 GB, 9.4 GB on disk with the pool's lz4. The shard texts
  and the whole text stay in `/var/tmp`, rebuildable from mobius's data.
