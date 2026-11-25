---
title:    FM-index on 1 GB of books, timed as a prior
kind:     experiment
status:   concluded
outcome:  positive
question: >-
  How fast can an FM-index answer what a count prior needs? Measured on the
  first 1 GB of the books mobius counts the prior from: the build's time and
  memory, the index's size, and per held-out character the counts of its
  contexts and the follower histograms of depth 1 to 24. mobius estimated the
  histograms at 0.5-1.5 ms a character and would not start a 1-2 week build
  without a measured figure under about 0.1 ms.
answer: >-
  Fast enough at 1 GB. With plain bit vectors the follower histograms take
  15.2 us a character on one core (851 MB index, 6.8 bits a character).
  With mobius's RRR design they take 179 us (250 MB, 2.0 bits). Both builds
  took about 2.2 min at a 4.8 GB peak. From 250 MB to 1 GB the plain
  histograms grew 1.21x per doubling of the text, which extrapolates to
  about 27-29 us at 10 GB. Depths 1-3 take a third of the plain time and
  almost half of RRR's. The one-pass build used 32-bit positions, though, so
  past 2^31 characters it needs about twice the memory (about 90 GB at
  10 GB). A 10 GB index needs another build method or shards.
opened:   2026-11-24
updated:  2026-11-24
code: >-
  branch fm-index-1gb, merged into main: fm_time.cpp and run.sh at 95ef95e
  (tag rnd/fm-index-1gb/1gb); extras.sh and the per-depth pass in
  fm_time.cpp at 085218c (tag rnd/fm-index-1gb/extras). sdsl-lite from xxsds
  at 7696536 (2026-11-16), header-only, built with g++ 14 -O3 -march=native
  -std=c++20.
data: >-
  the first 999,999,959 bytes of mobius's data/pg19/imm-split-20m/books.txt
  (1 GB cut at a line end; sha256 ffb80e9f...); queries: that split's
  heldout_corpus.txt (250,000 characters, the same 50 pieces as the 100M
  split's).
slots:    []
tags:     [prior, fm-index, scale, timing, mobius]
related:  [mobius-prior-rows, transformer-prior-pg19]
---

# FM-index on 1 GB of books, timed as a prior

## Question

The prior imm's residual sits on is a table of counts, built by mobius from
up to 10 GB of books. A prior from a much larger corpus (the user's 1 TB
goal) can't be held as such a table. An FM-index can hold the text itself
in about the text's size or less and answer counts of any context. The
worry, raised on 2026-11-22, is that it's slow to read the counts out.

mobius sized the test (its plan B, mail 20261122-172735Z). The user
approved it on 2026-11-24 and chose that imm runs it, on the user's second
machine: a 4-core Xeon E-2414 with 62 GB of RAM and an NVMe disk, which
has room for the build that the laptop didn't.

## Design

- **The index:** sdsl-lite's `csa_wt`, an FM-index whose BWT is a
  Huffman-shaped wavelet tree. It's built on the **reversed** text. There,
  the interval of reverse(c) holds in its BWT exactly the characters that
  follow c in the text. So the follower histogram of c is the distinct
  symbols of that interval with their counts (`interval_symbols`).
  Extending c by one character at its end is one backward-search step.
- **Two variants of the wavelet tree's bit vectors:**
  - `rrr`: RRR-compressed (`rrr_vector<127>`), mobius's design.
  - `plain`: plain bit vectors with rank support, larger but without a block
    to decode at each rank.
  - SA and ISA samples are taken one in 2^20, since the prior never locates
    or extracts.
- **The walk:** each held-out start j extends its context H[j..j+k-1] one
  character at a time, k = 1..24, and stops when the context no longer
  occurs. Each step gives the histogram that predicts H[j+k] at depth k,
  so every (position, depth) pair the prior needs is visited once, and the
  24 depths are incremental.
- **The passes, each the best of 2:**
  - Counts: backward steps only.
  - Histograms: `interval_symbols` at each depth. The next interval comes
    from the next character's ranks.
- **Four at once:** four query processes, a quarter of the held-out text
  each, to see what the machine's four cores give together.
- **Check:** 300 sampled held-out contexts. Half are 1-8 characters long, the
  rest 1-24. Each one's histogram from the index is compared with a
  brute-force scan of the text (`memmem`).
- **mobius's extras** (its mail of 2026-11-24, `extras.sh`, run after
  `run.sh` with the program as it is now, which adds a pass timing each
  histogram by depth):
  - the same builds and query passes on the first 250 MB and 500 MB, for
    how the cost grows with the text;
  - the histogram cost by depth, 1 to 24: microseconds a call, distinct
    symbols a call, and the depth's share of the time;
  - two query processes at once on the 1 GB index, half the held-out text
    each.
- **Not measured:** the prior's use of the counts (smoothing, the gate). That
  costs the same whichever structure gave the counts.

## How to rerun

On the machine, in a work directory on a fast disk:

```
git clone https://github.com/xxsds/sdsl-lite.git && git -C sdsl-lite checkout 7696536affb18144db909e8cf8f7eafef1431b6c
g++ -O3 -march=native -std=c++20 -DNDEBUG -I sdsl-lite/include fm_time.cpp -o fm_time_rrr
g++ -O3 -march=native -std=c++20 -DNDEBUG -DFM_PLAIN -I sdsl-lite/include fm_time.cpp -o fm_time_plain
head -c 1000000000 books.txt > books1g.raw   # then cut at the last \n: books1g.txt
cp heldout_corpus.txt heldout.txt
sh run.sh
cp fm_time.cpp fm2.cpp && sh extras.sh   # compiles fm2_rrr and fm2_plain itself
```

Results files (`results/`), as `key value` lines: `build_<v>[_<size>].txt`,
`check_<v>.txt`, `query_<v>[_<size>].txt`, `query_<v>_1g_bydepth.txt` (the
rerun with the per-depth pass), `query2_<v>_<p>.txt` and
`query4_<v>_<p>.txt` (the processes run at once).

## Results

### 2026-11-24: the program checked on 10 MB (laptop)

Both variants were built on the first 10 MB of the books (cut at a line end).
The check matched the brute-force scan on all 400 sampled contexts (270 that
occur, 130 that don't), with 0 mismatches.

- **Index size:** `rrr` 2.39 bits a character, `plain` 6.81.
- **Timings,** on a laptop busy with other jobs, so a check that it works,
  not a measurement:
  - counts 23 µs a character (2.2 µs a step);
  - histograms 145 µs a character (15 µs a histogram, 17 symbols on
    average).
- **Depth reached:** contexts reached 9.7 characters on average at 10 MB.

### 2026-11-24: 250 MB, 500 MB and 1 GB on the second machine

The machine: a 4-core Xeon E-2414 with 62 GB of RAM, otherwise idle; the
work directory was on its NVMe disk. All results are in `results/`.

**The check on 1 GB:** all 300 sampled contexts (243 that occur, 57 that
don't) matched the brute-force scan of the whole text, in both variants.

**The builds** (`build_*.txt`):

| text | variant | build | peak RSS | index | bits a char |
|---|---|---:|---:|---:|---:|
| 250 MB | rrr | 29 s | 1.20 GB | 65 MB | 2.09 |
| 500 MB | rrr | 62 s | 2.39 GB | 128 MB | 2.05 |
| 1 GB | rrr | 135 s | 4.77 GB | 250 MB | 2.00 |
| 250 MB | plain | 28 s | 1.20 GB | 213 MB | 6.81 |
| 500 MB | plain | 59 s | 2.39 GB | 426 MB | 6.81 |
| 1 GB | plain | 131 s | 4.77 GB | 851 MB | 6.81 |

- **The peak is 5 bytes a character:** the text plus a 32-bit suffix
  array. sdsl-lite gives the suffix array `bits::hi(n) + 1` bits a position
  and runs a 64-bit divsufsort past 32 bits (`construct_sa.hpp`). So 10 GB
  would take about 9 bytes a character, about 90 GB, more than the
  machine's 62 GB.
- **What 10 GB would need instead:**
  - sdsl's semi-external SA-IS (`construct_config().byte_algo_sa = SE_SAIS`;
    untested), or
  - shards under 2^31 characters each.
- **The wavelet tree is all of the index:** the SA and ISA samples at one
  in 2^20 add 8 KB.

**The queries, one process** (`query_*.txt`), in µs per held-out character,
depth 1-24, best of 2 passes:

| text | variant | counts | histograms | µs a histogram | symbols a histogram | mean depth reached |
|---|---|---:|---:|---:|---:|---:|
| 250 MB | rrr | 20.3 | 149.7 | 11.67 | 21.8 | 12.8 |
| 500 MB | rrr | 22.3 | 164.9 | 12.19 | 22.6 | 13.5 |
| 1 GB | rrr | 23.4 | 179.1 | 12.60 | 23.9 | 14.2 |
| 250 MB | plain | 2.05 | 10.4 | 0.813 | 21.8 | 12.8 |
| 500 MB | plain | 2.44 | 12.9 | 0.953 | 22.6 | 13.5 |
| 1 GB | plain | 2.81 | 15.2 | 1.073 | 23.9 | 14.2 |

- **Reproducibility:** the 1 GB passes, rerun by `extras.sh`, gave 178.5
  (rrr) and 15.4 (plain).
- **How histograms per character arise:** a held-out character gets one
  histogram per depth its context reaches, about 14 at 1 GB.
- **The slope, per doubling of the text:**
  - **plain:** histograms 1.24x then 1.18x a character. That's partly
    deeper contexts and more followers, and partly slower ranks: 0.037, 0.042
    and 0.045 µs a symbol. So it's memory-bound.
  - **rrr:** 1.10x then 1.09x a character. The cost per symbol is flat at
    0.53-0.54 µs, so it's the decoding of RRR blocks, not the memory.
- **Extrapolated to 10 GB in one index** (3.3 doublings past 1 GB):
  - plain: about 27-29 µs a character;
  - rrr: about 235-240 µs a character.
  - This extends a trend measured over only 4x of text.

**Several processes at once on the 1 GB index** (`query2_*`, `query4_*`),
histograms, in µs per character for each process, and for the machine
(the total time over all characters):

| variant | 1 process | 2 at once, each | 2, machine | 4 at once, each | 4, machine |
|---|---:|---:|---:|---:|---:|
| rrr | 179 | 181 | 90 | 192 | 48 |
| plain | 15.2 | 16.8 | 8.4 | 18.8 | 4.7 |

- **rrr scales almost perfectly with processes,** as a decoding-bound pass
  should.
- **plain slows each process by 10% at two and 23% at four,** which is
  memory contention. It still gains 3.2x on the machine with 4.

**By depth, 1 GB** (`query_*_1g_bydepth.txt`, each `interval_symbols` call
timed on its own):

| depth | symbols a call | rrr µs a call | rrr share | plain µs a call | plain share |
|---:|---:|---:|---:|---:|---:|
| 1 | 84.1 | 31.6 | 17.7% | 1.06 | 6.9% |
| 2 | 60.1 | 28.9 | 16.2% | 1.74 | 11.3% |
| 3 | 45.2 | 23.7 | 13.2% | 2.30 | 15.0% |
| 4 | 33.5 | 18.4 | 10.3% | 2.07 | 13.5% |
| 5 | 25.5 | 14.8 | 8.2% | 1.68 | 10.9% |
| 8 | 13.7 | 8.7 | 4.6% | 0.90 | 5.5% |
| 12 | 6.6 | 5.0 | 2.0% | 0.51 | 2.3% |
| 16 | 3.9 | 3.5 | 0.7% | 0.37 | 0.9% |
| 24 | 3.1 | 3.6 | 0.1% | 0.73 | 0.2% |

- **Depths 1-3 carry 47% of rrr's time and 33% of plain's.** Depths 1-5
  carry 66% and 58%.
- **plain's depth 1 is cheap** (1.06 µs for 84 symbols), presumably because
  the top of its tree stays in cache. Its cost per call peaks at depth 3.
- **What a table for the shortest contexts would save,** worked out by
  subtracting their share, not measured. Such a table would hold each
  short context's followers and the interval that continues the walk.
  Contexts of up to 3 characters number at most 96 + 96^2 + 96^3.
  - Taking out depths 1-3 would leave about 95 µs a character for rrr and
    10 for plain.
  - Taking out depths 1-5 would leave about 62 and 6.5.

## What we learned, or what's open

- **The readout is not the obstacle at 1 GB.** mobius's threshold for a 10
  GB build was under 100 µs a character at 1 GB, and plain bit vectors give
  15.2 µs. That's about 4.7 µs a character over the machine's 4 cores. So
  the histograms for a 100M-character training text would take about 8
  minutes at 1 GB, and roughly a quarter of an hour at 10 GB (both on 4
  cores).
- **The trade-off is memory against speed:**
  - RRR is 3.4x smaller (2.0 against 6.8 bits a character).
  - It is 12x slower, from decoding.
  - In between, untested: `rrr_vector` with smaller blocks, `hyb_vector`, or
    `bit_vector_il`.
- **The build was cheaper than estimated** (2.2 min and 4.8 GB at 1 GB,
  against 15-30 min and 8-9 GB). It doesn't carry to 10 GB in one piece,
  because of the 64-bit suffix array. The open question for 10 GB is
  SE_SAIS against shards, and its cost has to be measured.
- **Size is the limit for a much bigger corpus:**
  - At 6.8 bits a character, 10 GB is an 8.5 GB index.
  - 100 GB would be 85 GB, more than this machine's RAM.
  - At 2 bits, 1 TB is 250 GB, which needs shards, disk, or a bigger machine.
- **Not measured:** the prior's use of the counts (smoothing and the gate),
  and the effect on imm of a prior from an FM-index rather than mobius's
  table. Built from the same text, the counts are the same.
