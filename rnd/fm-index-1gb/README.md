---
title:    FM-index on 1 GB of books, timed as a prior
kind:     experiment
status:   running
outcome:  n/a
question: >-
  How fast can an FM-index answer what a count prior needs? Measured on the
  first 1 GB of the books mobius counts the prior from: the build's time and
  memory, the index's size, and per held-out character the counts of its
  contexts and the follower histograms of depth 1 to 24. mobius estimated the
  histograms at 0.5-1.5 ms a character and would not start a 1-2 week build
  without a measured figure under about 0.1 ms.
answer:   Running.
opened:   2026-11-24
updated:  2026-11-24
code: >-
  branch fm-index-1gb: fm_time.cpp, run.sh. sdsl-lite from xxsds at 7696536
  (2026-11-16), header-only, built with g++ 14 -O3 -march=native -std=c++20.
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
```

Results files (`results/`): `build_<v>.txt`, `check_<v>.txt`,
`query_<v>.txt`, `query4_<v>_<p>.txt`, as `key value` lines.

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

## What we learned, or what's open

Open until the 1 GB run is in.
