#!/bin/sh
# The 10 GB as 25 subject shards, run in the work directory on the second
# machine, beside books.txt, books.starts, books.names, groups_k25.books
# (mobius's list) and heldout.txt, with fm_shards compiled from fm_shards.cpp
# (see README). Results go to results/.
#
# 1. Cut the shards (every book followed by 0x01) and the whole text with
#    the same separators.
# 2. Build the 25 indexes, one after another.
# 3. The summed histograms on one core: each shard alone, then all 25
#    summed per start (best of 2).
# 4. Four processes at once, a quarter of the held-out text each, every one
#    with all 25 shards loaded.
# 5. The check: 300 contexts, the shards' summed counts against a scan of
#    the whole text (the scans on 4 threads).
set -eu
cd "$(dirname "$0")"
mkdir -p results tmp shards
nice -n 19 python3 -I split_shards.py books.txt books.starts books.names groups_k25.books shards all_sep.txt > results/split.txt 2>&1
for t in shards/shard_*.txt; do
    s=$(basename "$t" .txt)
    nice -n 19 ./fm_shards build "$t" "shards/$s.fm" tmp > "results/build_$s.txt" 2>&1
done
ls shards/shard_*.fm > shards.list
nice -n 19 ./fm_shards query shards.list heldout.txt 24 2 1 > results/query.txt 2>&1
split -n 4 -d heldout.txt heldout.part
for p in 0 1 2 3; do
    nice -n 19 ./fm_shards query shards.list heldout.part0$p 24 1 0 > results/query4_$p.txt 2>&1 &
done
wait
nice -n 19 ./fm_shards check shards.list all_sep.txt heldout.txt 300 24 4 > results/check.txt 2>&1
date -u > results/DONE
