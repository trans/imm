#!/bin/sh
# The 10 GB as 25 subject shards and a 26th, run in the work directory on
# the second machine, beside books.txt, books.starts, books.names,
# groups_k25.books (mobius's list), extra160.{txt,starts,names} (mobius's
# 26th shard) and heldout.txt, with fm_shards compiled from fm_shards.cpp
# (see README). Results go to results/.
#
# 1. Cut the shards (every book between two 0x01) and the whole text the
#    same way, all 26 shards' books in it.
# 2. The rev indexes (followers), one after another; the open question is
#    their summed lookup, so they are measured first:
#    a. on one core: each shard alone, then all 26 summed per start (best of 2);
#    b. four processes at once, a quarter of the held-out text each, every
#       one with all 26 loaded;
#    c. the check: 300 contexts, the summed counts against a scan of the
#       whole text (the scans on 4 threads).
# 3. The fwd indexes (predecessors, for the gate's suffix features), then
#    their one-core pass and their check.
set -eu
cd "$(dirname "$0")"
mkdir -p results tmp shards
nice -n 19 python3 -I split_shards.py books.txt books.starts books.names groups_k25.books shards all_sep.txt \
    extra160.txt extra160.starts extra160.names > results/split.txt 2>&1

for t in shards/shard_*.txt; do
    s=$(basename "$t" .txt)
    nice -n 19 ./fm_shards build "$t" "shards/$s.rev.fm" tmp rev > "results/build_$s.rev.txt" 2>&1
done
ls shards/shard_*.rev.fm > rev.list
nice -n 19 ./fm_shards query rev.list heldout.txt 24 2 1 followers > results/query_rev.txt 2>&1
split -n 4 -d heldout.txt heldout.part
for p in 0 1 2 3; do
    nice -n 19 ./fm_shards query rev.list heldout.part0$p 24 1 0 followers > results/query4_rev_$p.txt 2>&1 &
done
wait
nice -n 19 ./fm_shards check rev.list all_sep.txt heldout.txt 300 24 4 followers > results/check_rev.txt 2>&1
date -u > results/DONE_rev

for t in shards/shard_*.txt; do
    s=$(basename "$t" .txt)
    nice -n 19 ./fm_shards build "$t" "shards/$s.fwd.fm" tmp fwd > "results/build_$s.fwd.txt" 2>&1
done
ls shards/shard_*.fwd.fm > fwd.list
nice -n 19 ./fm_shards query fwd.list heldout.txt 24 2 1 predecessors > results/query_fwd.txt 2>&1
nice -n 19 ./fm_shards check fwd.list all_sep.txt heldout.txt 300 24 4 predecessors > results/check_fwd.txt 2>&1
date -u > results/DONE
