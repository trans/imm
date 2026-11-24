#!/bin/sh
# The 1 GB timing, run in the work directory on the second machine, where
# fm_time_rrr and fm_time_plain were compiled from fm_time.cpp (see README),
# beside books1g.txt and heldout.txt. Results go to results/.
#
# For each variant: the build, the brute-force check, the query pass (best of
# 2), then four query passes at once, a quarter of the held-out text each, to
# see what the four cores give together.
set -eu
cd "$(dirname "$0")"
mkdir -p results tmp
split -n 4 -d heldout.txt heldout.part
for v in rrr plain; do
    nice -n 19 ./fm_time_$v build books1g.txt books1g.$v.fm tmp > results/build_$v.txt 2>&1
    nice -n 19 ./fm_time_$v check books1g.$v.fm books1g.txt heldout.txt 300 24 > results/check_$v.txt 2>&1
    nice -n 19 ./fm_time_$v query books1g.$v.fm heldout.txt 24 2 > results/query_$v.txt 2>&1
    for p in 0 1 2 3; do
        nice -n 19 ./fm_time_$v query books1g.$v.fm heldout.part0$p 24 1 > results/query4_${v}_$p.txt 2>&1 &
    done
    wait
done
date -u > results/DONE
