#!/bin/sh
# mobius's three extras (mail 20261124-190109Z), run after run.sh, in the same
# work directory, with fm_time.cpp as it is now (the per-depth pass added),
# compiled as fm2_rrr and fm2_plain:
#   1. builds on the first 250 MB and 500 MB of books1g.txt, with the same
#      query passes, for the slope of the cost with the text's size;
#   2. the per-depth cost of the histograms (fm2's query prints it), on
#      every index, the 1 GB ones included;
#   3. two query processes at once on the 1 GB index, half the held-out
#      text each.
set -eu
cd "$(dirname "$0")"
while [ ! -f results/DONE ]; do sleep 60; done
for v in rrr plain; do
    f=""
    [ $v = plain ] && f=-DFM_PLAIN
    nice -n 19 g++ -O3 -march=native -std=c++20 -DNDEBUG $f -I sdsl-lite/include fm2.cpp -o fm2_$v
done
for size in 250 500; do
    head -c ${size}000000 books1g.txt | python3 -I -c "import sys; d=sys.stdin.buffer.read(); sys.stdout.buffer.write(d[:d.rindex(b'\n')+1])" > books${size}m.txt
done
split -n 2 -d heldout.txt heldout.half
for v in rrr plain; do
    for size in 250m 500m; do
        nice -n 19 ./fm2_$v build books$size.txt books$size.$v.fm tmp > results/build_${v}_$size.txt 2>&1
        nice -n 19 ./fm2_$v query books$size.$v.fm heldout.txt 24 2 > results/query_${v}_$size.txt 2>&1
    done
    nice -n 19 ./fm2_$v query books1g.$v.fm heldout.txt 24 2 > results/query_${v}_1g_bydepth.txt 2>&1
    for p in 0 1; do
        nice -n 19 ./fm2_$v query books1g.$v.fm heldout.half0$p 24 1 > results/query2_${v}_$p.txt 2>&1 &
    done
    wait
done
date -u > results/DONE2
