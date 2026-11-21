#!/bin/sh
# Paired buckets of two held-out per-character loss files (A against B: A's
# change), on the d24 prior's loss and features: compare_tx.py (place in the
# word, what is predicted, the prior's loss, entropy, depth) and
# topic_buckets.py (place in the piece, new or repeated word and distance
# back, the prior's misses). In the output, "imm" is A and "tx" is B.
# Usage: scripts/rnd/rpair.sh A.npz B.npz OUT.txt
# The buckets read mobius's d24 prior rows on the held-out text: PRIOR_ROWS
# (default below) and topic_buckets.py's path.
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
: "${PRIOR_ROWS:=/home/trans/Projects/air/mobius/runs/imm/pg19split20m_gated_d24_pg10g}"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PRIOR_ROWS
{
  echo "A = $1"; echo "B = $2"; echo
  nice -n 19 python3 "$here/compare_tx.py" "$repo" "$1" "$2"
  echo
  nice -n 19 python3 "$here/topic_buckets.py" "$1" "$2"
} > "$3" 2>&1
echo "rpair exit=$? -> $3"
