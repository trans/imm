# Laptop limits for every job here: nice 19, ionice idle, 2 threads, no GPU.
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export TMPDIR=/home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad
cd /home/trans/.cache/tmp/claude/claude-1000/-home-trans-air-imm/df897c15-423f-4248-a6c9-51e21de981da/scratchpad/imm-tree || exit 1
exec nice -n 19 ionice -c 3 python3 -B $TMPDIR/tree/rss.py "$@"
