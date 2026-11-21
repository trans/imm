"""Run a command and print its peak RSS (the largest child's) to stderr."""
import resource, subprocess, sys
r = subprocess.run(sys.argv[1:])
print(f"peak_rss_mb={resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024:.0f} exit={r.returncode}", file=sys.stderr)
sys.exit(r.returncode)
