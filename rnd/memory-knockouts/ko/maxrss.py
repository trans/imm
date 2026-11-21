"""Runs argv[1:] and prints the child's peak RSS (maxrss_kb=...) to stderr."""
import resource, subprocess, sys
rc = subprocess.call(sys.argv[1:])
print(f"maxrss_kb={resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}", file=sys.stderr)
sys.exit(rc)
