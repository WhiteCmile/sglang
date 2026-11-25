"""Summarize an interleaved A/B run: per-cell throughput/latency and per-arm mean, spread, delta vs first arm.

Usage: ab_summary.py <run-dir>
Uses the long-window benchmarks (long*.json); the discarded warm run is ignored.
"""
import json, glob, os, sys, statistics as st, collections
run = sys.argv[1]
cells = sorted(d for d in glob.glob(run + "/c[0-9]*") if os.path.isdir(d) and ".failed" not in d)
by = collections.defaultdict(list)
print("%-14s %8s %9s %9s" % ("cell", "tok/s", "mean_lat", "p95_e2e"))
for d in cells:
    arm = open(d + "/arm.txt").read().strip()
    for f in sorted(glob.glob(d + "/long*.json")):
        j = json.load(open(f))
        if j.get("successful_requests") != j.get("requests"): print("  incomplete", f); continue
        by[arm].append((j["output_tokens_per_s"], j["mean_latency_s"], j["p95_e2e_latency_s"]))
        print("%-14s %8.1f %9.4f %9.3f" % (os.path.basename(d), j["output_tokens_per_s"], j["mean_latency_s"], j["p95_e2e_latency_s"]))
arms = list(dict.fromkeys(open(d + "/arm.txt").read().strip() for d in cells))
print("\n%-8s %3s %9s %8s %9s  %s" % ("arm", "n", "tok/s", "spread", "mean_lat", "delta tok/s vs " + arms[0]))
base = st.mean(x[0] for x in by[arms[0]])
for a in arms:
    v = by[a]; t = [x[0] for x in v]
    print("%-8s %3d %9.1f %7.1f%% %9.4f  %+.1f%%" % (a, len(v), st.mean(t), 100 * (max(t) - min(t)) / st.mean(t), st.mean(x[1] for x in v), 100 * (st.mean(t) / base - 1)))
