"""TPOT p50/p90/p99 per cell (proxy: request latency_s / completed tokens; the server returns no tpot_s).

Usage: tpot.py <run-dir-or-cell> [...]
Prompts are ~16 tokens, so latency/tokens over-states true TPOT slightly (conservative for an SLO).
"""
import json, glob, sys, os
def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs)-1, int(q*len(xs)))] if xs else float('nan')
for arg in sys.argv[1:]:
    cells = [arg] if os.path.exists(arg + "/long1.json") else sorted(glob.glob(arg + "/c[0-9]*"))
    for c in cells:
        for f in sorted(glob.glob(c + "/long*.json")):
            d = json.load(open(f))
            t = [r["latency_s"] / r["completed_output_tokens"] * 1e3 for r in d["records"] if r.get("success") and r.get("completed_output_tokens")]
            print("%-60s conc %4s tok/s %8.1f  TPOT ms p50 %6.2f p90 %6.2f p99 %6.2f" % (
                c.split("results/")[-1] + "/" + os.path.basename(f), d.get("global_concurrency"), d["output_tokens_per_s"], pct(t, .5), pct(t, .9), pct(t, .99)))
