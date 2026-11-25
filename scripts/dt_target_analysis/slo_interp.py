"""Max load at TPOT p90 <= SLO (primary metric, 2026-11-24) from one run_ab run.

Usage: slo_interp.py <run-dir> [slo_ms=30]
Cells are c<NN>-<arm>; an arm name is <group><conc> (e.g. b384, o448). Per group and concurrency the
cells are averaged (TPOT p90 via latency_s / completed tokens, see tpot.py); the SLO crossing is
linearly interpolated between the two neighbouring concurrencies, for both concurrency and tok/s.
"""
import json, glob, sys, re, os, statistics as st, collections
def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs)-1, int(q*len(xs)))]
run = sys.argv[1]; slo = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
pts = collections.defaultdict(lambda: collections.defaultdict(list))
for c in sorted(glob.glob(run + "/c[0-9]*")):
    m = re.match(r"c\d+-([a-z]+)(\d+)$", os.path.basename(c))
    f = c + "/long1.json"
    if not m or not os.path.exists(f): continue
    d = json.load(open(f))
    t = [r["latency_s"] / r["completed_output_tokens"] * 1e3 for r in d["records"] if r.get("success") and r.get("completed_output_tokens")]
    pts[m.group(1)][int(m.group(2))].append((pct(t, .9), d["output_tokens_per_s"]))
for g, byc in sorted(pts.items()):
    rows = sorted((conc, st.mean(p for p, _ in v), st.mean(x for _, x in v), len(v)) for conc, v in byc.items())
    print("group %s" % g)
    for conc, p90, tps, n in rows: print("  conc %4d  n=%d  TPOT p90 %6.2f ms  tok/s %8.1f" % (conc, n, p90, tps))
    cross = None
    for (c0, p0, t0, _), (c1, p1, t1, _) in zip(rows, rows[1:]):
        if p0 <= slo < p1:
            w = (slo - p0) / (p1 - p0); cross = (c0 + w * (c1 - c0), t0 + w * (t1 - t0)); break
    if cross: print("  -> max load at TPOT p90 <= %.0f ms: conc %.0f, %.0f tok/s" % (slo, *cross))
    elif rows and rows[-1][1] <= slo: print("  -> SLO not reached up to conc %d (>= %.0f tok/s)" % (rows[-1][0], rows[-1][2]))
    else: print("  -> SLO already violated at the lowest concurrency")
