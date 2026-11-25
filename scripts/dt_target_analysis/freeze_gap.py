"""Per wave: was it ready-held, and how far is freeze from its last candidate arrival.
Usage: freeze_gap.py <cell-dir>"""
import json, glob, gzip, sys, collections, statistics as st
def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs)-1, int(q*len(xs)))] if xs else float('nan')
cell = sys.argv[1]
for f in sorted(glob.glob(cell + "/profiles/sys3/target-*.jsonl*")):
    rows = collections.defaultdict(list); holds = []; reasons = collections.Counter(); lf = collections.Counter()
    rb = {}
    for l in gzip.open(f, 'rt'):
        r = json.loads(l); e = r.get("event")
        if e == "target_p0_hop" and r.get("hop") == "frozen_for_verify":
            rows[r["freeze_ns"]].append(r["candidate_observed_ns"])
        elif e == "target_ready_hold" and r.get("phase") == "release":
            holds.append(r); reasons[r["reason"]] += 1
        elif e == "target_late_freeze_release": lf[r["release_reason"]] += 1
        elif e == "target_ready_batch": rb[r["freeze_ns"]] = r["batch_size"]
    gaps_last = [(fz - max(c)) / 1e6 for fz, c in rows.items()]
    gaps_first = [(fz - min(c)) / 1e6 for fz, c in rows.items()]
    print(f.split('/')[-1], "waves", len(rows), "ready-hold releases", len(holds), dict(reasons), "late-freeze", dict(lf))
    print("  freeze - last candidate ms p10/p50/p90 %.3f %.3f %.3f | freeze - first candidate p50 %.3f mean %.3f" % (
        pct(gaps_last, .1), pct(gaps_last, .5), pct(gaps_last, .9), pct(gaps_first, .5), st.mean(gaps_first)))
    near = sum(1 for g in gaps_last if g < 0.3)
    print("  waves frozen <0.3 ms after their last candidate: %.0f%%; rows/wave mean %.1f" % (100 * near / len(rows), st.mean(rb.values())))
    break
