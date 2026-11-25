"""Candidate observed -> freeze, per row via p0 hops, for the LAST row of each wave (the critical one).
Usage: cand_freeze.py <cell-dir>"""
import json, glob, gzip, sys, collections, statistics as st
def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs)-1, int(q*len(xs)))] if xs else float('nan')
cell = sys.argv[1]
f = sorted(glob.glob(cell + "/profiles/sys3/target-0*.jsonl*"))[0]
T = collections.defaultdict(dict); polls = []
for l in gzip.open(f, 'rt'):
    r = json.loads(l); e = r.get("event")
    if e == "target_p0_hop":
        h = r["hop"]; t = T[r["transfer_id"]]
        if h == "candidate_observed": t["obs"] = r["completion_observed_ns"]
        elif h == "ready_snapshot": t.update(snap=r["ready_snapshot_ns"], bs=r["tp_broadcast_start_ns"], be=r["tp_broadcast_end_ns"])
        elif h == "tp_ready_applied": t["apply"] = r["tp_apply_ns"]
        elif h == "frozen_for_verify": t.update(sel=r["selection_started_ns"], fz=r["freeze_ns"])
    elif e == "target_result_poll": polls.append(r)
waves = collections.defaultdict(list)
for tid, t in T.items():
    if "fz" in t and all(k in t for k in ("obs", "snap", "bs", "be", "apply", "sel")): waves[t["fz"]].append(t)
chain = [("obs", "snap", "candidate observed -> ready snapshot"), ("snap", "bs", "snapshot -> TP broadcast start"),
         ("bs", "be", "TP broadcast"), ("be", "apply", "broadcast end -> TP apply"), ("apply", "sel", "apply -> selection start"),
         ("sel", "fz", "selection start -> freeze"), ("obs", "fz", "TOTAL observed -> freeze")]
crit = [max(ts, key=lambda t: t["obs"]) for ts in waves.values()]
print("waves", len(crit), "(last-arriving row of each wave)")
for a, b, n in chain:
    v = [(t[b] - t[a]) / 1e6 for t in crit]
    print("  %-40s p50 %.3f mean %.3f p90 %.3f" % (n, pct(v, .5), st.mean(v), pct(v, .9)))
keys = [k for k in polls[0] if k.endswith("_ns") and k not in ("ts_ns", "started_ns", "finished_ns")]
print("target_result_poll mean ms:", {k: round(st.mean(p.get(k) or 0 for p in polls) / 1e6, 3) for k in keys})
