"""Local metrics per cell: result-consume lag, per-round segments (pooled over all Targets), idle split."""
import json, glob, gzip, sys, collections, statistics as st
def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs)-1, int(q*len(xs)))] if xs else float('nan')
def opener(f): return gzip.open(f, 'rt') if f.endswith('.gz') else open(f)
skip = 5e9
print("%-10s | copy_ready->process ms p50/mean | submit_ret->next coord_submit ms p50/mean | draft rt mean | cand->freeze mean | freeze->submit_ret mean | round sum" % "cell")
for cell in sys.argv[1:]:
    lag, post, dr, cf, fs = [], [], [], [], []
    for f in glob.glob(cell + "/profiles/*/*.jsonl*"):
        hops = collections.defaultdict(dict); t0 = None; copies = []
        for l in opener(f):
            try: r = json.loads(l)
            except Exception: continue
            e = r.get("event")
            if t0 is None and "ts_ns" in r: t0 = r["ts_ns"] + skip
            if e == "target_old_result_copy_ready" and r.get("copy_done_ready") and r["copy_done_observed_ns"] > t0:
                lag.append((r["process_started_ns"] - r["copy_done_observed_ns"]) / 1e6)
            elif e == "target_p0_hop" and r["ts_ns"] > t0:
                hops[r["transfer_id"]][r["hop"]] = r
        byreq = collections.defaultdict(dict)
        for h in hops.values():
            if "feature_published" in h: byreq[h["feature_published"]["request_id"]][h["feature_published"]["round_id"]] = h
        for rounds in byreq.values():
            for rid, h in rounds.items():
                n = rounds.get(rid + 1)
                if n and "verify_submitted" in h and "dispatch_registered" in n:
                    post.append((n["dispatch_registered"]["submitted_ns"] - h["verify_submitted"]["submission_returned_ns"]) / 1e6)
                if all(k in h for k in ("dispatch_registered", "candidate_observed", "frozen_for_verify", "verify_submitted")):
                    dr.append((h["candidate_observed"]["completion_observed_ns"] - h["dispatch_registered"]["submitted_ns"]) / 1e6)
                    cf.append((h["frozen_for_verify"]["freeze_ns"] - h["candidate_observed"]["completion_observed_ns"]) / 1e6)
                    fs.append((h["verify_submitted"]["submission_returned_ns"] - h["frozen_for_verify"]["freeze_ns"]) / 1e6)
    tot = st.mean(post) + st.mean(dr) + st.mean(cf) + st.mean(fs)
    print("%-10s | %13.3f / %6.3f | %20.2f / %6.2f | %12.2f | %16.2f | %22.2f | %8.2f" % (
        cell.rstrip("/").split("/")[-1], pct(lag, .5), st.mean(lag), pct(post, .5), st.mean(post), st.mean(dr), st.mean(cf), st.mean(fs), tot))
