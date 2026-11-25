import json, glob, sys, collections, statistics as st
def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs)-1, int(q*len(xs)))] if xs else float('nan')
root, skip_s = sys.argv[1], float(sys.argv[2])
by = collections.defaultdict(dict); host = {}
for f in glob.glob(root + "/profiles/*/*.jsonl"):
    h = f.split("/")[-2]
    for l in open(f):
        r = json.loads(l)
        if r.get("event") == "target_p0_hop":
            by[r["transfer_id"]][r["hop"]] = r; host[r["transfer_id"]] = h
rows = collections.defaultdict(list)
for tid, h in by.items():
    if all(k in h for k in ("feature_published","dispatch_registered","candidate_observed","frozen_for_verify","verify_submitted")):
        rows[host[tid]].append(h)
for hst, hs in sorted(rows.items()):
    t0 = min(h["dispatch_registered"]["submitted_ns"] for h in hs); t1 = max(h["dispatch_registered"]["submitted_ns"] for h in hs)
    keep = [h for h in hs if h["dispatch_registered"]["submitted_ns"] - t0 > skip_s*1e9]
    print("%s span %.1fs, kept %d/%d rounds after skipping first %.1fs" % (hst, (t1-t0)/1e9, len(keep), len(hs), skip_s))
    segs = {
     "launch->coord_submit (verify+result+park)": [(h["dispatch_registered"]["submitted_ns"]-h["feature_published"]["hidden_produced_ns"])/1e6 for h in keep],
     "draft roundtrip": [(h["candidate_observed"]["completion_observed_ns"]-h["dispatch_registered"]["submitted_ns"])/1e6 for h in keep],
     "candidate->freeze (incl. hold)": [(h["frozen_for_verify"]["freeze_ns"]-h["candidate_observed"]["completion_observed_ns"])/1e6 for h in keep],
     "freeze->submit_return": [(h["verify_submitted"]["submission_returned_ns"]-h["frozen_for_verify"]["freeze_ns"])/1e6 for h in keep],
    }
    tot = sum(st.mean(v) for v in segs.values())
    for k, v in segs.items():
        print("   %-42s median %6.2f mean %6.2f p95 %7.2f p99 %7.2f  share %4.1f%%" % (k, pct(v,.5), st.mean(v), pct(v,.95), pct(v,.99), 100*st.mean(v)/tot))
    dr = segs["draft roundtrip"]
    print("   draft >20ms: %.1f%% of rounds, %.1f%% of draft time; mean if capped at 20ms: %.2f" % (100*sum(x>20 for x in dr)/len(dr), 100*sum(x for x in dr if x>20)/sum(dr), st.mean(min(x,20) for x in dr)))
    print("   sum of segment means (≈ round cycle): %.2f ms" % tot)
