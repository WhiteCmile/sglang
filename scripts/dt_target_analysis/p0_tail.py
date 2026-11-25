import json, glob, sys, collections, statistics as st
def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs)-1, int(q*len(xs)))] if xs else float('nan')
root = sys.argv[1]
by = collections.defaultdict(dict); host = {}
for f in glob.glob(root + "/profiles/*/*.jsonl"):
    h = f.split("/")[-2]
    for l in open(f):
        r = json.loads(l)
        if r.get("event") == "target_p0_hop":
            by[r["transfer_id"]][r["hop"]] = r; host[r["transfer_id"]] = h
d = collections.defaultdict(list); cf = collections.defaultdict(list); dd = collections.defaultdict(list); allv = []
for tid, h in by.items():
    if "dispatch_registered" in h and "candidate_observed" in h and "frozen_for_verify" in h and "feature_published" in h:
        x = (h["candidate_observed"]["completion_observed_ns"] - h["dispatch_registered"]["submitted_ns"]) / 1e6
        d[host[tid]].append(x); allv.append((h["dispatch_registered"]["submitted_ns"], x))
        cf[host[tid]].append((h["frozen_for_verify"]["freeze_ns"] - h["candidate_observed"]["completion_observed_ns"]) / 1e6)
        dd[host[tid]].append((h["dispatch_registered"]["submitted_ns"] - h["feature_published"]["descriptor_published_ns"]) / 1e6)
for name, m in (("draft roundtrip coord_submit->candidate", d), ("candidate->freeze", cf), ("descriptor->coord_submit", dd)):
    print(name)
    for h, v in sorted(m.items()):
        print("  %s n=%d median %.2f mean %.2f p90 %.2f p95 %.2f p99 %.2f  >20ms %.1f%%  share of sum from >20ms %.1f%%" % (
            h, len(v), pct(v,.5), st.mean(v), pct(v,.9), pct(v,.95), pct(v,.99), 100*sum(x>20 for x in v)/len(v), 100*sum(x for x in v if x>20)/sum(v)))
v = [x for _, x in allv]
print("draft roundtrip histogram (ms buckets):", sorted(collections.Counter(min(int(x//4)*4, 100) for x in v).items()))
# time structure: is the tail clustered in time?
allv.sort(); t0 = allv[0][0]
buckets = collections.defaultdict(list)
for t, x in allv: buckets[int((t - t0)/5e8)].append(x)
print("per 0.5s window: median / p90 of draft roundtrip")
print(" ", " ".join("%d:%.0f/%.0f" % (k, pct(b,.5), pct(b,.9)) for k, b in sorted(buckets.items())))
