"""Per-request round-cycle decomposition and wave-size distribution from target_p0_hop traces."""
import json, glob, sys, collections, statistics as st

def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs)-1, int(q*len(xs)))] if xs else float('nan')

root = sys.argv[1]
by_tid = collections.defaultdict(dict)
waves = collections.defaultdict(dict)   # (target, wave) -> batch_size
for f in glob.glob(root + "/profiles/*/*.jsonl"):
    tgt = f.split("/")[-1].rsplit("-", 1)[0]
    for l in open(f):
        r = json.loads(l)
        if r.get("event") != "target_p0_hop": continue
        by_tid[r["transfer_id"]][r["hop"]] = r
        if r["hop"] == "frozen_for_verify":
            waves[tgt][r["wave_id"]] = r["batch_size"]

# wave sizes
allw = [b for t in waves.values() for b in t.values()]
print("waves:", len(allw), "rows/wave mean %.2f median %d p10 %d p90 %d max %d" % (st.mean(allw), pct(allw,.5), pct(allw,.1), pct(allw,.9), max(allw)))
hist = collections.Counter(allw); print("  size histogram:", sorted(hist.items()))
for t in sorted(waves): w=list(waves[t].values()); print("  %s waves=%d mean=%.2f" % (t, len(w), st.mean(w)))

# per transfer segments (one transfer == one request round)
seg = collections.defaultdict(list)
rounds = collections.defaultdict(list)  # request -> [(round, hidden_ns, submit_returned_ns)]
for tid, h in by_tid.items():
    need = ("feature_published","dispatch_registered","candidate_observed","frozen_for_verify","verify_submitted")
    if not all(k in h for k in need): continue
    fp, dr, co, fr, vs = (h[k] for k in need)
    t0 = fp["hidden_produced_ns"]
    pts = [("hidden->descriptor", fp["descriptor_published_ns"]),
           ("descriptor->coord_submit", dr["submitted_ns"]),
           ("coord_submit->candidate(draft)", co["completion_observed_ns"]),
           ("candidate->freeze", fr["freeze_ns"]),
           ("freeze->build_done", fr["build_finished_ns"]),
           ("build_done->submit_start", vs["submission_started_ns"]),
           ("submit_start->submit_return", vs["submission_returned_ns"])]
    prev = t0
    for name, t in pts:
        seg[name].append((t - prev)/1e6); prev = t
    rounds[fp["request_id"]].append((fp["round_id"], t0, vs["submission_returned_ns"]))

# round cycle: hidden(r) -> hidden(r+1); remainder after submit_return = verify GPU + result until next hidden
cycle, tail = [], []
for req, rs in rounds.items():
    rs.sort()
    for a, b in zip(rs, rs[1:]):
        if b[0] == a[0] + 1:
            cycle.append((b[1]-a[1])/1e6); tail.append((b[1]-a[2])/1e6)
seg["submit_return->next hidden (verify GPU+result)"] = tail
print("\nper-round segments (ms): name | median | mean | p90 | share of mean cycle")
mc = st.mean(cycle)
for k, v in seg.items():
    print("  %-48s %7.2f %7.2f %7.2f  %5.1f%%" % (k, pct(v,.5), st.mean(v), pct(v,.9), 100*st.mean(v)/mc))
print("  %-48s %7.2f %7.2f %7.2f" % ("ROUND CYCLE hidden->next hidden", pct(cycle,.5), mc, pct(cycle,.9)))
print("rounds:", len(cycle))
