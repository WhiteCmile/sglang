"""Where do Draft pickup stalls come from? coordinator dispatch -> Draft pull submit > THRESH ms, by Draft and time."""
import json, glob, gzip, sys, collections
def opener(f): return gzip.open(f, 'rt') if f.endswith('.gz') else open(f)
cell = sys.argv[1]; TH = float(sys.argv[2]) if len(sys.argv) > 2 else 10
coord, pull, inst, props = {}, {}, {}, collections.defaultdict(list)
for f in glob.glob(cell + "/profiles/sys3/*.jsonl*"):
    for l in opener(f):
        if '"coordinator_task_dispatch"' in l:
            r = json.loads(l)
            for t in r.get("accepted_transfer_ids") or r.get("transfer_ids") or []: coord.setdefault(t, r["ts_ns"])
        elif '"draft_feature_pull_submit"' in l:
            r = json.loads(l); pull[r["transfer_id"]] = r["submitted_ns"]; inst[r["transfer_id"]] = r.get("draft_instance") or r.get("component")
        elif '"draft_flashmtp_proposal"' in l:
            r = json.loads(l); props[r.get("draft_instance") or r.get("component")].append((r["execute_started_ns"], r["total_ns"], r["batch_size"]))
lat = [(coord[t], (pull[t] - coord[t]) / 1e6, inst[t]) for t in pull if t in coord]
lat.sort(); t0 = lat[0][0]
slow = [x for x in lat if x[1] > TH]
print("transfers %d, pickup > %.0f ms: %d (%.2f%%)" % (len(lat), TH, len(slow), 100 * len(slow) / len(lat)))
print("by Draft:", dict(collections.Counter(x[2] for x in slow)))
ep = collections.defaultdict(lambda: collections.Counter())
for t, v, i in slow: ep[int((t - t0) / 1e8)][i] += 1
print("100 ms windows with stalls (window: per-draft count):")
print("  " + "  ".join("%.1fs:%s" % (k / 10, ",".join("%s=%d" % (i[-1], n) for i, n in sorted(c.items()))) for k, c in sorted(ep.items())[:60]))
# gaps between consecutive proposals per Draft: long gaps = Draft loop not running proposals
for i, ps in sorted(props.items()):
    ps.sort(); gaps = [((b[0] - (a[0] + a[1])) / 1e6, (a[0] - t0) / 1e9) for a, b in zip(ps, ps[1:])]
    big = sorted(gaps, reverse=True)[:8]
    print("%s largest gaps between proposals (ms @ s): %s" % (i, ", ".join("%.1f@%.1f" % g for g in big)))
