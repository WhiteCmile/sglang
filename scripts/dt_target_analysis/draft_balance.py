"""Per-Draft load balance over time vs pickup stalls.
For 500 ms windows: rows proposed per Draft, distinct requests per Draft, exec occupancy, and pickup p90."""
import json, glob, gzip, sys, collections
def opener(f): return gzip.open(f, 'rt') if f.endswith('.gz') else open(f)
def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs)-1, int(q*len(xs)))] if xs else float('nan')
cell = sys.argv[1]; W = 5e8
coord, rows = {}, []
pull = []
for f in glob.glob(cell + "/profiles/sys3/*.jsonl*"):
    for l in opener(f):
        if '"coordinator_task_dispatch"' in l:
            r = json.loads(l)
            for t in r.get("accepted_transfer_ids") or r.get("transfer_ids") or []: coord.setdefault(t, r["ts_ns"])
        elif '"draft_feature_pull_submit"' in l:
            r = json.loads(l); pull.append((r["submitted_ns"], r["transfer_id"], r.get("draft_instance"), r["request_id"]))
        elif '"draft_flashmtp_proposal"' in l:
            r = json.loads(l); rows.append((r["execute_started_ns"], r["total_ns"], r["batch_size"], r.get("draft_instance")))
t0 = min(p[0] for p in pull)
win = collections.defaultdict(lambda: {"d0": [0, 0, set(), []], "d1": [0, 0, set(), []]})
for s, tot, bs, i in rows:
    k = "d0" if i.endswith("0") else "d1"; w = win[int((s - t0) / W)][k]; w[0] += bs; w[1] += tot
for s, t, i, req in pull:
    k = "d0" if i.endswith("0") else "d1"; w = win[int((s - t0) / W)][k]; w[2].add(req)
    if t in coord: w[3].append((s - coord[t]) / 1e6)
print("win(s) | d0 reqs rows occ pickup_p90 | d1 reqs rows occ pickup_p90")
for k in sorted(win):
    a, b = win[k]["d0"], win[k]["d1"]
    if a[0] + b[0] < 100: continue
    print("%5.1f | %3d %5d %4.0f%% %6.1f | %3d %5d %4.0f%% %6.1f" % (k * W / 1e9, len(a[2]), a[0], 100 * a[1] / W, pct(a[3], .9), len(b[2]), b[0], 100 * b[1] / W, pct(b[3], .9)))
