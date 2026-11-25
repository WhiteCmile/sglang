"""What was the TP0 scheduler doing when each verify finished on the GPU (and candidates arrived)?
Attributes the GPU end -> host observes lag (and candidate observed -> snapshot) to the traced activity
interval covering the GPU end instant. Usage: observe_lag.py <cell-dir> [skip_s]"""
import json, glob, gzip, sys, bisect, collections, statistics as st
cell = sys.argv[1]; skip = float(sys.argv[2]) * 1e9 if len(sys.argv) > 2 else 10e9
f = sorted(glob.glob(cell + "/profiles/sys3/target-0*.jsonl*"))[0]
recs = [json.loads(l) for l in gzip.open(f, 'rt')]
lo = min(r["ts_ns"] for r in recs) + skip
iv = []  # (start, end, label)
for r in recs:
    e = r.get("event")
    if e == "target_old_result_process": iv.append((r["started_ns"], r["finished_ns"], "result process"))
    elif e == "target_batch_submit":
        iv.append((r["selection_started_ns"], r["selection_finished_ns"], "selection+build+update_running"))
        iv.append((r["submission_started_ns"], r["submission_returned_ns"], "submission (launch verify)"))
    elif e == "target_result_poll": iv.append((r["started_ns"], r["finished_ns"], "poll"))
iv.sort(); starts = [s for s, _, _ in iv]
def where(t):
    i = bisect.bisect_right(starts, t) - 1
    best = None
    for j in range(max(0, i - 3), i + 1):
        s, e, lab = iv[j]
        if s <= t < e: best = (lab, e - t)
    return best or ("untraced (idle loop / other work)", 0)
for title, pairs in (("GPU end -> host observes", [(r["gpu_end_clock_ns"], r["gpu_end_observed_ns"]) for r in recs if r.get("event") == "target_verify_cuda_observed" and r.get("gpu_end_clock_ns") and r["gpu_end_clock_ns"] > lo]),
                     ("candidate completion -> observed", [(r["producer_globaltimer_ns"], r["completion_observed_ns"]) for r in recs if r.get("event") == "target_p0_hop" and r.get("hop") == "candidate_observed" and r.get("producer_globaltimer_ns") and r["completion_observed_ns"] > lo])):
    agg = collections.defaultdict(list)
    for t, obs in pairs:
        lab, rem = where(t); agg[lab].append((obs - t) / 1e6)
    n = sum(len(v) for v in agg.values())
    print("== %s (n=%d, mean %.2f ms)" % (title, n, st.mean(x for v in agg.values() for x in v) if n else 0))
    for lab, v in sorted(agg.items(), key=lambda kv: -sum(kv[1])):
        print("  %-36s %5.1f%% of events, mean lag %.2f ms, share of total lag %5.1f%%" % (lab, 100 * len(v) / n, st.mean(v), 100 * sum(v) / sum(x for vv in agg.values() for x in vv)))
