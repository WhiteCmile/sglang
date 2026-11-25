"""Where does the Target scheduler thread (TP0) spend wall time? Per-second budget from traced events.
Usage: loop_budget.py <cell-dir> [skip_s]"""
import json, glob, gzip, sys, collections, statistics as st
cell = sys.argv[1]; skip = float(sys.argv[2]) * 1e9 if len(sys.argv) > 2 else 10e9
for f in sorted(glob.glob(cell + "/profiles/sys3/target-0*.jsonl*")):
    tot = collections.Counter(); cnt = collections.Counter(); t_lo = t_hi = None; first = None
    recs = []
    for l in gzip.open(f, 'rt'):
        r = json.loads(l); recs.append(r)
        first = r["ts_ns"] if first is None else min(first, r["ts_ns"])
    lo = first + skip
    for r in recs:
        e = r.get("event"); ts = r["ts_ns"]
        if ts < lo: continue
        t_lo = ts if t_lo is None else min(t_lo, ts); t_hi = ts if t_hi is None else max(t_hi, ts)
        if e == "target_old_result_process": tot["result process (old_result_process)"] += r["total_ns"]; cnt["result process (old_result_process)"] += 1
        elif e == "target_ready_batch": tot["build_ready_batch"] += r["total_ns"]; cnt["build_ready_batch"] += 1
        elif e == "target_batch_submit":
            tot["selection (get_next_batch_to_run)"] += r["selection_finished_ns"] - r["selection_started_ns"]
            tot["submission (run_batch launch)"] += r["submission_wall_ns"]; cnt["submission (run_batch launch)"] += 1
            tot["update_running"] += r.get("update_running_ns", 0)
        elif e == "target_iteration_tail":
            tot["iteration tail (sample/detach)"] += r.get("sample_ns", 0) + r.get("detach_ns", 0)
    wall = t_hi - t_lo
    print(f.split("/")[-1], "window %.1f s" % (wall / 1e9))
    for k, v in tot.most_common():
        print("  %-40s %5.1f%% of wall  (%d events, %.3f ms each)" % (k, 100 * v / wall, cnt[k] or 0, v / 1e6 / max(1, cnt[k])))
    polls = [r for r in recs if r.get("event") == "target_result_poll" and r["ts_ns"] >= lo]
    print("  traced polls %d (sampled: only polls with events or slow ones)" % len(polls))
