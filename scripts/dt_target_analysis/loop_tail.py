"""Launch-iteration breakdown on TP0 (sys-3 target-0): run_batch tail, FlashMTP park phases, result processing.
Usage: loop_tail.py <cell-dir> [skip_s]
"""
import collections
import glob
import gzip
import json
import statistics as st
import sys

cell = sys.argv[1]
skip = float(sys.argv[2]) * 1e9 if len(sys.argv) > 2 else 10e9


def q(x, p):
    x = sorted(x)
    return x[min(len(x) - 1, int(p * len(x)))] if x else float("nan")


def show(name, xs, unit=1e6):
    if xs:
        print("  %-44s n %6d p50 %6.3f mean %6.3f p90 %6.3f" % (name, len(xs), q(xs, .5) / unit, st.mean(xs) / unit, q(xs, .9) / unit))


ev = collections.defaultdict(list)
for f in sorted(glob.glob(cell + "/profiles/sys3/target-0-*.jsonl.gz")):
    for line in gzip.open(f, "rt"):
        r = json.loads(line)
        ev[r["event"]].append(r)
t0 = min(r["ts_ns"] for v in ev.values() for r in v) + skip
for k in ev:
    ev[k] = [r for r in ev[k] if r["ts_ns"] >= t0]

print("== run_batch tail (after the FlashMTP worker returns)")
seg = collections.defaultdict(list)
for r in ev["target_run_batch_tail"]:
    m = r["marks"]
    for (a, ta), (b, tb) in zip(m, m[1:]):
        seg[f"{a} -> {b}"].append(tb - ta)
    seg["TOTAL worker_returned -> return"].append(m[-1][1] - m[0][1])
for k, v in seg.items():
    show(k, v)

print("== FlashMTP park (target_park_batch)")
P = ev["target_park_batch"]
show("park total", [r["total_ns"] for r in P])
for k in ("flashmtp_release_inflight_ns", "flashmtp_retire_ns", "flashmtp_select_ns", "flashmtp_scalar_copy_ns",
          "flashmtp_task_build_ns", "flashmtp_queue_flush_ns", "flashmtp_finalize_ns"):
    show(k, [r.get(k, 0) for r in P])
if P:
    print("  batch size mean %.1f" % st.mean(r["batch_size"] for r in P))

print("== previous-result processing (target_old_result_process) vs park")
proc = {r["wave_id"]: r for r in ev["target_old_result_process"]}
park = {r["wave_id"]: r for r in P}
show("process_batch_result total", [r["total_ns"] for r in proc.values()])
show("  of which outside park", [proc[w]["total_ns"] - park[w]["total_ns"] for w in proc if w in park])

print("== launch iterations (target_loop_iteration, has_batch)")
L = [r for r in ev["target_loop_iteration"] if r["has_batch"]]
for k in ("total_ns", "recv_ns", "poll_ns", "select_ns", "launch_tail_ns"):
    show(k, [r[k] for r in L])
print("== submission (target_batch_submit)")
S = ev["target_batch_submit"]
show("submission wall (run_batch)", [r["submission_returned_ns"] - r["submission_started_ns"] for r in S])
show("update_running", [r["update_running_ns"] for r in S])
show("selection (selection_started -> finished)", [r["selection_finished_ns"] - r["selection_started_ns"] for r in S])
R = ev["target_ready_batch"]
show("build_ready_batch total", [r["total_ns"] for r in R])
T = ev["target_iteration_tail"]
show("sample", [r["sample_ns"] for r in T])
show("detach", [r["detach_ns"] for r in T])

print("== FlashMTP pivot batch publish (target_flashmtp_pivot_batch_publish)")
B = ev["target_flashmtp_pivot_batch_publish"]
for k in ("selection_ns", "gather_submission_ns", "pool_publish_ns", "reverse_edge_ns", "ownership_ns",
          "descriptor_profile_ns", "scalar_stage_submission_ns", "total_ns"):
    show(k, [r.get(k, 0) for r in B])
