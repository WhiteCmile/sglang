"""Per-wave Target host critical path: freeze -> GPU start, split by the existing profile fields.

Usage: hostpath.py <cell-dir> [skip_s]
Joins target_ready_batch / target_batch_submit / target_verify_cuda_observed on wave_id
(sys-3 Targets; GPU start = gpu_end_clock_ns - cuda_total_ns, DT_GPU_CLOCK=1).
"""
import json, glob, gzip, sys, collections, statistics as st
def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs)-1, int(q*len(xs)))] if xs else float('nan')
def opener(f): return gzip.open(f, 'rt') if f.endswith('.gz') else open(f)
cell = sys.argv[1]; skip = float(sys.argv[2]) * 1e9 if len(sys.argv) > 2 else 10e9
agg = collections.defaultdict(list); order = []
def add(k, v):
    if k not in agg: order.append(k)
    agg[k].append(v / 1e6)
for f in sorted(glob.glob(cell + "/profiles/sys3/target-*.jsonl*")):
    rb, bs, cu, wk = {}, {}, {}, {}
    for l in opener(f):
        try: r = json.loads(l)
        except Exception: continue
        e = r.get("event")
        if e == "target_ready_batch": rb[r["wave_id"]] = r
        elif e == "target_batch_submit": bs[r["wave_id"]] = r
        elif e == "target_verify_cuda_observed": cu[r["wave_id"]] = r
        elif e == "target_hostpath_worker": wk[r["wave_id"]] = r["marks"]
    if not rb: continue
    t0 = min(x["freeze_ns"] for x in rb.values()) + skip
    for k, r in rb.items():
        if r["freeze_ns"] < t0 or k not in bs or k not in cu or not cu[k].get("gpu_end_clock_ns"): continue
        b, c = bs[k], cu[k]
        gstart = c["gpu_end_clock_ns"] - c["cuda_total_ns"]
        add("freeze -> build start", r["build_started_ns"] - r["freeze_ns"])
        for fld in ("initialize_ns", "req_pool_cpu_ns", "req_pool_device_ns", "seq_lens_cpu_ns", "seq_lens_device_ns",
                    "seq_lens_derived_ns", "bonus_tokens_ns", "candidate_cpu_ns", "candidate_device_ns", "relay_ns",
                    "sampling_ns", "spec_tensors_ns", "tensors_ns"):
            if fld in r: add("  build." + fld[:-3], r[fld])
        add("build total (build start -> finished)", r["build_finished_ns"] - r["build_started_ns"])
        add("build finished -> selection finished", b["selection_finished_ns"] - r["build_finished_ns"])
        add("  of which update_running", b.get("update_running_ns", 0))
        hm = b.get("hostpath_marks")
        if hm:
            prev_name, prev = "build_finished", r["build_finished_ns"]
            for name, ns in hm + [["selection_finished", b["selection_finished_ns"]]]:
                add("    %s -> %s" % (prev_name, name), ns - prev); prev_name, prev = name, ns
        add("selection finished -> submission start", b["submission_started_ns"] - b["selection_finished_ns"])
        add("submission start -> return (host launch)", b["submission_returned_ns"] - b["submission_started_ns"])
        add("submission start -> GPU start", gstart - b["submission_started_ns"])
        wm = wk.get(k)
        if wm:
            prev_name, prev = "submission_started", b["submission_started_ns"]
            for name, ns in sorted(wm, key=lambda m: m[1]) + [["GPU start", gstart]]:
                add("    %s -> %s" % (prev_name, name), ns - prev); prev_name, prev = name, ns
        add("submission return -> GPU start", gstart - b["submission_returned_ns"])
        add("TOTAL freeze -> GPU start", gstart - r["freeze_ns"])
n = len(agg.get("TOTAL freeze -> GPU start", []))
print("waves %d" % n)
print("%-44s %7s %7s %7s" % ("segment (ms)", "p50", "mean", "p90"))
for k in order: print("%-44s %7.3f %7.3f %7.3f" % (k, pct(agg[k], .5), st.mean(agg[k]), pct(agg[k], .9)))
