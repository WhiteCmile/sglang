"""Split each Target verify wave on the GPU (needs FlashMTP verify CUDA events, DT_GPU_CLOCK=1).

Usage: gpu_verify.py <cell-dir> [skip_s]
Per wave (host CLOCK_MONOTONIC via the GPU globaltimer stamp):
  launch->GPU start : submission_started -> verify start on GPU (host prepare + wait behind the other lane)
  GPU model/accept/finish : CUDA event split (target_verify_gpu_breakdown)
  GPU end -> host sees it : gpu_end_observed_ns - gpu_end_clock_ns
and the previous-wave end -> this start gap (<=0.05 ms means it queued directly behind the other lane).
"""
import json, glob, gzip, sys, collections, statistics as st

def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs)-1, int(q*len(xs)))] if xs else float('nan')
def opener(f): return gzip.open(f, 'rt') if f.endswith('.gz') else open(f)
def mean(xs): return st.mean(xs) if xs else float('nan')

cell = sys.argv[1]; skip = float(sys.argv[2]) * 1e9 if len(sys.argv) > 2 else 5e9
agg = collections.defaultdict(list); bybs = collections.defaultdict(lambda: collections.defaultdict(list))
n_queued = n_total = 0
for f in sorted(glob.glob(cell + "/profiles/*/target-*.jsonl*")):
    obs, brk = {}, {}
    for l in opener(f):
        try: r = json.loads(l)
        except Exception: continue
        e = r.get("event")
        if e == "target_verify_cuda_observed": obs[r["wave_id"]] = r
        elif e == "target_verify_gpu_breakdown": brk[r["wave_id"]] = r
    if not obs: continue
    t0 = min(o["submission_started_ns"] for o in obs.values() if o.get("submission_started_ns")) + skip
    for wid, o in obs.items():
        if not o.get("gpu_end_clock_ns") or o["submission_started_ns"] < t0: continue
        start = o["gpu_end_clock_ns"] - o["cuda_total_ns"]
        b = brk.get(wid)
        bs = b["batch_size"] if b else len(o.get("transfer_ids") or [])
        vals = {
            "launch -> GPU start": (start - o["submission_started_ns"]) / 1e6,
            "submit return -> GPU start": (start - o["submission_returned_ns"]) / 1e6,
            "GPU total (event0->3)": o["cuda_total_ns"] / 1e6,
            "GPU end -> host observes": (o["gpu_end_observed_ns"] - o["gpu_end_clock_ns"]) / 1e6,
        }
        if b:
            vals["  GPU model forward"] = b["model_ns"] / 1e6
            vals["  GPU accept"] = b["accept_ns"] / 1e6
            vals["  GPU finish"] = b["finish_ns"] / 1e6
        gap = o.get("cuda_previous_verify_end_to_start_ns")
        if gap is not None:
            n_total += 1; n_queued += gap <= 50_000
            vals["prev verify end -> this start"] = gap / 1e6
        for k, v in vals.items(): agg[k].append(v); bybs[min(bs, 8)][k].append(v)
print("waves: %d; started within 0.05 ms of the previous verify end (queued behind other lane): %.1f%%" % (n_total, 100 * n_queued / max(1, n_total)))
print("%-32s %7s %7s %7s %7s" % ("segment (ms)", "p50", "mean", "p95", "p99"))
for k, v in agg.items(): print("%-32s %7.2f %7.2f %7.2f %7.2f" % (k, pct(v, .5), mean(v), pct(v, .95), pct(v, .99)))
print("\nby wave rows: n | GPU total | model | launch->GPU start")
for bs in sorted(bybs):
    d = bybs[bs]
    print("  %s%d %6d | %6.2f | %6.2f | %6.2f" % (">=" if bs == 8 else "  ", bs, len(d["GPU total (event0->3)"]), mean(d["GPU total (event0->3)"]), mean(d["  GPU model forward"]), mean(d["launch -> GPU start"])))
