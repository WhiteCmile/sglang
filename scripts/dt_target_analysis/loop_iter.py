"""Scheduler loop composition from target_loop_iteration (TP0). Usage: loop_iter.py <cell-dir> [skip_s]
rest = total - recv - poll - select - launch_tail (result processing of the previous wave, defer turns, ...)."""
import json, glob, gzip, sys, collections, statistics as st
def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs)-1, int(q*len(xs)))] if xs else float('nan')
cell = sys.argv[1]; skip = float(sys.argv[2]) * 1e9 if len(sys.argv) > 2 else 10e9
f = sorted(glob.glob(cell + "/profiles/sys3/target-0*.jsonl*"))[0]
its = [json.loads(l) for l in gzip.open(f, 'rt')]
its = [r for r in its if r.get("event") == "target_loop_iteration"]
lo = its[0]["started_ns"] + skip; its = [r for r in its if r["started_ns"] >= lo]
wall = its[-1]["finish_ns"] - its[0]["started_ns"]
parts = ("recv_ns", "poll_ns", "select_ns", "launch_tail_ns")
for kind, sel in (("all", lambda r: True), ("launch", lambda r: r["has_batch"]), ("no batch, held", lambda r: not r["has_batch"] and r["held"]),
                  ("no batch, deferred", lambda r: not r["has_batch"] and r["deferred"]), ("no batch, other", lambda r: not r["has_batch"] and not r["held"] and not r["deferred"])):
    v = [r for r in its if sel(r)]
    if not v: continue
    tot = sum(r["total_ns"] for r in v)
    rest = [r["total_ns"] - sum(r[p] for p in parts) for r in v]
    print("%-20s n=%6d  %5.1f%% of wall  per-iter mean %.3f ms p99 %.3f | %s | rest %.3f" % (
        kind, len(v), 100 * tot / wall, st.mean(r["total_ns"] for r in v) / 1e6, pct([r["total_ns"] for r in v], .99) / 1e6,
        " ".join("%s %.3f" % (p[:-3], st.mean(r[p] for r in v) / 1e6) for p in parts), st.mean(rest) / 1e6))
long = sorted(its, key=lambda r: -r["total_ns"])[: max(1, len(its) // 100)]
print("slowest 1%% iterations (n=%d): mean %.2f ms | %s | rest %.2f" % (len(long), st.mean(r["total_ns"] for r in long) / 1e6,
      " ".join("%s %.2f" % (p[:-3], st.mean(r[p] for r in long) / 1e6) for p in parts),
      st.mean(r["total_ns"] - sum(r[p] for p in parts) for r in long) / 1e6))
