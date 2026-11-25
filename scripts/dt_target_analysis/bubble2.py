"""Step 2 (FlashMTP has no per-wave CUDA events): approximate GPU idle from result-queue depth.

Per Target: GPU has no verify work from the moment the main loop observes the last outstanding
result's copy_done (result queue depth -> 0) until the next verify submission starts.
Each such idle gap [E, S] before wave w is attributed using w's rows:
  starved : before the first candidate of w arrived (waiting on Draft / transport)
  wait    : candidates present, wave not frozen yet (READY hold / late-freeze / loop)
  host    : freeze -> submission start (batch build)
Also reports result-processing time and the submission wall (both host work on the round path).
"""
import json, glob, gzip, sys, collections, statistics as st

def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs)-1, int(q*len(xs)))] if xs else float('nan')
def opener(f): return gzip.open(f, 'rt') if f.endswith('.gz') else open(f)

cell, skip_s = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
TOT = collections.defaultdict(float)
proc_all, subwall_all, holds_all = [], [], collections.Counter()
print("%-9s %5s %6s | idle%% | idle split %% of wall: starved  wait  host | rows/wave | result-proc ms p50/mean | submit wall ms p50" % ("target", "waves", "span s"))
for f in sorted(glob.glob(cell + "/profiles/*/*.jsonl*")):
    tgt = f.split("/")[-1].rsplit("-", 1)[0]
    subs, readies, copies, procs, holds = [], {}, [], [], []
    cand = {}
    for l in opener(f):
        try: r = json.loads(l)
        except Exception: continue
        e = r.get("event")
        if e == "target_batch_submit": subs.append(r)
        elif e == "target_ready_batch": readies[r["wave_id"]] = r
        elif e == "target_old_result_copy_ready" and r.get("copy_done_ready"): copies.append(r)
        elif e == "target_old_result_process": procs.append(r)
        elif e == "target_ready_hold" and r.get("phase") == "release": holds.append(r)
        elif e == "target_p0_hop" and r.get("hop") == "candidate_observed": cand[r["transfer_id"]] = r["completion_observed_ns"]
    if not subs: continue
    t0 = min(s["submission_started_ns"] for s in subs) + skip_s * 1e9
    t1 = max(s["submission_started_ns"] for s in subs)
    subs = [s for s in subs if s["submission_started_ns"] >= t0]
    copies = [c for c in copies if c["copy_done_observed_ns"] >= t0]
    ev = [(c["copy_done_observed_ns"], 0, c) for c in copies] + [(s["submission_started_ns"], 1, s) for s in subs]
    ev.sort(key=lambda x: (x[0], x[1]))
    att = collections.defaultdict(float); idle_start = None
    for t, kind, r in ev:
        if kind == 0:
            if r["result_queue_depth_before"] - 1 <= 0: idle_start = t
        else:
            if idle_start is not None and t > idle_start:
                E, S = idle_start, t
                c0 = min((cand[x] for x in r.get("transfer_ids") or [] if x in cand), default=None)
                rb = readies.get(r["wave_id"])
                fz = rb["freeze_ns"] if rb else r["selection_finished_ns"]
                clip = lambda x: E if x is None else min(max(x, E), S)
                a, b = clip(c0), clip(fz); b = max(a, b)
                att["starved"] += a - E; att["wait"] += b - a; att["host"] += S - b
            idle_start = None
    span = t1 - t0
    idle = sum(att.values())
    pv = [p["total_ns"] / 1e6 for p in procs if p["started_ns"] >= t0]
    sw = [s["submission_wall_ns"] / 1e6 for s in subs]
    rows = [s["batch_size"] for s in subs]
    print("%-9s %5d %6.1f | %4.1f%% | %26.1f %5.1f %5.1f | %5.2f | %5.2f / %5.2f | %5.2f" % (
        tgt, len(subs), span / 1e9, 100 * idle / span, 100 * att["starved"] / span, 100 * att["wait"] / span, 100 * att["host"] / span,
        st.mean(rows), pct(pv, .5), st.mean(pv), pct(sw, .5)))
    for k, v in att.items(): TOT[k] += v
    TOT["span"] += span; proc_all += pv; subwall_all += sw
    for h in holds:
        if h.get("start_ns", h.get("ts_ns")) >= t0: holds_all[h.get("reason")] += 1
print("\nALL targets: idle %.1f%% of wall = starved %.1f%% + wait %.1f%% + host %.1f%%" % (
    100 * (TOT["starved"] + TOT["wait"] + TOT["host"]) / TOT["span"], 100 * TOT["starved"] / TOT["span"], 100 * TOT["wait"] / TOT["span"], 100 * TOT["host"] / TOT["span"]))
print("result processing (old_result_process) ms: p50 %.2f mean %.2f p95 %.2f" % (pct(proc_all, .5), st.mean(proc_all), pct(proc_all, .95)))
print("submission wall ms: p50 %.2f mean %.2f p95 %.2f" % (pct(subwall_all, .5), st.mean(subwall_all), pct(subwall_all, .95)))
print("READY hold releases by reason:", dict(holds_all))
