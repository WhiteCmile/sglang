"""Step 2: GPU bubble attribution per Target + round-cycle decomposition with GPU timing.

Input: a cell dir with profiles/sys{3,4}/*.jsonl[.gz] written with target_p0_hop,
target_verify_cuda_observed, target_batch_submit, target_ready_hold ... and DT_GPU_CLOCK=1.
Each gap between consecutive verify waves on one Target (GPU end -> next GPU start) is split into
  starved : no row of the next wave had a candidate yet (waiting on Draft / transfer)
  wait    : rows present but not yet frozen (READY hold, late-freeze, loop busy)
  host    : frozen -> GPU start (build + submit + launch queue)
"""
import json, glob, gzip, sys, collections, statistics as st

def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs)-1, int(q*len(xs)))] if xs else float('nan')
def opener(f): return gzip.open(f, 'rt') if f.endswith('.gz') else open(f)

cell, skip_s = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
per_tgt = collections.defaultdict(lambda: {"waves": {}, "hops": collections.defaultdict(dict), "holds": []})
for f in glob.glob(cell + "/profiles/*/*.jsonl*"):
    for l in opener(f):
        try: r = json.loads(l)
        except Exception: continue
        T = per_tgt[r.get("component")]
        ev = r.get("event")
        if ev == "target_verify_cuda_observed":
            T["waves"][r["wave_id"]] = r
        elif ev == "target_p0_hop":
            T["hops"][r["transfer_id"]][r["hop"]] = r
        elif ev == "target_ready_hold" and r.get("phase") == "release":
            T["holds"].append(r)

agg = collections.defaultdict(float); agg_span = 0.0
print("%-9s %6s %6s %7s | GPU busy | gap/wave ms: starved  wait  host | wave rows | verify ms p50" % ("target", "waves", "span", ""))
cycle_rows = collections.defaultdict(list)
for tgt in sorted(t for t in per_tgt if t and t.startswith("target")):
    T = per_tgt[tgt]
    waves = []
    for wid, w in T["waves"].items():
        if not w.get("gpu_end_clock_ns"): continue
        end = w["gpu_end_clock_ns"]; start = end - w["cuda_total_ns"]
        tids = w.get("transfer_ids") or []
        cands = [T["hops"][t]["candidate_observed"]["completion_observed_ns"] for t in tids if "candidate_observed" in T["hops"][t]]
        frz = [T["hops"][t]["frozen_for_verify"]["freeze_ns"] for t in tids if "frozen_for_verify" in T["hops"][t]]
        waves.append(dict(id=wid, start=start, end=end, n=len(tids), c0=min(cands) if cands else None,
                          frz=min(frz) if frz else w.get("selection_finished_ns"), sub=w.get("submission_returned_ns")))
    if len(waves) < 10: continue
    waves.sort(key=lambda w: w["start"])
    t0 = waves[0]["start"] + skip_s * 1e9
    waves = [w for w in waves if w["start"] >= t0]
    span = (waves[-1]["end"] - waves[0]["start"]) / 1e6
    busy_intervals = sorted((w["start"], w["end"]) for w in waves)
    busy = 0; cur_s, cur_e = busy_intervals[0]
    for s, e in busy_intervals[1:]:
        if s > cur_e: busy += cur_e - cur_s; cur_s, cur_e = s, e
        else: cur_e = max(cur_e, e)
    busy += cur_e - cur_s; busy /= 1e6
    att = collections.defaultdict(float)
    prev_end = waves[0]["end"]
    for w in waves[1:]:
        E, S = prev_end, w["start"]
        if S > E:
            clip = lambda x: E if x is None else min(max(x, E), S)
            c0, fz = clip(w["c0"]), clip(w["frz"])
            fz = max(fz, c0)
            att["starved"] += (c0 - E) / 1e6; att["wait"] += (fz - c0) / 1e6; att["host"] += (S - fz) / 1e6
        prev_end = max(prev_end, w["end"])
    nw = len(waves)
    print("%-9s %6d %6.0f %7s | %5.1f%%  | %21.2f %5.2f %5.2f | %5.2f | %5.2f" % (
        tgt, nw, span, "", 100 * busy / span, att["starved"] / nw, att["wait"] / nw, att["host"] / nw,
        st.mean(w["n"] for w in waves), pct([(w["end"] - w["start"]) / 1e6 for w in waves], .5)))
    for k, v in att.items(): agg[k] += v
    agg["busy"] += busy; agg_span += span
    # round decomposition with GPU timing: verify(r) GPU end -> coordinator submit of round r+1
    by_req = collections.defaultdict(dict)
    for tid, h in T["hops"].items():
        if "feature_published" in h: by_req[h["feature_published"]["request_id"]][h["feature_published"]["round_id"]] = h
    wave_of = {}
    for w in waves:
        for t in T["waves"][w["id"]].get("transfer_ids") or []: wave_of[t] = w
    for req, rounds in by_req.items():
        for r, h in rounds.items():
            nxt = rounds.get(r + 1); tid = h["feature_published"]["transfer_id"]
            if not nxt or tid not in wave_of or "verify_submitted" not in h or "dispatch_registered" not in nxt: continue
            w = wave_of[tid]
            cycle_rows["submit_return -> GPU start (lane queue)"].append((w["start"] - h["verify_submitted"]["submission_returned_ns"]) / 1e6)
            cycle_rows["verify GPU"].append((w["end"] - w["start"]) / 1e6)
            cycle_rows["GPU end -> coord submit (result+park+dispatch)"].append((nxt["dispatch_registered"]["submitted_ns"] - w["end"]) / 1e6)
            if "candidate_observed" in nxt:
                cycle_rows["draft roundtrip"].append((nxt["candidate_observed"]["completion_observed_ns"] - nxt["dispatch_registered"]["submitted_ns"]) / 1e6)
            if "frozen_for_verify" in nxt and "candidate_observed" in nxt:
                cycle_rows["candidate -> freeze"].append((nxt["frozen_for_verify"]["freeze_ns"] - nxt["candidate_observed"]["completion_observed_ns"]) / 1e6)
            if "verify_submitted" in nxt and "frozen_for_verify" in nxt:
                cycle_rows["freeze -> submit_return"].append((nxt["verify_submitted"]["submission_returned_ns"] - nxt["frozen_for_verify"]["freeze_ns"]) / 1e6)

print("\nALL: GPU busy %.1f%%; idle split: starved %.1f%%  wait %.1f%%  host %.1f%% of wall" % (
    100 * agg["busy"] / agg_span, 100 * agg["starved"] / agg_span, 100 * agg["wait"] / agg_span, 100 * agg["host"] / agg_span))
print("\nround cycle (ms): segment | median | mean | p95 | share")
tot = sum(st.mean(v) for v in cycle_rows.values() if v)
for k, v in cycle_rows.items():
    if v: print("  %-48s %7.2f %7.2f %7.2f  %5.1f%%" % (k, pct(v, .5), st.mean(v), pct(v, .95), 100 * st.mean(v) / tot))
print("  sum of means %.2f ms" % tot)
