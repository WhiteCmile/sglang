"""Draft-side decomposition of the Draft roundtrip from one traced cell.

Usage: draft.py <cell-dir> [skip_s]

Joins per transfer_id (all on sys-3's CLOCK_MONOTONIC: coordinator, both Drafts, target-0..2):
  target dispatch_registered.submitted_ns   (target-0..2 only; sys-4 Targets have another clock)
  -> coordinator_task_dispatch ts           coordinator receive + routing
  -> draft_feature_pull_submit.submitted_ns Draft picked the task up
  -> draft_feature_pull_complete.owner_observed_ns   hidden/pivot pulled from the Target
  -> draft_input_ready.ready_ns
  -> draft_flashmtp_proposal.execute_started_ns      waits for a Draft batch slot
  -> execute_started + total_ns              proposal done (GPU + D2H)
  -> draft_candidate_push_complete.completed_ns      candidates written back to the Target
  -> target candidate_observed.completion_observed_ns (target-0..2)
Also: per-Draft batch size, proposal GPU time vs batch size, Draft GPU busy fraction.
"""
import json, glob, gzip, sys, collections, statistics as st

def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs)-1, int(q*len(xs)))] if xs else float('nan')
def opener(f): return gzip.open(f, 'rt') if f.endswith('.gz') else open(f)
def mean(xs): return st.mean(xs) if xs else float('nan')

cell = sys.argv[1]; skip = float(sys.argv[2]) * 1e9 if len(sys.argv) > 2 else 5e9
T = collections.defaultdict(dict)        # transfer -> stage -> ns (sys-3 clock)
props = collections.defaultdict(list)    # draft instance -> proposals
pull_dur = collections.defaultdict(list) # source host -> pull ms
push_dur = collections.defaultdict(list)
t_first = None
for f in glob.glob(cell + "/profiles/sys3/*.jsonl*"):
    for l in opener(f):
        try: r = json.loads(l)
        except Exception: continue
        e = r.get("event"); ts = r.get("ts_ns", 0)
        if t_first is None or ts < t_first: t_first = ts
        if e == "coordinator_task_dispatch":
            for t in r.get("accepted_transfer_ids") or r.get("transfer_ids") or []:
                T[t].setdefault("coord", ts)
        elif e == "draft_feature_pull_submit":
            T[r["transfer_id"]]["pull_sub"] = r["submitted_ns"]
        elif e == "draft_feature_pull_complete":
            T[r["transfer_id"]]["pull_done"] = r["owner_observed_ns"]
            src = "sys4" if r["transfer_id"].split(":")[0] in ("target-3", "target-4", "target-5", "target-6") else "sys3"
            pull_dur[src].append((r["owner_observed_ns"] - r["controller_submitted_ns"]) / 1e6)
        elif e == "draft_input_ready":
            T[r["transfer_id"]]["ready"] = r["ready_ns"]
        elif e == "draft_flashmtp_proposal":
            props[r.get("draft_instance") or r.get("component")].append(r)
            for t in r.get("transfer_ids") or []:
                T[t]["exec"] = r["execute_started_ns"]; T[t]["prop_done"] = r["execute_started_ns"] + r["total_ns"]
                T[t]["batch"] = r["batch_size"]
        elif e == "draft_candidate_push_complete":
            src_push = (r["completed_ns"] - r["submitted_ns"]) / 1e6
            for t in r.get("transfer_ids") or []:
                T[t]["push_done"] = r["completed_ns"]
                push_dur["sys4" if t.split(":")[0] in ("target-3", "target-4", "target-5", "target-6") else "sys3"].append(src_push)
        elif e == "target_p0_hop":
            if r["hop"] == "dispatch_registered": T[r["transfer_id"]]["t_disp"] = r["submitted_ns"]
            elif r["hop"] == "candidate_observed": T[r["transfer_id"]]["t_cand"] = r["completion_observed_ns"]
t0 = (t_first or 0) + skip

print("== Draft batches (after first %.0fs)" % (skip / 1e9))
for inst, ps in sorted(props.items()):
    ps = [p for p in ps if p["execute_started_ns"] > t0]
    if not ps: continue
    span = (max(p["execute_started_ns"] + p["total_ns"] for p in ps) - min(p["execute_started_ns"] for p in ps))
    gpu = sum(p.get("proposal_gpu_ns") or 0 for p in ps)
    busy = sum(p["total_ns"] for p in ps)
    bs = [p["batch_size"] for p in ps]
    print("  %s: %d batches, rows/batch mean %.1f p50 %d p90 %d | proposal GPU %.0f%% of wall, total(exec) %.0f%% | %.0f batches/s" % (
        inst, len(ps), mean(bs), pct(bs, .5), pct(bs, .9), 100 * gpu / span, 100 * busy / span, len(ps) / (span / 1e9)))
    byb = collections.defaultdict(list)
    for p in ps: byb[min(p["batch_size"] // 8 * 8, 64)].append(p)
    print("    rows-bucket: n | GPU ms | total ms | GPU us/row")
    for b in sorted(byb):
        g = [p.get("proposal_gpu_ns", 0) / 1e6 for p in byb[b]]; tt = [p["total_ns"] / 1e6 for p in byb[b]]
        rows = mean([p["batch_size"] for p in byb[b]])
        print("    %3d-%-3d %6d | %6.2f | %6.2f | %6.0f" % (b, b + 7, len(byb[b]), mean(g), mean(tt), 1000 * mean(g) / rows))

print("\n== feature pull (controller submit -> owner observed) ms by source host")
for h, v in sorted(pull_dur.items()): print("  %s n=%d p50 %.2f mean %.2f p95 %.2f p99 %.2f" % (h, len(v), pct(v, .5), mean(v), pct(v, .95), pct(v, .99)))
print("== candidate push (submit -> complete) ms by destination host")
for h, v in sorted(push_dur.items()): print("  %s n=%d p50 %.2f mean %.2f p95 %.2f" % (h, len(v), pct(v, .5), mean(v), pct(v, .95)))

chain = [("t_disp", "coord", "target dispatch -> coordinator"), ("coord", "pull_sub", "coordinator -> Draft pull submit"),
         ("pull_sub", "pull_done", "feature pull"), ("pull_done", "ready", "pull done -> input ready"),
         ("ready", "exec", "input ready -> proposal start (queue)"), ("exec", "prop_done", "proposal (GPU+D2H)"),
         ("prop_done", "push_done", "proposal done -> push complete"), ("push_done", "t_cand", "push complete -> Target observes")]
for group, pred in (("sys-3 Targets (full join)", lambda t: t.split(":")[0] in ("target-0", "target-1", "target-2")),
                    ("sys-4 Targets (Draft-local stages)", lambda t: t.split(":")[0] in ("target-3", "target-4", "target-5", "target-6"))):
    print("\n== Draft roundtrip stages, %s: median | mean | p95 | p99 (ms)" % group)
    rows = [s for t, s in T.items() if pred(t) and s.get("pull_sub", 0) > t0]
    tot = 0
    for a, b, name in chain:
        v = [(s[b] - s[a]) / 1e6 for s in rows if a in s and b in s]
        if len(v) < 50: continue
        tot += mean(v)
        print("  %-40s %7.2f %7.2f %7.2f %7.2f  (n=%d)" % (name, pct(v, .5), mean(v), pct(v, .95), pct(v, .99), len(v)))
    full = [(s["t_cand"] - s["t_disp"]) / 1e6 for s in rows if "t_cand" in s and "t_disp" in s]
    if full: print("  %-40s %7.2f %7.2f %7.2f %7.2f" % ("TOTAL target dispatch -> observe", pct(full, .5), mean(full), pct(full, .95), pct(full, .99)))
    print("  sum of stage means %.2f" % tot)
    # where does the tail come from: stage means for rounds whose total is > 20 ms
    slow = [s for s in rows if all(k in s for k in ("pull_sub", "push_done")) and (s["push_done"] - s["pull_sub"]) > 20e6]
    if slow:
        print("  rounds with pull_sub->push_done > 20 ms: %d (%.1f%%); their stage means:" % (len(slow), 100 * len(slow) / max(1, len(rows))))
        for a, b, name in chain[2:7]:
            v = [(s[b] - s[a]) / 1e6 for s in slow if a in s and b in s]
            if v: print("    %-38s %7.2f" % (name, mean(v)))
