"""Local metrics for P3 (READY-critical poll / post-launch control split) and P2 (pre-reserve).
Per cell, pooled over Targets: result_poll total and its parts, P0 control/poll phase, candidate->freeze,
freeze->submit, GPU end->coordinator submit, and P2 pre-reserve event counts."""
import json, glob, gzip, sys, collections, statistics as st
def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs)-1, int(q*len(xs)))] if xs else float('nan')
def mean(x): return st.mean(x) if x else float('nan')
print("%-10s %9s %9s %9s %9s %9s %9s %9s %7s" % ("cell", "poll", "receive", "bcast", "ctrlphase", "pollphase", "cand>frz", "frz>sub", "p2ev"))
for cell in sys.argv[1:]:
    poll, recv, bc, ctrl, pph, cf, fs = [], [], [], [], [], [], []
    p2 = 0
    for f in glob.glob(cell + "/profiles/*/target-*.jsonl*"):
        t0 = None
        for l in gzip.open(f, 'rt'):
            r = json.loads(l)
            if t0 is None: t0 = r["ts_ns"] + 5e9
            if r["ts_ns"] < t0: continue
            e = r["event"]
            if e == "target_result_poll":
                poll.append(r["total_ns"] / 1e6); recv.append(r.get("receive_ns", 0) / 1e6); bc.append(r.get("broadcast_ns", 0) / 1e6)
            elif e == "target_p0_hop":
                h = r["hop"]
                if h == "verify_submitted":
                    ctrl.append(r.get("control_phase_ns", 0) / 1e6); pph.append(r.get("poll_phase_ns", 0) / 1e6)
                    if "frz" in r: pass
                elif h == "frozen_for_verify":
                    cf.append((r["freeze_ns"] - r["candidate_observed_ns"]) / 1e6)
            elif e.startswith("target_flashmtp_pre_reserve"):
                p2 += 1
    print("%-10s %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f %9s %7d" % (
        cell.rstrip("/").split("/")[-1], mean(poll), mean(recv), mean(bc), mean(ctrl), mean(pph), mean(cf), "", p2))
