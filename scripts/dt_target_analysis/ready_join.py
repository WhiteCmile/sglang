"""Cross-host ready-record timeline: sys-4 Target publisher marks joined with sys-3 Draft pull events.
Usage: ready_join.py <cell-dir> <clock.json> [skip_s]
Target clock (sys-4) -> sys-3 clock with clock.json best.sys4_minus_sys3_ns (calibrate_monotonic.py).
Per transfer (target_p0_hop gives transfer_id -> source slot/generation and hidden_produced_ns):
  hidden produced -> listener enqueue -> publisher dequeue -> producer event done -> record posted   (sys-4)
  record posted -> Draft pull submitted (negative = the Draft asked before the record was posted)
  record posted -> Draft observed the record (mapping_done_ns on the RDMA path)                      (cross)
"""
import collections
import glob
import gzip
import json
import statistics as st
import sys

cell, clock = sys.argv[1], sys.argv[2]
skip_ns = float(sys.argv[3]) * 1e9 if len(sys.argv) > 3 else 10e9
off = json.load(open(clock))["best"]["sys4_minus_sys3_ns"]
SYS4 = {"target-3", "target-4", "target-5", "target-6"}


def q(x, p):
    x = sorted(x)
    return x[min(len(x) - 1, int(p * len(x)))] if x else float("nan")


def show(name, xs):
    if xs:
        print("  %-46s n %7d p50 %7.3f mean %7.3f p90 %7.3f p99 %7.3f" % (
            name, len(xs), q(xs, .5) / 1e6, st.mean(xs) / 1e6, q(xs, .9) / 1e6, q(xs, .99) / 1e6))


pub = {}      # (component, slot, gen) -> record
hop = {}      # transfer_id -> (component, slot, gen, hidden_produced_ns)
for f in glob.glob(cell + "/profiles/sys4/target-*.jsonl.gz"):
    for line in gzip.open(f, "rt"):
        if "target_rdma_ready_publish" in line:
            r = json.loads(line)
            pub[(r["component"], r["slot_id"], r["slot_generation"])] = r
        elif "target_p0_hop" in line and "source_slot_id" in line:
            r = json.loads(line)
            if r.get("source_slot_id") is None:
                continue
            hop.setdefault(r["transfer_id"], (r["component"], r["source_slot_id"], r["slot_generation"],
                                               r.get("hidden_produced_ns")))
pulls = []
for f in glob.glob(cell + "/profiles/sys3/draft-*.jsonl.gz"):
    for line in gzip.open(f, "rt"):
        if "draft_feature_pull_complete" in line:
            r = json.loads(line)
            if r["target_instance"] in SYS4:
                pulls.append(r)
print("publish marks", len(pub), "hops with slot", len(hop), "sys-4 pulls", len(pulls))
if pub:
    t0 = min(r["enqueued_ns"] for r in pub.values()) + skip_ns
    P = [r for r in pub.values() if r["enqueued_ns"] >= t0]
    print("== Target publisher (sys-4 clock), after skip")
    show("enqueue -> dequeue", [r["dequeued_ns"] - r["enqueued_ns"] for r in P])
    show("dequeue -> producer event done", [r["event_done_ns"] - r["dequeued_ns"] for r in P])
    show("event done -> posted", [r["posted_ns"] - r["event_done_ns"] for r in P])
    show("enqueue -> posted", [r["posted_ns"] - r["enqueued_ns"] for r in P])
    print("  producer event already done at dequeue: %.1f%%; queue depth after post mean %.2f" % (
        100 * st.mean(r["done_at_dequeue"] for r in P), st.mean(r["queue_depth"] for r in P)))
seg = collections.defaultdict(list)
joined = 0
for r in pulls:
    h = hop.get(r["transfer_id"])
    if not h:
        continue
    comp, slot, gen, produced = h
    p = pub.get((comp, slot, gen))
    if not p:
        continue
    joined += 1
    posted3 = p["posted_ns"] - off
    seg["hidden produced -> enqueue (sys-4)"].append(p["enqueued_ns"] - produced) if produced else None
    seg["record posted -> Draft pull submitted"].append(r["controller_submitted_ns"] - posted3)
    seg["record posted -> Draft observed record"].append(r["mapping_done_ns"] - posted3)
    seg["Draft pull submitted -> observed record"].append(r["mapping_done_ns"] - r["controller_submitted_ns"])
    seg["observed record -> READ posted"].append(r["driver_submit_started_ns"] - r["mapping_done_ns"])
    seg["READ posted -> READ done"].append(r["driver_submit_done_ns"] - r["driver_submit_started_ns"])
    seg["READ done -> worker done"].append(r["gpu_completed_ns"] - r["driver_submit_done_ns"])
    seg["worker done -> Draft owner observed"].append(r["owner_observed_ns"] - r["gpu_completed_ns"])
print("== joined pulls", joined, "(clock offset %d ns, uncertainty %s ns)" % (off, json.load(open(clock))["best"]["uncertainty_ns"]))
for k, v in seg.items():
    show(k, v)
late = seg["record posted -> Draft pull submitted"]
if late:
    print("  pulls submitted BEFORE the record was posted: %.1f%%" % (100 * sum(x < 0 for x in late) / len(late)))
