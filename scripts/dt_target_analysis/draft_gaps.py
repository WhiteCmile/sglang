"""Find the longest silences in one Draft's trace (no event of any kind) during the measured window and print
what happened right before/after each, plus long individual stages that overlap the silence."""
import json, glob, gzip, sys
def opener(f): return gzip.open(f, 'rt') if f.endswith('.gz') else open(f)
cell, inst = sys.argv[1], sys.argv[2]
f = glob.glob(cell + "/profiles/sys3/%s-*.jsonl*" % inst)[0]
ev = []
for l in opener(f):
    r = json.loads(l); ev.append(r)
ev.sort(key=lambda r: r["ts_ns"]); t0 = ev[0]["ts_ns"]
gaps = sorted(((b["ts_ns"] - a["ts_ns"], i) for i, (a, b) in enumerate(zip(ev, ev[1:]))), reverse=True)
shown = 0
for g, i in gaps:
    a, b = ev[i], ev[i + 1]
    ts = (a["ts_ns"] - t0) / 1e9
    if g > 2e9: continue  # idle between benchmark phases
    print("gap %.1f ms at %.2fs: before=%s after=%s" % (g / 1e6, ts, a["event"], b["event"]))
    for r in ev[i + 1:i + 4]:
        keep = {k: v for k, v in r.items() if k.endswith("_ns") and isinstance(v, int) and v < 10**12 and k != "ts_ns"}
        print("    after: %s %s" % (r["event"], {k: round(v / 1e6, 2) for k, v in keep.items()}))
    shown += 1
    if shown >= 8: break
