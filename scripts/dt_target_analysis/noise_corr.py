"""Per cell: long-window tok/s vs cross-host slow feature pulls (>20 ms) and the time they cost.
Usage: noise_corr.py <cell-dir>..."""
import json, glob, gzip, sys, os
for cell in sys.argv[1:]:
    jf = glob.glob(cell + "/long1.json")
    if not jf: continue
    tok = json.load(open(jf[0]))["output_tokens_per_s"]
    n = slow = 0; slow_ms = 0.0
    for f in glob.glob(cell + "/profiles/sys3/draft-*.jsonl*"):
        for l in gzip.open(f, 'rt'):
            if '"draft_feature_pull_complete"' not in l: continue
            r = json.loads(l); n += 1
            d = (r["owner_observed_ns"] - r["controller_submitted_ns"]) / 1e6
            if d > 20: slow += 1; slow_ms += d
    print("%-40s tok/s %7.1f  slow pulls %5d  (%.0f s total)  arm=%s" % (
        "/".join(cell.rstrip("/").split("/")[-2:]), tok, slow, slow_ms / 1e3, open(cell + "/arm.txt").read().strip()))
