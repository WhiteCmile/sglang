"""Pick Target verify CUDA-graph buckets from per-row cost estimates.

Inputs (either or both):
  --bench shapes-kvN.tsv.. bench_verify_shapes.py outputs, averaged (--op, default layer_sum x 40 layers)
  --trace RUN_DIR ...      traced runs: median target_verify_gpu_breakdown.model_ns per padded bucket
Rule: a bucket is kept only if no larger candidate is cheaper (a dominated bucket just makes the rows
that pad to it slower than padding further); among the kept ones, candidates are every --step rows up
to --dense-max, then the default list.  Prints the comma-separated list for --target-cuda-graph-bs and a
per-row table of the padded cost under the default and the picked list.
"""
import argparse
import bisect
import collections
import glob
import gzip
import json
import statistics

DEFAULT = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 40, 44, 48, 52, 56,
           60, 64] + list(range(72, 257, 8))


def bench_costs(paths, op):
    # op layer_sum (default): per-layer GEMMs + attention x layers; lm_head is excluded because its
    # bench timing is bimodal run to run (330 vs 600 us at the same M) and would dominate the pick.
    acc = collections.defaultdict(list)
    for path in paths:
        for line in open(path):
            p = line.rstrip("\n").split("\t")
            if p[0] == "rows" or p[2] != op:
                continue
            acc[int(p[0])].append(float(p[3]) / 1000.0)  # ms
    scale = 40 if op == "layer_sum" else 1
    return {r: statistics.mean(v) * scale for r, v in acc.items()}


def trace_costs(dirs):
    g = collections.defaultdict(list)
    for d in dirs:
        for f in glob.glob(d + "/c*/profiles/sys3/target-*.jsonl.gz"):
            for line in gzip.open(f, "rt"):
                if "target_verify_gpu_breakdown" not in line:
                    continue
                r = json.loads(line)
                b = DEFAULT[bisect.bisect_left(DEFAULT, r["batch_size"])]
                g[b].append(r["model_ns"] / 1e6)
    return {b: statistics.median(v) for b, v in g.items() if len(v) >= 20}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="*", default=[])
    ap.add_argument("--op", default="layer_sum")
    ap.add_argument("--trace", nargs="*", default=[])
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--dense-max", type=int, default=64)
    a = ap.parse_args()
    cost = bench_costs(a.bench, a.op) if a.bench else trace_costs(a.trace)
    cands = sorted(set([r for r in range(1, a.dense_max + 1) if r <= 8 or r % a.step == 0] + DEFAULT))
    known = [r for r in cands if r in cost]
    keep = []
    for r in known:
        if all(cost[r] <= cost[s] for s in known if s > r):
            keep.append(r)
    picked = sorted(set(keep + [r for r in cands if r not in cost and r > max(known)]))
    if picked[-1] != DEFAULT[-1]:
        picked.append(DEFAULT[-1])
    print(",".join(map(str, picked)))
    dropped = [r for r in known if r not in keep]
    print("# dominated (dropped):", dropped)

    def padded(lst, rows):
        b = lst[bisect.bisect_left(lst, rows)]
        return b, cost.get(b)

    print("# rows  default(bucket,ms)  picked(bucket,ms)")
    for rows in range(1, max(known) + 1):
        bd, cd = padded(DEFAULT, rows)
        bp, cp = padded(picked, rows)
        if cd is None or cp is None:
            continue
        mark = "  <-" if cp < cd - 0.05 else ""
        print("# %3d  %3d %6.2f   %3d %6.2f%s" % (rows, bd, cd, bp, cp, mark))


if __name__ == "__main__":
    main()
