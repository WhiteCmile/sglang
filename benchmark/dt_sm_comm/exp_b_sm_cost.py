"""Experiment B (synthetic): proposal slowdown while a resident kernel holds K SMs.

A resident pull kernel with K CTAs (one CTA per SM at 512 threads plus the
kernel's shared memory) spins on an empty descriptor ring on the same GPU,
then a CUDA graph of DFlash-draft-shaped layers is replayed and timed.

Layer stand-in per draft layer (Qwen3-14B width 5120, intermediate 17408):
qkv/o projections, gate_up/down MLP, at M = batch * 16 block tokens. The real
DFlash proposal adds attention and materialize; this measures what holding SMs
costs the dense GEMMs, which dominate the proposal's GPU time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sm_comm_ext import Driver, load_ext  # noqa: E402
from sm_comm_runtime import OFF_DESC, OFF_READY, RING, TMASK, Region, load_runtime, torch_int64  # noqa: E402

HIDDEN = 5120
INTER = 17408
QKV = 5120 + 2 * 1024


def build_graph(device, m, layers):
    w = {
        "qkv": torch.randn(HIDDEN, QKV, device=device, dtype=torch.bfloat16) * 0.01,
        "o": torch.randn(HIDDEN, HIDDEN, device=device, dtype=torch.bfloat16) * 0.01,
        "gu": torch.randn(HIDDEN, 2 * INTER, device=device, dtype=torch.bfloat16) * 0.01,
        "down": torch.randn(INTER, HIDDEN, device=device, dtype=torch.bfloat16) * 0.01,
    }
    x = torch.randn(m, HIDDEN, device=device, dtype=torch.bfloat16)

    def body():
        h = x
        for _ in range(layers):
            qkv = h @ w["qkv"]
            h = h + qkv[:, :HIDDEN] @ w["o"]
            gu = h @ w["gu"]
            h = h + (torch.nn.functional.silu(gu[:, :INTER]) * gu[:, INTER:]) @ w["down"]
        return h

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        body()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            body()
    torch.cuda.synchronize()
    return graph, (w, x)


def time_graph(graph, reps):
    # Never torch.cuda.synchronize() here: a device-wide sync waits on the resident kernel forever.
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(5):
        graph.replay()
    torch.cuda.current_stream().synchronize()
    times = []
    for _ in range(reps):
        s.record()
        graph.replay()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2], times[int(0.9 * (len(times) - 1))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=1)
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 9, 24, 48])
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--ctas", nargs="+", type=int, default=[0, 1, 2, 4, 8, 16])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--build-dir", default=os.path.join(os.path.dirname(__file__), ".build"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev = args.device
    torch.cuda.set_device(dev)
    ext = load_ext(args.build_dir)
    rt = load_runtime(args.build_dir)
    drv = Driver()
    path = f"/dev/shm/dt-sm-comm-b-{os.getpid()}"
    reg = Region(path, drv, create=True)
    reg.reset()
    staging = ext.dev_alloc(1 << 20)
    ticket = ext.dev_alloc(256)
    t_pull = torch_int64(TMASK + 1, dev)
    resident = torch.cuda.Stream(priority=-1)
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    graphs = {b: build_graph(dev, b * args.block, args.layers) for b in args.batches}
    out = open(args.out, "a") if args.out else None
    base = {}
    for k in args.ctas:
        reg.reset()
        ext.memset_async(ticket, 256)
        torch.cuda.synchronize()
        done = None
        if k:
            with torch.cuda.stream(resident):
                rt.resident_pull(reg.dev(OFF_DESC), reg.dev(OFF_READY), reg.stop_dev, ticket,
                                 staging, staging, 1, t_pull, k, 512, RING, TMASK)
                done = torch.cuda.Event()
                done.record(resident)
        for b, (graph, _) in graphs.items():
            p50, p90 = time_graph(graph, args.reps)
            if k == 0:
                base[b] = p50
            rec = {"sms": sms, "resident_ctas": k, "batch": b, "tokens": b * args.block,
                   "layers": args.layers, "graph_ms_p50": round(p50, 4), "graph_ms_p90": round(p90, 4),
                   "slowdown_pct": round(100 * (p50 / base[b] - 1), 2)}
            line = json.dumps(rec)
            print(line, flush=True)
            if out:
                out.write(line + "\n")
                out.flush()
        reg.ctrl[0] = 1
        if done is not None:
            done.synchronize()
    os.unlink(path)


if __name__ == "__main__":
    main()
