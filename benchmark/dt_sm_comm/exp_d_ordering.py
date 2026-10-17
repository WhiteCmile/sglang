"""Experiment D: cross-GPU data-before-flag ordering for Draft-side communication.

Two processes: this one plays Target (GPU --target), a spawned child plays
Draft (GPU --draft). Buffers are exchanged with CUDA IPC (torch reductions).
Every iteration stamps the whole payload with the iteration number; the reader
counts words that still hold an older value after it observed the flag. A
ping-pong ack keeps the writer from overwriting a payload still being read.

Variants (writer -> payload location -> flag location -> observer):
  push-sm-host-cpu   Draft persistent kernel -> Target mem -> /dev/shm -> Target CPU poll, then check kernel
  push-sm-host-wait  Draft persistent kernel -> Target mem -> /dev/shm -> Target stream WaitValue, check kernel
  push-sm-dev-wait   Draft persistent kernel -> Target mem -> Target mem -> Target stream WaitValue, check kernel
  push-ce-host-wait  Draft stream memcpy (copy engine) + WriteValue -> /dev/shm -> Target stream WaitValue
  push-ce-dev-wait   Draft stream memcpy + WriteValue into Target mem -> Target stream WaitValue
  pull-sm-host       Target stream fill + WriteValue desc in /dev/shm -> Draft persistent kernel reads Target mem
  pull-sm-dev        Target stream fill + desc into Draft mem -> Draft persistent kernel reads Target mem

fence=0 is the negative control for the persistent-kernel variants: it shows
whether the check can detect reordering at all.

Each config prints one JSON line.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time

import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sm_comm_ext import Driver, SharedFlags, load_ext  # noqa: E402

VARIANTS = (
    "push-sm-host-cpu",
    "push-sm-host-wait",
    "push-sm-dev-wait",
    "push-ce-host-wait",
    "push-ce-dev-wait",
    "pull-sm-host",
    "pull-sm-dev",
)
TIMEOUT_S = 120.0


def build_configs(args):
    configs = []
    for variant in args.variants:
        persistent = "-sm-" in variant
        for size in args.sizes:
            iters = max(200, int(args.iters * min(1.0, (65536 / size) ** 0.5)))
            fences = (1, 0) if persistent and args.negative_control else (1,)
            load_modes = (0, 1) if variant.startswith("pull") else (0,)
            for fence in fences:
                for load_mode in load_modes:
                    if fence == 0 and load_mode != 0:
                        continue
                    for blocks in (args.blocks if persistent else (0,)):
                        configs.append(dict(variant=variant, bytes=size, iters=iters,
                                            fence=fence, load_mode=load_mode, blocks=blocks))
    return configs


def _wait_event(ev: torch.cuda.Event, what: str) -> None:
    start = time.perf_counter()
    while not ev.query():
        if time.perf_counter() - start > TIMEOUT_S:
            print(json.dumps({"fatal": f"timeout waiting for {what}"}), flush=True)
            os._exit(3)
        time.sleep(0.0002)


def _bg_gemm(device: int, stop: threading.Event) -> None:
    torch.cuda.set_device(device)
    s = torch.cuda.Stream(priority=0)
    a = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
    with torch.cuda.stream(s):
        while not stop.is_set():
            for _ in range(8):
                a = torch.matmul(a, b).clamp_(-1, 1)
            s.synchronize()


def draft_main(args, configs, to_draft, from_draft, barrier, shm_path):
    d = args.draft
    torch.cuda.set_device(d)
    ext = load_ext(args.build_dir)
    drv = Driver()
    shm = SharedFlags(shm_path, drv, create=False)
    max_bytes = max(args.sizes)
    # Payload, flag and hidden live in Target memory; map them on the Draft device.
    target_data, flag_dev, hidden = (ext.ipc_open(h) for h in to_draft.get())
    staging = ext.dev_alloc(max_bytes)
    desc_dev = ext.dev_alloc(256)
    done = torch.zeros(1, dtype=torch.int32, device=d)
    err = torch.zeros(1, dtype=torch.int64, device=d)
    max_iters = max(c["iters"] for c in configs)
    t_a = torch.zeros(max_iters, dtype=torch.int64, device=d)
    t_b = torch.zeros(max_iters, dtype=torch.int64, device=d)
    peer_write_rc = drv.write(torch.cuda.current_stream(), flag_dev, 0)
    torch.cuda.synchronize()
    from_draft.put({"desc_dev": ext.ipc_get(desc_dev), "peer_write_rc": peer_write_rc})
    stop = threading.Event()
    if args.bg_gemm:
        threading.Thread(target=_bg_gemm, args=(d, stop), daemon=True).start()
    stream = torch.cuda.Stream(priority=-1)

    for cfg in configs:
        done.zero_(); err.zero_(); t_a.zero_(); t_b.zero_()
        ext.memset_async(desc_dev, 256)
        torch.cuda.synchronize()
        barrier.wait(TIMEOUT_S)
        n_vec = cfg["bytes"] // 16
        iters = cfg["iters"]
        v = cfg["variant"]
        t0 = time.perf_counter()
        with torch.cuda.stream(stream):
            if v.startswith("push-sm"):
                flag = shm.addr("flag") if "-host-" in v else flag_dev
                ext.push_writer(target_data, n_vec, flag, shm.addr("ack"), done,
                                t_a, t_b, iters, cfg["fence"], cfg["blocks"], 512)
            elif v.startswith("push-ce"):
                flag = shm.addr("flag") if "-host-" in v else flag_dev
                for it in range(1, iters + 1):
                    drv.wait_eq(stream, shm.addr("ack"), it - 1)
                    ext.fill(staging, n_vec, it)
                    ext.memcpy_async(target_data, staging, cfg["bytes"])
                    if drv.write(stream, flag, it) != 0:
                        ext.store_flag(flag, it)
            else:
                desc = shm.addr("desc") if v.endswith("host") else desc_dev
                ext.pull_reader(hidden, n_vec, desc, shm.addr("ackp"), done, err,
                                t_a, iters, cfg["fence"], cfg["load_mode"], cfg["blocks"], 512)
        ev = torch.cuda.Event()
        ev.record(stream)
        _wait_event(ev, f"draft {v}")
        wall = time.perf_counter() - t0
        res = {"draft_wall_s": wall, "draft_err": int(err.item())}
        if v.startswith("push-sm"):
            rtt = ((t_b[:iters] - t_a[:iters]).double() / 1e3).tolist()
            res["rtt_us"] = _stats(rtt)
        elif v.startswith("pull"):
            gaps = (t_a[1:iters] - t_a[: iters - 1]).double().div(1e3).tolist()
            res["rtt_us"] = _stats(gaps)
        barrier.wait(TIMEOUT_S)
        from_draft.put(res)
    stop.set()
    barrier.wait(TIMEOUT_S)


def _stats(xs):
    if not xs:
        return {}
    xs = sorted(xs)
    pick = lambda q: round(xs[min(len(xs) - 1, int(q * (len(xs) - 1)))], 1)  # noqa: E731
    return {"p50": pick(0.5), "p90": pick(0.9), "p99": pick(0.99), "max": round(xs[-1], 1),
            "mean": round(statistics.fmean(xs), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=0)
    ap.add_argument("--draft", type=int, default=1)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--sizes", nargs="+", type=int, default=[4096, 262144, 8388608])
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--blocks", nargs="+", type=int, default=[4])
    ap.add_argument("--no-negative-control", dest="negative_control", action="store_false")
    ap.add_argument("--bg-gemm", action="store_true", help="saturate both GPUs with GEMMs")
    ap.add_argument("--build-dir", default=os.path.join(os.path.dirname(__file__), ".build"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    configs = build_configs(args)
    t = args.target
    torch.cuda.set_device(t)
    ext = load_ext(args.build_dir)
    drv = Driver()
    shm_path = f"/dev/shm/dt-sm-comm-d-{os.getpid()}"
    shm = SharedFlags(shm_path, drv, create=True)
    shm.reset()

    max_bytes = max(args.sizes)
    target_data = ext.dev_alloc(max_bytes)
    hidden = ext.dev_alloc(max_bytes)
    flag_dev = ext.dev_alloc(256)
    err = torch.zeros(1, dtype=torch.int64, device=t)

    ctx = mp.get_context("spawn")
    to_draft, from_draft = ctx.Queue(), ctx.Queue()
    barrier = ctx.Barrier(2)
    child = ctx.Process(target=draft_main,
                        args=(args, configs, to_draft, from_draft, barrier, shm_path))
    child.start()
    to_draft.put(tuple(ext.ipc_get(p) for p in (target_data, flag_dev, hidden)))
    hello = from_draft.get(timeout=600)
    desc_dev = ext.ipc_open(hello["desc_dev"])
    target_peer_write_rc = drv.write(torch.cuda.current_stream(), desc_dev, 0)
    torch.cuda.synchronize()
    header = {"header": True, "torch": torch.__version__, "target": t, "draft": args.draft,
              "bg_gemm": args.bg_gemm,
              "draft_peer_write_value_rc": hello["peer_write_rc"],
              "target_peer_write_value_rc": target_peer_write_rc}
    out = open(args.out, "a") if args.out else None
    _emit(header, out)

    stop = threading.Event()
    if args.bg_gemm:
        threading.Thread(target=_bg_gemm, args=(t, stop), daemon=True).start()
    stream = torch.cuda.Stream(priority=-1)

    for cfg in configs:
        shm.reset()
        err.zero_()
        for p, nbytes in ((target_data, max_bytes), (hidden, max_bytes), (flag_dev, 256)):
            ext.memset_async(p, nbytes)
        torch.cuda.synchronize()
        barrier.wait(TIMEOUT_S)
        n_vec = cfg["bytes"] // 16
        iters = cfg["iters"]
        v = cfg["variant"]
        t0 = time.perf_counter()
        with torch.cuda.stream(stream):
            if v == "push-sm-host-cpu":
                for it in range(1, iters + 1):
                    spin_start = time.perf_counter()
                    while shm.get("flag") != it:
                        if time.perf_counter() - spin_start > TIMEOUT_S:
                            _emit({"fatal": "target cpu poll timeout", **cfg}, out)
                            os._exit(3)
                    ext.check(target_data, n_vec, it, err)
                    drv.write(stream, shm.addr("ack"), it)
            elif v.startswith("push"):
                flag = shm.addr("flag") if "-host-" in v else flag_dev
                for it in range(1, iters + 1):
                    drv.wait_eq(stream, flag, it)
                    ext.check(target_data, n_vec, it, err)
                    drv.write(stream, shm.addr("ack"), it)
            else:
                desc = shm.addr("desc") if v.endswith("host") else desc_dev
                for it in range(1, iters + 1):
                    drv.wait_eq(stream, shm.addr("ackp"), it - 1)
                    ext.fill(hidden, n_vec, it)
                    if drv.write(stream, desc, it) != 0:
                        ext.store_flag(desc, it)
        ev = torch.cuda.Event()
        ev.record(stream)
        _wait_event(ev, f"target {v}")
        wall = time.perf_counter() - t0
        barrier.wait(TIMEOUT_S)
        dres = from_draft.get(timeout=TIMEOUT_S)
        stale = int(err.item()) + dres.pop("draft_err")
        rec = {**cfg, "stale_thread_reads": stale, "wall_s": round(wall, 3),
               "mean_iter_us": round(wall / iters * 1e6, 1), **dres}
        rec["draft_wall_s"] = round(rec["draft_wall_s"], 3)
        _emit(rec, out)
    stop.set()
    barrier.wait(TIMEOUT_S)
    child.join(60)
    os.unlink(shm_path)


def _emit(rec, out) -> None:
    line = json.dumps(rec)
    print(line, flush=True)
    if out:
        out.write(line + "\n")
        out.flush()


if __name__ == "__main__":
    main()
