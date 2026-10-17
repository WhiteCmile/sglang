"""Experiment A: Draft-side communication mechanisms in a closed-loop round.

Target process (GPU --target) emulates the Target CPU loop: N requests, waves
every --wave-ms; a request that became READY before wave k is published at
wave k+1 as a hidden descriptor (CPU store into the shared /dev/shm ring, sizes
from the r8 DFlash trace). A separate process keeps the Target GPU saturated
with GEMMs (verify stand-in).

Draft process (GPU --draft) emulates the Draft engine: it learns INPUT_READY,
spends --launch-host-ms of host time per batch, replays a CUDA graph of
--proposal-ms GPU work, then pushes 28-byte candidates into Target memory and
publishes READY in /dev/shm.

Pull mechanisms (descriptor -> hidden in Draft staging -> INPUT_READY):
  python      Draft Python main loop reads descriptors, cudaMemcpyAsync + WriteValue (current inline pull)
  cpp         dedicated C++ thread (no GIL) polls descriptors, cudaMemcpyAsync + WriteValue, 0 SM
  residentK   resident kernel with K CTAs polls descriptors and copies over NVLink with SM loads
Push mechanisms (proposal end -> candidates in Target memory -> READY visible to Target CPU):
  python      Draft Python loop polls the proposal event, then memcpy + WriteValue (current)
  queued      push kernel queued on the proposal stream right after the proposal, 0 resident SM, no CPU

C++ observer threads stamp INPUT_READY and READY independently of the Python
loops. Each config prints one JSON line.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sm_comm_ext import Driver, load_ext  # noqa: E402
from sm_comm_runtime import (  # noqa: E402
    MAX_PUSH, OFF_COMPLETE, OFF_DESC, OFF_READY, RING, TMASK, Region, busy_wait_ms, load_runtime,
    torch_int64,
)

ROW_BYTES = 51200  # DFlash Qwen3-14B: 5 layers x 5120 x bf16
MAX_ROWS = 7
MAX_REQ_BYTES = ROW_BYTES * MAX_ROWS
STAGING_SLOTS = 256
# Rows per request per round, r8 DFlash trace (slo-claude-matg1-r8, 268,819 pulls).
ROW_WEIGHTS = np.array([91465, 60365, 37229, 23489, 15646, 10813, 29812], dtype=np.float64)
TIMEOUT_S = 300.0


def build_configs(args):
    configs = []
    for pull in args.pulls:
        for push in args.pushes:
            configs.append({"pull": pull, "push": push,
                            "start_seq": (len(configs) + 1) << 20})
    return configs


def _target_load(device: int, stop) -> None:
    torch.cuda.set_device(device)
    a = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device=device, dtype=torch.bfloat16)
    while not stop.is_set():
        for _ in range(8):
            a = torch.matmul(a, b).clamp_(-1, 1)
        torch.cuda.synchronize()


def _capture_proposal(device: int, target_ms: float):
    """CUDA graph of GEMMs lasting about target_ms (proposal stand-in)."""
    a = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    out = torch.empty_like(a)
    for _ in range(3):
        torch.matmul(a, b, out=out)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(20):
        torch.matmul(a, b, out=out)
    e.record()
    torch.cuda.synchronize()
    reps = max(1, round(target_ms / (s.elapsed_time(e) / 20)))
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        torch.matmul(a, b, out=out)
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(reps):
                torch.matmul(a, b, out=out)
    torch.cuda.synchronize()
    s.record()
    graph.replay()
    e.record()
    torch.cuda.synchronize()
    return graph, (a, b, out), reps, s.elapsed_time(e)


def draft_main(args, configs, q_in, q_out, barrier, region_path):
    d = args.draft
    torch.cuda.set_device(d)
    ext = load_ext(args.build_dir)
    rt = load_runtime(args.build_dir)
    drv = Driver()
    reg = Region(region_path, drv, create=False)
    hidden_h, cand_h = q_in.get()
    hidden = ext.ipc_open(hidden_h)
    cand_dst = ext.ipc_open(cand_h)
    staging = ext.dev_alloc(STAGING_SLOTS * MAX_REQ_BYTES)
    cand_src = ext.dev_alloc(MAX_PUSH * 32)
    ticket = ext.dev_alloc(256)
    t_end = torch_int64(1 << 16, d)
    t_pull = torch_int64(TMASK + 1, d)
    graph, _keep, reps, graph_ms = _capture_proposal(d, args.proposal_ms)
    q_out.put({"proposal_reps": reps, "proposal_graph_ms_idle": round(graph_ms, 3)})

    compute = torch.cuda.Stream()
    comm = torch.cuda.Stream(priority=-1)
    resident_stream = torch.cuda.Stream(priority=-1)
    obs_ready = np.zeros(TMASK + 1, dtype=np.int64)
    seen_at = np.zeros(TMASK + 1, dtype=np.int64)
    launch_at = np.zeros(TMASK + 1, dtype=np.int64)
    batch_of = np.full(TMASK + 1, -1, dtype=np.int64)
    mono = rt.monotonic_ns

    for cfg in configs:
        barrier.wait(TIMEOUT_S)
        for arr in (obs_ready, seen_at, launch_at):
            arr[:] = 0
        batch_of[:] = -1
        t_end.zero_()
        ext.memset_async(ticket, 256)
        torch.cuda.synchronize()
        start = cfg["start_seq"]
        workers = [rt.start_observer(reg.cpu(OFF_READY), obs_ready.ctypes.data, args.cpu_base,
                                     RING, TMASK)]
        resident_done = None
        if cfg["pull"] == "cpp":
            workers.append(rt.start_pull_thread(reg.cpu(OFF_DESC), reg.dev(OFF_READY), hidden,
                                                staging, start, d, args.cpu_base + 1, RING))
        elif cfg["pull"].startswith("resident"):
            with torch.cuda.stream(resident_stream):
                rt.resident_pull(reg.dev(OFF_DESC), reg.dev(OFF_READY), reg.stop_dev, ticket, hidden,
                                 staging, start, t_pull, int(cfg["pull"][8:]), 512, RING, TMASK)
                resident_done = torch.cuda.Event()
                resident_done.record(resident_stream)
        # globaltimer drifts against CLOCK_MONOTONIC; calibrate around every config.
        off0, c0 = rt.clock_offset(*reg.calib_slot(0), 100), mono()
        barrier.wait(TIMEOUT_S)

        lo = start
        desc_next = start
        ready_q: list[int] = []
        inflight = collections.deque()
        timing = []
        batch_idx = 0
        window = np.arange(512, dtype=np.int64)
        while not reg.ctrl[0]:
            if cfg["pull"] == "python":
                while int(reg.desc[desc_next % RING, 0]) == desc_next & 0xFFFFFFFF:
                    slot = desc_next % RING
                    nbytes = int(reg.desc[slot, 2])
                    src_off, dst_off = int(reg.desc64[slot, 2]), int(reg.desc64[slot, 3])
                    with torch.cuda.stream(comm):
                        ext.memcpy_async(staging + dst_off, hidden + src_off, nbytes)
                    drv.write(comm, reg.dev(OFF_READY) + 4 * slot, desc_next)
                    desc_next += 1
            seqs = lo + window
            hits = (reg.ready[seqs % RING] == (seqs & 0xFFFFFFFF)) & (seen_at[seqs & TMASK] == 0)
            if hits.any():
                new = seqs[hits]
                seen_at[new & TMASK] = mono()
                ready_q.extend(new.tolist())
                while seen_at[lo & TMASK]:
                    lo += 1
            while inflight and inflight[0][1].query():
                _, _, batch = inflight.popleft()
                if cfg["push"] == "python":
                    for i, seq in enumerate(batch):
                        slot = seq % RING
                        with torch.cuda.stream(comm):
                            ext.memcpy_async(cand_dst + slot * 32, cand_src + i * 32, 32)
                        drv.write(comm, reg.dev(OFF_COMPLETE) + 4 * slot, seq)
            if ready_q and len(inflight) < args.max_inflight:
                batch, ready_q = ready_q[:MAX_PUSH], ready_q[MAX_PUSH:]
                idx = np.asarray(batch, dtype=np.int64) & TMASK
                launch_at[idx] = mono()
                batch_of[idx] = batch_idx
                busy_wait_ms(args.launch_host_ms)
                ev_s, ev_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(compute):
                    ev_s.record(compute)
                    graph.replay()
                    ev_e.record(compute)
                    rt.stamp(t_end, batch_idx)
                    if cfg["push"] == "queued":
                        rt.push_scatter(cand_src, cand_dst, reg.dev(OFF_COMPLETE), batch, RING)
                    done = torch.cuda.Event()
                    done.record(compute)
                inflight.append((batch_idx, done, batch))
                timing.append((ev_s, ev_e, len(batch)))
                batch_idx += 1

        if resident_done is not None:
            t0 = time.perf_counter()
            while not resident_done.query():
                if time.perf_counter() - t0 > 30:
                    print(json.dumps({"fatal": "resident kernel did not stop"}), flush=True)
                    os._exit(3)
        torch.cuda.synchronize()
        for w in workers:
            rt.stop_worker(w)
        off1, c1 = rt.clock_offset(*reg.calib_slot(0), 100), mono()
        prop_ms = [s.elapsed_time(e) for s, e, _ in timing]
        g = t_end[:batch_idx].cpu().numpy()
        ends = g + off0 + (off1 - off0) * ((g + off0 - c0) / max(1, c1 - c0))
        q_out.put({
            "obs_ready": obs_ready.copy(), "seen": seen_at.copy(), "launch": launch_at.copy(),
            "batch_of": batch_of.copy(), "prop_end": ends,
            "prop_ms": prop_ms, "batch_sizes": [n for _, _, n in timing],
            "clock_drift_us": round((off1 - off0) / 1e3, 1),
        })
        barrier.wait(TIMEOUT_S)


def _q(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {}
    return {"p50": round(float(np.percentile(x, 50)), 3), "p90": round(float(np.percentile(x, 90)), 3),
            "p99": round(float(np.percentile(x, 99)), 3), "mean": round(float(x.mean()), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=0)
    ap.add_argument("--draft", type=int, default=1)
    ap.add_argument("--pulls", nargs="+",
                    default=["python", "cpp", "resident1", "resident2", "resident4", "resident8"])
    ap.add_argument("--pushes", nargs="+", default=["python", "queued"])
    ap.add_argument("--requests", type=int, default=48)
    ap.add_argument("--wave-ms", type=float, default=12.0)
    ap.add_argument("--proposal-ms", type=float, default=3.6)
    ap.add_argument("--launch-host-ms", type=float, default=2.5)
    ap.add_argument("--max-inflight", type=int, default=2)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--no-target-load", dest="target_load", action="store_false")
    ap.add_argument("--cpu-base", type=int, default=40, help="first CPU for C++ helper threads")
    ap.add_argument("--build-dir", default=os.path.join(os.path.dirname(__file__), ".build"))
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    configs = build_configs(args)
    t = args.target
    torch.cuda.set_device(t)
    ext = load_ext(args.build_dir)
    rt = load_runtime(args.build_dir)
    drv = Driver()
    region_path = f"/dev/shm/dt-sm-comm-a-{os.getpid()}"
    reg = Region(region_path, drv, create=True)
    reg.reset()
    hidden = ext.dev_alloc(args.requests * MAX_REQ_BYTES)
    cand = ext.dev_alloc(RING * 32)
    mono = rt.monotonic_ns

    ctx = mp.get_context("spawn")
    q_in, q_out = ctx.Queue(), ctx.Queue()
    barrier = ctx.Barrier(2)
    child = ctx.Process(target=draft_main, args=(args, configs, q_in, q_out, barrier, region_path))
    child.start()
    q_in.put((ext.ipc_get(hidden), ext.ipc_get(cand)))
    hello = q_out.get(timeout=900)
    load_stop = ctx.Event()
    loader = None
    if args.target_load:
        loader = ctx.Process(target=_target_load, args=(t, load_stop), daemon=True)
        loader.start()
        time.sleep(5)
    out = open(args.out, "a") if args.out else None
    _emit({"header": True, **{k: v for k, v in vars(args).items() if k not in ("pulls", "pushes")},
           **hello}, out)

    rng = np.random.default_rng(17)
    probs = ROW_WEIGHTS / ROW_WEIGHTS.sum()
    t_pub = np.zeros(TMASK + 1, dtype=np.int64)
    t_seen = np.zeros(TMASK + 1, dtype=np.int64)
    obs_complete = np.zeros(TMASK + 1, dtype=np.int64)

    for cfg in configs:
        reg.reset()
        for arr in (t_pub, t_seen, obs_complete):
            arr[:] = 0
        barrier.wait(TIMEOUT_S)
        observer = rt.start_observer(reg.cpu(OFF_COMPLETE), obs_complete.ctypes.data,
                                     args.cpu_base + 2, RING, TMASK)
        barrier.wait(TIMEOUT_S)

        start = cfg["start_seq"]
        next_seq = start
        outstanding: dict[int, int] = {}
        to_publish = list(range(args.requests))
        completed: list[int] = []
        wave = 0
        t0 = mono()
        next_wave = t0
        end = t0 + int(args.duration * 1e9)
        while True:
            now = mono()
            if now >= end:
                break
            if now >= next_wave:
                for req in to_publish:
                    rows = int(rng.choice(MAX_ROWS, p=probs)) + 1
                    reg.publish(next_seq, wave, rows * ROW_BYTES, req * MAX_REQ_BYTES,
                                (next_seq % STAGING_SLOTS) * MAX_REQ_BYTES)
                    t_pub[next_seq & TMASK] = mono()
                    outstanding[next_seq] = req
                    next_seq += 1
                to_publish, completed = completed, []
                wave += 1
                next_wave += int(args.wave_ms * 1e6)
            if outstanding:
                keys = np.fromiter(outstanding.keys(), dtype=np.int64, count=len(outstanding))
                done = keys[reg.complete[keys % RING] == (keys & 0xFFFFFFFF)]
                if done.size:
                    t_seen[done & TMASK] = mono()
                    for s in done.tolist():
                        completed.append(outstanding.pop(s))
        reg.ctrl[0] = 1
        dres = q_out.get(timeout=TIMEOUT_S)
        rt.stop_worker(observer)
        barrier.wait(TIMEOUT_S)
        _emit(_summarize(cfg, start, next_seq, t_pub, t_seen, obs_complete, dres, args), out)

    load_stop.set()
    if loader is not None:
        loader.join(30)
    child.join(60)
    os.unlink(region_path)


def _summarize(cfg, start, next_seq, t_pub, t_seen, obs_complete, d, args):
    seqs = np.arange(start, next_seq, dtype=np.int64)
    i = seqs & TMASK
    ok = (t_seen[i] > 0) & (d["seen"][i] > 0) & (d["launch"][i] > 0) & (d["batch_of"][i] >= 0)
    i = i[ok]
    pub, ready_obs, seen, launch = t_pub[i], d["obs_ready"][i], d["seen"][i], d["launch"][i]
    prop_end = d["prop_end"][d["batch_of"][i]]
    comp_obs, tseen = obs_complete[i], t_seen[i]
    ms = lambda a, b: (a - b) / 1e6  # noqa: E731
    prop = np.asarray(d["prop_ms"])
    return {
        **{k: v for k, v in cfg.items() if k != "start_seq"},
        "rounds": int(i.size),
        "rounds_per_s": round(i.size / args.duration, 1),
        "clock_drift_us": d["clock_drift_us"],
        "mean_batch": round(float(np.mean(d["batch_sizes"])), 2) if d["batch_sizes"] else 0,
        "proposal_gpu_ms": _q(prop),
        "ms": {
            "pub_to_input_ready": _q(ms(ready_obs, pub)),
            "input_ready_to_draft_cpu": _q(ms(seen, ready_obs)),
            "draft_cpu_to_launch": _q(ms(launch, seen)),
            "launch_to_proposal_end": _q(ms(prop_end, launch)),
            "proposal_end_to_ready": _q(ms(comp_obs, prop_end)),
            "ready_to_target_cpu": _q(ms(tseen, comp_obs)),
            "pub_to_target_cpu": _q(ms(tseen, pub)),
            "pub_to_target_cpu_minus_draft_work": _q(ms(tseen, pub) - args.launch_host_ms - prop[
                d["batch_of"][i]]),
        },
    }


def _emit(rec, out) -> None:
    line = json.dumps(rec)
    print(line, flush=True)
    if out:
        out.write(line + "\n")
        out.flush()


if __name__ == "__main__":
    main()
