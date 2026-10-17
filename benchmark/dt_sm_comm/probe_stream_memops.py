"""Probe cuStreamWaitValue32 / cuStreamWriteValue32 support on this host.

The DT V2 notification bridge failed with "operation not supported" while
passing CU_STREAM_WAIT_VALUE_FLUSH. This probe separates the flag from the
memory type: local device memory, pinned host memory, a /dev/shm mapping
registered with cuMemHostRegister (the cross-process host flag case), and
peer device memory.

For every supported combination it also measures release latency: CPU (or a
peer stream) writes the flag, the waiting stream records an event, CPU polls
the event.

Run: python-cu130-compat probe_stream_memops.py [--device 0 --peer 1]
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import mmap
import os
import statistics
import time

import torch

CU_STREAM_WAIT_VALUE_GEQ = 0x0
CU_STREAM_WAIT_VALUE_EQ = 0x1
CU_STREAM_WAIT_VALUE_FLUSH = 1 << 30
CU_STREAM_WRITE_VALUE_DEFAULT = 0x0
CU_MEMHOSTREGISTER_PORTABLE = 0x01
CU_MEMHOSTREGISTER_DEVICEMAP = 0x02

ATTRS = {
    "CAN_USE_STREAM_MEM_OPS_V1": 92,
    "CAN_USE_64_BIT_STREAM_MEM_OPS_V1": 93,
    "CAN_USE_STREAM_WAIT_VALUE_NOR_V1": 94,
    "CAN_FLUSH_REMOTE_WRITES": 98,
    "CAN_USE_64_BIT_STREAM_MEM_OPS": 122,
    "CAN_USE_STREAM_WAIT_VALUE_NOR": 123,
}


class Driver:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL(ctypes.util.find_library("cuda") or "libcuda.so.1")
        self.wait = getattr(self.lib, "cuStreamWaitValue32_v2")
        self.write = getattr(self.lib, "cuStreamWriteValue32_v2")
        for fn in (self.wait, self.write):
            fn.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_uint]
            fn.restype = ctypes.c_int
        self.lib.cuDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int
        ]
        self.lib.cuMemHostRegister_v2.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        self.lib.cuMemHostUnregister.argtypes = [ctypes.c_void_p]
        self.lib.cuMemHostGetDevicePointer_v2.argtypes = [
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_void_p, ctypes.c_uint
        ]
        self.lib.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self.lib.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]

    def err(self, rc: int) -> str:
        if rc == 0:
            return "ok"
        msg = ctypes.c_char_p()
        self.lib.cuGetErrorString(rc, ctypes.byref(msg))
        return f"{rc}:{(msg.value or b'?').decode()}"

    def attr(self, name: str, device: int) -> int:
        out = ctypes.c_int()
        rc = self.lib.cuDeviceGetAttribute(ctypes.byref(out), ATTRS[name], device)
        return out.value if rc == 0 else -rc


def _wait_released(event: torch.cuda.Event, timeout_s: float) -> float | None:
    start = time.perf_counter()
    while not event.query():
        if time.perf_counter() - start > timeout_s:
            return None
    return time.perf_counter() - start


def probe_case(drv, name, addr, set_flag, reset_flag, flags, stream, iters):
    """Return support status and CPU-observed release latency (us)."""
    result = {"case": name, "flags": hex(flags)}
    reset_flag()
    torch.cuda.synchronize()
    rc = drv.wait(ctypes.c_void_p(stream.cuda_stream), addr, 1, flags)
    if rc != 0:
        result["wait"] = drv.err(rc)
        return result
    ev = torch.cuda.Event()
    ev.record(stream)
    time.sleep(0.05)
    blocked = not ev.query()
    set_flag()
    first = _wait_released(ev, 5.0)
    result["wait"] = "ok"
    result["blocked_before_set"] = blocked
    if first is None:
        result["released"] = False
        return result
    result["released"] = True
    lat = []
    for _ in range(iters):
        reset_flag()
        torch.cuda.synchronize()
        assert drv.wait(ctypes.c_void_p(stream.cuda_stream), addr, 1, flags) == 0
        ev = torch.cuda.Event()
        ev.record(stream)
        time.sleep(0.0005)
        t0 = time.perf_counter()
        set_flag()
        while not ev.query():
            pass
        lat.append((time.perf_counter() - t0) * 1e6)
    lat.sort()
    result["release_us_p50"] = round(statistics.median(lat), 1)
    result["release_us_p99"] = round(lat[int(0.99 * (len(lat) - 1))], 1)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--peer", type=int, default=1)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    drv = Driver()
    torch.cuda.set_device(args.device)
    torch.cuda.init()
    ver = ctypes.c_int()
    drv.lib.cuDriverGetVersion(ctypes.byref(ver))
    report = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "driver_api_version": ver.value,
        "attrs": {k: drv.attr(k, args.device) for k in ATTRS},
        "cases": [],
    }
    stream = torch.cuda.Stream(priority=-1)
    writer = torch.cuda.Stream()

    def device_flag_case(tensor, label, writer_stream):
        addr = tensor.data_ptr()

        def set_flag():
            rc = drv.write(ctypes.c_void_p(writer_stream.cuda_stream), addr, 1,
                           CU_STREAM_WRITE_VALUE_DEFAULT)
            assert rc == 0, drv.err(rc)

        def reset_flag():
            tensor.zero_()

        return label, addr, set_flag, reset_flag

    cases = []
    local = torch.zeros(1, dtype=torch.int32, device=f"cuda:{args.device}")
    cases.append(device_flag_case(local, "device_local", writer))

    pinned = torch.zeros(1, dtype=torch.int32, pin_memory=True)
    pinned_np = pinned.numpy()

    def pinned_set():
        pinned_np[0] = 1

    def pinned_reset():
        pinned_np[0] = 0

    cases.append(("host_pinned_cudaHostAlloc", pinned.data_ptr(), pinned_set, pinned_reset))

    shm_path = f"/dev/shm/dt-sm-comm-probe-{os.getpid()}"
    fd = os.open(shm_path, os.O_CREAT | os.O_RDWR, 0o600)
    os.ftruncate(fd, 4096)
    mm = mmap.mmap(fd, 4096)
    host_ptr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
    rc = drv.lib.cuMemHostRegister_v2(
        ctypes.c_void_p(host_ptr), 4096,
        CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP,
    )
    report["shm_register"] = drv.err(rc)
    shm_view = (ctypes.c_uint32 * 1).from_buffer(mm)
    if rc == 0:
        dptr = ctypes.c_uint64()
        drv.lib.cuMemHostGetDevicePointer_v2(ctypes.byref(dptr), ctypes.c_void_p(host_ptr), 0)

        def shm_set():
            shm_view[0] = 1

        def shm_reset():
            shm_view[0] = 0

        cases.append(("host_shm_cuMemHostRegister", dptr.value, shm_set, shm_reset))

    if torch.cuda.device_count() > args.peer and torch.cuda.can_device_access_peer(
        args.device, args.peer
    ):
        with torch.cuda.device(args.peer):
            peer = torch.zeros(1, dtype=torch.int32, device=f"cuda:{args.peer}")
            peer_writer = torch.cuda.Stream(device=args.peer)
        # Touch peer memory from args.device to force peer access enablement.
        torch.cuda.synchronize()
        _ = local.copy_(peer)
        cases.append(device_flag_case(peer, "device_peer", peer_writer))

    for label, addr, set_flag, reset_flag in cases:
        for flags in (CU_STREAM_WAIT_VALUE_EQ, CU_STREAM_WAIT_VALUE_EQ | CU_STREAM_WAIT_VALUE_FLUSH):
            report["cases"].append(
                probe_case(drv, label, addr, set_flag, reset_flag, flags, stream, args.iters)
            )
            torch.cuda.synchronize()

    del shm_view
    drv.lib.cuMemHostUnregister(ctypes.c_void_p(host_ptr))
    try:
        mm.close()
    except BufferError:
        pass
    os.close(fd)
    os.unlink(shm_path)

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
