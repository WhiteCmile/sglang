"""Profile-only absolute GPU timestamps for DT traces (DT_GPU_CLOCK=1).

``GpuClock.stamp_after(event)`` queues, on a private high-priority stream, a
wait on ``event`` and a one-thread kernel that writes ``%globaltimer`` into a
pinned host slot.  ``read`` converts it to CLOCK_MONOTONIC with an offset
re-calibrated every few seconds, so a CUDA boundary can be joined with host
timestamps without the host having to observe it.  The kernel is hand-written
PTX loaded through the driver API: no compiler is needed at runtime.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import time

import torch

_PTX = b"""
.version 7.8
.target sm_90
.address_size 64
.visible .entry dt_stamp(.param .u64 slot)
{
    .reg .u64 %rd<3>;
    ld.param.u64 %rd1, [slot];
    mov.u64 %rd2, %globaltimer;
    st.volatile.global.u64 [%rd1], %rd2;
    membar.sys;
    ret;
}
"""

SLOTS = 4096
RECALIBRATE_NS = 2_000_000_000


class GpuClock:
    def __init__(self, device: torch.device) -> None:
        self._driver = ctypes.CDLL(ctypes.util.find_library("cuda") or "libcuda.so.1")
        torch.cuda.set_device(device)
        torch.cuda.current_stream(device).synchronize()
        module = ctypes.c_void_p()
        self._check(self._driver.cuModuleLoadData(ctypes.byref(module), _PTX), "cuModuleLoadData")
        function = ctypes.c_void_p()
        self._check(
            self._driver.cuModuleGetFunction(ctypes.byref(function), module, b"dt_stamp"),
            "cuModuleGetFunction",
        )
        self._function = function
        self._driver.cuLaunchKernel.argtypes = [
            ctypes.c_void_p, *([ctypes.c_uint] * 6), ctypes.c_uint, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
        ]
        self.device = device
        self.stream = torch.cuda.Stream(device=device, priority=-1)
        self._calibration_stream = torch.cuda.Stream(device=device, priority=-1)
        self._slots = torch.zeros(SLOTS, dtype=torch.int64, pin_memory=True)
        self._calibration = torch.zeros(1, dtype=torch.int64, pin_memory=True)
        self._next = 0
        self._offset_ns = 0
        self._calibrated_ns = 0
        self._calibrate()

    def _check(self, rc: int, what: str) -> None:
        if rc != 0:
            raise RuntimeError(f"{what} failed: CUresult {rc}")

    def _launch(self, address: int, stream: torch.cuda.Stream) -> None:
        argument = ctypes.c_uint64(address)
        params = (ctypes.c_void_p * 1)(ctypes.cast(ctypes.byref(argument), ctypes.c_void_p))
        self._check(
            self._driver.cuLaunchKernel(
                self._function, 1, 1, 1, 1, 1, 1, 0, ctypes.c_void_p(stream.cuda_stream),
                params, None,
            ),
            "cuLaunchKernel",
        )

    def _calibrate(self) -> None:
        best = None
        for _ in range(8):
            self._calibration.zero_()
            before = time.monotonic_ns()
            self._launch(self._calibration.data_ptr(), self._calibration_stream)
            self._calibration_stream.synchronize()
            after = time.monotonic_ns()
            gpu = int(self._calibration[0])
            if best is None or after - before < best[0]:
                best = (after - before, (before + after) // 2 - gpu)
        self._offset_ns = best[1]
        self._calibrated_ns = time.monotonic_ns()

    def stamp_after(self, event: torch.cuda.Event) -> tuple[int, torch.cuda.Event]:
        """Return (slot, done_event) for a timestamp taken right after ``event``."""

        slot = self._next
        self._next = (slot + 1) % SLOTS
        self._slots[slot] = 0
        self.stream.wait_event(event)
        self._launch(self._slots.data_ptr() + 8 * slot, self.stream)
        done = torch.cuda.Event()
        done.record(self.stream)
        return slot, done

    def read(self, slot: int) -> int:
        """CLOCK_MONOTONIC ns of a completed stamp (0 if not written)."""

        if time.monotonic_ns() - self._calibrated_ns > RECALIBRATE_NS:
            self._calibrate()
        value = int(self._slots[slot])
        return value + self._offset_ns if value else 0
