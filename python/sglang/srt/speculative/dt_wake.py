"""ctypes owner of the Draft scheduler wake word (csrc/dt_wake.cpp, issue #274).

Design §4.4: the scheduler blocks in one native call with the GIL released
until a reason bit is set or its timeout passes.  The caller reads ``epoch()``
before it drains its queues and passes that value to ``wait``, so a ``set``
that lands while it drains makes the wait return at once.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

# Reason bits (§4.4).  READY is set by the native ingress (D6), CONTROL by
# the control thread (D5); DONE (2) belongs to the send loop (D8).
WAKE_READY = 1 << 0
WAKE_CONTROL = 1 << 2


def _build_wake_library() -> Path:
    from sglang.srt.speculative.dt_direct_ring import _build_native_library

    source = Path(__file__).with_name("csrc") / "dt_wake.cpp"
    return _build_native_library(
        "dt_wake",
        source,
        [os.environ.get("CXX", "g++"), "-O2", "-std=c++17", "-shared", "-fPIC"],
        ["--version"],
        "DT_WAKE_BUILD_DIR",
    )


class DraftWake:
    def __init__(self) -> None:
        lib = ctypes.CDLL(str(_build_wake_library()))
        u32, i64, vp = ctypes.c_uint32, ctypes.c_int64, ctypes.c_void_p
        lib.dt_wake_create.argtypes = []
        lib.dt_wake_create.restype = vp
        lib.dt_wake_destroy.argtypes = [vp]
        lib.dt_wake_destroy.restype = None
        lib.dt_wake_epoch.argtypes = [vp]
        lib.dt_wake_epoch.restype = u32
        lib.dt_wake_set.argtypes = [vp, u32]
        lib.dt_wake_set.restype = None
        lib.dt_wake_wait.argtypes = [vp, u32, i64]
        lib.dt_wake_wait.restype = u32
        self._lib = lib
        self._handle = lib.dt_wake_create()

    def epoch(self) -> int:
        return self._lib.dt_wake_epoch(self._handle)

    def set(self, reasons: int) -> None:
        self._lib.dt_wake_set(self._handle, reasons)

    def native_setter(self) -> tuple[int, int]:
        """(dt_wake_set address, wake handle) for a native thread that sets
        reason bits without Python (the ring worker's READY wake)."""

        return ctypes.cast(self._lib.dt_wake_set, ctypes.c_void_p).value, self._handle

    def wait(self, seen_epoch: int, timeout_s: float) -> int:
        """Return the reason bits taken (0 on timeout); the GIL is released."""

        return self._lib.dt_wake_wait(
            self._handle, seen_epoch, int(timeout_s * 1_000_000_000)
        )

    def close(self) -> None:
        # Only after every setter (the control thread) has stopped.
        if self._handle:
            self._lib.dt_wake_destroy(self._handle)
            self._handle = None
