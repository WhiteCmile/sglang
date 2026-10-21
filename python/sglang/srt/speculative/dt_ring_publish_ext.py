"""CUDA extension loader for the §7.1 GPU ring publish kernel (DT_RING_PUBLISH=1).

Builds ``csrc/dt_ring_publish.cu`` with nvcc at first use and loads it with
ctypes — the same pattern as ``_build_worker_library`` for
``csrc/dt_ring_worker.cpp`` in ``dt_direct_ring.py``.  The .so uses the CUDA
runtime API on the primary context, so kernels launch onto torch streams by
raw stream handle.

The kernel fills one 4096-byte ring record per verify wave and publishes the
seq word last with a system-scope release; see the .cu file for the layout.
The scheduler's only per-wave work is a credit check, a few pinned stores
into the device-mapped staging buffer allocated here, one event-handle
pre-write, and this launch.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

_NVCC_CANDIDATES = (
    "/usr/local/cuda-12.4/bin/nvcc",
    "/usr/local/cuda/bin/nvcc",
    "nvcc",
)


def _nvcc() -> str:
    env = os.environ.get("NVCC")
    if env:
        return env
    for candidate in _NVCC_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return "nvcc"


def _sm_arch() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return f"{major}{minor}"
    except Exception:
        pass
    return "90"


def _build_library() -> Path:
    """Build the publish kernel through the shared speculative/csrc path.

    Runtime compilation with the module's pinned toolchain is the supported
    build path for this directory (the dt_ring_worker.cpp pattern); the
    cache identity includes the nvcc binary, its version, the target arch,
    and every flag (see _build_native_library).
    """

    from sglang.srt.speculative.dt_direct_ring import _build_native_library

    source = Path(__file__).with_name("csrc") / "dt_ring_publish.cu"
    nvcc = _nvcc()
    cuda_home = str(Path(nvcc).resolve().parent.parent)
    arch = _sm_arch()
    return _build_native_library(
        "dt_ring_publish",
        source,
        [
            nvcc,
            "-O3",
            "-std=c++17",
            "-shared",
            "-Xcompiler",
            "-fPIC",
            f"-gencode=arch=compute_{arch},code=sm_{arch}",
            f"-L{cuda_home}/lib64",
            "-lcudart",
        ],
        ["--version"],
        "DT_RING_PUBLISH_BUILD_DIR",
    )


class RingPublishExt:
    """ctypes owner of the publish kernel and its staging buffer."""

    def __init__(self, *, device: int) -> None:
        lib = ctypes.CDLL(str(_build_library()))
        i64, i32 = ctypes.c_int64, ctypes.c_int32
        lib.rp_init.argtypes = [i32]
        lib.rp_init.restype = i32
        lib.rp_staging_alloc.argtypes = [i64, ctypes.POINTER(i32)]
        lib.rp_staging_alloc.restype = i64
        lib.rp_dev_ptr.argtypes = [i64, ctypes.POINTER(i32)]
        lib.rp_dev_ptr.restype = i64
        lib.rp_staging_free.argtypes = [i64]
        lib.rp_staging_free.restype = i32
        lib.rp_publish.argtypes = [i64] * 19
        lib.rp_publish.restype = i32
        self._lib = lib
        rc = lib.rp_init(device)
        if rc:
            raise RuntimeError(f"cudaSetDevice({device}) rc={rc}")

    @staticmethod
    def _check(rc: int, what: str) -> None:
        if rc:
            raise RuntimeError(f"{what} rc={rc}")

    def staging_alloc(self, nbytes: int) -> tuple[int, int]:
        """Return (host pointer, device pointer) of a zeroed mapped buffer."""

        rc = ctypes.c_int32()
        host = int(self._lib.rp_staging_alloc(nbytes, ctypes.byref(rc)))
        self._check(rc.value, "cudaHostAlloc")
        dev = int(self._lib.rp_dev_ptr(host, ctypes.byref(rc)))
        self._check(rc.value, "cudaHostGetDevicePointer")
        return host, dev

    def staging_free(self, host: int) -> None:
        self._check(self._lib.rp_staging_free(host), "cudaFreeHost")

    def publish(
        self,
        *,
        stream: int,
        ring_dev: int,
        capacity: int,
        seq: int,
        source_slot_id: int,
        slot_generation: int,
        source_byte_offset: int,
        nbytes: int,
        row_count: int,
        row_bytes: int,
        request_count: int,
        block_size: int,
        staging_dev: int,
        staging_entries_offset: int,
        staging_active_offset: int,
        accept_dev: int,
        bonus_dev: int,
        seq_lens_dev: int,
        offset_stride: int = 0,
    ) -> None:
        self._check(
            self._lib.rp_publish(
                stream,
                ring_dev,
                capacity,
                seq,
                source_slot_id,
                slot_generation,
                source_byte_offset,
                nbytes,
                row_count,
                row_bytes,
                request_count,
                block_size,
                offset_stride,
                staging_dev,
                staging_entries_offset,
                staging_active_offset,
                accept_dev,
                bonus_dev,
                seq_lens_dev,
            ),
            "rp_publish",
        )


_EXT_CACHE: dict[int, RingPublishExt] = {}


def load_ext(device: int) -> RingPublishExt:
    ext = _EXT_CACHE.get(device)
    if ext is None:
        ext = RingPublishExt(device=device)
        _EXT_CACHE[device] = ext
    return ext
