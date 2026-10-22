"""Same-host CUDA IPC primitives for DT Draft KV migration.

The transport deliberately mirrors Mooncake's IntraNodeNvlinkTransport:
publish the real CUDA allocation base and an offset for each registered
suballocation, cache opened IPC mappings, and enqueue one coalesced batch on a
dedicated stream.  It contains no DT protocol or request-lifecycle policy.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from threading import Lock

CUDA_IPC_HANDLE_BYTES = 64
CUDA_IPC_LAZY_ENABLE_PEER_ACCESS = 1
CUDA_MEMCPY_DEFAULT = 4
CU_STREAM_WAIT_VALUE_EQ = 0x1
CU_STREAM_WAIT_VALUE_FLUSH = 1 << 30
CU_MEMHOSTREGISTER_PORTABLE = 0x01
CU_MEMHOSTREGISTER_DEVICEMAP = 0x02


class CudaIpcError(RuntimeError):
    pass


class _CudaIpcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * CUDA_IPC_HANDLE_BYTES)]


@dataclass(frozen=True, slots=True)
class AllocationRegistration:
    base_address: int
    length: int
    ipc_handle: bytes


@dataclass(frozen=True, slots=True)
class BufferRegistration:
    allocation_index: int
    allocation_offset: int
    address: int
    length: int
    item_length: int


@dataclass(frozen=True, slots=True)
class CopySegment:
    source_address: int
    destination_address: int
    length: int


class CudaIpcRuntime:
    """Thin checked binding around the CUDA IPC and batch-copy runtime APIs."""

    def __init__(self) -> None:
        driver_name = ctypes.util.find_library("cuda") or "libcuda.so.1"
        runtime_name = os.environ.get("DT_CUDA_RUNTIME_LIBRARY") or (
            ctypes.util.find_library("cudart") or "libcudart.so"
        )
        try:
            self._driver = ctypes.CDLL(driver_name)
            self._runtime = ctypes.CDLL(runtime_name)
        except OSError as exc:
            raise CudaIpcError(f"CUDA IPC libraries are unavailable: {exc}") from exc

        get_range = getattr(
            self._driver,
            "cuMemGetAddressRange_v2",
            getattr(self._driver, "cuMemGetAddressRange", None),
        )
        if get_range is None:
            raise CudaIpcError("cuMemGetAddressRange is unavailable")
        self._get_address_range = get_range
        get_range.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_uint64,
        ]
        get_range.restype = ctypes.c_int

        self._driver.cuGetErrorString.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self._driver.cuGetErrorString.restype = ctypes.c_int

        # Stream memory operations are Driver APIs. PyTorch's cudaStream_t is
        # ABI-compatible with CUstream in the same CUDA context.
        self._stream_wait_value32 = getattr(
            self._driver,
            "cuStreamWaitValue32_v2",
            getattr(self._driver, "cuStreamWaitValue32", None),
        )
        self._stream_write_value32 = getattr(
            self._driver,
            "cuStreamWriteValue32_v2",
            getattr(self._driver, "cuStreamWriteValue32", None),
        )
        if self._stream_wait_value32 is not None:
            self._stream_wait_value32.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_uint32,
                ctypes.c_uint,
            ]
            self._stream_wait_value32.restype = ctypes.c_int
        if self._stream_write_value32 is not None:
            self._stream_write_value32.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_uint32,
                ctypes.c_uint,
            ]
            self._stream_write_value32.restype = ctypes.c_int

        self._runtime.cudaGetErrorString.argtypes = [ctypes.c_int]
        self._runtime.cudaGetErrorString.restype = ctypes.c_char_p
        self._runtime.cudaRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self._runtime.cudaRuntimeGetVersion.restype = ctypes.c_int
        runtime_version = ctypes.c_int()
        self._check_runtime(
            self._runtime.cudaRuntimeGetVersion(ctypes.byref(runtime_version)),
            "cudaRuntimeGetVersion",
        )
        self.runtime_version = runtime_version.value
        self._runtime.cudaIpcGetMemHandle.argtypes = [
            ctypes.POINTER(_CudaIpcMemHandle),
            ctypes.c_void_p,
        ]
        self._runtime.cudaIpcGetMemHandle.restype = ctypes.c_int
        self._runtime.cudaIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            _CudaIpcMemHandle,
            ctypes.c_uint,
        ]
        self._runtime.cudaIpcOpenMemHandle.restype = ctypes.c_int
        self._runtime.cudaIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        self._runtime.cudaIpcCloseMemHandle.restype = ctypes.c_int

        self._runtime.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._runtime.cudaMemcpyAsync.restype = ctypes.c_int

    def _check_runtime(self, code: int, operation: str) -> None:
        if code == 0:
            return
        detail = self._runtime.cudaGetErrorString(code)
        message = detail.decode(errors="replace") if detail else f"error {code}"
        raise CudaIpcError(f"{operation} failed: {message}")

    def _check_driver(self, code: int, operation: str) -> None:
        if code == 0:
            return
        detail = ctypes.c_char_p()
        decoded = self._driver.cuGetErrorString(code, ctypes.byref(detail))
        message = (
            detail.value.decode(errors="replace")
            if decoded == 0 and detail.value
            else f"CUresult {code}"
        )
        raise CudaIpcError(f"{operation} failed: {message}")

    def register_allocation(
        self, address: int, visible_length: int
    ) -> AllocationRegistration:
        if address <= 0 or visible_length <= 0:
            raise ValueError("CUDA buffer address and length must be positive")
        base = ctypes.c_uint64()
        length = ctypes.c_size_t()
        code = self._get_address_range(
            ctypes.byref(base), ctypes.byref(length), ctypes.c_uint64(address)
        )
        if code != 0:
            raise CudaIpcError(f"cuMemGetAddressRange failed with CUresult {code}")
        if address < base.value or address + visible_length > base.value + length.value:
            raise CudaIpcError("CUDA buffer lies outside its reported allocation")
        handle = _CudaIpcMemHandle()
        self._check_runtime(
            self._runtime.cudaIpcGetMemHandle(
                ctypes.byref(handle), ctypes.c_void_p(base.value)
            ),
            "cudaIpcGetMemHandle",
        )
        return AllocationRegistration(
            base_address=base.value,
            length=length.value,
            ipc_handle=bytes(handle),
        )

    def open_allocation(self, ipc_handle: bytes) -> int:
        handle = self._decode_handle(ipc_handle)
        mapped = ctypes.c_void_p()
        self._check_runtime(
            self._runtime.cudaIpcOpenMemHandle(
                ctypes.byref(mapped), handle, CUDA_IPC_LAZY_ENABLE_PEER_ACCESS
            ),
            "cudaIpcOpenMemHandle",
        )
        if not mapped.value:
            raise CudaIpcError("cudaIpcOpenMemHandle returned a null mapping")
        return int(mapped.value)

    def close_allocation(self, mapped_address: int) -> None:
        self._check_runtime(
            self._runtime.cudaIpcCloseMemHandle(ctypes.c_void_p(mapped_address)),
            "cudaIpcCloseMemHandle",
        )

    def copy_batch_async(self, segments: Sequence[CopySegment], stream: int) -> None:
        # CUDA 13's batched copy rejected peer IPC mappings on the validated
        # H800 forward-compat stack.  Enqueue the bounded segments on the same
        # dedicated stream instead.  The following publication/event remains
        # ordered after every copy, so transport and lifecycle semantics do not
        # change and this also works with the CUDA 12.4 prototype runtime.
        if not segments:
            return
        if stream == 0:
            raise ValueError("DT P2P copy must use a non-default CUDA stream")
        for index, segment in enumerate(segments):
            code = self._runtime.cudaMemcpyAsync(
                ctypes.c_void_p(segment.destination_address),
                ctypes.c_void_p(segment.source_address),
                ctypes.c_size_t(segment.length),
                ctypes.c_int(CUDA_MEMCPY_DEFAULT),
                ctypes.c_void_p(stream),
            )
            if code == 0:
                continue
            # Earlier segments are already queued: drain them so callers may
            # treat the failure as "no copy references these buffers".
            fenced = index == 0 or self._stream_synchronize(stream) == 0
            try:
                self._check_runtime(code, "cudaMemcpyAsync")
            except CudaIpcError as exc:
                exc.resources_quiescent = fenced
                raise

    def _stream_synchronize(self, stream: int) -> int:
        synchronize = self._runtime.cudaStreamSynchronize
        synchronize.argtypes = [ctypes.c_void_p]
        synchronize.restype = ctypes.c_int
        return int(synchronize(ctypes.c_void_p(stream)))

    def stream_synchronize(self, stream: int) -> None:
        """Block until one CUDA stream drains."""

        self._check_runtime(
            self._stream_synchronize(stream), "cudaStreamSynchronize"
        )

    def stream_wait_value32(self, *, address: int, value: int, stream: int) -> None:
        """Queue an acquire-style equality wait without blocking the CPU."""

        if self._stream_wait_value32 is None:
            raise CudaIpcError("cuStreamWaitValue32 is unavailable")
        if address <= 0 or stream == 0 or not 0 <= value <= 0xFFFFFFFF:
            raise ValueError("stream wait requires a valid address, stream, and uint32")
        self._check_driver(
            self._stream_wait_value32(
                ctypes.c_void_p(stream),
                ctypes.c_uint64(address),
                ctypes.c_uint32(value),
                ctypes.c_uint(
                    CU_STREAM_WAIT_VALUE_EQ | CU_STREAM_WAIT_VALUE_FLUSH
                ),
            ),
            "cuStreamWaitValue32",
        )

    def stream_write_value32(self, *, address: int, value: int, stream: int) -> None:
        """Queue one device-memory publication word on an existing CUDA stream."""

        if self._stream_write_value32 is None:
            raise CudaIpcError("cuStreamWriteValue32 is unavailable")
        if address <= 0 or not 0 <= value <= 0xFFFFFFFF:
            raise ValueError("stream write requires a valid address and uint32")
        self._check_driver(
            self._stream_write_value32(
                ctypes.c_void_p(stream),
                ctypes.c_uint64(address),
                ctypes.c_uint32(value),
                ctypes.c_uint(0),
            ),
            "cuStreamWriteValue32",
        )

    def host_register(self, address: int, length: int) -> int:
        """Page-lock a host mapping for every context; return its device address."""

        if address <= 0 or length <= 0:
            raise ValueError("host registration requires a valid address and length")
        register = self._driver.cuMemHostRegister_v2
        register.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        register.restype = ctypes.c_int
        self._check_driver(
            register(
                ctypes.c_void_p(address),
                ctypes.c_size_t(length),
                ctypes.c_uint(CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP),
            ),
            "cuMemHostRegister",
        )
        device_pointer = ctypes.c_uint64()
        get_pointer = self._driver.cuMemHostGetDevicePointer_v2
        get_pointer.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        get_pointer.restype = ctypes.c_int
        try:
            self._check_driver(
                get_pointer(ctypes.byref(device_pointer), ctypes.c_void_p(address), 0),
                "cuMemHostGetDevicePointer",
            )
        except Exception:
            self.host_unregister(address)
            raise
        return int(device_pointer.value)

    def host_unregister(self, address: int) -> None:
        unregister = self._driver.cuMemHostUnregister
        unregister.argtypes = [ctypes.c_void_p]
        unregister.restype = ctypes.c_int
        self._check_driver(unregister(ctypes.c_void_p(address)), "cuMemHostUnregister")

    @staticmethod
    def _decode_handle(value: bytes) -> _CudaIpcMemHandle:
        if len(value) != CUDA_IPC_HANDLE_BYTES:
            raise ValueError(f"CUDA IPC handle must be {CUDA_IPC_HANDLE_BYTES} bytes")
        return _CudaIpcMemHandle.from_buffer_copy(value)


class CudaIpcMappingCache:
    """Generation-fenced IPC mapping cache for warm migration reuse."""

    def __init__(self, runtime: CudaIpcRuntime) -> None:
        self._runtime = runtime
        self._mapped: dict[tuple[str, str, str, bytes], int] = {}
        self._latest_generation: dict[str, tuple[str, str]] = {}
        self._lock = Lock()

    def map(
        self,
        *,
        peer_instance_id: str,
        peer_session_id: str,
        pool_generation: str,
        ipc_handle: bytes,
    ) -> tuple[int, bool]:
        if not peer_instance_id or not peer_session_id or not pool_generation:
            raise ValueError("peer identity and pool generation must not be empty")
        with self._lock:
            generation = (peer_session_id, pool_generation)
            if self._latest_generation.get(peer_instance_id) != generation:
                self._invalidate_locked(peer_instance_id)
                self._latest_generation[peer_instance_id] = generation
            key = (
                peer_instance_id,
                peer_session_id,
                pool_generation,
                ipc_handle,
            )
            mapped = self._mapped.get(key)
            if mapped is not None:
                return mapped, True
            mapped = self._runtime.open_allocation(ipc_handle)
            self._mapped[key] = mapped
            return mapped, False

    def invalidate(self, peer_instance_id: str) -> None:
        with self._lock:
            self._invalidate_locked(peer_instance_id)
            self._latest_generation.pop(peer_instance_id, None)

    def close(self) -> None:
        with self._lock:
            failures = []
            for mapped in tuple(self._mapped.values()):
                try:
                    self._runtime.close_allocation(mapped)
                except Exception as exc:  # noqa: BLE001 - preserve cleanup progress
                    failures.append(exc)
            self._mapped.clear()
            self._latest_generation.clear()
        if failures:
            raise CudaIpcError(f"failed to close {len(failures)} CUDA IPC mappings")

    def _invalidate_locked(self, peer_instance_id: str) -> None:
        keys = [key for key in self._mapped if key[0] == peer_instance_id]
        failures = []
        for key in keys:
            mapped = self._mapped.pop(key)
            try:
                self._runtime.close_allocation(mapped)
            except Exception as exc:  # noqa: BLE001 - preserve cleanup progress
                failures.append(exc)
        if failures:
            raise CudaIpcError(
                f"failed to invalidate {len(failures)} CUDA IPC mappings for "
                f"{peer_instance_id}"
            )


def register_buffers(
    runtime: CudaIpcRuntime,
    buffers: Iterable[tuple[int, int, int]],
) -> tuple[tuple[AllocationRegistration, ...], tuple[BufferRegistration, ...]]:
    """Register ``(address, length, item_length)`` buffers once per allocation."""
    allocations: list[AllocationRegistration] = []
    allocation_indices: dict[tuple[int, int, bytes], int] = {}
    registered_buffers: list[BufferRegistration] = []
    for address, length, item_length in buffers:
        if item_length <= 0 or length <= 0 or length % item_length:
            raise ValueError("CUDA buffer length must be item-aligned")
        allocation = runtime.register_allocation(address, length)
        allocation_key = (
            allocation.base_address,
            allocation.length,
            allocation.ipc_handle,
        )
        allocation_index = allocation_indices.get(allocation_key)
        if allocation_index is None:
            allocation_index = len(allocations)
            allocation_indices[allocation_key] = allocation_index
            allocations.append(allocation)
        registered_buffers.append(
            BufferRegistration(
                allocation_index=allocation_index,
                allocation_offset=address - allocation.base_address,
                address=address,
                length=length,
                item_length=item_length,
            )
        )
    return tuple(allocations), tuple(registered_buffers)


def coalesce_page_runs(
    source_indices: Sequence[int], destination_indices: Sequence[int]
) -> tuple[tuple[int, int, int], ...]:
    """Return ``(src_start, dst_start, page_count)`` pairwise-contiguous runs."""
    if len(source_indices) != len(destination_indices):
        raise ValueError("source and destination page counts differ")
    if not source_indices:
        return ()
    runs: list[tuple[int, int, int]] = []
    src_start = int(source_indices[0])
    dst_start = int(destination_indices[0])
    run_length = 1
    previous_src = src_start
    previous_dst = dst_start
    for source, destination in zip(source_indices[1:], destination_indices[1:]):
        source = int(source)
        destination = int(destination)
        if source == previous_src + 1 and destination == previous_dst + 1:
            run_length += 1
        else:
            runs.append((src_start, dst_start, run_length))
            src_start, dst_start, run_length = source, destination, 1
        previous_src, previous_dst = source, destination
    runs.append((src_start, dst_start, run_length))
    return tuple(runs)


def build_copy_segments(
    *,
    mapped_source_bases: Sequence[int],
    source_buffers: Sequence[BufferRegistration],
    destination_buffers: Sequence[BufferRegistration],
    source_indices: Sequence[int],
    destination_indices: Sequence[int],
) -> tuple[CopySegment, ...]:
    """Expand coalesced page runs across corresponding K/V layer buffers."""
    if len(source_buffers) != len(destination_buffers):
        raise ValueError("source and destination KV buffer layouts differ")
    runs = coalesce_page_runs(source_indices, destination_indices)
    segments = []
    for source, destination in zip(source_buffers, destination_buffers):
        if source.item_length != destination.item_length:
            raise ValueError("source and destination KV item lengths differ")
        mapped_allocation = mapped_source_bases[source.allocation_index]
        source_buffer_base = mapped_allocation + source.allocation_offset
        for source_start, destination_start, page_count in runs:
            source_offset = source_start * source.item_length
            destination_offset = destination_start * destination.item_length
            length = page_count * source.item_length
            if source_offset < 0 or source_offset + length > source.length:
                raise ValueError("source page run exceeds the registered KV buffer")
            if (
                destination_offset < 0
                or destination_offset + length > destination.length
            ):
                raise ValueError(
                    "destination page run exceeds the registered KV buffer"
                )
            segments.append(
                CopySegment(
                    source_address=source_buffer_base + source_offset,
                    destination_address=destination.address + destination_offset,
                    length=length,
                )
            )
    return tuple(segments)
