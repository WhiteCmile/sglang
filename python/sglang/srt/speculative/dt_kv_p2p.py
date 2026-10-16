"""Mooncake-shaped same-host P2P transport for standalone DT Draft KV.

The lifecycle remains in ``dt_draft_engine``.  This module owns only stable KV
pool registration, generation-fenced CUDA IPC mappings, range planning, and a
bounded copy worker so the Draft scheduler never waits for transfer completion.
"""

from __future__ import annotations

import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock

import torch
from sglang.srt.speculative.dt_cuda_ipc import (
    BufferRegistration,
    CopySegment,
    CudaIpcMappingCache,
    CudaIpcRuntime,
    build_copy_segments,
    register_buffers,
)


@dataclass(frozen=True, slots=True)
class TransferMetrics:
    submitted_ns: int
    gpu_completed_ns: int
    mapping_ns: int
    driver_submit_ns: int
    gpu_copy_ns: int
    kv_bytes: int
    page_count: int
    segment_count: int
    cold_mapping_count: int


class TransferQuiescenceError(RuntimeError):
    """The copy stream could not prove that failed GPU work is complete."""

    resources_quiescent = False


class DraftKvP2pTransport:
    """Register one Draft KV pool and pull peer ranges on a copy stream."""

    def __init__(self, *, model_runner, device: torch.device) -> None:
        requested_device = torch.device(device)
        self.device_index = (
            torch.cuda.current_device()
            if requested_device.index is None
            else int(requested_device.index)
        )
        self.device = torch.device(requested_device.type, self.device_index)
        self._runtime = CudaIpcRuntime()
        self._mapping_cache = CudaIpcMappingCache(self._runtime)
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="dt-kv-p2p",
        )
        self._copy_stream = torch.cuda.Stream(device=device)
        self.pool_generation = uuid.uuid4().hex
        self.transport_session = uuid.uuid4().hex

        kv_cache = model_runner.token_to_kv_pool_allocator._kvcache
        pointers, lengths, item_lengths = kv_cache.get_contiguous_buf_infos()
        self._allocations, self._buffers = register_buffers(
            self._runtime,
            zip(pointers, lengths, item_lengths),
        )
        self._fatal_error: TransferQuiescenceError | None = None
        self._permanent_quarantine_error: RuntimeError | None = None
        self._closed = False
        self._lock = Lock()

    def pool_ref(self, *, instance_session_id: str, profile_id: str):
        from dt_protocol import (
            DraftKvAllocation,
            DraftKvBuffer,
            DraftKvPool,
        )

        properties = torch.cuda.get_device_properties(self.device)
        return DraftKvPool(
            transport="cuda-ipc",
            instance_session_id=instance_session_id,
            pool_generation=self.pool_generation,
            transport_session=self.transport_session,
            profile_id=profile_id,
            layout="NHD",
            page_size=1,
            device_id=self.device_index,
            device_uuid=str(properties.uuid),
            allocations=tuple(
                DraftKvAllocation(
                    base_address=allocation.base_address,
                    length=allocation.length,
                    ipc_handle=allocation.ipc_handle,
                )
                for allocation in self._allocations
            ),
            buffers=tuple(
                DraftKvBuffer(
                    allocation_index=buffer.allocation_index,
                    allocation_offset=buffer.allocation_offset,
                    visible_length=buffer.length,
                    item_length=buffer.item_length,
                )
                for buffer in self._buffers
            ),
        )

    def producer_event(self) -> tuple[torch.cuda.Event, bytes]:
        """Record source-producer visibility on the current scheduler stream."""
        event = torch.cuda.Event(interprocess=True)
        event.record(torch.cuda.current_stream(self.device))
        return event, event.ipc_handle()

    def submit(
        self,
        *,
        source_draft_instance_id: str,
        source_pool,
        transfer,
        reservation,
    ) -> Future[TransferMetrics]:
        with self._lock:
            self._raise_if_admission_fenced_locked()
            return self._executor.submit(
                self._run_transfer,
                source_draft_instance_id,
                source_pool,
                transfer,
                reservation,
            )

    def invalidate_peer(self, peer_instance_id: str) -> None:
        with self._lock:
            self._raise_if_admission_fenced_locked()
            invalidated = self._executor.submit(
                self._invalidate_peer_if_healthy, peer_instance_id
            )
        # Keep mapping retirement in the same FIFO as transfers.  In
        # particular, never invalidate a generation still referenced by a
        # preceding copy callback.
        invalidated.result()

    def _invalidate_peer_if_healthy(self, peer_instance_id: str) -> None:
        with self._lock:
            self._raise_if_admission_fenced_locked()
        self._mapping_cache.invalidate(peer_instance_id)

    def quarantine(self) -> None:
        """Permanently fence migration GPU work without releasing resources.

        This engine-policy fence is deliberately separate from
        ``_fatal_error``.  The latter represents an individual unquiesced
        transfer for which the lifecycle layer may explicitly probe
        quiescence and retire its reservation.  Permanent quarantine admits
        no probe, invalidation, mapping, copy, or close operation: uncertain
        mappings and ownership survive until CUDA context/process teardown.
        """

        with self._lock:
            if self._permanent_quarantine_error is None:
                self._permanent_quarantine_error = RuntimeError(
                    "DT CUDA IPC migration transport was permanently "
                    "quarantined by engine lifecycle policy"
                )

    def probe_quiescence(self) -> Future[bool]:
        """Fence a quarantined copy stream without blocking the scheduler."""
        with self._lock:
            if self._closed:
                raise RuntimeError("DT CUDA IPC transport is closed")
            self._raise_if_permanently_quarantined_locked(
                "probe migration-copy quiescence"
            )
            return self._executor.submit(self._probe_quiescence)

    def _probe_quiescence(self) -> bool:
        with self._lock:
            self._raise_if_permanently_quarantined_locked(
                "run a queued migration-copy quiescence probe"
            )
        torch.cuda.set_device(self.device)
        try:
            self._copy_stream.synchronize()
        except Exception:
            return False
        return True

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._raise_if_permanently_quarantined_locked(
                "close the migration transport"
            )
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
        # If stream synchronization failed, cudaIpcCloseMemHandle is not safe:
        # an already submitted transfer may still reference the mapping. Leak
        # it until CUDA context teardown instead of recycling remote memory.
        with self._lock:
            if (
                self._fatal_error is None
                and self._permanent_quarantine_error is None
            ):
                self._mapping_cache.close()

    def _raise_if_permanently_quarantined_locked(self, operation: str) -> None:
        error = self._permanent_quarantine_error
        if error is not None:
            raise RuntimeError(
                f"permanent DT CUDA IPC migration quarantine prevents {operation}; "
                "uncertain GPU resources must remain owned until process teardown"
            ) from error

    def _raise_if_admission_fenced_locked(self) -> None:
        if self._closed:
            raise RuntimeError("DT CUDA IPC transport is closed")
        self._raise_if_permanently_quarantined_locked("migration GPU work")
        if self._fatal_error is not None:
            raise RuntimeError(
                "DT CUDA IPC transport is quarantined after an unquiesced "
                "GPU-copy failure"
            ) from self._fatal_error

    def _run_transfer(
        self,
        source_draft_instance_id: str,
        source_pool,
        transfer,
        reservation,
    ) -> TransferMetrics:
        with self._lock:
            # Submission and execution are separated by an independent
            # executor queue.  Recheck both fences here before selecting a
            # device or allowing generation changes to touch the IPC cache.
            self._raise_if_admission_fenced_locked()
        torch.cuda.set_device(self.device)
        mapping_started_ns = time.monotonic_ns()
        mapped_bases = []
        cold_mapping_count = 0
        for allocation in source_pool.allocations:
            mapped, warm = self._mapping_cache.map(
                peer_instance_id=source_draft_instance_id,
                peer_session_id=source_pool.instance_session_id,
                pool_generation=source_pool.pool_generation,
                ipc_handle=allocation.ipc_handle,
            )
            mapped_bases.append(mapped)
            cold_mapping_count += int(not warm)
        mapping_done_ns = time.monotonic_ns()

        source_buffers = tuple(
            BufferRegistration(
                allocation_index=buffer.allocation_index,
                allocation_offset=buffer.allocation_offset,
                address=0,
                length=buffer.visible_length,
                item_length=buffer.item_length,
            )
            for buffer in source_pool.buffers
        )
        segments = build_copy_segments(
            mapped_source_bases=mapped_bases,
            source_buffers=source_buffers,
            destination_buffers=self._buffers,
            source_indices=transfer.page_indices,
            destination_indices=reservation.page_indices,
        )
        kv_bytes = sum(segment.length for segment in segments)
        if kv_bytes != transfer.kv_bytes:
            raise ValueError(
                "planned CUDA IPC byte count does not match source transfer metadata"
            )

        source_ready = torch.cuda.Event.from_ipc_handle(
            self.device_index,
            transfer.producer_event_handle,
        )
        return self._copy_and_wait(
            segments=segments,
            source_ready=source_ready,
            mapping_started_ns=mapping_started_ns,
            mapping_done_ns=mapping_done_ns,
            kv_bytes=kv_bytes,
            page_count=len(transfer.page_indices),
            cold_mapping_count=cold_mapping_count,
        )

    def _copy_and_wait(
        self,
        *,
        segments: tuple[CopySegment, ...],
        source_ready: torch.cuda.Event,
        mapping_started_ns: int,
        mapping_done_ns: int,
        kv_bytes: int,
        page_count: int,
        cold_mapping_count: int,
    ) -> TransferMetrics:
        """Submit one batch and never complete before possible GPU writes stop."""
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_done = torch.cuda.Event(enable_timing=True)
        submitted_ns = time.monotonic_ns()
        stream_may_reference_source = False
        try:
            with torch.cuda.stream(self._copy_stream):
                # From the first stream operation onward, a failure may leave
                # an enqueued dependency on the remote producer event.  Fence
                # that dependency before allowing the source event or pages to
                # be released, even if no memcpy was submitted yet.
                stream_may_reference_source = True
                self._copy_stream.wait_event(source_ready)
                gpu_start.record(self._copy_stream)
                driver_submit_started_ns = time.monotonic_ns()
                # The runtime may enqueue a prefix of the bounded segments
                # before returning an error. From this point onward every failure
                # must fence the stream before its Future becomes done.
                self._runtime.copy_batch_async(segments, self._copy_stream.cuda_stream)
                driver_submit_done_ns = time.monotonic_ns()
                gpu_done.record(self._copy_stream)
            gpu_done.synchronize()
        except Exception:
            if stream_may_reference_source:
                try:
                    self._copy_stream.synchronize()
                except Exception as fence_exc:
                    fatal = TransferQuiescenceError(
                        "failed CUDA IPC batch may still be writing destination "
                        "pages; transport and reservation must remain quarantined"
                    )
                    with self._lock:
                        self._fatal_error = fatal
                    raise fatal from fence_exc
            raise
        completed_ns = time.monotonic_ns()
        return TransferMetrics(
            submitted_ns=submitted_ns,
            gpu_completed_ns=completed_ns,
            mapping_ns=mapping_done_ns - mapping_started_ns,
            driver_submit_ns=driver_submit_done_ns - driver_submit_started_ns,
            gpu_copy_ns=int(gpu_start.elapsed_time(gpu_done) * 1_000_000),
            kv_bytes=kv_bytes,
            page_count=page_count,
            segment_count=len(segments),
            cold_mapping_count=cold_mapping_count,
        )
