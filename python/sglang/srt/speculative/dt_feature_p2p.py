"""Direct same-host CUDA-IPC transport for authoritative DT hidden states.

The Target owns a small ring of registered GPU feature buffers.  A published
descriptor names an allocation through its CUDA IPC handle plus offsets; it
never asks a peer to interpret a source virtual address.  The Draft maps that
allocation lazily, waits for the Target's interprocess producer event on its
own copy stream, and copies the visible range straight into a Draft CUDA
tensor.

This module intentionally has no Coordinator, request, or KV policy.  Its
callers own descriptor routing, deadlines, and the reliable read-done
handshake.  Keeping that boundary small lets the existing DT fencing policy
cover feature and KV transfers consistently.
"""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from threading import Lock

import torch

from sglang.srt.speculative.dt_cuda_ipc import (
    CopySegment,
    CudaIpcMappingCache,
    CudaIpcRuntime,
    register_buffers,
)


class FeatureTransportError(RuntimeError):
    """A feature descriptor or transport lifecycle invariant was violated."""


class FeatureTransportQuiescenceError(FeatureTransportError):
    """A failed copy stream may still reference a peer allocation."""

    resources_quiescent = False


class FeatureSlotState(StrEnum):
    FREE = "free"
    PUBLISHED = "published"
    QUARANTINED = "quarantined"


@dataclass(frozen=True, slots=True)
class FeatureTransferSlice:
    """One request-visible row range in a shared Target source slot."""

    transfer_id: str
    row_offset: int
    row_count: int


@dataclass(frozen=True, slots=True)
class FeatureSourceDescriptor:
    """Control-plane-only description of a Target VRAM feature range.

    ``allocation_offset`` is relative to the allocation represented by
    ``ipc_handle``; ``byte_offset`` is relative to the source-slot
    visible range.  Consumers must map the handle and calculate the address
    locally.  ``allocation_base_address`` is deliberately not published.

    The field names align with the DFlash extension of ``dt_protocol.SourceRef``
    so the controller can construct its wire struct without translating any
    payload data.
    """

    target_instance_id: str
    source_device_id: int
    source_slot_id: int
    slot_generation: int
    transfer_id: str
    feature_schema_id: str
    nbytes: int
    byte_offset: int
    transport: str
    source_session_id: str
    pool_generation: str
    transport_session: str
    device_uuid: str
    allocation_length: int
    allocation_offset: int
    ipc_handle: bytes
    dtype: str
    shape: tuple[int, int]
    producer_event_handle: bytes

    def source_ref_kwargs(self) -> dict[str, object]:
        """Return names/values used by the feature-capable ``SourceRef`` wire type."""
        return {
            "target_instance_id": self.target_instance_id,
            "source_device_id": self.source_device_id,
            "source_slot_id": self.source_slot_id,
            "slot_generation": self.slot_generation,
            "transfer_id": self.transfer_id,
            "feature_schema_id": self.feature_schema_id,
            "nbytes": self.nbytes,
            "byte_offset": self.byte_offset,
            "transport": self.transport,
            "source_session_id": self.source_session_id,
            "pool_generation": self.pool_generation,
            "transport_session": self.transport_session,
            "device_uuid": self.device_uuid,
            "allocation_length": self.allocation_length,
            "allocation_offset": self.allocation_offset,
            "ipc_handle": self.ipc_handle,
            "dtype": self.dtype,
            "shape": self.shape,
            "producer_event_handle": self.producer_event_handle,
        }


@dataclass(frozen=True, slots=True)
class FeaturePublish:
    """Descriptors produced atomically from one shared source-slot lease."""

    source_slot_id: int
    slot_generation: int
    producer_event: torch.cuda.Event
    descriptors: tuple[FeatureSourceDescriptor, ...]
    hidden_produced_ns: int
    descriptor_published_ns: int


@dataclass(frozen=True, slots=True)
class FeaturePullMetrics:
    submitted_ns: int
    gpu_completed_ns: int
    mapping_ns: int
    driver_submit_ns: int
    gpu_copy_ns: int
    feature_bytes: int
    cold_mapping_count: int


@dataclass(slots=True)
class _FeatureSlot:
    generation: int = 0
    state: FeatureSlotState = FeatureSlotState.FREE
    active_transfers: set[str] | None = None
    producer_event: torch.cuda.Event | None = None

    def __post_init__(self) -> None:
        if self.active_transfers is None:
            self.active_transfers = set()


class TargetFeatureSourcePool:
    """Bounded Target VRAM source slots protected by transfer leases.

    The pool has exactly one contiguous BF16 allocation of shape
    ``[slot_count, max_tokens, feature_width]``.  It is registered once during
    construction, and each descriptor refers to a slot-sized subrange.  A slot
    becomes reusable only after every published transfer has received a valid
    read-done acknowledgement (or has been explicitly cancelled by policy).
    """

    def __init__(
        self,
        *,
        target_instance_id: str,
        source_session_id: str,
        device: torch.device,
        slot_count: int,
        max_tokens: int,
        feature_width: int,
    ) -> None:
        if not target_instance_id or not source_session_id:
            raise ValueError("Target identity and source session must not be empty")
        if slot_count <= 0 or max_tokens <= 0 or feature_width <= 0:
            raise ValueError("feature source-pool dimensions must be positive")
        requested_device = torch.device(device)
        if requested_device.type != "cuda":
            raise ValueError("DFlash feature source buffers require a CUDA device")
        self.device_index = (
            torch.cuda.current_device()
            if requested_device.index is None
            else int(requested_device.index)
        )
        self.device = torch.device("cuda", self.device_index)
        self.target_instance_id = target_instance_id
        self.source_session_id = source_session_id
        self.slot_count = int(slot_count)
        self.max_tokens = int(max_tokens)
        self.feature_width = int(feature_width)
        self.dtype = torch.bfloat16
        self._slot_nbytes = self.max_tokens * self.feature_width * torch.empty(
            (), dtype=self.dtype
        ).element_size()
        self.pool_generation = uuid.uuid4().hex
        self.transport_session = uuid.uuid4().hex
        self._runtime = CudaIpcRuntime()
        torch.cuda.set_device(self.device)
        self._storage = torch.empty(
            (self.slot_count, self.max_tokens, self.feature_width),
            dtype=self.dtype,
            device=self.device,
        )
        self._allocations, buffers = register_buffers(
            self._runtime,
            ((self._storage.data_ptr(), self._storage.nbytes, self._slot_nbytes),),
        )
        if len(buffers) != 1 or len(self._allocations) != 1:
            raise FeatureTransportError("feature source pool must use one allocation")
        self._buffer = buffers[0]
        self._slots = [_FeatureSlot() for _ in range(self.slot_count)]
        self._active: dict[str, tuple[int, int]] = {}
        # Bounded history makes duplicate read-done idempotent without allowing
        # a stale acknowledgement to free a reissued slot generation.
        self._completed: OrderedDict[str, tuple[int, int]] = OrderedDict()
        self._completed_limit = max(64, self.slot_count * 16)
        self._lock = Lock()
        self._closed = False

        properties = torch.cuda.get_device_properties(self.device)
        self.device_uuid = str(properties.uuid)

    @property
    def slot_nbytes(self) -> int:
        return self._slot_nbytes

    @property
    def free_slot_count(self) -> int:
        with self._lock:
            return sum(slot.state is FeatureSlotState.FREE for slot in self._slots)

    @property
    def quarantined_slot_count(self) -> int:
        with self._lock:
            return sum(slot.state is FeatureSlotState.QUARANTINED for slot in self._slots)

    def publish(
        self,
        *,
        hidden_states: torch.Tensor | Sequence[torch.Tensor],
        feature_schema_id: str,
        transfers: Sequence[FeatureTransferSlice],
        hidden_produced_ns: int | None = None,
    ) -> FeaturePublish:
        """Copy packed BF16 features into a free source slot and publish leases.

        The copy and producer event are enqueued on the current Target CUDA
        stream.  This method never synchronizes that stream and never stages
        feature bytes on the CPU.  If every fixed slot is leased, it raises
        ``FeatureTransportError`` so the caller can apply scheduler-level
        backpressure rather than allocating an unbounded buffer.
        """
        if self._closed:
            raise FeatureTransportError("feature source pool is closed")
        if not feature_schema_id:
            raise ValueError("feature schema id must not be empty")
        chunks = (
            (hidden_states,)
            if isinstance(hidden_states, torch.Tensor)
            else tuple(hidden_states)
        )
        if not chunks:
            raise ValueError("feature publication requires at least one hidden chunk")
        for chunk in chunks:
            self._validate_hidden_states(chunk)
        token_count = sum(int(chunk.shape[0]) for chunk in chunks)
        if token_count > self.max_tokens:
            raise ValueError("feature token count exceeds bounded source-slot capacity")
        slices = tuple(transfers)
        self._validate_slices(slices, token_count)
        if not slices:
            raise ValueError("a published feature range needs at least one transfer")

        with self._lock:
            slot_id = next(
                (
                    index
                    for index, slot in enumerate(self._slots)
                    if slot.state is FeatureSlotState.FREE
                ),
                None,
            )
            if slot_id is None:
                raise FeatureTransportError("all bounded feature source slots are leased")
            duplicate = next(
                (item.transfer_id for item in slices if item.transfer_id in self._active),
                None,
            )
            if duplicate is not None:
                raise FeatureTransportError(
                    f"feature transfer {duplicate!r} is already active"
                )
            slot = self._slots[slot_id]
            slot.generation += 1
            generation = slot.generation
            slot.state = FeatureSlotState.PUBLISHED
            slot.active_transfers = {item.transfer_id for item in slices}
            for item in slices:
                self._active[item.transfer_id] = (slot_id, generation)

        try:
            target = self._storage[slot_id]
            row_offset = 0
            # Every chunk is CUDA BF16. Copy directly into the fixed registered
            # allocation without constructing a per-round packed tensor.
            for chunk in chunks:
                row_end = row_offset + int(chunk.shape[0])
                target[row_offset:row_end].copy_(chunk, non_blocking=True)
                row_offset = row_end
            producer_event = torch.cuda.Event(interprocess=True)
            producer_event.record(torch.cuda.current_stream(self.device))
            event_handle = producer_event.ipc_handle()
        except Exception:
            # A failed enqueue is not safe to reuse: CUDA may have accepted a
            # prefix of work.  Keep the fixed slot quarantined until process
            # teardown rather than risking a late remote read.
            with self._lock:
                self._quarantine_locked(slot_id, generation)
            raise

        descriptor_published_ns = time.monotonic_ns()
        allocation = self._allocations[self._buffer.allocation_index]
        slot_allocation_offset = self._buffer.allocation_offset + slot_id * self._slot_nbytes
        descriptors = tuple(
            FeatureSourceDescriptor(
                target_instance_id=self.target_instance_id,
                source_device_id=self.device_index,
                source_slot_id=slot_id,
                slot_generation=generation,
                transfer_id=item.transfer_id,
                feature_schema_id=feature_schema_id,
                nbytes=item.row_count * self.feature_width * 2,
                byte_offset=item.row_offset * self.feature_width * 2,
                transport="cuda-ipc",
                source_session_id=self.source_session_id,
                pool_generation=self.pool_generation,
                transport_session=self.transport_session,
                device_uuid=self.device_uuid,
                allocation_length=allocation.length,
                allocation_offset=slot_allocation_offset,
                ipc_handle=allocation.ipc_handle,
                dtype="bfloat16",
                shape=(item.row_count, self.feature_width),
                producer_event_handle=event_handle,
            )
            for item in slices
        )
        with self._lock:
            # Retain the local event object for the entire source lease; its
            # IPC handle must remain usable by every assigned Draft.
            current = self._slots[slot_id]
            if current.generation != generation or current.state is not FeatureSlotState.PUBLISHED:
                raise FeatureTransportError("published source slot was invalidated")
            current.producer_event = producer_event
        return FeaturePublish(
            source_slot_id=slot_id,
            slot_generation=generation,
            producer_event=producer_event,
            descriptors=descriptors,
            hidden_produced_ns=(
                time.monotonic_ns() if hidden_produced_ns is None else hidden_produced_ns
            ),
            descriptor_published_ns=descriptor_published_ns,
        )

    def read_done(
        self,
        *,
        transfer_id: str,
        source_slot_id: int,
        slot_generation: int,
        source_session_id: str,
        pool_generation: str,
    ) -> bool:
        """Acknowledge one Draft read and return whether it freed the slot.

        Repeating the exact acknowledgement is idempotent and returns
        ``False``.  A session, pool, slot, or generation mismatch raises
        instead of allowing a stale Draft to release current source memory.
        """
        self._validate_fence(
            source_session_id=source_session_id,
            pool_generation=pool_generation,
        )
        with self._lock:
            active = self._active.get(transfer_id)
            if active is None:
                completed = self._completed.get(transfer_id)
                if completed is None:
                    raise FeatureTransportError(f"unknown feature transfer {transfer_id!r}")
                if completed != (source_slot_id, slot_generation):
                    raise FeatureTransportError("stale read-done conflicts with completed lease")
                return False
            if active != (source_slot_id, slot_generation):
                raise FeatureTransportError("read-done slot or generation does not match lease")
            slot = self._slot_for_generation_locked(
                source_slot_id, slot_generation, allow_quarantined=True
            )
            self._active.pop(transfer_id)
            slot.active_transfers.remove(transfer_id)
            self._remember_completed_locked(transfer_id, active)
            if slot.active_transfers:
                return False
            slot.state = FeatureSlotState.FREE
            slot.producer_event = None
            return True

    def cancel(
        self,
        *,
        transfer_id: str,
        source_slot_id: int,
        slot_generation: int,
        source_session_id: str,
        pool_generation: str,
    ) -> bool:
        """Release an undelivered descriptor through the same fenced lease path."""
        return self.read_done(
            transfer_id=transfer_id,
            source_slot_id=source_slot_id,
            slot_generation=slot_generation,
            source_session_id=source_session_id,
            pool_generation=pool_generation,
        )

    def quarantine(
        self,
        *,
        source_slot_id: int,
        slot_generation: int,
        source_session_id: str,
        pool_generation: str,
    ) -> None:
        """Fence a failed/uncertain source range permanently for this process."""
        self._validate_fence(
            source_session_id=source_session_id,
            pool_generation=pool_generation,
        )
        with self._lock:
            if source_slot_id < 0 or source_slot_id >= self.slot_count:
                raise FeatureTransportError("source slot id is out of range")
            slot = self._slots[source_slot_id]
            if slot.generation != slot_generation:
                raise FeatureTransportError("stale source slot generation")
            if slot.state is FeatureSlotState.QUARANTINED:
                return
            if slot.state is not FeatureSlotState.PUBLISHED:
                raise FeatureTransportError("feature source slot is not published")
            self._quarantine_locked(source_slot_id, slot_generation)

    def close(self) -> None:
        """Stop publishing. CUDA allocation lifetime remains with this process."""
        with self._lock:
            self._closed = True

    def _validate_hidden_states(self, hidden_states: torch.Tensor) -> None:
        if hidden_states.device != self.device:
            raise ValueError("feature hidden states must be on the source CUDA device")
        if hidden_states.dtype is not self.dtype:
            raise ValueError("DFlash feature hidden states must be bfloat16")
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.feature_width:
            raise ValueError(
                "feature hidden states must have shape [tokens, feature_width]"
            )
        if not hidden_states.is_contiguous():
            raise ValueError("feature hidden states must be contiguous before publishing")
        if hidden_states.shape[0] <= 0:
            raise ValueError("feature hidden-state chunks must not be empty")

    @staticmethod
    def _validate_slices(
        slices: Sequence[FeatureTransferSlice], token_count: int
    ) -> None:
        transfer_ids = set()
        for item in slices:
            if not item.transfer_id or item.transfer_id in transfer_ids:
                raise ValueError("feature transfer ids must be unique and non-empty")
            if item.row_offset < 0 or item.row_count <= 0:
                raise ValueError("feature transfer rows must be positive and in range")
            if item.row_offset + item.row_count > token_count:
                raise ValueError("feature transfer range exceeds published hidden states")
            transfer_ids.add(item.transfer_id)

    def _validate_fence(
        self, *, source_session_id: str, pool_generation: str
    ) -> None:
        if source_session_id != self.source_session_id:
            raise FeatureTransportError("stale source session")
        if pool_generation != self.pool_generation:
            raise FeatureTransportError("stale source pool generation")

    def _slot_for_generation_locked(
        self,
        source_slot_id: int,
        slot_generation: int,
        *,
        allow_quarantined: bool = False,
    ) -> _FeatureSlot:
        if source_slot_id < 0 or source_slot_id >= self.slot_count:
            raise FeatureTransportError("source slot id is out of range")
        slot = self._slots[source_slot_id]
        allowed_states = (
            {FeatureSlotState.PUBLISHED, FeatureSlotState.QUARANTINED}
            if allow_quarantined
            else {FeatureSlotState.PUBLISHED}
        )
        if slot.generation != slot_generation or slot.state not in allowed_states:
            raise FeatureTransportError("stale source slot generation")
        return slot

    def _remember_completed_locked(
        self, transfer_id: str, lease: tuple[int, int]
    ) -> None:
        self._completed[transfer_id] = lease
        self._completed.move_to_end(transfer_id)
        while len(self._completed) > self._completed_limit:
            self._completed.popitem(last=False)

    def _quarantine_locked(self, slot_id: int, generation: int) -> None:
        slot = self._slot_for_generation_locked(slot_id, generation)
        # Quarantine prevents publication but retains the exact outstanding
        # readers.  A later valid read-done (or authoritative retirement
        # proof) can still drain the lease and recover the bounded slot.
        slot.state = FeatureSlotState.QUARANTINED


class DraftFeaturePullTransport:
    """Asynchronously pull one Target feature range into Draft-local VRAM."""

    def __init__(self, *, device: torch.device, max_workers: int = 1) -> None:
        if max_workers <= 0:
            raise ValueError("feature copy worker count must be positive")
        requested_device = torch.device(device)
        if requested_device.type != "cuda":
            raise ValueError("DFlash feature pulls require a CUDA device")
        self.device_index = (
            torch.cuda.current_device()
            if requested_device.index is None
            else int(requested_device.index)
        )
        self.device = torch.device("cuda", self.device_index)
        torch.cuda.set_device(self.device)
        self._runtime = CudaIpcRuntime()
        self._mapping_cache = CudaIpcMappingCache(self._runtime)
        self._copy_stream = torch.cuda.Stream(device=self.device)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="dt-feature-p2p",
        )
        self._fatal_error: FeatureTransportQuiescenceError | None = None
        self._closed = False
        self._lock = Lock()

    def submit(
        self,
        *,
        source: FeatureSourceDescriptor,
        destination: torch.Tensor,
    ) -> Future[FeaturePullMetrics]:
        """Queue a direct source-to-destination pull and return immediately."""
        self._validate_submission(source, destination)
        with self._lock:
            if self._closed:
                raise FeatureTransportError("feature pull transport is closed")
            if self._fatal_error is not None:
                raise FeatureTransportError(
                    "feature pull transport is quarantined after an unquiesced GPU-copy failure"
                ) from self._fatal_error
            return self._executor.submit(self._run_pull, source, destination)

    @staticmethod
    def query(future: Future[FeaturePullMetrics]) -> FeaturePullMetrics | None:
        """Return completed metrics without waiting, or ``None`` while in flight."""
        return future.result() if future.done() else None

    def wait(self, future: Future[FeaturePullMetrics]) -> FeaturePullMetrics:
        """Wait only in a worker/shutdown path, never a scheduler hot path."""
        return future.result()

    def invalidate_peer(self, target_instance_id: str) -> None:
        self._mapping_cache.invalidate(target_instance_id)

    def quarantine(self) -> None:
        """Reject new pulls after an external lifecycle failure."""
        with self._lock:
            if self._fatal_error is None:
                self._fatal_error = FeatureTransportQuiescenceError(
                    "feature pull transport was quarantined by lifecycle policy"
                )

    def probe_quiescence(self) -> Future[bool]:
        """Fence the copy stream asynchronously before retrying/releasing state."""
        with self._lock:
            if self._closed:
                raise FeatureTransportError("feature pull transport is closed")
            return self._executor.submit(self._probe_quiescence)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
        # Do not close mappings after an unquiesced failure: CUDA could still
        # access them. Context teardown is safer than remote-memory reuse.
        if self._fatal_error is None:
            self._mapping_cache.close()

    def _probe_quiescence(self) -> bool:
        torch.cuda.set_device(self.device)
        try:
            self._copy_stream.synchronize()
        except Exception:
            return False
        return True

    def _run_pull(
        self, source: FeatureSourceDescriptor, destination: torch.Tensor
    ) -> FeaturePullMetrics:
        torch.cuda.set_device(self.device)
        mapping_started_ns = time.monotonic_ns()
        mapped_base, warm = self._mapping_cache.map(
            peer_instance_id=source.target_instance_id,
            peer_session_id=source.source_session_id,
            pool_generation=source.pool_generation,
            ipc_handle=source.ipc_handle,
        )
        mapping_done_ns = time.monotonic_ns()
        source_address = mapped_base + source.allocation_offset + source.byte_offset
        source_ready = torch.cuda.Event.from_ipc_handle(
            self.device_index, source.producer_event_handle
        )
        segment = CopySegment(
            source_address=source_address,
            destination_address=destination.data_ptr(),
            length=source.nbytes,
        )
        return self._copy_and_wait(
            source_ready=source_ready,
            segment=segment,
            mapping_started_ns=mapping_started_ns,
            mapping_done_ns=mapping_done_ns,
            cold_mapping_count=int(not warm),
        )

    def _copy_and_wait(
        self,
        *,
        source_ready: torch.cuda.Event,
        segment: CopySegment,
        mapping_started_ns: int,
        mapping_done_ns: int,
        cold_mapping_count: int,
    ) -> FeaturePullMetrics:
        """Finish a submitted batch only once it cannot touch source memory."""
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_done = torch.cuda.Event(enable_timing=True)
        submitted_ns = time.monotonic_ns()
        stream_may_reference_source = False
        try:
            with torch.cuda.stream(self._copy_stream):
                stream_may_reference_source = True
                self._copy_stream.wait_event(source_ready)
                gpu_start.record(self._copy_stream)
                driver_submit_started_ns = time.monotonic_ns()
                self._runtime.copy_batch_async((segment,), self._copy_stream.cuda_stream)
                driver_submit_done_ns = time.monotonic_ns()
                gpu_done.record(self._copy_stream)
            # This synchronization happens only in the bounded copy worker;
            # Target and Draft scheduler loops poll the returned Future.
            gpu_done.synchronize()
        except Exception:
            if stream_may_reference_source:
                try:
                    self._copy_stream.synchronize()
                except Exception as fence_exc:
                    fatal = FeatureTransportQuiescenceError(
                        "failed feature pull may still access source or destination VRAM"
                    )
                    with self._lock:
                        self._fatal_error = fatal
                    raise fatal from fence_exc
            raise
        return FeaturePullMetrics(
            submitted_ns=submitted_ns,
            gpu_completed_ns=time.monotonic_ns(),
            mapping_ns=mapping_done_ns - mapping_started_ns,
            driver_submit_ns=driver_submit_done_ns - driver_submit_started_ns,
            gpu_copy_ns=int(gpu_start.elapsed_time(gpu_done) * 1_000_000),
            feature_bytes=segment.length,
            cold_mapping_count=cold_mapping_count,
        )

    def _validate_submission(
        self, source: FeatureSourceDescriptor, destination: torch.Tensor
    ) -> None:
        if source.transport != "cuda-ipc":
            raise ValueError("DFlash feature transport requires cuda-ipc descriptors")
        if not source.target_instance_id or not source.source_session_id:
            raise ValueError("feature descriptor source identity is incomplete")
        if not source.pool_generation or not source.transport_session:
            raise ValueError("feature descriptor generation fence is incomplete")
        if source.dtype != "bfloat16" or len(source.shape) != 2:
            raise ValueError("DFlash feature descriptor must describe a BF16 matrix")
        if source.shape[0] <= 0 or source.shape[1] <= 0:
            raise ValueError("feature descriptor shape must be positive")
        expected_nbytes = source.shape[0] * source.shape[1] * 2
        if source.nbytes != expected_nbytes:
            raise ValueError("feature descriptor byte length does not match BF16 shape")
        if source.byte_offset < 0 or source.allocation_offset < 0:
            raise ValueError("feature descriptor offsets must be non-negative")
        if (
            source.allocation_offset + source.byte_offset + source.nbytes
            > source.allocation_length
        ):
            raise ValueError("feature descriptor range exceeds mapped allocation")
        if not source.ipc_handle or not source.producer_event_handle:
            raise ValueError("feature descriptor lacks IPC handle or producer event")
        if destination.device != self.device or destination.dtype is not torch.bfloat16:
            raise ValueError("feature destination must be BF16 on the Draft CUDA device")
        if not destination.is_contiguous() or destination.numel() * 2 != source.nbytes:
            raise ValueError("feature destination does not exactly match source visible range")
