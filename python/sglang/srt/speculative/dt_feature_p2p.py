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

import os
import time
import uuid
from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from threading import Condition, Lock, local

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
    # Leased to an in-flight verify that writes its hidden rows in place.
    RESERVED = "reserved"
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

    The field names align with the direct-GPU extension of ``dt_protocol.SourceRef``
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
    pack_strategy: str
    pack_chunk_count: int


@dataclass(frozen=True, slots=True)
class FeaturePullMetrics:
    submitted_ns: int
    gpu_completed_ns: int
    mapping_ns: int
    driver_submit_ns: int
    gpu_copy_ns: int
    feature_bytes: int
    cold_mapping_count: int
    worker_started_ns: int = 0
    mapping_done_ns: int = 0
    driver_submit_started_ns: int = 0
    driver_submit_done_ns: int = 0
    cuda_pre_copy_ns: int = 0
    gpu_start_observed_ns: int = 0


@dataclass(frozen=True, slots=True)
class FeaturePullBatchMetrics:
    submitted_ns: int
    gpu_completed_ns: int
    mapping_ns: int
    driver_submit_ns: int
    gpu_copy_ns: int
    feature_bytes: int
    cold_mapping_count: int
    segment_count: int
    peer_count: int
    worker_started_ns: int = 0
    mapping_done_ns: int = 0
    driver_submit_started_ns: int = 0
    driver_submit_done_ns: int = 0
    cuda_pre_copy_ns: int = 0
    gpu_start_observed_ns: int = 0


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
            raise ValueError("direct GPU source buffers require a CUDA device")
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
        # Cancelling an undelivered descriptor is not equivalent to a remote
        # read completion: the Target publication copy can still be using the
        # slot.  Keep those leases active until the local producer event can be
        # queried complete.  This also makes existing active-lease lifecycle
        # accounting include deferred producer retirement.
        self._cancelled_transfers: set[str] = set()
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
        self.poll_cancelled_producers()
        with self._lock:
            return sum(slot.state is FeatureSlotState.FREE for slot in self._slots)

    @property
    def quarantined_slot_count(self) -> int:
        with self._lock:
            return sum(slot.state is FeatureSlotState.QUARANTINED for slot in self._slots)

    @property
    def active_lease_count(self) -> int:
        """Active remote leases plus cancellations awaiting producer completion."""

        with self._lock:
            return len(self._active)

    def publish(
        self,
        *,
        hidden_states: torch.Tensor | Sequence[torch.Tensor],
        feature_schema_id: str,
        transfers: Sequence[FeatureTransferSlice],
        hidden_produced_ns: int | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> FeaturePublish:
        """Copy packed BF16 features into a free source slot and publish leases.

        The copy and producer event are enqueued on the current Target CUDA
        stream, or on ``stream`` when the caller has already proven on the host
        that every chunk is complete.  This method never synchronizes a stream
        and never stages feature bytes on the CPU.  If every fixed slot is
        leased, it raises ``FeatureTransportError`` so the caller can apply
        scheduler-level backpressure rather than allocating an unbounded buffer.
        """
        if self._closed:
            raise FeatureTransportError("feature source pool is closed")
        self.poll_cancelled_producers()
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
            # ``close()`` uses this same lock to fence publication and snapshot
            # ownership. Validation above is intentionally outside the lock,
            # so admission must be rechecked here before reserving a slot; an
            # initial fast-path check alone would allow post-close CUDA work.
            if self._closed:
                raise FeatureTransportError("feature source pool is closed")
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

        producer_event = None
        if stream is not None:
            for chunk in chunks:
                chunk.record_stream(stream)
        publish_stream = (
            torch.cuda.current_stream(self.device) if stream is None else stream
        )
        try:
            with torch.cuda.stream(publish_stream):
                target = self._storage[slot_id]
                cat = getattr(torch, "cat", None)
                if len(chunks) > 1 and callable(cat):
                    # Pack ragged request chunks directly into the leased,
                    # registered slot. ``out=`` avoids a transient packed tensor
                    # while replacing one D2D enqueue per request with one ordered
                    # CUDA operation. The producer event below fences the complete
                    # pack exactly as it fenced the former copy loop.
                    cat(chunks, dim=0, out=target[:token_count])
                    pack_strategy = "cat-out"
                else:
                    # A single chunk is already one enqueue. Keep the original
                    # loop as the capability fallback for runtimes without cat.
                    row_offset = 0
                    for chunk in chunks:
                        row_end = row_offset + int(chunk.shape[0])
                        target[row_offset:row_end].copy_(chunk, non_blocking=True)
                        row_offset = row_end
                    pack_strategy = "copy-loop"
                producer_event = torch.cuda.Event(interprocess=True)
                producer_event.record(publish_stream)
                event_handle = producer_event.ipc_handle()
        except Exception:
            # A failed enqueue is not safe to reuse: CUDA may have accepted a
            # prefix of work.  Keep the fixed slot quarantined until process
            # teardown rather than risking a late remote read. If event
            # creation or IPC-handle export failed after recording, retain that
            # event with the quarantined slot as part of the uncertain work.
            with self._lock:
                if producer_event is not None:
                    self._slots[slot_id].producer_event = producer_event
                self._quarantine_locked(slot_id, generation)
            raise

        return self._commit_publication(
            slot_id=slot_id,
            generation=generation,
            slices=slices,
            feature_schema_id=feature_schema_id,
            producer_event=producer_event,
            event_handle=event_handle,
            hidden_produced_ns=hidden_produced_ns,
            pack_strategy=pack_strategy,
            pack_chunk_count=len(chunks),
        )

    def reserve(self, token_count: int) -> tuple[int, int, torch.Tensor] | None:
        """Lease a free slot whose rows a producer will write in place.

        Returns ``(slot_id, generation, rows)`` or ``None`` when no slot is
        free or the request exceeds slot capacity.  The caller must later call
        :meth:`publish_reserved` or :meth:`release_reserved`.
        """
        if token_count <= 0 or token_count > self.max_tokens:
            return None
        with self._lock:
            if self._closed:
                return None
            slot_id = next(
                (
                    index
                    for index, slot in enumerate(self._slots)
                    if slot.state is FeatureSlotState.FREE
                ),
                None,
            )
            if slot_id is None:
                return None
            slot = self._slots[slot_id]
            slot.generation += 1
            slot.state = FeatureSlotState.RESERVED
            return slot_id, slot.generation, self._storage[slot_id][:token_count]

    def release_reserved(self, slot_id: int, generation: int) -> None:
        """Return an unpublished reservation once its producer write finished."""
        with self._lock:
            slot = self._slots[slot_id]
            if slot.generation == generation and slot.state is FeatureSlotState.RESERVED:
                slot.state = FeatureSlotState.FREE

    def quarantine_reserved(self, slot_id: int, generation: int) -> None:
        """Fence a reservation whose in-place write may be incomplete."""
        with self._lock:
            slot = self._slots[slot_id]
            if slot.generation == generation and slot.state is FeatureSlotState.RESERVED:
                slot.state = FeatureSlotState.QUARANTINED

    @property
    def reserved_slot_count(self) -> int:
        with self._lock:
            return sum(slot.state is FeatureSlotState.RESERVED for slot in self._slots)

    def publish_reserved(
        self,
        *,
        slot_id: int,
        generation: int,
        token_count: int,
        feature_schema_id: str,
        transfers: Sequence[FeatureTransferSlice],
        producer_event: torch.cuda.Event,
        hidden_produced_ns: int | None = None,
    ) -> FeaturePublish:
        """Publish rows already written into a reserved slot.

        ``producer_event`` is an interprocess event recorded after the in-place
        write; Draft pulls order on it exactly as on a publication copy.
        """
        if self._closed:
            raise FeatureTransportError("feature source pool is closed")
        if not feature_schema_id:
            raise ValueError("feature schema id must not be empty")
        slices = tuple(transfers)
        if not slices:
            raise ValueError("a published feature range needs at least one transfer")
        self._validate_slices(slices, token_count)
        with self._lock:
            if self._closed:
                raise FeatureTransportError("feature source pool is closed")
            slot = self._slots[slot_id]
            if slot.generation != generation or slot.state is not FeatureSlotState.RESERVED:
                raise FeatureTransportError("feature source reservation is stale")
            duplicate = next(
                (item.transfer_id for item in slices if item.transfer_id in self._active),
                None,
            )
            if duplicate is not None:
                raise FeatureTransportError(
                    f"feature transfer {duplicate!r} is already active"
                )
            slot.state = FeatureSlotState.PUBLISHED
            slot.active_transfers = {item.transfer_id for item in slices}
            for item in slices:
                self._active[item.transfer_id] = (slot_id, generation)
            slot.producer_event = producer_event
        try:
            event_handle = producer_event.ipc_handle()
        except Exception:
            with self._lock:
                self._quarantine_locked(slot_id, generation)
            raise
        return self._commit_publication(
            slot_id=slot_id,
            generation=generation,
            slices=slices,
            feature_schema_id=feature_schema_id,
            producer_event=producer_event,
            event_handle=event_handle,
            hidden_produced_ns=hidden_produced_ns,
            pack_strategy="in-place",
            pack_chunk_count=0,
        )

    def _commit_publication(
        self,
        *,
        slot_id: int,
        generation: int,
        slices: tuple[FeatureTransferSlice, ...],
        feature_schema_id: str,
        producer_event: torch.cuda.Event,
        event_handle: bytes,
        hidden_produced_ns: int | None,
        pack_strategy: str,
        pack_chunk_count: int,
    ) -> FeaturePublish:
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
            pack_strategy=pack_strategy,
            pack_chunk_count=pack_chunk_count,
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
            active = self._validate_read_done_locked(
                transfer_id=transfer_id,
                source_slot_id=source_slot_id,
                slot_generation=slot_generation,
            )
            if active is None:
                return False
            slot = self._slot_for_generation_locked(
                source_slot_id, slot_generation, allow_quarantined=True
            )
            self._active.pop(transfer_id)
            self._cancelled_transfers.discard(transfer_id)
            slot.active_transfers.remove(transfer_id)
            self._remember_completed_locked(transfer_id, active)
            if slot.active_transfers:
                return False
            slot.state = FeatureSlotState.FREE
            slot.producer_event = None
            return True

    def validate_read_done(
        self,
        *,
        transfer_id: str,
        source_slot_id: int,
        slot_generation: int,
        source_session_id: str,
        pool_generation: str,
    ) -> None:
        """Validate a read completion without retiring its source lease."""

        self._validate_fence(
            source_session_id=source_session_id,
            pool_generation=pool_generation,
        )
        with self._lock:
            self._validate_read_done_locked(
                transfer_id=transfer_id,
                source_slot_id=source_slot_id,
                slot_generation=slot_generation,
            )

    def _validate_read_done_locked(
        self,
        *,
        transfer_id: str,
        source_slot_id: int,
        slot_generation: int,
    ) -> tuple[int, int] | None:
        active = self._active.get(transfer_id)
        if active is None:
            completed = self._completed.get(transfer_id)
            if completed is None:
                raise FeatureTransportError(f"unknown feature transfer {transfer_id!r}")
            if completed != (source_slot_id, slot_generation):
                raise FeatureTransportError(
                    "stale read-done conflicts with completed lease"
                )
            return None
        if active != (source_slot_id, slot_generation):
            raise FeatureTransportError(
                "read-done slot or generation does not match lease"
            )
        self._slot_for_generation_locked(
            source_slot_id, slot_generation, allow_quarantined=True
        )
        return active

    def cancel(
        self,
        *,
        transfer_id: str,
        source_slot_id: int,
        slot_generation: int,
        source_session_id: str,
        pool_generation: str,
    ) -> bool:
        """Retire an undelivered descriptor after its producer copy completes.

        A remote read-done proves that the consumer waited for the producer
        event. Cancellation has no such ordering guarantee, so it leaves the
        lease active and polls the event without synchronizing. Normal serving
        retires delayed cancellations from ``free_slot_count``; shutdown sees
        the still-active lease and retains the pool if the event is pending.
        """
        self._validate_fence(
            source_session_id=source_session_id,
            pool_generation=pool_generation,
        )
        with self._lock:
            active = self._validate_read_done_locked(
                transfer_id=transfer_id,
                source_slot_id=source_slot_id,
                slot_generation=slot_generation,
            )
            if active is None:
                return False
            self._cancelled_transfers.add(transfer_id)

        self.poll_cancelled_producers()
        with self._lock:
            return (
                transfer_id not in self._active
                and self._slots[source_slot_id].generation == slot_generation
                and self._slots[source_slot_id].state is FeatureSlotState.FREE
            )

    def poll_cancelled_producers(self) -> int:
        """Nonblocking retirement for cancelled, producer-owned source leases.

        A failed event query leaves the event, storage, and lease reachable in
        a permanently quarantined slot.  No synchronization is attempted: an
        uncertain CUDA fence must remain owned until process teardown.
        """
        with self._lock:
            candidates: list[tuple[int, int, torch.cuda.Event]] = []
            seen: set[tuple[int, int]] = set()
            for transfer_id in self._cancelled_transfers:
                active = self._active.get(transfer_id)
                if active is None or active in seen:
                    continue
                slot_id, generation = active
                slot = self._slot_for_generation_locked(
                    slot_id, generation, allow_quarantined=True
                )
                if slot.state is FeatureSlotState.QUARANTINED:
                    continue
                if slot.producer_event is None:
                    self._quarantine_locked(slot_id, generation)
                    raise FeatureTransportQuiescenceError(
                        "cancelled feature source has no producer completion event"
                    )
                seen.add(active)
                candidates.append((slot_id, generation, slot.producer_event))

        retired = 0
        for slot_id, generation, producer_event in candidates:
            try:
                producer_complete = bool(producer_event.query())
            except Exception as exc:
                with self._lock:
                    slot = self._slots[slot_id]
                    if (
                        slot.generation == generation
                        and slot.state is FeatureSlotState.PUBLISHED
                        and slot.producer_event is producer_event
                    ):
                        self._quarantine_locked(slot_id, generation)
                raise FeatureTransportQuiescenceError(
                    "failed to query cancelled feature producer completion"
                ) from exc
            if not producer_complete:
                continue
            with self._lock:
                slot = self._slots[slot_id]
                if (
                    slot.generation != generation
                    or slot.state is not FeatureSlotState.PUBLISHED
                    or slot.producer_event is not producer_event
                ):
                    continue
                cancelled = slot.active_transfers.intersection(
                    self._cancelled_transfers
                )
                for transfer_id in tuple(cancelled):
                    active = self._active.get(transfer_id)
                    if active != (slot_id, generation):
                        continue
                    self._active.pop(transfer_id)
                    self._cancelled_transfers.discard(transfer_id)
                    slot.active_transfers.remove(transfer_id)
                    self._remember_completed_locked(transfer_id, active)
                    retired += 1
                if not slot.active_transfers:
                    slot.state = FeatureSlotState.FREE
                    slot.producer_event = None
        return retired

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

    def close(self) -> tuple[int, int]:
        """Fence publication and snapshot ownership without releasing CUDA state.

        The returned pair is ``(active_leases, quarantined_slots)``.  Taking
        that snapshot under the same lock that closes admission lets the
        Target controller decide whether it may detach the pool without a new
        publication racing the shutdown ownership check.
        """
        with self._lock:
            self._closed = True
        # Publication is now fenced. Retire only cancellations whose producer
        # event is already complete; pending work remains counted in _active,
        # and a failed query quarantines the exact owning slot.
        self.poll_cancelled_producers()
        with self._lock:
            return (
                len(self._active),
                sum(
                    slot.state is FeatureSlotState.QUARANTINED
                    for slot in self._slots
                ),
            )

    def _validate_hidden_states(self, hidden_states: torch.Tensor) -> None:
        if hidden_states.device != self.device:
            raise ValueError("feature hidden states must be on the source CUDA device")
        if hidden_states.dtype is not self.dtype:
            raise ValueError("direct GPU source rows must be bfloat16")
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


@dataclass(slots=True)
class _InlinePull:
    copy_stream: torch.cuda.Stream
    gpu_queued: torch.cuda.Event
    gpu_start: torch.cuda.Event
    gpu_done: torch.cuda.Event
    submitted_ns: int
    mapping_started_ns: int
    mapping_done_ns: int
    driver_submit_started_ns: int
    driver_submit_done_ns: int
    feature_bytes: int
    cold_mapping_count: int
    gpu_start_observed_ns: int = 0


class _InlinePullFuture:
    """Future-compatible handle for a pull enqueued on the caller's thread.

    ``done()`` is a nonblocking CUDA event query, so the serving loop observes
    copy completion directly instead of waiting for a helper thread to win the
    GIL from the busy-polling scheduler loop.
    """

    __slots__ = ("_transport", "_pending", "_result", "_exception", "_done")

    def __init__(self, transport: "DraftFeaturePullTransport", pending: _InlinePull):
        self._transport = transport
        self._pending = pending
        self._result: FeaturePullMetrics | None = None
        self._exception: BaseException | None = None
        self._done = False

    def done(self) -> bool:
        if not self._done:
            self._transport._reap_inline(self)
        return self._done

    def result(self, timeout: float | None = None) -> FeaturePullMetrics:
        if not self.done():
            # Only shutdown/worker paths block; the scheduler polls done().
            self._pending.gpu_done.synchronize()
            self._transport._reap_inline(self)
        if self._exception is not None:
            raise self._exception
        return self._result

    def exception(self, timeout: float | None = None) -> BaseException | None:
        if not self.done():
            self._pending.gpu_done.synchronize()
            self._transport._reap_inline(self)
        return self._exception


class DraftFeaturePullTransport:
    """Asynchronously pull one Target feature range into Draft-local VRAM."""

    def __init__(
        self,
        *,
        device: torch.device,
        max_workers: int = 1,
        profile_timing: bool = False,
    ) -> None:
        if max_workers <= 0:
            raise ValueError("feature copy transport requires positive workers")
        requested_device = torch.device(device)
        if requested_device.type != "cuda":
            raise ValueError("direct GPU source pulls require a CUDA device")
        self.device_index = (
            torch.cuda.current_device()
            if requested_device.index is None
            else int(requested_device.index)
        )
        self.device = torch.device("cuda", self.device_index)
        self._profile_timing = bool(profile_timing)
        torch.cuda.set_device(self.device)
        self._runtime = CudaIpcRuntime()
        self._mapping_cache = CudaIpcMappingCache(self._runtime)
        self._stream_local = local()
        self._copy_streams: list[torch.cuda.Stream] = []
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="dt-feature-p2p",
        )
        self._fatal_error: FeatureTransportQuiescenceError | None = None
        self._closed = False
        self._lock = Lock()
        self._admission = Condition(self._lock)
        self._admitted_pulls = 0
        self._lifecycle_fence = False
        # Inline mode enqueues the copy on the calling scheduler thread and
        # completes it by event query. No helper thread is on the hot path.
        self._inline_submit = os.environ.get("DT_FEATURE_PULL_INLINE", "1") == "1"
        self._inline_inflight: set[_InlinePullFuture] = set()
        self._inline_streams: list[torch.cuda.Stream] = []
        self._inline_last_done: list[torch.cuda.Event | None] = []
        self._inline_cursor = 0
        if self._inline_submit:
            for _ in range(min(max_workers, 8)):
                stream = torch.cuda.Stream(device=self.device)
                self._inline_streams.append(stream)
                self._inline_last_done.append(None)
                self._copy_streams.append(stream)

    def submit(
        self,
        *,
        source: FeatureSourceDescriptor,
        destination: torch.Tensor,
    ) -> Future[FeaturePullMetrics]:
        """Queue a direct source-to-destination pull and return immediately."""
        self._validate_submission(source, destination)
        if getattr(self, "_inline_submit", False):
            return self._submit_inline(source, destination)
        admission = getattr(self, "_admission", None)
        lock = self._lock if admission is None else admission
        with lock:
            if self._closed:
                raise FeatureTransportError("feature pull transport is closed")
            if self._fatal_error is not None:
                raise FeatureTransportError(
                    "feature pull transport is quarantined after an unquiesced GPU-copy failure"
                ) from self._fatal_error
            if admission is None:
                return self._executor.submit(self._run_pull, source, destination)
            if self._lifecycle_fence:
                raise FeatureTransportError(
                    "feature pull admission is fenced by peer lifecycle work"
                )
            self._admitted_pulls += 1
            try:
                return self._executor.submit(
                    self._run_admitted, self._run_pull, source, destination
                )
            except Exception:
                self._admitted_pulls -= 1
                self._admission.notify_all()
                raise

    def submit_batch(
        self,
        *,
        sources: Sequence[FeatureSourceDescriptor],
        destinations: Sequence[torch.Tensor],
    ) -> Future[FeaturePullBatchMetrics]:
        """Queue one ordered CUDA batch for a scheduler-owned feature batch."""
        sources = tuple(sources)
        destinations = tuple(destinations)
        self._validate_batch_submission(sources, destinations)
        admission = getattr(self, "_admission", None)
        lock = self._lock if admission is None else admission
        with lock:
            if self._closed:
                raise FeatureTransportError("feature pull transport is closed")
            if self._fatal_error is not None:
                raise FeatureTransportError(
                    "feature pull transport is quarantined after an unquiesced GPU-copy failure"
                ) from self._fatal_error
            if admission is None:
                return self._executor.submit(self._run_batch, sources, destinations)
            if self._lifecycle_fence:
                raise FeatureTransportError(
                    "feature pull admission is fenced by peer lifecycle work"
                )
            self._admitted_pulls += 1
            try:
                return self._executor.submit(
                    self._run_admitted, self._run_batch, sources, destinations
                )
            except Exception:
                self._admitted_pulls -= 1
                self._admission.notify_all()
                raise

    def _run_admitted(self, callback, *args):
        try:
            return callback(*args)
        finally:
            with self._admission:
                self._admitted_pulls -= 1
                self._admission.notify_all()

    @staticmethod
    def query(future: Future[FeaturePullMetrics]) -> FeaturePullMetrics | None:
        """Return completed metrics without waiting, or ``None`` while in flight."""
        return future.result() if future.done() else None

    def wait(self, future: Future[FeaturePullMetrics]) -> FeaturePullMetrics:
        """Wait only in a worker/shutdown path, never a scheduler hot path."""
        return future.result()

    def invalidate_peer(self, target_instance_id: str) -> None:
        admission = getattr(self, "_admission", None)
        if admission is None:
            with self._lock:
                if self._closed:
                    raise FeatureTransportError("feature pull transport is closed")
                if self._fatal_error is not None:
                    raise FeatureTransportError(
                        "cannot invalidate a peer while feature pulls are quarantined"
                    ) from self._fatal_error
                invalidated = self._executor.submit(
                    self._invalidate_peer_if_healthy, target_instance_id
                )
            invalidated.result()
            return
        with admission:
            if self._closed:
                raise FeatureTransportError("feature pull transport is closed")
            if self._fatal_error is not None:
                raise FeatureTransportError(
                    "cannot invalidate a peer while feature pulls are quarantined"
                ) from self._fatal_error
            self._lifecycle_fence = True
            try:
                self._wait_admitted_drained_locked()
                self._mapping_cache.invalidate(target_instance_id)
            finally:
                self._lifecycle_fence = False
                admission.notify_all()

    def _invalidate_peer_if_healthy(self, target_instance_id: str) -> None:
        with self._lock:
            if self._fatal_error is not None:
                raise FeatureTransportError(
                    "peer invalidation was fenced by feature-pull quarantine"
                ) from self._fatal_error
        self._mapping_cache.invalidate(target_instance_id)

    def quarantine(self) -> None:
        """Permanently fence admission without touching CUDA-owned resources.

        Already-running work may be uncertain and is intentionally not
        synchronized, cancelled, unmapped, or released here.  Queued callbacks
        recheck ``_fatal_error`` before mapping IPC memory or issuing a copy.
        """
        with self._lock:
            if self._fatal_error is None:
                self._fatal_error = FeatureTransportQuiescenceError(
                    "feature pull transport was quarantined by lifecycle policy"
                )

    def probe_quiescence(self) -> Future[bool]:
        """Fence all admitted copy work before retrying/releasing state."""
        admission = getattr(self, "_admission", None)
        if admission is None:
            with self._lock:
                if self._closed:
                    raise FeatureTransportError("feature pull transport is closed")
                return self._executor.submit(self._probe_stream_quiescence)
        with admission:
            if self._closed:
                raise FeatureTransportError("feature pull transport is closed")
            if self._lifecycle_fence:
                raise FeatureTransportError(
                    "feature pull lifecycle work is already in progress"
                )
            self._lifecycle_fence = True
            try:
                return self._executor.submit(self._run_quiescence_probe)
            except Exception:
                self._lifecycle_fence = False
                admission.notify_all()
                raise

    def _run_quiescence_probe(self) -> bool:
        try:
            with self._admission:
                self._wait_admitted_drained_locked()
            return self._probe_stream_quiescence()
        finally:
            with self._admission:
                self._lifecycle_fence = False
                self._admission.notify_all()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
        if getattr(self, "_inline_inflight", None):
            try:
                for stream in self._inline_streams:
                    stream.synchronize()
            except Exception:
                with self._lock:
                    self._fatal_error = FeatureTransportQuiescenceError(
                        "inline feature pull did not quiesce at close"
                    )
        # Do not close mappings after an unquiesced failure: CUDA could still
        # access them. Context teardown is safer than remote-memory reuse.
        if self._fatal_error is None:
            self._mapping_cache.close()

    def _probe_stream_quiescence(self) -> bool:
        torch.cuda.set_device(self.device)
        try:
            if hasattr(self, "_copy_streams"):
                with self._lock:
                    streams = tuple(self._copy_streams)
            else:
                streams = (self._copy_stream,)
            for stream in streams:
                stream.synchronize()
        except Exception:
            return False
        return True

    def _wait_admitted_drained_locked(self) -> None:
        """Wait (holding ``_admission``) until no admitted pull remains."""
        while self._admitted_pulls:
            handles = tuple(getattr(self, "_inline_inflight", ()))
            if not handles:
                self._admission.wait()
                continue
            self._admission.release()
            try:
                for handle in handles:
                    self._reap_inline(handle)
            finally:
                self._admission.acquire()
            if self._admitted_pulls:
                self._admission.wait(0.0002)

    def _next_inline_stream(self) -> int:
        """Prefer an idle copy stream so one producer wait does not block others."""
        count = len(self._inline_streams)
        for offset in range(count):
            index = (self._inline_cursor + offset) % count
            last = self._inline_last_done[index]
            if last is None or last.query():
                self._inline_cursor = (index + 1) % count
                return index
        index = self._inline_cursor
        self._inline_cursor = (index + 1) % count
        return index

    def _submit_inline(
        self, source: FeatureSourceDescriptor, destination: torch.Tensor
    ) -> Future[FeaturePullMetrics] | _InlinePullFuture:
        with self._admission:
            if self._closed:
                raise FeatureTransportError("feature pull transport is closed")
            if self._fatal_error is not None:
                raise FeatureTransportError(
                    "feature pull transport is quarantined after an unquiesced GPU-copy failure"
                ) from self._fatal_error
            if self._lifecycle_fence:
                raise FeatureTransportError(
                    "feature pull admission is fenced by peer lifecycle work"
                )
            self._admitted_pulls += 1
        try:
            pending = self._enqueue_inline(source, destination)
        except BaseException as exc:
            with self._admission:
                self._admitted_pulls -= 1
                self._admission.notify_all()
            failed: Future[FeaturePullMetrics] = Future()
            failed.set_exception(exc)
            return failed
        handle = _InlinePullFuture(self, pending)
        with self._admission:
            self._inline_inflight.add(handle)
        return handle

    def _enqueue_inline(
        self, source: FeatureSourceDescriptor, destination: torch.Tensor
    ) -> _InlinePull:
        mapping_started_ns = time.monotonic_ns()
        mapped_base, warm = self._mapping_cache.map(
            peer_instance_id=source.target_instance_id,
            peer_session_id=source.source_session_id,
            pool_generation=source.pool_generation,
            ipc_handle=source.ipc_handle,
        )
        segment = CopySegment(
            source_address=mapped_base + source.allocation_offset + source.byte_offset,
            destination_address=destination.data_ptr(),
            length=source.nbytes,
        )
        source_ready = torch.cuda.Event.from_ipc_handle(
            self.device_index, source.producer_event_handle
        )
        mapping_done_ns = time.monotonic_ns()
        index = self._next_inline_stream()
        copy_stream = self._inline_streams[index]
        timing = self._profile_timing
        gpu_queued = torch.cuda.Event(enable_timing=timing)
        gpu_start = torch.cuda.Event(enable_timing=timing)
        gpu_done = torch.cuda.Event(enable_timing=timing)
        submitted_ns = time.monotonic_ns()
        stream_may_reference_source = False
        try:
            with torch.cuda.stream(copy_stream):
                stream_may_reference_source = True
                gpu_queued.record(copy_stream)
                copy_stream.wait_event(source_ready)
                gpu_start.record(copy_stream)
                driver_submit_started_ns = time.monotonic_ns()
                self._runtime.copy_batch_async((segment,), copy_stream.cuda_stream)
                driver_submit_done_ns = time.monotonic_ns()
                gpu_done.record(copy_stream)
        except Exception:
            if stream_may_reference_source:
                try:
                    copy_stream.synchronize()
                except Exception as fence_exc:
                    fatal = FeatureTransportQuiescenceError(
                        "failed feature pull may still access source or destination VRAM"
                    )
                    with self._lock:
                        self._fatal_error = fatal
                    raise fatal from fence_exc
            raise
        self._inline_last_done[index] = gpu_done
        return _InlinePull(
            copy_stream=copy_stream,
            gpu_queued=gpu_queued,
            gpu_start=gpu_start,
            gpu_done=gpu_done,
            submitted_ns=submitted_ns,
            mapping_started_ns=mapping_started_ns,
            mapping_done_ns=mapping_done_ns,
            driver_submit_started_ns=driver_submit_started_ns,
            driver_submit_done_ns=driver_submit_done_ns,
            feature_bytes=source.nbytes,
            cold_mapping_count=int(not warm),
        )

    def _reap_inline(self, handle: _InlinePullFuture) -> None:
        """Resolve ``handle`` if its copy finished; never blocks."""
        if handle._done:
            return
        pending = handle._pending
        result = None
        error = None
        try:
            if (
                self._profile_timing
                and not pending.gpu_start_observed_ns
                and pending.gpu_start.query()
            ):
                pending.gpu_start_observed_ns = time.monotonic_ns()
            if not pending.gpu_done.query():
                return
            completed_ns = time.monotonic_ns()
            if not pending.gpu_start_observed_ns:
                pending.gpu_start_observed_ns = completed_ns
            timing = self._profile_timing
            result = FeaturePullMetrics(
                submitted_ns=pending.submitted_ns,
                gpu_completed_ns=completed_ns,
                mapping_ns=pending.mapping_done_ns - pending.mapping_started_ns,
                driver_submit_ns=(
                    pending.driver_submit_done_ns - pending.driver_submit_started_ns
                ),
                gpu_copy_ns=(
                    int(pending.gpu_start.elapsed_time(pending.gpu_done) * 1_000_000)
                    if timing
                    else 0
                ),
                feature_bytes=pending.feature_bytes,
                cold_mapping_count=pending.cold_mapping_count,
                worker_started_ns=pending.mapping_started_ns,
                mapping_done_ns=pending.mapping_done_ns,
                driver_submit_started_ns=pending.driver_submit_started_ns,
                driver_submit_done_ns=pending.driver_submit_done_ns,
                cuda_pre_copy_ns=(
                    int(pending.gpu_queued.elapsed_time(pending.gpu_start) * 1_000_000)
                    if timing
                    else 0
                ),
                gpu_start_observed_ns=(
                    pending.gpu_start_observed_ns if timing else 0
                ),
            )
        except Exception as exc:  # noqa: BLE001 - a failed query proves nothing
            error = FeatureTransportQuiescenceError(
                "feature pull completion could not be observed"
            )
            error.__cause__ = exc
            with self._lock:
                self._fatal_error = error
        with self._admission:
            if handle._done:
                return
            handle._result = result
            handle._exception = error
            handle._done = True
            self._inline_inflight.discard(handle)
            self._admitted_pulls -= 1
            self._admission.notify_all()

    def _worker_stream(self):
        """Return one bounded CUDA stream owned by the current copy worker."""

        stream_local = getattr(self, "_stream_local", None)
        if stream_local is None:
            # Compatibility for narrow CPU transport fixtures.
            return self._copy_stream
        stream = getattr(stream_local, "copy_stream", None)
        if stream is None:
            stream = torch.cuda.Stream(device=self.device)
            stream_local.copy_stream = stream
            with self._lock:
                self._copy_streams.append(stream)
        return stream

    def _run_pull(
        self,
        source: FeatureSourceDescriptor,
        destination: torch.Tensor,
    ) -> FeaturePullMetrics:
        batch = self._run_batch((source,), (destination,))
        return FeaturePullMetrics(
            submitted_ns=batch.submitted_ns,
            gpu_completed_ns=batch.gpu_completed_ns,
            mapping_ns=batch.mapping_ns,
            driver_submit_ns=batch.driver_submit_ns,
            gpu_copy_ns=batch.gpu_copy_ns,
            feature_bytes=batch.feature_bytes,
            cold_mapping_count=batch.cold_mapping_count,
            worker_started_ns=batch.worker_started_ns,
            mapping_done_ns=batch.mapping_done_ns,
            driver_submit_started_ns=batch.driver_submit_started_ns,
            driver_submit_done_ns=batch.driver_submit_done_ns,
            cuda_pre_copy_ns=batch.cuda_pre_copy_ns,
            gpu_start_observed_ns=batch.gpu_start_observed_ns,
        )

    def _run_batch(
        self,
        sources: tuple[FeatureSourceDescriptor, ...],
        destinations: tuple[torch.Tensor, ...],
    ) -> FeaturePullBatchMetrics:
        with self._lock:
            if self._fatal_error is not None:
                raise FeatureTransportError(
                    "feature pull was queued before transport quarantine"
                ) from self._fatal_error
        # Only callbacks admitted by the transport fence may touch CUDA state.
        # A callback that passes this check is considered already running; a
        # later quarantine retains all of its possibly referenced resources.
        torch.cuda.set_device(self.device)
        mapping_started_ns = time.monotonic_ns()
        segments = []
        source_events: dict[tuple[str, str, bytes], torch.cuda.Event] = {}
        cold_mapping_count = 0
        feature_bytes = 0
        for source, destination in zip(sources, destinations, strict=True):
            mapped_base, warm = self._mapping_cache.map(
                peer_instance_id=source.target_instance_id,
                peer_session_id=source.source_session_id,
                pool_generation=source.pool_generation,
                ipc_handle=source.ipc_handle,
            )
            cold_mapping_count += int(not warm)
            feature_bytes += source.nbytes
            source_address = (
                mapped_base + source.allocation_offset + source.byte_offset
            )
            segments.append(
                CopySegment(
                    source_address=source_address,
                    destination_address=destination.data_ptr(),
                    length=source.nbytes,
                )
            )
            event_key = (
                source.target_instance_id,
                source.source_session_id,
                source.producer_event_handle,
            )
            if event_key not in source_events:
                source_events[event_key] = torch.cuda.Event.from_ipc_handle(
                    self.device_index, source.producer_event_handle
                )
        mapping_done_ns = time.monotonic_ns()
        return self._copy_batch_and_wait(
            source_ready=tuple(source_events.values()),
            segments=tuple(segments),
            mapping_started_ns=mapping_started_ns,
            mapping_done_ns=mapping_done_ns,
            feature_bytes=feature_bytes,
            cold_mapping_count=cold_mapping_count,
            peer_count=len({source.target_instance_id for source in sources}),
        )

    def _copy_batch_and_wait(
        self,
        *,
        source_ready: tuple[torch.cuda.Event, ...],
        segments: tuple[CopySegment, ...],
        mapping_started_ns: int,
        mapping_done_ns: int,
        feature_bytes: int,
        cold_mapping_count: int,
        peer_count: int,
    ) -> FeaturePullBatchMetrics:
        """Finish a submitted batch only once it cannot touch source memory."""
        copy_stream = self._worker_stream()
        gpu_queued = torch.cuda.Event(enable_timing=True)
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_done = torch.cuda.Event(enable_timing=True)
        submitted_ns = time.monotonic_ns()
        stream_may_reference_source = False
        try:
            with torch.cuda.stream(copy_stream):
                stream_may_reference_source = True
                gpu_queued.record(copy_stream)
                for event in source_ready:
                    copy_stream.wait_event(event)
                gpu_start.record(copy_stream)
                driver_submit_started_ns = time.monotonic_ns()
                self._runtime.copy_batch_async(
                    segments, copy_stream.cuda_stream
                )
                driver_submit_done_ns = time.monotonic_ns()
                gpu_done.record(copy_stream)
            # This synchronization happens only in the bounded copy worker;
            # Target and Draft scheduler loops poll the returned Future.
            gpu_start_observed_ns = 0
            if getattr(self, "_profile_timing", False):
                # This is a host observation of the CUDA start event, not a
                # conversion of the device clock to monotonic wall time.  The
                # worker already blocks for gpu_done below, so observing the
                # earlier boundary does not serialize either scheduler loop.
                gpu_start.synchronize()
                gpu_start_observed_ns = time.monotonic_ns()
            gpu_done.synchronize()
        except Exception:
            if stream_may_reference_source:
                try:
                    copy_stream.synchronize()
                except Exception as fence_exc:
                    fatal = FeatureTransportQuiescenceError(
                        "failed feature pull may still access source or destination VRAM"
                    )
                    with self._lock:
                        self._fatal_error = fatal
                    raise fatal from fence_exc
            raise
        return FeaturePullBatchMetrics(
            submitted_ns=submitted_ns,
            gpu_completed_ns=time.monotonic_ns(),
            mapping_ns=mapping_done_ns - mapping_started_ns,
            driver_submit_ns=driver_submit_done_ns - driver_submit_started_ns,
            gpu_copy_ns=int(gpu_start.elapsed_time(gpu_done) * 1_000_000),
            feature_bytes=feature_bytes,
            cold_mapping_count=cold_mapping_count,
            segment_count=len(segments),
            peer_count=peer_count,
            worker_started_ns=mapping_started_ns,
            mapping_done_ns=mapping_done_ns,
            driver_submit_started_ns=driver_submit_started_ns,
            driver_submit_done_ns=driver_submit_done_ns,
            cuda_pre_copy_ns=int(gpu_queued.elapsed_time(gpu_start) * 1_000_000),
            gpu_start_observed_ns=gpu_start_observed_ns,
        )

    def _validate_batch_submission(
        self,
        sources: tuple[FeatureSourceDescriptor, ...],
        destinations: tuple[torch.Tensor, ...],
    ) -> None:
        if not sources or len(sources) != len(destinations):
            raise ValueError("feature batch requires matching non-empty inputs")
        peer_generations: dict[str, tuple[str, str, str]] = {}
        destination_ranges = []
        for source, destination in zip(sources, destinations, strict=True):
            self._validate_submission(source, destination)
            generation = (
                source.source_session_id,
                source.pool_generation,
                source.transport_session,
            )
            previous = peer_generations.setdefault(
                source.target_instance_id, generation
            )
            if previous != generation:
                raise FeatureTransportError(
                    "one feature batch cannot mix generations from the same peer"
                )
            start = destination.data_ptr()
            destination_ranges.append((start, start + source.nbytes))
        destination_ranges.sort()
        if any(
            destination_ranges[index][1] > destination_ranges[index + 1][0]
            for index in range(len(destination_ranges) - 1)
        ):
            raise ValueError("feature batch destinations must not overlap")

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
