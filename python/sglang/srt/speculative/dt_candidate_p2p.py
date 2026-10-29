"""Target-owned direct-GPU candidate return for Draft-Target serving.

The Target allocates one bounded CUDA arena and one notification row per
candidate slot. A Draft maps the advertised allocations, copies candidate
tokens directly into the reserved Target row, writes authenticated metadata,
and publishes the slot generation last. After the Draft transport reports its
stream completion, the Target enqueues one tiny metadata D2H and observes a
CUDA event without blocking the scheduler thread.

With DT_GPU_READY_RECORD=1 the Target also shares a same-host /dev/shm file
of per-slot READY records.  The Draft queues the candidate copy, metadata words
and then the record (generation last) on its push stream, so the Target CPU
observes READY from host memory without waiting for a Draft-side completion
poll or the direct notice.  The notice stays as the retirement/fallback path.

With DT_PUSH_COMPLETION_RECORD=1 the Draft instead queues a one-block kernel
on its push stream after the candidate copy.  The kernel writes a per-slot
completion record into a second shared file — payload words first, the
reservation's authorization nonce (the seq word) published last with a
system-scope release store.  For a negotiated Target the record carries a
superset of the READY record words, so the legacy READY record writes and
the ZMQ completion notice are both skipped: the Target observes READY only
by scanning the completion records from host memory.

Request, assignment, timeout, and retirement policy remain in the existing DT
controller/router.  This module owns only allocation-relative CUDA transport
and the local slot lifecycle.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import logging
import mmap
import os
import secrets
import subprocess
import time
import uuid
import zlib
from collections import OrderedDict, deque
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import Lock

import msgspec
import numpy as np
import torch
import zmq

from sglang.srt.speculative.dt_cuda_ipc import (
    CopySegment,
    CudaIpcMappingCache,
    CudaIpcRuntime,
    register_buffers,
)

METADATA_WORDS = 5
GENERATION_WORD = 0
NONCE_LOW_WORD = 1
NONCE_HIGH_WORD = 2
VALID_LENGTH_WORD = 3
ROOT_TOKEN_WORD = 4
COMPLETION_RETRY_NS = 100_000_000
COMPLETION_MAX_ATTEMPTS = 100
# Queue bursts instead of silently dropping them: at the measured standalone
# rate (~3k notices/s per Target with retries) a multi-second Target prefill
# stall alone overflows a 4096-deep ROUTER queue.
COMPLETION_SOCKET_HWM = 65536
# One poll call must be able to empty a full queue; the time cap keeps a
# pathological flood from taxing the scheduler thread beyond a few ms.
COMPLETION_POLL_MAX_MESSAGES = COMPLETION_SOCKET_HWM
COMPLETION_POLL_MAX_NS = 4_000_000
# Stop burn-retrying into a full pipe after this many consecutive EAGAINs
# per Target socket; the next progress call tries again.
COMPLETION_RETRY_EAGAIN_LIMIT = 16
# Bounded, deduplicated deferred-ACK queue (one queued ACK per transfer).
COMPLETION_PENDING_ACK_LIMIT = 4096
RETIRED_TRANSFER_LIMIT = 65536
RING_DEBUG = os.environ.get("DT_RING_DEBUG", "0") == "1"
METADATA_READBACK = os.environ.get("DT_CANDIDATE_METADATA_READBACK", "0") == "1"
GPU_READY_RECORD = os.environ.get("DT_GPU_READY_RECORD", "1") == "1"
# One 32-byte host READY record per slot: the metadata words plus the
# producer tag.  The generation word is written last.
READY_RECORD_WORDS = 8
READY_RECORD_BYTES = READY_RECORD_WORDS * 4
PRODUCER_TAG_WORD = 5
PUSH_COMPLETION_RECORD = os.environ.get("DT_PUSH_COMPLETION_RECORD", "1") == "1"
# One 64-byte host completion record per slot: eight u64 words, the seq word
# published last with a system-scope release store.
COMPLETION_RECORD_WORDS = 8
COMPLETION_RECORD_BYTES = COMPLETION_RECORD_WORDS * 8
COMPLETION_SEQ_WORD = 0
COMPLETION_META_WORD = 1  # generation (low u32), valid length (high u32)
COMPLETION_HASH_WORD = 2  # transfer id hash
COMPLETION_TOKEN_WORD = 3  # root token (low u32), producer tag (high u32)
COMPLETION_PRODUCED_WORD = 4  # %globaltimer publication timestamp
COMPLETION_OFFSET_WORD = 5  # producer's globaltimer->realtime offset (0 = none)
GLOBALTIMER_CALIBRATION_SAMPLES = 64
# A best-sample round trip above this cannot bound the offset usefully.
GLOBALTIMER_CALIBRATION_MAX_UNCERTAINTY_NS = 1_000_000

logger = logging.getLogger(__name__)


def producer_tag(draft_session_id: str) -> int:
    """Nonzero 31-bit tag binding a host READY record to its Draft session."""

    return (zlib.crc32(draft_session_id.encode()) & 0x7FFFFFFF) or 1


def transfer_id_hash(transfer_id: str) -> int:
    """64-bit hash binding a host completion record to one transfer id."""

    return int.from_bytes(
        hashlib.blake2b(transfer_id.encode(), digest_size=8).digest(), "little"
    )


class _ReadyRecordFile:
    """A /dev/shm mapping of per-slot records shared by Target and Draft."""

    def __init__(
        self,
        path: str,
        *,
        record_bytes: int = READY_RECORD_BYTES,
        slot_count: int = 0,
        create: bool,
    ) -> None:
        if create and slot_count <= 0:
            raise ValueError("record file needs a positive slot count")
        page = mmap.PAGESIZE
        flags = os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0)
        fd = os.open(path, flags, 0o600)
        try:
            if create:
                nbytes = -(-slot_count * record_bytes // page) * page
                os.ftruncate(fd, nbytes)
            else:
                nbytes = os.fstat(fd).st_size
            if nbytes <= 0 or nbytes % record_bytes:
                raise CandidateTransportError("record file has an invalid size")
            self._mm = mmap.mmap(fd, nbytes)
        finally:
            os.close(fd)
        self.path = path
        self.nbytes = nbytes
        self.slot_capacity = nbytes // record_bytes
        self.words = np.frombuffer(self._mm, dtype=np.uint32).reshape(
            self.slot_capacity, record_bytes // 4
        )
        self.host_address = int(self.words.ctypes.data)

    def close(self, *, unlink: bool) -> None:
        self.words = None
        try:
            self._mm.close()
        except BufferError:
            # A caller still holds a view; the mapping dies with the process.
            pass
        if unlink:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass


def _build_push_completion_library() -> Path:
    source = Path(__file__).with_name("csrc") / "dt_push_completion.cu"
    text = source.read_bytes()
    digest = hashlib.sha256(text).hexdigest()[:16]
    cache = Path(
        os.environ.get(
            "DT_PUSH_COMPLETION_BUILD_DIR",
            Path.home() / ".cache" / "dt-push-completion",
        )
    )
    cache.mkdir(parents=True, exist_ok=True)
    library = cache / f"dt_push_completion-{digest}.so"
    if library.exists():
        return library
    arch = os.environ.get("DT_PUSH_COMPLETION_ARCH", "sm_90")
    with open(cache / ".build.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.exists():
            partial = library.with_suffix(f".so.{os.getpid()}")
            subprocess.run(
                [
                    os.environ.get("NVCC", "nvcc"),
                    "-O2",
                    "-std=c++17",
                    "-shared",
                    "-Xcompiler",
                    "-fPIC",
                    f"-arch={arch}",
                    "-cudart",
                    "static",
                    "-o",
                    str(partial),
                    str(source),
                ],
                check=True,
            )
            os.replace(partial, library)
    return library


class _PushCompletionKernel:
    """ctypes handle for the one-block completion record publication kernel."""

    def __init__(self) -> None:
        lib = ctypes.CDLL(str(_build_push_completion_library()))
        lib.dt_push_completion_launch.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_uint64,
        ]
        lib.dt_push_completion_launch.restype = ctypes.c_int
        lib.dt_push_completion_sample_globaltimer.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint64,
        ]
        lib.dt_push_completion_sample_globaltimer.restype = ctypes.c_int
        self._lib = lib

    def launch(
        self,
        *,
        device: int,
        record_address: int,
        seq: int,
        generation: int,
        valid_length: int,
        record_hash: int,
        root_token: int,
        tag: int,
        globaltimer_offset_ns: int,
        stream: int,
    ) -> None:
        rc = self._lib.dt_push_completion_launch(
            device,
            record_address,
            seq,
            generation,
            valid_length,
            record_hash,
            root_token,
            tag,
            globaltimer_offset_ns,
            stream,
        )
        if rc != 0:
            raise CandidateTransportError(
                f"push completion kernel launch failed with cudaError {rc}"
            )

    def sample_globaltimer(self, *, device: int, address: int, stream: int) -> None:
        """Store the device %globaltimer into one mapped u64."""

        rc = self._lib.dt_push_completion_sample_globaltimer(
            device, address, stream
        )
        if rc != 0:
            raise CandidateTransportError(
                f"globaltimer sample kernel launch failed with cudaError {rc}"
            )


def _calibrate_globaltimer_offset(
    runtime: CudaIpcRuntime, kernel: _PushCompletionKernel, device_index: int
) -> int:
    """Estimate CLOCK_REALTIME minus %globaltimer on the producer's device.

    The completion kernel stamps records with %globaltimer; turning those
    stamps into delivery latencies needs the offset from the GPU timebase to
    host realtime, measured on the GPU whose kernel writes the stamps —
    %globaltimer bases differ across same-host GPUs.  Each sample brackets a
    one-block kernel with realtime reads; the median offset is the estimate
    and the best round trip bounds its uncertainty.
    """

    page = mmap.PAGESIZE
    sample_map = mmap.mmap(-1, page)
    try:
        view = np.frombuffer(sample_map, dtype=np.uint64)
        device_ptr = 0
        try:
            device_ptr = runtime.host_register(int(view.ctypes.data), page)
            offsets = []
            best_uncertainty_ns = None
            for _ in range(GLOBALTIMER_CALIBRATION_SAMPLES):
                rt1 = time.time_ns()
                kernel.sample_globaltimer(
                    device=device_index, address=device_ptr, stream=0
                )
                runtime.stream_synchronize(0)
                rt2 = time.time_ns()
                offsets.append(((rt1 + rt2) // 2) - int(view[0]))
                uncertainty = rt2 - rt1
                if best_uncertainty_ns is None or uncertainty < best_uncertainty_ns:
                    best_uncertainty_ns = uncertainty
            if best_uncertainty_ns > GLOBALTIMER_CALIBRATION_MAX_UNCERTAINTY_NS:
                raise CandidateTransportError(
                    "globaltimer calibration round trip is too noisy"
                )
            return int(np.median(offsets))
        finally:
            if device_ptr:
                runtime.host_unregister(int(view.ctypes.data))
            # Drop the view before closing the mapping.
            view = None
    finally:
        sample_map.close()


class CandidateTransportError(RuntimeError):
    pass


class _RingDebugLog:
    """Rate-limited DT_RING_DEBUG=1 counters for the completion channel."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.total: dict[str, int] = {}
        self.window: dict[str, int] = {}
        self.last_emit_ns = time.monotonic_ns()
        self.last_call_ns = 0
        self.max_gap_ns = 0

    def note_call(self, now_ns: int) -> None:
        if self.last_call_ns:
            gap = now_ns - self.last_call_ns
            if gap > self.max_gap_ns:
                self.max_gap_ns = gap
        self.last_call_ns = now_ns

    def bump(self, key: str, count: int = 1) -> None:
        self.total[key] = self.total.get(key, 0) + count
        self.window[key] = self.window.get(key, 0) + count

    def maybe_emit(self, now_ns: int, *, force: bool = False, extra: str = "") -> None:
        if not force and now_ns - self.last_emit_ns < 1_000_000_000:
            return
        self.last_emit_ns = now_ns
        gap_ms = self.max_gap_ns / 1e6
        self.max_gap_ns = 0
        window, self.window = self.window, {}
        logger.info(
            "DT_RING_DEBUG %s: window=%s total=%s max_call_gap_ms=%.1f%s",
            self.label,
            dict(sorted(window.items())),
            dict(sorted(self.total.items())),
            gap_ms,
            extra,
        )


class CandidateSlotState(StrEnum):
    FREE = "free"
    ARMED = "armed"
    READY = "ready"
    READING = "reading"
    CANCELLED = "cancelled"
    QUARANTINED = "quarantined"


@dataclass(frozen=True, slots=True)
class CandidateCompletion:
    transfer_id: str
    slot_id: int
    generation: int
    authorization_nonce: int
    valid_length: int
    root_token: int
    draft_instance_id: str
    draft_session_id: str
    observed_ns: int
    notice_queued_ns: int = 0
    notice_received_ns: int = 0
    metadata_submitted_ns: int = 0
    metadata_gpu_ns: int = 0
    ready_source: str = "notice"
    # %globaltimer stamp written by the producer's completion kernel and the
    # publication->discovery latency derived from it using the record's
    # producer-stamped globaltimer offset; both 0 on paths that have no
    # GPU-stamped record (notice, READY record) or an uncalibrated producer.
    producer_globaltimer_ns: int = 0
    delivery_ns: int = 0


@dataclass(slots=True)
class _CandidateSlot:
    generation: int = 0
    state: CandidateSlotState = CandidateSlotState.FREE
    transfer_id: str = ""
    authorization_nonce: int = 0
    wait_stream: torch.cuda.Stream | None = None
    completion_event: torch.cuda.Event | None = None
    metadata_start_event: torch.cuda.Event | None = None
    metadata_host: torch.Tensor | None = None
    draft_instance_id: str = ""
    draft_session_id: str = ""
    completion_draft_instance_id: str = ""
    completion_draft_session_id: str = ""
    completion: CandidateCompletion | None = None
    read_event: torch.cuda.Event | None = None
    cancel_pending: bool = False
    notice_queued_ns: int = 0
    notice_received_ns: int = 0
    metadata_submitted_ns: int = 0


@dataclass(slots=True)
class _PendingCompletionNotice:
    notice: object
    socket: zmq.Socket
    encoded: bytes
    attempts: int = 0
    next_retry_ns: int = 0
    # Sends that failed with EAGAIN (peer queue full).  Tracked separately so
    # the retry budget measures Target silence, not local pipe pressure.
    send_failures: int = 0


class TargetCandidateArena:
    """Bounded fixed-address rows with transport host completion."""

    def __init__(
        self,
        *,
        target_instance_id: str,
        target_session_id: str,
        device: torch.device,
        slot_count: int,
        row_width: int,
        profile_timing: bool = False,
    ) -> None:
        if not target_instance_id or not target_session_id:
            raise ValueError("Target candidate identity must not be empty")
        if slot_count <= 0 or row_width <= 0:
            raise ValueError("candidate arena dimensions must be positive")
        requested = torch.device(device)
        if requested.type != "cuda":
            raise ValueError("candidate arena requires CUDA")
        self.device_index = (
            torch.cuda.current_device()
            if requested.index is None
            else int(requested.index)
        )
        self.device = torch.device("cuda", self.device_index)
        torch.cuda.set_device(self.device)
        self.target_instance_id = target_instance_id
        self.target_session_id = target_session_id
        self.pool_generation = uuid.uuid4().hex
        self.transport_session = uuid.uuid4().hex
        self.slot_count = int(slot_count)
        self.row_width = int(row_width)
        # Set whenever a completion parks in READY or an assignment lands;
        # poll()'s parked-completion pickup scan runs only while this is set,
        # because nothing else can newly expose a parked completion.
        self._ready_pickup_pending = False
        self._profile_timing = bool(profile_timing)
        self._row_nbytes = self.row_width * 4
        self._metadata_row_nbytes = METADATA_WORDS * 4
        self._runtime = CudaIpcRuntime()
        self.storage = torch.empty(
            (self.slot_count, self.row_width),
            dtype=torch.int32,
            device=self.device,
        )
        self._metadata = torch.zeros(
            (self.slot_count, METADATA_WORDS),
            dtype=torch.int32,
            device=self.device,
        )
        self._allocations, buffers = register_buffers(
            self._runtime,
            (
                (self.storage.data_ptr(), self.storage.nbytes, self._row_nbytes),
                (
                    self._metadata.data_ptr(),
                    self._metadata.nbytes,
                    self._metadata_row_nbytes,
                ),
            ),
        )
        if len(buffers) != 2:
            raise CandidateTransportError("candidate arena registration is incomplete")
        self._data_buffer, self._metadata_buffer = buffers
        self._slots = [_CandidateSlot() for _ in range(self.slot_count)]
        self._active: dict[str, int] = {}
        self._poll_cursor = 0
        self._retired_transfers: OrderedDict[str, None] = OrderedDict()
        self._lock = Lock()
        self._closed = False
        self.device_uuid = str(torch.cuda.get_device_properties(self.device).uuid)
        self._zmq_context = zmq.Context.instance()
        self._completion_socket = self._zmq_context.socket(zmq.ROUTER)
        self._completion_socket.setsockopt(zmq.LINGER, 0)
        self._completion_socket.setsockopt(zmq.RCVHWM, COMPLETION_SOCKET_HWM)
        self._completion_socket.setsockopt(zmq.SNDHWM, COMPLETION_SOCKET_HWM)
        # Without ROUTER_MANDATORY a ROUTER send to a backlogged or departed
        # peer silently succeeds while dropping the message, which hides ACK
        # loss from every counter.  Mandatory turns the two failure modes
        # into explicit errors: EAGAIN for a full peer queue (retryable),
        # EHOSTUNREACH for a departed peer (droppable).
        self._completion_socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self._completion_socket.bind("tcp://127.0.0.1:*")
        self.completion_endpoint = self._completion_socket.getsockopt_string(
            zmq.LAST_ENDPOINT
        )
        self._notice_decoder = msgspec.msgpack.Decoder()
        self._ack_encoder = msgspec.msgpack.Encoder()
        self._recent_notice_keys: deque[tuple[object, ...]] = deque()
        self._recent_notice_set: set[tuple[object, ...]] = set()
        self._recent_notice_limit = max(16, self.slot_count * 4)
        # ACKs whose send failed are retried on later polls instead of being
        # dropped: a backlogged Draft must not force notice retries.  The
        # queue is bounded and holds at most one ACK per transfer; a revoked
        # producer's entries are purged in revoke_producer.
        self._pending_acks: deque[tuple[bytes, str, bytes]] = deque()
        self._pending_ack_keys: set[tuple[bytes, str]] = set()
        self._ring_debug: _RingDebugLog | None = (
            _RingDebugLog(f"target-arena:{target_instance_id}")
            if RING_DEBUG
            else None
        )
        self._ready_records: _ReadyRecordFile | None = None
        self.ready_record_path = ""
        self._completion_records: _ReadyRecordFile | None = None
        self.completion_record_path = ""
        if GPU_READY_RECORD or PUSH_COMPLETION_RECORD:
            # Generation/tag a slot's record must carry to become READY; 0 means
            # the slot is not waiting for a record (unassigned, done, cancelled).
            self._record_generation = np.zeros(self.slot_count, dtype=np.uint32)
            self._record_tag = np.zeros(self.slot_count, dtype=np.uint32)
        if GPU_READY_RECORD:
            self._ready_records = _ReadyRecordFile(
                f"/dev/shm/dt-ready-{self.transport_session}",
                slot_count=self.slot_count,
                create=True,
            )
            self.ready_record_path = self._ready_records.path
        if PUSH_COMPLETION_RECORD:
            self._completion_records = _ReadyRecordFile(
                f"/dev/shm/dt-completion-{self.transport_session}",
                record_bytes=COMPLETION_RECORD_BYTES,
                slot_count=self.slot_count,
                create=True,
            )
            self.completion_record_path = self._completion_records.path
            try:
                # The Draft kernel writes through its own registration of the
                # same pages; pin this mapping for the symmetric lifecycle.
                self._runtime.host_register(
                    self._completion_records.host_address,
                    self._completion_records.nbytes,
                )
            except Exception:
                self._completion_words = None
                self.completion_record_path = ""
                self._completion_records.close(unlink=True)
                self._completion_records = None
                raise
            self._completion_words = self._completion_records.words.view(
                np.uint64
            ).reshape(
                self._completion_records.slot_capacity, COMPLETION_RECORD_WORDS
            )
            # Reservation nonce a slot's completion record must carry; 0 means
            # the slot is not waiting for a record (unassigned, done,
            # cancelled).
            self._record_seq_expected = np.zeros(self.slot_count, dtype=np.uint64)

    @property
    def free_slot_count(self) -> int:
        self.poll_releases()
        with self._lock:
            return sum(slot.state is CandidateSlotState.FREE for slot in self._slots)

    @property
    def quarantined_slot_count(self) -> int:
        with self._lock:
            return sum(
                slot.state is CandidateSlotState.QUARANTINED for slot in self._slots
            )

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    def registration(self):
        """Return the incarnation-scoped allocation material sent once."""

        from dt_protocol import CandidateArenaRegistration

        data_allocation = self._allocations[self._data_buffer.allocation_index]
        metadata_allocation = self._allocations[
            self._metadata_buffer.allocation_index
        ]
        return CandidateArenaRegistration(
            target_instance_id=self.target_instance_id,
            target_session_id=self.target_session_id,
            pool_generation=self.pool_generation,
            transport_session=self.transport_session,
            target_device_id=self.device_index,
            device_uuid=self.device_uuid,
            row_width=self.row_width,
            data_allocation_length=data_allocation.length,
            data_ipc_handle=data_allocation.ipc_handle,
            metadata_allocation_length=metadata_allocation.length,
            metadata_ipc_handle=metadata_allocation.ipc_handle,
            completion_endpoint=self.completion_endpoint,
            ready_record_path=self.ready_record_path,
            completion_record_path=self.completion_record_path,
        )

    def reserve(self, transfer_id: str):
        """Reserve and arm one entry, returning a protocol descriptor."""

        from dt_protocol import CandidateDestinationRef

        if not transfer_id:
            raise ValueError("candidate transfer id must not be empty")
        with self._lock:
            if self._closed:
                raise CandidateTransportError("candidate arena is closed")
            if transfer_id in self._active:
                raise CandidateTransportError("candidate transfer is already reserved")
            slot_id = next(
                (
                    index
                    for index, slot in enumerate(self._slots)
                    if slot.state is CandidateSlotState.FREE
                ),
                None,
            )
            if slot_id is None:
                raise CandidateTransportError("all candidate arena slots are active")
            slot = self._slots[slot_id]
            generation = slot.generation + 1
            if generation > 0x7FFFFFFF:
                raise CandidateTransportError("candidate slot generation exhausted")
            nonce = secrets.randbits(63) or 1
            wait_stream = torch.cuda.Stream(device=self.device)
            metadata_host = torch.empty(
                (METADATA_WORDS,), dtype=torch.int32, device="cpu", pin_memory=True
            )
            slot.generation = generation
            slot.state = CandidateSlotState.ARMED
            slot.transfer_id = transfer_id
            slot.authorization_nonce = nonce
            slot.wait_stream = wait_stream
            slot.completion_event = None
            slot.metadata_start_event = None
            slot.metadata_host = metadata_host
            slot.draft_instance_id = ""
            slot.draft_session_id = ""
            slot.completion = None
            slot.read_event = None
            slot.cancel_pending = False
            slot.notice_queued_ns = 0
            slot.notice_received_ns = 0
            slot.metadata_submitted_ns = 0
            self._active[transfer_id] = slot_id
            if getattr(self, "_record_generation", None) is not None:
                self._record_generation[slot_id] = 0
            if getattr(self, "_record_seq_expected", None) is not None:
                self._record_seq_expected[slot_id] = 0

        data_allocation = self._allocations[self._data_buffer.allocation_index]
        metadata_allocation = self._allocations[
            self._metadata_buffer.allocation_index
        ]
        return CandidateDestinationRef(
            target_instance_id=self.target_instance_id,
            target_session_id=self.target_session_id,
            pool_generation=self.pool_generation,
            transport_session=self.transport_session,
            transfer_id=transfer_id,
            target_device_id=self.device_index,
            device_uuid=self.device_uuid,
            slot_id=slot_id,
            generation=generation,
            authorization_nonce=nonce,
            dtype="int32",
            row_width=self.row_width,
            nbytes=self._row_nbytes,
            data_allocation_length=data_allocation.length,
            data_allocation_offset=(
                self._data_buffer.allocation_offset + slot_id * self._row_nbytes
            ),
            data_ipc_handle=b"",
            metadata_allocation_length=metadata_allocation.length,
            metadata_allocation_offset=(
                self._metadata_buffer.allocation_offset
                + slot_id * self._metadata_row_nbytes
            ),
            metadata_ipc_handle=b"",
            notification_backend="transport-host-completion",
        )

    def authorize(self, assignment) -> None:
        """Bind an armed descriptor to the Coordinator-selected Draft session."""

        with self._lock:
            slot_id = self._active.get(assignment.transfer_id)
            if slot_id is None:
                if assignment.transfer_id in getattr(self, "_retired_transfers", ()):
                    return
                raise CandidateTransportError("assignment names no active destination")
            slot = self._slots[slot_id]
            if slot.draft_instance_id and (
                slot.draft_instance_id,
                slot.draft_session_id,
            ) == (assignment.draft_instance_id, assignment.draft_session_id):
                if assignment.authorization_nonce != slot.authorization_nonce:
                    raise CandidateTransportError("candidate assignment nonce mismatch")
                # Already authorized for this producer (owner lease at ring
                # write); the slot may have moved on to READING or cancel.
                return
            assignable_cancel = (
                slot.state is CandidateSlotState.QUARANTINED
                and slot.cancel_pending
            )
            if slot.state not in {
                CandidateSlotState.ARMED,
                CandidateSlotState.READY,
            } and not assignable_cancel:
                raise CandidateTransportError("candidate destination is not assignable")
            if assignment.authorization_nonce != slot.authorization_nonce:
                raise CandidateTransportError("candidate assignment nonce mismatch")
            if not assignment.draft_instance_id or not assignment.draft_session_id:
                raise CandidateTransportError("candidate assignment identity is incomplete")
            if slot.draft_instance_id and (
                slot.draft_instance_id,
                slot.draft_session_id,
            ) != (assignment.draft_instance_id, assignment.draft_session_id):
                raise CandidateTransportError("candidate producer assignment changed")
            if slot.completion_draft_instance_id and (
                slot.completion_draft_instance_id,
                slot.completion_draft_session_id,
            ) != (assignment.draft_instance_id, assignment.draft_session_id):
                slot.state = CandidateSlotState.QUARANTINED
                raise CandidateTransportError(
                    "candidate host completion producer does not match assignment"
                )
            slot.draft_instance_id = assignment.draft_instance_id
            slot.draft_session_id = assignment.draft_session_id
            self._ready_pickup_pending = True
            if (
                getattr(self, "_record_generation", None) is not None
                and (
                    (slot.state is CandidateSlotState.ARMED and not slot.cancel_pending)
                    or (PUSH_COMPLETION_RECORD and assignable_cancel)
                )
                and not slot.completion_draft_instance_id
            ):
                # A record is accepted only from the assigned producer, so the
                # READY identity is the assignment's, as on the notice path.
                # A cancel-pending slot keeps the expectation so the cancelled
                # push's completion record can authenticate and retire it.
                self._record_tag[slot_id] = producer_tag(assignment.draft_session_id)
                self._record_generation[slot_id] = slot.generation
                if getattr(self, "_record_seq_expected", None) is not None:
                    self._record_seq_expected[slot_id] = slot.authorization_nonce

    def notify_host_completion(
        self,
        transfer_id: str,
        *,
        draft_instance_id: str,
        draft_session_id: str,
        notice_queued_ns: int = 0,
        notice_received_ns: int = 0,
        valid_length: int = 0,
        root_token: int = 0,
    ) -> None:
        """Arm the tiny metadata read after transport reports remote completion.

        A successful completion whose notice carries the published metadata is
        made READY directly: the producer sends the notice only after its push
        event completed, so the payload is already in this arena and the
        readback would only re-read words the authenticated notice states.
        Cancellation tombstones keep the readback, which proves remote
        quiescence before a slot is reused.
        """

        if not draft_instance_id or not draft_session_id:
            raise ValueError("candidate host completion identity is incomplete")
        with self._lock:
            slot_id = self._active.get(transfer_id)
            if slot_id is None:
                return
            slot = self._slots[slot_id]
            identity = (draft_instance_id, draft_session_id)
            observed = (
                slot.completion_draft_instance_id,
                slot.completion_draft_session_id,
            )
            if observed != ("", ""):
                if observed != identity:
                    slot.state = CandidateSlotState.QUARANTINED
                    raise CandidateTransportError(
                        "candidate host completion producer changed"
                    )
                return
            if slot.state not in {
                CandidateSlotState.ARMED,
                CandidateSlotState.QUARANTINED,
            }:
                return
            if slot.draft_instance_id and (
                slot.draft_instance_id,
                slot.draft_session_id,
            ) != identity:
                slot.state = CandidateSlotState.QUARANTINED
                raise CandidateTransportError(
                    "candidate host completion producer is unauthorized"
                )
            wait_stream = slot.wait_stream
            metadata_host = slot.metadata_host
            generation = slot.generation
            if wait_stream is None or metadata_host is None:
                slot.state = CandidateSlotState.QUARANTINED
                raise CandidateTransportError(
                    "candidate host completion resources are unavailable"
                )
            slot.completion_draft_instance_id = draft_instance_id
            slot.completion_draft_session_id = draft_session_id
            slot.notice_queued_ns = notice_queued_ns
            slot.notice_received_ns = notice_received_ns
            if (
                not METADATA_READBACK
                and valid_length == self.row_width
                and slot.state is CandidateSlotState.ARMED
                and not slot.cancel_pending
            ):
                observed_ns = time.monotonic_ns()
                slot.completion = CandidateCompletion(
                    transfer_id=slot.transfer_id,
                    slot_id=slot_id,
                    generation=slot.generation,
                    authorization_nonce=slot.authorization_nonce,
                    valid_length=valid_length,
                    root_token=root_token,
                    draft_instance_id=draft_instance_id,
                    draft_session_id=draft_session_id,
                    observed_ns=observed_ns,
                    notice_queued_ns=notice_queued_ns,
                    notice_received_ns=notice_received_ns,
                    # No readback is issued; record it as instantaneous so
                    # traces keep one monotonic receive->observe chain.
                    metadata_submitted_ns=observed_ns,
                )
                # Same parked-before-ACK semantics as the readback path: poll()
                # exposes it only once the assignment identity matches.
                slot.state = CandidateSlotState.READY
                self._ready_pickup_pending = True
                return

        timing_enabled = bool(getattr(self, "_profile_timing", False))
        event = torch.cuda.Event(enable_timing=timing_enabled)
        start_event = (
            torch.cuda.Event(enable_timing=True)
            if timing_enabled
            else None
        )
        metadata_submitted_ns = time.monotonic_ns()
        try:
            with torch.cuda.stream(wait_stream):
                if start_event is not None:
                    start_event.record(wait_stream)
                metadata_host.copy_(self._metadata[slot_id], non_blocking=True)
                event.record(wait_stream)
        except Exception:
            with self._lock:
                current_id = self._active.get(transfer_id)
                if current_id == slot_id:
                    self._slots[slot_id].state = CandidateSlotState.QUARANTINED
            raise
        with self._lock:
            if self._active.get(transfer_id) != slot_id:
                return
            slot = self._slots[slot_id]
            if slot.generation != generation:
                raise CandidateTransportError("candidate slot changed while arming")
            slot.completion_event = event
            slot.metadata_start_event = start_event
            slot.metadata_submitted_ns = metadata_submitted_ns

    @staticmethod
    def _notice_key(notice) -> tuple[object, ...]:
        return (
            notice.transfer_id,
            notice.slot_id,
            notice.generation,
            notice.authorization_nonce,
            notice.draft_instance_id,
            notice.draft_session_id,
        )

    def _remember_notice(self, key: tuple[object, ...]) -> None:
        if key in self._recent_notice_set:
            return
        self._recent_notice_keys.append(key)
        self._recent_notice_set.add(key)
        while len(self._recent_notice_keys) > self._recent_notice_limit:
            self._recent_notice_set.discard(self._recent_notice_keys.popleft())

    def _accept_completion_notice(
        self, notice, routing_id: bytes, *, received_ns: int
    ) -> bool:
        """Validate and arm one direct notice; return whether it is ACKable."""

        from dt_protocol import PROTOCOL_VERSION

        dbg = self._ring_debug
        key = self._notice_key(notice)
        if key in self._recent_notice_set:
            if dbg is not None:
                dbg.bump("accept_duplicate")
            return True
        if (
            notice.protocol_version != PROTOCOL_VERSION
            or notice.target_instance_id != self.target_instance_id
            or notice.target_session_id != self.target_session_id
            or notice.pool_generation != self.pool_generation
            or notice.transport_session != self.transport_session
            or not notice.draft_instance_id
            or not notice.draft_session_id
            or routing_id != notice.draft_session_id.encode()
        ):
            if dbg is not None:
                dbg.bump("reject_identity")
                if routing_id != notice.draft_session_id.encode():
                    dbg.bump("reject_identity_routing")
                elif notice.target_session_id != self.target_session_id:
                    dbg.bump("reject_identity_session")
                elif notice.transport_session != self.transport_session:
                    dbg.bump("reject_identity_transport")
                elif notice.pool_generation != self.pool_generation:
                    dbg.bump("reject_identity_pool_generation")
            return False
        with self._lock:
            slot_id = self._active.get(notice.transfer_id)
            if slot_id is None:
                # A notice for a transfer this arena already retired (READY
                # from its host record, read, and freed) is a late duplicate;
                # ACK it so the producer stops retrying.
                retired = notice.transfer_id in self._retired_transfers
                if dbg is not None:
                    dbg.bump(
                        "accept_retired" if retired else "reject_unknown_transfer"
                    )
                    if not retired:
                        dbg.maybe_emit(
                            time.monotonic_ns(),
                            force=True,
                            extra=f" unknown_transfer={notice.transfer_id}",
                        )
                return retired
            if slot_id != notice.slot_id:
                if dbg is not None:
                    dbg.bump("reject_slot_mismatch")
                    dbg.maybe_emit(
                        time.monotonic_ns(),
                        force=True,
                        extra=(
                            f" slot_mismatch transfer={notice.transfer_id}"
                            f" active_slot={slot_id} notice_slot={notice.slot_id}"
                        ),
                    )
                return False
            slot = self._slots[slot_id]
            if (
                slot.generation != notice.generation
                or slot.authorization_nonce != notice.authorization_nonce
            ):
                slot.state = CandidateSlotState.QUARANTINED
                raise CandidateTransportError(
                    "candidate direct completion generation/authentication mismatch"
                )
        self.notify_host_completion(
            notice.transfer_id,
            draft_instance_id=notice.draft_instance_id,
            draft_session_id=notice.draft_session_id,
            notice_queued_ns=notice.sent_ns,
            notice_received_ns=received_ns,
            valid_length=0 if notice.cancelled else notice.valid_length,
            root_token=notice.root_token,
        )
        self._remember_notice(key)
        if dbg is not None:
            dbg.bump("accept_new")
        return True

    def revoke_producer(self, draft_session_id: str, *, quiescent: bool) -> int:
        """Retire unfinished slots authorized to a revoked Draft session.

        ``quiescent`` means the producer process is proven gone (it can no
        longer write this arena), so its ARMED or cancel-pending slots free
        now; otherwise they stay quarantined.  READY/READING slots already hold
        a complete payload and follow the normal read lifecycle.
        """

        retired = 0
        with self._lock:
            if quiescent:
                # The producer can no longer write: later cancels of slots it
                # was (or is) assigned need no tombstone.
                revoked = getattr(self, "_revoked_producers", None)
                if revoked is None:
                    revoked = self._revoked_producers = set()
                revoked.add(draft_session_id)
            for slot_id, slot in enumerate(self._slots):
                if slot.draft_session_id != draft_session_id or slot.state not in {
                    CandidateSlotState.ARMED,
                    CandidateSlotState.QUARANTINED,
                }:
                    continue
                if quiescent:
                    self._free_locked(slot_id)
                else:
                    slot.state = CandidateSlotState.QUARANTINED
                retired += 1
        self._drop_peer_pending_acks(draft_session_id.encode())
        return retired

    def has_active(self, transfer_id: str) -> bool:
        with self._lock:
            return transfer_id in self._active

    def _drop_peer_pending_acks(self, routing_id: bytes) -> None:
        """Forget deferred ACKs for a peer that can no longer receive them."""

        if not self._pending_acks:
            return
        self._pending_acks = deque(
            entry for entry in self._pending_acks if entry[0] != routing_id
        )
        self._pending_ack_keys = {
            key for key in self._pending_ack_keys if key[0] != routing_id
        }

    def _defer_ack(self, routing_id: bytes, transfer_id: str, encoded: bytes) -> None:
        """Queue one ACK for a later resend, bounded and deduplicated.

        A full queue drops the newest ACK: the Draft's idempotent notice
        retry regenerates one within COMPLETION_RETRY_NS, so no ACK loss is
        permanent while the producer lives.
        """

        key = (routing_id, transfer_id)
        if key in self._pending_ack_keys:
            return
        if len(self._pending_acks) >= COMPLETION_PENDING_ACK_LIMIT:
            if self._ring_debug is not None:
                self._ring_debug.bump("ack_defer_queue_full")
            return
        self._pending_acks.append((routing_id, transfer_id, encoded))
        self._pending_ack_keys.add(key)
        if self._ring_debug is not None:
            self._ring_debug.bump("ack_deferred_eagain")

    def _flush_pending_acks(self, deadline_ns: int) -> None:
        """Best-effort resend of deferred ACKs within the poll's deadline."""

        dbg = self._ring_debug
        flushed = 0
        while self._pending_acks:
            if flushed and time.monotonic_ns() >= deadline_ns:
                if dbg is not None:
                    dbg.bump("ack_flush_deadline")
                break
            routing_id, transfer_id, encoded = self._pending_acks[0]
            try:
                self._completion_socket.send_multipart(
                    (routing_id, encoded), flags=zmq.NOBLOCK
                )
            except zmq.Again:
                # Peer queue still full; keep the entry for the next poll.
                break
            except zmq.ZMQError as exc:
                if exc.errno != zmq.EHOSTUNREACH:
                    raise
                # Peer is gone: drop its whole queue, the producer revoke
                # path retires its slots separately.
                if dbg is not None:
                    dbg.bump("ack_dropped_peer_gone")
                self._drop_peer_pending_acks(routing_id)
                continue
            self._pending_acks.popleft()
            self._pending_ack_keys.discard((routing_id, transfer_id))
            flushed += 1
            if dbg is not None:
                dbg.bump("ack_sent")
        if dbg is not None and flushed:
            dbg.bump("ack_resent", flushed)

    def poll_completion_notices(
        self,
        *,
        max_messages: int = COMPLETION_POLL_MAX_MESSAGES,
        max_ns: int = COMPLETION_POLL_MAX_NS,
    ) -> int:
        """Drain bounded payload-free Draft completions from the direct endpoint.

        The count cap lets one call empty a full socket queue; the time cap
        bounds how long a flood can hold the scheduler thread.  Both caps only
        engage while a backlog exists — steady-state drains return on the
        first empty read.
        """

        from dt_protocol import (
            PROTOCOL_VERSION,
            CandidateCompletionAck,
            CandidateCompletionNotice,
        )

        if max_messages <= 0:
            raise ValueError("max_messages must be positive")
        if max_ns <= 0:
            raise ValueError("max_ns must be positive")
        dbg = self._ring_debug
        started_ns = time.monotonic_ns()
        if dbg is not None:
            dbg.note_call(started_ns)
        self._flush_pending_acks(started_ns + max_ns)
        accepted = 0
        drained = 0
        timed_out = False
        for _ in range(max_messages):
            if (
                drained % 64 == 0
                and drained > 0
                and time.monotonic_ns() - started_ns >= max_ns
            ):
                timed_out = True
                break
            try:
                parts = self._completion_socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            drained += 1
            if len(parts) != 2:
                if dbg is not None:
                    dbg.bump("malformed_parts")
                continue
            routing_id, payload = parts
            try:
                notice = msgspec.msgpack.decode(payload, type=CandidateCompletionNotice)
            except (msgspec.DecodeError, msgspec.ValidationError):
                if dbg is not None:
                    dbg.bump("malformed_decode")
                continue
            received_ns = time.monotonic_ns()
            if not self._accept_completion_notice(
                notice, routing_id, received_ns=received_ns
            ):
                continue
            ack = CandidateCompletionAck(
                protocol_version=PROTOCOL_VERSION,
                target_instance_id=self.target_instance_id,
                target_session_id=self.target_session_id,
                transport_session=self.transport_session,
                transfer_id=notice.transfer_id,
                generation=notice.generation,
                authorization_nonce=notice.authorization_nonce,
                draft_instance_id=notice.draft_instance_id,
                draft_session_id=notice.draft_session_id,
                accepted_ns=time.monotonic_ns(),
            )
            encoded = self._ack_encoder.encode(ack)
            try:
                self._completion_socket.send_multipart(
                    (routing_id, encoded), flags=zmq.NOBLOCK
                )
                if dbg is not None:
                    dbg.bump("ack_sent")
            except zmq.Again:
                # Do not block Target; resend on a later poll instead of
                # dropping the ACK and depending on the Draft's notice retry.
                self._defer_ack(routing_id, notice.transfer_id, encoded)
            except zmq.ZMQError as exc:
                if exc.errno != zmq.EHOSTUNREACH:
                    raise
                # The Draft is gone; its producer revoke retires its slots.
                if dbg is not None:
                    dbg.bump("ack_dropped_peer_gone")
            accepted += 1
        if dbg is not None:
            if drained:
                dbg.bump("notices_drained", drained)
            if drained == max_messages or timed_out:
                dbg.bump("drain_cap_reached")
                dbg.maybe_emit(
                    time.monotonic_ns(),
                    force=True,
                    extra=(
                        f" drain cap reached drained={drained}"
                        f" timed_out={timed_out}"
                    ),
                )
            dbg.maybe_emit(time.monotonic_ns())
        return accepted

    def poll(self, *, max_queries: int = 64) -> tuple[CandidateCompletion, ...]:
        """Observe ready events with bounded round-robin scanning."""

        if max_queries <= 0:
            raise ValueError("max_queries must be positive")
        self.poll_ready_records()
        self.poll_completion_records()
        completed: list[CandidateCompletion] = []
        queries = min(max_queries, self.slot_count)
        for offset in range(queries):
            slot_id = (self._poll_cursor + offset) % self.slot_count
            with self._lock:
                slot = self._slots[slot_id]
                if slot.state is not CandidateSlotState.ARMED:
                    continue
                event = slot.completion_event
            if event is None:
                continue
            try:
                ready = bool(event.query())
            except Exception as exc:
                with self._lock:
                    slot.state = CandidateSlotState.QUARANTINED
                raise CandidateTransportError(
                    "candidate completion event query failed"
                ) from exc
            if not ready:
                continue
            with self._lock:
                if slot.state is not CandidateSlotState.ARMED:
                    continue
                values = [int(value) & 0xFFFFFFFF for value in slot.metadata_host]
                nonce = values[NONCE_LOW_WORD] | (values[NONCE_HIGH_WORD] << 32)
                if (
                    values[GENERATION_WORD] != slot.generation
                    or nonce != slot.authorization_nonce
                    or values[VALID_LENGTH_WORD] != self.row_width
                ):
                    slot.state = CandidateSlotState.QUARANTINED
                    raise CandidateTransportError(
                        "candidate notification metadata failed generation/authentication"
                    )
                root = values[ROOT_TOKEN_WORD]
                if root & 0x80000000:
                    root -= 1 << 32
                slot.completion = CandidateCompletion(
                    transfer_id=slot.transfer_id,
                    slot_id=slot_id,
                    generation=slot.generation,
                    authorization_nonce=slot.authorization_nonce,
                    valid_length=values[VALID_LENGTH_WORD],
                    root_token=root,
                    draft_instance_id=slot.completion_draft_instance_id,
                    draft_session_id=slot.completion_draft_session_id,
                    observed_ns=time.monotonic_ns(),
                    notice_queued_ns=slot.notice_queued_ns,
                    notice_received_ns=slot.notice_received_ns,
                    metadata_submitted_ns=slot.metadata_submitted_ns,
                    metadata_gpu_ns=(
                        int(
                            slot.metadata_start_event.elapsed_time(event)
                            * 1_000_000
                        )
                        if slot.metadata_start_event is not None
                        else 0
                    ),
                )
                # Completion may beat the assignment ACK.  Retain it in READY
                # but do not expose it to the scheduler until both identities match.
                slot.state = CandidateSlotState.READY
                self._ready_pickup_pending = True
                if (
                    slot.draft_instance_id
                    and (
                        slot.draft_instance_id,
                        slot.draft_session_id,
                    )
                    == (
                        slot.completion_draft_instance_id,
                        slot.completion_draft_session_id,
                    )
                ):
                    completed.append(slot.completion)
        self._poll_cursor = (self._poll_cursor + queries) % self.slot_count
        # Pick up completions that were parked before their assignment arrived.
        # Only an authorize() or a fresh READY entry (both flagged above, under
        # this same lock) can newly expose a parked completion, so an idle flag
        # makes the full-slot scan provably fruitless.
        with self._lock:
            if not self._ready_pickup_pending:
                return tuple(completed)
            self._ready_pickup_pending = False
            for slot in self._slots:
                if (
                    slot.state is CandidateSlotState.READY
                    and slot.completion is not None
                    and slot.draft_instance_id
                    and (
                        slot.draft_instance_id,
                        slot.draft_session_id,
                    )
                    == (
                        slot.completion_draft_instance_id,
                        slot.completion_draft_session_id,
                    )
                    and slot.completion not in completed
                ):
                    completed.append(slot.completion)
        return tuple(completed)

    def poll_ready_records(self) -> int:
        """Mark assigned ARMED slots READY from GPU-written host records.

        The Draft push stream writes payload, metadata, and record words in
        order, generation last, so a matching generation proves the payload is
        in this arena.  The identity is the assignment's; the producer tag and
        nonce reject records from any other writer.
        """

        records = getattr(self, "_ready_records", None)
        if records is None:
            return 0
        expected = self._record_generation
        hits = np.flatnonzero((expected != 0) & (records.words[: self.slot_count, 0] == expected))
        if hits.size == 0:
            return 0
        observed_ns = time.monotonic_ns()
        accepted = 0
        for slot_id in hits.tolist():
            values = [int(value) for value in records.words[slot_id]]
            with self._lock:
                generation = int(expected[slot_id])
                slot = self._slots[slot_id]
                if slot.cancel_pending and getattr(
                    self, "_completion_records", None
                ) is not None:
                    # The cancelled push's completion record retires this
                    # slot; keep the expectation for poll_completion_records.
                    continue
                expected[slot_id] = 0
                if (
                    slot.generation != generation
                    or slot.state is not CandidateSlotState.ARMED
                    or slot.cancel_pending
                    or slot.completion_draft_instance_id
                    or not slot.draft_instance_id
                ):
                    continue
                nonce = values[NONCE_LOW_WORD] | (values[NONCE_HIGH_WORD] << 32)
                if (
                    nonce != slot.authorization_nonce
                    or values[PRODUCER_TAG_WORD] != int(self._record_tag[slot_id])
                    or values[VALID_LENGTH_WORD] != self.row_width
                ):
                    slot.state = CandidateSlotState.QUARANTINED
                    raise CandidateTransportError(
                        "candidate READY record failed generation/authentication"
                    )
                root = values[ROOT_TOKEN_WORD]
                if root & 0x80000000:
                    root -= 1 << 32
                slot.completion_draft_instance_id = slot.draft_instance_id
                slot.completion_draft_session_id = slot.draft_session_id
                slot.completion = CandidateCompletion(
                    transfer_id=slot.transfer_id,
                    slot_id=slot_id,
                    generation=slot.generation,
                    authorization_nonce=slot.authorization_nonce,
                    valid_length=values[VALID_LENGTH_WORD],
                    root_token=root,
                    draft_instance_id=slot.draft_instance_id,
                    draft_session_id=slot.draft_session_id,
                    observed_ns=observed_ns,
                    # No notice or readback precedes a record; keep one
                    # monotonic chain for traces.
                    notice_queued_ns=observed_ns,
                    notice_received_ns=observed_ns,
                    metadata_submitted_ns=observed_ns,
                    ready_source="gpu_record",
                )
                slot.state = CandidateSlotState.READY
                self._ready_pickup_pending = True
                key = (
                    slot.transfer_id,
                    slot_id,
                    slot.generation,
                    slot.authorization_nonce,
                    slot.draft_instance_id,
                    slot.draft_session_id,
                )
            # The direct notice still follows; ACK it as a duplicate.
            self._remember_notice(key)
            accepted += 1
        return accepted

    def poll_completion_records(self) -> int:
        """Mark assigned ARMED slots READY from GPU-written completion records.

        The Draft push stream queues the candidate copy and then a one-block
        kernel that writes the record payload words, fences system-wide, and
        releases the reservation's authorization nonce (the seq word) last,
        so an observed matching nonce proves every payload word is visible in
        this arena.  The nonce is unique per reservation, so a stale record
        from a prior reservation or producer can never match and slot reuse
        needs no shared-counter reset.  The identity is the assignment's; the
        generation, transfer id hash and producer tag reject records from any
        other writer.  For a cancel-pending slot the authenticated record is
        the cancelled push's quiescence proof — its kernel still publishes
        the original nonce — and retires the slot directly, replacing the
        suppressed notice's tombstone readback.  Accepted completions also
        carry the record's %globaltimer stamp and, when the producer stamped
        a nonzero globaltimer offset, the publication-to-discovery
        delivery_ns derived from both; the monotonic observed-chain fields
        are unaffected.
        """

        records = getattr(self, "_completion_records", None)
        if records is None:
            return 0
        expected = self._record_generation
        expected_seq = self._record_seq_expected
        words = self._completion_words
        hits = np.flatnonzero(
            (expected_seq != 0)
            & (words[: self.slot_count, COMPLETION_SEQ_WORD] == expected_seq)
            & (
                (words[: self.slot_count, COMPLETION_META_WORD] & 0xFFFFFFFF)
                == expected
            )
        )
        if hits.size == 0:
            return 0
        observed_ns = time.monotonic_ns()
        accepted = 0
        for slot_id in hits.tolist():
            # The nonce was released last; the payload words cannot be
            # reordered before it on this CPU, and the GPU released them
            # beforehand.
            meta = int(words[slot_id, COMPLETION_META_WORD])
            record_hash = int(words[slot_id, COMPLETION_HASH_WORD])
            token = int(words[slot_id, COMPLETION_TOKEN_WORD])
            produced = int(words[slot_id, COMPLETION_PRODUCED_WORD])
            record_offset = int(words[slot_id, COMPLETION_OFFSET_WORD])
            with self._lock:
                # A record is consumable exactly once: a failed authentication
                # must not re-raise on every poll.
                expected_seq[slot_id] = 0
                generation = int(expected[slot_id])
                slot = self._slots[slot_id]
                if slot.generation != generation:
                    continue
                if (
                    meta & 0xFFFFFFFF != generation
                    or meta >> 32 != self.row_width
                    or record_hash != transfer_id_hash(slot.transfer_id)
                    or token >> 32 != int(self._record_tag[slot_id])
                ):
                    if slot.cancel_pending:
                        slot.cancel_pending = False
                    slot.state = CandidateSlotState.QUARANTINED
                    raise CandidateTransportError(
                        "candidate completion record failed generation/authentication"
                    )
                if (
                    slot.state is CandidateSlotState.QUARANTINED
                    and slot.cancel_pending
                ):
                    # Stream-ordered after the candidate copy, the record is
                    # the cancelled push's quiescence proof; no tombstone
                    # readback is needed.
                    self._free_locked(slot_id)
                    accepted += 1
                    continue
                if (
                    slot.state is not CandidateSlotState.ARMED
                    or slot.cancel_pending
                    or slot.completion_draft_instance_id
                    or not slot.draft_instance_id
                ):
                    continue
                root = token & 0xFFFFFFFF
                if root & 0x80000000:
                    root -= 1 << 32
                slot.completion_draft_instance_id = slot.draft_instance_id
                slot.completion_draft_session_id = slot.draft_session_id
                slot.completion = CandidateCompletion(
                    transfer_id=slot.transfer_id,
                    slot_id=slot_id,
                    generation=slot.generation,
                    authorization_nonce=slot.authorization_nonce,
                    valid_length=meta >> 32,
                    root_token=root,
                    draft_instance_id=slot.draft_instance_id,
                    draft_session_id=slot.draft_session_id,
                    observed_ns=observed_ns,
                    # No notice or readback precedes a record; keep one
                    # monotonic chain for traces.
                    notice_queued_ns=observed_ns,
                    notice_received_ns=observed_ns,
                    metadata_submitted_ns=observed_ns,
                    ready_source="completion_record",
                    producer_globaltimer_ns=produced,
                    delivery_ns=(
                        time.time_ns() - (produced + record_offset)
                        if record_offset
                        else 0
                    ),
                )
                expected[slot_id] = 0
                slot.state = CandidateSlotState.READY
                self._ready_pickup_pending = True
                key = (
                    slot.transfer_id,
                    slot_id,
                    slot.generation,
                    slot.authorization_nonce,
                    slot.draft_instance_id,
                    slot.draft_session_id,
                )
            # A straggler notice from a mixed-mode producer ACKs as a duplicate.
            self._remember_notice(key)
            accepted += 1
        return accepted

    def begin_read(self, completions: list[CandidateCompletion]) -> torch.Tensor:
        """Gather selected rows, fence their last read, and return device tokens."""

        if not completions:
            raise ValueError("candidate gather requires at least one completion")
        slot_ids = []
        with self._lock:
            for completion in completions:
                slot_id = self._active.get(completion.transfer_id)
                if slot_id != completion.slot_id:
                    raise CandidateTransportError("candidate completion slot changed")
                slot = self._slots[slot_id]
                if (
                    slot.state is not CandidateSlotState.READY
                    or slot.generation != completion.generation
                    or slot.completion != completion
                ):
                    raise CandidateTransportError("candidate completion is stale")
                slot.state = CandidateSlotState.READING
                slot_ids.append(slot_id)
        # A pageable list upload would block the scheduler until the verify
        # already queued on this stream finishes; stage it in pinned memory.
        indices_cpu = torch.tensor(
            slot_ids, dtype=torch.int64, pin_memory=torch.cuda.is_available()
        )
        indices = indices_cpu.to(self.device, non_blocking=indices_cpu.is_pinned())
        rows = self.storage.index_select(0, indices)
        return rows

    def finish_read(self, completions: list[CandidateCompletion]) -> None:
        """Fence the final gather/broadcast consumer before slot reuse."""

        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        with self._lock:
            for completion in completions:
                slot_id = self._active.get(completion.transfer_id)
                if slot_id != completion.slot_id:
                    raise CandidateTransportError("candidate read slot changed")
                slot = self._slots[slot_id]
                if slot.state is not CandidateSlotState.READING:
                    raise CandidateTransportError("candidate slot is not being read")
                slot.read_event = event

    def poll_releases(self) -> int:
        released = 0
        with self._lock:
            candidates = [
                (index, slot.generation, slot.read_event)
                for index, slot in enumerate(self._slots)
                if slot.state is CandidateSlotState.READING
            ]
        for slot_id, generation, event in candidates:
            if event is None:
                continue
            try:
                ready = bool(event.query())
            except Exception as exc:
                with self._lock:
                    self._slots[slot_id].state = CandidateSlotState.QUARANTINED
                raise CandidateTransportError("candidate read fence query failed") from exc
            if not ready:
                continue
            with self._lock:
                slot = self._slots[slot_id]
                if (
                    slot.state is CandidateSlotState.READING
                    and slot.generation == generation
                ):
                    self._free_locked(slot_id)
                    released += 1
        return released

    def cancel(self, transfer_id: str, *, remote_quiescent: bool = False) -> bool:
        """Cancel safely; uncertain assigned producers quarantine their slot."""

        with self._lock:
            slot_id = self._active.get(transfer_id)
            if slot_id is None:
                return False
            slot = self._slots[slot_id]
            if slot.state is CandidateSlotState.READY:
                self._free_locked(slot_id)
                return True
            if slot.state is CandidateSlotState.READING:
                return False
            if slot.draft_session_id and slot.draft_session_id in getattr(
                self, "_revoked_producers", ()
            ) and slot.state in {CandidateSlotState.ARMED, CandidateSlotState.QUARANTINED}:
                # Its producer was revoked as gone: nothing can still write it.
                self._free_locked(slot_id)
                return True
            if slot.state is CandidateSlotState.QUARANTINED:
                return False
            if not remote_quiescent:
                # Submission can win the Coordinator assignment while its ACK
                # is still in flight.  An unbound slot is therefore not proof
                # that no Draft can write it.  Retain the generation so a late
                # assignment can authenticate the producer and its tombstone.
                slot.state = CandidateSlotState.QUARANTINED
                slot.cancel_pending = True
                return False
            self._free_locked(slot_id)
            return True

    def poll_cancelled(self) -> int:
        released = 0
        with self._lock:
            candidates = [
                (index, slot.generation, slot.completion_event, slot.cancel_pending)
                for index, slot in enumerate(self._slots)
                if slot.state is CandidateSlotState.CANCELLED
                or (
                    slot.state is CandidateSlotState.QUARANTINED
                    and slot.cancel_pending
                )
            ]
        for slot_id, generation, event, needs_remote_proof in candidates:
            if event is None:
                continue
            try:
                ready = bool(event.query())
            except Exception as exc:
                with self._lock:
                    slot = self._slots[slot_id]
                    if slot.generation == generation:
                        slot.state = CandidateSlotState.QUARANTINED
                        slot.cancel_pending = False
                raise CandidateTransportError(
                    "cancelled candidate completion event query failed"
                ) from exc
            if ready:
                with self._lock:
                    slot = self._slots[slot_id]
                    if needs_remote_proof:
                        values = [
                            int(value) & 0xFFFFFFFF for value in slot.metadata_host
                        ]
                        nonce = values[NONCE_LOW_WORD] | (
                            values[NONCE_HIGH_WORD] << 32
                        )
                        if (
                            values[GENERATION_WORD] != generation
                            or nonce != slot.authorization_nonce
                            or values[VALID_LENGTH_WORD] not in {0, self.row_width}
                        ):
                            slot.cancel_pending = False
                            raise CandidateTransportError(
                                "cancelled candidate completion failed authentication"
                            )
                    if (
                        slot.state
                        in {
                            CandidateSlotState.CANCELLED,
                            CandidateSlotState.QUARANTINED,
                        }
                        and slot.generation == generation
                    ):
                        self._free_locked(slot_id)
                        released += 1
        return released

    def close(self) -> tuple[int, int]:
        with self._lock:
            self._closed = True
        self.poll_releases()
        self.poll_cancelled()
        self._completion_socket.close(linger=0)
        if getattr(self, "_ready_records", None) is not None:
            # Draft mappings keep the inode alive; only the name goes away.
            self._ready_records.close(unlink=True)
            self._ready_records = None
        if getattr(self, "_completion_records", None) is not None:
            self._runtime.host_unregister(self._completion_records.host_address)
            # Release the numpy view first: a live view fails the mmap close
            # with BufferError and leaks the mapping.
            self._completion_words = None
            self.completion_record_path = ""
            # Draft mappings keep the inode alive; only the name goes away.
            self._completion_records.close(unlink=True)
            self._completion_records = None
        return self.active_count, self.quarantined_slot_count

    def _free_locked(self, slot_id: int) -> None:
        slot = self._slots[slot_id]
        self._active.pop(slot.transfer_id, None)
        if slot.transfer_id:
            # A freed slot had a proven completion or a proven-quiescent
            # cancellation; a later assignment for it is stale, not an error.
            retired = self._retired_transfers
            retired[slot.transfer_id] = None
            if len(retired) > RETIRED_TRANSFER_LIMIT:
                retired.popitem(last=False)
        generation = slot.generation
        self._slots[slot_id] = _CandidateSlot(generation=generation)
        if getattr(self, "_record_generation", None) is not None:
            self._record_generation[slot_id] = 0
        if getattr(self, "_record_seq_expected", None) is not None:
            self._record_seq_expected[slot_id] = 0


@dataclass(frozen=True, slots=True)
class CandidatePush:
    event: torch.cuda.Event
    transfer_ids: tuple[str, ...]
    submitted_ns: int
    candidate_bytes: int
    cold_mapping_count: int
    start_event: torch.cuda.Event | None = None


class DraftCandidatePushTransport:
    """Map Target destinations and enqueue candidate copy+publication in place."""

    def __init__(
        self,
        *,
        draft_instance_id: str,
        draft_session_id: str,
        device: torch.device,
        profile_timing: bool = False,
    ) -> None:
        if not draft_instance_id or not draft_session_id:
            raise ValueError("Draft candidate identity must not be empty")
        requested = torch.device(device)
        if requested.type != "cuda":
            raise ValueError("candidate push requires CUDA")
        self.device_index = (
            torch.cuda.current_device()
            if requested.index is None
            else int(requested.index)
        )
        self.device = torch.device("cuda", self.device_index)
        torch.cuda.set_device(self.device)
        self.draft_instance_id = draft_instance_id
        self.draft_session_id = draft_session_id
        self._profile_timing = bool(profile_timing)
        self._runtime = CudaIpcRuntime()
        self._mapping_cache = CudaIpcMappingCache(self._runtime)
        self._registered = {}
        self._completion_sockets: dict[tuple[str, str, str, str], zmq.Socket] = {}
        self._pending_notices: dict[str, _PendingCompletionNotice] = {}
        self._acknowledgements: deque[dict[str, int | str]] | None = (
            deque() if self._profile_timing else None
        )
        self._zmq_context = zmq.Context.instance()
        self._notice_encoder = msgspec.msgpack.Encoder()
        # Target READY record mappings: key -> (file, device base address).
        self._ready_records: dict[tuple[str, str, str, str], tuple[_ReadyRecordFile, int]] = {}
        self._producer_tag = producer_tag(draft_session_id)
        # Target completion record mappings: key -> (file, device base
        # address), plus the publication kernel.
        self._completion_records: dict[tuple[str, str, str, str], tuple[_ReadyRecordFile, int]] = {}
        self._completion_kernel: _PushCompletionKernel | None = None
        # CLOCK_REALTIME minus this GPU's %globaltimer, calibrated once on
        # first use; None when uncalibrated (records then carry offset 0).
        self._globaltimer_offset_ns: int | None = None
        self._globaltimer_calibrated = False
        # Completed candidates are pushed here rather than on the compute
        # stream, where they would queue behind the next Draft batch.
        self.push_stream = torch.cuda.Stream(device=self.device)
        self._closed = False
        self._ring_debug: _RingDebugLog | None = (
            _RingDebugLog(f"draft-push:{draft_instance_id}") if RING_DEBUG else None
        )

    @staticmethod
    def _registration_key(value) -> tuple[str, str, str, str]:
        return (
            value.target_instance_id,
            value.target_session_id,
            value.pool_generation,
            value.transport_session,
        )

    def register(self, registration) -> None:
        """Cache one Target arena mapping for later logical descriptors."""

        if self._closed:
            raise CandidateTransportError("candidate push transport is closed")
        if (
            not registration.target_instance_id
            or not registration.target_session_id
            or not registration.pool_generation
            or not registration.transport_session
            or registration.row_width <= 0
            or registration.metadata_words != METADATA_WORDS
            or not registration.data_ipc_handle
            or not registration.metadata_ipc_handle
            or not registration.completion_endpoint
        ):
            raise ValueError("candidate arena registration is incomplete")
        key = self._registration_key(registration)
        if any(
            registered_key[0] == registration.target_instance_id
            and registered_key != key
            for registered_key in self._registered
        ):
            # Replacing an IPC allocation while an old transfer might still
            # write it requires a process-level quiescence proof. Transparent
            # recovery is out of scope, so fail closed instead of invalidating
            # a mapping that may still be referenced by queued CUDA work.
            raise CandidateTransportError(
                "candidate Target incarnation changed; Draft restart is required"
            )
        data_base, _ = self._mapping_cache.map(
            peer_instance_id=registration.target_instance_id,
            peer_session_id=registration.target_session_id,
            pool_generation=registration.pool_generation,
            ipc_handle=registration.data_ipc_handle,
        )
        metadata_base, _ = self._mapping_cache.map(
            peer_instance_id=registration.target_instance_id,
            peer_session_id=registration.target_session_id,
            pool_generation=registration.pool_generation,
            ipc_handle=registration.metadata_ipc_handle,
        )
        previous = self._registered.get(key)
        socket = self._completion_sockets.get(key)
        if socket is None:
            socket = self._zmq_context.socket(zmq.DEALER)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.IMMEDIATE, 1)
            socket.setsockopt(zmq.SNDHWM, COMPLETION_SOCKET_HWM)
            socket.setsockopt(zmq.RCVHWM, COMPLETION_SOCKET_HWM)
            socket.setsockopt(zmq.IDENTITY, self.draft_session_id.encode())
            socket.connect(registration.completion_endpoint)
            self._completion_sockets[key] = socket
        mapped = (registration, data_base, metadata_base, socket)
        if previous is not None and previous != mapped:
            raise CandidateTransportError("candidate arena registration changed")
        self._registered[key] = mapped
        if (
            GPU_READY_RECORD
            and getattr(registration, "ready_record_path", "")
            and key not in getattr(self, "_ready_records", {})
        ):
            try:
                records = _ReadyRecordFile(registration.ready_record_path, create=False)
            except (OSError, CandidateTransportError):
                # Not the same host or the file is gone: notice-only completion.
                logger.warning(
                    "Target READY record file unavailable, using notices: %s",
                    registration.ready_record_path,
                    exc_info=True,
                )
            else:
                try:
                    device_base = self._runtime.host_register(
                        records.host_address, records.nbytes
                    )
                except Exception:
                    records.close(unlink=False)
                    raise
                self._ready_records[key] = (records, device_base)
        if (
            PUSH_COMPLETION_RECORD
            and getattr(registration, "completion_record_path", "")
            and key not in getattr(self, "_completion_records", {})
        ):
            try:
                records = _ReadyRecordFile(
                    registration.completion_record_path,
                    record_bytes=COMPLETION_RECORD_BYTES,
                    create=False,
                )
            except (OSError, CandidateTransportError):
                # Not the same host or the file is gone: notice-only completion.
                logger.warning(
                    "Target completion record file unavailable, using notices: %s",
                    registration.completion_record_path,
                    exc_info=True,
                )
            else:
                device_base = 0
                try:
                    device_base = self._runtime.host_register(
                        records.host_address, records.nbytes
                    )
                    kernel = self._completion_kernel
                    if kernel is None:
                        kernel = self._completion_kernel = _PushCompletionKernel()
                except Exception:
                    if device_base:
                        self._runtime.host_unregister(records.host_address)
                    records.close(unlink=False)
                    raise
                self._completion_records[key] = (records, device_base)
                self._ensure_globaltimer_offset()

    def has_ready_records(self, tasks: list[object]) -> bool:
        """Whether every task's Target arena accepts GPU-written READY records."""

        records = getattr(self, "_ready_records", None)
        return bool(tasks and records) and all(
            self._registration_key(task.candidate_destination) in records
            for task in tasks
        )

    def has_completion_records(self, tasks: list[object]) -> bool:
        """Whether every task's Target arena accepts GPU-written completion records."""

        records = getattr(self, "_completion_records", None)
        return bool(tasks and records) and all(
            self._registration_key(task.candidate_destination) in records
            for task in tasks
        )

    def _ensure_globaltimer_offset(self) -> None:
        """Calibrate this GPU's globaltimer offset exactly once, best effort.

        The completion kernel stamps records on this device, so the offset
        must come from this device: %globaltimer bases differ across
        same-host GPUs.  Failure leaves the offset unset; records then carry
        0 and the Target leaves delivery_ns at 0.
        """

        if self._globaltimer_calibrated:
            return
        self._globaltimer_calibrated = True
        try:
            self._globaltimer_offset_ns = _calibrate_globaltimer_offset(
                self._runtime, self._completion_kernel, self.device_index
            )
        except Exception:
            logger.warning(
                "globaltimer calibration failed; completion records carry "
                "offset 0 and Target delivery_ns stays 0",
                exc_info=True,
            )
        else:
            logger.info(
                "globaltimer offset %d ns calibrated on device %d",
                self._globaltimer_offset_ns,
                self.device_index,
            )

    def _launch_completion_record(
        self,
        *,
        destination,
        record_hash: int,
        root_token: int,
        record_address: int,
        stream: int,
    ) -> None:
        self._completion_kernel.launch(
            device=self.device_index,
            record_address=record_address,
            seq=int(destination.authorization_nonce),
            generation=int(destination.generation),
            valid_length=int(destination.row_width),
            record_hash=record_hash,
            root_token=root_token,
            tag=self._producer_tag,
            globaltimer_offset_ns=self._globaltimer_offset_ns or 0,
            stream=stream,
        )

    def _mapping(self, destination):
        mapped = self._registered.get(self._registration_key(destination))
        if mapped is None:
            raise CandidateTransportError(
                "candidate destination references an unregistered arena"
            )
        registration, data_base, metadata_base, _socket = mapped
        if (
            registration.target_device_id != destination.target_device_id
            or registration.device_uuid != destination.device_uuid
            or registration.row_width != destination.row_width
            or registration.data_allocation_length
            != destination.data_allocation_length
            or registration.metadata_allocation_length
            != destination.metadata_allocation_length
        ):
            raise CandidateTransportError(
                "candidate destination conflicts with its arena registration"
            )
        return data_base, metadata_base

    def publish_completion(self, tasks: list[object], *, cancelled: bool) -> None:
        """Queue authenticated direct READY notices after producer CUDA completion."""

        from dt_protocol import PROTOCOL_VERSION, CandidateCompletionNotice

        if self._closed:
            raise CandidateTransportError("candidate push transport is closed")
        now_ns = time.monotonic_ns()
        for task in tasks:
            destination = task.candidate_destination
            mapped = self._registered.get(self._registration_key(destination))
            if mapped is None:
                raise CandidateTransportError(
                    "candidate completion references an unregistered arena"
                )
            registration, _data_base, _metadata_base, socket = mapped
            if (
                not cancelled
                and self._registration_key(destination)
                in getattr(self, "_completion_records", {})
            ):
                # The push stream already published the completion record
                # after the candidate copy; the Target polls it from host
                # memory, so no notice is queued or retried.
                continue
            notice = CandidateCompletionNotice(
                protocol_version=PROTOCOL_VERSION,
                target_instance_id=destination.target_instance_id,
                target_session_id=destination.target_session_id,
                pool_generation=destination.pool_generation,
                transport_session=destination.transport_session,
                transfer_id=task.transfer_id,
                slot_id=destination.slot_id,
                generation=destination.generation,
                authorization_nonce=destination.authorization_nonce,
                draft_instance_id=self.draft_instance_id,
                draft_session_id=self.draft_session_id,
                cancelled=cancelled,
                sent_ns=now_ns,
                valid_length=0 if cancelled else int(destination.row_width),
                root_token=int(task.next_token_id),
            )
            pending = self._pending_notices.get(task.transfer_id)
            if pending is not None:
                if pending.notice != notice:
                    raise CandidateTransportError(
                        "candidate completion transfer identity was reused"
                    )
                continue
            if registration.completion_endpoint == "":
                raise CandidateTransportError("candidate completion endpoint is empty")
            self._pending_notices[task.transfer_id] = _PendingCompletionNotice(
                notice=notice,
                socket=socket,
                encoded=self._notice_encoder.encode(notice),
                next_retry_ns=now_ns,
            )
            if self._ring_debug is not None:
                self._ring_debug.bump(
                    f"notice_queued:{destination.target_instance_id}"
                )
        self.progress_completions(now_ns=now_ns)

    @property
    def pending_completion_count(self) -> int:
        return len(self._pending_notices)

    def progress_completions(
        self, *, max_messages: int = 64, now_ns: int | None = None
    ) -> int:
        """ACK and retry bounded direct notices without blocking Draft."""

        from dt_protocol import PROTOCOL_VERSION, CandidateCompletionAck

        if self._closed:
            return 0
        if max_messages <= 0:
            raise ValueError("max_messages must be positive")
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        dbg = self._ring_debug
        if dbg is not None:
            dbg.note_call(now_ns)
        acknowledged = 0
        # Each Target socket gets its own budget: a busy peer must not starve
        # the ACK stream of the others.
        for key, socket in self._completion_sockets.items():
            budget = max_messages
            socket_received = 0
            while budget:
                try:
                    parts = socket.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                budget -= 1
                socket_received += 1
                payload = parts[-1]
                try:
                    ack = msgspec.msgpack.decode(payload, type=CandidateCompletionAck)
                except (msgspec.DecodeError, msgspec.ValidationError):
                    if dbg is not None:
                        dbg.bump("ack_malformed")
                    continue
                pending = self._pending_notices.get(ack.transfer_id)
                if pending is None:
                    if dbg is not None:
                        dbg.bump("ack_unknown_transfer")
                    continue
                notice = pending.notice
                if (
                    ack.protocol_version == PROTOCOL_VERSION
                    and ack.target_instance_id == notice.target_instance_id
                    and ack.target_session_id == notice.target_session_id
                    and ack.transport_session == notice.transport_session
                    and ack.generation == notice.generation
                    and ack.authorization_nonce == notice.authorization_nonce
                    and ack.draft_instance_id == self.draft_instance_id
                    and ack.draft_session_id == self.draft_session_id
                ):
                    acknowledgements = getattr(self, "_acknowledgements", None)
                    if acknowledgements is not None:
                        acknowledgements.append(
                            {
                                "transfer_id": ack.transfer_id,
                                "notice_queued_ns": notice.sent_ns,
                                "target_accepted_ns": ack.accepted_ns,
                                "ack_observed_ns": time.monotonic_ns(),
                            }
                        )
                    del self._pending_notices[ack.transfer_id]
                    acknowledged += 1
                elif dbg is not None:
                    dbg.bump("ack_rejected")
            if dbg is not None and socket_received:
                dbg.bump(f"ack_recv:{key[0]}", socket_received)
            if dbg is not None and not budget:
                dbg.bump("ack_budget_exhausted")
        if dbg is not None:
            if acknowledged:
                dbg.bump("ack_accepted", acknowledged)
            dbg.bump("pending_max", 0)  # ensure key exists
            pending_now = len(self._pending_notices)
            if pending_now > dbg.total.get("pending_max", 0):
                dbg.total["pending_max"] = pending_now
        # Backpressure is per Target socket: a full pipe to one Target must
        # not suppress retries owed to healthy Targets in the same call.
        eagain_streaks: dict[int, int] = {}
        for transfer_id, pending in tuple(self._pending_notices.items()):
            if pending.next_retry_ns > now_ns:
                continue
            if (
                pending.attempts >= COMPLETION_MAX_ATTEMPTS
                or pending.send_failures >= COMPLETION_MAX_ATTEMPTS
            ):
                if dbg is not None:
                    dbg.maybe_emit(
                        now_ns,
                        force=True,
                        extra=(
                            f" EXHAUSTED transfer={transfer_id}"
                            f" attempts={pending.attempts}"
                            f" send_failures={pending.send_failures}"
                            f" age_ms={(now_ns - pending.notice.sent_ns) / 1e6:.0f}"
                            f" pending={len(self._pending_notices)}"
                        ),
                    )
                raise CandidateTransportError(
                    "direct candidate completion retry budget exhausted for "
                    f"{transfer_id}"
                )
            socket_key = id(pending.socket)
            if eagain_streaks.get(socket_key, 0) >= COMPLETION_RETRY_EAGAIN_LIMIT:
                # This Target's pipe is full; further sends to it this call
                # are futile and burn CPU that the ACK drain above needs.
                if dbg is not None:
                    dbg.bump("retry_deferred_pipe_full")
                continue
            try:
                pending.socket.send(pending.encoded, flags=zmq.NOBLOCK)
            except zmq.Again:
                eagain_streaks[socket_key] = eagain_streaks.get(socket_key, 0) + 1
                pending.send_failures += 1
                # No attempt is burned: the notice never left this process, so
                # the budget keeps measuring Target silence, not pipe pressure.
                pending.next_retry_ns = now_ns + COMPLETION_RETRY_NS
                if dbg is not None:
                    dbg.bump("retry_send_eagain")
                continue
            eagain_streaks[socket_key] = 0
            if dbg is not None:
                dbg.bump("retry_sent")
            pending.attempts += 1
            pending.next_retry_ns = now_ns + COMPLETION_RETRY_NS
        if dbg is not None:
            dbg.maybe_emit(now_ns)
        return acknowledged

    def pop_acknowledgements(self) -> tuple[dict[str, int | str], ...]:
        queue = getattr(self, "_acknowledgements", None)
        if queue is None:
            return ()
        acknowledgements = tuple(queue)
        queue.clear()
        return acknowledgements

    def submit(self, *, tasks: list[object], candidates: torch.Tensor) -> CandidatePush:
        if self._closed:
            raise CandidateTransportError("candidate push transport is closed")
        if not tasks or tuple(candidates.shape) != (
            len(tasks),
            int(tasks[0].candidate_destination.row_width),
        ):
            raise ValueError("candidate batch shape does not match destinations")
        if (
            candidates.device != self.device
            or candidates.dtype is not torch.int32
            or not candidates.is_contiguous()
        ):
            raise ValueError("candidate source must be contiguous int32 CUDA")
        segments = []
        metadata_addresses = []
        cold = 0
        row_bytes = int(candidates.shape[1]) * 4
        for index, task in enumerate(tasks):
            destination = task.candidate_destination
            self._validate_destination(task, destination, row_bytes)
            data_base, metadata_base = self._mapping(destination)
            segments.append(
                CopySegment(
                    source_address=candidates[index].data_ptr(),
                    destination_address=(
                        data_base + destination.data_allocation_offset
                    ),
                    length=row_bytes,
                )
            )
            metadata_addresses.append(
                metadata_base + destination.metadata_allocation_offset
            )
        stream = torch.cuda.current_stream(self.device)
        start_event = (
            torch.cuda.Event(enable_timing=True)
            if getattr(self, "_profile_timing", False)
            else None
        )
        if start_event is not None:
            start_event.record(stream)
        self._runtime.copy_batch_async(tuple(segments), stream.cuda_stream)
        for task, address in zip(tasks, metadata_addresses, strict=True):
            destination = task.candidate_destination
            nonce = int(destination.authorization_nonce)
            words = (
                (NONCE_LOW_WORD, nonce & 0xFFFFFFFF),
                (NONCE_HIGH_WORD, (nonce >> 32) & 0xFFFFFFFF),
                (VALID_LENGTH_WORD, destination.row_width),
                (ROOT_TOKEN_WORD, int(task.next_token_id) & 0xFFFFFFFF),
                # Publish generation last: the Target wait is the acquire edge.
                (GENERATION_WORD, destination.generation),
            )
            for word, value in words:
                self._runtime.stream_write_value32(
                    address=address + word * 4,
                    value=value,
                    stream=stream.cuda_stream,
                )
            completion = getattr(self, "_completion_records", {}).get(
                self._registration_key(destination)
            )
            record = (
                None
                if completion is not None
                else getattr(self, "_ready_records", {}).get(
                    self._registration_key(destination)
                )
            )
            if record is not None:
                records, device_base = record
                if destination.slot_id >= records.slot_capacity:
                    raise CandidateTransportError("READY record slot is out of range")
                record_address = device_base + destination.slot_id * READY_RECORD_BYTES
                # Default stream memops order after the copy and metadata
                # above; the Target CPU treats a matching generation as READY.
                for word, value in (
                    *words[:-1],
                    (PRODUCER_TAG_WORD, self._producer_tag),
                    words[-1],
                ):
                    self._runtime.stream_write_value32(
                        address=record_address + word * 4,
                        value=value,
                        stream=stream.cuda_stream,
                    )
            if completion is not None:
                records, device_base = completion
                if destination.slot_id >= records.slot_capacity:
                    raise CandidateTransportError(
                        "completion record slot is out of range"
                    )
                # The completion record carries a superset of the READY
                # record words, so the legacy writes above are skipped for
                # this Target and READY comes only from the kernel.  The
                # launch is stream-ordered after the copy and metadata: the
                # kernel releases the reservation's authorization nonce last,
                # so a Target CPU that observes it also observes the
                # candidate payload.
                self._launch_completion_record(
                    destination=destination,
                    record_hash=transfer_id_hash(task.transfer_id),
                    root_token=int(task.next_token_id) & 0xFFFFFFFF,
                    record_address=(
                        device_base + destination.slot_id * COMPLETION_RECORD_BYTES
                    ),
                    stream=stream.cuda_stream,
                )
        event = torch.cuda.Event(enable_timing=start_event is not None)
        event.record(stream)
        return CandidatePush(
            event=event,
            transfer_ids=tuple(task.transfer_id for task in tasks),
            submitted_ns=time.monotonic_ns(),
            candidate_bytes=len(tasks) * row_bytes,
            cold_mapping_count=cold,
            start_event=start_event,
        )

    def publish_cancelled(self, tasks: list[object]) -> CandidatePush:
        """Publish authenticated quiescence for tasks cancelled before a push."""

        if self._closed:
            raise CandidateTransportError("candidate push transport is closed")
        if not tasks:
            raise ValueError("candidate cancellation publication is empty")
        stream = torch.cuda.current_stream(self.device)
        start_event = (
            torch.cuda.Event(enable_timing=True)
            if getattr(self, "_profile_timing", False)
            else None
        )
        if start_event is not None:
            start_event.record(stream)
        for task in tasks:
            destination = task.candidate_destination
            self._validate_destination(task, destination, destination.nbytes)
            _data_base, metadata_base = self._mapping(destination)
            address = metadata_base + destination.metadata_allocation_offset
            nonce = int(destination.authorization_nonce)
            for word, value in (
                (NONCE_LOW_WORD, nonce & 0xFFFFFFFF),
                (NONCE_HIGH_WORD, (nonce >> 32) & 0xFFFFFFFF),
                (VALID_LENGTH_WORD, 0),
                (ROOT_TOKEN_WORD, int(task.next_token_id) & 0xFFFFFFFF),
                (GENERATION_WORD, destination.generation),
            ):
                self._runtime.stream_write_value32(
                    address=address + word * 4,
                    value=value,
                    stream=stream.cuda_stream,
                )
        event = torch.cuda.Event(enable_timing=start_event is not None)
        event.record(stream)
        return CandidatePush(
            event=event,
            transfer_ids=tuple(task.transfer_id for task in tasks),
            submitted_ns=time.monotonic_ns(),
            candidate_bytes=0,
            cold_mapping_count=0,
            start_event=start_event,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        torch.cuda.current_stream(self.device).synchronize()
        records_by_key = getattr(self, "_ready_records", {})
        # Pushes run on push_stream; their sources and mappings must outlive them.
        self.push_stream.synchronize()
        for records, _device_base in records_by_key.values():
            self._runtime.host_unregister(records.host_address)
            records.close(unlink=False)
        records_by_key.clear()
        completion_by_key = getattr(self, "_completion_records", {})
        for records, _device_base in completion_by_key.values():
            self._runtime.host_unregister(records.host_address)
            records.close(unlink=False)
        completion_by_key.clear()
        self._mapping_cache.close()
        self._registered.clear()
        for socket in self._completion_sockets.values():
            socket.close(linger=0)
        self._completion_sockets.clear()

    @staticmethod
    def _validate_destination(task: object, destination, row_bytes: int) -> None:
        if destination is None:
            raise ValueError("direct candidate task has no destination")
        if (
            destination.transport_session == ""
            or destination.notification_backend != "transport-host-completion"
            or destination.transfer_id != task.transfer_id
            or destination.target_instance_id != task.target_instance_id
            or destination.dtype != "int32"
            or destination.row_width <= 0
            or destination.nbytes != row_bytes
            or destination.metadata_words != METADATA_WORDS
            or destination.slot_id < 0
            or not 0 < destination.generation <= 0x7FFFFFFF
            or not 0 < destination.authorization_nonce < (1 << 63)
            or destination.data_ipc_handle
            or destination.metadata_ipc_handle
        ):
            raise ValueError("candidate destination descriptor is incomplete")
        if (
            destination.data_allocation_offset < 0
            or destination.data_allocation_offset + destination.nbytes
            > destination.data_allocation_length
            or destination.metadata_allocation_offset < 0
            or destination.metadata_allocation_offset + METADATA_WORDS * 4
            > destination.metadata_allocation_length
        ):
            raise ValueError("candidate destination exceeds its CUDA allocation")
