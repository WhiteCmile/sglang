"""Same-host Target -> owner-Draft descriptor rings (DT_DIRECT_RING=1).

For owner-known persistent rounds (RECONCILE with a candidate destination)
the Target writes the round's task directly into a /dev/shm ring shared with
the owning Draft instead of sending it through the Coordinator and the Draft
I/O thread.  A native worker thread in the Draft (csrc/dt_ring_worker.cpp, no
GIL, no resident SMs) serves each published entry: it waits on the Target's
producer event and copies the hidden rows into a shared Draft landing buffer
on its own copy stream, then publishes INPUT_READY into the ring with a
stream memory write.  The Draft main loop turns ready entries directly into
INPUT_READY work; it no longer receives, submits, or polls these pulls.

The Coordinator still receives the same task through the Target's normal
submission and remains the confirmation and accounting authority: the Draft
holds the Coordinator-bound result and source-read-done of a ring task until
the Coordinator copy arrives, exactly like the earlier direct dispatch.

Ring file layout (one file per (Draft session, Target session), created by
the Draft, page-aligned regions):
  page 0      header words + Target source registration
  page 1..    ``capacity`` entries of ``ENTRY_BYTES``: 64-byte header (seq
              last), 64-byte producer event IPC handle, msgpack DraftTask
  last pages  ``capacity`` INPUT_READY words written by the Draft GPU stream
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import logging
import mmap
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DIRECT_RING = os.environ.get("DT_DIRECT_RING", "0") == "1"
RING_CAPACITY = int(os.environ.get("DT_DIRECT_RING_CAPACITY", "256"))
if RING_CAPACITY <= 0 or RING_CAPACITY & (RING_CAPACITY - 1):
    # Indices are derived from the 32-bit wire seq in the native worker; a
    # power of two keeps ``seq % capacity`` identical on all three sides.
    raise ValueError("DT_DIRECT_RING_CAPACITY must be a power of two")
SEQ_MASK = 0xFFFFFFFF
RING_MAGIC = 0x44545247
RING_VERSION = 1
ENTRY_BYTES = 4096
ENTRY_HEADER_BYTES = 64
EVENT_HANDLE_OFFSET = 64
EVENT_HANDLE_BYTES = 64
PAYLOAD_OFFSET = 128
MAX_PAYLOAD_BYTES = ENTRY_BYTES - PAYLOAD_OFFSET
FLAG_HAS_PULL = 1

STATE_CREATED = 1
STATE_REGISTERED = 2
STATE_SERVING = 3
STATE_CLOSED = 4

# Header u32 word indices.
H_MAGIC = 0
H_VERSION = 1
H_CAPACITY = 2
H_ENTRY_BYTES = 3
H_STATE = 4
H_CONSUMED = 5  # Draft: low 32 bits of the last consumed seq
H_MAX_ENTRY_BYTES = 6  # Draft: largest hidden range it accepts per entry
H_OWNER_PID = 7  # Draft process id: same-host liveness fence for lease revocation
H_SOURCE_LENGTH_LO = 8
H_SOURCE_LENGTH_HI = 9
# Header byte offsets of Target source registration fields.
SOURCE_HANDLE_OFFSET = 64
POOL_GENERATION_OFFSET = 128
SOURCE_SESSION_OFFSET = 192
TEXT_FIELD_BYTES = 64

# Entry header u32 word indices (mirrors EntryHeader in the worker).
E_SEQ = 0
E_FLAGS = 1
E_SOURCE_OFFSET_LO = 2
E_SOURCE_OFFSET_HI = 3
E_NBYTES = 4
E_SOURCE_SLOT = 5
E_SLOT_GENERATION = 6
E_PAYLOAD_LEN = 7
E_EPOCH = 8  # high 32 bits of the logical seq (written before E_SEQ)


class DirectRingError(RuntimeError):
    pass


def ring_path(draft_session_id: str, target_session_id: str) -> str:
    def clean(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", value)

    return f"/dev/shm/dt-ring-{clean(draft_session_id)}-{clean(target_session_id)}"


def next_seq(seq: int) -> int:
    """Logical successor; skips seqs whose wire value (low 32 bits) is 0.

    0 is the never-written value of every entry and INPUT_READY word.  The
    native worker skips it identically, and because the capacity divides
    2**32 the skip leaves ``seq % capacity`` equal on all sides.
    """

    seq += 1
    return seq + 1 if not seq & SEQ_MASK else seq


def _seq_after(later: int, earlier: int) -> bool:
    """32-bit modular ``later >= earlier`` for values less than 2**31 apart."""

    return ((later - earlier) & SEQ_MASK) < 0x80000000


def _layout(capacity: int) -> tuple[int, int, int]:
    page = mmap.PAGESIZE
    entries_offset = page
    ready_offset = entries_offset + -(-capacity * ENTRY_BYTES // page) * page
    total = ready_offset + -(-capacity * 4 // page) * page
    return entries_offset, ready_offset, total


class _RingMapping:
    def __init__(self, path: str, *, capacity: int = 0, create: bool) -> None:
        # A creator maps a private temp name and publishes it with rename()
        # once the header is complete, so an opener never sees a partial file.
        open_path = f"{path}.{os.getpid()}.init" if create else path
        flags = os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0)
        fd = os.open(open_path, flags, 0o600)
        try:
            if create:
                if capacity <= 0:
                    raise ValueError("ring capacity must be positive")
                _, _, total = _layout(capacity)
                os.ftruncate(fd, total)
            else:
                with mmap.mmap(fd, mmap.PAGESIZE, prot=mmap.PROT_READ) as head:
                    words = np.frombuffer(head, dtype=np.uint32, count=8).copy()
                if words[H_MAGIC] != RING_MAGIC or words[H_VERSION] != RING_VERSION:
                    raise DirectRingError(f"ring {path} has an unknown format")
                capacity = int(words[H_CAPACITY])
                _, _, total = _layout(capacity)
                if os.fstat(fd).st_size < total:
                    raise DirectRingError(f"ring {path} is truncated")
            self._mm = mmap.mmap(fd, total)
        finally:
            os.close(fd)
        self.path = path
        self.open_path = open_path
        self.capacity = capacity
        self.total_bytes = total
        self.entries_offset, self.ready_offset, _ = _layout(capacity)
        self.raw = np.frombuffer(self._mm, dtype=np.uint8)
        self.header = self.raw[: mmap.PAGESIZE].view(np.uint32)
        self.ready = self.raw[self.ready_offset : self.ready_offset + capacity * 4].view(
            np.uint32
        )
        self.host_address = int(self.raw.ctypes.data)

    def entry_words(self, index: int) -> np.ndarray:
        start = self.entries_offset + index * ENTRY_BYTES
        return self.raw[start : start + ENTRY_HEADER_BYTES].view(np.uint32)

    def entry_bytes(self, index: int, offset: int, length: int) -> bytes:
        start = self.entries_offset + index * ENTRY_BYTES + offset
        return self.raw[start : start + length].tobytes()

    def write_entry_bytes(self, index: int, offset: int, data: bytes) -> None:
        start = self.entries_offset + index * ENTRY_BYTES + offset
        self.raw[start : start + len(data)] = np.frombuffer(data, dtype=np.uint8)

    def read_text(self, offset: int) -> str:
        return self.raw[offset : offset + TEXT_FIELD_BYTES].tobytes().rstrip(b"\0").decode()

    def write_text(self, offset: int, value: str) -> None:
        data = value.encode()
        if len(data) > TEXT_FIELD_BYTES:
            raise ValueError("ring text field is too long")
        self.raw[offset : offset + TEXT_FIELD_BYTES] = 0
        self.raw[offset : offset + len(data)] = np.frombuffer(data, dtype=np.uint8)

    def publish(self) -> None:
        os.rename(self.open_path, self.path)
        self.open_path = self.path

    def close(self, *, unlink: bool) -> None:
        self.raw = self.header = self.ready = None
        try:
            self._mm.close()
        except BufferError:
            pass
        if unlink:
            try:
                os.unlink(self.open_path)
            except FileNotFoundError:
                pass


@dataclass(frozen=True, slots=True)
class RingEntry:
    seq: int
    has_pull: bool
    source_offset: int
    nbytes: int
    source_slot_id: int
    slot_generation: int
    payload: bytes
    landing_index: int
    landing_offset: int


class TargetRingWriter:
    """Target-side producer for one owner Draft session."""

    def __init__(self, path: str) -> None:
        self._ring = _RingMapping(path, create=False)
        self.path = path
        self.capacity = self._ring.capacity
        self.next_seq = 1
        self.max_entry_bytes = int(self._ring.header[H_MAX_ENTRY_BYTES])
        self.owner_pid = int(self._ring.header[H_OWNER_PID])
        self.source_ipc_handle = b""

    @property
    def state(self) -> int:
        return int(self._ring.header[H_STATE])

    def register_source(
        self,
        *,
        ipc_handle: bytes,
        allocation_length: int,
        pool_generation: str,
        source_session_id: str,
    ) -> None:
        """Publish the source allocation the Draft maps for every entry."""

        ring = self._ring
        if self.state != STATE_CREATED:
            return
        if len(ipc_handle) != EVENT_HANDLE_BYTES:
            raise ValueError("source IPC handle must be 64 bytes")
        ring.raw[SOURCE_HANDLE_OFFSET : SOURCE_HANDLE_OFFSET + 64] = np.frombuffer(
            ipc_handle, dtype=np.uint8
        )
        ring.header[H_SOURCE_LENGTH_LO] = allocation_length & 0xFFFFFFFF
        ring.header[H_SOURCE_LENGTH_HI] = allocation_length >> 32
        ring.write_text(POOL_GENERATION_OFFSET, pool_generation)
        ring.write_text(SOURCE_SESSION_OFFSET, source_session_id)
        ring.header[H_STATE] = STATE_REGISTERED

    def read_proven(self, seq: int) -> bool:
        """Whether the Draft worker published INPUT_READY (after its copy).

        The worker serves entries in seq order, so a later seq already
        published at the same index (a later lap) also proves this one.
        """

        # A never-written word (0) is 2**32 - seq behind any early seq.
        return _seq_after(int(self._ring.ready[seq % self.capacity]), seq & SEQ_MASK)

    def free_entries(self) -> int:
        # Counts a skipped wire 0 as outstanding: conservative by one entry.
        consumed = int(self._ring.header[H_CONSUMED])
        outstanding = ((self.next_seq - 1) - consumed) & SEQ_MASK
        return self.capacity - outstanding

    def write(
        self,
        *,
        payload: bytes,
        has_pull: bool,
        source_offset: int = 0,
        nbytes: int = 0,
        source_slot_id: int = 0,
        slot_generation: int = 0,
        event_handle: bytes = b"",
    ) -> int:
        """Publish one entry; the caller checked state and free_entries()."""

        if len(payload) > MAX_PAYLOAD_BYTES:
            raise DirectRingError("ring payload exceeds the entry size")
        if has_pull and len(event_handle) != EVENT_HANDLE_BYTES:
            raise DirectRingError("pull entry requires a producer event handle")
        seq = self.next_seq
        index = seq % self.capacity
        ring = self._ring
        ring.write_entry_bytes(index, PAYLOAD_OFFSET, payload)
        if has_pull:
            ring.write_entry_bytes(index, EVENT_HANDLE_OFFSET, event_handle)
        words = ring.entry_words(index)
        words[E_FLAGS] = FLAG_HAS_PULL if has_pull else 0
        words[E_SOURCE_OFFSET_LO] = source_offset & 0xFFFFFFFF
        words[E_SOURCE_OFFSET_HI] = source_offset >> 32
        words[E_NBYTES] = nbytes
        words[E_SOURCE_SLOT] = source_slot_id
        words[E_SLOT_GENERATION] = slot_generation
        words[E_PAYLOAD_LEN] = len(payload)
        words[E_EPOCH] = seq >> 32
        # x86 keeps stores in program order: seq becomes visible last.
        words[E_SEQ] = seq & SEQ_MASK
        self.next_seq = next_seq(seq)
        return seq

    def close(self, *, unlink: bool = False) -> None:
        # The Draft owns the file; the Target unlinks only a revoked dead one.
        self._ring.close(unlink=unlink)


class DraftRingEndpoint:
    """Draft-side consumer state of one ring."""

    def __init__(
        self,
        *,
        path: str,
        ring_id: int,
        target_session_id: str,
        capacity: int,
        max_entry_bytes: int,
        runtime,
    ) -> None:
        self._ring = _RingMapping(path, capacity=capacity, create=True)
        self.path = path
        self.ring_id = ring_id
        self.target_session_id = target_session_id
        self.capacity = capacity
        header = self._ring.header
        header[H_MAGIC] = RING_MAGIC
        header[H_VERSION] = RING_VERSION
        header[H_CAPACITY] = capacity
        header[H_ENTRY_BYTES] = ENTRY_BYTES
        header[H_CONSUMED] = 0
        header[H_MAX_ENTRY_BYTES] = max_entry_bytes
        header[H_OWNER_PID] = os.getpid()
        self._runtime = runtime
        self.device_base = runtime.host_register(
            self._ring.host_address, self._ring.total_bytes
        )
        self.ready_device = self.device_base + self._ring.ready_offset
        self.placement_buffer = np.full(capacity, -1, dtype=np.int32)
        self.placement_offset = np.zeros(capacity, dtype=np.uint64)
        self.next_seq = 1
        self.attached = False
        self.pool_generation = ""
        self.source_session_id = ""
        header[H_STATE] = STATE_CREATED
        self._ring.publish()

    @property
    def entries_address(self) -> int:
        return self._ring.host_address + self._ring.entries_offset

    def registration(self) -> tuple[bytes, int, str, str] | None:
        ring = self._ring
        if int(ring.header[H_STATE]) != STATE_REGISTERED:
            return None
        handle = ring.raw[SOURCE_HANDLE_OFFSET : SOURCE_HANDLE_OFFSET + 64].tobytes()
        length = int(ring.header[H_SOURCE_LENGTH_LO]) | (
            int(ring.header[H_SOURCE_LENGTH_HI]) << 32
        )
        return (
            handle,
            length,
            ring.read_text(POOL_GENERATION_OFFSET),
            ring.read_text(SOURCE_SESSION_OFFSET),
        )

    def mark_serving(self, *, pool_generation: str, source_session_id: str) -> None:
        self.pool_generation = pool_generation
        self.source_session_id = source_session_id
        self.attached = True
        self._ring.header[H_STATE] = STATE_SERVING

    def take_ready(self) -> RingEntry | None:
        """Return the next entry whose worker publication completed, in order."""

        seq = self.next_seq
        index = seq % self.capacity
        low = seq & SEQ_MASK
        words = self._ring.entry_words(index)
        if (
            int(words[E_SEQ]) != low
            or int(self._ring.ready[index]) != low
            or int(words[E_EPOCH]) != seq >> 32
        ):
            return None
        payload_len = int(words[E_PAYLOAD_LEN])
        if payload_len > MAX_PAYLOAD_BYTES:
            raise DirectRingError("ring entry payload length is invalid")
        entry = RingEntry(
            seq=seq,
            has_pull=bool(int(words[E_FLAGS]) & FLAG_HAS_PULL),
            source_offset=int(words[E_SOURCE_OFFSET_LO])
            | (int(words[E_SOURCE_OFFSET_HI]) << 32),
            nbytes=int(words[E_NBYTES]),
            source_slot_id=int(words[E_SOURCE_SLOT]),
            slot_generation=int(words[E_SLOT_GENERATION]),
            payload=self._ring.entry_bytes(index, PAYLOAD_OFFSET, payload_len),
            landing_index=int(self.placement_buffer[index]),
            landing_offset=int(self.placement_offset[index]),
        )
        # Entries are consumed in seq order; the Target may now reuse this one.
        self.next_seq = next_seq(seq)
        self._ring.header[H_CONSUMED] = low
        return entry

    def close(self) -> None:
        if self._ring is None:
            return
        self._ring.header[H_STATE] = STATE_CLOSED
        try:
            self._runtime.host_unregister(self._ring.host_address)
        finally:
            self._ring.close(unlink=True)
            self._ring = None


def _build_worker_library() -> Path:
    source = Path(__file__).with_name("csrc") / "dt_ring_worker.cpp"
    text = source.read_bytes()
    digest = hashlib.sha256(text).hexdigest()[:16]
    cache = Path(os.environ.get("DT_DIRECT_RING_BUILD_DIR", Path.home() / ".cache" / "dt-ring-worker"))
    cache.mkdir(parents=True, exist_ok=True)
    library = cache / f"dt_ring_worker-{digest}.so"
    if library.exists():
        return library
    with open(cache / ".build.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.exists():
            partial = library.with_suffix(f".so.{os.getpid()}")
            subprocess.run(
                [
                    os.environ.get("CXX", "g++"),
                    "-O2",
                    "-std=c++17",
                    "-shared",
                    "-fPIC",
                    "-o",
                    str(partial),
                    str(source),
                    "-ldl",
                    "-lpthread",
                ],
                check=True,
            )
            os.replace(partial, library)
    return library


class DraftRingWorker:
    """ctypes owner of the native ring worker thread and its landing buffers."""

    def __init__(self, *, device: int, cpu: int, landing, row_bytes: int) -> None:
        lib = ctypes.CDLL(str(_build_worker_library()))
        u64, u32, i32, vp = ctypes.c_uint64, ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p
        lib.dt_ring_worker_create.argtypes = [i32, i32, u64, u64, u32, u64]
        lib.dt_ring_worker_create.restype = vp
        lib.dt_ring_worker_attach.argtypes = [vp, i32, u64, u64, u32, u64, u64, u64, u64, u64, u32]
        lib.dt_ring_worker_attach.restype = i32
        lib.dt_ring_worker_detach.argtypes = [vp, i32]
        lib.dt_ring_worker_detach.restype = u32
        lib.dt_ring_worker_seal.argtypes = [vp, u32]
        lib.dt_ring_worker_seal.restype = u64
        lib.dt_ring_worker_try_free.argtypes = [vp, u32, u64]
        lib.dt_ring_worker_try_free.restype = i32
        lib.dt_ring_worker_start.argtypes = [vp]
        lib.dt_ring_worker_start.restype = i32
        lib.dt_ring_worker_error.argtypes = [vp, ctypes.POINTER(i32)]
        lib.dt_ring_worker_error.restype = i32
        lib.dt_ring_worker_entries_done.argtypes = [vp]
        lib.dt_ring_worker_entries_done.restype = u64
        lib.dt_ring_worker_landing_stalls.argtypes = [vp]
        lib.dt_ring_worker_landing_stalls.restype = u64
        lib.dt_ring_worker_sync.argtypes = [vp]
        lib.dt_ring_worker_sync.restype = i32
        lib.dt_ring_worker_destroy.argtypes = [vp]
        lib.dt_ring_worker_destroy.restype = i32
        self._lib = lib
        # ``landing`` is a contiguous [count, rows, width] CUDA tensor kept
        # alive by this object until destroy() drained the worker stream.
        self.landing = landing
        self.landing_count = int(landing.shape[0])
        self.landing_bytes = int(landing[0].numel() * landing.element_size())
        self.row_bytes = int(row_bytes)
        self._handle = lib.dt_ring_worker_create(
            device,
            cpu,
            landing.data_ptr(),
            self.landing_bytes,
            self.landing_count,
            self.row_bytes,
        )
        if not self._handle:
            raise DirectRingError("native ring worker could not be created")
        self._endpoints: dict[int, DraftRingEndpoint] = {}
        if lib.dt_ring_worker_start(self._handle) != 0:
            lib.dt_ring_worker_destroy(self._handle)
            self._handle = None
            raise DirectRingError("native ring worker did not start")

    def attach(self, endpoint: DraftRingEndpoint, *, source_base: int, source_length: int) -> None:
        rc = self._lib.dt_ring_worker_attach(
            self._handle,
            endpoint.ring_id,
            endpoint.entries_address,
            ENTRY_BYTES,
            endpoint.capacity,
            endpoint.ready_device,
            source_base,
            source_length,
            endpoint.placement_buffer.ctypes.data,
            endpoint.placement_offset.ctypes.data,
            endpoint.next_seq & SEQ_MASK,
        )
        if rc != 0:
            raise DirectRingError("native ring worker rejected a ring")
        self._endpoints[endpoint.ring_id] = endpoint

    def seal(self, landing_index: int) -> int:
        return int(self._lib.dt_ring_worker_seal(self._handle, landing_index))

    def try_free(self, landing_index: int, released: int) -> int:
        return int(self._lib.dt_ring_worker_try_free(self._handle, landing_index, released))

    def check(self) -> None:
        ring_id = ctypes.c_int(-1)
        code = self._lib.dt_ring_worker_error(self._handle, ctypes.byref(ring_id))
        if code:
            raise DirectRingError(
                f"native ring worker failed: code={code} ring={ring_id.value}"
            )

    @property
    def landing_stalls(self) -> int:
        return int(self._lib.dt_ring_worker_landing_stalls(self._handle))

    def destroy(self) -> int:
        if not self._handle:
            return 0
        # Endpoints stay SERVING: the owner closes them (CLOSED means "no
        # further candidate pushes") only after its push stream drained.
        rc = int(self._lib.dt_ring_worker_destroy(self._handle))
        self._handle = None
        self._endpoints.clear()
        return rc
