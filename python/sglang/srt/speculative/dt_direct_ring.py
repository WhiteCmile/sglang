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

With DT_RING_PUBLISH=1 a verify wave can instead be published by a GPU
kernel (csrc/dt_ring_publish.cu via RingGpuPublisher): a same-stride 4096-byte
record with a 128-byte header (u64 seq released last with system scope), a
56-entry request table, and a 512-token array, discriminated by version=2 at
byte 12 (CPU entries keep the always-zero source-offset high word there).
Both entry kinds share one seq space and the same INPUT_READY/placement
contract.

Reader/credit reconciliation for kernel-published records: the native
worker validates fences and performs the pull, but never resolves request
ids — the native rid ledger (dt_rid_ledger_*) is consulted only by the
Draft main loop when materializing tasks.  The credit word (H_CONSUMED)
therefore advances in ``take_ready`` (main loop), not in the worker: a
record is released for reuse only after its request table and token array
have been fully parsed.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import logging
import mmap
import os
import re
import shutil
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DIRECT_RING = os.environ.get("DT_DIRECT_RING", "1") == "1"
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

# §7.1 GPU ring publish (DT_RING_PUBLISH=1).  Kernel-written 4096-byte
# records share the ring with the CPU-written msgpack entries above: an
# entry is a v2 record when the u32 at byte offset 12 is V2_VERSION (old
# entries keep the high half of the source offset there, always 0 because
# source allocations are smaller than 4 GiB — enforced by RingGpuPublisher).
RING_PUBLISH = os.environ.get("DT_RING_PUBLISH", "1") == "1"
# Q15b (DT_RING_PUBLISH_AT_LAUNCH=1): eligible deferred verify waves enqueue
# the publish kernel on the verify stream at launch, right behind the verify
# forward, instead of from host park one result boundary later.  Launch-time
# publishes need their own staging slots: the host stages wave k+1 while the
# GPU may still hold wave k-1's kernel queued behind its forward, so a single
# staging buffer could be overwritten before the queued kernel reads it.  The
# launch slots rotate; slot 0 remains the park path's staging.
RING_PUBLISH_AT_LAUNCH = os.environ.get("DT_RING_PUBLISH_AT_LAUNCH", "0") == "1"
LAUNCH_STAGING_SLOTS = 8
V2_VERSION = 2
V2_HEADER_BYTES = 128
V2_MAX_REQ = 56
V2_REQ_ENTRY_BYTES = 32
V2_MAX_TOK = 512
V2_TOKEN_OFFSET = V2_HEADER_BYTES + V2_MAX_REQ * V2_REQ_ENTRY_BYTES  # 1920
V2_TAIL_OFFSET = V2_TOKEN_OFFSET + V2_MAX_TOK * 4  # 3968
V2_EVENT_HANDLE_OFFSET = V2_TAIL_OFFSET + 8  # 3976, 64 bytes
V2_MARKER_B_XOR = 0x9E3779B97F4A7C15
V2_FLAG_HAS_PULL = 1
V2_SOURCE_LENGTH_LIMIT = 1 << 32

# Kernel staging buffer: one 32-byte entry per batch slot (rid_hash u64,
# round_id u32, attempt u32, owner_epoch u64, token_src device pointer),
# followed by the compact u32 active-slot list.
V2_STAGING_ENTRY_BYTES = 32

_FNV1A64_OFFSET = 0xCBF29CE484222325
_FNV1A64_PRIME = 0x100000001B3
_MASK64 = 0xFFFFFFFFFFFFFFFF


def rid_hash(rid: str) -> int:
    """FNV-1a 64 of the request id; the Draft ledger maps it back."""

    h = _FNV1A64_OFFSET
    for byte in rid.encode():
        h = ((h ^ byte) * _FNV1A64_PRIME) & _MASK64
    return h

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
H_SOURCE_SLOT_NBYTES_LO = 10
H_SOURCE_SLOT_NBYTES_HI = 11
H_SOURCE_SLOT_COUNT = 12
H_SOURCE_ALLOC_OFFSET_LO = 13
H_SOURCE_ALLOC_OFFSET_HI = 14
# Header byte offsets of Target source registration fields.
SOURCE_HANDLE_OFFSET = 64
POOL_GENERATION_OFFSET = 128
SOURCE_SESSION_OFFSET = 192
TEXT_FIELD_BYTES = 64
# Per-slot current generation table (u32 x slot_count), written by the
# Target at reservation time; the worker rejects records naming a stale
# generation.  The table starts after the text fields in page 0.
GENERATION_TABLE_OFFSET = 256

# v2 record fence: the CPU pre-writes a pool-generation tag at
# reserved[1] before launching the kernel (the release publishes it, like
# the event handle); the Draft rejects records from a stale pool.
V2_POOL_TAG_OFFSET = 88


def pool_generation_tag(pool_generation: str) -> int:
    """64-bit tag of a pool generation, stored in v2 record reserved[1]."""

    return int.from_bytes(
        hashlib.sha256(pool_generation.encode()).digest()[:8], "little"
    )

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


# Per-entry candidate-destination stride for kernel-published waves: up to
# V2_MAX_REQ (slot_id u32, generation u32, authorization_nonce u64) triples,
# a u64 validity word (the wave seq, written last), and the two arena buffer
# offsets the Draft needs to build allocation-relative destination refs
# (constant per arena incarnation; covered by the same validity word).  The
# Target CPU-writes them when the flush-time arena reservation lands; the
# Draft's push then never waits on the Coordinator confirmation copy.
V2_DEST_ENTRY_BYTES = 16
V2_DEST_STRIDE = 1024
V2_DEST_SEQ_OFFSET = V2_MAX_REQ * V2_DEST_ENTRY_BYTES
V2_DEST_DATA_OFFSET = V2_DEST_SEQ_OFFSET + 8
V2_DEST_META_OFFSET = V2_DEST_DATA_OFFSET + 8

# Mailbox slot layout, mirroring dt_ring_worker.cpp.  The native worker
# snapshots every served entry here before it advances H_CONSUMED, so the
# main loop parses record content that a ring-entry rewrite cannot tear.
MAIL_SEQ_TAG = 0
MAIL_FORMAT = 8
MAIL_COUNT = 12
MAIL_TOK_TOTAL = 16
MAIL_HEADER = 24
MAIL_REQS = 152
MAIL_TOKENS = 1944
MAIL_V1_PAYLOAD = 3992
MAIL_RIDS = 7960
MAIL_MARKER_B = 22296
MAIL_EVENT = 22304
MAIL_DROPPED = 22368
MAIL_STRIDE = 24576


def _layout(capacity: int) -> tuple[int, int, int, int]:
    page = mmap.PAGESIZE
    entries_offset = page
    ready_offset = entries_offset + -(-capacity * ENTRY_BYTES // page) * page
    dest_offset = ready_offset + -(-capacity * 4 // page) * page
    total = dest_offset + -(-capacity * V2_DEST_STRIDE // page) * page
    return entries_offset, ready_offset, dest_offset, total


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
                _, _, _, total = _layout(capacity)
                os.ftruncate(fd, total)
            else:
                with mmap.mmap(fd, mmap.PAGESIZE, prot=mmap.PROT_READ) as head:
                    words = np.frombuffer(head, dtype=np.uint32, count=8).copy()
                if words[H_MAGIC] != RING_MAGIC or words[H_VERSION] != RING_VERSION:
                    raise DirectRingError(f"ring {path} has an unknown format")
                capacity = int(words[H_CAPACITY])
                _, _, _, total = _layout(capacity)
                if os.fstat(fd).st_size < total:
                    raise DirectRingError(f"ring {path} is truncated")
            self._mm = mmap.mmap(fd, total)
        finally:
            os.close(fd)
        self.path = path
        self.open_path = open_path
        self.capacity = capacity
        self.total_bytes = total
        self.entries_offset, self.ready_offset, self.dest_offset, _ = _layout(capacity)
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


@dataclass(frozen=True, slots=True)
class RingRecordV2Request:
    rid_hash: int
    round_id: int
    attempt: int
    token_offset: int
    token_count: int
    bonus_token: int
    seq_len: int
    # Resolved by the native worker into the mailbox; empty for pre-mailbox
    # callers that construct the entry by hand.
    rid: str = ""


@dataclass(frozen=True, slots=True)
class RingRecordV2:
    """One kernel-published verify wave (§7.1), read after the ready word."""

    seq: int
    owner_epoch: int
    source_slot_id: int
    slot_generation: int
    source_byte_offset: int
    nbytes: int
    row_count: int
    row_bytes: int
    block_size: int
    hidden_produced_ns: int
    requests: tuple[RingRecordV2Request, ...]
    token_ids: tuple[int, ...]
    producer_event_handle: bytes
    landing_index: int
    landing_offset: int
    # Bitmask of requests whose rid was already released when the worker
    # served the record; they are consumed without creating tasks.  A
    # fully dropped wave is served with landing_index = -1 (no pull).
    dropped_mask: int = 0


class NativeRidLedger:
    """Native rid_hash -> rid table in the ring worker library.

    Populated by the Draft main loop at request admission/release (the only
    writer) and read by the native worker thread while serving v2 records;
    a sequence counter keeps the table lock-free for concurrent readers.
    """

    def __init__(self, capacity: int = 1 << 16) -> None:
        lib = ctypes.CDLL(str(_build_worker_library()))
        i32, u32, u64, vp, cp = (
            ctypes.c_int,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_char_p,
        )
        lib.dt_rid_ledger_create.argtypes = [u32]
        lib.dt_rid_ledger_create.restype = vp
        lib.dt_rid_ledger_put.argtypes = [vp, u64, cp]
        lib.dt_rid_ledger_put.restype = i32
        lib.dt_rid_ledger_remove.argtypes = [vp, u64, cp]
        lib.dt_rid_ledger_remove.restype = i32
        lib.dt_rid_ledger_resolve.argtypes = [vp, u64, cp, u32]
        lib.dt_rid_ledger_resolve.restype = i32
        lib.dt_rid_ledger_destroy.argtypes = [vp]
        lib.dt_rid_ledger_destroy.restype = None
        self._lib = lib
        self._handle = lib.dt_rid_ledger_create(capacity)
        if not self._handle:
            raise DirectRingError("native rid ledger could not be created")

    def register(self, rid: str) -> int:
        value = rid_hash(rid)
        rc = self._lib.dt_rid_ledger_put(self._handle, value, rid.encode())
        if rc == -1:
            raise DirectRingError(f"rid hash collision at {rid!r}")
        if rc != 0:
            raise DirectRingError(f"native rid ledger rejected {rid!r}: rc={rc}")
        return value

    def resolve(self, value: int) -> str | None:
        buf = ctypes.create_string_buffer(256)
        rc = self._lib.dt_rid_ledger_resolve(self._handle, value, buf, len(buf))
        if rc == -5:
            return None  # released tombstone
        if rc < 0:
            raise DirectRingError(f"native rid ledger resolve failed: rc={rc}")
        return buf.value.decode() if rc else None

    def evict(self, rid: str) -> None:
        self._lib.dt_rid_ledger_remove(self._handle, rid_hash(rid), rid.encode())

    def close(self) -> None:
        if self._handle:
            self._lib.dt_rid_ledger_destroy(self._handle)
            self._handle = None


class RingGpuPublisher:
    """Target-side §7.1 kernel publication channel over one ring.

    The scheduler's per-wave work shrinks to: per-request scalar stores into
    a device-mapped staging buffer (``stage_request``), one CPU pre-write of
    the producer-event handle, a credit check, and one kernel launch
    (``publish``).  The kernel fills the whole 4096-byte record from the
    staging buffer and the on-GPU tensors and releases seq with system scope.
    """

    def __init__(
        self,
        writer: TargetRingWriter,
        *,
        runtime,
        device: int,
        max_batch_slots: int,
        block_size: int,
        row_bytes: int,
    ) -> None:
        if not RING_PUBLISH:
            raise DirectRingError("RingGpuPublisher requires DT_RING_PUBLISH=1")
        if int(writer._ring.header[H_SOURCE_LENGTH_HI]) != 0:
            # v2 records share the ring with old entries; the discriminator
            # (u32 at byte 12) collides with a nonzero source offset high word.
            raise DirectRingError("GPU ring publish needs a < 4 GiB source allocation")
        from sglang.srt.speculative.dt_ring_publish_ext import load_ext

        self._writer = writer
        self._ext = load_ext(device)
        self.block_size = int(block_size)
        self.row_bytes = int(row_bytes)
        self.max_batch_slots = int(max_batch_slots)
        self._pool_tag = pool_generation_tag(
            writer._ring.read_text(POOL_GENERATION_OFFSET)
        )
        self._runtime = runtime
        self._ring_dev = runtime.host_register(
            writer._ring.host_address, writer._ring.total_bytes
        )
        self._entries_dev = self._ring_dev + writer._ring.entries_offset
        self._active_offset = self.max_batch_slots * V2_STAGING_ENTRY_BYTES
        self._counts_offset = self._active_offset + 4 * V2_MAX_REQ
        nbytes = self._counts_offset + 4 * self.max_batch_slots
        if RING_PUBLISH_AT_LAUNCH:
            # Launch slots 1..LAUNCH_STAGING_SLOTS (slot 0 is the park path):
            # each carries its own compacted entries and active list.  The
            # host can stage at most a handful of unpublished waves ahead of
            # the GPU (every park syncs the previous wave's copy_done), so
            # eight slots bound the rotation with margin.
            self._launch_slot_stride = (
                V2_MAX_REQ * V2_STAGING_ENTRY_BYTES + 4 * V2_MAX_REQ
            )
            self._launch_entries_base = nbytes
            self._launch_active_base = self._launch_entries_base + (
                LAUNCH_STAGING_SLOTS * self._launch_slot_stride
            )
            nbytes = self._launch_active_base + LAUNCH_STAGING_SLOTS * 4 * V2_MAX_REQ
            self._launch_slot_cycle = -1
        self._staging_host, self._staging_dev = self._ext.staging_alloc(nbytes)
        buf = (ctypes.c_uint8 * nbytes).from_address(self._staging_host)
        flat = np.frombuffer(buf, dtype=np.uint8)
        self._stg_u8 = flat
        self._stg_u64 = flat.view(np.uint64)
        self._stg_u32 = flat.view(np.uint32)
        self._hash_by_rid: dict[str, int] = {}

    @property
    def writer(self) -> TargetRingWriter:
        return self._writer

    def free_entries(self) -> int:
        return self._writer.free_entries()

    def stage_counts(self, counts: list[int]) -> None:
        """Pinned per-batch-slot committed counts for the kernel, one slice."""

        base = self._counts_offset >> 2
        self._stg_u32[base : base + len(counts)] = counts

    @property
    def counts_dev(self) -> int:
        return self._staging_dev + self._counts_offset

    def rid_hash_for(self, rid: str) -> int:
        value = self._hash_by_rid.get(rid)
        if value is None:
            value = rid_hash(rid)
            self._hash_by_rid[rid] = value
        return value

    def next_launch_slot(self) -> int:
        """Rotate the launch-time staging slots (1..LAUNCH_STAGING_SLOTS)."""

        if not RING_PUBLISH_AT_LAUNCH:
            raise DirectRingError("launch staging requires DT_RING_PUBLISH_AT_LAUNCH=1")
        self._launch_slot_cycle = (self._launch_slot_cycle + 1) % LAUNCH_STAGING_SLOTS
        return self._launch_slot_cycle + 1

    def _slot_offsets(self, staging_slot: int) -> tuple[int, int]:
        if staging_slot == 0:
            return 0, self._active_offset
        base = staging_slot - 1
        return (
            self._launch_entries_base + base * self._launch_slot_stride,
            self._launch_active_base + base * 4 * V2_MAX_REQ,
        )

    def stage_requests(self, entries, *, staging_slot: int = 0) -> None:
        """Write the compacted request staging entries in one pack+copy.

        entries: [(rid_hash, round_id, attempt, owner_epoch, token_src)] in
        active order; the kernel reads entry i for active[i].
        """

        flat = []
        for value, round_id, attempt, owner_epoch, token_src in entries:
            flat += (value, round_id, attempt, owner_epoch, token_src)
        packed = struct.pack("<" + "QIIQQ" * len(entries), *flat)
        entries_offset, _ = self._slot_offsets(staging_slot)
        self._stg_u8[entries_offset : entries_offset + len(packed)] = np.frombuffer(
            packed, dtype=np.uint8
        )

    def stage_request(
        self,
        batch_slot: int,
        *,
        rid: str,
        round_id: int,
        attempt: int,
        owner_epoch: int,
        token_src: int,
    ) -> None:
        """One pinned 32-byte store; no CUDA API, no sync."""

        value = self._hash_by_rid.get(rid)
        if value is None:
            value = rid_hash(rid)
            self._hash_by_rid[rid] = value
        base = batch_slot * 4  # u64 element index of this slot's entry
        self._stg_u64[base] = value
        self._stg_u64[base + 1] = (round_id & 0xFFFFFFFF) | (attempt << 32)
        self._stg_u64[base + 2] = owner_epoch
        self._stg_u64[base + 3] = token_src

    def publish(
        self,
        *,
        stream: int,
        active_slots: list[int],
        slot_id: int,
        generation: int,
        source_byte_offset: int,
        nbytes: int,
        row_count: int,
        event_handle: bytes,
        accept_dev: int,
        bonus_dev: int,
        seq_lens_dev: int,
        offset_stride: int = 0,
        token_total: int | None = None,
        staging_slot: int = 0,
    ) -> int:
        """Launch the publish kernel; the caller staged every active slot.

        offset_stride 0 packs each request's rows at the prefix sum of the
        committed counts; a nonzero stride places them at batch_slot*stride
        (in-place reservation layout, where the slot holds the whole packed
        wave).  staging_slot selects the staging region the kernel reads;
        the launch-time path rotates nonzero slots so a queued kernel never
        reads entries a later wave's host staging has overwritten."""

        writer = self._writer
        request_count = len(active_slots)
        if writer.state != STATE_SERVING:
            raise DirectRingError("ring is not serving")
        # The record bounds the request table and the compacted token array;
        # physical feature rows (block-strided for in-place waves) are bounded
        # separately by the Draft landing capacity below.
        token_base = self._counts_offset >> 2
        if token_total is None:
            token_total = int(
                self._stg_u32[token_base : token_base + self.max_batch_slots][
                    list(active_slots)
                ].sum()
            )
        if not 0 < request_count <= V2_MAX_REQ or token_total > V2_MAX_TOK:
            raise DirectRingError("wave exceeds the v2 record geometry")
        if len(event_handle) != EVENT_HANDLE_BYTES:
            raise DirectRingError("pull record requires a producer event handle")
        if nbytes > writer.max_entry_bytes:
            raise DirectRingError("pull exceeds the Draft landing buffer")
        if writer.free_entries() <= 0:
            raise DirectRingError("ring has no free entries")
        seq = writer.next_seq
        index = seq % writer.capacity
        # CPU pre-write of the producer-event handle and the pool-generation
        # fence tag; the kernel's release store of seq publishes both
        # (validated cpu-field ordering).  The slot's current generation is
        # registered the same way so the worker can reject stale records.
        writer._ring.write_entry_bytes(index, V2_EVENT_HANDLE_OFFSET, event_handle)
        writer._ring.write_entry_bytes(
            index, V2_POOL_TAG_OFFSET, self._pool_tag.to_bytes(8, "little")
        )
        writer._ring.header[(GENERATION_TABLE_OFFSET >> 2) + slot_id] = generation
        entries_offset, active_offset = self._slot_offsets(staging_slot)
        base = active_offset >> 2
        self._stg_u32[base : base + request_count] = list(active_slots)
        self._ext.publish(
            stream=stream,
            ring_dev=self._entries_dev,
            capacity=writer.capacity,
            seq=seq,
            source_slot_id=slot_id,
            slot_generation=generation,
            source_byte_offset=source_byte_offset,
            nbytes=nbytes,
            row_count=row_count,
            row_bytes=self.row_bytes,
            request_count=request_count,
            block_size=self.block_size,
            offset_stride=offset_stride,
            staging_dev=self._staging_dev,
            staging_entries_offset=entries_offset,
            staging_active_offset=active_offset,
            accept_dev=accept_dev,
            bonus_dev=bonus_dev,
            seq_lens_dev=seq_lens_dev,
        )
        writer.next_seq = next_seq(seq)
        return seq

    def close(self) -> None:
        try:
            self._ext.staging_free(self._staging_host)
        finally:
            self._runtime.host_unregister(self._writer._ring.host_address)


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
        slot_nbytes: int = 0,
        slot_count: int = 0,
        allocation_offset: int = 0,
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
        ring.header[H_SOURCE_SLOT_NBYTES_LO] = slot_nbytes & 0xFFFFFFFF
        ring.header[H_SOURCE_SLOT_NBYTES_HI] = slot_nbytes >> 32
        ring.header[H_SOURCE_SLOT_COUNT] = slot_count
        ring.header[H_SOURCE_ALLOC_OFFSET_LO] = allocation_offset & 0xFFFFFFFF
        ring.header[H_SOURCE_ALLOC_OFFSET_HI] = allocation_offset >> 32
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

    def write_destinations(
        self,
        seq: int,
        entries: list[tuple[int, int, int]],
        *,
        data_buffer_offset: int = 0,
        metadata_buffer_offset: int = 0,
    ) -> None:
        """CPU-write (slot_id, generation, nonce) triples for a published wave.

        Seqlock discipline: the validity word is invalidated before any
        field store and republished last, so a concurrent reader either sees
        the previous lap fully intact or detects the rewrite and falls back.
        """

        index = seq % self.capacity
        base = self._ring.dest_offset + index * V2_DEST_STRIDE
        raw = self._ring.raw
        struct.pack_into("<Q", raw, base + V2_DEST_SEQ_OFFSET, 0)
        for position, (slot_id, generation, nonce) in enumerate(entries):
            struct.pack_into(
                "<IIQ",
                raw,
                base + position * V2_DEST_ENTRY_BYTES,
                slot_id,
                generation,
                nonce,
            )
        struct.pack_into("<Q", raw, base + V2_DEST_DATA_OFFSET, data_buffer_offset)
        struct.pack_into("<Q", raw, base + V2_DEST_META_OFFSET, metadata_buffer_offset)
        struct.pack_into("<Q", raw, base + V2_DEST_SEQ_OFFSET, seq)

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
        self._mailbox = np.zeros((capacity, MAIL_STRIDE), dtype=np.uint8)
        self._mail_ack = np.zeros(1, dtype=np.uint64)
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

    def destination(
        self, seq: int, position: int
    ) -> tuple[int, int, int, int, int] | None:
        """The arena destination for one request of a published wave.

        Returns (slot_id, generation, authorization_nonce, data_buffer_offset,
        metadata_buffer_offset) once the Target has written the wave's
        validity word, else None.  The buffer offsets let the Draft build the
        same allocation-relative destination the Coordinator copy carries.

        Seqlock read: accept only if the validity word equals ``seq`` both
        before and after the field reads, so a lap rewrite landing mid-read
        (fields partially overwritten under the old validity word) is
        rejected and the caller falls back to the Coordinator copy.  A full
        lap overwrite reads back the newer seq and likewise returns None.
        """

        index = seq % self.capacity
        base = self._ring.dest_offset + index * V2_DEST_STRIDE
        raw = self._ring.raw
        for _ in range(2):
            first = struct.unpack_from("<Q", raw, base + V2_DEST_SEQ_OFFSET)[0]
            if first != seq:
                return None
            slot_id, generation, nonce = struct.unpack_from(
                "<IIQ", raw, base + position * V2_DEST_ENTRY_BYTES
            )
            data_offset, metadata_offset = struct.unpack_from(
                "<QQ", raw, base + V2_DEST_DATA_OFFSET
            )
            if struct.unpack_from("<Q", raw, base + V2_DEST_SEQ_OFFSET)[0] == first:
                return (slot_id, generation, nonce, data_offset, metadata_offset)
        return None

    def mark_serving(self, *, pool_generation: str, source_session_id: str) -> None:
        self.pool_generation = pool_generation
        self.source_session_id = source_session_id
        self.attached = True
        self._ring.header[H_STATE] = STATE_SERVING

    def take_ready(self) -> RingEntry | RingRecordV2 | None:
        """Return the next entry whose worker publication completed, in order.

        Content comes from the worker's mailbox slot, not the ring entry:
        the native worker advances H_CONSUMED itself once the snapshot is
        sealed, so the Target may already have rewritten the ring entry.
        """

        seq = self.next_seq
        index = seq % self.capacity
        low = seq & SEQ_MASK
        if int(self._ring.ready[index]) != low:
            return None
        mailbox = self._mailbox[index]
        if struct.unpack_from("<Q", mailbox, MAIL_SEQ_TAG)[0] != seq:
            return None
        format_tag = struct.unpack_from("<I", mailbox, MAIL_FORMAT)[0]
        if format_tag == 2:
            entry = self._take_ready_v2(seq, index, mailbox)
            if entry is None:
                return None
        else:
            (
                v1_seq,
                flags,
                source_offset,
                nbytes,
                source_slot_id,
                slot_generation,
                payload_len,
                epoch,
            ) = struct.unpack_from("<IIQIIIII", mailbox, MAIL_HEADER)
            if v1_seq != low or epoch != seq >> 32:
                return None
            if payload_len > MAX_PAYLOAD_BYTES:
                raise DirectRingError("ring entry payload length is invalid")
            entry = RingEntry(
                seq=seq,
                has_pull=bool(flags & FLAG_HAS_PULL),
                source_offset=source_offset,
                nbytes=nbytes,
                source_slot_id=source_slot_id,
                slot_generation=slot_generation,
                payload=bytes(
                    mailbox[MAIL_V1_PAYLOAD : MAIL_V1_PAYLOAD + payload_len]
                ),
                landing_index=int(self.placement_buffer[index]),
                landing_offset=int(self.placement_offset[index]),
            )
        # Entries are consumed in seq order; the ack unblocks the next lap's
        # mailbox slot (credit itself moved to the native worker).
        self.next_seq = next_seq(seq)
        self._mail_ack[0] = seq
        return entry

    def _take_ready_v2(self, seq: int, index: int, mailbox) -> RingRecordV2 | None:
        (
            seq_word,
            _flags,
            _version,
            owner_epoch,
            source_slot_id,
            slot_generation,
            source_byte_offset,
            nbytes,
            row_count,
            row_bytes,
            request_count,
            block_size,
            hidden_produced_ns,
            marker_a,
        ) = struct.unpack_from("<QIIQIIQQIIIIQQ", mailbox, MAIL_HEADER)
        if seq_word != seq:
            return None
        marker_b = struct.unpack_from("<Q", mailbox, MAIL_MARKER_B)[0]
        pool_tag = struct.unpack_from("<Q", mailbox, MAIL_HEADER + V2_POOL_TAG_OFFSET)[0]
        if pool_tag != pool_generation_tag(self.pool_generation):
            raise DirectRingError(
                "kernel-published ring record has a stale pool generation"
            )
        if (
            marker_a != (~seq) & _MASK64
            or marker_b != (seq ^ V2_MARKER_B_XOR)
            or request_count > V2_MAX_REQ
            or nbytes != row_count * row_bytes
        ):
            raise DirectRingError("kernel-published ring record failed validation")
        dropped_mask = struct.unpack_from("<Q", mailbox, MAIL_DROPPED)[0]
        requests = tuple(
            RingRecordV2Request(
                *struct.unpack_from("<Q6I", mailbox, MAIL_REQS + i * V2_REQ_ENTRY_BYTES),
                rid=(
                    ""
                    if dropped_mask >> i & 1
                    else bytes(mailbox[MAIL_RIDS + i * 256 : MAIL_RIDS + (i + 1) * 256])
                    .split(b"\0", 1)[0]
                    .decode()
                ),
            )
            for i in range(request_count)
        )
        # The strided in-place layout leaves uncommitted rows between the
        # per-request ranges; token ids always sit at compacted prefix
        # positions in the token array.
        token_total = sum(request.token_count for request in requests)
        if token_total > row_count or token_total > V2_MAX_TOK:
            raise DirectRingError("kernel-published ring record row count mismatch")
        token_ids = struct.unpack_from(f"<{token_total}I", mailbox, MAIL_TOKENS)
        return RingRecordV2(
            seq=seq,
            owner_epoch=owner_epoch,
            source_slot_id=source_slot_id,
            slot_generation=slot_generation,
            source_byte_offset=source_byte_offset,
            nbytes=nbytes,
            row_count=row_count,
            row_bytes=row_bytes,
            block_size=block_size,
            hidden_produced_ns=hidden_produced_ns,
            requests=requests,
            token_ids=token_ids,
            producer_event_handle=bytes(mailbox[MAIL_EVENT : MAIL_EVENT + EVENT_HANDLE_BYTES]),
            landing_index=int(self.placement_buffer[index]),
            landing_offset=int(self.placement_offset[index]),
            dropped_mask=dropped_mask,
        )

    def close(self) -> None:
        if self._ring is None:
            return
        self._ring.header[H_STATE] = STATE_CLOSED
        try:
            self._runtime.host_unregister(self._ring.host_address)
        finally:
            self._ring.close(unlink=True)
            self._ring = None


def _toolchain_tag(compiler: str, version_args: list[str]) -> str:
    resolved = shutil.which(compiler) or compiler
    try:
        version = subprocess.run(
            [resolved, *version_args], capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:
        version = "unknown"
    return f"{resolved}\n{version}"


def _build_native_library(
    name: str,
    source: Path,
    command: list[str],
    version_args: list[str],
    cache_dir_env: str,
) -> Path:
    """Compile ``source`` once and cache it by content and toolchain.

    The supported build path for ``speculative/csrc`` is runtime compilation
    with the module's pinned toolchain (the dt_ring_worker.cpp pattern):
    there is no cmake path for this directory.  The cache identity covers the
    source bytes, the resolved compiler and its version, and every flag.
    """

    cache = Path(os.environ.get(cache_dir_env, Path.home() / ".cache" / name))
    cache.mkdir(parents=True, exist_ok=True)
    resolved = shutil.which(command[0]) or command[0]
    identity = hashlib.sha256()
    identity.update(source.read_bytes())
    identity.update(b"\0")
    identity.update(_toolchain_tag(resolved, version_args).encode())
    identity.update(b"\0")
    identity.update("\0".join(command[1:]).encode())
    library = cache / f"{name}-{identity.hexdigest()[:24]}.so"
    if library.exists():
        return library
    with open(cache / ".build.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.exists():
            partial = library.with_suffix(f".so.{os.getpid()}")
            subprocess.run(
                [resolved, *command[1:], "-o", str(partial), str(source)],
                check=True,
            )
            os.replace(partial, library)
    return library


def _build_worker_library() -> Path:
    source = Path(__file__).with_name("csrc") / "dt_ring_worker.cpp"
    return _build_native_library(
        "dt_ring_worker",
        source,
        [
            os.environ.get("CXX", "g++"),
            "-O2",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-ldl",
            "-lpthread",
        ],
        ["--version"],
        "DT_DIRECT_RING_BUILD_DIR",
    )


class DraftRingWorker:
    """ctypes owner of the native ring worker thread and its landing buffers."""

    def __init__(self, *, device: int, cpu: int, landing, row_bytes: int) -> None:
        lib = ctypes.CDLL(str(_build_worker_library()))
        u64, u32, i32, vp = ctypes.c_uint64, ctypes.c_uint32, ctypes.c_int, ctypes.c_void_p
        lib.dt_ring_worker_create.argtypes = [i32, i32, u64, u64, u32, u64]
        lib.dt_ring_worker_create.restype = vp
        lib.dt_ring_worker_attach.argtypes = [
            vp, i32, u64, u64, u32, u64, u64, u64, u64, u64, u64, u64, u32, u64, u64, u64,
            u64, u64, u64, u64, vp,
        ]
        lib.dt_ring_worker_attach.restype = i32
        lib.dt_ring_worker_detach.argtypes = [vp, i32]
        lib.dt_ring_worker_detach.restype = u64
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

    def attach(
        self,
        endpoint: DraftRingEndpoint,
        *,
        source_base: int,
        source_length: int,
        ledger=None,
    ) -> None:
        header = endpoint._ring.header
        slot_nbytes = int(header[H_SOURCE_SLOT_NBYTES_LO]) | (
            int(header[H_SOURCE_SLOT_NBYTES_HI]) << 32
        )
        allocation_offset = int(header[H_SOURCE_ALLOC_OFFSET_LO]) | (
            int(header[H_SOURCE_ALLOC_OFFSET_HI]) << 32
        )
        endpoint._mail_ack[0] = endpoint.next_seq - 1
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
            endpoint.next_seq,
            slot_nbytes,
            int(header[H_SOURCE_SLOT_COUNT]),
            pool_generation_tag(endpoint._ring.read_text(POOL_GENERATION_OFFSET)),
            endpoint._ring.host_address + GENERATION_TABLE_OFFSET,
            allocation_offset,
            endpoint._ring.header.ctypes.data,
            endpoint._mailbox.ctypes.data,
            MAIL_STRIDE,
            endpoint._mail_ack.ctypes.data,
            ledger,
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
