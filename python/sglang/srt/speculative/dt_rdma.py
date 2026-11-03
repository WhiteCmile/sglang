"""CPU-proxy RDMA backend for cross-host Draft-Target pairs (issue #165).

Same-host pairs keep the CUDA-IPC paths.  A Target whose arena registration
names a ``host_addr`` other than the Draft's own serves the Draft over RC
queue pairs, following the #163 measurements:

- **Ready record** (Target -> Draft, ``WRITE_WITH_IMM`` + receive CQ): when a
  feature publication's producer event completes, the Target's publisher
  thread writes one :data:`READY_RECORD_BYTES` record naming ``(slot,
  slot_generation)`` into every connected Draft's ready ring.  The record
  also carries a per-connection sequence number and the pool-generation and
  source-session tags; the Draft accepts a record only if all three match.
- **Hidden pull** (Draft, GPUDirect RDMA READ): once the ready record for the
  descriptor's exact slot generation has arrived, the Draft READs the range
  from the Target's registered feature pool into a registered landing
  buffer, fences GPU visibility (``cuFlushGPUDirectRDMAWrites``), and copies
  it into the destination on its stream.  INPUT_READY and SourceReadDone
  follow through the existing future, unchanged.
- **Candidate push** (Draft -> Target, ``WRITE_WITH_IMM`` into pinned host
  staging + receive CQ): the candidate row, its five metadata words, the
  transfer hash and the producer tag land in the staging slot of the arena
  slot; the Target's push thread validates them against the slot (session
  tag, generation, nonce), copies them into the GPU arena, synchronizes that
  copy (the GPU visibility barrier), and only then publishes the slot's
  64-byte completion record, so the existing observer/authenticate path makes
  the candidate READY.  #163 measured that a direct RDMA WRITE into Target
  VRAM is not reliably observable, so the payload never targets GPU memory.
- **Task record** (Target -> Draft, ``WRITE_WITH_IMM`` into a pre-posted
  pinned ring + receive CQ, issue #182): cross-host owner pairs get the same
  4096-byte task-record ring same-host pairs have in ``/dev/shm``.  The Target
  stages each record image in pinned memory and writes it into the Draft's
  ring; the Draft's main loop consumes completions in seq order and applies
  the local ring's validation (sequence, epoch, per-slot generation against a
  generation table the Target mirrors onto the ring, owner/session identity).
  The hidden rows are not copied eagerly: the round's RDMA hidden pull (#165)
  runs lazily at execution, so a ring record's source lease retires on the
  Coordinator's SourceReadDone exactly like a relayed task's.

Queue pairs are bootstrapped over the Target's framed-TCP server
(``dt_host_tcp``).  On a QP error, or when the Coordinator reports a Draft
session lost (heartbeat timeout), the Target destroys that session's queue
pairs before any of its leases are released: a destroyed QP is the proof
that the NIC no longer touches the feature pool or the staging region on its
behalf.  Until that proof exists the affected slots stay quarantined.
"""

from __future__ import annotations

import collections
import ctypes
import itertools
import logging
import mmap
import os
import queue
import secrets
import struct
import threading
import time
from pathlib import Path

import numpy as np
import torch

from sglang.srt.speculative.dt_candidate_p2p import METADATA_WORDS
from sglang.srt.speculative.dt_direct_ring import (
    ENTRY_BYTES,
    ENTRY_HEADER_BYTES,
    E_EPOCH,
    E_FLAGS,
    E_NBYTES,
    E_PAYLOAD_LEN,
    E_SEQ,
    E_SLOT_GENERATION,
    E_SOURCE_OFFSET_HI,
    E_SOURCE_OFFSET_LO,
    E_SOURCE_SLOT,
    EVENT_HANDLE_BYTES,
    EVENT_HANDLE_OFFSET,
    FLAG_HAS_PULL,
    MAX_PAYLOAD_BYTES,
    PAYLOAD_OFFSET,
    RING_CAPACITY,
    SEQ_MASK,
    STATE_CLOSED,
    STATE_SERVING,
    V2_VERSION,
    RingEntry,
    _build_native_library,
    next_seq,
    pool_generation_tag,
)
from sglang.srt.speculative.dt_host_tcp import (
    MSG_RDMA_CONNECT,
    MSG_RDMA_READY,
    HostTcpConnection,
)

logger = logging.getLogger(__name__)

LEG_PULL = "pull"
LEG_PUSH = "push"
LEG_RING = "ring"

# Ready record: seq, pool-generation tag, source-session tag, slot id, slot
# generation.  One record per published feature slot per connected Draft.
_READY_RECORD = struct.Struct("<QQQII")
READY_RECORD_BYTES = _READY_RECORD.size
# Ready records a Draft can hold unconsumed; each is one posted receive, so
# the Target's sends past this depth wait on RNR retry (backpressure).
READY_RING_DEPTH = 1024
# Candidate record header: transfer hash, producer tag, then the int32
# metadata words exactly as the arena stores them.  The int32 row follows.
_CANDIDATE_PREFIX = struct.Struct("<QI")
CANDIDATE_HEADER_BYTES = _CANDIDATE_PREFIX.size + METADATA_WORDS * 4
# Inline-data cap requested for every send queue; records must fit inline so
# their send buffers need no registration and are reusable at post time.
MAX_INLINE_BYTES = 256
# Outstanding signaled sends per Target-side ready-record queue pair.
READY_SEND_DEPTH = 1024

# Task-record ring (issue #182): one page holds the per-slot generation
# table the Target mirrors remotely (entry image offsets below share the
# local ring's SSOT in dt_direct_ring), followed by RING_CAPACITY entries of
# ENTRY_BYTES.  Posted receives are entry credits: the Target's writes past
# this depth wait on RNR retry, exactly like the ready-record ring.
REMOTE_RING_TABLE_BYTES = mmap.PAGESIZE
REMOTE_RING_ENTRY_BYTES = ENTRY_BYTES
# Header u32 word offsets of the record image (mirrors EntryHeader in the
# native worker; the E_* indices are dt_direct_ring's).
_REMOTE_ENTRY_HEADER = struct.Struct("<9I")
# How long an idle proxy thread waits before re-checking for shutdown or
# housekeeping; arrivals wake it at once, so this only bounds GIL wakeups.
_IDLE_WAIT_S = 0.05
# Wait budget of one operation against a healthy peer: a bootstrap request,
# a candidate write, or a hidden pull (its ready record and READ together).
RDMA_OP_TIMEOUT_S = 30.0
# Hidden-pull READs a Draft posts before polling completions; it is the pull
# queue pair's send-queue depth, so a read_batch never overruns the QP.
PULL_READ_WINDOW = 4

_IBV_ACCESS_LOCAL_WRITE = 1
_IBV_ACCESS_REMOTE_WRITE = 2
_IBV_ACCESS_REMOTE_READ = 4
_IBV_WC_SUCCESS = 0
_IBV_WC_RECV_RDMA_WITH_IMM = 1 << 7 | 1  # IBV_WC_RECV_RDMA_WITH_IMM
_POLL_BATCH = 64
_CU_FLUSH_TARGET_CURRENT_CTX = 0
_CU_FLUSH_TO_OWNER = 100


class RdmaTransportError(RuntimeError):
    """A cross-host RDMA leg failed or saw a record it must not accept."""


class StaleDescriptorError(RdmaTransportError):
    """One pull named a stale source; the leg itself stays healthy."""


def candidate_record_stride(row_width: int) -> int:
    """Bytes of one candidate staging slot, 8-byte aligned."""

    return -(-(CANDIDATE_HEADER_BYTES + row_width * 4) // 8) * 8


def encode_ready_record(
    *, seq: int, pool_tag: int, session_tag: int, slot_id: int, slot_generation: int
) -> bytes:
    return _READY_RECORD.pack(seq, pool_tag, session_tag, slot_id, slot_generation)


def decode_ready_record(
    record: bytes,
    *,
    expected_seq: int,
    pool_tag: int,
    session_tag: int,
    slot_count: int,
) -> tuple[int, int]:
    """Return ``(slot_id, slot_generation)`` or raise for a stale record."""

    seq, record_pool, record_session, slot_id, generation = _READY_RECORD.unpack(
        record
    )
    if record_session != session_tag:
        raise RdmaTransportError("ready record names a stale source session")
    if record_pool != pool_tag:
        raise RdmaTransportError("ready record names a stale pool generation")
    if seq != expected_seq:
        raise RdmaTransportError(
            f"ready record sequence {seq} is not the expected {expected_seq}"
        )
    if slot_id >= slot_count or generation == 0:
        raise RdmaTransportError("ready record names no valid source slot")
    return slot_id, generation


def encode_candidate_record(
    *, transfer_hash: int, producer_tag: int, metadata: bytes, row: bytes
) -> bytes:
    if len(metadata) != METADATA_WORDS * 4:
        raise RdmaTransportError("candidate metadata has an invalid length")
    return _CANDIDATE_PREFIX.pack(transfer_hash, producer_tag) + metadata + row


def decode_candidate_record(
    record: bytes, row_width: int
) -> tuple[int, int, bytes, bytes]:
    """Return ``(transfer_hash, producer_tag, metadata_bytes, row_bytes)``."""

    transfer_hash, tag = _CANDIDATE_PREFIX.unpack_from(record, 0)
    metadata = record[_CANDIDATE_PREFIX.size : CANDIDATE_HEADER_BYTES]
    row = record[CANDIDATE_HEADER_BYTES : CANDIDATE_HEADER_BYTES + row_width * 4]
    return transfer_hash, tag, metadata, row


# Task-record ring (issue #182).  The wire image is byte-identical to the
# local ring's v1 entry (dt_direct_ring.TargetRingWriter.write): a 64-byte
# header with the wire seq released last, the producer-event handle, and the
# msgpack DraftTask payload.  A single RDMA WRITE delivers the whole entry,
# so the release ordering is moot on the wire; keeping one image format lets
# both sides share the header SSOT and the validation below mirror the
# native worker's (csrc/dt_ring_worker.cpp serve_entry_v1), plus the
# per-slot generation fence the worker applies to kernel-published records.

def encode_task_record(
    *,
    seq: int,
    payload: bytes,
    has_pull: bool,
    source_offset: int = 0,
    nbytes: int = 0,
    source_slot_id: int = 0,
    slot_generation: int = 0,
    event_handle: bytes = b"",
) -> bytes:
    """Build one 4096-byte task-record image for the remote ring."""

    if len(payload) > MAX_PAYLOAD_BYTES:
        raise RdmaTransportError("task record payload exceeds the entry size")
    if has_pull and len(event_handle) != EVENT_HANDLE_BYTES:
        raise RdmaTransportError("task record with a pull needs a producer event handle")
    image = bytearray(ENTRY_BYTES)
    image[PAYLOAD_OFFSET : PAYLOAD_OFFSET + len(payload)] = payload
    if has_pull:
        image[EVENT_HANDLE_OFFSET : EVENT_HANDLE_OFFSET + EVENT_HANDLE_BYTES] = (
            event_handle
        )
    words = memoryview(image)[:ENTRY_HEADER_BYTES]
    struct.pack_into("<I", words, 4 * E_FLAGS, FLAG_HAS_PULL if has_pull else 0)
    struct.pack_into("<I", words, 4 * E_SOURCE_OFFSET_LO, source_offset & SEQ_MASK)
    struct.pack_into("<I", words, 4 * E_SOURCE_OFFSET_HI, source_offset >> 32)
    struct.pack_into("<I", words, 4 * E_NBYTES, nbytes)
    struct.pack_into("<I", words, 4 * E_SOURCE_SLOT, source_slot_id)
    struct.pack_into("<I", words, 4 * E_SLOT_GENERATION, slot_generation)
    struct.pack_into("<I", words, 4 * E_PAYLOAD_LEN, len(payload))
    struct.pack_into("<I", words, 4 * E_EPOCH, (seq >> 32) & SEQ_MASK)
    # Released last, mirroring the local writer.
    struct.pack_into("<I", words, 4 * E_SEQ, seq & SEQ_MASK)
    return bytes(image)


def decode_task_record(
    record: bytes, *, expected_seq: int, generation_table
) -> RingEntry:
    """Validate one arrived task-record image; raise on a stale record.

    Checks the same fences as the local ring worker: the wire seq and its
    epoch lap, the payload geometry, and — for records with a hidden pull —
    the named slot's generation against the table the Target mirrors onto
    the ring (so a record naming a recycled source slot is rejected before
    any pull, exactly like a kernel-published record on the local ring).
    """

    if len(record) != ENTRY_BYTES:
        raise RdmaTransportError("task record has an invalid size")
    (
        wire_seq,
        flags,
        offset_lo,
        offset_hi,
        nbytes,
        source_slot_id,
        slot_generation,
        payload_len,
        epoch,
    ) = _REMOTE_ENTRY_HEADER.unpack_from(record, 0)
    if wire_seq != expected_seq & SEQ_MASK or epoch != (expected_seq >> 32) & SEQ_MASK:
        raise RdmaTransportError(
            f"task record sequence is stale: wire {epoch << 32 | wire_seq}, "
            f"expected {expected_seq}"
        )
    if offset_hi == V2_VERSION:
        raise RdmaTransportError(
            "kernel-published records cannot arrive on a remote task ring"
        )
    if flags & ~FLAG_HAS_PULL:
        raise RdmaTransportError("task record has unknown flags")
    if payload_len > MAX_PAYLOAD_BYTES:
        raise RdmaTransportError("task record payload length is invalid")
    has_pull = bool(flags & FLAG_HAS_PULL)
    if has_pull:
        if source_slot_id >= len(generation_table):
            raise RdmaTransportError("task record names no valid source slot")
        if slot_generation == 0 or generation_table[source_slot_id] != slot_generation:
            raise RdmaTransportError("task record names a stale source generation")
        if nbytes == 0:
            raise RdmaTransportError("task record pull range is invalid")
    elif nbytes or source_slot_id or slot_generation or offset_lo or offset_hi:
        raise RdmaTransportError("task record without a pull carries source fields")
    return RingEntry(
        seq=expected_seq,
        has_pull=has_pull,
        source_offset=offset_lo | (offset_hi << 32),
        nbytes=nbytes,
        source_slot_id=source_slot_id,
        slot_generation=slot_generation,
        payload=record[PAYLOAD_OFFSET : PAYLOAD_OFFSET + payload_len],
        landing_index=-1,
        landing_offset=0,
    )


def _verbs_library() -> ctypes.CDLL:
    source = Path(__file__).with_name("csrc") / "dt_rdma_verbs.cpp"
    path = _build_native_library(
        "dt_rdma_verbs",
        source,
        [
            os.environ.get("CXX", "g++"),
            "-O2",
            "-std=c++17",
            "-shared",
            "-fPIC",
            # The shared build helper places flags before the source, where
            # an as-needed linker would drop the library.
            "-Wl,--no-as-needed",
            "-libverbs",
        ],
        ["--version"],
        "DT_RDMA_BUILD_DIR",
    )
    lib = ctypes.CDLL(str(path))
    vp, i32, u32, u64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32, ctypes.c_uint64
    pu32, pu64 = ctypes.POINTER(u32), ctypes.POINTER(u64)
    signatures = {
        "dt_rdma_open": ([ctypes.c_char_p, i32, ctypes.c_char_p, i32], vp),
        "dt_rdma_lid": ([vp], i32),
        "dt_rdma_close": ([vp], None),
        "dt_rdma_reg_mr": ([vp, u64, u64, i32, pu32, pu32], vp),
        "dt_rdma_dereg_mr": ([vp], i32),
        "dt_rdma_create_cq": ([vp, i32], vp),
        "dt_rdma_destroy_cq": ([vp], i32),
        "dt_rdma_create_qp": ([vp, vp, vp, i32, i32, i32, pu32, pu32], vp),
        "dt_rdma_connect_qp": ([vp, i32, u32, u32, u32, u32], i32),
        "dt_rdma_destroy_qp": ([vp], i32),
        "dt_rdma_post_recv": ([vp, u64, i32], i32),
        "dt_rdma_post_write": ([vp, u64, u64, u32, u32, u64, u32, i32, u32, i32], i32),
        "dt_rdma_post_write_imm_inline": ([vp, u64, ctypes.c_char_p, u32, u64, u32, u32], i32),
        "dt_rdma_post_read": ([vp, u64, u64, u32, u32, u64, u32], i32),
        "dt_rdma_poll": ([vp, i32, u64, pu64, pu32, pu32, pu32, pu32], i32),
    }
    for name, (argtypes, restype) in signatures.items():
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = restype
    return lib


_LIB: ctypes.CDLL | None = None
_LIB_LOCK = threading.Lock()


def _lib() -> ctypes.CDLL:
    global _LIB
    with _LIB_LOCK:
        if _LIB is None:
            _LIB = _verbs_library()
        return _LIB


def _cuda_driver() -> ctypes.CDLL:
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuFlushGPUDirectRDMAWrites.argtypes = [ctypes.c_int, ctypes.c_int]
    driver.cuFlushGPUDirectRDMAWrites.restype = ctypes.c_int
    driver.cuDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    driver.cuDeviceGetPCIBusId.restype = ctypes.c_int
    return driver


def make_gpu_visible(driver: ctypes.CDLL) -> None:
    """Block until completed GPUDirect RDMA writes into this thread's current
    CUDA context are visible to it (a READ's CQE alone does not prove it)."""

    rc = driver.cuFlushGPUDirectRDMAWrites(
        _CU_FLUSH_TARGET_CURRENT_CTX, _CU_FLUSH_TO_OWNER
    )
    if rc != 0:
        raise RdmaTransportError(f"cuFlushGPUDirectRDMAWrites failed: {rc}")


def _index(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else int(device.index)


def closest_ib_device(device_index: int) -> str:
    """The active InfiniBand HCA sharing the longest PCI path with the GPU."""

    driver = _cuda_driver()
    bus = ctypes.create_string_buffer(32)
    if driver.cuDeviceGetPCIBusId(bus, len(bus), device_index) != 0:
        raise RdmaTransportError("cuDeviceGetPCIBusId failed")
    gpu_path = Path(
        os.path.realpath(f"/sys/bus/pci/devices/{bus.value.decode().lower()}")
    ).parts
    best: tuple[int, str] | None = None
    root = Path("/sys/class/infiniband")
    for hca in sorted(root.iterdir()) if root.exists() else ():
        port = hca / "ports" / "1"
        try:
            state = (port / "state").read_text()
            link = (port / "link_layer").read_text().strip()
        except OSError:
            continue
        if "ACTIVE" not in state or link != "InfiniBand":
            continue
        hca_path = Path(os.path.realpath(hca / "device")).parts
        shared = 0
        for left, right in zip(gpu_path, hca_path):
            if left != right:
                break
            shared += 1
        if best is None or shared > best[0]:
            best = (shared, hca.name)
    if best is None:
        raise RdmaTransportError("no active InfiniBand HCA is available")
    return best[1]


class RdmaDevice:
    """One opened HCA port with its protection domain."""

    PORT = 1

    def __init__(self, name: str) -> None:
        self.lib = _lib()
        err = ctypes.create_string_buffer(256)
        self.handle = self.lib.dt_rdma_open(name.encode(), self.PORT, err, len(err))
        if not self.handle:
            raise RdmaTransportError(f"cannot open RDMA device {name}: {err.value.decode()}")
        self.name = name
        self.lid = int(self.lib.dt_rdma_lid(self.handle))

    def register(self, address: int, nbytes: int, *, remote: int) -> "MemoryRegion":
        return MemoryRegion(self, address, nbytes, _IBV_ACCESS_LOCAL_WRITE | remote)

    def completion_queue(self, depth: int) -> "CompletionQueue":
        return CompletionQueue(self, depth)

    def queue_pair(
        self, send_cq: "CompletionQueue", recv_cq: "CompletionQueue", *, max_send: int, max_recv: int
    ) -> "QueuePair":
        return QueuePair(self, send_cq, recv_cq, max_send=max_send, max_recv=max_recv)

    def close(self) -> None:
        if self.handle:
            self.lib.dt_rdma_close(self.handle)
            self.handle = None


class MemoryRegion:
    def __init__(self, device: RdmaDevice, address: int, nbytes: int, access: int) -> None:
        lkey, rkey = ctypes.c_uint32(), ctypes.c_uint32()
        self._lib = device.lib
        self.handle = self._lib.dt_rdma_reg_mr(
            device.handle, address, nbytes, access, ctypes.byref(lkey), ctypes.byref(rkey)
        )
        if not self.handle:
            raise RdmaTransportError(
                f"ibv_reg_mr failed for {nbytes} bytes at {address:#x} "
                f"(errno {ctypes.get_errno()})"
            )
        self.address = address
        self.nbytes = nbytes
        self.lkey = lkey.value
        self.rkey = rkey.value

    def close(self) -> None:
        if self.handle:
            self._lib.dt_rdma_dereg_mr(self.handle)
            self.handle = None


class CompletionQueue:
    def __init__(self, device: RdmaDevice, depth: int) -> None:
        self._lib = device.lib
        self.handle = self._lib.dt_rdma_create_cq(device.handle, depth)
        if not self.handle:
            raise RdmaTransportError("ibv_create_cq failed")
        u32_array = ctypes.c_uint32 * _POLL_BATCH
        self._wr_id = (ctypes.c_uint64 * _POLL_BATCH)()
        self._status = u32_array()
        self._opcode = u32_array()
        self._imm = u32_array()
        self._qpn = u32_array()

    def poll(self, timeout_ns: int = 0) -> list[tuple[int, int, int, int, int]]:
        """Completions as ``(wr_id, status, opcode, imm, qp_num)``; ``wr_id``
        is the posting queue pair's :attr:`QueuePair.tag`.

        Not thread-safe: each CQ has exactly one polling thread."""

        n = self._lib.dt_rdma_poll(
            self.handle, _POLL_BATCH, timeout_ns, self._wr_id, self._status,
            self._opcode, self._imm, self._qpn,
        )
        if n < 0:
            raise RdmaTransportError("ibv_poll_cq failed")
        return [
            (self._wr_id[i], self._status[i], self._opcode[i], self._imm[i], self._qpn[i])
            for i in range(n)
        ]

    def close(self) -> None:
        if self.handle:
            self._lib.dt_rdma_destroy_cq(self.handle)
            self.handle = None


# Work-request ids: every queue pair gets a process-unique tag and posts all
# its work requests with it, so a completion names its queue pair even after
# that QP was destroyed and the provider reused its QP number.
_QP_TAGS = itertools.count(1)


class QueuePair:
    """One RC queue pair, created in INIT so receives can be posted early."""

    def __init__(
        self,
        device: RdmaDevice,
        send_cq: CompletionQueue,
        recv_cq: CompletionQueue,
        *,
        max_send: int,
        max_recv: int,
    ) -> None:
        self._lib = device.lib
        self._port = device.PORT
        qpn, max_inline = ctypes.c_uint32(), ctypes.c_uint32()
        self.handle = self._lib.dt_rdma_create_qp(
            device.handle, send_cq.handle, recv_cq.handle, max_send, max_recv,
            MAX_INLINE_BYTES, ctypes.byref(qpn), ctypes.byref(max_inline),
        )
        if not self.handle:
            raise RdmaTransportError("ibv_create_qp failed")
        self.qpn = qpn.value
        self.tag = next(_QP_TAGS)
        self.max_inline = max_inline.value
        self.psn = secrets.randbits(24)

    def endpoint(self, lid: int) -> dict:
        return {"lid": lid, "qpn": self.qpn, "psn": self.psn}

    def connect(self, remote: dict) -> None:
        rc = self._lib.dt_rdma_connect_qp(
            self._live(), self._port, int(remote["qpn"]), int(remote["lid"]),
            self.psn, int(remote["psn"]),
        )
        if rc != 0:
            raise RdmaTransportError(f"queue pair connect failed ({rc})")

    def _live(self):
        if not self.handle:
            raise RdmaTransportError("queue pair is destroyed")
        return self.handle

    def post_recv(self, count: int = 1) -> None:
        if self._lib.dt_rdma_post_recv(self._live(), self.tag, count) != 0:
            raise RdmaTransportError("ibv_post_recv failed")

    def post_write_imm(self, data: bytes, remote_addr: int, rkey: int, imm: int) -> None:
        if len(data) > self.max_inline:
            raise RdmaTransportError("record exceeds the queue pair's inline capacity")
        rc = self._lib.dt_rdma_post_write_imm_inline(
            self._live(), self.tag, data, len(data), remote_addr, rkey, imm
        )
        if rc != 0:
            raise RdmaTransportError(f"WRITE_WITH_IMM post failed ({rc})")

    def post_read(
        self, local: MemoryRegion, local_addr: int, nbytes: int, remote_addr: int, rkey: int
    ) -> None:
        rc = self._lib.dt_rdma_post_read(
            self._live(), self.tag, local_addr, nbytes, local.lkey, remote_addr, rkey
        )
        if rc != 0:
            raise RdmaTransportError(f"RDMA READ post failed ({rc})")

    def post_write(
        self,
        local: MemoryRegion,
        local_addr: int,
        nbytes: int,
        remote_addr: int,
        rkey: int,
        *,
        with_imm: bool,
        imm: int = 0,
        signaled: bool,
    ) -> None:
        """RDMA WRITE from registered memory; with_imm consumes one of the
        peer's posted receives and delivers ``imm`` in its completion."""

        rc = self._lib.dt_rdma_post_write(
            self._live(),
            self.tag,
            local_addr,
            local.lkey,
            nbytes,
            remote_addr,
            rkey,
            1 if with_imm else 0,
            imm,
            1 if signaled else 0,
        )
        if rc != 0:
            raise RdmaTransportError(f"RDMA WRITE post failed ({rc})")

    def close(self) -> None:
        """Destroy the QP: afterwards the HCA performs no access for it."""

        if self.handle:
            if self._lib.dt_rdma_destroy_qp(self.handle) != 0:
                raise RdmaTransportError("ibv_destroy_qp failed")
            self.handle = None


def _wait_one(
    cq: CompletionQueue, what: str, deadline: float
) -> tuple[int, int, int, int, int]:
    while True:
        completions = cq.poll(1_000_000)
        if completions:
            completion = completions[0]
            if completion[1] != _IBV_WC_SUCCESS:
                raise RdmaTransportError(f"{what} completed with status {completion[1]}")
            return completion
        if time.monotonic() >= deadline:
            raise RdmaTransportError(f"{what} did not complete")


class _TargetSession:
    """One Draft session's queue pairs on the Target."""

    def __init__(self, draft_instance_id: str, draft_session_id: str) -> None:
        self.draft_instance_id = draft_instance_id
        self.draft_session_id = draft_session_id
        self.pull_qp: QueuePair | None = None
        self.push_qp: QueuePair | None = None
        # Task-record ring leg (issue #182): the Target writes 4096-byte
        # records into the Draft's pinned ring through this queue pair.
        self.ring_qp: QueuePair | None = None
        self.ring_addr = 0
        self.ring_rkey = 0
        self.ring_capacity = 0
        self.ring_writer: TargetRemoteRingWriter | None = None
        # Pinned staging of the per-slot generation snapshot written at ring
        # connect; the ring writer reuses it for per-change generation posts.
        self.table_staging = None
        self.table_staging_np: np.ndarray | None = None
        self.table_mr: MemoryRegion | None = None
        self.ready_addr = 0
        self.ready_rkey = 0
        self.ready_seq = 0
        self.active = False
        # Set once fence() destroyed the session's queue pairs; the session
        # id can never connect again (a new Draft incarnation has a new one).
        self.fenced = False

    def queue_pairs(self) -> tuple[QueuePair, ...]:
        return tuple(
            qp for qp in (self.pull_qp, self.push_qp, self.ring_qp) if qp is not None
        )


class TargetRdmaEndpoint:
    """Target side of the RDMA backend: registers the feature pool (READ
    source) and a pinned candidate staging region (WRITE_WITH_IMM sink),
    accepts Draft connections through the bootstrap server, publishes ready
    records, and lands candidate pushes into the arena."""

    def __init__(self, *, pool, arena, device_index: int) -> None:
        self._pool = pool
        self._arena = arena
        self._device_index = device_index
        self.device = RdmaDevice(closest_ib_device(device_index))
        self._lock = threading.Lock()
        self._sessions: dict[tuple[str, str], _TargetSession] = {}
        # Live queue pairs by QueuePair.tag (= every completion's wr_id).
        self._by_tag: dict[int, _TargetSession] = {}
        self._closed = False
        self._pool_mr: MemoryRegion | None = None
        self._ready_cq: CompletionQueue | None = None
        self._ready_generation: np.ndarray | None = None
        self._publications: queue.Queue = queue.Queue()
        # Signaled ready-record sends not yet reaped, per pull QP tag; a fenced
        # QP's entry goes with it, so its late completions count for nothing.
        self._outstanding: dict[int, int] = {}
        self._staging: torch.Tensor | None = None
        self._staging_mr: MemoryRegion | None = None
        self._push_cq: CompletionQueue | None = None
        # Task-record ring leg (issue #182): signaled record writes reap here.
        # Created for every endpoint: token-only Targets (no feature pool)
        # still serve pull-free ring records to remote Drafts.
        self._ring_send_cq: CompletionQueue | None = None
        self.rejected_pushes = 0
        self._threads: list[threading.Thread] = []
        if pool is not None or arena is not None:
            self._ring_send_cq = self.device.completion_queue(2 * RING_CAPACITY)
        if pool is not None:
            storage = pool._storage
            self._pool_mr = self.device.register(
                storage.data_ptr(), storage.nbytes, remote=_IBV_ACCESS_REMOTE_READ
            )
            self._pool_tag = pool_generation_tag(pool.pool_generation)
            self._session_tag = pool_generation_tag(pool.source_session_id)
            self._ready_generation = np.zeros(pool.slot_count, dtype=np.uint32)
            self._ready_cq = self.device.completion_queue(4 * READY_SEND_DEPTH)
            pool.set_ready_listener(self._publications.put)
            self._start(self._publisher_loop, "dt-rdma-ready-publisher")
        if arena is not None:
            self._stride = candidate_record_stride(arena.row_width)
            if self._stride > MAX_INLINE_BYTES:
                raise RdmaTransportError("candidate record exceeds the inline cap")
            self._staging = torch.zeros(
                arena.slot_count * self._stride, dtype=torch.uint8, pin_memory=True
            )
            self._staging_np = self._staging.numpy()
            self._staging_mr = self.device.register(
                self._staging.data_ptr(), self._staging.nbytes,
                remote=_IBV_ACCESS_REMOTE_WRITE,
            )
            self._push_cq = self.device.completion_queue(4 * arena.slot_count)
            self._start(self._push_loop, "dt-rdma-push-lander")

    def _start(self, target, name: str) -> None:
        thread = threading.Thread(target=target, name=name, daemon=True)
        self._threads.append(thread)
        thread.start()

    # -- bootstrap handlers (HostTcpServer threads) -------------------------

    def handle_connect(self, header: dict, _payload: bytes) -> tuple[dict, bytes]:
        leg = header["leg"]
        key = (str(header["draft_instance_id"]), str(header["draft_session_id"]))
        if (
            leg == LEG_PULL and self._pool_mr is None
            or leg == LEG_PUSH and self._staging_mr is None
        ):
            raise RdmaTransportError(f"this Target serves no {leg} leg")
        if leg == LEG_PULL:
            qp = self.device.queue_pair(
                self._ready_cq, self._ready_cq, max_send=READY_SEND_DEPTH, max_recv=1
            )
        elif leg == LEG_PUSH:
            qp = self.device.queue_pair(
                self._push_cq, self._push_cq, max_send=1,
                max_recv=self._arena.slot_count,
            )
            qp.post_recv(self._arena.slot_count)
        else:
            if leg != LEG_RING or self._ring_send_cq is None:
                raise RdmaTransportError(f"unknown RDMA leg {leg!r}")
            if int(header["ring_capacity"]) != RING_CAPACITY:
                raise RdmaTransportError("task-record ring capacity mismatch")
            if int(header["entry_bytes"]) != ENTRY_BYTES:
                raise RdmaTransportError("task-record ring entry size mismatch")
            qp = self.device.queue_pair(
                self._ring_send_cq, self._ring_send_cq,
                max_send=2 * RING_CAPACITY, max_recv=1,
            )
        try:
            qp.connect(header)
        except Exception:
            qp.close()
            raise
        with self._lock:
            if self._closed:
                qp.close()
                raise RdmaTransportError("RDMA endpoint is closed")
            session = self._sessions.setdefault(key, _TargetSession(*key))
            if session.fenced:
                qp.close()
                raise RdmaTransportError("draft session was fenced")
            if leg == LEG_PULL:
                previous = session.pull_qp
            elif leg == LEG_PUSH:
                previous = session.push_qp
            else:
                previous = session.ring_qp
            if previous is not None:
                qp.close()
                raise RdmaTransportError(f"draft session already holds a {leg} leg")
            if leg == LEG_PULL:
                session.pull_qp = qp
                session.ready_addr = int(header["ready_addr"])
                session.ready_rkey = int(header["ready_rkey"])
            elif leg == LEG_PUSH:
                session.push_qp = qp
            else:
                session.ring_qp = qp
                session.ring_addr = int(header["ring_addr"])
                session.ring_rkey = int(header["ring_rkey"])
                session.ring_capacity = int(header["ring_capacity"])
            self._by_tag[qp.tag] = session
        response = qp.endpoint(self.device.lid)
        if leg == LEG_PULL:
            pool = self._pool
            response.update(
                pool_addr=self._pool_mr.address,
                pool_rkey=self._pool_mr.rkey,
                slot_nbytes=pool.slot_nbytes,
                slot_count=pool.slot_count,
                pool_generation=pool.pool_generation,
                source_session_id=pool.source_session_id,
            )
        elif leg == LEG_PUSH:
            response.update(
                staging_addr=self._staging_mr.address,
                staging_rkey=self._staging_mr.rkey,
                record_stride=self._stride,
                slot_count=self._arena.slot_count,
                row_width=self._arena.row_width,
            )
        else:
            # Mirror the per-slot generation table into the Draft's ring
            # before any record can reference it: same-QP post order
            # guarantees the snapshot lands first, so a record naming a slot
            # reserved before this connect still validates.
            slot_count = self._pool.slot_count if self._pool is not None else 0
            if slot_count:
                snapshot = self._pool.generation_snapshot()
                staging = torch.zeros(slot_count, dtype=torch.uint32, pin_memory=True)
                staging_np = staging.numpy()
                staging_np[:] = np.asarray(snapshot, dtype=np.uint32)
                mr = self.device.register(
                    staging.data_ptr(), staging.nbytes, remote=0
                )
                try:
                    qp.post_write(
                        mr, mr.address, staging.nbytes,
                        session.ring_addr, session.ring_rkey,
                        with_imm=False, signaled=False,
                    )
                except Exception:
                    mr.close()
                    raise
                session.table_staging = staging
                session.table_staging_np = staging_np
                session.table_mr = mr
            response.update(
                slot_count=slot_count,
                pool_generation=(
                    self._pool.pool_generation if self._pool is not None else ""
                ),
                source_session_id=(
                    self._pool.source_session_id if self._pool is not None else ""
                ),
            )
        return response, b""

    def handle_ready(self, header: dict, _payload: bytes) -> tuple[dict, bytes]:
        """The Draft's pull QP is in RTS: start ready records, and return the
        slots already ready, so publications before this point are not lost."""

        key = (str(header["draft_instance_id"]), str(header["draft_session_id"]))
        with self._lock:
            session = self._sessions.get(key)
            if session is None or session.pull_qp is None:
                raise RdmaTransportError("draft session has no pull leg")
            session.active = True
            snapshot = self._ready_generation.tolist()
        return {"ready_generation": snapshot}, b""

    def has_session(self, draft_session_id: str) -> bool:
        """Whether any leg of this Draft session ever connected."""

        with self._lock:
            return any(
                session.draft_session_id == draft_session_id
                for session in self._sessions.values()
            )

    def ring_writer(self, draft_session_id: str) -> "TargetRemoteRingWriter | None":
        """The task-record ring writer for one Draft session, once its ring
        leg connected; None while pending, and forever after a fence."""

        with self._lock:
            for session in self._sessions.values():
                if session.draft_session_id != draft_session_id:
                    continue
                if session.ring_qp is None or session.fenced:
                    return None
                if session.ring_writer is None:
                    slot_count = self._pool.slot_count if self._pool is not None else 0
                    session.ring_writer = TargetRemoteRingWriter(
                        self, session, slot_count=slot_count
                    )
                return session.ring_writer
        return None

    # -- ready records ------------------------------------------------------

    def _publisher_loop(self) -> None:
        torch.cuda.set_device(self._device_index)
        while not self._closed:
            # Idle with nothing in flight: block (no GIL wakeups); otherwise
            # come back soon to reap the outstanding sends.
            try:
                item = self._publications.get(
                    timeout=0.001 if any(self._outstanding.values()) else _IDLE_WAIT_S
                )
            except queue.Empty:
                item = None
            if item is not None:
                slot_id, generation, producer_event = item
                try:
                    producer_event.synchronize()
                except Exception:  # noqa: BLE001 - the slot stays unready
                    logger.exception("feature producer event failed; slot %d not ready", slot_id)
                else:
                    self._publish_ready(slot_id, generation)
            self._reap_ready_sends()

    def _publish_ready(self, slot_id: int, generation: int) -> None:
        broken = []
        with self._lock:
            self._ready_generation[slot_id] = generation
            for session in self._sessions.values():
                if not session.active or session.pull_qp is None:
                    continue
                session.ready_seq += 1
                record = encode_ready_record(
                    seq=session.ready_seq, pool_tag=self._pool_tag,
                    session_tag=self._session_tag, slot_id=slot_id,
                    slot_generation=generation,
                )
                index = session.ready_seq % READY_RING_DEPTH
                try:
                    session.pull_qp.post_write_imm(
                        record,
                        session.ready_addr + index * READY_RECORD_BYTES,
                        session.ready_rkey, index,
                    )
                except RdmaTransportError:
                    broken.append(session)
                else:
                    tag = session.pull_qp.tag
                    self._outstanding[tag] = self._outstanding.get(tag, 0) + 1
        for session in broken:
            logger.error(
                "ready record post failed for draft session %s; fencing it",
                session.draft_session_id,
            )
            self.fence(session.draft_instance_id, session.draft_session_id)

    def _reap_ready_sends(self) -> None:
        if not any(self._outstanding.values()):
            return
        for tag, status, _opcode, _imm, _qpn in self._ready_cq.poll(0):
            with self._lock:
                if tag in self._outstanding:
                    self._outstanding[tag] -= 1
                session = self._by_tag.get(tag)
            if status == _IBV_WC_SUCCESS:
                continue
            if session is not None:
                logger.error(
                    "ready record to draft session %s failed (status %d); fencing it",
                    session.draft_session_id, status,
                )
                self.fence(session.draft_instance_id, session.draft_session_id)

    # -- candidate pushes ---------------------------------------------------

    def _push_loop(self) -> None:
        torch.cuda.set_device(self._device_index)
        pending: dict[int, tuple[str, int, int, bytes, bytes]] = {}
        while not self._closed:
            # The poll returns at the first completion; its timeout only bounds
            # how often an idle thread takes the GIL (or retries a deferred push).
            timeout_ns = 1_000_000 if pending else int(_IDLE_WAIT_S * 1e9)
            for tag, status, _opcode, slot_id, _qpn in self._push_cq.poll(timeout_ns):
                with self._lock:
                    # Under the lock: fence() may destroy the QP concurrently.
                    session = self._by_tag.get(tag)
                    if session is not None and status == _IBV_WC_SUCCESS:
                        session.push_qp.post_recv()
                if session is None:
                    continue  # a fenced session's late completion
                if status != _IBV_WC_SUCCESS:
                    logger.error(
                        "candidate push from draft session %s failed (status %d); fencing it",
                        session.draft_session_id, status,
                    )
                    self.fence(session.draft_instance_id, session.draft_session_id)
                    continue
                if slot_id >= self._arena.slot_count:
                    self.rejected_pushes += 1
                    continue
                start = slot_id * self._stride
                # Copy out of staging at once: the Draft may reuse the slot
                # only after this slot's generation retires, but a stale writer
                # must never alter a record while it is validated.
                record = self._staging_np[start : start + self._stride].tobytes()
                transfer_hash, tag, metadata, row = decode_candidate_record(
                    record, self._arena.row_width
                )
                pending[slot_id] = (session.draft_session_id, transfer_hash, tag, metadata, row)
            for slot_id, (session_id, transfer_hash, tag, metadata, row) in tuple(pending.items()):
                try:
                    landed = self._arena.land_remote_push(
                        slot_id=slot_id, transfer_hash=transfer_hash, producer_tag_value=tag,
                        metadata_bytes=metadata, data_bytes=row,
                    )
                except Exception as exc:  # noqa: BLE001 - a rejected record is dropped
                    self.rejected_pushes += 1
                    logger.warning(
                        "rejected candidate push for slot %d from %s: %s", slot_id, session_id, exc
                    )
                    landed = True
                if landed:
                    pending.pop(slot_id)

    # -- failure handling ---------------------------------------------------

    def fence(self, draft_instance_id: str, draft_session_id: str = "") -> dict[str, bool]:
        """Destroy the queue pairs of a lost Draft session (all sessions of
        the instance when ``draft_session_id`` is empty).  Returns, per fenced
        session id (including ones fenced earlier), whether quiescence is
        proven: none of that session's queue pairs remains."""

        fenced: dict[str, bool] = {}
        with self._lock:
            for key, session in self._sessions.items():
                if key[0] != draft_instance_id or (
                    draft_session_id and key[1] != draft_session_id
                ):
                    continue
                session.active = False
                session.fenced = True
                for qp in session.queue_pairs():
                    tag = qp.tag
                    try:
                        qp.close()
                    except RdmaTransportError:
                        continue
                    self._by_tag.pop(tag, None)
                    self._outstanding.pop(tag, None)
                fenced[key[1]] = all(qp.handle is None for qp in session.queue_pairs())
        if fenced:
            logger.warning(
                "fenced RDMA queue pairs of draft %s sessions %s",
                draft_instance_id, fenced,
            )
        return fenced

    def close(self) -> None:
        with self._lock:
            self._closed = True
            keys = list(self._sessions)
        for thread in self._threads:
            thread.join(timeout=5.0)
        for instance_id, session_id in keys:
            self.fence(instance_id, session_id)
        if self._pool is not None:
            self._pool.set_ready_listener(None)
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            if session.ring_writer is not None:
                session.ring_writer.close()
                session.ring_writer = None
            if session.table_mr is not None:
                session.table_mr.close()
                session.table_mr = None
                session.table_staging = None
                session.table_staging_np = None
        for resource in (
            self._pool_mr,
            self._staging_mr,
            self._ready_cq,
            self._push_cq,
            self._ring_send_cq,
        ):
            if resource is not None:
                resource.close()
        self.device.close()


class _DraftConnection:
    """Shared bootstrap of one Draft-side leg to one remote Target."""

    def __init__(
        self, *, leg: str, host: str, port: int, device_index: int,
        draft_instance_id: str, draft_session_id: str,
    ) -> None:
        self.leg = leg
        self._identity = {
            "draft_instance_id": draft_instance_id,
            "draft_session_id": draft_session_id,
        }
        self.bootstrap = HostTcpConnection(host, port, RDMA_OP_TIMEOUT_S)
        self.device = RdmaDevice(closest_ib_device(device_index))
        self.send_cq = self.device.completion_queue(64)
        self.recv_cq: CompletionQueue | None = None
        self.qp: QueuePair | None = None

    def connect(self, extra: dict) -> dict:
        header = {"leg": self.leg, **self._identity, **self.qp.endpoint(self.device.lid), **extra}
        remote, _ = self.bootstrap.request(MSG_RDMA_CONNECT, header)
        self.qp.connect(remote)
        return remote

    def ready(self) -> dict:
        header, _ = self.bootstrap.request(MSG_RDMA_READY, dict(self._identity))
        return header

    def close(self) -> None:
        self.bootstrap.close()
        if self.qp is not None:
            self.qp.close()
        for cq in (self.send_cq, self.recv_cq):
            if cq is not None:
                cq.close()


class DraftRdmaPuller:
    """Hidden-pull leg of one Draft to one remote Target.

    Owned by exactly one pull worker thread: it consumes ready records,
    posts READs into a registered landing buffer, fences GPU visibility, and
    copies the range into the destination on the worker's stream."""

    def __init__(self, *, host: str, port: int, device: torch.device,
                 draft_instance_id: str, draft_session_id: str) -> None:
        self._conn = _DraftConnection(
            leg=LEG_PULL, host=host, port=port, device_index=_index(device),
            draft_instance_id=draft_instance_id, draft_session_id=draft_session_id,
        )
        self._driver = _cuda_driver()
        self._landing_mr: MemoryRegion | None = None
        self._ring_mr: MemoryRegion | None = None
        try:
            conn = self._conn
            conn.recv_cq = conn.device.completion_queue(2 * READY_RING_DEPTH)
            conn.qp = conn.device.queue_pair(
                conn.send_cq, conn.recv_cq, max_send=PULL_READ_WINDOW, max_recv=READY_RING_DEPTH
            )
            self._ring = torch.zeros(
                READY_RING_DEPTH * READY_RECORD_BYTES, dtype=torch.uint8, pin_memory=True
            )
            self._ring_np = self._ring.numpy()
            self._ring_mr = conn.device.register(
                self._ring.data_ptr(), self._ring.nbytes, remote=_IBV_ACCESS_REMOTE_WRITE
            )
            conn.qp.post_recv(READY_RING_DEPTH)
            remote = conn.connect(
                {"ready_addr": self._ring_mr.address, "ready_rkey": self._ring_mr.rkey}
            )
            self.pool_generation = str(remote["pool_generation"])
            self.source_session_id = str(remote["source_session_id"])
            self._pool_tag = pool_generation_tag(self.pool_generation)
            self._session_tag = pool_generation_tag(self.source_session_id)
            self._pool_addr = int(remote["pool_addr"])
            self._pool_rkey = int(remote["pool_rkey"])
            self._slot_nbytes = int(remote["slot_nbytes"])
            self._slot_count = int(remote["slot_count"])
            self._device = device
            self._landing: torch.Tensor | None = None
            self._landing_mr = None
            self._expected_seq = 1
            snapshot = conn.ready()["ready_generation"]
            if len(snapshot) != self._slot_count:
                raise RdmaTransportError("ready snapshot does not match the pool")
            self._ready = np.asarray(snapshot, dtype=np.uint32)
        except Exception:
            self.close()
            raise

    def drain(self, timeout_ns: int = 0) -> None:
        """Consume every arrived ready record; raise on a stale one."""

        completions = self._conn.recv_cq.poll(timeout_ns)
        for _wr_id, status, opcode, index, _qpn in completions:
            if status != _IBV_WC_SUCCESS or opcode != _IBV_WC_RECV_RDMA_WITH_IMM:
                raise RdmaTransportError(f"ready record receive failed (status {status})")
            start = (index % READY_RING_DEPTH) * READY_RECORD_BYTES
            slot_id, generation = decode_ready_record(
                self._ring_np[start : start + READY_RECORD_BYTES].tobytes(),
                expected_seq=self._expected_seq,
                pool_tag=self._pool_tag,
                session_tag=self._session_tag,
                slot_count=self._slot_count,
            )
            self._expected_seq += 1
            if generation > self._ready[slot_id]:
                self._ready[slot_id] = generation
        if completions:
            self._conn.qp.post_recv(len(completions))

    def _ensure_landing(self, nbytes: int) -> None:
        """Grow the registered landing buffer until it holds ``nbytes``.

        A batch of rows of one (draft, target) pair shares the buffer at
        disjoint offsets; growth is geometric so re-registration amortizes.
        """

        if self._landing is not None and self._landing.nbytes >= nbytes:
            return
        size = max(self._slot_nbytes, 1)
        while size < nbytes:
            size *= 2
        landing = torch.empty(size, dtype=torch.uint8, device=self._device)
        mr = self._conn.device.register(landing.data_ptr(), landing.nbytes, remote=0)
        old_mr = self._landing_mr
        self._landing = landing
        self._landing_mr = mr
        if old_mr is not None:
            old_mr.close()

    def _pull_plan(self, source) -> dict:
        """Resolve one descriptor against the ready ring; raise for a stale
        source, else return its read plan and ready timestamp."""

        if (
            source.pool_generation != self.pool_generation
            or source.source_session_id != self.source_session_id
        ):
            raise StaleDescriptorError("feature descriptor names a stale source incarnation")
        slot_id, generation = int(source.source_slot_id), int(source.slot_generation)
        if (
            slot_id >= self._slot_count
            or source.byte_offset + source.nbytes > self._slot_nbytes
        ):
            raise StaleDescriptorError("feature descriptor range exceeds the source slot")
        deadline = time.monotonic() + RDMA_OP_TIMEOUT_S
        while self._ready[slot_id] < generation:
            if time.monotonic() >= deadline:
                raise RdmaTransportError("ready record for the source slot did not arrive")
            self.drain(1_000_000)
        if self._ready[slot_id] != generation:
            raise StaleDescriptorError("feature source slot was republished before the read")
        return {
            "slot_id": slot_id,
            "generation": generation,
            "remote": self._pool_addr + slot_id * self._slot_nbytes + source.byte_offset,
            "nbytes": int(source.nbytes),
            "ready_ns": time.monotonic_ns(),
        }

    @staticmethod
    def _copy_runs(planned: list) -> list:
        """Group consecutive pulls whose destinations are adjacent views of
        one storage — the engine stages a same-Target batch into consecutive
        rows — so their landing copies merge into one kernel per run."""

        runs: list[list] = []
        for plan in planned:
            if runs:
                previous = runs[-1][-1]
                destination = plan["destination"]
                if (
                    destination.untyped_storage().data_ptr()
                    == previous["destination"].untyped_storage().data_ptr()
                    and previous["destination"].data_ptr() + previous["nbytes"]
                    == destination.data_ptr()
                ):
                    runs[-1].append(plan)
                    continue
            runs.append([plan])
        return runs

    def read(
        self, source, destination: torch.Tensor, stream: torch.cuda.Stream
    ) -> tuple[int, int, int]:
        """Pull one descriptor's range into ``destination``.

        Returns the monotonic ns at which the ready record was in hand, the
        READ was posted, and its completion was polled."""

        timestamps, error = self.read_batch([(source, destination)], stream)[0]
        if error is not None:
            raise error
        return timestamps

    def read_batch(self, items, stream: torch.cuda.Stream) -> list:
        """Pull a batch of same-Target descriptors into their destinations.

        Every READ is posted into the landing buffer back to back (at most
        :data:`PULL_READ_WINDOW` outstanding) and polled to completion; then
        ONE ``cuFlushGPUDirectRDMAWrites`` covers the whole batch and the
        landing copies are enqueued — merged into one copy per run of
        adjacent destinations.  A row becomes visible to the consumer only
        after that fence, exactly as a per-row fence guaranteed before.

        Returns, aligned with ``items``, one ``((ready_ns, posted_ns,
        read_done_ns), None)`` per pulled range, or ``(None, error)`` for a
        descriptor that proved stale; its siblings still pull."""

        planned = []
        for source, destination in items:
            try:
                plan = self._pull_plan(source)
            except StaleDescriptorError as exc:
                plan = {"error": exc}
            plan["source"] = source
            plan["destination"] = destination
            planned.append(plan)
        live = [plan for plan in planned if "error" not in plan]
        self._ensure_landing(sum(plan["nbytes"] for plan in live))
        offset = 0
        for plan in live:
            plan["landing_offset"] = offset
            offset += plan["nbytes"]
        posted = 0
        completed = 0
        deadline = time.monotonic() + RDMA_OP_TIMEOUT_S
        while completed < len(live):
            while posted < len(live) and posted - completed < PULL_READ_WINDOW:
                plan = live[posted]
                self._conn.qp.post_read(
                    self._landing_mr,
                    self._landing_mr.address + plan["landing_offset"],
                    plan["nbytes"],
                    plan["remote"],
                    self._pool_rkey,
                )
                plan["posted_ns"] = time.monotonic_ns()
                posted += 1
            # One RC queue pair completes sends in posting order, so the
            # n-th completion belongs to the n-th posted READ.
            for _wr_id, status, _opcode, _index, _qpn in self._conn.send_cq.poll(1_000_000):
                if status != _IBV_WC_SUCCESS:
                    raise RdmaTransportError(f"hidden RDMA READ completed with status {status}")
                plan = live[completed]
                plan["read_done_ns"] = time.monotonic_ns()
                completed += 1
            if time.monotonic() >= deadline:
                raise RdmaTransportError("hidden RDMA READ did not complete")
        # A republication whose ready record arrived meanwhile may have
        # overwritten the range under a READ.
        self.drain()
        for plan in live:
            if self._ready[plan["slot_id"]] != plan["generation"]:
                plan["error"] = StaleDescriptorError(
                    "feature source slot was republished during the read"
                )
        if any("error" not in plan for plan in live):
            # One fence for the batch: every READ completion was polled
            # above, so the flush orders all their landings before the
            # copies below enqueue the GPU's reads of the landing buffer.
            make_gpu_visible(self._driver)
            with torch.cuda.stream(stream):
                for run in self._copy_runs(
                    [plan for plan in live if "error" not in plan]
                ):
                    first = run[0]
                    total = sum(plan["nbytes"] for plan in run)
                    source_span = self._landing[
                        first["landing_offset"] : first["landing_offset"] + total
                    ]
                    destination = first["destination"].view(torch.uint8).reshape(-1)
                    if total != first["nbytes"]:
                        destination = destination.as_strided((total,), (1,))
                    destination.copy_(source_span, non_blocking=True)
                copied = torch.cuda.Event()
                copied.record(stream)
            copied.synchronize()
        outcomes = []
        for plan in planned:
            if "error" in plan:
                outcomes.append((None, plan["error"]))
            else:
                outcomes.append(
                    (
                        (plan["ready_ns"], plan["posted_ns"], plan["read_done_ns"]),
                        None,
                    )
                )
        return outcomes

    def close(self) -> None:
        self._conn.close()
        for mr in (self._landing_mr, self._ring_mr):
            if mr is not None:
                mr.close()
        self._conn.device.close()


class DraftRdmaPusher:
    """Candidate-push leg of one Draft to one remote Target; owned by one
    sender thread.  Each record is one inline WRITE_WITH_IMM into the
    Target's staging slot for the arena slot, immediate = slot id."""

    def __init__(self, *, host: str, port: int, device: torch.device,
                 draft_instance_id: str, draft_session_id: str) -> None:
        self._conn = _DraftConnection(
            leg=LEG_PUSH, host=host, port=port, device_index=_index(device),
            draft_instance_id=draft_instance_id, draft_session_id=draft_session_id,
        )
        try:
            conn = self._conn
            conn.qp = conn.device.queue_pair(conn.send_cq, conn.send_cq, max_send=4, max_recv=1)
            remote = conn.connect({})
            self._staging_addr = int(remote["staging_addr"])
            self._staging_rkey = int(remote["staging_rkey"])
            self._stride = int(remote["record_stride"])
            self._slot_count = int(remote["slot_count"])
            self.row_width = int(remote["row_width"])
        except Exception:
            self.close()
            raise

    def push(self, slot_id: int, record: bytes) -> None:
        if slot_id >= self._slot_count or len(record) > self._stride:
            raise RdmaTransportError("candidate record does not fit the staging slot")
        self._conn.qp.post_write_imm(
            record, self._staging_addr + slot_id * self._stride,
            self._staging_rkey, slot_id,
        )
        _wait_one(
            self._conn.send_cq,
            "candidate WRITE_WITH_IMM",
            time.monotonic() + RDMA_OP_TIMEOUT_S,
        )

    def close(self) -> None:
        self._conn.close()
        self._conn.device.close()


class TargetRemoteRingWriter:
    """Target-side task-record producer for one remote owner Draft session
    (issue #182).  Duck-type compatible with
    ``dt_direct_ring.TargetRingWriter`` for the Target's ring planning.

    Each record is staged in pinned memory and written into the Draft's
    pre-posted pinned ring with one signaled ``WRITE_WITH_IMM`` (immediate =
    ring slot).  The Draft's posted receives are the credits: writes past the
    ring depth wait on RNR retry, so an unconsumed slot is never overwritten.
    Signaled send completions are reaped in order and bound the staging
    reuse; the Draft-side consumption credit is what actually protects the
    remote slot.  Unlike the local writer there is no early read proof (the
    hidden pull runs lazily at execution over the pull leg), so the Target's
    ``_write_direct_ring`` skips the ring-outstanding bookkeeping and the
    source lease retires on the Coordinator's SourceReadDone, exactly like a
    relayed task's.
    """

    is_remote = True
    # None: no eager landing copy exists remotely, so the planner's
    # landing-capacity bound does not apply (the execution-time pull bounds
    # the range through the feature staging geometry, as for relayed tasks).
    max_entry_bytes: int | None = None

    def __init__(
        self, endpoint: TargetRdmaEndpoint, session: _TargetSession, *, slot_count: int
    ) -> None:
        self._endpoint = endpoint
        self._session = session
        self.draft_instance_id = session.draft_instance_id
        self.draft_session_id = session.draft_session_id
        self.capacity = session.ring_capacity
        self.next_seq = 1
        self.source_ipc_handle = b""
        self._slot_count = slot_count
        self._posted = 0
        self._completed = 0
        self._broken: str | None = None
        self._sent_generations = np.zeros(max(slot_count, 1), dtype=np.uint32)
        self._staging = torch.zeros(
            self.capacity * ENTRY_BYTES, dtype=torch.uint8, pin_memory=True
        )
        self._staging_np = self._staging.numpy()
        self._staging_mr = endpoint.device.register(
            self._staging.data_ptr(), self._staging.nbytes, remote=0
        )

    @property
    def state(self) -> int:
        if self._broken is not None or self._session.fenced:
            return STATE_CLOSED
        return STATE_SERVING

    def _fail(self, reason: str) -> None:
        self._broken = reason
        logger.error(
            "task-record ring to draft session %s failed: %s; fencing it",
            self.draft_session_id,
            reason,
        )
        self._endpoint.fence(self.draft_instance_id, self.draft_session_id)

    def _reap(self) -> None:
        """Reap signaled record completions in post order."""

        if self._session.ring_qp is None:
            return
        for _tag, status, _opcode, _imm, _qpn in self._endpoint._ring_send_cq.poll(0):
            if self._broken is not None:
                continue  # a fenced session's late completions count for nothing
            if status != _IBV_WC_SUCCESS:
                self._fail(f"record write completed with status {status}")
                continue
            self._completed += 1

    def free_entries(self) -> int:
        self._reap()
        return self.capacity - (self._posted - self._completed)

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
        """Publish one record; the caller checked state and free_entries()."""

        session = self._session
        if self._broken is not None or session.fenced or session.ring_qp is None:
            raise RdmaTransportError("task-record ring session is fenced")
        self._reap()
        if self._posted - self._completed >= self.capacity:
            raise RdmaTransportError("task-record ring has no free staging entries")
        if has_pull and source_slot_id >= self._slot_count:
            raise RdmaTransportError("task record names no valid source slot")
        seq = self.next_seq
        index = seq % self.capacity
        if (
            has_pull
            and self._slot_count
            and self._sent_generations[source_slot_id] != slot_generation
        ):
            # Mirror the slot's current generation before the record that
            # names it: same-QP post order guarantees the table lands first,
            # so the Draft never rejects a fresh record as stale.
            session.table_staging_np[source_slot_id] = slot_generation
            session.ring_qp.post_write(
                session.table_mr,
                session.table_mr.address + 4 * source_slot_id,
                4,
                session.ring_addr + 4 * source_slot_id,
                session.ring_rkey,
                with_imm=False,
                signaled=False,
            )
            self._sent_generations[source_slot_id] = slot_generation
        image = encode_task_record(
            seq=seq,
            payload=payload,
            has_pull=has_pull,
            source_offset=source_offset,
            nbytes=nbytes,
            source_slot_id=source_slot_id,
            slot_generation=slot_generation,
            event_handle=event_handle,
        )
        start = index * ENTRY_BYTES
        self._staging_np[start : start + ENTRY_BYTES] = np.frombuffer(
            image, dtype=np.uint8
        )
        session.ring_qp.post_write(
            self._staging_mr,
            self._staging_mr.address + start,
            ENTRY_BYTES,
            session.ring_addr + REMOTE_RING_TABLE_BYTES + start,
            session.ring_rkey,
            with_imm=True,
            imm=index,
            signaled=True,
        )
        self._posted += 1
        self.next_seq = next_seq(seq)
        return seq

    def read_proven(self, seq: int) -> bool:
        raise RdmaTransportError(
            "a remote ring has no early read proof; leases retire on SourceReadDone"
        )

    def close(self, *, unlink: bool = False) -> None:
        if self._staging_mr is not None:
            self._staging_mr.close()
            self._staging_mr = None
        self._staging = None
        self._staging_np = None


class DraftRemoteRing:
    """Draft-side consumer of one remote Target's task-record ring.

    The ring buffer is one pinned region: page 0 holds the per-slot
    generation table the Target mirrors remotely, followed by
    ``capacity`` 4096-byte entries.  Posted receives are entry credits; each
    arrived ``WRITE_WITH_IMM`` completion (immediate = slot) releases one
    entry, validated by :func:`decode_task_record` with the same fences as
    the local ring worker.  A receive failure or a stale record is
    fail-stop: the leg must never silently skip a record, and a fenced queue
    pair proves the Target's NIC no longer touches this ring.
    """

    is_remote = True

    def __init__(
        self,
        *,
        host: str,
        port: int,
        device: torch.device,
        draft_instance_id: str,
        draft_session_id: str,
        target_instance_id: str,
        target_session_id: str,
        ring_id: int,
    ) -> None:
        self.target_session_id = target_session_id
        self.target_instance_id = target_instance_id
        self.ring_id = ring_id
        self.capacity = RING_CAPACITY
        self.next_seq = 1
        self.records_taken = 0
        self.pool_generation = ""
        self.source_session_id = ""
        self._conn = _DraftConnection(
            leg=LEG_RING, host=host, port=port, device_index=_index(device),
            draft_instance_id=draft_instance_id, draft_session_id=draft_session_id,
        )
        self._ring_mr: MemoryRegion | None = None
        try:
            conn = self._conn
            conn.recv_cq = conn.device.completion_queue(2 * RING_CAPACITY)
            conn.qp = conn.device.queue_pair(
                conn.send_cq, conn.recv_cq, max_send=4, max_recv=RING_CAPACITY
            )
            buffer = torch.zeros(
                REMOTE_RING_TABLE_BYTES + RING_CAPACITY * ENTRY_BYTES,
                dtype=torch.uint8,
                pin_memory=True,
            )
            self._buffer_np = buffer.numpy()
            self._ring_mr = conn.device.register(
                buffer.data_ptr(), buffer.nbytes, remote=_IBV_ACCESS_REMOTE_WRITE
            )
            self._buffer = buffer
            conn.qp.post_recv(RING_CAPACITY)
            remote = conn.connect(
                {
                    "ring_addr": self._ring_mr.address,
                    "ring_rkey": self._ring_mr.rkey,
                    "ring_capacity": RING_CAPACITY,
                    "entry_bytes": ENTRY_BYTES,
                }
            )
            self.pool_generation = str(remote["pool_generation"])
            self.source_session_id = str(remote["source_session_id"])
            self._slot_count = int(remote["slot_count"])
            if self._slot_count * 4 > REMOTE_RING_TABLE_BYTES:
                raise RdmaTransportError("source slot count exceeds the ring table")
        except Exception:
            self.close()
            raise
        # Completions polled in batches but consumed one admission at a time.
        self._pending = collections.deque()

    def take_ready(self) -> RingEntry | None:
        """The next validated record, in seq order, or None when empty."""

        if not self._pending:
            completions = self._conn.recv_cq.poll(0)
            if completions:
                self._pending.extend(completions)
        if not self._pending:
            return None
        _tag, status, opcode, imm, _qpn = self._pending.popleft()
        if status != _IBV_WC_SUCCESS or opcode != _IBV_WC_RECV_RDMA_WITH_IMM:
            raise RdmaTransportError(
                f"task-record ring receive failed (status {status})"
            )
        slot = int(imm) % self.capacity
        start = REMOTE_RING_TABLE_BYTES + slot * ENTRY_BYTES
        entry = decode_task_record(
            self._buffer_np[start : start + ENTRY_BYTES].tobytes(),
            expected_seq=self.next_seq,
            generation_table=self._table,
        )
        self.next_seq = next_seq(self.next_seq)
        self.records_taken += 1
        self._conn.qp.post_recv()
        return entry

    @property
    def _table(self) -> np.ndarray:
        return self._buffer_np[:REMOTE_RING_TABLE_BYTES].view(np.uint32)[
            : self._slot_count
        ]

    def close(self) -> None:
        self._conn.close()
        if self._ring_mr is not None:
            self._ring_mr.close()
            self._ring_mr = None
        self._conn.device.close()
