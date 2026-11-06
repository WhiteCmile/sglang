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
- **Candidate push** (Draft -> Target, GPUDirect RDMA WRITEs, issue #223):
  symmetric with the same-host CUDA-IPC push.  The Draft copies each row
  into its registered GPU mirror of the Target arena slot on the push
  stream, and a CUDA host function stream-ordered after that copy
  (csrc/dt_rdma_verbs.cpp ``dt_rdma_post_candidates``) posts, on one RC
  queue pair, the row WRITE straight into the Target's GPU arena slot and
  then the slot's 64-byte completion record into the Target's host-mapped
  record mapping, the seq (reservation nonce) word last.  The Target runs no
  thread for this leg: its native record observer issues one
  ``cuFlushGPUDirectRDMAWrites`` per scan pass that matched a record, before
  publishing it to its READY table, and the observer's existing validation
  (nonce, generation, transfer hash, producer tag) makes the candidate READY
  exactly like a same-host record.  #223 measured 0 stale rows in that order; #163's stale
  observations came from posting before its GPU source write completed.
- **Task record** (Target -> Draft, ``WRITE_WITH_IMM`` into a pre-posted
  pinned ring + receive CQ, issue #182): cross-host owner pairs get the same
  4096-byte task-record ring same-host pairs have in ``/dev/shm``.  The Target
  stages each record image in pinned memory and writes it into the Draft's
  ring.  The Draft's native ring worker (csrc/dt_ring_worker.cpp, issue #198)
  consumes the completions in seq order with the local ring's validation
  (sequence, epoch, per-slot generation against a generation table the
  Target mirrors onto the ring, a row-aligned range inside the slot), waits
  for the slot's ready record, READs the rows into its landing buffers over
  the ring queue pair, fences once per batch of READs and publishes
  INPUT_READY, so the main loop admits the row like a same-host one.  The
  source lease still retires on the Coordinator's SourceReadDone.

Queue pairs are bootstrapped over the Target's framed-TCP server
(``dt_host_tcp``).  On a QP error, or when the Coordinator reports a Draft
session lost (heartbeat timeout), the Target destroys that session's queue
pairs before any of its leases are released: a destroyed QP is the proof
that the NIC no longer touches the feature pool or the staging region on its
behalf.  Until that proof exists the affected slots stay quarantined.
"""

from __future__ import annotations

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

from sglang.srt.speculative.dt_candidate_p2p import (
    COMPLETION_RECORD_BYTES,
    COMPLETION_RECORD_WORDS,
)
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
    MailboxConsumer,
    PAYLOAD_OFFSET,
    RING_CAPACITY,
    SEQ_MASK,
    STATE_CLOSED,
    STATE_SERVING,
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
_READY_RECORD_FIELDS = (
    ("seq", "Q"),
    ("pool_tag", "Q"),
    ("session_tag", "Q"),
    ("slot_id", "I"),
    ("slot_generation", "I"),
)
_READY_RECORD = struct.Struct("<" + "".join(code for _, code in _READY_RECORD_FIELDS))
READY_RECORD_BYTES = _READY_RECORD.size
# Field name -> (byte offset, width); the native ring worker decodes by it.
READY_RECORD_LAYOUT = {
    name: (
        struct.calcsize("<" + "".join(code for _, code in _READY_RECORD_FIELDS[:i])),
        struct.calcsize("<" + code),
    )
    for i, (name, code) in enumerate(_READY_RECORD_FIELDS)
}
# Ready records a Draft can hold unconsumed; each is one posted receive, so
# the Target's sends past this depth wait on RNR retry (backpressure).
READY_RING_DEPTH = 1024
# Inline-data cap requested for every send queue; records must fit inline so
# their send buffers need no registration and are reusable at post time.
MAX_INLINE_BYTES = 256
# Outstanding signaled sends per Target-side ready-record queue pair.
READY_SEND_DEPTH = 1024
# Send-queue depth of a Draft's candidate-push queue pair, in work requests
# (a row is three, a tombstone two); a full queue makes the next group wait
# for completions of earlier ones.
PUSH_SEND_DEPTH = 4096
# One candidate-return group as csrc/dt_rdma_verbs.cpp lays it out
# (CandidateGroupHeader, then `rows` CandidateGroupRow).
CANDIDATE_GROUP_HEADER = np.dtype(
    [
        ("qp", "<u8"),
        ("wr_id", "<u8"),
        ("data_rkey", "<u4"),
        ("record_rkey", "<u4"),
        ("source_lkey", "<u4"),
        ("rows", "<u4"),
        ("status", "<i4"),
        ("reserved", "<u4", 7),
    ]
)
CANDIDATE_GROUP_ROW = np.dtype(
    [
        ("source_addr", "<u8"),
        ("data_addr", "<u8"),
        ("record_addr", "<u8"),
        ("nbytes", "<u4"),
        ("reserved", "<u4"),
        ("record", "<u8", COMPLETION_RECORD_WORDS),
    ]
)
assert CANDIDATE_GROUP_HEADER.itemsize == 64 and CANDIDATE_GROUP_ROW.itemsize == 96
# Group status before its post ran; the native post writes 0 or the error.
_GROUP_PENDING = -1

# Task-record ring (issue #182): one page holds the per-slot generation
# table the Target mirrors remotely (entry image offsets below share the
# local ring's SSOT in dt_direct_ring), followed by RING_CAPACITY entries of
# ENTRY_BYTES.  Posted receives are entry credits: the Target's writes past
# this depth wait on RNR retry, exactly like the ready-record ring.
REMOTE_RING_TABLE_BYTES = mmap.PAGESIZE
# How long an idle proxy thread waits before re-checking for shutdown or
# housekeeping; arrivals wake it at once, so this only bounds GIL wakeups.
_IDLE_WAIT_S = 0.05
# Wait budget of one operation against a healthy peer: a bootstrap request,
# a candidate write, or a hidden pull (its ready record and READ together).
RDMA_OP_TIMEOUT_S = 30.0
# Hidden-pull READs a Draft posts before polling completions; it is the pull
# queue pair's send-queue depth, so a read_batch never overruns the QP.  The
# native ring worker uses the same window on the ring queue pair.
PULL_READ_WINDOW = 4
# Poll period of a relayed pull waiting on a ready record the native ring
# worker consumes (DraftRdmaPuller native_ready).
_NATIVE_READY_WAIT_S = 50e-6

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


# Task-record ring (issue #182).  The wire image is byte-identical to the
# local ring's v1 entry (dt_direct_ring.TargetRingWriter.write): a 64-byte
# header with the wire seq released last, the producer-event handle, and the
# msgpack DraftTask payload.  A single RDMA WRITE delivers the whole entry,
# so the release ordering is moot on the wire; keeping one image format lets
# both sides share the header SSOT, and the Draft's native worker validate it
# with the local fences plus the per-slot generation fence it applies to
# kernel-published records (csrc/dt_ring_worker.cpp remote_record_check).

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
        "dt_rdma_post_candidates": ([vp], None),
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
    driver.cuLaunchHostFunc.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    driver.cuLaunchHostFunc.restype = ctypes.c_int
    driver.cuFlushGPUDirectRDMAWrites.argtypes = [ctypes.c_int, ctypes.c_int]
    driver.cuFlushGPUDirectRDMAWrites.restype = ctypes.c_int
    driver.cuDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    driver.cuDeviceGetPCIBusId.restype = ctypes.c_int
    return driver


_HOST_POST: tuple | None = None


def _host_post() -> tuple:
    """(cuLaunchHostFunc, the native post as a CUhostFn), resolved once."""

    global _HOST_POST
    if _HOST_POST is None:
        _HOST_POST = (
            _cuda_driver().cuLaunchHostFunc,
            ctypes.c_void_p(ctypes.cast(_lib().dt_rdma_post_candidates, ctypes.c_void_p).value),
        )
    return _HOST_POST


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
        # Its completions reap into a per-session completion queue: several
        # sessions share this endpoint, and each writer may only credit its
        # own queue pair's completions.
        self.ring_qp: QueuePair | None = None
        self.ring_cq: CompletionQueue | None = None
        self.ring_addr = 0
        self.ring_rkey = 0
        self.ring_capacity = 0
        # Largest hidden range the Draft's landing buffers take per record.
        self.ring_max_entry_bytes = 0
        self.ring_writer: TargetRemoteRingWriter | None = None
        # Pinned staging of the per-slot generation table the ring writer
        # posts into the Draft's ring ahead of a record naming a changed
        # slot; created at ring connect and published with the ring leg.
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
        # Set once fence_ring() destroyed the task-record ring leg; the
        # pull/push legs stay alive for the Coordinator-relay fallback.
        self.ring_fenced = False

    def queue_pairs(self) -> tuple[QueuePair, ...]:
        return tuple(
            qp for qp in (self.pull_qp, self.push_qp, self.ring_qp) if qp is not None
        )


class TargetRdmaEndpoint:
    """Target side of the RDMA backend: registers the feature pool (READ
    source) and the candidate arena's GPU rows and host completion records
    (GPUDirect WRITE sinks), accepts Draft connections through the bootstrap
    server, and publishes ready records.  Candidate pushes need no thread
    here: the arena's record observer consumes them (issue #223)."""

    def __init__(self, *, pool, arena, device_index: int, on_landed_push=None) -> None:
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
        self._arena_mr: MemoryRegion | None = None
        self._record_mr: MemoryRegion | None = None
        self._push_cq: CompletionQueue | None = None
        self._threads: list[threading.Thread] = []
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
            records = arena._completion_records
            if records is None:
                raise RdmaTransportError("candidate arena has no completion records")
            storage = arena.storage
            self._arena_mr = self.device.register(
                storage.data_ptr(), storage.nbytes, remote=_IBV_ACCESS_REMOTE_WRITE
            )
            self._record_mr = self.device.register(
                records.host_address, records.nbytes, remote=_IBV_ACCESS_REMOTE_WRITE
            )
            # Push queue pairs only receive WRITEs: nothing completes here.
            self._push_cq = self.device.completion_queue(1)
            arena.remote_record_listener = on_landed_push

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
            or leg == LEG_PUSH and self._arena_mr is None
        ):
            raise RdmaTransportError(f"this Target serves no {leg} leg")
        if leg == LEG_PULL:
            qp = self.device.queue_pair(
                self._ready_cq, self._ready_cq, max_send=READY_SEND_DEPTH, max_recv=1
            )
        elif leg == LEG_PUSH:
            qp = self.device.queue_pair(self._push_cq, self._push_cq, max_send=1, max_recv=1)
        else:
            if leg != LEG_RING or (self._pool is None and self._arena is None):
                raise RdmaTransportError(f"unknown RDMA leg {leg!r}")
            if int(header["ring_capacity"]) != RING_CAPACITY:
                raise RdmaTransportError("task-record ring capacity mismatch")
            if int(header["entry_bytes"]) != ENTRY_BYTES:
                raise RdmaTransportError("task-record ring entry size mismatch")
            ring_cq = self.device.completion_queue(2 * RING_CAPACITY)
            try:
                qp = self.device.queue_pair(
                    ring_cq, ring_cq,
                    max_send=2 * RING_CAPACITY, max_recv=1,
                )
            except Exception:
                ring_cq.close()
                raise
        try:
            qp.connect(header)
        except Exception:
            qp.close()
            if leg == LEG_RING:
                ring_cq.close()
            raise
        table_staging = None
        table_staging_np = None
        table_mr = None
        slot_count = 0
        if leg == LEG_RING:
            slot_count = self._pool.slot_count if self._pool is not None else 0
            if slot_count:
                # Pinned staging for the per-slot generation posts the ring
                # writer makes ahead of a record naming a changed slot.
                # Build it before the lock and publish it with the ring leg
                # in one step: a ring writer must never be visible while its
                # table state is still missing (issue #211).
                try:
                    table_staging = torch.zeros(
                        slot_count, dtype=torch.uint32, pin_memory=True
                    )
                    table_staging_np = table_staging.numpy()
                    table_mr = self.device.register(
                        table_staging.data_ptr(), table_staging.nbytes, remote=0
                    )
                except Exception:
                    if table_mr is not None:
                        table_mr.close()
                    qp.close()
                    ring_cq.close()
                    raise
        with self._lock:
            if self._closed:
                qp.close()
                if leg == LEG_RING:
                    ring_cq.close()
                    if table_mr is not None:
                        table_mr.close()
                raise RdmaTransportError("RDMA endpoint is closed")
            session = self._sessions.setdefault(key, _TargetSession(*key))
            if session.fenced or (leg == LEG_RING and session.ring_fenced):
                qp.close()
                if leg == LEG_RING:
                    ring_cq.close()
                    if table_mr is not None:
                        table_mr.close()
                raise RdmaTransportError("draft session was fenced")
            if leg == LEG_PULL:
                previous = session.pull_qp
            elif leg == LEG_PUSH:
                previous = session.push_qp
            else:
                previous = session.ring_qp
            if previous is not None:
                qp.close()
                if leg == LEG_RING:
                    ring_cq.close()
                    if table_mr is not None:
                        table_mr.close()
                raise RdmaTransportError(f"draft session already holds a {leg} leg")
            if leg == LEG_PULL:
                session.pull_qp = qp
                session.ready_addr = int(header["ready_addr"])
                session.ready_rkey = int(header["ready_rkey"])
            elif leg == LEG_PUSH:
                # Admitted before its first write can land: from here the
                # observer flushes GPUDirect writes ahead of every record.
                try:
                    self._arena.accept_remote_producer(
                        key[1],
                        flush_target=_CU_FLUSH_TARGET_CURRENT_CTX,
                        flush_scope=_CU_FLUSH_TO_OWNER,
                    )
                except Exception:
                    qp.close()
                    raise
                session.push_qp = qp
            else:
                session.ring_qp = qp
                session.ring_cq = ring_cq
                session.ring_addr = int(header["ring_addr"])
                session.ring_rkey = int(header["ring_rkey"])
                session.ring_capacity = int(header["ring_capacity"])
                session.ring_max_entry_bytes = int(header["max_entry_bytes"])
                session.table_staging = table_staging
                session.table_staging_np = table_staging_np
                session.table_mr = table_mr
            self._by_tag[qp.tag] = session
        response = qp.endpoint(self.device.lid)
        if leg == LEG_PULL:
            pool = self._pool
            response.update(
                pool_addr=self._pool_mr.address,
                pool_rkey=self._pool_mr.rkey,
                slot_nbytes=pool.slot_nbytes,
                slot_count=pool.slot_count,
                allocation_offset=pool.allocation_offset,
                pool_generation=pool.pool_generation,
                source_session_id=pool.source_session_id,
            )
        elif leg == LEG_PUSH:
            response.update(
                arena_addr=self._arena_mr.address,
                arena_rkey=self._arena_mr.rkey,
                record_addr=self._record_mr.address,
                record_rkey=self._record_mr.rkey,
                slot_count=self._arena.slot_count,
                row_width=self._arena.row_width,
            )
        else:
            # The per-record generation post is the only fence a fresh table
            # needs: the ring writer posts a slot's generation before its
            # first record naming that (slot, generation), and same-QP post
            # order makes the table land before the record.  Valid
            # generations are >= 1 (a slot's generation is incremented on
            # reservation) and the Draft's table starts zeroed like the
            # writer's _sent_generations, so a record naming a slot reserved
            # before this connect still triggers that first post.
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
                if (
                    session.ring_qp is None
                    or session.fenced
                    or session.ring_fenced
                ):
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
                if session.ring_cq is not None:
                    session.ring_cq.close()
                    session.ring_cq = None
                fenced[key[1]] = all(qp.handle is None for qp in session.queue_pairs())
        if fenced:
            logger.warning(
                "fenced RDMA queue pairs of draft %s sessions %s",
                draft_instance_id, fenced,
            )
        return fenced

    def fence_ring(self, draft_instance_id: str, draft_session_id: str) -> bool:
        """Destroy only the session's task-record ring queue pair.

        A ring-leg write or completion failure must not take the pull and
        push legs with it: the session's Draft stays alive and its
        Coordinator-relayed rounds still move features (pull leg) and
        candidates (push leg).  The ring leg itself can never reconnect;
        record planning falls back to the Coordinator relay.
        """
        with self._lock:
            session = self._sessions.get((draft_instance_id, draft_session_id))
            if session is None:
                return False
            session.ring_fenced = True
            qp = session.ring_qp
            session.ring_qp = None
            if qp is not None:
                self._by_tag.pop(qp.tag, None)
                try:
                    qp.close()
                except RdmaTransportError:
                    pass
            if session.ring_cq is not None:
                session.ring_cq.close()
                session.ring_cq = None
            session.ring_writer = None
            return qp is None or qp.handle is None

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
            if session.ring_cq is not None:
                session.ring_cq.close()
                session.ring_cq = None
            if session.table_mr is not None:
                session.table_mr.close()
                session.table_mr = None
                session.table_staging = None
                session.table_staging_np = None
        if self._arena is not None:
            self._arena.remote_record_listener = None
        for resource in (
            self._pool_mr,
            self._arena_mr,
            self._record_mr,
            self._ready_cq,
            self._push_cq,
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
    copies the range into the destination on the worker's stream.

    With ``native_ready`` the Draft's native ring worker is the only consumer
    of the ready records (it gates its own READs on them, issue #198) and
    keeps ``_ready`` current in place; this object then only pulls."""

    def __init__(self, *, host: str, port: int, device: torch.device,
                 draft_instance_id: str, draft_session_id: str,
                 native_ready: bool = False) -> None:
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
            self.allocation_offset = int(remote["allocation_offset"])
            self.native_ready = native_ready
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

        if self.native_ready:
            return
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
            if self.native_ready:
                time.sleep(_NATIVE_READY_WAIT_S)
            else:
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


class CandidateGroup:
    """One posted (or pending) candidate-return group to one Target.

    ``buffer`` is the native descriptor the post reads and whose status it
    writes last; the post may run in a CUDA host function after the
    submitting call returned, so the pusher keeps every group it retired
    until that status is written."""

    __slots__ = ("pusher", "buffer", "header", "ticket", "wrs", "done", "error", "post_list")

    def __init__(self, pusher, buffer, ticket: int, wrs: int) -> None:
        self.pusher = pusher
        self.buffer = buffer
        self.header = buffer[: CANDIDATE_GROUP_HEADER.itemsize].view(
            CANDIDATE_GROUP_HEADER
        )[0]
        self.ticket = ticket
        self.wrs = wrs
        self.done = False
        self.error: BaseException | None = None
        self.post_list = None

    def poll(self) -> bool:
        """True once the Target NIC acknowledged the group's last WRITE;
        raises if the group can never complete."""

        if not self.done and self.error is None:
            status = int(self.header["status"])
            if status == _GROUP_PENDING:
                return False
            if status != 0:
                self.pusher._fail(
                    RdmaTransportError(f"candidate WRITE post failed ({status})")
                )
            else:
                self.pusher.reap()
        if self.error is not None:
            raise self.error
        return self.done


class CandidatePost:
    """The native post of one submit's candidate groups (several Targets).

    ``launch`` runs it from a CUDA host function stream-ordered after
    everything already enqueued (the mirror-row copies); ``post`` runs it at
    once, for groups that depend on no GPU work (tombstones).  The list
    buffer must outlive the host function, so the groups keep it."""

    def __init__(self, groups: list[CandidateGroup]) -> None:
        self.groups = [group for group in groups if group.error is None]
        self._list = np.array(
            [len(self.groups), *(group.buffer.ctypes.data for group in self.groups)],
            dtype=np.uint64,
        )
        for group in self.groups:
            group.post_list = self._list

    def launch(self, stream: torch.cuda.Stream) -> None:
        if not self.groups:
            return
        launch_host_func, post_fn = _host_post()
        rc = launch_host_func(
            ctypes.c_void_p(stream.cuda_stream),
            post_fn,
            ctypes.c_void_p(self._list.ctypes.data),
        )
        if rc != 0:
            error = RdmaTransportError(f"cuLaunchHostFunc failed ({rc})")
            for group in self.groups:
                group.pusher._fail(error)

    def post(self) -> None:
        if self.groups:
            _lib().dt_rdma_post_candidates(ctypes.c_void_p(self._list.ctypes.data))

    def host_call(self) -> tuple[int, int]:
        """The native post as (CUhostFn, argument) for another thread to
        launch on a stream (issue #244); (0, 0) when no group is live."""
        if not self.groups:
            return 0, 0
        return _host_post()[1].value, self._list.ctypes.data


class DraftRdmaPusher:
    """Candidate-return leg of one Draft to one remote Target (issue #223).

    Holds the queue pair, the Target arena's GPU rows and host completion
    records (remote addresses and keys), and a registered GPU mirror of the
    arena rows that serves as the GPUDirect WRITE source: a row for slot s is
    copied there on the push stream, and slot s is reused only after the
    Target freed it, which needs the record that this leg posts after the
    row, so the mirror row is never rewritten under a pending WRITE.
    Not thread-safe: the scheduler thread owns it; only the native post may
    run on a CUDA host-function thread."""

    def __init__(self, *, host: str, port: int, device: torch.device,
                 draft_instance_id: str, draft_session_id: str) -> None:
        self._conn = _DraftConnection(
            leg=LEG_PUSH, host=host, port=port, device_index=_index(device),
            draft_instance_id=draft_instance_id, draft_session_id=draft_session_id,
        )
        self._mirror_mr: MemoryRegion | None = None
        self._groups: dict[int, CandidateGroup] = {}
        # Retired (completed or failed) groups whose native post has not
        # written its status yet: their descriptors must outlive it.
        self._unposted: list[CandidateGroup] = []
        self._tickets = itertools.count(1)
        self._pending_wrs = 0
        self._broken: BaseException | None = None
        try:
            conn = self._conn
            # One completion per group at most, so the queue depth bounds
            # them too.
            conn.send_cq.close()
            conn.send_cq = conn.device.completion_queue(PUSH_SEND_DEPTH)
            conn.qp = conn.device.queue_pair(
                conn.send_cq, conn.send_cq, max_send=PUSH_SEND_DEPTH, max_recv=1
            )
            remote = conn.connect({})
            self._arena_addr = int(remote["arena_addr"])
            self._arena_rkey = int(remote["arena_rkey"])
            self._record_addr = int(remote["record_addr"])
            self._record_rkey = int(remote["record_rkey"])
            self._slot_count = int(remote["slot_count"])
            self.row_width = int(remote["row_width"])
            self._mirror = torch.zeros(
                (self._slot_count, self.row_width), dtype=torch.int32, device=device
            )
            self._mirror_mr = conn.device.register(
                self._mirror.data_ptr(), self._mirror.nbytes, remote=0
            )
        except Exception:
            self.close()
            raise

    def mirror_row(self, slot_id: int) -> int:
        """Device address of the mirror row the push copies slot ``slot_id`` into."""

        if not 0 <= slot_id < self._slot_count:
            raise RdmaTransportError("candidate slot is outside the Target arena")
        return self._mirror.data_ptr() + slot_id * self.row_width * 4

    def group(self, rows) -> CandidateGroup:
        """Describe one group: ``rows`` are ``(slot_id, with_row, record)``
        with ``record`` the slot's eight completion-record words."""

        if self._broken is not None:
            group = CandidateGroup(self, np.zeros(64, np.uint8), 0, 0)
            group.error = self._broken
            return group
        wrs = sum(3 if with_row else 2 for _slot, with_row, _record in rows)
        deadline = time.monotonic() + RDMA_OP_TIMEOUT_S
        while self._pending_wrs + wrs > PUSH_SEND_DEPTH:
            # Queue full: earlier groups' host functions run as the push
            # stream advances, then their completions free the queue.
            self.reap(1_000_000)
            if self._broken is not None:
                return self.group(rows)
            if time.monotonic() >= deadline:
                raise RdmaTransportError("candidate push queue pair did not drain")
        buffer = np.zeros(
            CANDIDATE_GROUP_HEADER.itemsize + len(rows) * CANDIDATE_GROUP_ROW.itemsize,
            dtype=np.uint8,
        )
        ticket = next(self._tickets)
        header = buffer[: CANDIDATE_GROUP_HEADER.itemsize].view(CANDIDATE_GROUP_HEADER)
        header["qp"] = self._conn.qp.handle
        header["wr_id"] = ticket
        header["data_rkey"] = self._arena_rkey
        header["record_rkey"] = self._record_rkey
        header["source_lkey"] = self._mirror_mr.lkey
        header["rows"] = len(rows)
        header["status"] = _GROUP_PENDING
        table = buffer[CANDIDATE_GROUP_HEADER.itemsize :].view(CANDIDATE_GROUP_ROW)
        row_nbytes = self.row_width * 4
        for index, (slot_id, with_row, record) in enumerate(rows):
            entry = table[index]
            if with_row:
                entry["source_addr"] = self.mirror_row(slot_id)
                entry["nbytes"] = row_nbytes
            elif not 0 <= slot_id < self._slot_count:
                raise RdmaTransportError("candidate slot is outside the Target arena")
            entry["data_addr"] = self._arena_addr + slot_id * row_nbytes
            entry["record_addr"] = self._record_addr + slot_id * COMPLETION_RECORD_BYTES
            entry["record"] = record
        group = CandidateGroup(self, buffer, ticket, wrs)
        self._groups[ticket] = group
        self._pending_wrs += wrs
        return group

    def reap(self, timeout_ns: int = 0) -> None:
        if self._unposted:
            self._unposted = [
                group
                for group in self._unposted
                if int(group.header["status"]) == _GROUP_PENDING
            ]
        if self._broken is not None or not self._groups:
            return
        for ticket, status, _opcode, _imm, _qpn in self._conn.send_cq.poll(timeout_ns):
            if status != _IBV_WC_SUCCESS:
                self._fail(
                    RdmaTransportError(f"candidate WRITE completed with status {status}")
                )
                return
            group = self._groups.pop(ticket, None)
            if group is not None:
                group.done = True
                self._pending_wrs -= group.wrs
                if int(group.header["status"]) == _GROUP_PENDING:
                    self._unposted.append(group)

    def _fail(self, error: BaseException) -> None:
        """A failed post or completion: the queue pair is unusable, so every
        unfinished group and every later one fails."""

        if self._broken is None:
            self._broken = error
        for group in self._groups.values():
            if group.error is None:
                group.error = self._broken
            if int(group.header["status"]) == _GROUP_PENDING:
                self._unposted.append(group)
        self._groups.clear()
        self._pending_wrs = 0

    def close(self) -> None:
        self._conn.close()
        if self._mirror_mr is not None:
            self._mirror_mr.close()
            self._mirror_mr = None
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
    Draft's READ completion is not observable here), so the Target's
    planner publishes the record before the Coordinator copy goes out and
    skips the ring-outstanding bookkeeping; the source lease retires on the
    Coordinator's SourceReadDone, exactly like a relayed task's.
    """

    is_remote = True

    def __init__(
        self, endpoint: TargetRdmaEndpoint, session: _TargetSession, *, slot_count: int
    ) -> None:
        self._endpoint = endpoint
        self._session = session
        self.draft_instance_id = session.draft_instance_id
        self.draft_session_id = session.draft_session_id
        self.capacity = session.ring_capacity
        # The Draft worker READs each record's rows into one landing buffer.
        self.max_entry_bytes = session.ring_max_entry_bytes
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
            "task-record ring to draft session %s failed: %s; fencing its ring leg",
            self.draft_session_id,
            reason,
        )
        # Only the ring QP is destroyed: the session's pull and push legs
        # keep serving the Coordinator-relay fallback.
        self._endpoint.fence_ring(self.draft_instance_id, self.draft_session_id)

    def _reap(self) -> None:
        """Reap this session's signaled record completions in post order."""

        if self._session.ring_qp is None or self._session.ring_cq is None:
            return
        for _tag, status, _opcode, _imm, _qpn in self._session.ring_cq.poll(0):
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


class DraftRemoteRing(MailboxConsumer):
    """Draft-side end of one remote Target's task-record ring.

    The ring buffer is one pinned region: page 0 holds the per-slot
    generation table the Target mirrors remotely, followed by
    ``capacity`` 4096-byte entries.  Posted receives are entry credits.  The
    native ring worker (``DraftRingWorker.attach_remote``) is the only
    consumer: it validates each arrived record, re-posts its receive once the
    record is snapshotted, READs its hidden rows into the shared landing
    buffers over this ring's queue pair (gated on ``puller``'s ready records)
    and publishes INPUT_READY; the main loop takes entries through
    :class:`MailboxConsumer`.  A failed leg is fail-stop in the worker: a
    record is never silently skipped, and a fenced queue pair proves the
    Target's NIC no longer touches this ring.

    ``landing`` is the worker's landing tensor (None for token-only rounds);
    ``puller`` the Target's pull leg, opened in ``native_ready`` mode (None
    for token-only rounds), owned by the feature pull transport.  This object
    owns the ring and landing registrations and outlives the worker.
    """

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
        landing: torch.Tensor | None,
        max_entry_bytes: int,
        puller: DraftRdmaPuller | None,
    ) -> None:
        self.target_session_id = target_session_id
        self.target_instance_id = target_instance_id
        self.ring_id = ring_id
        self._init_mailbox(RING_CAPACITY)
        self.ready_words = np.zeros(RING_CAPACITY, dtype=np.uint32)
        self.records_taken = 0
        self.pool_generation = ""
        self.source_session_id = ""
        self._puller = puller
        self._landing = landing
        self._landing_mr: MemoryRegion | None = None
        self._ring_mr: MemoryRegion | None = None
        self._conn: _DraftConnection | None = None
        try:
            self._conn = conn = _DraftConnection(
                leg=LEG_RING, host=host, port=port, device_index=_index(device),
                draft_instance_id=draft_instance_id, draft_session_id=draft_session_id,
            )
            if landing is not None:
                # Registered before the leg connects: once connected, the
                # Target may write records the worker must be able to READ.
                self._landing_mr = conn.device.register(
                    landing.data_ptr(), landing.numel() * landing.element_size(), remote=0
                )
            conn.recv_cq = conn.device.completion_queue(2 * RING_CAPACITY)
            conn.qp = conn.device.queue_pair(
                conn.send_cq, conn.recv_cq, max_send=PULL_READ_WINDOW,
                max_recv=RING_CAPACITY,
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
                    "max_entry_bytes": max_entry_bytes,
                }
            )
            self.pool_generation = str(remote["pool_generation"])
            self.source_session_id = str(remote["source_session_id"])
            self._slot_count = int(remote["slot_count"])
            if self._slot_count * 4 > REMOTE_RING_TABLE_BYTES:
                raise RdmaTransportError("source slot count exceeds the ring table")
            if puller is not None and (
                puller.pool_generation != self.pool_generation
                or puller.source_session_id != self.source_session_id
            ):
                raise RdmaTransportError("ring and pull legs name different sources")
        except Exception:
            self.close()
            raise

    @property
    def entries_address(self) -> int:
        return self._ring_mr.address + REMOTE_RING_TABLE_BYTES

    def native_spec(self):
        """The worker's view of this ring (``RemoteRingSpec``)."""

        from sglang.srt.speculative.dt_direct_ring import RemoteRingSpec

        lib = _lib()
        conn = self._conn
        spec = RemoteRingSpec(
            post_read=ctypes.cast(lib.dt_rdma_post_read, ctypes.c_void_p).value,
            post_recv=ctypes.cast(lib.dt_rdma_post_recv, ctypes.c_void_p).value,
            poll=ctypes.cast(lib.dt_rdma_poll, ctypes.c_void_p).value,
            record_qp=conn.qp.handle,
            record_recv_cq=conn.recv_cq.handle,
            record_send_cq=conn.send_cq.handle,
            record_tag=conn.qp.tag,
            table_host=self._ring_mr.address,
            read_window=PULL_READ_WINDOW,
            drain_timeout_ns=int(RDMA_OP_TIMEOUT_S * 1e9),
            wc_success=_IBV_WC_SUCCESS,
            wc_recv_with_imm=_IBV_WC_RECV_RDMA_WITH_IMM,
            flush_target=_CU_FLUSH_TARGET_CURRENT_CTX,
            flush_scope=_CU_FLUSH_TO_OWNER,
            ready_record_bytes=READY_RECORD_BYTES,
        )
        for name, (offset, width) in READY_RECORD_LAYOUT.items():
            setattr(spec, f"ready_{name}_at", offset)
            setattr(spec, f"ready_{name}_bytes", width)
        puller = self._puller
        if puller is not None:
            ready = puller._conn
            spec.landing_lkey = self._landing_mr.lkey
            spec.pool_addr = puller._pool_addr
            spec.pool_rkey = puller._pool_rkey
            spec.pool_allocation_offset = puller.allocation_offset
            spec.slot_nbytes = puller._slot_nbytes
            spec.slot_count = puller._slot_count
            spec.ready_qp = ready.qp.handle
            spec.ready_recv_cq = ready.recv_cq.handle
            spec.ready_tag = ready.qp.tag
            spec.ready_ring_host = puller._ring_mr.address
            spec.ready_depth = READY_RING_DEPTH
            spec.ready_first_seq = puller._expected_seq
            spec.ready_table_host = puller._ready.ctypes.data
            spec.pool_tag = puller._pool_tag
            spec.session_tag = puller._session_tag
        return spec

    def take_ready(self) -> RingEntry | None:
        entry = super().take_ready()
        if entry is not None:
            self.records_taken += 1
        return entry

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
        self._puller = None
        for mr in (self._ring_mr, self._landing_mr):
            if mr is not None:
                mr.close()
        self._ring_mr = None
        self._landing_mr = None
        self._landing = None
        if self._conn is not None:
            self._conn.device.close()
            self._conn = None
