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

Queue pairs are bootstrapped over the Target's framed-TCP server
(``dt_host_tcp``).  On a QP error, or when the Coordinator reports a Draft
session lost (heartbeat timeout), the Target destroys that session's queue
pairs before any of its leases are released: a destroyed QP is the proof
that the NIC no longer touches the feature pool or the staging region on its
behalf.  Until that proof exists the affected slots stay quarantined.
"""

from __future__ import annotations

import ctypes
import logging
import os
import queue
import secrets
import struct
import threading
import time
from pathlib import Path

import numpy as np
import torch

from sglang.srt.speculative.dt_direct_ring import (
    _build_native_library,
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

# Ready record: seq, pool-generation tag, source-session tag, slot id, slot
# generation.  One record per published feature slot per connected Draft.
_READY_RECORD = struct.Struct("<QQQII")
READY_RECORD_BYTES = _READY_RECORD.size
# Ready records a Draft can hold unconsumed; each is one posted receive, so
# the Target's sends past this depth wait on RNR retry (backpressure).
READY_RING_DEPTH = 1024
# Candidate record header: transfer hash, producer tag, then the five int32
# metadata words exactly as the arena stores them (generation, nonce low,
# nonce high, valid length, root token).  The int32 row follows.
_CANDIDATE_HEADER = struct.Struct("<QI5I")
CANDIDATE_HEADER_BYTES = _CANDIDATE_HEADER.size
# Inline-data cap requested for every send queue; records must fit inline so
# their send buffers need no registration and are reusable at post time.
MAX_INLINE_BYTES = 256
# Outstanding signaled sends per Target-side ready-record queue pair.
READY_SEND_DEPTH = 1024
# How long an idle proxy thread waits before re-checking for shutdown or
# housekeeping; arrivals wake it at once, so this only bounds GIL wakeups.
_IDLE_WAIT_S = 0.05
# Wait budget of one operation against a healthy peer: a bootstrap request,
# a candidate write, or a hidden pull (its ready record and READ together).
RDMA_OP_TIMEOUT_S = 30.0

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
    return (
        _CANDIDATE_HEADER.pack(transfer_hash, producer_tag, *struct.unpack("<5I", metadata))
        + row
    )


def decode_candidate_record(
    record: bytes, row_width: int
) -> tuple[int, int, bytes, bytes]:
    """Return ``(transfer_hash, producer_tag, metadata_bytes, row_bytes)``."""

    transfer_hash, tag = struct.unpack_from("<QI", record, 0)
    metadata = record[12:CANDIDATE_HEADER_BYTES]
    row = record[CANDIDATE_HEADER_BYTES : CANDIDATE_HEADER_BYTES + row_width * 4]
    return transfer_hash, tag, metadata, row


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
        "dt_rdma_qp_error": ([vp], i32),
        "dt_rdma_destroy_qp": ([vp], i32),
        "dt_rdma_post_recv": ([vp, u64, i32], i32),
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
        """Completions as ``(wr_id, status, opcode, imm, qp_num)``.

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
        if self._lib.dt_rdma_post_recv(self._live(), 0, count) != 0:
            raise RdmaTransportError("ibv_post_recv failed")

    def post_write_imm(
        self, wr_id: int, data: bytes, remote_addr: int, rkey: int, imm: int
    ) -> None:
        if len(data) > self.max_inline:
            raise RdmaTransportError("record exceeds the queue pair's inline capacity")
        rc = self._lib.dt_rdma_post_write_imm_inline(
            self._live(), wr_id, data, len(data), remote_addr, rkey, imm
        )
        if rc != 0:
            raise RdmaTransportError(f"WRITE_WITH_IMM post failed ({rc})")

    def post_read(
        self, wr_id: int, local: MemoryRegion, local_addr: int, nbytes: int,
        remote_addr: int, rkey: int,
    ) -> None:
        rc = self._lib.dt_rdma_post_read(
            self._live(), wr_id, local_addr, nbytes, local.lkey, remote_addr, rkey
        )
        if rc != 0:
            raise RdmaTransportError(f"RDMA READ post failed ({rc})")

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
        self.ready_addr = 0
        self.ready_rkey = 0
        self.ready_seq = 0
        self.active = False

    def queue_pairs(self) -> tuple[QueuePair, ...]:
        return tuple(qp for qp in (self.pull_qp, self.push_qp) if qp is not None)


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
        self._by_pull_qpn: dict[int, _TargetSession] = {}
        self._by_push_qpn: dict[int, _TargetSession] = {}
        self._closed = False
        self._pool_mr: MemoryRegion | None = None
        self._ready_cq: CompletionQueue | None = None
        self._ready_generation: np.ndarray | None = None
        self._publications: queue.Queue = queue.Queue()
        # Signaled ready-record sends not yet reaped, per pull QP number; a
        # fenced QP's entry goes with it (a destroyed QP completes nothing).
        self._outstanding: dict[int, int] = {}
        self._staging: torch.Tensor | None = None
        self._staging_mr: MemoryRegion | None = None
        self._push_cq: CompletionQueue | None = None
        self.rejected_pushes = 0
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
        if leg == LEG_PULL and self._pool_mr is None or leg == LEG_PUSH and self._staging_mr is None:
            raise RdmaTransportError(f"this Target serves no {leg} leg")
        if leg == LEG_PULL:
            qp = self.device.queue_pair(
                self._ready_cq, self._ready_cq, max_send=READY_SEND_DEPTH, max_recv=1
            )
        else:
            qp = self.device.queue_pair(
                self._push_cq, self._push_cq, max_send=1,
                max_recv=self._arena.slot_count,
            )
            qp.post_recv(self._arena.slot_count)
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
            previous = session.pull_qp if leg == LEG_PULL else session.push_qp
            if previous is not None:
                qp.close()
                raise RdmaTransportError(f"draft session already holds a {leg} leg")
            if leg == LEG_PULL:
                session.pull_qp = qp
                session.ready_addr = int(header["ready_addr"])
                session.ready_rkey = int(header["ready_rkey"])
                self._by_pull_qpn[qp.qpn] = session
            else:
                session.push_qp = qp
                self._by_push_qpn[qp.qpn] = session
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
        else:
            response.update(
                staging_addr=self._staging_mr.address,
                staging_rkey=self._staging_mr.rkey,
                record_stride=self._stride,
                slot_count=self._arena.slot_count,
                row_width=self._arena.row_width,
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
                        session.ready_seq, record,
                        session.ready_addr + index * READY_RECORD_BYTES,
                        session.ready_rkey, index,
                    )
                except RdmaTransportError:
                    broken.append(session)
                else:
                    qpn = session.pull_qp.qpn
                    self._outstanding[qpn] = self._outstanding.get(qpn, 0) + 1
        for session in broken:
            logger.error(
                "ready record post failed for draft session %s; fencing it",
                session.draft_session_id,
            )
            self.fence(session.draft_instance_id, session.draft_session_id)

    def _reap_ready_sends(self) -> None:
        if not any(self._outstanding.values()):
            return
        for _wr_id, status, _opcode, _imm, qpn in self._ready_cq.poll(0):
            with self._lock:
                if qpn in self._outstanding:
                    self._outstanding[qpn] -= 1
                session = self._by_pull_qpn.get(qpn)
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
            for _wr_id, status, _opcode, slot_id, qpn in self._push_cq.poll(timeout_ns):
                with self._lock:
                    # Under the lock: fence() may destroy the QP concurrently.
                    session = self._by_push_qpn.get(qpn)
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

    def fence(self, draft_instance_id: str, draft_session_id: str = "") -> bool:
        """Destroy the queue pairs of a lost Draft session (all sessions of
        the instance when ``draft_session_id`` is empty).  Returns whether
        quiescence is proven: no queue pair of those sessions remains."""

        with self._lock:
            sessions = [
                session
                for key, session in self._sessions.items()
                if key[0] == draft_instance_id
                and (not draft_session_id or key[1] == draft_session_id)
            ]
            proven = True
            for session in sessions:
                session.active = False
                for qp in session.queue_pairs():
                    try:
                        qp.close()
                    except RdmaTransportError:
                        proven = False
                        continue
                    self._by_pull_qpn.pop(qp.qpn, None)
                    self._by_push_qpn.pop(qp.qpn, None)
                    self._outstanding.pop(qp.qpn, None)
                if all(qp.handle is None for qp in session.queue_pairs()):
                    self._sessions.pop(
                        (session.draft_instance_id, session.draft_session_id), None
                    )
        if sessions:
            logger.warning(
                "fenced RDMA queue pairs of draft %s session %s (proven=%s)",
                draft_instance_id, draft_session_id or "<all>", proven,
            )
        return proven

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
        for resource in (self._pool_mr, self._staging_mr, self._ready_cq, self._push_cq):
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
                conn.send_cq, conn.recv_cq, max_send=4, max_recv=READY_RING_DEPTH
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
            self._landing = torch.empty(self._slot_nbytes, dtype=torch.uint8, device=device)
            self._landing_mr = conn.device.register(
                self._landing.data_ptr(), self._landing.nbytes, remote=0
            )
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

    def read(
        self, source, destination: torch.Tensor, stream: torch.cuda.Stream
    ) -> tuple[int, int, int]:
        """Pull one descriptor's range into ``destination``.

        Returns the monotonic ns at which the ready record was in hand, the
        READ was posted, and its completion was polled."""

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
        ready_ns = time.monotonic_ns()
        remote = self._pool_addr + slot_id * self._slot_nbytes + source.byte_offset
        self._conn.qp.post_read(
            generation, self._landing_mr, self._landing_mr.address, source.nbytes,
            remote, self._pool_rkey,
        )
        posted_ns = time.monotonic_ns()
        _wait_one(self._conn.send_cq, "hidden RDMA READ", deadline)
        completed_ns = time.monotonic_ns()
        # A republication whose ready record arrived meanwhile may have
        # overwritten the range under the READ.
        self.drain()
        if self._ready[slot_id] != generation:
            raise StaleDescriptorError("feature source slot was republished during the read")
        make_gpu_visible(self._driver)
        with torch.cuda.stream(stream):
            destination.view(torch.uint8).reshape(-1).copy_(
                self._landing[: source.nbytes], non_blocking=True
            )
            copied = torch.cuda.Event()
            copied.record(stream)
        copied.synchronize()
        return ready_ns, posted_ns, completed_ns

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
            slot_id, record, self._staging_addr + slot_id * self._stride,
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
