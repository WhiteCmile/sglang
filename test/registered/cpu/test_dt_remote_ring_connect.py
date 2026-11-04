"""CPU contract for the remote ring leg's connect publication (issue #211).

PR #207's review p1: ``TargetRdmaEndpoint.handle_connect`` installed the ring
QP under the endpoint lock but created the generation-table staging and MR
outside it, so concurrent task planning could obtain a ring writer whose
``table_staging_np``/``table_mr`` were still None and crash the Target
controller with an uncaught TypeError.  A failure in that window further left
the session half-installed, permanently shadowing the Coordinator-relay
fallback.

These tests pin the fix against a stub RDMA device: the whole ring-leg state
(table staging, MR, QP, CQ) is built before the lock and published in one
step, so while the table registration is delayed ``ring_writer()`` keeps
returning None and a registration failure rolls the session back to "no ring
leg" (other legs and the relay fallback stay usable).  They also pin p2 of
the same review: no whole-table generation snapshot is posted at connect —
the writer's per-record generation post is the only fence, and it must be
posted ahead of the first record naming a changed slot.

Run only after the task's external release/gate; imports can load heavy modules.
"""

import itertools
import threading
import time

import numpy as np
import pytest
import torch

from sglang.srt.speculative.dt_rdma import (
    ENTRY_BYTES,
    LEG_PULL,
    LEG_RING,
    RING_CAPACITY,
    REMOTE_RING_TABLE_BYTES,
    TargetRdmaEndpoint,
)

_SLOT_COUNT = 4


class _StubMR:
    def __init__(self, address, nbytes, rkey):
        self.address = address
        self.nbytes = nbytes
        self.rkey = rkey
        self.closed = False

    def close(self):
        self.closed = True


class _StubCQ:
    def __init__(self):
        self.closed = False

    def poll(self, _n):
        return []

    def close(self):
        self.closed = True


class _StubQP:
    def __init__(self):
        self.tag = next(_StubQP._tags)
        self.closed = False
        self.posts = []

    _tags = itertools.count(1)

    def connect(self, _header):
        pass

    def endpoint(self, _lid):
        return {"tag": self.tag}

    def post_write(self, mr, local, nbytes, remote, rkey, **_kw):
        self.posts.append((mr, local, nbytes, remote, rkey))

    def post_recv(self, _n=1):
        pass

    def close(self):
        self.closed = True


class _StubDevice:
    """register() blocks on ``gate`` while one is set and raises while
    ``fail_small_register`` is set; both faults apply only to the small
    generation-table registration, never the writer's record staging."""

    def __init__(self):
        self.lid = 7
        self.mrs = []
        self.small_mrs = []
        self.gate = None
        self.fail_small_register = False

    def completion_queue(self, _n):
        return _StubCQ()

    def queue_pair(self, _scq, _rcq, max_send=0, max_recv=0):
        return _StubQP()

    def register(self, addr, nbytes, remote=0):
        if self.fail_small_register and nbytes < ENTRY_BYTES:
            raise RuntimeError("injected pinned-allocation/MR failure")
        if self.gate is not None and nbytes < ENTRY_BYTES:
            assert self.gate.wait(30), "fault-injection gate never opened"
        mr = _StubMR(addr, nbytes, rkey=0x1000 + len(self.mrs))
        self.mrs.append(mr)
        if nbytes < ENTRY_BYTES:
            self.small_mrs.append(mr)
        return mr


class _StubPool:
    def __init__(self, slot_count=_SLOT_COUNT):
        self.slot_count = slot_count
        self.slot_nbytes = 4096
        self.allocation_offset = 0
        self.pool_generation = "pool-gen"
        self.source_session_id = "source-session"


def _make_endpoint(monkeypatch, device, pool):
    # The stub device never dereferences host memory, so pinning is
    # irrelevant; without a GPU, torch.zeros(pin_memory=True) raises.  The
    # race under test is at the Python publication level.
    real_zeros = torch.zeros

    def zeros_no_pin(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return real_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", zeros_no_pin)
    ep = TargetRdmaEndpoint.__new__(TargetRdmaEndpoint)
    ep._arena = None
    ep._on_landed_push = None
    ep._device_index = 0
    ep.device = device
    ep._lock = threading.Lock()
    ep._sessions = {}
    ep._by_tag = {}
    ep._closed = False
    ep._pool_mr = _StubMR(0xA000, 1 << 20, 0x2000)
    ep._ready_cq = None
    ep._ready_generation = None
    ep._publications = None
    ep._outstanding = {}
    ep._staging = None
    ep._staging_mr = None
    ep._push_cq = None
    ep.rejected_pushes = 0
    ep._threads = []
    ep._pool = pool
    return ep


def _ring_header():
    return {
        "leg": LEG_RING,
        "draft_instance_id": "inst",
        "draft_session_id": "sess",
        "ring_capacity": RING_CAPACITY,
        "entry_bytes": ENTRY_BYTES,
        "max_entry_bytes": 1 << 20,
        "ring_addr": 0xDEAD0000,
        "ring_rkey": 0xBEEF,
    }


def _write_one(writer, slot_id=0, generation=1):
    return writer.write(
        payload=b"task",
        has_pull=True,
        nbytes=8,
        source_slot_id=slot_id,
        slot_generation=generation,
        event_handle=b"e" * 64,
    )


def test_ring_writer_never_visible_before_table_state(monkeypatch):
    device = _StubDevice()
    device.gate = threading.Event()  # table register() blocks until released
    ep = _make_endpoint(monkeypatch, device, _StubPool())
    outcome = {}

    def connect():
        response, _ = ep.handle_connect(_ring_header(), b"")
        outcome["slot_count"] = response["slot_count"]

    t = threading.Thread(target=connect)
    t.start()
    # On dt-v3 the ring QP was already installed at this point while the
    # table state was still None; a planner hitting the window crashed with
    # TypeError.  Poll for the half-installed session for a bounded window.
    deadline = time.monotonic() + 5
    half_installed = None
    while time.monotonic() < deadline and t.is_alive():
        session = ep._sessions.get(("inst", "sess"))
        if session is not None and session.ring_qp is not None:
            half_installed = session
            break
        assert ep.ring_writer("sess") is None
        time.sleep(0.0005)
    device.gate.set()
    t.join(30)
    assert not t.is_alive()
    assert half_installed is None, (
        "ring writer state became visible before its generation-table state"
    )
    assert outcome["slot_count"] == _SLOT_COUNT
    session = ep._sessions[("inst", "sess")]
    assert session.ring_qp is not None
    assert session.table_staging_np is not None
    assert session.table_mr is not None

    writer = ep.ring_writer("sess")
    assert writer is not None
    _write_one(writer)
    ring_qp = session.ring_qp
    # The per-record generation post is the only fence: it lands ahead of
    # the record it names (same-QP post order), and connect posted no
    # whole-table snapshot.
    assert len(ring_qp.posts) == 2
    gen_mr, gen_local, gen_nbytes, gen_remote, gen_rkey = ring_qp.posts[0]
    assert gen_mr is session.table_mr
    assert gen_nbytes == 4
    assert gen_remote == session.ring_addr
    assert gen_rkey == session.ring_rkey
    assert gen_local == session.table_mr.address + 4 * 0
    rec_mr, _rec_local, rec_nbytes, rec_remote, _rec_rkey = ring_qp.posts[1]
    assert rec_nbytes == ENTRY_BYTES
    # seq 1 maps to ring index 1 (both sides skip wire seq 0).
    assert rec_remote == session.ring_addr + REMOTE_RING_TABLE_BYTES + ENTRY_BYTES


def test_connect_rolls_back_to_no_ring_writer_on_table_failure(monkeypatch):
    device = _StubDevice()
    device.fail_small_register = True
    ep = _make_endpoint(monkeypatch, device, _StubPool())
    with pytest.raises(RuntimeError, match="injected"):
        ep.handle_connect(_ring_header(), b"")
    # Rolled back: no ring writer is visible, so planning keeps using the
    # Coordinator-relay fallback instead of crashing on a half-built writer.
    assert ep.ring_writer("sess") is None
    session = ep._sessions.get(("inst", "sess"))
    if session is not None:
        assert session.ring_qp is None
        assert session.table_staging_np is None
        assert session.table_mr is None
        assert not session.fenced
        assert not session.ring_fenced
    # The failed ring leg leaves the session connectable on other legs.
    response, _ = ep.handle_connect(
        {
            "leg": LEG_PULL,
            "draft_instance_id": "inst",
            "draft_session_id": "sess",
            "ready_addr": 123,
            "ready_rkey": 456,
        },
        b"",
    )
    assert response["pool_generation"] == "pool-gen"
    assert ep.ring_writer("sess") is None


def test_no_whole_table_snapshot_posted_at_connect(monkeypatch):
    device = _StubDevice()
    ep = _make_endpoint(monkeypatch, device, _StubPool())
    ep.handle_connect(_ring_header(), b"")
    session = ep._sessions[("inst", "sess")]
    # The ring QP must carry no posts right after connect: the first
    # generation word travels with the per-record fence, not a snapshot.
    assert session.ring_qp.posts == []
    assert len(device.small_mrs) == 1
    staging_np = session.table_staging_np
    assert isinstance(staging_np, np.ndarray)
    assert staging_np.dtype == np.uint32
    assert len(staging_np) == _SLOT_COUNT
    assert not staging_np.any()  # zeroed like the Draft's table and writer
