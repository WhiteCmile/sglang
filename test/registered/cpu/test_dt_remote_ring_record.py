"""CPU contract for the cross-host task-record ring (issues #182, #198).

The wire image is a local-ring v1 entry (dt_direct_ring header SSOT), so the
Draft's native ring worker validates it with the local ring's fences plus the
registered-source fences it applies to kernel-published records.  These tests
pin that parity and the fault fences through the worker's pure validator
(``dt_ring_remote_record_check``, no CUDA): a record whose sequence, epoch,
per-slot generation, or geometry is stale or forged is rejected before the
worker places, READs or publishes anything for it.

Run only after the task's external release/gate; imports can load heavy modules.
"""

import ctypes
import struct

import numpy as np
import pytest

from sglang.srt.speculative.dt_direct_ring import (
    ENTRY_BYTES,
    ENTRY_V1_HEADER,
    E_EPOCH,
    E_PAYLOAD_LEN,
    E_SOURCE_OFFSET_HI,
    PAYLOAD_OFFSET,
    RING_CAPACITY,
    V2_VERSION,
    _build_worker_library,
    next_seq,
)
from sglang.srt.speculative.dt_rdma import encode_task_record

_GENERATIONS = np.array([3, 7, 1, 42], dtype=np.uint32)
_ALLOCATION_OFFSET = 8192
_SLOT_NBYTES = 1 << 20
_ROW_BYTES = 256
_LANDING_BYTES = 1 << 19
_MALFORMED = -4
_STALE_SOURCE = -5


@pytest.fixture(scope="module")
def check():
    lib = ctypes.CDLL(str(_build_worker_library()))
    fn = lib.dt_ring_remote_record_check
    u64 = ctypes.c_uint64
    fn.argtypes = [ctypes.c_char_p, u64, u64, ctypes.c_void_p, u64, u64, u64, u64, u64]
    fn.restype = ctypes.c_int

    def run(record, expected_seq, table=_GENERATIONS):
        return fn(
            record,
            ENTRY_BYTES,
            expected_seq,
            table.ctypes.data,
            len(table),
            _ALLOCATION_OFFSET,
            _SLOT_NBYTES,
            _ROW_BYTES,
            _LANDING_BYTES,
        )

    return run


def _offset(slot_id, row=0):
    return _ALLOCATION_OFFSET + slot_id * _SLOT_NBYTES + row * _ROW_BYTES


def _pull(seq, *, slot_id=1, generation=None, row=0, rows=4, payload=b"task", **kwargs):
    fields = dict(
        source_offset=_offset(slot_id, row),
        nbytes=rows * _ROW_BYTES,
        source_slot_id=slot_id,
        slot_generation=int(_GENERATIONS[slot_id]) if generation is None else generation,
        event_handle=bytes(range(64)),
    )
    fields.update(kwargs)
    return encode_task_record(seq=seq, payload=payload, has_pull=True, **fields)


def _plain(seq, payload=b"task", **kwargs):
    return encode_task_record(seq=seq, payload=payload, has_pull=False, **kwargs)


def test_round_trip_matches_local_v1_layout(check):
    seq = next_seq(1)  # 2: the wire-0 skip both sides apply identically
    record = _pull(seq, slot_id=2, row=3, rows=16, payload=b"payload-bytes")
    assert len(record) == ENTRY_BYTES
    (
        wire_seq,
        flags,
        source_offset,
        nbytes,
        source_slot,
        slot_generation,
        payload_len,
        epoch,
    ) = ENTRY_V1_HEADER.unpack_from(record, 0)
    assert wire_seq == seq & 0xFFFFFFFF
    assert epoch == seq >> 32
    assert flags == 1
    assert source_offset == _offset(2, 3)
    assert (nbytes, source_slot, slot_generation, payload_len) == (
        16 * _ROW_BYTES,
        2,
        int(_GENERATIONS[2]),
        len(b"payload-bytes"),
    )
    assert record[PAYLOAD_OFFSET : PAYLOAD_OFFSET + payload_len] == b"payload-bytes"
    assert check(record, seq) == 0
    assert check(_plain(seq), seq) == 0


def test_epoch_lap_is_validated(check):
    lap_seq = (1 << 32) + 3
    record = _plain(lap_seq)
    assert check(record, lap_seq) == 0
    # Same wire word with the wrong lap must not alias into this seq.
    stale = bytearray(record)
    struct.pack_into("<I", stale, 4 * E_EPOCH, 0)  # back to lap 0
    assert check(bytes(stale), lap_seq) == _MALFORMED


def test_replayed_record_with_stale_sequence_is_rejected(check):
    assert check(_plain(1, payload=b"first"), 1) == 0
    assert check(_plain(1, payload=b"forged"), next_seq(1)) == _MALFORMED


def test_stale_slot_generation_is_rejected(check):
    assert check(_pull(1, slot_id=1, generation=6), 1) == _STALE_SOURCE  # table holds 7
    assert check(_pull(1, slot_id=1, generation=0), 1) == _STALE_SOURCE
    unknown = _pull(1, slot_id=1, source_slot_id=len(_GENERATIONS), slot_generation=1)
    assert check(unknown, 1) == _STALE_SOURCE


def test_kernel_published_record_cannot_arrive_remotely(check):
    record = bytearray(_pull(1))
    struct.pack_into("<I", record, 4 * E_SOURCE_OFFSET_HI, V2_VERSION)
    assert check(bytes(record), 1) == _MALFORMED


def test_geometry_faults_are_rejected(check):
    oversized = bytearray(_plain(1, payload=b"x"))
    struct.pack_into("<I", oversized, 4 * E_PAYLOAD_LEN, ENTRY_BYTES)
    assert check(bytes(oversized), 1) == _MALFORMED
    # Malformed pull ranges: empty, not whole rows, larger than one landing
    # buffer.
    assert check(_pull(1, nbytes=0), 1) == _MALFORMED
    assert check(_pull(1, nbytes=_ROW_BYTES + 2), 1) == _MALFORMED
    assert check(_pull(1, rows=_LANDING_BYTES // _ROW_BYTES + 1), 1) == _MALFORMED
    # Source ranges outside the named slot, or not row-aligned inside it.
    assert check(_pull(1, slot_id=1, source_offset=_offset(0)), 1) == _STALE_SOURCE
    last_row = _SLOT_NBYTES // _ROW_BYTES - 1
    assert check(_pull(1, slot_id=1, row=last_row, rows=2), 1) == _STALE_SOURCE
    assert check(_pull(1, slot_id=1, source_offset=_offset(1) + 2), 1) == _STALE_SOURCE
    assert check(_plain(1, nbytes=16, source_slot_id=0), 1) == _MALFORMED


def test_sequence_advances_across_laps(check):
    # Drive the ring past one full lap: the slot index wraps (seq %
    # RING_CAPACITY aliases seq 1 onto slot 1 again) while the logical
    # sequence must keep validating in order, epoch word included.
    seq = 1
    for expected in range(1, RING_CAPACITY + 2):
        assert check(_pull(seq, slot_id=0, rows=1), expected) == 0
        seq = next_seq(seq)
    # The lap boundary aliases slots, not sequences: the previous lap's
    # record must not validate as the freshly written one.
    assert check(_plain(1), RING_CAPACITY + 1) == _MALFORMED
    assert check(_plain(RING_CAPACITY + 1), 1) == _MALFORMED
