"""CPU contract for the cross-host task-record ring (issue #182).

The wire image is a local-ring v1 entry (dt_direct_ring header SSOT), so the
remote ring's admission can apply the local ring worker's validation without
a second format.  These tests pin that parity and the fault fences: a record
whose sequence, epoch, per-slot generation, or geometry is stale is rejected
before admission, and the consumed entry always carries the lazy-pull
placement (landing_index -1) the Draft engine's remote branch expects.

Run only after the task's external release/gate; imports can load heavy modules.
"""

import struct

import numpy as np
import pytest

from sglang.srt.speculative.dt_direct_ring import (
    ENTRY_BYTES,
    PAYLOAD_OFFSET,
    V2_VERSION,
    next_seq,
)
from sglang.srt.speculative.dt_rdma import (
    RdmaTransportError,
    decode_task_record,
    encode_task_record,
)

# The local ring worker parses v1 entries exactly like this (its mailbox
# snapshot copies the 64-byte entry header verbatim); the remote image must.
_LOCAL_V1_HEADER = struct.Struct("<IIQIIIII")

_SLOT_COUNT = 4
_GENERATIONS = np.array([3, 7, 1, 42], dtype=np.uint32)


def _encode(seq, *, payload=b"task", has_pull=True, **kwargs):
    if has_pull and "event_handle" not in kwargs:
        kwargs["event_handle"] = bytes(range(64))
    return encode_task_record(seq=seq, payload=payload, has_pull=has_pull, **kwargs)


def _decode(record, expected_seq):
    return decode_task_record(
        record, expected_seq=expected_seq, generation_table=_GENERATIONS
    )


def test_round_trip_matches_local_v1_layout():
    seq = next_seq(1)  # 2: the wire-0 skip both sides apply identically
    record = _encode(
        seq,
        payload=b"payload-bytes",
        has_pull=True,
        source_offset=1 << 20,
        nbytes=4096,
        source_slot_id=2,
        slot_generation=int(_GENERATIONS[2]),
    )
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
    ) = _LOCAL_V1_HEADER.unpack_from(record, 0)
    assert wire_seq == seq & 0xFFFFFFFF
    assert epoch == seq >> 32
    assert flags == 1
    assert source_offset == 1 << 20
    assert (nbytes, source_slot, slot_generation, payload_len) == (
        4096,
        2,
        int(_GENERATIONS[2]),
        len(b"payload-bytes"),
    )
    assert record[PAYLOAD_OFFSET : PAYLOAD_OFFSET + payload_len] == b"payload-bytes"

    entry = _decode(record, seq)
    assert entry.seq == seq
    assert entry.has_pull
    assert entry.source_offset == 1 << 20
    assert entry.payload == b"payload-bytes"
    # Remote records never carry a local landing placement.
    assert entry.landing_index == -1
    assert entry.landing_offset == 0


def test_epoch_lap_is_validated():
    lap_seq = (1 << 32) + 3
    record = _encode(lap_seq, has_pull=False)
    assert _decode(record, lap_seq).seq == lap_seq
    # Same wire word with the wrong lap must not alias into this seq.
    stale = bytearray(record)
    struct.pack_into("<I", stale, 4 * 8, 0)  # E_EPOCH back to lap 0
    with pytest.raises(RdmaTransportError, match="stale"):
        _decode(bytes(stale), lap_seq)


def test_replayed_record_with_stale_sequence_is_rejected():
    first = _encode(1, payload=b"first", has_pull=False)
    assert _decode(first, 1).payload == b"first"
    replayed = _encode(1, payload=b"forged", has_pull=False)
    with pytest.raises(RdmaTransportError, match="stale"):
        _decode(replayed, next_seq(1))


def test_stale_slot_generation_is_rejected():
    record = _encode(
        1, source_slot_id=1, slot_generation=6  # table holds 7
    )
    with pytest.raises(RdmaTransportError, match="stale source generation"):
        _decode(record, 1)
    zero = _encode(1, source_slot_id=1, slot_generation=0)
    with pytest.raises(RdmaTransportError, match="stale source generation"):
        _decode(zero, 1)
    unknown = _encode(1, source_slot_id=_SLOT_COUNT, slot_generation=1)
    with pytest.raises(RdmaTransportError, match="no valid source slot"):
        _decode(unknown, 1)


def test_kernel_published_record_cannot_arrive_remotely():
    record = bytearray(_encode(1, has_pull=True, slot_generation=int(_GENERATIONS[1])))
    struct.pack_into("<I", record, 4 * 3, V2_VERSION)  # E_SOURCE_OFFSET_HI
    with pytest.raises(RdmaTransportError, match="kernel-published"):
        _decode(bytes(record), 1)


def test_geometry_faults_are_rejected():
    oversized = bytearray(_encode(1, has_pull=False, payload=b"x"))
    struct.pack_into("<I", oversized, 4 * 7, ENTRY_BYTES)  # E_PAYLOAD_LEN
    with pytest.raises(RdmaTransportError, match="payload length"):
        _decode(bytes(oversized), 1)

    empty_pull = _encode(1, nbytes=0, source_slot_id=0, slot_generation=3)
    with pytest.raises(RdmaTransportError, match="pull range"):
        _decode(empty_pull, 1)

    source_fields = _encode(1, has_pull=False, nbytes=16, source_slot_id=0)
    with pytest.raises(RdmaTransportError, match="carries source fields"):
        _decode(source_fields, 1)


def test_sequence_advances_across_laps():
    seq = 1
    table = np.zeros(_SLOT_COUNT, dtype=np.uint32)
    table[0] = 5
    for expected in range(1, 6):
        record = _encode(
            seq,
            has_pull=True,
            nbytes=256,
            source_slot_id=0,
            slot_generation=int(table[0]),
        )
        entry = decode_task_record(
            record, expected_seq=expected, generation_table=table
        )
        assert entry.seq == expected
        seq = next_seq(seq)
    # The sixth consume expects seq 6; a replay of the previous record fails.
    replayed = _encode(5, has_pull=False)
    with pytest.raises(RdmaTransportError, match="stale"):
        decode_task_record(replayed, expected_seq=6, generation_table=table)
