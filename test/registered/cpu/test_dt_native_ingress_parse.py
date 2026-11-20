"""CPU contract of the native ingress v2 parse (Draft redesign D6, issue #275).

The ring worker turns each kernel-published v2 record into per-row READY
entries (design §6.2).  It computes the completion-record transfer hash
natively, so a change to the transfer-id format, to the hash, or to the
record/request-table layout on either side must break here: the hash is
checked against ``dt_candidate_p2p.transfer_id_hash`` over the Python
transfer id (``dt_draft_engine._accept_ring_record``).  The test runs the
worker's own resolution and parse through ``dt_ring_parse_v2_record`` (no
CUDA).
"""

import ctypes
import struct

import numpy as np
import pytest

from sglang.srt.speculative.dt_candidate_p2p import transfer_id_hash
from sglang.srt.speculative.dt_direct_ring import (
    ENTRY_BYTES,
    V2_DEST_ENTRY_BYTES,
    V2_DEST_SEQ_OFFSET,
    V2_DEST_STRIDE,
    V2_HEADER_BYTES,
    V2_REQ_ENTRY_BYTES,
    V2_TOKEN_OFFSET,
    V2_VERSION,
    NativeRidLedger,
    _build_worker_library,
    ready_abi,
    rid_hash,
)

_TARGET = "target-instance-7f3c"
_SEQ = 0x1_0000_0005  # above 2**32: the record seq is the full u64
_ROW0 = 96  # landing row of the record's first row
# rid, round, attempt, token_offset, token_count, bonus, seq_len.  The last
# rid makes the transfer id longer than one 128-byte BLAKE2b block.
_REQS = [
    ("req-a", 3, 0, 0, 2, 11, 40),
    ("req-b", 9, 1, 4, 1, 12, 17),
    ("req-c-" + "x" * 140, 1, 2, 8, 3, 13, 3),
]
_TOKENS = [101, 102, 201, 301, 302, 303]


@pytest.fixture(scope="module")
def native():
    lib = ctypes.CDLL(str(_build_worker_library()))
    dtype, constants = ready_abi(lib)
    lib.dt_ring_parse_v2_record.argtypes = [
        ctypes.c_char_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p,
        ctypes.c_uint64, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.dt_ring_parse_v2_record.restype = ctypes.c_int
    ledger = NativeRidLedger(capacity=64)
    for rid, *_ in _REQS:
        ledger.register(rid)
    ledger.evict("req-b")  # released: its row is dropped

    def parse(record, dest=None):
        out = np.zeros(8, dtype=dtype)
        tokens = np.zeros(512, dtype=np.uint32)
        count = lib.dt_ring_parse_v2_record(
            record, ledger._handle, _TARGET.encode(), dest, V2_DEST_SEQ_OFFSET, 4, _ROW0,
            out.ctypes.data, tokens.ctypes.data,
        )
        return count, out[: max(count, 0)], tokens

    yield parse, constants, ledger
    ledger.close()


def _record(reqs=_REQS):
    record = bytearray(ENTRY_BYTES)
    struct.pack_into(
        "<QIIQIIQQIIIIQQ", record, 0, _SEQ, 1, V2_VERSION, 0, 0, 1, 0, 12 * 256, 12, 256,
        len(reqs), 4, 0, ~_SEQ & (2**64 - 1),
    )
    for i, (rid, *scalars) in enumerate(reqs):
        struct.pack_into("<Q6I", record, V2_HEADER_BYTES + i * V2_REQ_ENTRY_BYTES,
                         rid_hash(rid), *scalars)
    struct.pack_into(f"<{len(_TOKENS)}I", record, V2_TOKEN_OFFSET, *_TOKENS)
    return bytes(record)


def _dest(triples):
    region = bytearray(V2_DEST_STRIDE)
    for position, triple in enumerate(triples):
        struct.pack_into("<IIQ", region, position * V2_DEST_ENTRY_BYTES, *triple)
    struct.pack_into("<Q", region, V2_DEST_SEQ_OFFSET, _SEQ)
    return ctypes.create_string_buffer(bytes(region), len(region))


def _check_row(entry, req, prefix, tokens, ledger, constants):
    rid, round_id, attempt, token_offset, token_count, bonus, seq_len = req
    assert entry["transfer_hash"] == transfer_id_hash(f"{_TARGET}:{rid}:{round_id}:{attempt}")
    assert (entry["round_id"], entry["attempt"], entry["seq_len"], entry["bonus_token"]) == (
        round_id, attempt, seq_len, bonus,
    )
    assert (entry["ring_id"], entry["record_seq"], entry["landing_k"]) == (4, _SEQ, 0)
    assert entry["row_first"] == _ROW0 + token_offset
    assert entry["copy_event_idx"] == constants["NO_EVENT"]
    at = int(entry["token_at"])
    assert entry["token_count"] == token_count
    assert tokens[at : at + token_count].tolist() == _TOKENS[prefix : prefix + token_count]
    assert (entry["req_handle"], entry["req_gen"]) == ledger.handle(rid)


def test_live_rows_hash_and_fields_match_python(native):
    parse, constants, ledger = native
    count, entries, tokens = parse(_record())
    # req-b is released and has no known destination: no entry.
    assert count == 2
    assert entries["kind"].tolist() == [constants["ROW"]] * 2
    assert entries["position"].tolist() == [0, 2]
    _check_row(entries[0], _REQS[0], 0, tokens, ledger, constants)
    # req-b's token id still occupies its compacted position.
    _check_row(entries[1], _REQS[2], 3, tokens, ledger, constants)


def test_destination_capture_and_dropped_rows(native):
    parse, constants, ledger = native
    count, entries, tokens = parse(_record(), _dest([(5, 6, 7), (8, 9, 10), (0, 0, 0)]))
    known = constants["DEST_KNOWN"]
    assert entries["kind"].tolist() == [
        constants["ROW"] | known, constants["DROPPED"] | known, constants["ROW"] | known,
    ]
    triples = [tuple(int(e[f]) for f in ("dest_slot", "dest_gen", "dest_nonce")) for e in entries]
    # The zero triple is the Target's tombstone of a live row: passed through.
    assert triples == [(5, 6, 7), (8, 9, 10), (0, 0, 0)]
    dropped = entries[1]
    assert dropped["transfer_hash"] == transfer_id_hash(f"{_TARGET}:req-b:9:1")
    assert dropped["token_count"] == 0
    # A stale validity word (a later lap's region) is "not written yet".
    stale = _dest([(5, 6, 7), (8, 9, 10), (0, 0, 0)])
    struct.pack_into("<Q", stale, V2_DEST_SEQ_OFFSET, _SEQ + 256)
    count, entries, _ = parse(_record(), stale)
    assert count == 2 and not (entries["kind"] & known).any()


def test_malformed_and_unregistered_requests(native):
    parse, _, _ = native
    assert parse(_record([("req-a", 0, 0, 0, 2, 11, 40)]))[0] == -4  # round 0
    assert parse(_record([("req-a", 3, 0, 0, 2, 11, 1)]))[0] == -4  # seq_len < tokens
    assert parse(_record([("req-a", 3, 0, 11, 2, 11, 40)]))[0] == -4  # past row_count
    assert parse(_record([("req-unknown", 3, 0, 0, 2, 11, 40)]))[0] == 0  # retried
