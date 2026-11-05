"""CPU contract for the cross-host candidate record (issue #223).

A cross-host Draft GPUDirect-writes its row and then the completion record
words built by ``completion_record_words`` into the Target's record mapping;
the Target's native record observer validates them and publishes them into
its READY table exactly like a same-host record, and an authenticated record
of a cross-host producer -- a row or a tombstone -- is the proof that retires
a retired-unaccepted round's source lease (``remote_record_listener``).
These tests drive the real native observer and pin that the Draft's words
authenticate, that exactly those records deliver the proof, and that stale
or forged records never reach READY or deliver one.

Run only after the task's external release/gate; imports can load heavy modules.
"""

import shutil
import time
from collections import OrderedDict, deque
from threading import Lock

import numpy as np
import pytest

from sglang.srt.speculative.dt_candidate_p2p import (
    COMPLETION_RECORD_WORDS,
    COMPLETION_SEQ_WORD,
    CandidateSlotState,
    CandidateTransportError,
    TargetCandidateArena,
    _CandidateSlot,
    _RecordObserver,
    completion_record_words,
    producer_tag,
    transfer_id_hash,
)

pytestmark = pytest.mark.skipif(
    shutil.which("g++") is None, reason="needs a C++ compiler for the observer"
)

_ROW_WIDTH = 16
_REMOTE = "remote-draft"
_LOCAL = "local-draft"


@pytest.fixture
def arena_and_proofs():
    proofs = []
    arena = object.__new__(TargetCandidateArena)
    arena.slot_count = 4
    arena.row_width = _ROW_WIDTH
    arena._lock = Lock()
    arena._slots = [_CandidateSlot() for _ in range(arena.slot_count)]
    arena._active = {}
    arena._retired_transfers = OrderedDict()
    arena._completion_records = object()
    arena._completion_words = np.zeros(
        (arena.slot_count, COMPLETION_RECORD_WORDS), dtype=np.uint64
    )
    arena._record_generation = np.zeros(arena.slot_count, dtype=np.uint32)
    arena._record_seq_expected = np.zeros(arena.slot_count, dtype=np.uint64)
    arena._record_hash_expected = np.zeros(arena.slot_count, dtype=np.uint64)
    arena._record_tag = np.zeros(arena.slot_count, dtype=np.uint32)
    arena._recent_notice_keys = deque()
    arena._recent_notice_set = set()
    arena._recent_notice_limit = 100
    arena._event_pool = []
    arena._remote_producers = {_REMOTE}
    arena.remote_record_listener = proofs.append
    arena._observer = _RecordObserver(
        slot_count=arena.slot_count,
        completion_words=arena._completion_words,
        ready_words=None,
        record_seq_expected=arena._record_seq_expected,
        record_generation=arena._record_generation,
        record_hash_expected=arena._record_hash_expected,
        record_tag=arena._record_tag,
        row_width=_ROW_WIDTH,
    )
    try:
        yield arena, proofs
    finally:
        arena._observer.destroy()


def _arm(arena, slot_id, transfer_id, session, *, generation=3, nonce=0x1234_5678_9ABC):
    slot = arena._slots[slot_id]
    slot.generation = generation
    slot.state = CandidateSlotState.ARMED
    slot.transfer_id = transfer_id
    slot.authorization_nonce = nonce
    slot.draft_instance_id = "d"
    slot.draft_session_id = session
    arena._active[transfer_id] = slot_id
    arena._record_hash_expected[slot_id] = transfer_id_hash(transfer_id)
    arena._record_generation[slot_id] = generation
    arena._record_seq_expected[slot_id] = nonce
    arena._record_tag[slot_id] = producer_tag(session)
    return generation, nonce


def _write(arena, slot_id, transfer_id, session, *, generation, nonce, valid_length=_ROW_WIDTH):
    words = completion_record_words(
        nonce=nonce,
        generation=generation,
        valid_length=valid_length,
        transfer_hash=transfer_id_hash(transfer_id),
        root_token=-5,
        producer_tag_value=producer_tag(session),
    )
    # The seq (nonce) word lands last, as on the wire.
    record = arena._completion_words[slot_id]
    for index, word in enumerate(words):
        if index != COMPLETION_SEQ_WORD:
            record[index] = word
    record[COMPLETION_SEQ_WORD] = words[COMPLETION_SEQ_WORD]


def _poll_until_applied(arena, deadline_s=10.0):
    """Poll until the native observer publishes one record (it needs a
    scheduling quantum to see it)."""

    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        if arena.poll_completion_records():
            return True
        time.sleep(0.001)
    return False


def _settle(arena, slot_id):
    """Arm and write a valid same-host record in ``slot_id`` and wait until
    it is applied: the observer scans every slot per pass, so by then it has
    seen every other slot's record at least once."""

    generation, nonce = _arm(arena, slot_id, f"sentinel-{slot_id}", _LOCAL, nonce=77)
    _write(arena, slot_id, f"sentinel-{slot_id}", _LOCAL, generation=generation, nonce=nonce)
    assert _poll_until_applied(arena)


def test_remote_row_authenticates_and_proves_its_source_read(arena_and_proofs):
    arena, proofs = arena_and_proofs
    generation, nonce = _arm(arena, 0, "t0", _REMOTE)
    _write(arena, 0, "t0", _REMOTE, generation=generation, nonce=nonce)
    assert _poll_until_applied(arena)
    slot = arena._slots[0]
    assert slot.state is CandidateSlotState.READY
    assert slot.completion.valid_length == _ROW_WIDTH
    assert slot.completion.root_token == -5
    assert proofs == ["t0"]


def test_remote_tombstone_retires_the_slot_and_proves_its_source_read(arena_and_proofs):
    arena, proofs = arena_and_proofs
    generation, nonce = _arm(arena, 1, "t1", _REMOTE)
    arena._slots[1].state = CandidateSlotState.QUARANTINED
    arena._slots[1].cancel_pending = True
    _write(arena, 1, "t1", _REMOTE, generation=generation, nonce=nonce, valid_length=0)
    assert _poll_until_applied(arena)
    assert arena._slots[1].state is CandidateSlotState.FREE
    assert proofs == ["t1"]


def test_same_host_record_delivers_no_remote_proof(arena_and_proofs):
    arena, proofs = arena_and_proofs
    _settle(arena, 2)
    assert arena._slots[2].state is CandidateSlotState.READY
    assert proofs == []


def test_late_record_of_a_recycled_slot_never_becomes_ready(arena_and_proofs):
    arena, proofs = arena_and_proofs
    generation, nonce = _arm(arena, 0, "t0", _REMOTE)
    # A late record of the slot's previous reservation (recycled slot).
    _write(arena, 0, "t-old", _REMOTE, generation=generation - 1, nonce=nonce + 1)
    _settle(arena, 3)
    assert arena.poll_completion_records() == 0
    assert arena._slots[0].state is CandidateSlotState.ARMED
    assert proofs == []


def test_forged_producer_is_quarantined_without_a_proof(arena_and_proofs):
    arena, proofs = arena_and_proofs
    generation, nonce = _arm(arena, 3, "t3", _REMOTE)
    _write(arena, 3, "t3", "someone-else", generation=generation, nonce=nonce)
    with pytest.raises(CandidateTransportError):
        _poll_until_applied(arena)
    assert arena._slots[3].state is CandidateSlotState.QUARANTINED
    assert proofs == []
