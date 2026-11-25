"""CPU contract: a direct candidate completion may land before its round parks
(issue #330).

With the destination written at launch (Target P2, #321), a same-host Draft
can push its candidates before the Target's park pass registers the round.
The arena's poll() returns a completion once; the Target must keep it and
apply it when the round parks -- dropping it starved the request until the
client aborted -- and must not apply it once its slot was let go.

These tests drive the real arena slot machine (legacy record scan, no
native observer) through the Target's export-rank poll.

Run only after the task's external release/gate; imports can load heavy modules.
"""

from collections import OrderedDict, deque
from threading import Lock
from types import SimpleNamespace

import numpy as np

from sglang.srt.speculative.dt_candidate_p2p import (
    COMPLETION_RECORD_WORDS,
    CandidateSlotState,
    TargetCandidateArena,
    _CandidateSlot,
    completion_record_words,
    producer_tag,
    transfer_id_hash,
)
from sglang.srt.speculative.dt_target import DTTargetController

_ROW_WIDTH = 16
_SESSION = "draft-session"
_TRANSFER = "target-0:req-0:3:1"
_ROOT = 7


def _arena():
    arena = object.__new__(TargetCandidateArena)
    arena.slot_count = 4
    arena.row_width = _ROW_WIDTH
    arena._lock = Lock()
    arena._slots = [_CandidateSlot() for _ in range(arena.slot_count)]
    arena._active = {}
    arena._poll_cursor = 0
    arena._ready_pickup_pending = False
    arena._scanned_slot_epoch = 0
    arena._release_scan = (-1, True)
    arena._cancel_scan = (-1, True)
    arena._retired_transfers = OrderedDict()
    arena._event_pool = []
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
    arena.poll_completion_notices = lambda: 0  # no notice socket here
    return arena


def _authorize_and_push(arena, slot_id=0, generation=3, nonce=0x1234_5678_9ABC):
    """The launch-time reserve + authorize, then the Draft's record landing."""

    slot = arena._slots[slot_id]
    slot.generation = generation
    slot.state = CandidateSlotState.ARMED
    slot.transfer_id = _TRANSFER
    slot.authorization_nonce = nonce
    slot.draft_instance_id = "draft-0"
    slot.draft_session_id = _SESSION
    arena._active[_TRANSFER] = slot_id
    arena._record_generation[slot_id] = generation
    arena._record_seq_expected[slot_id] = nonce
    arena._record_tag[slot_id] = producer_tag(_SESSION)
    arena._completion_words[slot_id] = completion_record_words(
        nonce=nonce,
        generation=generation,
        valid_length=_ROW_WIDTH,
        transfer_hash=transfer_id_hash(_TRANSFER),
        root_token=_ROOT,
        producer_tag_value=producer_tag(_SESSION),
    )


def _controller(arena):
    c = object.__new__(DTTargetController)
    c._candidate_arena = arena
    c._parked = {}
    c._unparked_completions = {}
    rounds = {}
    c._ledger = SimpleNamespace(
        get=rounds.get, mark_direct_ready=lambda *_args, **_kwargs: None
    )
    c._draft_progress_ns = {}
    c._profile_active = False
    c._ring_outstanding = {}
    c._client = SimpleNamespace(recv_nowait=lambda: ())
    c._flush_pending_send = lambda: None
    c._expire_local_rounds = lambda: []
    c._expire_source_leases = lambda **_kwargs: None
    c._maybe_emit_lifecycle_status = lambda: None
    return c, rounds


def _park(controller, rounds):
    rounds[_TRANSFER] = object()
    parked = SimpleNamespace(
        candidate_completion=None, bonus_token=_ROOT, seq_len=40,
        req=SimpleNamespace(rid="req-0"), round_id=3, result_received_ns=0,
    )
    controller._parked[_TRANSFER] = parked
    return parked


def _ready(events):
    return [payload for kind, _tid, payload in events if kind == "ready_direct"]


def test_completion_that_lands_before_park_is_applied_once_the_round_parks():
    arena = _arena()
    controller, rounds = _controller(arena)
    _authorize_and_push(arena)

    events, _ = controller._poll_export_rank()
    assert _ready(events) == []  # observed, but the round is not parked yet
    assert arena._slots[0].state is CandidateSlotState.READY
    assert _ready(controller._poll_export_rank()[0]) == []  # poll() returns it once

    parked = _park(controller, rounds)
    (payload,) = _ready(controller._poll_export_rank()[0])
    completion, bonus, seq_len = payload
    assert (completion.transfer_id, completion.root_token) == (_TRANSFER, _ROOT)
    assert (bonus, seq_len) == (parked.bonus_token, parked.seq_len)
    assert controller._unparked_completions == {}


def test_completion_whose_slot_is_let_go_before_park_is_never_applied():
    arena = _arena()
    controller, rounds = _controller(arena)
    _authorize_and_push(arena)
    assert _ready(controller._poll_export_rank()[0]) == []

    assert arena.cancel(_TRANSFER)  # e.g. the request ended first
    _park(controller, rounds)
    assert _ready(controller._poll_export_rank()[0]) == []
    assert controller._unparked_completions == {}
