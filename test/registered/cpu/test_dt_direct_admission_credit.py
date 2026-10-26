"""CPU contract for the cuda-ipc proposal-admission credit (issue #42).

The direct path no longer draws on the mmap result-slot pool; an explicit
credit counter (default 64) bounds proposal admission instead.  These tests
pin the accounting semantics: the counter charges exactly what the old pool
accounting charged, and every acquire is released exactly once (push
completion, held-push drop, cancel, or exception).
"""

from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.speculative.dt_draft_engine import FlashMTPDraftEngine


def _task(transfer_id, *, destination="dest", ring_seq=0):
    return SimpleNamespace(
        transfer_id=transfer_id,
        request_id=f"req-{transfer_id}",
        round_id=1,
        attempt=0,
        target_instance_id="target-0",
        requires_proposal=True,
        candidate_destination=destination,
        direct_ring_seq=ring_seq,
    )


def _ready_push(ready=False):
    event = SimpleNamespace(query=lambda: ready)
    return SimpleNamespace(
        event=event,
        start_event=None,
        candidate_bytes=16,
        cold_mapping_count=0,
        submitted_ns=1,
    )


class _Transport:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.push_stream = object()
        self.submitted = []
        self.published_completions = []

    def submit(self, *, tasks, candidates):
        if self.fail:
            raise RuntimeError("injected submit failure")
        push = _ready_push()
        self.submitted.append((list(tasks), push))
        return push

    def publish_completion(self, tasks, *, cancelled):
        self.published_completions.append((list(tasks), cancelled))


def _engine(*, limit=4, transport="cuda-ipc"):
    engine = FlashMTPDraftEngine.__new__(FlashMTPDraftEngine)
    engine.instance_id = "draft-0"
    engine.candidate_transport = transport
    engine.direct_proposal_admission_limit = limit
    engine._direct_proposal_admission_inflight = 0
    engine._is_feature_backed = True
    engine._persistent = True
    engine._profile_active = False
    engine._fatal_gpu_quarantine_reason = None
    engine._cancelled_ids = set()
    engine._pending_ids = set()
    engine._ring_pending_ids = set()
    engine._received_ns_by_transfer = {}
    engine._ring_record_positions = {}
    engine._ring_dest_region = False
    engine._early_pushed_ids = set()
    engine._unconfirmed_direct = {}
    engine._direct_tasks_by_transfer = {}
    engine._held_ring_pushes = []
    engine._pending_candidate_pushes = deque()
    engine._pending = deque()
    engine._ring_admission = deque()
    engine._pending_feature_batches = deque()
    engine._input_ready_features = deque()
    engine._pending_flashmtp_completions = deque()
    engine._pending_dflash_completions = deque()
    engine._pending_eagle3_completions = deque()
    engine._free_result_slots = deque(range(2))
    engine._result_sequence = 0
    engine._sent_results = []
    engine._send_result = engine._sent_results.append
    engine._publish_capacity = lambda: None
    engine.EntryStatus = SimpleNamespace(OK="OK", CANCELLED="CANCELLED")
    return engine


def test_availability_counts_reservations_and_inflight_waves():
    engine = _engine(limit=4)
    assert engine._available_direct_proposal_admission() == 4
    engine._input_ready_features.append(
        SimpleNamespace(pending=SimpleNamespace(task=_task("t1")))
    )
    assert engine._available_direct_proposal_admission() == 3
    engine._pending_feature_batches.append(
        SimpleNamespace(pending=[SimpleNamespace(task=_task("t2"))])
    )
    assert engine._available_direct_proposal_admission() == 2
    engine._pending_dflash_completions.append(
        SimpleNamespace(proposal_tasks=(_task("t3"),))
    )
    engine._pending_eagle3_completions.append(
        SimpleNamespace(candidates=torch.zeros(1))
    )
    assert engine._available_direct_proposal_admission() == 0
    assert engine._proposal_admission_blocked()
    # A held in-flight wave also blocks; reservations alone clamp at zero.
    engine._acquire_direct_proposal_admission()
    assert engine._available_direct_proposal_admission() == 0
    engine._release_direct_proposal_admission()
    assert engine._available_direct_proposal_admission() == 0
    engine._pending_dflash_completions.pop()
    assert not engine._proposal_admission_blocked()


def test_publish_wave_credit_retires_with_gpu_push_completion():
    engine = _engine()
    engine._candidate_push_transport = _Transport()
    engine._publish_successes([_task("t1")], torch.zeros((1, 2), dtype=torch.int32))
    assert engine._direct_proposal_admission_inflight == 1
    pending = engine._pending_candidate_pushes[0]
    assert pending.result_slot is None
    assert pending.admission_credit
    # The mmap pool is untouched on the direct path.
    assert len(engine._free_result_slots) == 2

    engine._poll_candidate_pushes()  # event not ready: still held
    assert engine._direct_proposal_admission_inflight == 1

    pending.push.event = SimpleNamespace(query=lambda: True)
    engine._poll_candidate_pushes()
    assert engine._direct_proposal_admission_inflight == 0
    assert not engine._pending_candidate_pushes


def test_failed_push_submit_releases_the_credit():
    engine = _engine()
    engine._candidate_push_transport = _Transport(fail=True)
    with pytest.raises(RuntimeError, match="injected submit failure"):
        engine._publish_successes(
            [_task("t1")], torch.zeros((1, 2), dtype=torch.int32)
        )
    assert engine._direct_proposal_admission_inflight == 0
    assert not engine._pending_candidate_pushes


def test_held_ring_push_drop_releases_the_credit():
    engine = _engine()
    engine._candidate_push_transport = _Transport()
    # §7.1 record task without a destination yet: the wave is held.
    engine._publish_successes(
        [_task("t1", destination=None, ring_seq=7)],
        torch.zeros((1, 2), dtype=torch.int32),
    )
    assert engine._direct_proposal_admission_inflight == 1
    assert len(engine._held_ring_pushes) == 1
    assert not engine._pending_candidate_pushes

    # Every task cancelled while held: the credit retires like a failed submit.
    engine._cancelled_ids.add("t1")
    engine._release_held_ring_pushes()
    assert engine._direct_proposal_admission_inflight == 0
    assert not engine._held_ring_pushes
    assert not engine._pending_candidate_pushes


def test_held_ring_push_submit_moves_the_credit_to_the_push(monkeypatch):
    # The held-release path gathers rows on the push stream; stub the two CUDA
    # touchpoints so the credit ownership transfer runs on CPU.
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    engine = _engine()
    engine._candidate_push_transport = _Transport()
    engine._publish_successes(
        [_task("t1", destination=None, ring_seq=7)],
        torch.zeros((1, 2), dtype=torch.int32),
    )
    assert len(engine._held_ring_pushes) == 1

    # Confirmation arrived: the task now has a destination, so the held wave
    # submits and the push owns the credit until its completion retires.
    engine._direct_tasks_by_transfer["t1"] = _task("t1")
    engine._release_held_ring_pushes()
    assert not engine._held_ring_pushes
    assert engine._direct_proposal_admission_inflight == 1
    pending = engine._pending_candidate_pushes[0]
    assert pending.admission_credit
    pending.push.event = SimpleNamespace(query=lambda: True)
    engine._poll_candidate_pushes()
    assert engine._direct_proposal_admission_inflight == 0


def test_mmap_paths_still_use_the_result_pool():
    engine = _engine(transport="mmap")
    engine._free_result_slots = deque()
    assert engine._proposal_wave_buffer_exhausted()
    assert engine._proposal_admission_blocked()
    engine._free_result_slots.append(0)
    assert not engine._proposal_wave_buffer_exhausted()
    assert not engine._proposal_admission_blocked()
    # mmap gates never consult the direct credit.
    engine._direct_proposal_admission_inflight = 99
    assert not engine._proposal_admission_blocked()
