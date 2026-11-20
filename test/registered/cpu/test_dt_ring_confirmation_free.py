"""CPU contracts of the Draft's confirmation-free mode (DT_RING_CONFIRMATION=0).

The per-PR GPU gate runs the default mode, so these pin the two flag-on
contracts nothing else checks before the default flips (design §11 D11):

- a held v2 wave push is released by re-reading the ring destination region,
  never by a Coordinator copy (design §7.3);
- ring rounds send the Coordinator no result and leave the cumulative
  capacity pair, which the Coordinator checks for consistency, relayed-only
  (design §8.3).
"""

import contextlib
from types import SimpleNamespace

import torch
from dt_protocol import (
    PROTOCOL_VERSION,
    DraftExecutionResultBatch,
    DraftResultEntry,
    DraftTask,
    EntryStatus,
)

import sglang.srt.speculative.dt_draft_engine as draft_mod


class Rows(torch.Tensor):
    def record_stream(self, _stream):
        pass


def _task(transfer_id, deadline_ns):
    return DraftTask(
        transfer_id=transfer_id,
        request_id=transfer_id,
        round_id=1,
        attempt=0,
        target_instance_id="target",
        draft_profile_id="p",
        k=3,
        next_token_id=1,
        source=None,
        created_ns=0,
        deadline_ns=deadline_ns,
        direct_ring_seq=7,
    )


def _engine():
    engine = object.__new__(draft_mod.FlashMTPDraftEngine)
    engine._ring_confirmation = False
    engine._cancelled_ids = set()
    engine._unconfirmed_direct = {}
    return engine


def test_held_push_resolves_from_region_without_coordinator(monkeypatch):
    monkeypatch.setattr(draft_mod.torch.cuda, "stream", contextlib.nullcontext)
    future = draft_mod.time.monotonic_ns() + 60 * 10**9
    live, zombie, late = "t:live:1:0", "t:zombie:1:0", "t:late:1:0"
    tasks = [_task(live, future), _task(zombie, future), _task(late, 0)]
    region = {0: None, 1: (0, 0, 0, 0, 0), 2: None}
    endpoint = SimpleNamespace(destination=lambda _seq, position: region[position])
    engine = _engine()
    engine._direct_tasks_by_transfer = {task.transfer_id: task for task in tasks}
    engine._ring_record_positions = {
        task.transfer_id: (endpoint, index) for index, task in enumerate(tasks)
    }
    engine._pending_ids = {live, zombie, late}
    engine._ring_pending_ids = {live, zombie, late}
    engine._received_ns_by_transfer = {}
    engine._ring_destination_ref = lambda _task, resolved: (
        None if resolved[:3] == (0, 0, 0) else ("arena", resolved[:3])
    )
    engine._candidate_push_transport = SimpleNamespace(push_stream=None)
    submitted = []
    engine._submit_candidate_push = lambda tasks, rows, *, result_slot: submitted.append(
        (tasks, rows.tolist(), result_slot)
    )
    candidates = torch.arange(12, dtype=torch.int32).reshape(3, 4).as_subclass(Rows)
    engine._held_ring_pushes = [(tasks, candidates, None)]

    # The live row's region is not written yet: the tombstoned row and the
    # row past its deadline are cancelled, the wave stays held for the live
    # row, and nothing is pushed or retired yet.
    engine._resolve_held_ring_destinations()
    assert engine._cancelled_ids == {zombie, late}
    assert len(engine._held_ring_pushes) == 1
    assert submitted == []

    # The Target writes the live row's destination: the next pass adopts it
    # and pushes only that row, with no confirmation having arrived.
    region[0] = (3, 9, 123, 4096, 2048)
    engine._resolve_held_ring_destinations()
    assert engine._held_ring_pushes == []
    [(pushed, rows, slot)] = submitted
    assert [task.transfer_id for task in pushed] == [live]
    assert pushed[0].candidate_destination == ("arena", (3, 9, 123))
    assert rows == [[0, 1, 2, 3]] and slot is None
    # The cancelled rows retired through the release path.
    assert engine._pending_ids == engine._ring_pending_ids == {live}
    assert not engine._cancelled_ids
    assert set(engine._ring_record_positions) == {live}


def test_ring_rounds_send_no_result_and_count_relayed_only():
    ring, relayed = "t:ring:1:0", "t:relayed:0:0"
    engine = _engine()
    engine.instance_id = "draft"
    engine.candidate_transport = "cuda-ipc"
    engine.max_pending_tasks = 64
    engine.max_batch_size = 8
    engine._stop = False
    engine._pending_ids = {ring, relayed}
    engine._ring_pending_ids = {ring}
    engine._received_tasks = 1  # the relayed round only
    engine._capacity_sequence = 0
    engine._prepared_migrations = {}
    engine._persistent_requests = {}
    engine._draft_model_runner = None
    engine._last_queue_delay_ns = engine._last_batch_size = 0
    engine._available_direct_proposal_admission = lambda: 1
    engine._maybe_log_draft_stats = lambda _pending: None
    updates, sent = [], []
    engine._client = SimpleNamespace(update_capacity=updates.append)
    engine._delivery_outbox = SimpleNamespace(
        add=lambda _op, _id, message: sent.append(message)
    )
    engine._flush_reliable_outbox = lambda: None

    engine._publish_capacity()
    assert (updates[-1].received_tasks, updates[-1].pending_tasks) == (1, 1)
    assert updates[-1].pending_work == 2
    assert updates[-1].ring_pending_tasks == 1

    def entry(transfer_id):
        return DraftResultEntry(
            transfer_id=transfer_id,
            request_id=transfer_id,
            round_id=1,
            attempt=0,
            target_instance_id="target",
            status=EntryStatus.OK,
        )

    engine._send_result(
        DraftExecutionResultBatch(
            protocol_version=PROTOCOL_VERSION,
            batch_id="b",
            draft_instance_id="draft",
            entries=(entry(ring), entry(relayed)),
            sent_ns=0,
        )
    )
    # Only the relayed entry reaches the Coordinator; the ring round retired
    # at once, as its ACK would have.
    [delivered] = sent
    assert [e.transfer_id for e in delivered.entries] == [relayed]
    assert engine._pending_ids == {relayed} and not engine._ring_pending_ids

    engine._publish_capacity()
    # retired = received - pending stays monotonic: the ring round never
    # entered either side of the cumulative pair.
    assert (updates[-1].received_tasks, updates[-1].pending_tasks) == (1, 1)
    assert updates[-1].pending_work == 1
