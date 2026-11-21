"""CPU contracts of the Draft send/completion loop (DT_DRAFT_SEND_LOOP, D8).

The per-PR GPU gate runs the default mode, so these pin the two hand-over
rules nothing else checks, and that the native loop cannot exercise without
a GPU (design §4.3, §5.1, §7.2):

- a landing placement is released exactly once, with the last READY row of
  its record, however those rows are split across batches.  The send loop
  pushes that release to ingress at the batch's GPU done, so releasing it
  early frees a buffer the GPU still reads, and releasing it late (or
  twice) stalls or corrupts ingress placement;
- ABSORB takes the DONE records in FIFO order and drops the batch from the
  in-flight deque *before* its Python bookkeeping, which is what makes the
  depth gate open at the GPU done rather than after the retire work.
"""

import collections

import numpy as np
import pytest

import sglang.srt.speculative.dt_draft_engine as draft_mod


def _engine():
    engine = object.__new__(draft_mod.FlashMTPDraftEngine)
    engine._ring_worker = None
    engine._ready_records = {}
    return engine


def test_landing_placement_is_released_with_the_records_last_row():
    engine = _engine()
    # One record of three rows in landing buffer 2 (generation 5), and one
    # single-row record in buffer 3; rows are keyed by (ring id, record seq).
    engine._ready_records = {(0, 11): [3, 2, 5], (0, 12): [1, 3, 9]}

    # A batch that took two of the three rows frees nothing yet.
    assert engine._resolve_ready_releases([(0, 11), (0, 11)]) == ()
    assert engine._ready_records[(0, 11)][0] == 1

    # The next batch takes the last row of one record and the whole other.
    assert engine._resolve_ready_releases([(0, 11), (0, 12)]) == ((2, 5), (3, 9))
    assert engine._ready_records == {}


def test_absorb_takes_done_records_in_order_and_opens_the_gate_first():
    engine = _engine()
    engine._fatal_gpu_quarantine_reason = None
    engine._profile_active = False
    engine._loop_stage_ns = collections.Counter()
    engine._loop_stage_calls = collections.Counter()
    engine._ready_release_fenced = collections.deque()
    engine._publish_capacity = lambda: None

    fence = object()
    engine._ready_release_fenced.append((fence, ()))
    batches = [
        draft_mod.PendingDFlashCompletion(
            pending=[],
            staging_slots=(),
            staging_slot_counts=(),
            completion_event=None,
            materialize_start_event=None,
            materialize_done_event=None,
            proposal_done_event=None,
            proposal_tasks=(),
            candidates=None,
            candidate_rows_cpu=None,
            launch_started_ns=0,
            tasks_prepared_ns=0,
            proposal_metadata_ready_ns=0,
            materialize_submitted_ns=0,
            launched_ns=0,
            proposal_execution=None,
            proposal_fallback_reason=None,
            reconcile_prepare_phase_ns=(),
            prior_unfinished_proposal_count=0,
            prior_candidate_copy_pending_count=0,
            batch_seq=seq,
        )
        for seq in (0, 1)
    ]
    engine._pending_dflash_completions = collections.deque(batches)

    retired = []
    engine._retire_dflash_completion = lambda completion, **stamps: retired.append(
        (completion.batch_seq, stamps, len(engine._pending_dflash_completions))
    )
    # A release-only fence (seq 0), then the two batches in submit order.
    records = np.array(
        [[0, 0, 0, 0, 0], [1, 7_000, 300_000, 3_100_000, 0], [2, 9_000, 310_000, 3_200_000, 0]],
        dtype=np.uint64,
    )
    engine._send_loop_worker = type("W", (), {"drain_done": lambda self: records})()

    engine._absorb_dflash_done()

    # The fence event is let go only by its own record, and both batches left
    # the in-flight deque before the first retire ran: the gate is already
    # open while the bookkeeping of a completed batch is still pending.
    assert not engine._ready_release_fenced
    assert [seq for seq, _, _ in retired] == [0, 1]
    assert [inflight for _, _, inflight in retired] == [0, 0]
    # The GPU stamp and the CUDA elapsed come from the record, not from this
    # thread's clock or an elapsed_time() call on the events.
    assert retired[0][1]["completed_ns"] == 7_000
    assert retired[0][1]["gpu_materialization_ns"] == 300_000
    assert retired[0][1]["proposal_gpu_ns"] == 3_100_000


def test_absorb_fails_stop_on_a_done_record_out_of_order():
    engine = _engine()
    engine._fatal_gpu_quarantine_reason = None
    engine._profile_active = False
    engine._loop_stage_ns = collections.Counter()
    engine._loop_stage_calls = collections.Counter()
    engine._pending_dflash_completions = collections.deque(
        [draft_mod.PendingDFlashCompletion.__new__(draft_mod.PendingDFlashCompletion)]
    )
    engine._pending_dflash_completions[0].batch_seq = 4
    records = np.array([[9, 0, 0, 0, 0]], dtype=np.uint64)
    engine._send_loop_worker = type("W", (), {"drain_done": lambda self: records})()

    with pytest.raises(RuntimeError, match="oldest in-flight batch"):
        engine._absorb_dflash_done()
