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
  depth gate open at the GPU done rather than after the retire work.  Phase
  one has already popped every batch by then, so a failure record must
  quarantine the batches behind it too: nothing else still holds them;
- under DT_DRAFT_SCHEDULER=v2 (D10) the gate counts frozen_seq - done_seq
  and absorbs a batch done since ABSORB before FREEZE, and the late freeze
  waits for the running batch's GPU start + G - C (design §5.2).
"""

import collections
from types import SimpleNamespace

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


def test_absorb_quarantines_every_popped_batch_behind_a_failure():
    """A failure record stops the retire, but phase one already popped the
    batches behind it: they must be quarantined, not dropped on the floor
    with their KV and staging still owned."""

    engine = _engine()
    engine._fatal_gpu_quarantine_reason = None
    engine._profile_active = False
    engine._loop_stage_ns = collections.Counter()
    engine._loop_stage_calls = collections.Counter()
    engine._quarantined_dflash_completions = []
    engine._quarantined_feature_draft_handles = set()
    engine._publish_capacity = lambda: None
    failures = []
    engine._publish_failures = failures.append
    engine._enter_fatal_gpu_quarantine = lambda reason: setattr(
        engine, "_fatal_gpu_quarantine_reason", reason
    )
    engine._retire_dflash_completion = lambda *a, **k: pytest.fail(
        "a batch behind a send-loop failure was retired as if it were clean"
    )

    batches = []
    for seq in (0, 1):
        completion = draft_mod.PendingDFlashCompletion.__new__(
            draft_mod.PendingDFlashCompletion
        )
        completion.batch_seq = seq
        completion.pending = [
            SimpleNamespace(task=SimpleNamespace(draft_kv_handle=f"kv{seq}",
                                                 transfer_id=f"t{seq}"))
        ]
        batches.append(completion)
    engine._pending_dflash_completions = collections.deque(batches)
    # The failed batch, then a clean record for the batch behind it.
    records = np.array([[1, 0, 0, 0, 999], [2, 0, 0, 0, 0]], dtype=np.uint64)
    engine._send_loop_worker = type("W", (), {"drain_done": lambda self: records})()

    engine._absorb_dflash_done()

    assert not engine._pending_dflash_completions
    assert engine._quarantined_dflash_completions == batches
    assert engine._quarantined_feature_draft_handles == {"kv0", "kv1"}
    assert len(failures) == 2
    assert engine._fatal_gpu_quarantine_reason


def _gate_engine(done_seq, frozen_seq, batches, last_done_ns=0):
    engine = _engine()
    engine._fatal_gpu_quarantine_reason = None
    # READY-table row: (seq, ident, deadline_ns, requires_proposal, count, k,
    # row_first, drained_ns, fields, marks).
    engine._ready_pending = [(1, 0, 0, True, 40, 0, 0, 0, (), ())]  # 40 tokens READY
    engine._input_ready_features = collections.deque()
    engine.feature_buffer_max_tokens = 8192
    engine._batch_estimator = draft_mod.DraftBatchEstimator()
    engine._dflash_batch_seq = frozen_seq
    engine._dflash_last_done_ns = last_done_ns
    engine._send_loop_worker = type("W", (), {"done_seq": lambda self: done_seq})()
    engine._pending_dflash_completions = collections.deque(
        SimpleNamespace(
            batch_seq=seq, handoff_ns=handoff_ns,
            pending=[SimpleNamespace(task=SimpleNamespace(feature_token_count=40))],
        )
        for seq, handoff_ns in batches
    )
    return engine


def test_v2_gate_counts_in_flight_by_done_seq_and_absorbs_before_a_freeze():
    """DT_DRAFT_SCHEDULER=v2 (D10): in-flight is frozen_seq - done_seq.  A
    batch the send loop finished since ABSORB still sits in the deque; the
    gate must absorb it before letting FREEZE run, or FREEZE sees its owners
    as busy and the launch finds its per-batch scratch still owned."""

    engine = _gate_engine(done_seq=1, frozen_seq=3, batches=[(1, 0), (2, 0)])
    assert engine._dflash_gate(0) is None  # two in flight: wait for a DONE

    engine = _gate_engine(done_seq=3, frozen_seq=3, batches=[(1, 0), (2, 0)])
    absorbed = []
    engine._absorb_dflash_done = lambda: absorbed.append(
        engine._pending_dflash_completions.clear()
    )
    assert engine._dflash_gate(0) == 0  # none in flight: freeze now
    assert absorbed and not engine._pending_dflash_completions

    # The record of batch 3 was absorbed before the loop stored done_seq 3
    # (it publishes the record first): the counter still says one in
    # flight, the deque says none.  Seen on 7T2D r64 as an IndexError.
    engine = _gate_engine(done_seq=2, frozen_seq=3, batches=[])
    assert engine._dflash_gate(0) == 0


def test_v2_late_freeze_deadline():
    """With one batch running, the next freezes at its GPU start + G(its
    tokens) - C(READY tokens) (design §5.2).  Its GPU start is its hand-off,
    or the GPU done of the batch it queued behind; without an estimate, or
    once it ran past 2 G, the freeze is immediate."""

    # Queued behind a batch that finished at 1.0 ms: start is that done.
    engine = _gate_engine(2, 3, [(2, 200)], last_done_ns=1_000_000)
    assert engine._dflash_gate(1_200_000) == 0  # no estimate yet
    engine._batch_estimator.observe(rows=1, tokens=40, gpu_ns=3_000_000, cpu_ns=1_500_000)
    assert engine._dflash_gate(1_200_000) == 2_500_000
    assert engine._dflash_gate(2_500_000) == 0
    # Not queued: start is its own hand-off.
    engine = _gate_engine(2, 3, [(2, 5_000_000)], last_done_ns=1_000_000)
    engine._batch_estimator.observe(rows=1, tokens=40, gpu_ns=3_000_000, cpu_ns=1_000_000)
    assert engine._dflash_gate(5_100_000) == 7_000_000
    # The 2 G guard holds even when C would push the deadline past it.
    engine._batch_estimator._ema[("tokens", engine._batch_estimator.bucket(40), "c")] = -9e6
    assert engine._dflash_gate(5_000_000 + 6_000_001) == 0
