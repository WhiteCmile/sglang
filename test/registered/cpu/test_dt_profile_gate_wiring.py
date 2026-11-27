"""The per-event profiling gates keep producers, consumers and drains together.

Issue #362 armed every expensive profiling path behind the events that
consume its output; issue #366 fixes the three combinations a partial
DT_PROFILE_EVENTS set then broke.  The contract a later gate change is most
likely to break: for every gated producer, enabling ANY single one of its
consumer events must arm the producer, and the drain of the producer's
records must run whenever the producer does.  These tests snapshot the
gates per single-event filter (the wiring) and run the two record paths
end to end on bare instances (the drains):

- the push transport collects completion-ACK records whenever any
  push-timing consumer is on; _push_transport_progress must drain them even
  when draft_candidate_notice_ack itself is filtered out, or the deque
  grows without bound;
- the FREEZE skip accounting must run when only draft_ready_skip is on, or
  no draft_ready_skip record is ever emitted;
- the submission record must be taken when only
  target_verify_cuda_observation_error is on, or the error diagnostics go
  silent.
"""

import collections
from types import SimpleNamespace

import pytest

import sglang.srt.speculative.dt_draft_engine as draft_mod
import sglang.srt.speculative.dt_target as target_mod

# Producer flag -> every event whose enablement alone must arm it.  A gate
# edit that drops one of these consumers starves the consumer or orphans
# the producer's records; that is the regression this file exists to catch.
DRAFT_PRODUCER_CONSUMERS = {
    "_profile_push_timing": (
        "draft_candidate_push_submit",
        "draft_candidate_producer_completion_observed",
        "draft_candidate_push_complete",
        "draft_candidate_notice_ack",
    ),
    "_profile_ready_skip_accounting": (
        "draft_batch_estimate",
        "draft_ready_skip",
    ),
}
TARGET_PRODUCER_CONSUMERS = {
    "target_submission_profile_enabled": (
        "target_batch_submit",
        "target_verify_cuda_observed",
        "target_verify_cuda_observation_error",
    ),
    "_profile_verify_observation": (
        "target_verify_cuda_observed",
        "target_verify_cuda_observation_error",
    ),
}

DRAFT_GATE_FLAGS = (
    "_profile_batch_estimate",
    "_profile_batch_complete",
    "_profile_input_ready",
    "_profile_ring_take",
    "_profile_input_ready_submit",
    "_profile_input_ready_select",
    "_profile_ready_state_event",
    "_profile_kv_or_proposal",
    "_profile_feature_pull_complete",
    "_profile_done_absorb",
    "_profile_candidate_push_submit",
    "_profile_candidate_push_retire",
    "_profile_notice_ack",
    "_profile_result_publish",
    "_profile_ready_skip",
    "_profile_push_timing",
    "_profile_ready_skip_accounting",
)
TARGET_GATE_FLAGS = (
    "target_loop_profile_enabled",
    "target_submission_profile_enabled",
    "target_phase_profile_enabled",
    "_profile_verify_observation",
    "_profile_park_batch",
    "_profile_maintenance",
    "_profile_poll_splits",
    "_profile_tp_apply",
    "_profile_capacity_gauge",
    "_profile_rdma_sender_trace",
    "_profile_ready_snapshot",
    "empty_prefill_arbitration_profile_enabled",
)


def _only(enabled_events):
    return lambda event: event in enabled_events


def _draft_gates(monkeypatch, enabled_events):
    monkeypatch.setattr(draft_mod, "profile_event_enabled", _only(enabled_events))
    engine = object.__new__(draft_mod.FlashMTPDraftEngine)
    engine._snapshot_profile_gates()
    return engine


def _target_gates(monkeypatch, enabled_events, *, is_export_rank=True):
    monkeypatch.setattr(target_mod, "profile_event_enabled", _only(enabled_events))
    controller = object.__new__(target_mod.DTTargetController)
    controller.is_export_rank = is_export_rank
    controller._snapshot_profile_gates()
    return controller


def test_every_event_on_arms_every_draft_gate(monkeypatch):
    # event_enabled maps an empty DT_PROFILE_EVENTS to all-on; the wiring
    # must not drop a gate when the predicate says yes to everything.
    engine = _draft_gates(
        monkeypatch,
        {
            "draft_batch_estimate",
            "draft_dflash_batch_complete",
            "draft_input_ready",
            "draft_ring_take",
            "draft_input_ready_submit",
            "draft_input_ready_select",
            "draft_input_ready_state",
            "draft_kv_materialization",
            "draft_proposal",
            "draft_feature_pull_complete",
            "draft_dflash_done_absorb",
            "draft_candidate_push_submit",
            "draft_candidate_producer_completion_observed",
            "draft_candidate_push_complete",
            "draft_candidate_notice_ack",
            "draft_result_publish",
            "draft_ready_skip",
        },
    )
    for flag in DRAFT_GATE_FLAGS:
        assert getattr(engine, flag), flag


def test_nothing_enabled_disarms_every_draft_gate(monkeypatch):
    engine = _draft_gates(monkeypatch, set())
    for flag in DRAFT_GATE_FLAGS:
        assert not getattr(engine, flag), flag


@pytest.mark.parametrize(
    "flag,event",
    [
        (flag, event)
        for flag, consumers in DRAFT_PRODUCER_CONSUMERS.items()
        for event in consumers
    ],
)
def test_each_draft_consumer_arms_its_producer(monkeypatch, flag, event):
    engine = _draft_gates(monkeypatch, {event})
    assert getattr(engine, flag), f"{event} must arm {flag}"


def test_nothing_enabled_disarms_every_target_gate(monkeypatch):
    controller = _target_gates(monkeypatch, set())
    for flag in TARGET_GATE_FLAGS:
        assert not getattr(controller, flag), flag


@pytest.mark.parametrize(
    "flag,event",
    [
        (flag, event)
        for flag, consumers in TARGET_PRODUCER_CONSUMERS.items()
        for event in consumers
    ],
)
def test_each_target_consumer_arms_its_producer(monkeypatch, flag, event):
    controller = _target_gates(monkeypatch, {event})
    assert getattr(controller, flag), f"{event} must arm {flag}"


def test_non_export_rank_never_arms_rank_gated_producers(monkeypatch):
    controller = _target_gates(
        monkeypatch,
        {"target_batch_submit", "target_verify_cuda_observation_error"},
        is_export_rank=False,
    )
    assert not controller.target_submission_profile_enabled
    assert not controller._profile_verify_observation


class _FakePushTransport:
    def __init__(self, acknowledgements):
        self.acknowledgements = list(acknowledgements)
        self.pops = 0

    def progress_completions(self):
        pass

    def pop_acknowledgements(self):
        self.pops += 1
        drained, self.acknowledgements = self.acknowledgements, []
        return tuple(drained)


def _push_engine(acknowledgements, *, push_timing, notice_ack):
    engine = object.__new__(draft_mod.FlashMTPDraftEngine)
    engine.instance_id = "draft-0"
    engine._candidate_push_transport = _FakePushTransport(acknowledgements)
    engine._profile_push_timing = push_timing
    engine._profile_notice_ack = notice_ack
    return engine


def test_ack_records_drain_whenever_the_producer_runs(monkeypatch):
    emitted = []
    monkeypatch.setattr(
        draft_mod, "profile_emit", lambda event, /, **fields: emitted.append(event)
    )
    ack = {
        "transfer_id": "t1",
        "notice_queued_ns": 1,
        "target_accepted_ns": 2,
        "ack_observed_ns": 3,
    }

    # Push timing on for push_submit/push_complete only: the records must
    # still be drained, just not emitted.
    engine = _push_engine([ack], push_timing=True, notice_ack=False)
    engine._push_transport_progress()
    assert engine._candidate_push_transport.acknowledgements == []
    assert emitted == []

    # The ack event itself on: drained AND emitted.
    engine = _push_engine([ack], push_timing=True, notice_ack=True)
    engine._push_transport_progress()
    assert engine._candidate_push_transport.acknowledgements == []
    assert emitted == ["draft_candidate_notice_ack"]

    # Producer off: the transport collected nothing, so there is no drain.
    engine = _push_engine([], push_timing=False, notice_ack=False)
    engine._push_transport_progress()
    assert engine._candidate_push_transport.pops == 0


def test_observation_error_alone_still_records_the_submission(monkeypatch):
    # p1-3 of the #362 review: with only target_verify_cuda_observation_error
    # enabled the submission gate was off, so the missing-events diagnostic
    # could never fire.
    controller = _target_gates(monkeypatch, {"target_verify_cuda_observation_error"})
    controller._pending_target_gpu_observations = collections.deque()
    controller._target_compute_batches_inflight = 0
    emitted = []
    controller._profile_emit = lambda event, /, **fields: emitted.append(
        (event, fields)
    )
    batch = SimpleNamespace(
        dt_full_wave_id=7,
        dt_full_wave_transfer_ids=("t1",),
        dt_full_wave_ready_ns=10,
        dt_update_running_ns=0,
        reqs=[object()],
    )
    batch_result = SimpleNamespace(dt_target_verify_gpu_events=None)

    controller.record_target_batch_submission(
        batch,
        batch_result,
        selection_started_ns=1,
        selection_finished_ns=2,
        submission_started_ns=3,
        submission_returned_ns=4,
        result_queue_depth_before=0,
        iteration_started_ns=0,
        receive_started_ns=0,
        receive_finished_ns=0,
        poll_started_ns=0,
        poll_finished_ns=0,
    )

    errors = [
        fields
        for event, fields in emitted
        if event == "target_verify_cuda_observation_error"
    ]
    assert len(errors) == 1
    assert errors[0]["reason"] == "missing_events"
    assert errors[0]["wave_id"] == 7
