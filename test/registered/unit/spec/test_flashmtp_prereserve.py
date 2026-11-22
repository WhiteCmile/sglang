"""Focused CPU tests for FlashMTP destination pre-reserve (issue #281).

Launch staging only arms export-rank arena rows; authorization, ledger begin
and Coordinator submission stay at the ordinary park boundary. These tests
drive the real controller methods with lightweight fakes, so they need no GPU,
RDMA, distributed init, model weights or controller constructor.
"""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

_PROTOCOL_NAMES = (
    "ControlAck", "DraftTask", "DraftTaskKind", "ErrorMessage", "SourceReadDone",
    "SourceReadDoneBatch", "SourceRef", "TargetResultBatch", "TargetTaskBatch",
    "TaskBatchAck",
)


def _load_dt_target():
    """Import dt_target with the external dt-runtime modules stubbed out."""
    dt_profile = types.ModuleType("dt_profile")
    dt_profile.event_enabled = lambda *args, **kwargs: False
    dt_profile.explicit_event_enabled = lambda *args, **kwargs: False
    dt_protocol = types.ModuleType("dt_protocol")
    dt_protocol.PROTOCOL_VERSION = 1
    for name in _PROTOCOL_NAMES:
        setattr(dt_protocol, name, type(name, (), {}))
    with mock.patch.dict(
        sys.modules, {"dt_profile": dt_profile, "dt_protocol": dt_protocol}
    ):
        import sglang.srt.speculative.dt_target as dt_target
    return dt_target

class _FakeArena:
    def __init__(self, *, free_slots, fail_reserve_at=None, fail_cancel_for=()):
        self.free_slot_count = free_slots
        self.reserve_calls = []
        self.cancel_calls = []
        self.forbidden_calls = []
        self._fail_reserve_at = fail_reserve_at
        self._fail_cancel_for = set(fail_cancel_for)

    def __getattr__(self, name):
        def _forbidden(*args, **kwargs):
            self.forbidden_calls.append(name)
            raise AssertionError(f"pre-reserve must not call arena.{name}")
        return _forbidden

    def reserve(self, transfer_id):
        self.reserve_calls.append(transfer_id)
        if len(self.reserve_calls) == self._fail_reserve_at:
            raise RuntimeError("reserve failed")
        return SimpleNamespace(transfer_id=transfer_id)

    def cancel(self, transfer_id, *, remote_quiescent=False):
        self.cancel_calls.append((transfer_id, remote_quiescent))
        if transfer_id in self._fail_cancel_for:
            raise RuntimeError("cancel failed")

class _RecordingLedger:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def _record(*args, **kwargs):
            self.calls.append(name)
            raise AssertionError(f"pre-reserve must not call ledger.{name}")
        return _record

class _Req:
    def __init__(self, rid, *, inflight=0):
        self.rid = rid
        self.is_retracted = False
        self.inflight_middle_chunks = inflight

    def finished(self):
        return False

def _specs(*transfer_ids):
    return [(f"req-{tid}", index, tid) for index, tid in enumerate(transfer_ids)]

def _reserve_controller(module, arena):
    controller = object.__new__(module.DTTargetController)
    controller._candidate_arena = arena
    controller._profile_active = False
    controller._profile_emit = lambda *args, **kwargs: None
    controller.instance_id = "target:0"
    controller._ledger = _RecordingLedger()
    return controller

def _park_controller(module, arena):
    controller = _reserve_controller(module, arena)
    controller.flashmtp_pre_reserve_dest = True
    controller.is_export_rank = True
    controller.is_token_only = False
    controller.is_feature_backed = False
    controller._parked = {}
    controller._staged_flashmtp_source_leases = set()
    controller._source_deadline_by_transfer = {}
    controller._source_slot_by_transfer = {}
    controller._lane_target_sizes = {0: 0}
    controller._next_round = {}
    controller._verify_transfer_by_request = {}
    controller._queued = []
    controller._release_target_inflight_batch = lambda batch: None
    controller._target_inflight_count = lambda rid: 0
    controller._ensure_lifecycle_attempt = lambda rid: 0
    controller._lane_for_request = lambda req: 0
    controller._new_round_deadline_ns = lambda: 0
    controller._flush_pending_send = lambda: None
    controller._make_task = lambda **kwargs: SimpleNamespace(**kwargs)
    controller._queue_task_batch = lambda tasks: controller._queued.append(tasks)
    return controller

def _batch(module, req, bonus_token):
    return SimpleNamespace(
        reqs=[req],
        spec_info=module.DFlashDraftInputV2(
            topk_p=torch.empty(0),
            topk_index=torch.empty(0, dtype=torch.long),
            bonus_tokens=torch.tensor([bonus_token]),
            new_seq_lens=torch.tensor([bonus_token + 1]),
            hidden_states=torch.empty(0),
        ),
        enable_overlap=False,
    )

def _staged(module, transfer_id, pre_reserved):
    task = module.StagedFlashMTPTask(
        request_id=f"req-{transfer_id}", request_index=0, source_lease_id=transfer_id
    )
    return module.StagedFlashMTPPivotBatch(
        tasks=(task,),
        descriptors_by_transfer={},
        pre_reserved_destinations=pre_reserved,
    )

class TestFlashMTPPreReserve(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_dt_target()

    def test_default_metadata_and_capacity_shortage_arm_nothing(self):
        staged = self.module.StagedFlashMTPPivotBatch(
            tasks=(), descriptors_by_transfer={}
        )
        self.assertIsNone(staged.pre_reserved_destinations)
        arena = _FakeArena(free_slots=2)
        controller = _reserve_controller(self.module, arena)
        result = controller._pre_reserve_flashmtp_destinations(_specs("t0", "t1"))
        self.assertIsNone(result)
        self.assertEqual(arena.reserve_calls, [])
        self.assertEqual(arena.cancel_calls, [])

    def test_full_wave_success_never_authorizes_or_touches_ledger(self):
        arena = _FakeArena(free_slots=8)
        controller = _reserve_controller(self.module, arena)
        result = controller._pre_reserve_flashmtp_destinations(
            _specs("t0", "t1", "t2")
        )
        self.assertEqual(list(result), ["t0", "t1", "t2"])
        self.assertEqual(arena.reserve_calls, ["t0", "t1", "t2"])
        self.assertEqual(arena.cancel_calls, [])
        self.assertEqual(arena.forbidden_calls, [])
        self.assertEqual(controller._ledger.calls, [])

    def test_mid_wave_reserve_failure_rolls_back_reserved_ids(self):
        arena = _FakeArena(free_slots=8, fail_reserve_at=3)
        controller = _reserve_controller(self.module, arena)
        result = controller._pre_reserve_flashmtp_destinations(
            _specs("t0", "t1", "t2", "t3")
        )
        self.assertIsNone(result)
        self.assertEqual(arena.reserve_calls, ["t0", "t1", "t2"])
        # t2 never reserved, so exactly the two successfully armed ids roll back.
        self.assertEqual(arena.cancel_calls, [("t0", True), ("t1", True)])

    def test_one_cancel_failure_does_not_stop_later_cleanup(self):
        arena = _FakeArena(free_slots=8, fail_cancel_for={"t1"})
        controller = _reserve_controller(self.module, arena)
        reservations = {
            tid: (SimpleNamespace(transfer_id=tid), 0)
            for tid in ("t0", "t1", "t2", "t3")
        }
        with self.assertLogs(self.module.logger, level="ERROR") as logs:
            controller._cancel_flashmtp_pre_reserved_destinations(
                reservations, reason="reserve_failure"
            )
        self.assertEqual(
            arena.cancel_calls,
            [("t0", True), ("t1", True), ("t2", True), ("t3", True)],
        )
        self.assertEqual(reservations, {})
        self.assertEqual(len(logs.records), 1)

    def test_park_handoff_and_ineligible_cleanup_cancel_exactly_once(self):
        # Eligible row: the queued task owns the destination, so later staging
        # cleanup must not cancel it a second time.
        arena = _FakeArena(free_slots=8)
        controller = _park_controller(self.module, arena)
        reservations = {"t0": (SimpleNamespace(transfer_id="t0"), 0)}
        staged = _staged(self.module, "t0", reservations)
        batch_result = SimpleNamespace(dt_staged_flashmtp_pivots=staged)
        controller._park_completed_batch(
            _batch(self.module, _Req("req-t0"), 7), batch_result=batch_result
        )
        self.assertEqual(len(controller._queued), 1)
        self.assertEqual(
            controller._queued[0][0].candidate_destination.transfer_id, "t0"
        )
        self.assertEqual(reservations, {})
        controller._cancel_staged_flashmtp_pivots(staged, batch_result)
        self.assertEqual(arena.cancel_calls, [])
        # Ineligible row: launch-time reserve is cancelled once, never requeued.
        arena = _FakeArena(free_slots=8)
        controller = _park_controller(self.module, arena)
        reservations = {"t0": (SimpleNamespace(transfer_id="t0"), 0)}
        staged = _staged(self.module, "t0", reservations)
        batch_result = SimpleNamespace(dt_staged_flashmtp_pivots=staged)
        controller._park_completed_batch(
            _batch(self.module, _Req("req-t0", inflight=1), 7),
            batch_result=batch_result,
        )
        self.assertEqual(controller._queued, [])
        self.assertEqual(arena.cancel_calls, [("t0", True)])
        controller._cancel_staged_flashmtp_pivots(staged, batch_result)
        self.assertEqual(arena.cancel_calls, [("t0", True)])

if __name__ == "__main__":
    unittest.main()
