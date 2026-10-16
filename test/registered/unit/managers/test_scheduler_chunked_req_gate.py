"""Regression tests for the SWA chunked-req stash gate (#24252)."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import NextBatchPlan, Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.speculative.dt_target import DTTargetController
from sglang.srt.utils.common import Range

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _make_req(
    *,
    req_pool_idx: int,
    fill_ids: list,
    prefix_indices: torch.Tensor,
    extend_input_len: int,
    fill_len: int,
) -> Req:
    req = Req.__new__(Req)
    req.rid = "test-req"
    req.origin_input_ids = array("q", fill_ids)
    req.output_ids = array("q")
    req.full_untruncated_fill_ids = array("q", fill_ids)
    req.prefix_indices = prefix_indices
    req.req_pool_idx = req_pool_idx
    req.extend_range = Range(fill_len - extend_input_len, fill_len)
    req.inflight_middle_chunks = 0
    req.host_hit_length = 0
    req.cache_protected_len = 0
    req.skip_radix_cache_insert = False
    req.last_node = None
    req.swa_uuid_for_lock = None
    req.session = None
    req.return_logprob = False
    req.logprob_start_len = -1
    req.positional_embed_overrides = None
    req.extra_key = None
    req.mamba_pool_idx = None
    req.sampling_params = SimpleNamespace(max_new_tokens=128, ignore_eos=False)
    return req


def _make_req_to_token_pool(num_slots: int, max_context: int) -> SimpleNamespace:
    # Slot s contains a recognizable fingerprint [s*1000, s*1000+1, ...]
    # so we can tell a corrupted prefix_indices from a healthy one by content.
    pool = SimpleNamespace()
    pool.req_to_token = (
        torch.arange(max_context, dtype=torch.int32).unsqueeze(0).repeat(num_slots, 1)
        + torch.arange(num_slots, dtype=torch.int32).unsqueeze(1) * 1000
    )
    return pool


def _make_chunk_cache(req_to_token_pool) -> ChunkCache:
    return ChunkCache(
        SimpleNamespace(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=None,
            page_size=1,
        )
    )


def _scheduler_for_get_next_batch(*, tree_cache, chunked_req) -> Scheduler:
    s = Scheduler.__new__(Scheduler)
    s._abort_on_waiting_timeout = MagicMock()
    s._abort_on_running_timeout = MagicMock()
    s.dllm_config = None
    s.dllm_manager = None
    s.enable_hisparse = False
    s.enable_fpm = False
    s.dt_target_controller = None
    s.last_batch = None
    s.require_mlp_sync = False
    s.spec_algorithm = MagicMock()
    s.server_args = MagicMock(speculative_skip_dp_mlp_sync=True)
    s.running_batch = MagicMock()
    s.running_batch.is_empty.return_value = True
    s.running_batch.is_prefill_only = False
    s.running_batch.batch_is_full = False
    s.running_batch.reqs = []
    s.get_new_batch_prefill = MagicMock(
        return_value=NextBatchPlan(batch_to_run=None, running_batch=s.running_batch)
    )
    s.dp_attn_adapter = MagicMock()
    s.dp_attn_adapter.maybe_prepare_mlp_sync_batch = MagicMock(
        side_effect=lambda batch, **_: batch
    )
    s.ngram_embedding_manager = MagicMock()
    s.ngram_embedding_manager.prepare_for_forward = MagicMock(
        side_effect=lambda batch, **_: batch
    )
    s.update_running_batch = MagicMock(side_effect=lambda batch: batch)
    s.tree_cache = tree_cache
    s.chunked_req = chunked_req
    s.waiting_queue = []
    s._pending_chunked_abort_req = None
    return s


class TestStashGatePreservesPrefixIndices(CustomTestCase):
    """Consumer side: real ChunkCache.cache_unfinished_req mutates
    req.prefix_indices iff stash actually runs, so prefix_indices content
    is the bug-detection signal. The stash gate is content-based:
    `fill_len > len(prefix_indices)` means there is freshly computed KV to
    cache; otherwise the chunk was parked and stashing must be skipped."""

    POOL_IDX = 4
    INITIAL_PREFIX_LEN = 8  # what was really cached last iter
    POST_RESET_FILL_LEN = 32  # length after init_next_round_input rebuilds
    NUM_SLOTS = 8
    MAX_CONTEXT = 64

    def _build(self, *, fill_len: int):
        pool = _make_req_to_token_pool(self.NUM_SLOTS, self.MAX_CONTEXT)
        cache = _make_chunk_cache(pool)
        initial_prefix = pool.req_to_token[self.POOL_IDX, : self.INITIAL_PREFIX_LEN].to(
            dtype=torch.int64, copy=True
        )
        req = _make_req(
            req_pool_idx=self.POOL_IDX,
            fill_ids=list(range(self.POST_RESET_FILL_LEN)),
            prefix_indices=initial_prefix,
            extend_input_len=fill_len - self.INITIAL_PREFIX_LEN,
            fill_len=fill_len,
        )
        s = _scheduler_for_get_next_batch(tree_cache=cache, chunked_req=req)
        return s, req, initial_prefix, pool

    def test_parked_chunked_req_keeps_real_prefix_indices(self):
        # A parked chunk has fill_len == len(prefix_indices): no new KV was
        # computed, so the gate must skip stash and leave prefix_indices intact.
        s, req, initial_prefix, _ = self._build(fill_len=self.INITIAL_PREFIX_LEN)

        Scheduler.get_next_batch_to_run(
            s, running_batch=s.running_batch, last_batch=s.last_batch
        )

        self.assertEqual(req.prefix_indices.shape[0], self.INITIAL_PREFIX_LEN)
        self.assertTrue(torch.equal(req.prefix_indices, initial_prefix))

    def test_scheduled_chunked_req_advances_prefix_indices_via_real_stash(self):
        # Symmetric guard against over-gating: when fill_len has advanced past
        # the cached prefix, stash must run and advance prefix_indices.
        s, req, _, pool = self._build(fill_len=self.POST_RESET_FILL_LEN)

        Scheduler.get_next_batch_to_run(
            s, running_batch=s.running_batch, last_batch=s.last_batch
        )

        expected = pool.req_to_token[self.POOL_IDX, : self.POST_RESET_FILL_LEN].to(
            dtype=torch.int64
        )
        self.assertEqual(req.prefix_indices.shape[0], self.POST_RESET_FILL_LEN)
        self.assertTrue(torch.equal(req.prefix_indices, expected))

    def test_no_chunked_req_never_mutates_state(self):
        # The outer `if chunked_req is not None` guard must hold on the retract
        # path that clears chunked_req.
        pool = _make_req_to_token_pool(self.NUM_SLOTS, self.MAX_CONTEXT)
        cache = _make_chunk_cache(pool)
        s = _scheduler_for_get_next_batch(tree_cache=cache, chunked_req=None)

        Scheduler.get_next_batch_to_run(
            s, running_batch=s.running_batch, last_batch=s.last_batch
        )
        self.assertIsNone(s.chunked_req)


class TestDTPrefillArbitration(CustomTestCase):
    def _build(self):
        pool = _make_req_to_token_pool(num_slots=2, max_context=8)
        scheduler = _scheduler_for_get_next_batch(
            tree_cache=_make_chunk_cache(pool), chunked_req=None
        )
        controller = MagicMock()
        controller.pop_failures.return_value = []
        controller.has_feature_source_capacity.return_value = True
        controller.should_defer_prefill.return_value = False
        controller.empty_prefill_arbitration_profile_enabled = True
        controller.eagle3_host_span_profile_enabled = False
        scheduler.dt_target_controller = controller
        return scheduler, controller

    def test_admissible_prefill_runs_before_unsubmitted_ready_wave(self):
        scheduler, controller = self._build()
        prefill = MagicMock(name="prefill")
        scheduler.get_new_batch_prefill.return_value = NextBatchPlan(
            batch_to_run=prefill, running_batch=scheduler.running_batch
        )

        plan = Scheduler.get_next_batch_to_run(
            scheduler,
            running_batch=scheduler.running_batch,
            last_batch=None,
        )

        self.assertIs(plan.batch_to_run, prefill)
        controller.build_ready_batch.assert_not_called()
        controller.profile_empty_prefill_arbitration.assert_called_once()
        profile = controller.profile_empty_prefill_arbitration.call_args.kwargs
        self.assertEqual(profile["prefill_decision"], "prefill_selected")
        self.assertEqual(profile["selected"], "prefill")

    def test_stale_empty_full_alias_is_normalized_before_prefill_choice(self):
        scheduler, controller = self._build()
        scheduler.running_batch.batch_is_full = True
        controller.detach_parked_requests.side_effect = (
            lambda batch: DTTargetController.detach_parked_requests(
                controller, batch
            )
        )
        prefill = MagicMock(name="prefill")

        def choose_prefill(running_batch):
            self.assertFalse(running_batch.batch_is_full)
            return NextBatchPlan(
                batch_to_run=prefill,
                running_batch=running_batch,
            )

        scheduler.get_new_batch_prefill.side_effect = choose_prefill

        plan = Scheduler.get_next_batch_to_run(
            scheduler,
            running_batch=scheduler.running_batch,
            last_batch=None,
        )

        self.assertIs(plan.batch_to_run, prefill)
        controller.build_ready_batch.assert_not_called()
        profile = controller.profile_empty_prefill_arbitration.call_args.kwargs
        self.assertTrue(profile["empty_full_before_detach"])
        self.assertTrue(profile["fullness_cleared_by_detach"])

    def test_ready_wave_runs_when_no_prefill_is_admissible(self):
        scheduler, controller = self._build()
        ready = MagicMock(name="ready")
        ready.is_empty.return_value = False
        ready.is_prefill_only = False
        scheduler.get_new_batch_prefill.return_value = NextBatchPlan(
            batch_to_run=None, running_batch=scheduler.running_batch
        )
        controller.build_ready_batch.return_value = ready

        plan = Scheduler.get_next_batch_to_run(
            scheduler,
            running_batch=scheduler.running_batch,
            last_batch=None,
        )

        self.assertIs(plan.batch_to_run, ready)
        controller.build_ready_batch.assert_called_once_with()
        profile = controller.profile_empty_prefill_arbitration.call_args.kwargs
        self.assertEqual(profile["prefill_decision"], "no_waiting")
        self.assertEqual(profile["selected"], "ready")

    def test_reserved_source_headroom_blocks_prefill_before_ready_choice(self):
        scheduler, controller = self._build()
        ready = MagicMock(name="ready")
        ready.is_empty.return_value = False
        ready.is_prefill_only = False
        controller.should_defer_prefill.return_value = True
        controller.build_ready_batch.return_value = ready

        plan = Scheduler.get_next_batch_to_run(
            scheduler,
            running_batch=scheduler.running_batch,
            last_batch=None,
        )

        self.assertIs(plan.batch_to_run, ready)
        scheduler.get_new_batch_prefill.assert_not_called()
        profile = controller.profile_empty_prefill_arbitration.call_args.kwargs
        self.assertEqual(profile["prefill_decision"], "one_slot_headroom")
        self.assertEqual(profile["selected"], "ready")

    def test_previous_prefill_result_is_reconciled_before_ready_choice(self):
        scheduler, controller = self._build()
        previous_prefill = MagicMock(name="previous_prefill")
        previous_prefill.forward_mode.is_extend.return_value = True
        previous_prefill.chunked_req = None
        previous_prefill.reqs = []
        previous_prefill.batch_size.return_value = 1
        previous_prefill.is_empty.return_value = False
        previous_prefill.is_prefill_only = False
        scheduler.get_new_batch_prefill.side_effect = lambda running: NextBatchPlan(
            batch_to_run=None, running_batch=running
        )

        plan = Scheduler.get_next_batch_to_run(
            scheduler,
            running_batch=scheduler.running_batch,
            last_batch=previous_prefill,
        )

        self.assertIs(plan.batch_to_run, previous_prefill)
        controller.build_ready_batch.assert_not_called()
        profile = controller.profile_empty_prefill_arbitration.call_args.kwargs
        self.assertEqual(profile["prefill_decision"], "not_empty_running")
        self.assertEqual(profile["selected"], "running")


def test_overlap_snapshot_preserves_dt_full_wave_profile_identity():
    batch = ScheduleBatch(reqs=[])
    batch.dt_full_wave_id = 17
    batch.dt_full_wave_transfer_ids = ("transfer-a", "transfer-b")
    batch.dt_full_wave_ready_ns = 123456

    copied = batch.copy()

    self_identity = (
        copied.dt_full_wave_id,
        copied.dt_full_wave_transfer_ids,
        copied.dt_full_wave_ready_ns,
    )
    assert self_identity == (17, ("transfer-a", "transfer-b"), 123456)


if __name__ == "__main__":
    unittest.main()
