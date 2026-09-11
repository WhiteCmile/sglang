from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace

import pytest

from dt_client import RetryOutbox
from dt_protocol import (
    EntryStatus,
    FINALIZE_DRAFT_KV_MIGRATION,
    PROTOCOL_VERSION,
    RELEASE_DRAFT_REQUEST,
    DraftResultEntry,
    DraftKvMigrationFinalize,
    DraftRequestRelease,
    PullDone,
    SourceReadDone,
    TargetResultBatch,
    draft_request_release_id,
    draft_kv_migration_finalized_id,
    pull_done_id,
)
from sglang.srt.speculative.dt_draft_engine import (
    FlashMTPDraftEngine,
    PublishedResult,
)
from sglang.srt.speculative.dt_target import DTTargetController
from sglang.srt.speculative import dt_kv_p2p
from sglang.srt.speculative.dt_kv_p2p import (
    DraftKvP2pTransport,
    TransferQuiescenceError,
)


def _init_lifetime_state(engine) -> None:
    engine._is_dflash = False
    engine._pending = deque()
    engine._pending_feature_batches = deque()
    engine._pending_dflash_completions = deque()
    engine._migration_release_source_leases = {}
    engine._migration_release_feature_transfers = {}
    engine._migration_release_feature_sources = {}
    engine._migration_feature_sources_released = {}
    engine._migration_destination_copy_quiescent = {}
    engine._migration_source_leases_released = {}
    engine._migration_source_handles = {}
    engine._deferred_persistent_releases = {}
    engine._pending_request_release_acks = {}
    engine._migration_quiescence_probes = {}
    engine._migration_quiescence_retry_ns = {}
    engine._deferred_feature_source_tasks = {}
    engine._feature_source_inflight = set()
    engine._feature_source_quiescent = set()
    engine._feature_source_unquiescent = set()


def test_late_unquiescent_failure_quarantines_after_result_is_stale() -> None:
    controller = object.__new__(DTTargetController)
    controller.is_dflash = True
    quarantined = []
    controller._quarantine_source_slot = quarantined.append
    controller._ledger = SimpleNamespace(
        apply_results=lambda _batch: SimpleNamespace(
            ready_transfer_ids=(), failed_transfer_ids=()
        )
    )
    controller.block_size = 8
    controller._cancelled_transfers = set()
    controller._result_stores = {}

    events = controller._consume_result_batch_body(
        TargetResultBatch(
            protocol_version=PROTOCOL_VERSION,
            batch_id="late-lease-abandon",
            target_instance_id="target-0",
            draft_instance_id="draft-0",
            entries=(
                DraftResultEntry(
                    transfer_id="transfer-0",
                    request_id="request-0",
                    round_id=1,
                    attempt=0,
                    target_instance_id="target-0",
                    status=EntryStatus.FINAL_ERROR,
                    error_code="draft_session_replaced",
                    source_read_quiescent=False,
                ),
            ),
            sent_ns=1,
        )
    )

    assert not events
    assert quarantined == ["transfer-0"]


def test_source_read_done_rejects_a_stale_coordinator_epoch() -> None:
    controller = object.__new__(DTTargetController)
    controller.instance_id = "target-0"
    controller._client = SimpleNamespace(coordinator_session_id="coordinator-new")
    message = SourceReadDone(
        protocol_version=PROTOCOL_VERSION,
        transfer_id="transfer-0",
        target_instance_id="target-0",
        draft_instance_id="draft-0",
        source_slot_id=0,
        slot_generation=1,
        completed_ns=1,
        instance_session_id="draft-session",
        coordinator_session_id="coordinator-old",
        source_session_id="target-session",
        pool_generation="pool-0",
    )

    with pytest.raises(ValueError, match="session fence"):
        controller._consume_source_read_done(message)


def test_reused_request_id_gets_a_new_lifecycle_attempt() -> None:
    controller = object.__new__(DTTargetController)
    controller._lifecycle_attempt_by_request = {}
    controller._next_lifecycle_attempt = 0

    assert controller._ensure_lifecycle_attempt("request-0") == 0
    controller._lifecycle_attempt_by_request.pop("request-0")
    assert controller._ensure_lifecycle_attempt("request-0") == 1
    assert controller._advance_lifecycle_attempt("request-0") == 2


def test_terminal_request_retires_late_dflash_control_failure() -> None:
    controller = object.__new__(DTTargetController)
    req = SimpleNamespace(rid="request-0")
    state = SimpleNamespace(req=req, transfer_id="transfer-0")
    controller.instance_id = "target-0"
    controller.is_dflash = True
    controller.is_export_rank = True
    controller.block_size = 8
    controller._parked = {}
    controller._feature_control_rounds = {state.transfer_id: state}
    controller._cancelled_transfers = set()
    controller._source_slot_by_transfer = {state.transfer_id: (0, 1)}
    controller._pending_send = deque()
    controller._result_stores = {}
    controller._delivery_outbox = RetryOutbox(retry_interval_ns=1)
    controller._drop_cancelled_unsent = lambda: None
    quarantined = []
    controller._quarantine_source_slot = quarantined.append
    finished = []
    controller._ledger = SimpleNamespace(
        get=lambda _transfer_id: object(),
        apply_results=lambda _batch: SimpleNamespace(
            ready_transfer_ids=(), failed_transfer_ids=(state.transfer_id,)
        ),
        finish=finished.append,
    )

    controller._retire_dflash_rounds_for_request(
        req.rid, reason="request_finished"
    )
    events = controller._consume_result_batch_body(
        TargetResultBatch(
            protocol_version=PROTOCOL_VERSION,
            batch_id="late-control-failure",
            target_instance_id="target-0",
            draft_instance_id="draft-0",
            entries=(
                DraftResultEntry(
                    transfer_id=state.transfer_id,
                    request_id=req.rid,
                    round_id=0,
                    attempt=0,
                    target_instance_id="target-0",
                    status=EntryStatus.FINAL_ERROR,
                    source_read_quiescent=False,
                ),
            ),
            sent_ns=1,
        )
    )

    assert not controller._feature_control_rounds
    assert not events
    assert quarantined == [state.transfer_id]
    assert finished == [state.transfer_id]
    assert state.transfer_id not in controller._cancelled_transfers


def test_oom_cleanup_detaches_failed_parked_alias_before_kv_release() -> None:
    controller = object.__new__(DTTargetController)
    req = SimpleNamespace(rid="request-0")
    state = SimpleNamespace(
        req=req,
        transfer_id="transfer-0",
        failure="draft_request_released",
    )
    controller.is_persistent = True
    controller.is_export_rank = False
    controller._parked = {state.transfer_id: state}
    controller._feature_control_rounds = {}
    controller._next_round = {req.rid: 1}
    controller._lifecycle_attempt_by_request = {req.rid: 0}
    controller._next_lifecycle_attempt = 1
    controller._forget_request_lane = lambda _req: None
    controller._finish_verified_request = lambda _rid: None
    controller._release_draft_request = lambda _rid: None

    controller.cleanup_running_after_oom(
        retracted_reqs=[req], aborted_reqs=[]
    )

    assert not controller._parked
    assert not controller.pop_failures()
    assert controller._lifecycle_attempt_by_request[req.rid] == 1


def test_dflash_overlap_routes_staged_feature_boundaries_in_fifo_order() -> None:
    controller = object.__new__(DTTargetController)
    controller.is_token_only = False
    controller.is_dflash = True
    controller.instance_id = "target-0"
    controller.profile_id = "dflash-profile"
    controller.block_size = 7
    controller.round_timeout_ns = 1_000_000
    controller._draft_committed_len = {"request-0": 20}
    controller._draft_dispatched_len = {}
    controller._draft_kv_handle_by_request = {"request-0": "kv-0"}
    req = SimpleNamespace(rid="request-0")

    def descriptor(transfer_id):
        return SimpleNamespace(
            source_ref_kwargs=lambda: {
                "target_instance_id": "target-0",
                "source_device_id": 0,
                "source_slot_id": 0,
                "slot_generation": 1,
                "transfer_id": transfer_id,
                "feature_schema_id": "dflash-features",
                "nbytes": 10 * 25600 * 2,
            }
        )

    first = controller._make_task(
        transfer_id="transfer-0",
        req=req,
        round_id=0,
        attempt=0,
        bonus_token=11,
        source_slot=None,
        source_generation=None,
        pivot_nbytes=10 * 25600 * 2,
        previous_transfer_id=None,
        feature_descriptor=descriptor("transfer-0"),
        feature_token_ids=tuple(range(10)),
        feature_token_offset=0,
        requires_proposal=False,
    )
    second = controller._make_task(
        transfer_id="transfer-1",
        req=req,
        round_id=1,
        attempt=0,
        bonus_token=12,
        source_slot=None,
        source_generation=None,
        pivot_nbytes=10 * 25600 * 2,
        previous_transfer_id="transfer-0",
        feature_descriptor=descriptor("transfer-1"),
        feature_token_ids=tuple(range(10, 20)),
        feature_token_offset=10,
    )

    assert first.committed_kv_len == 10
    assert second.feature_token_offset == first.committed_kv_len
    assert second.committed_kv_len == 20
    assert controller._draft_dispatched_len[req.rid] == 20


def test_request_release_ack_waits_for_persistent_kv_retirement() -> None:
    engine = object.__new__(FlashMTPDraftEngine)
    release = DraftRequestRelease(
        protocol_version=PROTOCOL_VERSION,
        target_instance_id="target-0",
        request_id="request-0",
        draft_kv_handle="draft-kv-0",
        released_ns=1,
        draft_instance_id="draft-0",
    )
    release_id = draft_request_release_id(release)
    engine._persistent_requests = {release.draft_kv_handle: object()}
    engine._pending_request_release_acks = {release_id: release}
    engine._delivery_outbox = RetryOutbox(retry_interval_ns=1)
    engine._flush_reliable_outbox = lambda: None

    engine._ack_applied_request_releases(release.draft_kv_handle)
    assert engine._delivery_outbox.get(RELEASE_DRAFT_REQUEST, release_id) is None

    engine._persistent_requests.clear()
    engine._ack_applied_request_releases(release.draft_kv_handle)
    acknowledgement = engine._delivery_outbox.get(RELEASE_DRAFT_REQUEST, release_id)
    assert acknowledgement is not None
    assert acknowledgement.operation == RELEASE_DRAFT_REQUEST
    assert acknowledgement.message_id == release_id


def test_chunked_feature_dispatch_waits_for_prior_request_transfer() -> None:
    controller = object.__new__(DTTargetController)
    task = SimpleNamespace(
        request_id="request-0", transfer_id="transfer-1", round_id=1
    )
    batch = SimpleNamespace(batch_id="batch-1", entries=(task,))
    controller._pending_send = deque((batch,))
    controller._last_pending_send_block = None
    controller._flush_reliable_outbox = lambda: None
    controller._profile_emit = lambda *_args, **_kwargs: None
    active = {"request-0": "transfer-0"}
    active_state = SimpleNamespace(
        task=SimpleNamespace(round_id=0, requires_proposal=False),
        phase=SimpleNamespace(value="wait_draft"),
    )
    begun = []
    submitted = []
    controller._ledger = SimpleNamespace(
        active_transfer_id=lambda request_id: active.get(request_id),
        get=lambda transfer_id: (
            active_state if transfer_id == "transfer-0" else None
        ),
        begin=begun.append,
    )
    controller._client = SimpleNamespace(
        submit=lambda value: submitted.append(value) or True
    )
    controller._delivery_outbox = SimpleNamespace(
        add=lambda *_args: None,
        mark_sent=lambda *_args, **_kwargs: None,
    )

    controller._flush_pending_send()
    assert not submitted
    assert tuple(controller._pending_send) == (batch,)

    active.clear()
    controller._flush_pending_send()
    assert submitted == [batch]
    assert begun == [batch]
    assert not controller._pending_send


def test_dflash_migration_waits_for_every_feature_consumer_stage() -> None:
    engine = object.__new__(FlashMTPDraftEngine)
    handle = "draft-kv-0"
    queued_task = SimpleNamespace(draft_kv_handle=handle)
    feature_item = SimpleNamespace(task=queued_task)
    engine._pending = deque((queued_task,))
    engine._pending_feature_batches = deque()
    engine._pending_dflash_completions = deque()
    assert engine._dflash_handle_has_active_consumer(handle)

    engine._pending.clear()
    engine._pending_feature_batches.append(
        SimpleNamespace(pending=(feature_item,))
    )
    assert engine._dflash_handle_has_active_consumer(handle)

    engine._pending_feature_batches.clear()
    engine._pending_dflash_completions.append(
        SimpleNamespace(pending=(feature_item,))
    )
    assert engine._dflash_handle_has_active_consumer(handle)

    engine._pending_dflash_completions.clear()
    assert not engine._dflash_handle_has_active_consumer(handle)


def test_failed_enqueued_copy_future_waits_for_gpu_quiescence(monkeypatch) -> None:
    """A post-enqueue submit error cannot make reservation cleanup runnable."""

    trace = []
    gpu_may_complete = Event()
    worker_is_fencing = Event()

    class FakeCudaEvent:
        def __init__(self, *, enable_timing=False) -> None:
            self.enable_timing = enable_timing

        def record(self, _stream) -> None:
            trace.append("event_recorded")

        def synchronize(self) -> None:
            raise AssertionError("submit failure must use the stream fence")

        def elapsed_time(self, _other) -> float:
            return 0.0

    class FakeCopyStream:
        cuda_stream = 17

        def wait_event(self, _event) -> None:
            trace.append("producer_wait")

        def synchronize(self) -> None:
            worker_is_fencing.set()
            assert gpu_may_complete.wait(timeout=2)
            trace.append("gpu_complete")

    class FailingAfterEnqueueRuntime:
        @staticmethod
        def copy_batch_async(_segments, _stream) -> None:
            trace.append("work_enqueued")
            raise RuntimeError("injected failure after enqueue")

    monkeypatch.setattr(dt_kv_p2p.torch.cuda, "Event", FakeCudaEvent)
    monkeypatch.setattr(dt_kv_p2p.torch.cuda, "stream", lambda _stream: nullcontext())
    transport = object.__new__(DraftKvP2pTransport)
    transport._runtime = FailingAfterEnqueueRuntime()
    transport._copy_stream = FakeCopyStream()
    transport._fatal_error = None

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            transport._copy_and_wait,
            segments=(),
            source_ready=object(),
            mapping_started_ns=1,
            mapping_done_ns=2,
            kv_bytes=0,
            page_count=0,
            cold_mapping_count=0,
        )
        cleanup_order = []
        future.add_done_callback(lambda _future: cleanup_order.append(len(trace)))

        assert worker_is_fencing.wait(timeout=2)
        assert not future.done()
        assert not cleanup_order
        gpu_may_complete.set()
        with pytest.raises(RuntimeError, match="injected failure after enqueue"):
            future.result(timeout=2)

    assert trace[-1] == "gpu_complete"
    assert cleanup_order == [len(trace)]


def test_failed_event_record_after_source_wait_still_fences_remote_read(
    monkeypatch,
) -> None:
    """A recorded source wait is a remote-read lease before memcpy submission."""

    trace = []
    gpu_may_complete = Event()
    worker_is_fencing = Event()

    class FailingCudaEvent:
        def __init__(self, *, enable_timing=False) -> None:
            self.enable_timing = enable_timing

        def record(self, _stream) -> None:
            trace.append("event_record_failed")
            raise RuntimeError("injected event record failure")

    class FakeCopyStream:
        cuda_stream = 17

        def wait_event(self, _event) -> None:
            trace.append("producer_wait_enqueued")

        def synchronize(self) -> None:
            worker_is_fencing.set()
            assert gpu_may_complete.wait(timeout=2)
            trace.append("source_wait_quiescent")

    class UnusedRuntime:
        @staticmethod
        def copy_batch_async(_segments, _stream) -> None:
            raise AssertionError("event-record failure must precede batch submission")

    monkeypatch.setattr(dt_kv_p2p.torch.cuda, "Event", FailingCudaEvent)
    monkeypatch.setattr(dt_kv_p2p.torch.cuda, "stream", lambda _stream: nullcontext())
    transport = object.__new__(DraftKvP2pTransport)
    transport._runtime = UnusedRuntime()
    transport._copy_stream = FakeCopyStream()
    transport._fatal_error = None

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            transport._copy_and_wait,
            segments=(),
            source_ready=object(),
            mapping_started_ns=1,
            mapping_done_ns=2,
            kv_bytes=0,
            page_count=0,
            cold_mapping_count=0,
        )

        assert worker_is_fencing.wait(timeout=2)
        assert not future.done()
        gpu_may_complete.set()
        with pytest.raises(RuntimeError, match="injected event record failure"):
            future.result(timeout=2)

    assert trace == [
        "producer_wait_enqueued",
        "event_record_failed",
        "source_wait_quiescent",
    ]


def test_failed_event_record_with_failed_fence_is_unquiesced(monkeypatch) -> None:
    """Failure to fence an enqueued source wait quarantines the transport."""

    class FailingCudaEvent:
        def __init__(self, *, enable_timing=False) -> None:
            self.enable_timing = enable_timing

        def record(self, _stream) -> None:
            raise RuntimeError("injected event record failure")

    class FailingCopyStream:
        cuda_stream = 17

        def wait_event(self, _event) -> None:
            return None

        def synchronize(self) -> None:
            raise RuntimeError("stream fence failed")

    monkeypatch.setattr(dt_kv_p2p.torch.cuda, "Event", FailingCudaEvent)
    monkeypatch.setattr(dt_kv_p2p.torch.cuda, "stream", lambda _stream: nullcontext())
    transport = object.__new__(DraftKvP2pTransport)
    transport._runtime = object()
    transport._copy_stream = FailingCopyStream()
    transport._fatal_error = None

    with pytest.raises(TransferQuiescenceError, match="must remain quarantined"):
        transport._copy_and_wait(
            segments=(),
            source_ready=object(),
            mapping_started_ns=1,
            mapping_done_ns=2,
            kv_bytes=0,
            page_count=0,
            cold_mapping_count=0,
        )

    assert isinstance(transport._fatal_error, TransferQuiescenceError)


def test_unquiesced_copy_failure_quarantines_destination_reservation() -> None:
    engine = object.__new__(FlashMTPDraftEngine)
    _init_lifetime_state(engine)
    migration_id = "migration-unquiesced"
    prepared = SimpleNamespace(req=object(), scheduler_submit_ns=0)
    engine._prepared_migrations = {migration_id: prepared}
    engine._persistent_requests = {}
    engine._quarantined_migration_ids = set()
    engine._deferred_migration_finalizations = {}
    engine.instance_id = "draft-destination"
    published = []
    cleaned = []
    engine._publish_migration_status = published.append
    engine._cleanup_partial_migration_request = cleaned.append
    message = SimpleNamespace(
        migration_id=migration_id,
        source_draft_instance_id="draft-source",
        destination_draft_instance_id="draft-destination",
        target_instance_id="target-0",
        request_id="request-0",
        draft_kv_handle="kv-0",
        draft_profile_id="profile-0",
        round_id=1,
        committed_kv_len=4,
        reservation=object(),
        source_pool=None,
        transfer=None,
    )

    engine._complete_migration_import(
        message,
        error=TransferQuiescenceError("GPU completion is unknown"),
    )

    assert not cleaned
    assert engine._prepared_migrations[migration_id] is prepared
    assert migration_id in engine._quarantined_migration_ids
    assert published[0].error_code == "TransferQuiescenceError"


def test_done_unquiesced_copy_is_consumed_before_deferred_finalize_cleanup() -> None:
    """A done worker Future is not safe until the scheduler consumes its result."""

    engine = object.__new__(FlashMTPDraftEngine)
    _init_lifetime_state(engine)
    migration_id = "migration-done-unquiesced"
    req = SimpleNamespace(req_pool_idx=7)
    future = Future()
    future.set_exception(TransferQuiescenceError("GPU completion is unknown"))
    import_message = SimpleNamespace(
        migration_id=migration_id,
        source_draft_instance_id="draft-source",
        destination_draft_instance_id="draft-destination",
        target_instance_id="target-0",
        request_id="request-0",
        draft_kv_handle="kv-0",
        draft_profile_id="profile-0",
        round_id=1,
        committed_kv_len=4,
        reservation=object(),
        source_pool=None,
        transfer=None,
    )
    prepared = SimpleNamespace(
        req=req,
        scheduler_submit_ns=0,
        transfer_future=future,
        import_message=import_message,
        message=SimpleNamespace(draft_kv_handle="kv-0"),
    )
    engine._prepared_migrations = {migration_id: prepared}
    engine._persistent_requests = {}
    engine._quarantined_migration_ids = set()
    engine._deferred_migration_finalizations = {}
    engine._migration_roles = {migration_id: "destination"}
    engine._migration_decisions = {}
    engine._migration_release_requests = {}
    engine._migration_cleanup_ids = {}
    engine._migration_source_events = {}
    engine.instance_id = "draft-destination"
    quiescence_probe = Future()
    engine._p2p_transport = SimpleNamespace(
        probe_quiescence=lambda: quiescence_probe
    )
    published_status = []
    published_finalized = []
    cleaned = []
    released = []
    engine._publish_migration_status = published_status.append
    engine._publish_migration_finalized = (
        lambda *args, **kwargs: published_finalized.append((args, kwargs))
    )
    engine._cleanup_partial_migration_request = cleaned.append
    engine._release_persistent_request = lambda *args, **kwargs: released.append(
        (args, kwargs)
    )
    engine._delivery_outbox = SimpleNamespace(ack=lambda *_args: None)
    finalize = DraftKvMigrationFinalize(
        protocol_version=PROTOCOL_VERSION,
        migration_id=migration_id,
        source_draft_instance_id="draft-source",
        destination_draft_instance_id="draft-destination",
        target_instance_id="target-0",
        request_id="request-0",
        draft_kv_handle="kv-0",
        committed=False,
    )

    engine._handle_migration_finalize(finalize)

    assert engine._deferred_migration_finalizations[migration_id] == finalize
    assert engine._prepared_migrations[migration_id] is prepared
    assert not cleaned
    assert not released
    assert not published_finalized

    assert engine._poll_migration_transfers()

    assert migration_id in engine._quarantined_migration_ids
    assert engine._prepared_migrations[migration_id] is prepared
    assert migration_id not in engine._deferred_migration_finalizations
    assert not cleaned
    assert released
    assert published_status[0].error_code == "TransferQuiescenceError"
    assert published_finalized
    assert not published_finalized[-1][1]["destination_copy_quiescent"]

    quiescence_probe.set_result(True)
    assert engine._poll_migration_transfers()
    assert migration_id not in engine._quarantined_migration_ids
    assert migration_id not in engine._prepared_migrations
    assert cleaned == [req]
    assert published_finalized[-1][1]["destination_copy_quiescent"]


def test_close_does_not_recycle_an_unconsumed_transfer_reservation() -> None:
    """Shutdown leaves any attached worker result for CUDA context teardown."""

    class Closable:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class ResultStore(Closable):
        def __init__(self) -> None:
            super().__init__()
            self.unlinked = False

        def unlink(self) -> None:
            self.unlinked = True

    engine = object.__new__(FlashMTPDraftEngine)
    _init_lifetime_state(engine)
    future = Future()
    future.set_exception(TransferQuiescenceError("GPU completion is unknown"))
    prepared = SimpleNamespace(
        transfer_future=future,
        message=SimpleNamespace(draft_kv_handle="kv-0"),
        req=SimpleNamespace(req_pool_idx=7),
    )
    transport = Closable()
    client = Closable()
    result_store = ResultStore()
    cleaned = []
    engine._p2p_transport = transport
    engine._prepared_migrations = {"migration-0": prepared}
    engine._quarantined_migration_ids = set()
    engine._persistent_requests = {}
    engine._source_stores = {}
    engine._migration_source_events = {}
    engine._client = client
    engine._result_store = result_store
    engine._cleanup_partial_migration_request = cleaned.append

    engine.close()

    assert transport.closed
    assert not cleaned
    assert not engine._prepared_migrations
    assert client.closed
    assert result_store.closed
    assert result_store.unlinked


def test_migration_published_result_release_returns_the_draft_slot() -> None:
    class RecordingClient:
        def __init__(self) -> None:
            self.finalized = []
            self.instance_session_id = "draft-session"

        def migration_finalized(self, message) -> bool:
            self.finalized.append(message)
            return True

    engine = object.__new__(FlashMTPDraftEngine)
    _init_lifetime_state(engine)
    engine._result_pool_generation = "result-pool"
    key = ("draft-session", "result-pool", 7, 3, 0, 1)
    published = PublishedResult(
        batch_id="migration-result",
        slot_id=5,
        generation=3,
        transfer_ids=("transfer-0",),
        transfer_rows={"transfer-0": ("target-0", 0)},
        range_transfer_ids={key: frozenset({"transfer-0"})},
    )
    engine._published_by_batch = {published.batch_id: published}
    engine._published_by_range = {key: published.batch_id}
    engine._free_result_slots = deque()
    engine.instance_id = "draft-0"
    engine._migration_roles = {"migration-0": "source"}
    engine._migration_decisions = {}
    engine._migration_release_requests = {}
    engine._migration_cleanup_ids = {}
    engine._prepared_migrations = {}
    engine._deferred_migration_finalizations = {}
    engine._migration_source_events = {}
    engine._persistent_requests = {}
    engine._delivery_outbox = RetryOutbox(retry_interval_ns=1)
    engine._client = RecordingClient()
    result_release = PullDone(
        protocol_version=PROTOCOL_VERSION,
        target_instance_id="target-0",
        draft_instance_id="draft-0",
        buffer_id=7,
        generation=3,
        row_offset=0,
        row_count=1,
        completed_ns=100,
        draft_session_id="draft-session",
        result_pool_generation="result-pool",
    )

    old_finalize = DraftKvMigrationFinalize(
        protocol_version=PROTOCOL_VERSION,
        migration_id="migration-0",
        source_draft_instance_id="draft-0",
        destination_draft_instance_id="draft-1",
        target_instance_id="target-0",
        request_id="request-0",
        draft_kv_handle="kv-0",
        committed=False,
    )
    engine._handle_migration_finalize(old_finalize)
    old_ack = engine._client.finalized[-1]
    assert old_ack.cleanup_id == ""
    assert engine._published_by_batch

    new_finalize = DraftKvMigrationFinalize(
        protocol_version=PROTOCOL_VERSION,
        migration_id="migration-0",
        source_draft_instance_id="draft-0",
        destination_draft_instance_id="draft-1",
        target_instance_id="target-0",
        request_id="request-0",
        draft_kv_handle="kv-0",
        committed=False,
        published_result_release=result_release,
        cleanup_id=pull_done_id(result_release),
    )
    engine._handle_migration_finalize(new_finalize)

    assert not engine._published_by_batch
    assert not engine._published_by_range
    assert list(engine._free_result_slots) == [5]
    new_ack = engine._client.finalized[-1]
    assert new_ack.cleanup_id == pull_done_id(result_release)
    pending_ids = {
        message_id
        for operation, message_id, _ in engine._delivery_outbox.pending()
        if operation == FINALIZE_DRAFT_KV_MIGRATION
    }
    assert pending_ids == {draft_kv_migration_finalized_id(new_ack)}

    engine._delivery_outbox.ack(
        FINALIZE_DRAFT_KV_MIGRATION,
        draft_kv_migration_finalized_id(old_ack),
    )
    assert (
        engine._delivery_outbox.get(
            FINALIZE_DRAFT_KV_MIGRATION,
            draft_kv_migration_finalized_id(new_ack),
        )
        == new_ack
    )


def test_request_release_waits_for_every_source_read_lease() -> None:
    class RecordingClient:
        def __init__(self) -> None:
            self.finalized = []

        def migration_finalized(self, message) -> bool:
            self.finalized.append(message)
            return True

    engine = object.__new__(FlashMTPDraftEngine)
    _init_lifetime_state(engine)
    handle = "kv-shared"
    request_id = "request-shared"
    engine.instance_id = "draft-source"
    engine._migration_roles = {
        "migration-0": "source",
        "migration-1": "source",
    }
    engine._migration_decisions = {}
    engine._migration_release_requests = {}
    engine._migration_cleanup_ids = {}
    engine._prepared_migrations = {}
    engine._deferred_migration_finalizations = {}
    engine._migration_source_events = {
        "migration-0": object(),
        "migration-1": object(),
    }
    engine._migration_source_handles = {
        "migration-0": handle,
        "migration-1": handle,
    }
    engine._persistent_requests = {
        handle: SimpleNamespace(
            request_id=request_id,
            draft_kv_handle=handle,
            req=SimpleNamespace(kv=None, req_pool_idx=None),
        )
    }
    engine._delivery_outbox = RetryOutbox(retry_interval_ns=1)
    engine._client = RecordingClient()

    assert not engine._release_persistent_request(handle, request_id=request_id)
    assert handle in engine._persistent_requests

    def finalize(migration_id: str) -> DraftKvMigrationFinalize:
        return DraftKvMigrationFinalize(
            protocol_version=PROTOCOL_VERSION,
            migration_id=migration_id,
            source_draft_instance_id=engine.instance_id,
            destination_draft_instance_id="draft-destination",
            target_instance_id="target-0",
            request_id=request_id,
            draft_kv_handle=handle,
            committed=False,
            release_request=True,
            release_source_lease=True,
        )

    first = finalize("migration-0")
    engine._handle_migration_finalize(first)
    engine._handle_migration_finalize(first)  # delayed duplicate is harmless

    assert "migration-0" not in engine._migration_source_events
    assert "migration-1" in engine._migration_source_events
    assert handle in engine._persistent_requests
    assert engine._client.finalized[-1].source_lease_released

    engine._handle_migration_finalize(finalize("migration-1"))

    assert not engine._migration_source_events
    assert handle not in engine._persistent_requests
    assert engine._client.finalized[-1].source_lease_released
