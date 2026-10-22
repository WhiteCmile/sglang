from __future__ import annotations

import gc
import time
import weakref
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from threading import Event, Lock
from types import SimpleNamespace

import pytest
import torch
from dt_protocol import (
    PROTOCOL_VERSION,
    DraftKvMigrationExport,
    DraftKvMigrationPrepare,
    EntryStatus,
)
from sglang.srt.speculative.dt_draft_engine import (
    FlashMTPDraftEngine,
    PendingDFlashCompletion,
    PendingDraftTask,
    PendingEagle3Completion,
    PendingFeatureBatch,
)
from sglang.srt.speculative.dt_feature_p2p import (
    DraftFeaturePullTransport,
    FeatureTransportError,
    FeatureTransportQuiescenceError,
)


class _FakeStream:
    cuda_stream = 17

    def __init__(self, *, fail_fence: bool = False) -> None:
        self.fail_fence = fail_fence
        self.waited = []
        self.synchronize_count = 0

    def wait_event(self, event) -> None:
        self.waited.append(event)

    def synchronize(self) -> None:
        self.synchronize_count += 1
        if self.fail_fence:
            raise RuntimeError("injected fence failure")


class _FakeMappingCache:
    def __init__(self) -> None:
        self.maps = []
        self.invalidated = []
        self.closed = False

    def map(self, **kwargs):
        self.maps.append(kwargs)
        return 10_000 * len(self.maps), len(self.maps) > 1

    def invalidate(self, peer: str) -> None:
        self.invalidated.append(peer)

    def close(self) -> None:
        self.closed = True


class _FakeTensor:
    def __init__(self, address: int) -> None:
        self._address = address

    def data_ptr(self) -> int:
        return self._address


class _TrackingCompletionEvent:
    def __init__(self, *, query_error: Exception | None = None) -> None:
        self.query_error = query_error
        self.query_calls = 0
        self.synchronize_calls = 0

    def query(self) -> bool:
        self.query_calls += 1
        if self.query_error is not None:
            raise self.query_error
        return False

    def synchronize(self) -> None:
        self.synchronize_calls += 1


class _TrackingFuture:
    def __init__(self) -> None:
        self.done_calls = 0

    def done(self) -> bool:
        self.done_calls += 1
        return False


class _TrackingFeatureTransport:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def _source(
    peer: str,
    *,
    session: str = "session-1",
    generation: str = "generation-1",
    transport_session: str = "transport-1",
    event: bytes = b"event-1",
) -> SimpleNamespace:
    return SimpleNamespace(
        target_instance_id=peer,
        source_session_id=session,
        pool_generation=generation,
        transport_session=transport_session,
        ipc_handle=f"handle-{peer}".encode(),
        allocation_offset=4,
        byte_offset=8,
        nbytes=16,
        producer_event_handle=event,
    )


def _transport() -> DraftFeaturePullTransport:
    transport = object.__new__(DraftFeaturePullTransport)
    transport.device = SimpleNamespace()
    transport.device_index = 0
    transport._runtime = SimpleNamespace()
    transport._mapping_cache = _FakeMappingCache()
    transport._copy_stream = _FakeStream()
    transport._executor = ThreadPoolExecutor(max_workers=1)
    transport._fatal_error = None
    transport._closed = False
    transport._lock = Lock()
    transport._validate_submission = lambda *_args, **_kwargs: None
    return transport


def test_successful_measured_feature_pulls_account_for_proposal_boundaries() -> None:
    """Successful copies retain materialization work without inflating proposals."""

    def task(
        suffix: str, *, rows: int, requires_proposal: bool
    ) -> SimpleNamespace:
        return SimpleNamespace(
            transfer_id=f"transfer-{suffix}",
            request_id=f"dt-benchmark-measurement-{suffix}",
            round_id=0,
            draft_kv_handle=f"handle-{suffix}",
            source_read_migration_id="",
            deadline_ns=time.monotonic_ns() + 1_000_000_000,
            requires_proposal=requires_proposal,
            source=SimpleNamespace(
                nbytes=rows * 2,
                shape=(rows, 1),
                target_instance_id="target-0",
                source_session_id="target-session",
                pool_generation="target-pool",
                transport_session="target-transport",
            ),
        )

    materialize = task("000000", rows=2, requires_proposal=False)
    propose = task("000001", rows=3, requires_proposal=True)
    pending_batches = []
    for index, measured_task in enumerate((materialize, propose)):
        future = Future()
        future.set_result(
            SimpleNamespace(
                submitted_ns=1,
                gpu_completed_ns=2,
                mapping_ns=0,
                driver_submit_ns=0,
                gpu_copy_ns=1,
                feature_bytes=measured_task.source.nbytes,
                cold_mapping_count=0,
                segment_count=1,
                peer_count=1,
            )
        )
        pending_batches.append(
            PendingFeatureBatch(
                batch_id=f"measured-{index}",
                pending=[PendingDraftTask(task=measured_task)],
                staging_slot=index,
                row_ranges=((0, measured_task.source.shape[0]),),
                future=future,
                submitted_ns=1,
            )
        )

    engine = FlashMTPDraftEngine.__new__(FlashMTPDraftEngine)
    engine.instance_id = "draft-0"
    engine.profile = SimpleNamespace(method="dflash")
    engine._persistent = True
    engine._is_feature_backed = True
    engine._is_eagle3 = False
    engine.dflash_submit_pipeline = False
    engine.batch_source_read_done = False
    engine._stop = False
    engine._fatal_gpu_quarantine_reason = None
    engine._pending_feature_batches = deque(pending_batches)
    engine._input_ready_features = deque()
    engine._feature_staging_refcounts = {0: 1, 1: 1}
    engine._feature_staging_compute_busy = set()
    engine._pending_flashmtp_completions = deque()
    engine._pending_dflash_completions = deque()
    engine._pending_eagle3_completions = deque()
    engine._feature_source_inflight = set()
    engine._feature_source_quiescent = set()
    engine._feature_source_unquiescent = set()
    engine._deferred_feature_source_tasks = {}
    engine._quarantined_feature_draft_handles = set()
    engine._free_feature_staging_slots = deque()
    engine._cancelled_ids = set()
    engine._direct_gpu_measurement_evidence_emitted = False
    engine._direct_gpu_measurement_epochs = {}
    engine._direct_gpu_measurement_requests = {}
    engine._last_direct_gpu_measurement_ns = 0
    engine.max_batch_size = 8
    engine.feature_buffer_max_tokens = 8

    source_done = []
    launched = []
    capacity_updates = []
    engine._publish_source_done = source_done.append
    engine._launch_dflash_materialization = lambda **kwargs: launched.append(kwargs)
    engine._resume_feature_finalizations = lambda: None
    engine._publish_capacity = lambda: capacity_updates.append("capacity")

    engine._poll_feature_batches()
    assert len(engine._input_ready_features) == 2
    assert not launched
    while ready := engine._take_input_ready_batch():
        engine._execute_input_ready_batch(ready)

    assert [task.transfer_id for task in source_done] == [
        materialize.transfer_id,
        propose.transfer_id,
    ]
    assert len(launched) == 2
    assert capacity_updates == ["capacity"]
    assert engine._direct_gpu_measurement_epoch_evidence() == [
        {
            "target_instance_id": "target-0",
            "source_session_id": "target-session",
            "pool_generation": "target-pool",
            "transport_session": "target-transport",
            "batches": 2,
            "measurement_rows": 5,
            "measurement_requests": 2,
            "measurement_proposal_requests": 1,
        }
    ]
    assert engine._direct_gpu_measurement_request_evidence() == [
        {
            "target_instance_id": "target-0",
            "source_session_id": "target-session",
            "pool_generation": "target-pool",
            "transport_session": "target-transport",
            "request_id": materialize.request_id,
            "measurement_rows": 2,
            "measurement_requests": 1,
            "measurement_proposal_requests": 0,
        },
        {
            "target_instance_id": "target-0",
            "source_session_id": "target-session",
            "pool_generation": "target-pool",
            "transport_session": "target-transport",
            "request_id": propose.request_id,
            "measurement_rows": 3,
            "measurement_requests": 1,
            "measurement_proposal_requests": 1,
        },
    ]


def test_feature_batch_promotes_each_request_to_input_ready_independently() -> None:
    def make_task(transfer_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            transfer_id=transfer_id,
            request_id=f"request-{transfer_id}",
            round_id=1,
            draft_kv_handle="",
            source_read_migration_id="",
            deadline_ns=time.monotonic_ns() + 1_000_000_000,
            requires_proposal=True,
            source=SimpleNamespace(
                nbytes=2,
                shape=(1, 1),
                target_instance_id="target-0",
                source_session_id="target-session",
                pool_generation="target-pool",
                transport_session="target-transport",
            ),
        )

    def metrics() -> SimpleNamespace:
        return SimpleNamespace(
            submitted_ns=1,
            gpu_completed_ns=2,
            mapping_ns=0,
            driver_submit_ns=0,
            gpu_copy_ns=1,
            feature_bytes=2,
            cold_mapping_count=0,
        )

    fast = make_task("fast")
    slow = make_task("slow")
    fast_future = Future()
    fast_future.set_result(metrics())
    slow_future = Future()
    engine = FlashMTPDraftEngine.__new__(FlashMTPDraftEngine)
    engine.instance_id = "draft-0"
    engine._persistent = False
    engine._is_feature_backed = False
    engine._fatal_gpu_quarantine_reason = None
    engine._pending_feature_batches = deque(
        (
            PendingFeatureBatch(
                batch_id="mixed-readiness",
                pending=[PendingDraftTask(fast), PendingDraftTask(slow)],
                staging_slot=0,
                row_ranges=((0, 1), (1, 2)),
                futures=(fast_future, slow_future),
                submitted_ns=1,
            ),
        )
    )
    engine._input_ready_features = deque()
    engine._feature_staging_refcounts = {0: 2}
    engine._feature_staging_compute_busy = set()
    engine._pending_flashmtp_completions = deque()
    engine._pending_dflash_completions = deque()
    engine._pending_eagle3_completions = deque()
    engine._feature_source_inflight = set()
    engine._feature_source_quiescent = set()
    engine._feature_source_unquiescent = set()
    engine._deferred_feature_source_tasks = {}
    engine._quarantined_feature_draft_handles = set()
    engine._cancelled_ids = set()
    engine.batch_source_read_done = False
    engine.max_batch_size = 8
    engine.feature_buffer_max_tokens = 8
    source_done = []
    engine._publish_source_done = source_done.append
    engine._publish_failures = lambda failures: pytest.fail(str(failures))
    engine._publish_capacity = lambda: None

    engine._poll_feature_batches()

    assert [entry.pending.task.transfer_id for entry in engine._input_ready_features] == [
        "fast"
    ]
    assert [item.task.transfer_id for item in engine._pending_feature_batches[0].pending] == [
        "slow"
    ]
    selected = engine._take_input_ready_batch()
    assert [entry.pending.task.transfer_id for entry in selected] == ["fast"]
    assert source_done == [fast]

    # The unfinished entry is not part of that compute selection and can
    # independently become READY later in the same staging arena.
    engine._feature_staging_compute_busy.clear()
    slow_future.set_result(metrics())
    engine._poll_feature_batches()
    assert [entry.pending.task.transfer_id for entry in engine._input_ready_features] == [
        "slow"
    ]


def test_feature_batch_rejects_mixed_generation_before_queueing() -> None:
    transport = _transport()
    sources = (
        _source("target-a"),
        _source("target-a", generation="generation-2"),
    )
    with pytest.raises(FeatureTransportError, match="mix generations"):
        transport.submit_batch(
            sources=sources,
            destinations=(_FakeTensor(100), _FakeTensor(200)),
        )
    assert transport._mapping_cache.maps == []
    transport.close()


def test_flashmtp_pivot_layout_canonicalizes_string_runner_device() -> None:
    engine = FlashMTPDraftEngine.__new__(FlashMTPDraftEngine)
    engine.device = "cuda"
    engine.gpu_id = 0
    engine._pivot_width = 12
    engine._feature_staging_buffers = [
        SimpleNamespace(device=torch.device("cuda", 0)),
        SimpleNamespace(device=torch.device("cuda", 0)),
    ]
    valid = SimpleNamespace(
        device=torch.device("cuda", 0),
        dtype=torch.bfloat16,
        shape=(2, 12),
    )

    assert engine._require_flashmtp_pivot_layout(valid, batch_size=2) == torch.device(
        "cuda", 0
    )

    for invalid in (
        SimpleNamespace(
            device=torch.device("cuda", 1),
            dtype=torch.bfloat16,
            shape=(2, 12),
        ),
        SimpleNamespace(
            device=torch.device("cuda", 0),
            dtype=torch.float32,
            shape=(2, 12),
        ),
        SimpleNamespace(
            device=torch.device("cuda", 0),
            dtype=torch.bfloat16,
            shape=(2, 11),
        ),
    ):
        with pytest.raises(ValueError, match="invalid GPU layout"):
            engine._require_flashmtp_pivot_layout(invalid, batch_size=2)


def test_feature_batch_uses_one_copy_and_deduplicates_producer_waits(
    monkeypatch,
) -> None:
    transport = _transport()
    copied = []

    class _Runtime:
        @staticmethod
        def copy_batch_async(segments, stream) -> None:
            copied.append((segments, stream))

    class _CudaEvent:
        def __init__(self, *, enable_timing=False) -> None:
            self.enable_timing = enable_timing

        @classmethod
        def from_ipc_handle(cls, _device, handle):
            return ("imported", handle)

        def record(self, _stream) -> None:
            pass

        def synchronize(self) -> None:
            pass

        def elapsed_time(self, _other) -> float:
            return 0.25

    transport._runtime = _Runtime()
    monkeypatch.setattr("torch.cuda.set_device", lambda _device: None)
    monkeypatch.setattr("torch.cuda.Event", _CudaEvent)
    monkeypatch.setattr("torch.cuda.stream", lambda _stream: nullcontext())
    sources = (
        _source("target-a", event=b"shared-event"),
        _source("target-b", event=b"shared-event"),
        _source("target-a", event=b"shared-event"),
    )
    metrics = transport.submit_batch(
        sources=sources,
        destinations=(_FakeTensor(100), _FakeTensor(200), _FakeTensor(300)),
    ).result(timeout=1)

    assert len(copied) == 1
    assert len(copied[0][0]) == 3
    # Event handles are process/session scoped, so equal bytes from two peers
    # remain distinct while repeated descriptors from one peer are deduplicated.
    assert len(transport._copy_stream.waited) == 2
    assert metrics.segment_count == 3
    assert metrics.peer_count == 2
    assert metrics.feature_bytes == 48
    assert metrics.gpu_copy_ns == 250_000
    transport.close()


@pytest.mark.parametrize("fail_fence", [False, True])
def test_feature_batch_failure_fences_or_quarantines(monkeypatch, fail_fence) -> None:
    transport = _transport()
    transport._copy_stream = _FakeStream(fail_fence=fail_fence)

    class _Runtime:
        @staticmethod
        def copy_batch_async(_segments, _stream) -> None:
            raise RuntimeError("injected copy failure")

    class _CudaEvent:
        def __init__(self, *, enable_timing=False) -> None:
            self.enable_timing = enable_timing

        def record(self, _stream) -> None:
            pass

    transport._runtime = _Runtime()
    monkeypatch.setattr("torch.cuda.Event", _CudaEvent)
    monkeypatch.setattr("torch.cuda.stream", lambda _stream: nullcontext())

    expected = FeatureTransportQuiescenceError if fail_fence else RuntimeError
    with pytest.raises(expected):
        transport._copy_batch_and_wait(
            source_ready=(object(),),
            segments=(SimpleNamespace(length=16),),
            mapping_started_ns=1,
            mapping_done_ns=2,
            feature_bytes=16,
            cold_mapping_count=1,
            peer_count=1,
        )
    assert transport._copy_stream.synchronize_count == 1
    assert (transport._fatal_error is not None) is fail_fence
    transport.close()


def test_feature_batch_close_finishes_running_and_cancels_queued_work() -> None:
    transport = _transport()
    started = Event()
    release = Event()

    def blocked_batch(_sources, _destinations):
        started.set()
        assert release.wait(timeout=2)
        return "done"

    transport._run_batch = blocked_batch
    first = transport.submit_batch(
        sources=(_source("target-a"),),
        destinations=(_FakeTensor(100),),
    )
    second = transport.submit_batch(
        sources=(_source("target-b"),),
        destinations=(_FakeTensor(200),),
    )
    assert started.wait(timeout=1)
    with ThreadPoolExecutor(max_workers=1) as caller:
        closing = caller.submit(transport.close)
        deadline = time.monotonic() + 1
        while not second.cancelled() and time.monotonic() < deadline:
            time.sleep(0.001)
        assert second.cancelled()
        release.set()
        closing.result(timeout=1)
    assert first.done() and first.result() == "done"
    assert second.done() and second.cancelled()
    assert transport._mapping_cache.closed


def test_unquiesced_feature_batch_keeps_staging_and_draft_kv_quarantined() -> None:
    future = Future()
    future.set_exception(FeatureTransportQuiescenceError("injected fence failure"))
    task = SimpleNamespace(
        transfer_id="transfer-1",
        request_id="request-1",
        round_id=1,
        draft_kv_handle="handle-1",
        source_read_migration_id="",
        source=SimpleNamespace(source_session_id="source-session-1"),
    )
    next_future = Future()
    next_future.set_exception(FeatureTransportError("transport quarantined"))
    next_task = SimpleNamespace(
        transfer_id="transfer-2",
        request_id="request-1",
        round_id=2,
        draft_kv_handle="handle-1",
        source_read_migration_id="",
        source=SimpleNamespace(source_session_id="source-session-1"),
    )
    engine = FlashMTPDraftEngine.__new__(FlashMTPDraftEngine)
    engine._is_feature_backed = True
    engine._persistent = True
    engine._pending_feature_batches = deque(
        (
            PendingFeatureBatch(
                batch_id="batch-1",
                pending=[PendingDraftTask(task=task)],
                staging_slot=0,
                row_ranges=((0, 1),),
                future=future,
                submitted_ns=1,
            ),
            PendingFeatureBatch(
                batch_id="batch-2",
                pending=[PendingDraftTask(task=next_task)],
                staging_slot=1,
                row_ranges=((0, 1),),
                future=next_future,
                submitted_ns=2,
            ),
        )
    )
    engine._input_ready_features = deque()
    engine._feature_staging_refcounts = {0: 1, 1: 1}
    engine._feature_staging_compute_busy = set()
    engine._pending_flashmtp_completions = deque()
    engine._pending_dflash_completions = deque()
    engine._pending_eagle3_completions = deque()
    engine._quarantined_eagle3_completions = []
    engine.dflash_submit_pipeline = False
    engine._feature_source_inflight = set()
    engine._feature_source_quiescent = set()
    engine._feature_source_unquiescent = set()
    engine._quarantined_feature_draft_handles = set()
    engine._deferred_feature_source_tasks = {}
    engine._free_feature_staging_slots = deque()
    engine._stop = False
    engine.batch_source_read_done = False
    releases = []
    failures = []
    source_done = []
    engine._release_persistent_request = releases.append
    engine._publish_failures = failures.extend
    engine._publish_source_done = source_done.append
    engine._resume_feature_finalizations = lambda: None
    engine._publish_capacity = lambda: None

    FlashMTPDraftEngine._poll_feature_batches(engine)

    assert releases == []
    assert list(engine._free_feature_staging_slots) == []
    assert source_done == []
    assert engine._stop is True
    assert engine._fatal_gpu_quarantine_reason == "unquiesced feature-copy failure"
    assert len(engine._pending_feature_batches) == 1
    assert engine._quarantined_feature_draft_handles == {"handle-1"}
    assert [(failed.transfer_id, type(exc)) for failed, exc in failures] == [
        ("transfer-1", FeatureTransportQuiescenceError)
    ]

    engine._persistent_requests = {
        "handle-1": SimpleNamespace(request_id="request-1")
    }
    engine._deferred_persistent_releases = {}
    engine._migration_source_handles = {}
    assert not FlashMTPDraftEngine._release_persistent_request(engine, "handle-1")
    assert "handle-1" in engine._persistent_requests
    del engine._release_persistent_request

    engine._pending = deque()
    engine._cancelled_ids = set()
    engine._pending_ids = set()
    engine._received_ns_by_transfer = {}
    engine._feature_pull_transport = None
    engine._poll_deferred_migration_gpu_controls = lambda: False
    engine._delivery_outbox = SimpleNamespace(pending=lambda: ())
    engine._flush_source_done = lambda: None
    with pytest.raises(RuntimeError, match="process teardown"):
        FlashMTPDraftEngine.close(engine)
    assert "handle-1" in engine._persistent_requests


@pytest.mark.parametrize("failure_stage", ("copy", "proposal"))
def test_fatal_gpu_quarantine_blocks_deferred_migration_and_resource_reuse(
    failure_stage: str,
) -> None:
    """An unquiesced feature failure makes later GPU ownership immutable."""

    task = SimpleNamespace(
        transfer_id=f"transfer-{failure_stage}",
        request_id="request-1",
        round_id=1,
        draft_kv_handle="handle-1",
        source_read_migration_id="",
        deadline_ns=time.monotonic_ns() + 1_000_000_000,
        requires_proposal=True,
        source=SimpleNamespace(
            nbytes=32,
            target_instance_id="target-0",
            source_session_id="target-session",
            pool_generation="pool-generation",
            transport_session="transport-session",
        ),
    )
    feature_future = Future()
    if failure_stage == "copy":
        feature_future.set_exception(
            FeatureTransportQuiescenceError("injected unquiesced copy failure")
        )
    else:
        feature_future.set_result(
            SimpleNamespace(
                submitted_ns=1,
                gpu_completed_ns=2,
                mapping_ns=0,
                driver_submit_ns=0,
                gpu_copy_ns=1,
                feature_bytes=32,
                cold_mapping_count=0,
                segment_count=1,
                peer_count=1,
            )
        )

    engine = FlashMTPDraftEngine.__new__(FlashMTPDraftEngine)
    engine.instance_id = "draft-0"
    engine.device = torch.device("cuda:0")
    engine._is_feature_backed = True
    engine._is_eagle3 = False
    engine._persistent = True
    engine.dflash_submit_pipeline = False
    engine._pending_feature_batches = deque(
        (
            PendingFeatureBatch(
                batch_id=f"batch-{failure_stage}",
                pending=[PendingDraftTask(task=task)],
                staging_slot=7,
                row_ranges=((0, 1),),
                future=feature_future,
                submitted_ns=1,
            ),
        )
    )
    engine._input_ready_features = deque()
    engine._feature_staging_refcounts = {7: 1}
    engine._feature_staging_compute_busy = set()
    engine._pending_flashmtp_completions = deque()
    engine._pending_dflash_completions = deque()
    engine._pending_eagle3_completions = deque()
    engine._quarantined_eagle3_completions = []
    engine._feature_source_inflight = set()
    engine._feature_source_quiescent = set()
    engine._feature_source_unquiescent = set()
    engine._quarantined_feature_draft_handles = set()
    engine._deferred_feature_source_tasks = {}
    engine._free_feature_staging_slots = deque()
    engine._cancelled_ids = set()
    engine._stop = False
    engine._fatal_gpu_quarantine_reason = None
    engine.batch_source_read_done = False
    engine._persistent_requests = {
        "handle-1": SimpleNamespace(request_id="request-1")
    }
    engine._deferred_persistent_releases = {}
    engine._migration_source_handles = {}
    engine._prepared_migrations = {}
    engine.max_batch_size = 8
    engine.feature_buffer_max_tokens = 8

    control_type = (
        DraftKvMigrationPrepare
        if failure_stage == "copy"
        else DraftKvMigrationExport
    )
    migration_control = control_type(
        protocol_version=PROTOCOL_VERSION,
        migration_id=f"migration-{failure_stage}",
        source_draft_instance_id=(
            "draft-source" if failure_stage == "copy" else "draft-0"
        ),
        destination_draft_instance_id=(
            "draft-0" if failure_stage == "copy" else "draft-destination"
        ),
        target_instance_id="target-0",
        request_id="request-1",
        draft_kv_handle="handle-1",
        draft_profile_id="profile-0",
        round_id=1,
        committed_kv_len=2,
        deadline_ns=time.monotonic_ns() + 1_000_000_000,
    )
    control_key = (type(migration_control), migration_control.migration_id)
    engine._deferred_migration_gpu_controls = {
        control_key: SimpleNamespace(message=migration_control, deferred_ns=1)
    }
    finalize = SimpleNamespace(
        migration_id="migration-finalize", feature_transfer_id=""
    )
    engine._deferred_migration_finalizations = {
        finalize.migration_id: finalize
    }

    migration_gpu_calls = []
    finalization_calls = []
    failures = []
    source_done = []
    capacity_updates = []
    engine._handle_migration_prepare = migration_gpu_calls.append
    engine._handle_migration_export = migration_gpu_calls.append
    engine._handle_migration_finalize = finalization_calls.append
    engine._publish_failures = failures.extend
    engine._publish_source_done = source_done.append
    engine._publish_capacity = lambda: capacity_updates.append("capacity")

    if failure_stage == "proposal":
        proposal_error = RuntimeError("injected unquiesced proposal failure")
        proposal_error.resources_quiescent = False

        def fail_proposal(**_kwargs) -> None:
            raise proposal_error

        engine._launch_dflash_materialization = fail_proposal

    FlashMTPDraftEngine._poll_feature_batches(engine)
    if failure_stage == "proposal":
        ready = engine._take_input_ready_batch()
        assert len(ready) == 1
        engine._execute_input_ready_batch(ready)

    assert engine._stop
    assert engine._fatal_gpu_quarantine_active()
    assert [(failed.transfer_id, str(exc)) for failed, exc in failures] == [
        (task.transfer_id, str(feature_future.exception()))
        if failure_stage == "copy"
        else (task.transfer_id, str(proposal_error))
    ]
    assert capacity_updates == ["capacity"]
    assert source_done == ([] if failure_stage == "copy" else [task])
    assert list(engine._free_feature_staging_slots) == []
    assert engine._quarantined_feature_draft_handles == {"handle-1"}
    assert "handle-1" in engine._persistent_requests
    assert not engine._release_persistent_request("handle-1")
    assert "handle-1" in engine._persistent_requests
    assert migration_gpu_calls == []
    assert finalization_calls == []
    assert control_key in engine._deferred_migration_gpu_controls
    assert finalize.migration_id in engine._deferred_migration_finalizations

    # Explicit retries remain inert after the feature poll has returned.
    assert not engine._poll_deferred_migration_gpu_controls()
    engine._resume_feature_finalizations()
    assert migration_gpu_calls == []
    assert finalization_calls == []

    class FeatureTransport:
        closed = False

        def close(self) -> None:
            self.closed = True

    feature_transport = FeatureTransport()
    engine._feature_pull_transport = feature_transport
    with pytest.raises(RuntimeError, match="fatal GPU quarantine.*process teardown"):
        engine._quiesce_direct_gpu_source_shutdown()
    assert capacity_updates == ["capacity", "capacity"]
    assert not feature_transport.closed
    assert list(engine._free_feature_staging_slots) == []
    assert "handle-1" in engine._persistent_requests


def test_dflash_proposal_recovery_fence_failure_retains_all_gpu_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recovery-fence error itself must force permanent quarantine."""

    proposal_error = RuntimeError("injected DFlash proposal failure")
    synchronization_error = RuntimeError("injected DFlash recovery fence failure")

    class FailingProposalAllocation:
        def view(self, *_shape):
            raise proposal_error

    class TrackingAllocator:
        def __init__(self) -> None:
            self.allocation = FailingProposalAllocation()
            self.free_calls = []

        def alloc(self, _size):
            return self.allocation

        def free(self, allocation) -> None:
            self.free_calls.append(allocation)

    class FailOnceRecoveryStream:
        def __init__(self) -> None:
            self.synchronize_calls = 0

        def synchronize(self) -> None:
            self.synchronize_calls += 1
            if self.synchronize_calls == 1:
                raise synchronization_error
            # A second synchronization would appear to make the resources safe.
            # The outer poll must never attempt it after the first uncertain fence.

    allocator = TrackingAllocator()
    recovery_stream = FailOnceRecoveryStream()
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda *_args, **_kwargs: recovery_stream,
    )

    task = SimpleNamespace(
        transfer_id="transfer-proposal-recovery",
        request_id="request-1",
        round_id=1,
        draft_kv_handle="handle-1",
        source_read_migration_id="",
        deadline_ns=time.monotonic_ns() + 1_000_000_000,
        requires_proposal=True,
        source=SimpleNamespace(
            nbytes=32,
            target_instance_id="target-0",
            source_session_id="target-session",
            pool_generation="pool-generation",
            transport_session="transport-session",
        ),
    )
    completed_copy = Future()
    completed_copy.set_result(
        SimpleNamespace(
            submitted_ns=1,
            gpu_completed_ns=2,
            mapping_ns=0,
            driver_submit_ns=0,
            gpu_copy_ns=1,
            feature_bytes=32,
            cold_mapping_count=0,
            segment_count=1,
            peer_count=1,
        )
    )

    engine = FlashMTPDraftEngine.__new__(FlashMTPDraftEngine)
    engine.instance_id = "draft-0"
    engine.device = torch.device("cuda:0")
    engine.profile = SimpleNamespace(block_size=2)
    engine._draft_model_runner = SimpleNamespace(
        req_to_token_pool=object(),
        token_to_kv_pool_allocator=allocator,
    )
    engine._persistent = True
    engine._is_feature_backed = True
    engine._is_eagle3 = False
    engine.dflash_submit_pipeline = False
    engine.batch_source_read_done = False
    engine._stop = False
    engine._fatal_gpu_quarantine_reason = None
    engine._feature_pull_transport = None
    engine._p2p_transport = None
    engine._pending_feature_batches = deque(
        (
            PendingFeatureBatch(
                batch_id="proposal-recovery-batch",
                pending=[PendingDraftTask(task=task)],
                staging_slot=7,
                row_ranges=((0, 1),),
                future=completed_copy,
                submitted_ns=1,
            ),
        )
    )
    engine._input_ready_features = deque()
    engine._feature_staging_refcounts = {7: 1}
    engine._feature_staging_compute_busy = set()
    engine._pending_flashmtp_completions = deque()
    engine._pending_dflash_completions = deque()
    engine._pending_eagle3_completions = deque()
    engine._feature_source_inflight = set()
    engine._feature_source_quiescent = set()
    engine._feature_source_unquiescent = set()
    engine._deferred_feature_source_tasks = {}
    engine._quarantined_feature_draft_handles = set()
    engine._free_feature_staging_slots = deque()
    engine._cancelled_ids = set()
    engine.max_batch_size = 8
    engine.feature_buffer_max_tokens = 8

    persistent_owner = SimpleNamespace(request_id="request-1")
    prepared_owner = SimpleNamespace(transfer_future=None)
    source_handle_owner = object()
    deferred_finalization = SimpleNamespace(
        migration_id="migration-1", feature_transfer_id=""
    )
    engine._persistent_requests = {"handle-1": persistent_owner}
    engine._deferred_persistent_releases = {}
    engine._prepared_migrations = {"migration-1": prepared_owner}
    engine._migration_source_handles = {"migration-1": source_handle_owner}
    engine._deferred_migration_finalizations = {
        "migration-1": deferred_finalization
    }

    release_calls = []

    def release_persistent(handle: str) -> bool:
        release_calls.append(handle)
        engine._persistent_requests.pop(handle, None)
        return True

    failures = []
    source_done = []
    capacity_updates = []
    migration_finalizations = []
    engine._release_persistent_request = release_persistent
    engine._publish_failures = failures.extend
    engine._publish_source_done = source_done.append
    engine._publish_capacity = lambda: capacity_updates.append("capacity")
    engine._handle_migration_finalize = migration_finalizations.append

    proposal_state = SimpleNamespace(request_id="request-1")

    def launch_through_real_proposal_handler(**_kwargs) -> None:
        FlashMTPDraftEngine._run_dflash_proposal(
            engine,
            states=[proposal_state],
            req_pool_indices=object(),
            prefix_lens=object(),
            seq_lens_cpu=object(),
            bonus_tokens=object(),
        )

    engine._launch_dflash_materialization = launch_through_real_proposal_handler

    FlashMTPDraftEngine._poll_feature_batches(engine)
    ready = engine._take_input_ready_batch()
    assert len(ready) == 1
    engine._execute_input_ready_batch(ready)

    assert recovery_stream.synchronize_calls == 1
    assert synchronization_error.resources_quiescent is False
    assert synchronization_error.__cause__ is proposal_error
    assert proposal_error.resources_quiescent is False
    assert engine._stop
    assert (
        engine._fatal_gpu_quarantine_reason
        == "unquiesced feature proposal/materialization failure"
    )
    assert [(failed.transfer_id, exc) for failed, exc in failures] == [
        (task.transfer_id, synchronization_error)
    ]
    assert release_calls == []
    assert engine._persistent_requests == {"handle-1": persistent_owner}
    assert engine._quarantined_feature_draft_handles == {"handle-1"}
    assert list(engine._free_feature_staging_slots) == []
    assert allocator.free_calls == []
    assert engine._prepared_migrations == {"migration-1": prepared_owner}
    assert engine._migration_source_handles == {
        "migration-1": source_handle_owner
    }
    assert engine._deferred_migration_finalizations == {
        "migration-1": deferred_finalization
    }
    assert migration_finalizations == []
    assert source_done == [task]
    assert capacity_updates == ["capacity"]


def test_dflash_candidate_copy_fence_failure_retains_registered_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A submitted D2H copy keeps every local owner after its fence fails."""

    trace = []

    class OwnedCandidate:
        pass

    class OwnedEvent:
        pass

    class CandidateRows:
        def copy_(self, _source, *, non_blocking: bool):
            assert non_blocking
            trace.append("copy_submitted")
            return self

    completion_event_refs = []

    class FailingCompletionEvent:
        def __init__(self) -> None:
            completion_event_refs.append(weakref.ref(self))

        def record(self, _stream) -> None:
            trace.append("completion_record_failed")
            raise RuntimeError("injected candidate completion record failure")

    class FailingCopyStream:
        def __init__(self) -> None:
            self.synchronize_calls = 0

        def wait_event(self, _event) -> None:
            trace.append("proposal_wait_submitted")

        def synchronize(self) -> None:
            self.synchronize_calls += 1
            trace.append("copy_fence_failed")
            raise RuntimeError("injected candidate copy fence failure")

    monkeypatch.setattr(torch.cuda, "Event", FailingCompletionEvent)
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())

    candidates = OwnedCandidate()
    candidate_rows = CandidateRows()
    materialize_start = OwnedEvent()
    materialize_done = OwnedEvent()
    proposal_done = OwnedEvent()
    owner_refs = {
        "candidates": weakref.ref(candidates),
        "candidate_rows": weakref.ref(candidate_rows),
        "materialize_start": weakref.ref(materialize_start),
        "materialize_done": weakref.ref(materialize_done),
        "proposal_done": weakref.ref(proposal_done),
    }
    completion = PendingDFlashCompletion(
        pending=[],
        staging_slots=(7,),
        staging_slot_counts=(0,),
        completion_event=proposal_done,
        materialize_start_event=materialize_start,
        materialize_done_event=materialize_done,
        proposal_done_event=proposal_done,
        proposal_tasks=(),
        candidates=candidates,
        candidate_rows_cpu=candidate_rows,
        launch_started_ns=1,
        tasks_prepared_ns=2,
        proposal_metadata_ready_ns=3,
        materialize_submitted_ns=4,
        launched_ns=0,
        proposal_execution="graph",
        proposal_fallback_reason=None,
        reconcile_prepare_phase_ns=(),
        prior_unfinished_proposal_count=0,
        prior_candidate_copy_pending_count=0,
    )
    completion_identity = id(completion)
    copy_stream = FailingCopyStream()

    engine = FlashMTPDraftEngine.__new__(FlashMTPDraftEngine)
    engine._pending_dflash_completions = deque()
    engine._quarantined_dflash_completions = []
    engine._fatal_gpu_quarantine_reason = None
    engine._stop = False
    engine._p2p_transport = None
    engine._feature_pull_transport = None

    try:
        FlashMTPDraftEngine._register_dflash_completion(
            engine,
            completion=completion,
            copy_stream=copy_stream,
        )
    except RuntimeError as propagated:
        assert str(propagated) == "injected candidate copy fence failure"
        record_error = propagated.__cause__
        assert record_error is not None
        assert str(record_error) == "injected candidate completion record failure"
        assert propagated.resources_quiescent is False
        assert record_error.resources_quiescent is False
        # The engine-owned quarantine, not exception tracebacks, must keep CUDA
        # inputs/events alive after result publication discards this exception.
        propagated.__traceback__ = None
        record_error.__traceback__ = None
        propagated.__cause__ = None
        propagated.__context__ = None
        record_error.__context__ = None
        del record_error
    else:  # pragma: no cover - the fault injection must reach both failures
        raise AssertionError("candidate-copy fault injection did not fail")

    assert trace == [
        "proposal_wait_submitted",
        "copy_submitted",
        "completion_record_failed",
        "copy_fence_failed",
    ]
    assert copy_stream.synchronize_calls == 1
    assert list(engine._pending_dflash_completions) == []
    assert len(engine._quarantined_dflash_completions) == 1
    assert engine._stop
    assert (
        engine._fatal_gpu_quarantine_reason
        == "unquiesced DFlash candidate-copy failure"
    )

    del completion
    del candidates
    del candidate_rows
    del materialize_start
    del materialize_done
    del proposal_done
    gc.collect()

    quarantined = engine._quarantined_dflash_completions[0]
    assert id(quarantined) == completion_identity
    assert quarantined.candidates is owner_refs["candidates"]()
    assert quarantined.candidate_rows_cpu is owner_refs["candidate_rows"]()
    assert quarantined.materialize_start_event is owner_refs["materialize_start"]()
    assert quarantined.materialize_done_event is owner_refs["materialize_done"]()
    assert quarantined.proposal_done_event is owner_refs["proposal_done"]()
    assert quarantined.completion_event is completion_event_refs[0]()
    assert all(reference() is not None for reference in owner_refs.values())

    engine._is_feature_backed = True
    engine._persistent = True
    capacity_updates = []
    engine._publish_capacity = lambda: capacity_updates.append("capacity")
    for _attempt in range(2):
        with pytest.raises(
            RuntimeError,
            match="fatal GPU quarantine.*unquiesced DFlash candidate-copy failure",
        ):
            FlashMTPDraftEngine.close(engine)
    assert capacity_updates == ["capacity", "capacity"]
    assert copy_stream.synchronize_calls == 1
    assert len(engine._quarantined_dflash_completions) == 1
    assert id(engine._quarantined_dflash_completions[0]) == completion_identity


@pytest.mark.parametrize("failure_stage", ("copy", "eagle3"))
def test_fatal_gpu_quarantine_precedes_fallible_failure_publication(
    failure_stage: str,
) -> None:
    """A reporting failure cannot reopen the normal GPU shutdown path."""

    task = SimpleNamespace(
        transfer_id=f"transfer-{failure_stage}",
        request_id="request-1",
        round_id=1,
        attempt=0,
        target_instance_id="target-0",
        draft_kv_handle="handle-1",
        source_read_migration_id="",
        source=SimpleNamespace(source_session_id="target-session"),
    )
    engine = FlashMTPDraftEngine.__new__(FlashMTPDraftEngine)
    engine.instance_id = "draft-0"
    engine.EntryStatus = EntryStatus
    engine._result_sequence = 0
    engine._cancelled_ids = set()
    engine._persistent = True
    engine._is_feature_backed = True
    engine._is_eagle3 = failure_stage == "eagle3"
    engine._stop = False
    engine._fatal_gpu_quarantine_reason = None
    engine._quarantined_feature_draft_handles = set()
    engine._quarantined_eagle3_completions = []
    engine._pending_dflash_completions = deque()
    engine._pending_eagle3_completions = deque()
    engine._pending_feature_batches = deque()
    engine._free_feature_staging_slots = deque()
    engine._feature_source_inflight = set()
    engine._feature_source_quiescent = set()
    engine._feature_source_unquiescent = set()
    engine._deferred_feature_source_tasks = {}
    engine.dflash_submit_pipeline = False
    engine.batch_source_read_done = False
    capacity_updates = []
    engine._publish_capacity = lambda: capacity_updates.append("capacity")

    def fail_publication(_message) -> None:
        raise RuntimeError("injected failure publication error")

    engine._send_result = fail_publication

    if failure_stage == "copy":
        copy_future = Future()
        copy_future.set_exception(
            FeatureTransportQuiescenceError("injected unquiesced copy failure")
        )
        engine._pending_feature_batches.append(
            PendingFeatureBatch(
                batch_id="copy-batch",
                pending=[PendingDraftTask(task=task)],
                staging_slot=7,
                row_ranges=((0, 1),),
                future=copy_future,
                submitted_ns=1,
            )
        )
        poll = engine._poll_feature_batches
        expected_reason = "unquiesced feature-copy failure"
    else:
        completion_event = _TrackingCompletionEvent(
            query_error=RuntimeError("injected EAGLE3 completion query failure")
        )
        completion = PendingEagle3Completion(
            pending=[PendingDraftTask(task=task)],
            staging_slots=(7,),
            staging_slot_counts=(1,),
            completion_event=completion_event,
            submitted_ns=1,
        )
        engine._pending_eagle3_completions.append(completion)
        poll = engine._poll_eagle3_completions
        expected_reason = "unquiesced EAGLE3 completion-event failure"

    with pytest.raises(RuntimeError, match="failure publication"):
        poll()

    assert engine._stop
    assert engine._fatal_gpu_quarantine_reason == expected_reason
    assert engine._quarantined_feature_draft_handles == {"handle-1"}
    assert list(engine._free_feature_staging_slots) == []
    if failure_stage == "eagle3":
        assert list(engine._pending_eagle3_completions) == []
        assert engine._quarantined_eagle3_completions == [completion]

    engine._persistent_requests = {
        "handle-1": SimpleNamespace(request_id="request-1")
    }
    engine._deferred_persistent_releases = {}
    engine._migration_source_handles = {}
    assert not engine._release_persistent_request("handle-1")
    assert "handle-1" in engine._persistent_requests

    # If normal shutdown were reached, it would close the transport and query
    # or synchronize this retained completion.  Fatal shutdown must do neither.
    later_event = _TrackingCompletionEvent()
    engine._pending_dflash_completions.append(
        SimpleNamespace(completion_event=later_event, pending=())
    )
    feature_transport = _TrackingFeatureTransport()
    engine._feature_pull_transport = feature_transport
    with pytest.raises(RuntimeError, match="fatal GPU quarantine.*process teardown"):
        engine._quiesce_direct_gpu_source_shutdown()
    assert capacity_updates == ["capacity"]
    assert feature_transport.close_calls == 0
    assert later_event.query_calls == 0
    assert later_event.synchronize_calls == 0
    assert "handle-1" in engine._persistent_requests


def test_dflash_completion_query_failure_permanently_quarantines_gpu_work() -> None:
    """An uncertain DFlash completion blocks later batches and migration."""

    task = SimpleNamespace(
        transfer_id="transfer-dflash",
        request_id="request-1",
        round_id=1,
        draft_kv_handle="handle-1",
    )
    completion_event = _TrackingCompletionEvent(
        query_error=RuntimeError("injected DFlash completion query failure")
    )
    completion = PendingDFlashCompletion(
        pending=[PendingDraftTask(task=task)],
        staging_slots=(9,),
        staging_slot_counts=(1,),
        completion_event=completion_event,
        materialize_start_event=object(),
        materialize_done_event=object(),
        proposal_done_event=object(),
        proposal_tasks=(),
        candidates=None,
        candidate_rows_cpu=None,
        launch_started_ns=1,
        tasks_prepared_ns=2,
        proposal_metadata_ready_ns=3,
        materialize_submitted_ns=4,
        launched_ns=5,
        proposal_execution=None,
        proposal_fallback_reason=None,
        reconcile_prepare_phase_ns=(),
        prior_unfinished_proposal_count=0,
        prior_candidate_copy_pending_count=0,
    )
    queued_future = _TrackingFuture()
    queued_task = SimpleNamespace(transfer_id="transfer-queued")

    engine = FlashMTPDraftEngine.__new__(FlashMTPDraftEngine)
    engine._stop = False
    engine._fatal_gpu_quarantine_reason = None
    engine._persistent = True
    engine._is_feature_backed = True
    engine._pending_dflash_completions = deque((completion,))
    engine._pending_eagle3_completions = deque()
    engine._pending_feature_batches = deque(
        (
            PendingFeatureBatch(
                batch_id="queued-feature-batch",
                pending=[PendingDraftTask(task=queued_task)],
                staging_slot=10,
                row_ranges=((0, 1),),
                future=queued_future,
                submitted_ns=6,
            ),
        )
    )
    engine._quarantined_feature_draft_handles = set()
    engine._free_feature_staging_slots = deque()
    engine._deferred_migration_gpu_controls = {
        ("prepare", "migration-1"): SimpleNamespace(
            message=SimpleNamespace(migration_id="migration-1"), deferred_ns=1
        )
    }
    engine._deferred_migration_finalizations = {
        "migration-1": SimpleNamespace(
            migration_id="migration-1", feature_transfer_id=""
        )
    }
    migration_gpu_calls = []
    finalization_calls = []
    engine._handle_migration_prepare = migration_gpu_calls.append
    engine._handle_migration_export = migration_gpu_calls.append
    engine._handle_migration_finalize = finalization_calls.append
    failures = []
    capacity_updates = []
    engine._publish_failures = failures.extend
    engine._publish_capacity = lambda: capacity_updates.append("capacity")

    engine._poll_dflash_completions()

    assert engine._stop
    assert (
        engine._fatal_gpu_quarantine_reason
        == "unquiesced DFlash completion-event failure"
    )
    assert list(engine._pending_dflash_completions) == [completion]
    assert completion_event.query_calls == 1
    assert completion_event.synchronize_calls == 0
    assert engine._quarantined_feature_draft_handles == {"handle-1"}
    assert list(engine._free_feature_staging_slots) == []
    assert [(failed.transfer_id, str(exc)) for failed, exc in failures] == [
        ("transfer-dflash", "injected DFlash completion query failure")
    ]
    assert capacity_updates == ["capacity"]

    # Every explicit retry remains inert and retains the queued ownership.
    engine._poll_dflash_completions()
    engine._poll_feature_batches()
    assert not engine._poll_deferred_migration_gpu_controls()
    engine._resume_feature_finalizations()
    assert completion_event.query_calls == 1
    assert queued_future.done_calls == 0
    assert migration_gpu_calls == []
    assert finalization_calls == []
    assert len(engine._deferred_migration_gpu_controls) == 1
    assert "migration-1" in engine._deferred_migration_finalizations

    engine._persistent_requests = {
        "handle-1": SimpleNamespace(request_id="request-1")
    }
    engine._deferred_persistent_releases = {}
    engine._migration_source_handles = {}
    assert not engine._release_persistent_request("handle-1")
    assert "handle-1" in engine._persistent_requests

    feature_transport = _TrackingFeatureTransport()
    engine._feature_pull_transport = feature_transport
    with pytest.raises(RuntimeError, match="fatal GPU quarantine.*process teardown"):
        engine._quiesce_direct_gpu_source_shutdown()
    assert capacity_updates == ["capacity", "capacity"]
    assert feature_transport.close_calls == 0
    assert completion_event.query_calls == 1
    assert completion_event.synchronize_calls == 0
    assert queued_future.done_calls == 0
    assert list(engine._free_feature_staging_slots) == []
    assert "handle-1" in engine._persistent_requests


def test_engine_fatal_quarantine_survives_transport_notification_failure() -> None:
    """The engine latch does not depend on transport notification success."""

    calls = []

    def fail_quarantine() -> None:
        calls.append("quarantine")
        raise RuntimeError("injected transport quarantine failure")

    engine = FlashMTPDraftEngine.__new__(FlashMTPDraftEngine)
    engine._stop = False
    engine._fatal_gpu_quarantine_reason = None
    engine._feature_pull_transport = SimpleNamespace(quarantine=fail_quarantine)

    engine._enter_fatal_gpu_quarantine("injected engine failure")

    assert calls == ["quarantine"]
    assert engine._stop
    assert engine._fatal_gpu_quarantine_reason == "injected engine failure"


def test_engine_fatal_quarantine_fences_queued_feature_executor_callback(
    monkeypatch,
) -> None:
    """A queued pull cannot touch CUDA or IPC mappings after engine failure."""

    transport = _transport()
    worker_started = Event()
    release_worker = Event()
    cuda_device_calls = []
    copy_calls = []
    monkeypatch.setattr(
        "torch.cuda.set_device", lambda device: cuda_device_calls.append(device)
    )

    class _Runtime:
        @staticmethod
        def copy_batch_async(segments, stream) -> None:
            copy_calls.append((segments, stream))

    transport._runtime = _Runtime()

    def occupy_only_worker() -> None:
        worker_started.set()
        assert release_worker.wait(timeout=2)

    blocker = transport._executor.submit(occupy_only_worker)
    queued_pull = None
    try:
        assert worker_started.wait(timeout=1)
        queued_source = _source("target-queued")
        queued_pull = transport.submit_batch(
            sources=(queued_source,),
            destinations=(_FakeTensor(200),),
        )

        proposal_task = SimpleNamespace(
            transfer_id="transfer-proposal",
            request_id="request-1",
            round_id=1,
            draft_kv_handle="handle-1",
            source_read_migration_id="",
            deadline_ns=time.monotonic_ns() + 1_000_000_000,
            requires_proposal=True,
            source=SimpleNamespace(
                nbytes=32,
                target_instance_id="target-0",
                source_session_id="target-session",
                pool_generation="pool-generation",
                transport_session="transport-session",
            ),
        )
        queued_task = SimpleNamespace(
            transfer_id="transfer-queued",
            request_id="request-2",
            draft_kv_handle="handle-2",
            source_read_migration_id="",
            source=queued_source,
        )
        completed_copy = Future()
        completed_copy.set_result(
            SimpleNamespace(
                submitted_ns=1,
                gpu_completed_ns=2,
                mapping_ns=0,
                driver_submit_ns=0,
                gpu_copy_ns=1,
                feature_bytes=32,
                cold_mapping_count=0,
                segment_count=1,
                peer_count=1,
            )
        )

        engine = FlashMTPDraftEngine.__new__(FlashMTPDraftEngine)
        engine.instance_id = "draft-0"
        engine._feature_pull_transport = transport
        engine._stop = False
        engine._fatal_gpu_quarantine_reason = None
        engine._persistent = True
        engine._is_feature_backed = True
        engine._is_eagle3 = False
        engine.dflash_submit_pipeline = False
        engine.batch_source_read_done = False
        engine._pending = deque()
        engine._pending_feature_batches = deque(
            (
                PendingFeatureBatch(
                    batch_id="proposal-batch",
                    pending=[PendingDraftTask(task=proposal_task)],
                    staging_slot=7,
                    row_ranges=((0, 1),),
                    future=completed_copy,
                    submitted_ns=1,
                ),
                PendingFeatureBatch(
                    batch_id="queued-pull-batch",
                    pending=[PendingDraftTask(task=queued_task)],
                    staging_slot=8,
                    row_ranges=((0, 1),),
                    future=queued_pull,
                    submitted_ns=2,
                ),
            )
        )
        engine._input_ready_features = deque()
        engine._feature_staging_refcounts = {7: 1, 8: 1}
        engine._feature_staging_compute_busy = set()
        engine._pending_flashmtp_completions = deque()
        engine._pending_dflash_completions = deque()
        engine._pending_eagle3_completions = deque()
        engine._feature_source_inflight = set()
        engine._feature_source_quiescent = set()
        engine._feature_source_unquiescent = set()
        engine._deferred_feature_source_tasks = {}
        engine._quarantined_feature_draft_handles = set()
        engine._free_feature_staging_slots = deque()
        engine._cancelled_ids = set()
        engine.max_batch_size = 8
        engine.feature_buffer_max_tokens = 8
        source_done = []
        failures = []
        capacity_updates = []
        engine._publish_source_done = source_done.append
        engine._publish_failures = failures.extend
        engine._publish_capacity = lambda: capacity_updates.append("capacity")

        proposal_error = RuntimeError("injected unquiesced proposal failure")
        proposal_error.resources_quiescent = False

        def fail_proposal(**_kwargs) -> None:
            raise proposal_error

        engine._launch_dflash_materialization = fail_proposal
        engine._poll_feature_batches()
        ready = engine._take_input_ready_batch()
        assert len(ready) == 1
        engine._execute_input_ready_batch(ready)

        assert engine._stop
        assert (
            engine._fatal_gpu_quarantine_reason
            == "unquiesced feature proposal/materialization failure"
        )
        assert transport._fatal_error is not None
        assert len(engine._pending_feature_batches) == 1
        assert engine._pending_feature_batches[0].future is queued_pull
        assert list(engine._free_feature_staging_slots) == []
        assert engine._quarantined_feature_draft_handles == {"handle-1"}
        assert source_done == [proposal_task]
        assert [(task.transfer_id, exc) for task, exc in failures] == [
            (proposal_task.transfer_id, proposal_error)
        ]
        assert capacity_updates == ["capacity"]

        # Let the executor reach the callback only after engine quarantine.
        # Its transport fence must reject it before CUDA selection, mapping, or
        # copy submission, while the engine retains the pending source/KV owner.
        release_worker.set()
        blocker.result(timeout=1)
        with pytest.raises(
            FeatureTransportError, match="queued before transport quarantine"
        ):
            queued_pull.result(timeout=1)
        assert cuda_device_calls == []
        assert transport._mapping_cache.maps == []
        assert copy_calls == []
        assert not transport._mapping_cache.closed
        assert not transport._closed

        engine._persistent_requests = {
            "handle-1": SimpleNamespace(request_id="request-1"),
            "handle-2": SimpleNamespace(request_id="request-2"),
        }
        engine._deferred_persistent_releases = {}
        engine._migration_source_handles = {}
        assert not engine._release_persistent_request("handle-1")
        assert not engine._release_persistent_request("handle-2")
        assert set(engine._persistent_requests) == {"handle-1", "handle-2"}
    finally:
        release_worker.set()
        transport.close()

    # Test cleanup may stop the executor, but a quarantined transport must not
    # close mappings whose ownership could still be referenced by GPU work.
    assert transport._closed
    assert not transport._mapping_cache.closed
