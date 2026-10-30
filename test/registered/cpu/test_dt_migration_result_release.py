import json
import struct
import time
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from threading import Event, Lock
from types import SimpleNamespace

import msgspec
import pytest
import torch

import sglang.srt.speculative.dt_draft_engine as dt_draft_engine_module
import sglang.srt.speculative.dt_target as dt_target_module

from dt_client import RetryOutbox
from dt_protocol import (
    EntryStatus,
    FINALIZE_DRAFT_KV_MIGRATION,
    PROTOCOL_VERSION,
    RELEASE_DRAFT_REQUEST,
    CANCEL_TRANSFERS,
    CancelTransfers,
    DraftResultEntry,
    DraftTask,
    DraftKvMigrationFinalize,
    DraftRequestRelease,
    DraftTaskKind,
    PullDone,
    ResultBatchAck,
    SourceReadDone,
    SourceReadDoneBatch,
    SourceReadDoneBatchAck,
    SourceRef,
    TargetResultBatch,
    TargetTaskBatch,
    cancel_transfers_id,
    draft_request_release_id,
    draft_kv_migration_finalized_id,
    pull_done_id,
    source_read_done_id,
)
from sglang.srt.speculative.dt_draft_engine import (
    FlashMTPDraftEngine,
    PendingDraftTask,
    PendingFlashMTPCompletion,
    PublishedResult,
    _eagle3_feature_staging_view_bounds,
)
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_worker_v2 import split_dflash_committed_rows
from sglang.srt.speculative.dt_feature_p2p import (
    FeatureSlotState,
    FeatureTransportError,
    FeatureTransportQuiescenceError,
    FeatureTransferSlice,
    TargetFeatureSourcePool,
)
from sglang.srt.speculative.dt_candidate_p2p import (
    CandidateSlotState,
    TargetCandidateArena,
    _CandidateSlot,
)
from sglang.srt.speculative.dt_target import (
    DTTargetController,
    ParkedTargetRound,
    StagedFlashMTPPivotBatch,
    StagedFlashMTPTask,
    _batched_eagle3_token_metadata,
    _flashmtp_park_scalar_values,
    _flashmtp_pivot_preservation_indices,
    _flashmtp_pivot_publish_view,
    _parse_pipeline_lane_weights,
    _ready_batch_h2d,
    _ready_candidate_host_tensor,
)
from sglang.srt.speculative.eagle3_target_worker_v2 import (
    EAGLE3DisaggregatedTargetWorkerV2,
)
from sglang.srt.arg_groups.speculative_hook import _validate_dflash_draft_target
from sglang.srt.speculative import dt_kv_p2p
from sglang.srt.speculative.dt_kv_p2p import (
    DraftKvP2pTransport,
    TransferQuiescenceError,
)
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator


def test_direct_cancel_before_assignment_ack_retains_authenticated_generation() -> None:
    arena = TargetCandidateArena.__new__(TargetCandidateArena)
    arena._lock = Lock()
    arena._active = {"transfer": 0}
    arena._slots = [
        _CandidateSlot(
            generation=7,
            state=CandidateSlotState.ARMED,
            transfer_id="transfer",
            authorization_nonce=91,
        )
    ]

    assert arena.cancel("transfer", remote_quiescent=False) is False
    slot = arena._slots[0]
    assert slot.state is CandidateSlotState.QUARANTINED
    assert slot.cancel_pending is True
    assert arena._active == {"transfer": 0}

    arena.authorize(
        SimpleNamespace(
            transfer_id="transfer",
            authorization_nonce=91,
            draft_instance_id="draft-0",
            draft_session_id="draft-session",
        )
    )
    assert (slot.draft_instance_id, slot.draft_session_id) == (
        "draft-0",
        "draft-session",
    )
    assert slot.state is CandidateSlotState.QUARANTINED
    assert slot.cancel_pending is True


@pytest.mark.parametrize(
    ("component_class", "component_module", "role"),
    (
        (DTTargetController, dt_target_module, "target"),
        (FlashMTPDraftEngine, dt_draft_engine_module, "draft"),
    ),
)
def test_direct_gpu_measurement_evidence_is_one_shot(
    monkeypatch, component_class, component_module, role
) -> None:
    component = component_class.__new__(component_class)
    component._direct_gpu_measurement_evidence_emitted = False
    records = []
    monkeypatch.setattr(
        component_module.logger, "info", lambda *args: records.append(args)
    )
    payload = {
        "role": role,
        "method": "flashmtp",
        "transport": "cuda-ipc",
        "measurement_rows": 4,
        "source_session_id": "source-session",
        "pool_generation": "pool-generation",
        "transport_session": "transport-session",
    }

    # Warmup/unmeasured work never calls this qualifying-success helper and
    # therefore cannot consume the evidence latch.
    assert component._direct_gpu_measurement_evidence_emitted is False
    assert component._emit_direct_gpu_measurement_evidence_once(payload) is True
    assert component._emit_direct_gpu_measurement_evidence_once(payload) is False

    assert len(records) == 1
    assert records[0][1] == component_module.DT_DIRECT_GPU_SOURCE_MARKER
    assert json.loads(records[0][2]) == payload
    assert component._direct_gpu_measurement_evidence_emitted is True


def test_flashmtp_pivot_publish_view_skips_only_identity_gather() -> None:
    pivot = torch.arange(24, dtype=torch.bfloat16).reshape(4, 6)

    full, materialized = _flashmtp_pivot_publish_view(pivot, [0, 1, 2, 3])
    assert not materialized
    assert full.data_ptr() == pivot.data_ptr()
    assert full.is_contiguous()

    filtered, materialized = _flashmtp_pivot_publish_view(pivot, [0, 2])
    assert materialized
    assert filtered.is_contiguous()
    assert torch.equal(filtered, pivot[[0, 2]])
    assert filtered.data_ptr() != pivot.data_ptr()

    reordered, materialized = _flashmtp_pivot_publish_view(
        pivot, [3, 2, 1, 0]
    )
    assert materialized
    assert torch.equal(reordered, pivot[[3, 2, 1, 0]])

    noncontiguous = pivot[:, ::2]
    copied, materialized = _flashmtp_pivot_publish_view(
        noncontiguous, [0, 1, 2, 3]
    )
    assert materialized
    assert copied.is_contiguous()
    assert torch.equal(copied, noncontiguous)


def test_flashmtp_park_scalars_prefer_staged_cpu_rows_and_preserve_order() -> None:
    draft_input = DFlashDraftInputV2.create_idle_input(torch.device("cpu"))
    draft_input.bonus_tokens = torch.tensor([10, 11, 12], dtype=torch.int64)
    draft_input.new_seq_lens = torch.tensor([100, 101, 102], dtype=torch.int64)
    batch_result = SimpleNamespace(
        dt_flashmtp_bonus_tokens_cpu=torch.tensor([20, 21, 22], dtype=torch.int64),
        dt_flashmtp_seq_lens_cpu=torch.tensor([200, 201, 202], dtype=torch.int64),
    )

    bonus, seq_lens, staged = _flashmtp_park_scalar_values(
        batch_result=batch_result,
        draft_input=draft_input,
        active_indices=[2, 0],
        original_batch_size=3,
    )
    assert staged
    assert bonus == [22, 20]
    assert seq_lens == [202, 200]

    bonus, seq_lens, staged = _flashmtp_park_scalar_values(
        batch_result=SimpleNamespace(),
        draft_input=draft_input,
        active_indices=[1],
        original_batch_size=3,
    )
    assert not staged
    assert bonus == [11]
    assert seq_lens == [101]

    with pytest.raises(RuntimeError, match="staging is incomplete"):
        _flashmtp_park_scalar_values(
            batch_result=SimpleNamespace(
                dt_flashmtp_bonus_tokens_cpu=torch.tensor([20, 21, 22])
            ),
            draft_input=draft_input,
            active_indices=[0, 1, 2],
            original_batch_size=3,
        )


@pytest.mark.parametrize("is_token_only", (False, True))
def test_linear_chain_scalar_stage_adds_reverse_stream_edge(monkeypatch, is_token_only) -> None:
    from sglang.srt.managers import utils as manager_utils

    copied = []

    def async_d2h(value):
        copied.append(value)
        return value.detach().cpu().clone()

    monkeypatch.setattr(manager_utils, "_async_d2h", async_d2h)

    class Event:
        recorded = False

        def record(self) -> None:
            self.recorded = True

    event = Event()
    waited = []
    controller = object.__new__(DTTargetController)
    controller._profile_active = False
    controller.is_feature_backed = False
    controller.is_token_only = is_token_only
    pivots = []
    controller._stage_flashmtp_pivot_copy = lambda *args: pivots.append(args)
    controller.scheduler = SimpleNamespace(
        device_module=SimpleNamespace(Event=lambda: event),
        forward_stream=SimpleNamespace(wait_event=waited.append),
    )
    draft_input = DFlashDraftInputV2.create_idle_input(torch.device("cpu"))
    draft_input.bonus_tokens = torch.tensor([7, 8], dtype=torch.int64)
    draft_input.new_seq_lens = torch.tensor([70, 80], dtype=torch.int64)
    batch_result = SimpleNamespace(next_draft_input=draft_input)
    batch = SimpleNamespace(
        reqs=[
            SimpleNamespace(finished=lambda: False, is_retracted=False),
            SimpleNamespace(finished=lambda: False, is_retracted=False),
        ]
    )

    controller.stage_overlap_pivot_copy(batch, batch_result)
    assert len(pivots) == (0 if is_token_only else 1)

    assert copied == [draft_input.bonus_tokens, draft_input.new_seq_lens]
    assert torch.equal(
        batch_result.dt_flashmtp_bonus_tokens_cpu, draft_input.bonus_tokens
    )
    assert torch.equal(
        batch_result.dt_flashmtp_seq_lens_cpu, draft_input.new_seq_lens
    )
    assert event.recorded
    assert waited == [event]
    assert batch_result.dt_flashmtp_scalar_copy_done is event


@pytest.mark.parametrize("pinned", (False, True))
def test_ready_batch_h2d_is_nonblocking_only_for_pinned_source(pinned) -> None:
    calls = []

    class FakeHostTensor:
        def is_pinned(self):
            return pinned

        def to(self, **kwargs):
            calls.append(kwargs)
            return "device-tensor"

    result = _ready_batch_h2d(
        FakeHostTensor(), device="cuda:0", dtype=torch.int32
    )

    assert result == "device-tensor"
    assert calls == [
        {
            "device": "cuda:0",
            "dtype": torch.int32,
            "non_blocking": pinned,
        }
    ]


def test_ready_candidate_host_tensor_preserves_rows_in_final_dtype() -> None:
    values = (11, 12, 13, 14, 21, 22, 23, 24)
    staged = _ready_candidate_host_tensor(
        struct.pack("<8i", *values),
        batch_size=2,
        block_size=4,
        pin_memory=False,
    )

    assert staged.device.type == "cpu"
    assert staged.dtype == torch.int64
    assert staged.tolist() == [[11, 12, 13, 14], [21, 22, 23, 24]]


def test_flashmtp_candidate_copy_is_enqueued_without_early_publish(
    monkeypatch,
) -> None:
    class FakeEvent:
        def __init__(self) -> None:
            self.recorded_on = None

        def record(self, stream) -> None:
            self.recorded_on = stream

        def query(self) -> bool:
            return False

    event = FakeEvent()
    stream = object()
    monkeypatch.setattr(
        dt_draft_engine_module.torch.cuda,
        "Event",
        lambda **_kwargs: event,
    )
    monkeypatch.setattr(
        dt_draft_engine_module.torch.cuda,
        "current_stream",
        lambda _device: stream,
    )

    candidates = torch.tensor(
        [[7, 1, 2, 3], [8, 4, 5, 6]], dtype=torch.int32
    )
    pivots = torch.empty((2, 1), dtype=torch.bfloat16)
    staged_bonus_tokens = torch.tensor([7, 8], dtype=torch.int64)
    engine = object.__new__(FlashMTPDraftEngine)
    engine.profile = SimpleNamespace(block_size=4)
    engine.device = torch.device("cpu")
    engine._profile_active = False
    operation_order = []
    replay_calls = []

    def replay(**kwargs):
        operation_order.append("graph_replay")
        replay_calls.append(kwargs)
        return candidates

    engine._draft_cuda_graph = SimpleNamespace(replay=replay)
    engine._require_flashmtp_pivot_layout = (
        lambda _pivots, *, batch_size: torch.device("cpu")
    )
    engine._stage_flashmtp_bonus_tokens = (
        lambda tasks, **_kwargs: (
            operation_order.append("bonus_stage")
            or staged_bonus_tokens,
            None,
            None,
            0,
            0,
        )
    )
    engine._candidate_result_host_buffers = [
        torch.empty((4, 4), dtype=torch.int32)
    ]
    engine._pending_flashmtp_completions = deque()
    engine._record_graph_measurement = lambda **_kwargs: None
    published = []
    engine._publish_successes = lambda *_args: published.append(True)
    pending = [
        PendingDraftTask(
            SimpleNamespace(
                next_token_id=7,
                request_id="warmup-0",
                transfer_id="transfer-0",
            )
        ),
        PendingDraftTask(
            SimpleNamespace(
                next_token_id=8,
                request_id="warmup-1",
                transfer_id="transfer-1",
            )
        ),
    ]

    engine._execute_flashmtp_batch(
        pending,
        pivots,
        staging_slot=0,
    )

    assert not published
    assert len(engine._pending_flashmtp_completions) == 1
    completion = engine._pending_flashmtp_completions[0]
    assert completion.pending is pending
    assert completion.staging_slot == 0
    assert completion.completion_event is event
    assert event.recorded_on is stream
    assert torch.equal(completion.candidate_rows_cpu, candidates)
    assert operation_order == ["bonus_stage", "graph_replay"]
    assert len(replay_calls) == 1
    assert set(replay_calls[0]) == {"pivots", "bonus_tokens"}
    assert replay_calls[0]["pivots"] is pivots
    assert replay_calls[0]["bonus_tokens"] is staged_bonus_tokens
    assert replay_calls[0]["bonus_tokens"].tolist() == [7, 8]


def test_flashmtp_eager_proposal_consumes_staged_bonus_tokens(monkeypatch) -> None:
    class FakeEvent:
        def record(self, _stream) -> None:
            pass

    class DraftModel:
        markov_head = object()

        def __call__(self, **kwargs):
            return SimpleNamespace(hidden_states=kwargs["input_embeds"])

        def sample_draft_tokens(self, *, block_hidden, first_prev_token_ids):
            del block_hidden
            increments = torch.tensor([1, 2, 3], dtype=torch.int64)
            return first_prev_token_ids[:, None] + increments[None, :]

    monkeypatch.setattr(
        dt_draft_engine_module.torch.cuda,
        "Event",
        lambda **_kwargs: FakeEvent(),
    )
    monkeypatch.setattr(
        dt_draft_engine_module.torch.cuda,
        "current_stream",
        lambda _device: object(),
    )
    engine = object.__new__(FlashMTPDraftEngine)
    engine.profile = SimpleNamespace(block_size=4)
    engine.device = torch.device("cpu")
    engine._profile_active = False
    engine._draft_cuda_graph = None
    engine._draft_block_len = 4
    engine._mask_token_id = 0
    engine._num_proposals = 3
    engine._local_positions = torch.arange(4, dtype=torch.int64)
    engine._token_embedding = lambda ids: ids.to(torch.float32).unsqueeze(-1)
    engine._draft_model = DraftModel()
    engine._target_lm_head = None
    engine._require_flashmtp_pivot_layout = (
        lambda _pivots, *, batch_size: torch.device("cpu")
    )
    engine._stage_flashmtp_bonus_tokens = (
        lambda tasks, **_kwargs: (
            torch.tensor(
                [task.next_token_id for task in tasks], dtype=torch.int64
            ),
            None,
            None,
            0,
            0,
        )
    )
    engine._candidate_result_host_buffers = [
        torch.empty((4, 4), dtype=torch.int32)
    ]
    engine._pending_flashmtp_completions = deque()
    engine._record_graph_measurement = lambda **_kwargs: None
    pending = [
        PendingDraftTask(
            SimpleNamespace(
                next_token_id=10,
                request_id="warmup-0",
                transfer_id="transfer-0",
            )
        ),
        PendingDraftTask(
            SimpleNamespace(
                next_token_id=20,
                request_id="warmup-1",
                transfer_id="transfer-1",
            )
        ),
    ]

    engine._execute_flashmtp_batch(
        pending,
        torch.empty((2, 1), dtype=torch.bfloat16),
        staging_slot=0,
    )

    completion = engine._pending_flashmtp_completions[0]
    assert completion.execution == "eager"
    assert completion.candidate_rows_cpu.tolist() == [
        [10, 11, 12, 13],
        [20, 21, 22, 23],
    ]


def test_flashmtp_bonus_staging_allocates_one_pinned_device_pair_per_slot(
    monkeypatch,
) -> None:
    calls = []

    def fake_empty(*shape, **kwargs):
        value = object()
        calls.append((shape, kwargs, value))
        return value

    monkeypatch.setattr(dt_draft_engine_module.torch, "empty", fake_empty)
    engine = object.__new__(FlashMTPDraftEngine)
    engine.max_batch_size = 128
    engine.feature_buffer_slot_count = 2
    engine.device = torch.device("cuda:0")

    engine._initialize_flashmtp_bonus_token_staging()

    assert len(engine._proposal_bonus_tokens_host_buffers) == 2
    assert len(engine._proposal_bonus_tokens_buffers) == 2
    assert len({id(value) for value in engine._proposal_bonus_tokens_host_buffers}) == 2
    assert len({id(value) for value in engine._proposal_bonus_tokens_buffers}) == 2
    assert [
        (shape, kwargs)
        for shape, kwargs, _value in calls
    ] == [
        ((128,), {"dtype": torch.int64, "device": "cpu", "pin_memory": True}),
        ((128,), {"dtype": torch.int64, "device": "cpu", "pin_memory": True}),
        ((128,), {"dtype": torch.int64, "device": torch.device("cuda:0")}),
        ((128,), {"dtype": torch.int64, "device": torch.device("cuda:0")}),
    ]


def test_flashmtp_bonus_staging_is_slot_owned_nonblocking_and_profile_only(
    monkeypatch,
) -> None:
    operations = []

    class Buffer:
        def __init__(self, *, device, pinned, slot, values=None, bounds=None):
            self.device = torch.device(device)
            self.dtype = torch.int64
            self.shape = (4,) if bounds is None else (bounds[1] - bounds[0],)
            self._pinned = pinned
            self.slot = slot
            self.values = [None] * 4 if values is None else values
            self.bounds = (0, 4) if bounds is None else bounds

        def is_pinned(self):
            return self._pinned

        def __getitem__(self, item):
            assert isinstance(item, slice) and item.start in (None, 0)
            stop = 4 if item.stop is None else item.stop
            return Buffer(
                device=self.device,
                pinned=self._pinned,
                slot=self.slot,
                values=self.values,
                bounds=(0, stop),
            )

        def copy_(self, source, non_blocking=False):
            if isinstance(source, Buffer):
                values = source.values[source.bounds[0] : source.bounds[1]]
                kind = "device_copy"
            else:
                values = source.tolist()
                kind = "host_copy"
            start, end = self.bounds
            self.values[start:end] = values
            operations.append((kind, self.slot, non_blocking, tuple(values)))
            return self

    class TimingEvent:
        def __init__(self, name):
            self.name = name

        def record(self, stream):
            operations.append((self.name, stream))

    host_buffers = [
        Buffer(device="cpu", pinned=True, slot=0),
        Buffer(device="cpu", pinned=True, slot=1),
    ]
    device_buffers = [
        Buffer(device="cuda:0", pinned=False, slot=0),
        Buffer(device="cuda:0", pinned=False, slot=1),
    ]
    engine = object.__new__(FlashMTPDraftEngine)
    engine.max_batch_size = 4
    engine.gpu_id = 0
    engine.device = torch.device("cuda:0")
    engine._feature_staging_buffers = [object(), object()]
    engine._proposal_bonus_tokens_host_buffers = host_buffers
    engine._proposal_bonus_tokens_buffers = device_buffers
    engine._profile_active = False
    stream = object()
    monkeypatch.setattr(
        dt_draft_engine_module.torch.cuda,
        "current_stream",
        lambda _device: operations.append(("current_stream", stream)) or stream,
    )
    monkeypatch.setattr(
        dt_draft_engine_module.torch.cuda,
        "Event",
        lambda **_kwargs: pytest.fail("unprofiled staging created a timing event"),
    )

    first = engine._stage_flashmtp_bonus_tokens(
        [SimpleNamespace(next_token_id=11), SimpleNamespace(next_token_id=12)],
        staging_slot=0,
        execution_device=torch.device("cuda:0"),
    )
    second = engine._stage_flashmtp_bonus_tokens(
        [SimpleNamespace(next_token_id=21), SimpleNamespace(next_token_id=22)],
        staging_slot=1,
        execution_device=torch.device("cuda:0"),
    )

    assert first[1:] == (None, None, 0, 0)
    assert second[1:] == (None, None, 0, 0)
    assert host_buffers[0].values == [11, 12, None, None]
    assert device_buffers[0].values == [11, 12, None, None]
    assert host_buffers[1].values == [21, 22, None, None]
    assert device_buffers[1].values == [21, 22, None, None]
    assert [item for item in operations if item[0].endswith("copy")] == [
        ("host_copy", 0, False, (11, 12)),
        ("device_copy", 0, True, (11, 12)),
        ("host_copy", 1, False, (21, 22)),
        ("device_copy", 1, True, (21, 22)),
    ]

    operations.clear()
    engine._profile_active = True
    event_names = iter(("h2d_start", "h2d_done"))
    monkeypatch.setattr(
        dt_draft_engine_module.torch.cuda,
        "Event",
        lambda **kwargs: TimingEvent(next(event_names))
        if kwargs == {"enable_timing": True}
        else pytest.fail(f"unexpected Event kwargs: {kwargs}"),
    )
    times = iter((100, 103, 200, 204))
    monkeypatch.setattr(
        dt_draft_engine_module.time, "monotonic_ns", lambda: next(times)
    )

    profiled = engine._stage_flashmtp_bonus_tokens(
        [SimpleNamespace(next_token_id=31), SimpleNamespace(next_token_id=32)],
        staging_slot=0,
        execution_device=torch.device("cuda:0"),
    )

    assert profiled[3:] == (3, 4)
    assert operations == [
        ("host_copy", 0, False, (31, 32)),
        ("current_stream", stream),
        ("h2d_start", stream),
        ("device_copy", 0, True, (31, 32)),
        ("h2d_done", stream),
    ]


def test_flashmtp_bonus_staging_rejects_unpinned_or_mismatched_layout() -> None:
    bad_host = SimpleNamespace(
        dtype=torch.int64,
        shape=(4,),
        device=torch.device("cpu"),
        is_pinned=lambda: False,
    )
    device = SimpleNamespace(
        dtype=torch.int64,
        shape=(4,),
        device=torch.device("cuda:0"),
    )
    engine = object.__new__(FlashMTPDraftEngine)
    engine.max_batch_size = 4
    engine.gpu_id = 0
    engine.device = torch.device("cuda:0")
    engine._profile_active = False
    engine._feature_staging_buffers = [object()]
    engine._proposal_bonus_tokens_host_buffers = [bad_host]
    engine._proposal_bonus_tokens_buffers = [device]

    with pytest.raises(RuntimeError, match="pinned CPU int64"):
        engine._stage_flashmtp_bonus_tokens(
            [SimpleNamespace(next_token_id=1)],
            staging_slot=0,
            execution_device=torch.device("cuda:0"),
        )

    engine._proposal_bonus_tokens_host_buffers = []
    with pytest.raises(RuntimeError, match="slots are inconsistent"):
        engine._stage_flashmtp_bonus_tokens(
            [SimpleNamespace(next_token_id=1)],
            staging_slot=0,
            execution_device=torch.device("cuda:0"),
        )


def test_flashmtp_candidate_completion_retains_slot_and_filters_rows(
    monkeypatch,
) -> None:
    class Event:
        ready = False

        def query(self) -> bool:
            return self.ready

    event = Event()
    tasks = [
        SimpleNamespace(transfer_id="active", deadline_ns=200),
        SimpleNamespace(transfer_id="cancelled", deadline_ns=200),
        SimpleNamespace(transfer_id="expired", deadline_ns=50),
    ]
    completion = PendingFlashMTPCompletion(
        pending=[PendingDraftTask(task) for task in tasks],
        staging_slot=3,
        completion_event=event,
        bonus_h2d_start_event=None,
        bonus_h2d_done_event=None,
        proposal_start_event=None,
        proposal_done_event=None,
        candidate_rows_cpu=torch.tensor(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            dtype=torch.int32,
        ),
        execute_started_ns=1,
        inputs_ready_ns=2,
        submitted_ns=3,
        bonus_pack_ns=0,
        bonus_h2d_submit_ns=0,
        bonus_device="cuda:0",
        execution="graph",
        measurement_rows=0,
        prior_pending_flashmtp_completions=0,
    )
    engine = object.__new__(FlashMTPDraftEngine)
    engine.profile = SimpleNamespace(block_size=4)
    engine._profile_active = False
    engine._fatal_gpu_quarantine_reason = None
    engine._pending_flashmtp_completions = deque((completion,))
    engine._quarantined_flashmtp_completions = []
    engine._cancelled_ids = {"cancelled"}
    engine._pending_ids = {task.transfer_id for task in tasks}
    engine._received_ns_by_transfer = {}
    engine._free_feature_staging_slots = deque()
    successes = []
    failures = []
    capacity = []
    engine._publish_successes = lambda rows, payload: successes.append(
        ([task.transfer_id for task in rows], payload)
    )
    engine._publish_failures = lambda rows: failures.extend(rows)
    engine._publish_capacity = lambda: capacity.append(True)
    monkeypatch.setattr(dt_draft_engine_module.time, "monotonic_ns", lambda: 100)

    engine._poll_flashmtp_completions()
    assert list(engine._pending_flashmtp_completions) == [completion]
    assert not engine._free_feature_staging_slots
    assert not successes
    assert not failures

    event.ready = True
    engine._poll_flashmtp_completions()

    assert not engine._pending_flashmtp_completions
    assert list(engine._free_feature_staging_slots) == [3]
    assert engine._cancelled_ids == set()
    assert successes[0][0] == ["active"]
    assert struct.unpack("<4i", successes[0][1]) == (1, 2, 3, 4)
    assert [(task.transfer_id, type(error)) for task, error in failures] == [
        ("expired", TimeoutError)
    ]
    assert capacity == [True]


def test_flashmtp_candidate_completions_retire_in_fifo_order() -> None:
    class Event:
        def __init__(self, ready: bool) -> None:
            self.ready = ready

        def query(self) -> bool:
            return self.ready

    head_event = Event(ready=False)
    tail_event = Event(ready=True)
    head_task = SimpleNamespace(transfer_id="head", deadline_ns=2**63 - 1)
    tail_task = SimpleNamespace(transfer_id="tail", deadline_ns=2**63 - 1)

    def completion(task, slot, event, values):
        return PendingFlashMTPCompletion(
            pending=[PendingDraftTask(task)],
            staging_slot=slot,
            completion_event=event,
            bonus_h2d_start_event=None,
            bonus_h2d_done_event=None,
            proposal_start_event=None,
            proposal_done_event=None,
            candidate_rows_cpu=torch.tensor([values], dtype=torch.int32),
            execute_started_ns=1,
            inputs_ready_ns=2,
            submitted_ns=3,
            bonus_pack_ns=0,
            bonus_h2d_submit_ns=0,
            bonus_device="cuda:0",
            execution="graph",
            measurement_rows=0,
            prior_pending_flashmtp_completions=0,
        )

    head = completion(head_task, 4, head_event, [1, 2, 3, 4])
    tail = completion(tail_task, 5, tail_event, [5, 6, 7, 8])
    engine = object.__new__(FlashMTPDraftEngine)
    engine.profile = SimpleNamespace(block_size=4)
    engine._profile_active = False
    engine._fatal_gpu_quarantine_reason = None
    engine._pending_flashmtp_completions = deque((head, tail))
    engine._quarantined_flashmtp_completions = []
    engine._cancelled_ids = set()
    engine._free_feature_staging_slots = deque()
    published = []
    capacity = []
    engine._publish_successes = lambda tasks, payload: published.append(
        ([task.transfer_id for task in tasks], struct.unpack("<4i", payload))
    )
    engine._publish_failures = lambda _rows: pytest.fail(
        "ready FIFO completions must not fail"
    )
    engine._publish_capacity = lambda: capacity.append(True)

    # A ready tail cannot overtake an unready head and neither slot is released.
    engine._poll_flashmtp_completions()
    assert list(engine._pending_flashmtp_completions) == [head, tail]
    assert not published
    assert not engine._free_feature_staging_slots
    assert not capacity

    head_event.ready = True
    engine._poll_flashmtp_completions()

    assert not engine._pending_flashmtp_completions
    assert published == [
        (["head"], (1, 2, 3, 4)),
        (["tail"], (5, 6, 7, 8)),
    ]
    assert list(engine._free_feature_staging_slots) == [4, 5]
    assert capacity == [True]


def test_flashmtp_profiled_completion_emits_async_gpu_timings(monkeypatch) -> None:
    class CompletionEvent:
        def query(self) -> bool:
            return True

    class TimingEvent:
        def __init__(self, elapsed_ms: float) -> None:
            self.elapsed_ms = elapsed_ms

        def elapsed_time(self, _other) -> float:
            return self.elapsed_ms

    task = SimpleNamespace(transfer_id="transfer-profiled", deadline_ns=1_000)
    completion_event = CompletionEvent()
    completion = PendingFlashMTPCompletion(
        pending=[PendingDraftTask(task)],
        staging_slot=6,
        completion_event=completion_event,
        bonus_h2d_start_event=TimingEvent(0.025),
        bonus_h2d_done_event=TimingEvent(0.0),
        proposal_start_event=TimingEvent(1.25),
        proposal_done_event=TimingEvent(0.5),
        candidate_rows_cpu=torch.tensor([[11, 12, 13, 14]], dtype=torch.int32),
        execute_started_ns=10,
        inputs_ready_ns=20,
        submitted_ns=30,
        bonus_pack_ns=3,
        bonus_h2d_submit_ns=4,
        bonus_device="cuda:0",
        execution="graph",
        measurement_rows=1,
        prior_pending_flashmtp_completions=1,
    )
    engine = object.__new__(FlashMTPDraftEngine)
    engine.instance_id = "draft-0"
    engine.profile = SimpleNamespace(block_size=4)
    engine._profile_active = True
    engine._fatal_gpu_quarantine_reason = None
    engine._pending_flashmtp_completions = deque((completion,))
    engine._quarantined_flashmtp_completions = []
    engine._cancelled_ids = set()
    engine._free_feature_staging_slots = deque()
    records = []
    successes = []
    engine._publish_successes = lambda tasks, payload: successes.append(
        ([task.transfer_id for task in tasks], struct.unpack("<4i", payload))
    )
    engine._publish_failures = lambda _rows: pytest.fail(
        "profiled ready completion must not fail"
    )
    engine._publish_capacity = lambda: None
    monkeypatch.setattr(dt_draft_engine_module.time, "monotonic_ns", lambda: 100)
    monkeypatch.setattr(
        dt_draft_engine_module,
        "profile_emit",
        lambda event, **fields: records.append((event, fields)),
    )

    engine._poll_flashmtp_completions()

    assert successes == [(["transfer-profiled"], (11, 12, 13, 14))]
    assert list(engine._free_feature_staging_slots) == [6]
    assert records == [
        (
            "draft_flashmtp_proposal",
            {
                "draft_instance": "draft-0",
                "execute_started_ns": 10,
                "inputs_ready_ns": 20,
                "proposal_enqueued_ns": 30,
                "batch_size": 1,
                "transfer_ids": ("transfer-profiled",),
                "execution": "graph",
                "measurement_rows": 1,
                "input_prepare_ns": 10,
                "proposal_to_host_ns": 80,
                "total_ns": 90,
                "candidate_bytes": 16,
                "proposal_gpu_ns": 1_250_000,
                "candidate_d2h_gpu_ns": 500_000,
                "completion_poll_ns": 70,
                "asynchronous_completion": True,
                "bonus_stage_async": True,
                "bonus_tokens_pinned_h2d": True,
                "bonus_staging_slot": 6,
                "bonus_token_rows": 1,
                "bonus_token_bytes": 8,
                "bonus_host_pinned": True,
                "bonus_host_dtype": "torch.int64",
                "bonus_device_dtype": "torch.int64",
                "bonus_device": "cuda:0",
                "bonus_pack_ns": 3,
                "bonus_h2d_submit_ns": 4,
                "bonus_h2d_gpu_ns": 25_000,
                "prior_pending_flashmtp_completions": 1,
            },
        )
    ]


def test_flashmtp_candidate_event_query_failure_quarantines_slot() -> None:
    class Event:
        def query(self) -> bool:
            raise RuntimeError("injected candidate event failure")

    task = SimpleNamespace(transfer_id="transfer-0", deadline_ns=200)
    completion = PendingFlashMTPCompletion(
        pending=[PendingDraftTask(task)],
        staging_slot=4,
        completion_event=Event(),
        bonus_h2d_start_event=None,
        bonus_h2d_done_event=None,
        proposal_start_event=None,
        proposal_done_event=None,
        candidate_rows_cpu=torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
        execute_started_ns=1,
        inputs_ready_ns=2,
        submitted_ns=3,
        bonus_pack_ns=0,
        bonus_h2d_submit_ns=0,
        bonus_device="cuda:0",
        execution="graph",
        measurement_rows=0,
        prior_pending_flashmtp_completions=0,
    )
    engine = object.__new__(FlashMTPDraftEngine)
    engine._fatal_gpu_quarantine_reason = None
    engine._stop = False
    engine._p2p_transport = None
    engine._feature_pull_transport = None
    engine._pending_flashmtp_completions = deque((completion,))
    engine._quarantined_flashmtp_completions = []
    engine._free_feature_staging_slots = deque()
    failures = []
    capacity = []
    engine._publish_failures = lambda rows: failures.extend(rows)
    engine._publish_capacity = lambda: capacity.append(True)

    engine._poll_flashmtp_completions()

    assert engine._stop
    assert "FlashMTP completion-event failure" in engine._fatal_gpu_quarantine_reason
    assert not engine._pending_flashmtp_completions
    assert engine._quarantined_flashmtp_completions == [completion]
    assert not engine._free_feature_staging_slots
    assert failures[0][0] is task
    assert getattr(failures[0][1], "resources_quiescent") is False
    assert capacity == [True]


def test_flashmtp_shutdown_synchronizes_and_drains_candidate_completion() -> None:
    class Event:
        ready = False
        synchronize_calls = 0
        query_calls = 0

        def synchronize(self) -> None:
            self.synchronize_calls += 1
            self.ready = True

        def query(self) -> bool:
            self.query_calls += 1
            return self.ready

    class FeatureTransport:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    event = Event()
    transport = FeatureTransport()
    task = SimpleNamespace(transfer_id="transfer-0", deadline_ns=2**63 - 1)
    completion = PendingFlashMTPCompletion(
        pending=[PendingDraftTask(task)],
        staging_slot=5,
        completion_event=event,
        bonus_h2d_start_event=None,
        bonus_h2d_done_event=None,
        proposal_start_event=None,
        proposal_done_event=None,
        candidate_rows_cpu=torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
        execute_started_ns=1,
        inputs_ready_ns=2,
        submitted_ns=3,
        bonus_pack_ns=0,
        bonus_h2d_submit_ns=0,
        bonus_device="cuda:0",
        execution="graph",
        measurement_rows=0,
        prior_pending_flashmtp_completions=0,
    )
    engine = object.__new__(FlashMTPDraftEngine)
    engine._fatal_gpu_quarantine_reason = None
    engine._stop = True
    engine._p2p_transport = None
    engine._feature_pull_transport = transport
    engine._pending = deque()
    engine._pending_ids = set()
    engine._received_ns_by_transfer = {}
    engine._cancelled_ids = set()
    engine._pending_feature_batches = deque()
    engine._pending_flashmtp_completions = deque((completion,))
    engine._quarantined_flashmtp_completions = []
    engine._pending_dflash_completions = deque()
    engine._pending_eagle3_completions = deque()
    engine._quarantined_eagle3_completions = []
    engine._free_feature_staging_slots = deque()
    engine._deferred_migration_gpu_controls = {}
    engine._deferred_feature_source_tasks = {}
    engine._feature_source_inflight = set()
    engine._feature_source_quiescent = set()
    engine._feature_source_unquiescent = set()
    engine._quarantined_feature_draft_handles = set()
    engine._delivery_outbox = SimpleNamespace(pending=lambda: ())
    engine._profile_active = False
    engine.profile = SimpleNamespace(block_size=4)
    engine._poll_feature_batches = lambda: None
    engine._flush_source_done = lambda: None
    engine._publish_successes = lambda *_args: pytest.fail(
        "shutdown-cancelled candidate must not publish"
    )
    engine._publish_failures = lambda *_args: None
    engine._publish_capacity = lambda: None

    engine._quiesce_direct_gpu_source_shutdown()

    assert event.synchronize_calls == 1
    assert event.query_calls == 1
    assert not engine._pending_flashmtp_completions
    assert not engine._quarantined_flashmtp_completions
    assert list(engine._free_feature_staging_slots) == [5]
    assert engine._cancelled_ids == set()
    assert transport.close_calls == 1
    assert engine._feature_pull_transport is None


def test_direct_gpu_measurement_epoch_counters_accumulate_by_source():
    target = DTTargetController.__new__(DTTargetController)
    target.instance_id = "target-0"
    target._direct_gpu_measurement_epochs = {}
    for rows, requests in ((4, 2), (3, 1)):
        target._record_direct_gpu_measurement_epoch(
            measurement_rows=rows,
            measurement_requests=requests,
            source_session_id="target-session",
            pool_generation="target-pool",
            transport_session="target-transport",
        )
    assert target._direct_gpu_measurement_epoch_evidence() == [
        {
            "target_instance_id": "target-0",
            "source_session_id": "target-session",
            "pool_generation": "target-pool",
            "transport_session": "target-transport",
            "batches": 2,
            "measurement_rows": 7,
            "measurement_requests": 3,
        }
    ]

    draft = FlashMTPDraftEngine.__new__(FlashMTPDraftEngine)
    draft._direct_gpu_measurement_epochs = {}
    draft._record_direct_gpu_measurement_epoch(
        target_instance_id="target-1",
        source_session_id="source-1",
        pool_generation="pool-1",
        transport_session="transport-1",
        measurement_rows=5,
        measurement_requests=2,
        measurement_proposal_requests=1,
    )
    assert draft._direct_gpu_measurement_epoch_evidence()[0] == {
        "target_instance_id": "target-1",
        "source_session_id": "source-1",
        "pool_generation": "pool-1",
        "transport_session": "transport-1",
        "batches": 1,
        "measurement_rows": 5,
        "measurement_requests": 2,
        "measurement_proposal_requests": 1,
    }
    draft._direct_gpu_measurement_requests = {}
    for rows, proposals in ((3, 0), (2, 1)):
        draft._record_direct_gpu_measurement_request(
            target_instance_id="target-1",
            source_session_id="source-1",
            pool_generation="pool-1",
            transport_session="transport-1",
            request_id="dt-benchmark-measurement-000001",
            measurement_rows=rows,
            measurement_requests=1,
            measurement_proposal_requests=proposals,
        )
    assert draft._direct_gpu_measurement_request_evidence() == [
        {
            "target_instance_id": "target-1",
            "source_session_id": "source-1",
            "pool_generation": "pool-1",
            "transport_session": "transport-1",
            "request_id": "dt-benchmark-measurement-000001",
            "measurement_rows": 5,
            "measurement_requests": 2,
            "measurement_proposal_requests": 1,
        }
    ]


def _init_lifetime_state(engine) -> None:
    engine._is_dflash = False
    engine._is_feature_backed = False
    engine._persistent = True
    engine._pending = deque()
    engine._pending_feature_batches = deque()
    engine._pending_flashmtp_completions = deque()
    engine._pending_dflash_completions = deque()
    engine._pending_eagle3_completions = deque()
    engine._quarantined_flashmtp_completions = []
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
    engine._quarantined_feature_draft_handles = set()


def _init_p2p_transport_fence_state(transport) -> None:
    transport._fatal_error = None
    transport._permanent_quarantine_error = None
    transport._closed = False
    transport._lock = Lock()


def test_disaggregated_dflash_target_validation_fails_closed() -> None:
    args = SimpleNamespace(
        draft_target_disaggregation_role="target",
        device="cuda",
        dp_size=1,
        disable_overlap_schedule=False,
        draft_target_overlap_schedule=True,
        draft_target_defer_dflash_feature_stage=True,
        draft_target_coordinator_endpoint="tcp://127.0.0.1:19001",
        draft_target_instance_id="target-0",
        draft_target_profile_id="dflash-profile",
        draft_target_feature_schema_id="dflash-schema",
        draft_target_round_timeout_ms=60_000,
        draft_target_registration_timeout_s=60.0,
    )

    _validate_dflash_draft_target(args)

    args.device = "npu"
    with pytest.raises(ValueError, match="CUDA direct feature transport"):
        _validate_dflash_draft_target(args)
    args.device = "cuda"
    args.draft_target_defer_dflash_feature_stage = False
    with pytest.raises(ValueError, match="defer-dflash-feature-stage"):
        _validate_dflash_draft_target(args)
    args.draft_target_defer_dflash_feature_stage = True
    args.disable_overlap_schedule = True
    with pytest.raises(ValueError, match="requires overlap scheduling"):
        _validate_dflash_draft_target(args)


def test_flashmtp_draft_admission_rejects_legacy_mmap_pivot() -> None:
    engine = object.__new__(FlashMTPDraftEngine)
    engine.profile = SimpleNamespace(
        profile_id="flashmtp-profile",
        draft_k=7,
        feature_schema=SimpleNamespace(schema_id="flashmtp-pivot"),
    )
    engine._is_feature_backed = False
    engine._persistent = False
    engine._pivot_width = 16
    source = SourceRef(
        target_instance_id="target-0",
        source_device_id=0,
        source_slot_id=1,
        slot_generation=2,
        transfer_id="transfer-0",
        feature_schema_id="flashmtp-pivot",
        nbytes=32,
        transport="cuda-ipc",
        source_session_id="target-session",
        pool_generation="target-pool",
        transport_session="target-transport",
        device_uuid="GPU-target",
        allocation_length=4096,
        allocation_offset=128,
        ipc_handle=b"ipc-handle",
        dtype="bfloat16",
        shape=(1, 16),
        producer_event_handle=b"event-handle",
    )
    task = DraftTask(
        transfer_id="transfer-0",
        request_id="request-0",
        round_id=0,
        attempt=0,
        target_instance_id="target-0",
        draft_profile_id="flashmtp-profile",
        k=7,
        next_token_id=42,
        source=source,
        created_ns=1,
        deadline_ns=2,
    )

    engine._validate_task(task)
    with pytest.raises(ValueError, match="complete fenced CUDA IPC"):
        engine._validate_task(
            msgspec.structs.replace(
                task,
                source=msgspec.structs.replace(
                    source,
                    transport="mmap",
                    uri="mmap:///tmp/legacy-pivots.slots",
                    source_session_id="",
                    pool_generation="",
                    transport_session="",
                    device_uuid="",
                    allocation_length=0,
                    ipc_handle=b"",
                    dtype="",
                    shape=(),
                    producer_event_handle=b"",
                ),
            )
        )


def test_flashmtp_overlap_preserves_final_chunk_until_ordered_proposal() -> None:
    class FeaturePool:
        source_session_id = "target-session"
        pool_generation = "pool-generation"

        def __init__(self) -> None:
            self.active = {
                "pivot-middle": (0, 1),
                "pivot-final": (1, 1),
            }
            self.cancelled = []
            self.completed = []

        def _retire(
            self,
            *,
            transfer_id,
            source_slot_id,
            slot_generation,
            source_session_id,
            pool_generation,
        ) -> None:
            assert source_session_id == self.source_session_id
            assert pool_generation == self.pool_generation
            assert self.active.pop(transfer_id) == (
                source_slot_id,
                slot_generation,
            )

        def cancel(self, **kwargs) -> None:
            self._retire(**kwargs)
            self.cancelled.append(kwargs["transfer_id"])

        def read_done(self, **kwargs) -> None:
            self._retire(**kwargs)
            self.completed.append(kwargs["transfer_id"])

    req = SimpleNamespace(
        rid="request-0",
        is_retracted=False,
        inflight_middle_chunks=1,
        finished=lambda: False,
    )
    # Byte preservation must ignore proposal-eligibility counters. The final
    # chunk is already launched while an older middle result still owns req.
    assert _flashmtp_pivot_preservation_indices(
        SimpleNamespace(reqs=[req], chunked_req=None)
    ) == [0]

    middle_descriptor = object()
    final_descriptor = object()
    middle_result = SimpleNamespace(
        dt_staged_flashmtp_pivots=StagedFlashMTPPivotBatch(
            tasks=(
                StagedFlashMTPTask(
                    request_id=req.rid,
                    request_index=0,
                    source_lease_id="pivot-middle",
                ),
            ),
            descriptors_by_transfer={"pivot-middle": middle_descriptor},
        )
    )
    final_result = SimpleNamespace(
        dt_staged_flashmtp_pivots=StagedFlashMTPPivotBatch(
            tasks=(
                StagedFlashMTPTask(
                    request_id=req.rid,
                    request_index=0,
                    source_lease_id="pivot-final",
                ),
            ),
            descriptors_by_transfer={"pivot-final": final_descriptor},
        )
    )

    def draft_input(bonus_token: int, seq_len: int) -> DFlashDraftInputV2:
        value = DFlashDraftInputV2.create_idle_input(torch.device("cpu"))
        value.bonus_tokens = torch.tensor([bonus_token], dtype=torch.int64)
        value.new_seq_lens = torch.tensor([seq_len], dtype=torch.int64)
        return value

    middle_batch = SimpleNamespace(
        reqs=[req],
        spec_info=draft_input(11, 128),
        chunked_req=req,
        enable_overlap=True,
    )
    final_batch = SimpleNamespace(
        reqs=[req],
        spec_info=draft_input(12, 192),
        chunked_req=None,
        enable_overlap=True,
    )

    controller = object.__new__(DTTargetController)
    controller.instance_id = "target-0"
    controller.is_token_only = False
    controller.is_feature_backed = False
    controller.is_export_rank = True
    controller.pipeline_lanes = 2
    controller._lane_target_sizes = [0, 0]
    controller._parked = {}
    controller._verify_transfer_by_request = {}
    controller._next_round = {}
    controller._lifecycle_attempt_by_request = {}
    controller._next_lifecycle_attempt = 0
    controller._target_inflight_counts = {req.rid: 2}
    controller._target_inflight_reqs = {req.rid: req}
    controller._source_slot_by_transfer = {
        "pivot-middle": (0, 1),
        "pivot-final": (1, 1),
    }
    controller._staged_flashmtp_source_leases = {
        "pivot-middle",
        "pivot-final",
    }
    controller._source_deadline_by_transfer = {
        "pivot-middle": 10,
        "pivot-final": 10,
    }
    controller._feature_source_pool = FeaturePool()
    controller._source_done_applied_ids = set()
    controller._cancelled_transfers = set()
    controller._delivery_outbox = RetryOutbox(retry_interval_ns=1)
    controller._ledger = SimpleNamespace(get=lambda _transfer_id: None)
    controller._forget_request_lane = lambda _req: None
    controller._release_draft_request = lambda _rid: None
    controller._lane_for_request = lambda _req: 0
    queued = []

    def make_task(**kwargs):
        return SimpleNamespace(
            transfer_id=kwargs["transfer_id"],
            request_id=kwargs["req"].rid,
            round_id=kwargs["round_id"],
            attempt=kwargs["attempt"],
            deadline_ns=100,
            source=kwargs["feature_descriptor"],
        )

    def queue_tasks(tasks) -> None:
        queued.extend(tasks)
        for task in tasks:
            controller._source_deadline_by_transfer[task.transfer_id] = (
                task.deadline_ns
            )

    controller._make_task = make_task
    controller._queue_task_batch = queue_tasks
    controller._flush_pending_send = lambda: None

    # The middle result is consumed first. Its provisional lease is cancelled,
    # but it cannot create a proposal while the final Target forward is pending.
    # Generic prefill result processing has already retired the middle counter.
    req.inflight_middle_chunks = 0
    controller._park_completed_batch(middle_batch, batch_result=middle_result)
    assert queued == []
    assert controller._feature_source_pool.cancelled == ["pivot-middle"]
    assert controller._target_inflight_counts == {req.rid: 1}
    assert controller._staged_flashmtp_source_leases == {"pivot-final"}

    # The already-preserved final pivot is now the sole proposal source.
    controller._park_completed_batch(final_batch, batch_result=final_result)
    assert len(queued) == 1
    assert queued[0].transfer_id == "pivot-final"
    assert queued[0].source is final_descriptor
    assert queued[0].round_id == 0
    assert queued[0].attempt == 0
    assert not controller._target_inflight_counts
    assert not controller._staged_flashmtp_source_leases

    controller._consume_source_read_done(
        SourceReadDone(
            protocol_version=PROTOCOL_VERSION,
            transfer_id="pivot-final",
            target_instance_id="target-0",
            draft_instance_id="draft-0",
            source_slot_id=1,
            slot_generation=1,
            completed_ns=1,
            source_session_id="target-session",
            pool_generation="pool-generation",
        ),
        flush=False,
        prevalidated=True,
    )
    assert controller._feature_source_pool.completed == ["pivot-final"]
    assert not controller._feature_source_pool.active
    assert not controller._source_slot_by_transfer
    assert not controller._source_deadline_by_transfer
    assert not controller._staged_flashmtp_source_leases


def test_target_shutdown_missing_source_ack_retains_pool_on_repeated_close() -> None:
    events = []
    storage = object()
    producer_event = object()

    class RetainedFeaturePool:
        def __init__(self) -> None:
            self._storage = storage
            self._slots = [SimpleNamespace(producer_event=producer_event)]
            self._active = {"transfer-0": (0, 1)}
            self._closed = False
            self.close_calls = 0

        def close(self) -> None:
            events.append("pool-close")
            self.close_calls += 1
            self._closed = True

        def publish(self) -> None:
            if self._closed:
                raise RuntimeError("feature source pool is closed")

    class Closable:
        def __init__(self, label: str) -> None:
            self.label = label
            self.close_calls = 0

        def close(self) -> None:
            events.append(self.label)
            self.close_calls += 1

    controller = object.__new__(DTTargetController)
    controller.instance_id = "target-0"
    controller.is_export_rank = True
    controller.is_persistent = False
    controller.is_feature_backed = False
    controller.is_token_only = False
    controller.round_timeout_ns = 0
    controller._source_shutdown_quarantine_error = None
    controller._staged_flashmtp_source_leases = set()
    controller._pending_send = deque()
    controller._parked = {}
    controller._feature_control_rounds = {}
    controller._verify_transfer_by_request = {}
    controller._source_slot_by_transfer = {"transfer-0": (0, 1)}
    controller._source_deadline_by_transfer = {"transfer-0": 10}
    controller._lane_by_request = {}
    controller._next_round = {}
    controller._target_inflight_counts = {}
    controller._target_inflight_reqs = {}
    controller._lifecycle_attempt_by_request = {}
    controller._draft_kv_handle_by_request = {}
    controller._delivery_outbox = RetryOutbox(retry_interval_ns=1)
    controller._ledger = SimpleNamespace(
        get=lambda _transfer_id: None,
        finish=lambda _transfer_id: None,
    )
    controller._flush_reliable_outbox = lambda: None
    controller._consume_shutdown_acks = lambda: None

    pool = RetainedFeaturePool()
    store = Closable("store-close")
    client = Closable("client-close")
    controller._feature_source_pool = pool
    controller._result_stores = {"draft-0": store}
    controller._client = client

    # The source lease never receives read-done. The first close fences future
    # publication before host-side cleanup but keeps all GPU ownership rooted.
    with pytest.raises(RuntimeError, match="timed out") as first_failure:
        controller.close()

    assert events == ["pool-close", "store-close", "client-close"]
    assert controller._source_shutdown_quarantine_error is first_failure.value
    assert controller._feature_source_pool is pool
    assert pool._storage is storage
    assert pool._slots[0].producer_event is producer_event
    assert pool._active == {"transfer-0": (0, 1)}
    assert controller._source_slot_by_transfer == {"transfer-0": (0, 1)}
    assert controller._source_deadline_by_transfer == {"transfer-0": 10}
    with pytest.raises(RuntimeError, match="source pool is closed"):
        pool.publish()

    # A second close must not retry the drain, close the pool again, or release
    # any unresolved ownership. It reports the latched first failure instead.
    with pytest.raises(RuntimeError, match="ownership is retained") as repeated:
        controller.close()

    assert repeated.value.__cause__ is first_failure.value
    assert events == ["pool-close", "store-close", "client-close"]
    assert pool.close_calls == 1
    assert store.close_calls == 1
    assert client.close_calls == 1
    assert controller._feature_source_pool is pool
    assert pool._storage is storage
    assert pool._slots[0].producer_event is producer_event
    assert pool._active == {"transfer-0": (0, 1)}
    assert controller._source_slot_by_transfer == {"transfer-0": (0, 1)}


def _cpu_feature_source_pool_for_publish_test(
    monkeypatch, *, max_tokens: int, feature_width: int
):
    class ProducerEvent:
        def __init__(self) -> None:
            self.recorded_stream = None

        def record(self, stream) -> None:
            self.recorded_stream = stream

        def ipc_handle(self) -> bytes:
            return b"producer-event"

    current_stream = object()
    producer_event = ProducerEvent()
    monkeypatch.setattr(
        torch.cuda,
        "Event",
        lambda *, interprocess: producer_event if interprocess else None,
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: current_stream)

    pool = object.__new__(TargetFeatureSourcePool)
    pool.target_instance_id = "target-0"
    pool.source_session_id = "target-session"
    pool.pool_generation = "pool-generation"
    pool.transport_session = "transport-session"
    pool.device_uuid = "GPU-target"
    pool.device_index = 0
    pool.device = torch.device("cuda", 0)
    pool.slot_count = 1
    pool.max_tokens = max_tokens
    pool.feature_width = feature_width
    pool._slot_nbytes = max_tokens * feature_width * 2
    storage = torch.full(
        (1, max_tokens, feature_width),
        -1,
        dtype=torch.bfloat16,
    )
    pool._storage = storage
    pool._allocations = (SimpleNamespace(length=4096, ipc_handle=b"allocation"),)
    pool._buffer = SimpleNamespace(allocation_index=0, allocation_offset=128)
    pool._slots = [
        SimpleNamespace(
            generation=0,
            state=FeatureSlotState.FREE,
            active_transfers=set(),
            producer_event=None,
        )
    ]
    pool._active = {}
    pool._cancelled_transfers = set()
    pool._completed = OrderedDict()
    pool._completed_limit = 64
    pool._closed = False
    pool._lock = Lock()
    pool._validate_hidden_states = lambda _hidden: None
    return pool, storage, producer_event, current_stream


def test_feature_source_publish_batches_ragged_chunks_into_registered_slot(
    monkeypatch,
) -> None:
    pool, storage, producer_event, current_stream = (
        _cpu_feature_source_pool_for_publish_test(
            monkeypatch, max_tokens=6, feature_width=2
        )
    )
    chunks = (
        torch.tensor([[1, 2]], dtype=torch.bfloat16),
        torch.tensor([[3, 4], [5, 6]], dtype=torch.bfloat16),
        torch.tensor([[7, 8]], dtype=torch.bfloat16),
    )
    transfers = (
        FeatureTransferSlice(transfer_id="middle", row_offset=1, row_count=2),
        FeatureTransferSlice(transfer_id="first", row_offset=0, row_count=1),
        FeatureTransferSlice(transfer_id="last", row_offset=3, row_count=1),
    )
    real_cat = torch.cat
    cat_calls = []

    def traced_cat(tensors, *args, **kwargs):
        cat_calls.append((tuple(tensors), args, kwargs))
        return real_cat(tensors, *args, **kwargs)

    monkeypatch.setattr(torch, "cat", traced_cat)
    published = pool.publish(
        hidden_states=chunks,
        feature_schema_id="feature-schema",
        transfers=transfers,
        hidden_produced_ns=17,
    )

    assert len(cat_calls) == 1
    assert cat_calls[0][0] == chunks
    assert cat_calls[0][1] == ()
    assert cat_calls[0][2]["dim"] == 0
    assert cat_calls[0][2]["out"].data_ptr() == storage[0, :4].data_ptr()
    assert torch.equal(storage[0, :4], real_cat(chunks, dim=0))
    assert torch.equal(
        storage[0, 4:], torch.full((2, 2), -1, dtype=torch.bfloat16)
    )
    assert producer_event.recorded_stream is current_stream
    assert published.producer_event is producer_event
    assert published.hidden_produced_ns == 17
    assert published.pack_strategy == "cat-out"
    assert published.pack_chunk_count == 3
    assert [descriptor.transfer_id for descriptor in published.descriptors] == [
        "middle",
        "first",
        "last",
    ]
    assert [descriptor.byte_offset for descriptor in published.descriptors] == [
        4,
        0,
        12,
    ]
    assert [descriptor.nbytes for descriptor in published.descriptors] == [8, 4, 4]
    assert [descriptor.shape for descriptor in published.descriptors] == [
        (2, 2),
        (1, 2),
        (1, 2),
    ]
    assert all(
        descriptor.allocation_offset == 128
        for descriptor in published.descriptors
    )


def test_feature_source_publish_keeps_copy_loop_capability_fallback(
    monkeypatch,
) -> None:
    pool, storage, producer_event, current_stream = (
        _cpu_feature_source_pool_for_publish_test(
            monkeypatch, max_tokens=3, feature_width=2
        )
    )
    chunks = (
        torch.tensor([[1, 2]], dtype=torch.bfloat16),
        torch.tensor([[3, 4], [5, 6]], dtype=torch.bfloat16),
    )
    monkeypatch.setattr(torch, "cat", None)

    published = pool.publish(
        hidden_states=chunks,
        feature_schema_id="feature-schema",
        transfers=(
            FeatureTransferSlice(
                transfer_id="all-rows", row_offset=0, row_count=3
            ),
        ),
    )

    assert torch.equal(
        storage[0],
        torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.bfloat16),
    )
    assert producer_event.recorded_stream is current_stream
    assert published.producer_event is producer_event
    assert published.pack_strategy == "copy-loop"
    assert published.pack_chunk_count == 2


def test_feature_source_publish_quarantines_batched_pack_failure(monkeypatch) -> None:
    pool, storage, producer_event, _current_stream = (
        _cpu_feature_source_pool_for_publish_test(
            monkeypatch, max_tokens=2, feature_width=2
        )
    )
    chunks = (
        torch.tensor([[1, 2]], dtype=torch.bfloat16),
        torch.tensor([[3, 4]], dtype=torch.bfloat16),
    )

    def fail_cat(*_args, **_kwargs):
        raise RuntimeError("injected batched pack failure")

    monkeypatch.setattr(torch, "cat", fail_cat)
    with pytest.raises(RuntimeError, match="batched pack failure"):
        pool.publish(
            hidden_states=chunks,
            feature_schema_id="feature-schema",
            transfers=(
                FeatureTransferSlice(
                    transfer_id="first", row_offset=0, row_count=1
                ),
                FeatureTransferSlice(
                    transfer_id="second", row_offset=1, row_count=1
                ),
            ),
        )

    assert torch.equal(
        storage[0], torch.full((2, 2), -1, dtype=torch.bfloat16)
    )
    assert producer_event.recorded_stream is None
    assert pool._active == {"first": (0, 1), "second": (0, 1)}
    assert pool._slots[0].state is FeatureSlotState.QUARANTINED
    assert pool._slots[0].active_transfers == {"first", "second"}
    assert pool._slots[0].producer_event is None


@pytest.mark.parametrize("failure_phase", ("inside_publish", "after_publish"))
def test_target_shutdown_retains_pool_ownership_hidden_from_controller(
    monkeypatch, failure_phase
) -> None:
    class FakeTarget:
        def __getitem__(self, _rows):
            return self

        def copy_(self, _source, *, non_blocking):
            assert non_blocking is True

    class FakeStorage:
        def __init__(self) -> None:
            self.target = FakeTarget()

        def __getitem__(self, slot_id):
            assert slot_id == 0
            return self.target

    class FakeProducerEvent:
        def __init__(self) -> None:
            self.recorded = False

        def record(self, _stream) -> None:
            self.recorded = True

        def ipc_handle(self) -> bytes:
            if failure_phase == "inside_publish":
                raise RuntimeError("injected event export failure")
            return b"producer-event"

    producer_event = FakeProducerEvent()
    monkeypatch.setattr(
        torch.cuda,
        "Event",
        lambda *, interprocess: producer_event if interprocess else None,
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: object())

    pool = object.__new__(TargetFeatureSourcePool)
    pool.target_instance_id = "target-0"
    pool.source_session_id = "target-session"
    pool.pool_generation = "pool-generation"
    pool.transport_session = "transport-session"
    pool.device_uuid = "GPU-target"
    pool.device_index = 0
    pool.device = torch.device("cuda", 0)
    pool.slot_count = 1
    pool.max_tokens = 1
    pool.feature_width = 2
    pool._slot_nbytes = 4
    storage = FakeStorage()
    pool._storage = storage
    pool._allocations = (SimpleNamespace(length=4096, ipc_handle=b"allocation"),)
    pool._buffer = SimpleNamespace(allocation_index=0, allocation_offset=0)
    pool._slots = [
        SimpleNamespace(
            generation=0,
            state=FeatureSlotState.FREE,
            active_transfers=set(),
            producer_event=None,
        )
    ]
    pool._active = {}
    pool._cancelled_transfers = set()
    pool._completed = OrderedDict()
    pool._completed_limit = 64
    pool._closed = False
    pool._lock = Lock()
    pool._validate_hidden_states = lambda _hidden: None

    controller = object.__new__(DTTargetController)
    controller.instance_id = "target-0"
    controller.is_export_rank = True
    controller.is_persistent = False
    controller.is_feature_backed = False
    controller.is_token_only = False
    controller.round_timeout_ns = 1
    controller._source_shutdown_quarantine_error = None
    controller._staged_flashmtp_source_leases = set()
    controller._pending_send = deque()
    controller._parked = {}
    controller._feature_control_rounds = {}
    controller._verify_transfer_by_request = {}
    controller._source_slot_by_transfer = {}
    controller._source_deadline_by_transfer = {}
    controller._lane_by_request = {}
    controller._next_round = {}
    controller._target_inflight_counts = {}
    controller._target_inflight_reqs = {}
    controller._lifecycle_attempt_by_request = {}
    controller._draft_kv_handle_by_request = {}
    controller._delivery_outbox = RetryOutbox(retry_interval_ns=1)
    controller._ledger = SimpleNamespace(
        get=lambda _transfer_id: None,
        finish=lambda _transfer_id: None,
    )
    controller._flush_reliable_outbox = lambda: None
    controller._consume_shutdown_acks = lambda: None

    hidden = torch.empty((1, 2), dtype=torch.bfloat16)
    transfer = FeatureTransferSlice(
        transfer_id="transfer-0", row_offset=0, row_count=1
    )
    if failure_phase == "inside_publish":
        with pytest.raises(RuntimeError, match="event export failure"):
            pool.publish(
                hidden_states=hidden,
                feature_schema_id="feature-schema",
                transfers=(transfer,),
            )
    else:
        published = pool.publish(
            hidden_states=hidden,
            feature_schema_id="feature-schema",
            transfers=(transfer,),
        )
        descriptor = published.descriptors[0]
        controller._source_slot_by_transfer[descriptor.transfer_id] = (
            descriptor.source_slot_id,
            descriptor.slot_generation,
        )
        pool.quarantine(
            source_slot_id=descriptor.source_slot_id,
            slot_generation=descriptor.slot_generation,
            source_session_id=descriptor.source_session_id,
            pool_generation=descriptor.pool_generation,
        )
        # This mirrors both Target publication exception handlers: once the
        # pool owns an uncertain quarantined range, controller-local routing
        # ownership is deliberately removed.
        controller._source_slot_by_transfer.pop(descriptor.transfer_id)

    assert producer_event.recorded
    assert controller._source_slot_by_transfer == {}
    assert pool.active_lease_count == 1
    assert pool.quarantined_slot_count == 1
    assert pool._slots[0].producer_event is producer_event

    close_calls = []
    real_pool_close = pool.close

    def tracked_pool_close():
        close_calls.append(True)
        return real_pool_close()

    pool.close = tracked_pool_close
    controller._feature_source_pool = pool
    host_close_calls = []
    controller._result_stores = {
        "draft-0": SimpleNamespace(close=lambda: host_close_calls.append("store"))
    }
    controller._client = SimpleNamespace(
        close=lambda: host_close_calls.append("client")
    )
    drain_calls = []

    def drain() -> None:
        drain_calls.append(True)
        DTTargetController._drain_token_only_shutdown(controller)

    controller._drain_token_only_shutdown = drain

    with pytest.raises(
        RuntimeError, match="unresolved direct-GPU source pool ownership"
    ) as first_failure:
        controller.close()

    assert drain_calls == [True]
    assert close_calls == [True]
    assert host_close_calls == ["store", "client"]
    assert controller._source_shutdown_quarantine_error is first_failure.value
    assert controller._feature_source_pool is pool
    assert pool._closed
    assert pool._storage is storage
    assert pool._slots[0].producer_event is producer_event
    assert pool._active == {"transfer-0": (0, 1)}
    assert pool.quarantined_slot_count == 1

    with pytest.raises(RuntimeError, match="ownership is retained") as repeated:
        controller.close()

    assert repeated.value.__cause__ is first_failure.value
    assert drain_calls == [True]
    assert close_calls == [True]
    assert host_close_calls == ["store", "client"]
    assert controller._feature_source_pool is pool
    assert pool._storage is storage
    assert pool._slots[0].producer_event is producer_event
    assert pool._active == {"transfer-0": (0, 1)}


def test_feature_pool_close_fences_a_publisher_paused_before_reservation() -> None:
    validation_started = Event()
    resume_publication = Event()

    class UnreachableStorage:
        def __init__(self) -> None:
            self.accesses = 0

        def __getitem__(self, _slot_id):
            self.accesses += 1
            raise AssertionError("closed publication must not access GPU storage")

    pool = object.__new__(TargetFeatureSourcePool)
    pool.max_tokens = 1
    pool.feature_width = 2
    pool.slot_count = 1
    pool._closed = False
    pool._lock = Lock()
    pool._active = {}
    pool._cancelled_transfers = set()
    pool._completed = OrderedDict()
    pool._completed_limit = 64
    pool._slots = [
        SimpleNamespace(
            generation=0,
            state=FeatureSlotState.FREE,
            active_transfers=set(),
            producer_event=None,
        )
    ]
    storage = UnreachableStorage()
    pool._storage = storage
    pool._validate_hidden_states = lambda _hidden: None

    def pause_before_reservation(_slices, _token_count) -> None:
        validation_started.set()
        assert resume_publication.wait(timeout=2)

    pool._validate_slices = pause_before_reservation
    hidden = torch.empty((1, 2), dtype=torch.bfloat16)
    transfer = FeatureTransferSlice(
        transfer_id="transfer-raced", row_offset=0, row_count=1
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        publishing = executor.submit(
            pool.publish,
            hidden_states=hidden,
            feature_schema_id="feature-schema",
            transfers=(transfer,),
        )
        assert validation_started.wait(timeout=2)
        assert pool.close() == (0, 0)
        resume_publication.set()
        with pytest.raises(FeatureTransportError, match="source pool is closed"):
            publishing.result(timeout=2)

    assert storage.accesses == 0
    assert pool._active == {}
    assert pool._slots[0].generation == 0
    assert pool._slots[0].state is FeatureSlotState.FREE
    assert pool._slots[0].active_transfers == set()
    assert pool._slots[0].producer_event is None


def _source_pool_with_pending_producer(producer_event):
    storage = object()
    pool = object.__new__(TargetFeatureSourcePool)
    pool.source_session_id = "target-session"
    pool.pool_generation = "pool-generation"
    pool.slot_count = 1
    pool._storage = storage
    pool._slots = [
        SimpleNamespace(
            generation=1,
            state=FeatureSlotState.PUBLISHED,
            active_transfers={"transfer-0"},
            producer_event=producer_event,
        )
    ]
    pool._active = {"transfer-0": (0, 1)}
    pool._cancelled_transfers = set()
    pool._completed = OrderedDict()
    pool._completed_limit = 64
    pool._closed = False
    pool._lock = Lock()
    return pool, storage


def test_feature_source_cancel_defers_retirement_until_producer_completion() -> None:
    class DelayedProducerEvent:
        ready = False

        def __init__(self) -> None:
            self.query_calls = 0

        def query(self) -> bool:
            self.query_calls += 1
            return self.ready

    producer_event = DelayedProducerEvent()
    pool, storage = _source_pool_with_pending_producer(producer_event)

    assert not pool.cancel(
        transfer_id="transfer-0",
        source_slot_id=0,
        slot_generation=1,
        source_session_id="target-session",
        pool_generation="pool-generation",
    )
    assert pool._active == {"transfer-0": (0, 1)}
    assert pool._cancelled_transfers == {"transfer-0"}
    assert pool._slots[0].producer_event is producer_event
    assert pool._storage is storage

    producer_event.ready = True
    assert pool.free_slot_count == 1
    assert pool._active == {}
    assert pool._cancelled_transfers == set()
    assert pool._slots[0].state is FeatureSlotState.FREE
    assert pool._slots[0].producer_event is None
    assert pool.close() == (0, 0)


def test_feature_source_close_retains_pending_cancelled_producer() -> None:
    class PendingProducerEvent:
        def __init__(self) -> None:
            self.query_calls = 0

        def query(self) -> bool:
            self.query_calls += 1
            return False

    producer_event = PendingProducerEvent()
    pool, storage = _source_pool_with_pending_producer(producer_event)

    assert not pool.cancel(
        transfer_id="transfer-0",
        source_slot_id=0,
        slot_generation=1,
        source_session_id="target-session",
        pool_generation="pool-generation",
    )
    assert pool.close() == (1, 0)
    assert pool.close() == (1, 0)
    assert pool._closed
    assert pool._active == {"transfer-0": (0, 1)}
    assert pool._cancelled_transfers == {"transfer-0"}
    assert pool._slots[0].state is FeatureSlotState.PUBLISHED
    assert pool._slots[0].producer_event is producer_event
    assert pool._storage is storage


def test_feature_source_cancel_query_failure_quarantines_without_retry() -> None:
    class BrokenProducerEvent:
        def __init__(self) -> None:
            self.query_calls = 0

        def query(self) -> bool:
            self.query_calls += 1
            raise RuntimeError("injected producer query failure")

    producer_event = BrokenProducerEvent()
    pool, storage = _source_pool_with_pending_producer(producer_event)

    with pytest.raises(
        FeatureTransportQuiescenceError,
        match="failed to query cancelled feature producer completion",
    ) as failure:
        pool.cancel(
            transfer_id="transfer-0",
            source_slot_id=0,
            slot_generation=1,
            source_session_id="target-session",
            pool_generation="pool-generation",
        )

    assert isinstance(failure.value.__cause__, RuntimeError)
    assert producer_event.query_calls == 1
    assert pool.close() == (1, 1)
    assert pool.close() == (1, 1)
    assert producer_event.query_calls == 1
    assert pool._active == {"transfer-0": (0, 1)}
    assert pool._cancelled_transfers == {"transfer-0"}
    assert pool._slots[0].state is FeatureSlotState.QUARANTINED
    assert pool._slots[0].producer_event is producer_event
    assert pool._storage is storage


def test_eagle3_feature_staging_view_requires_proposal_fence_and_adjacency() -> None:
    contiguous = ((2, 5), (5, 9), (9, 12))
    assert _eagle3_feature_staging_view_bounds(
        requires_proposal=True, row_ranges=contiguous
    ) == (2, 12)
    assert (
        _eagle3_feature_staging_view_bounds(
            requires_proposal=False, row_ranges=contiguous
        )
        is None
    )


def test_eagle3_feature_staging_fence_marks_failed_stream_unquiescent(
    monkeypatch,
) -> None:
    engine = object.__new__(FlashMTPDraftEngine)
    engine.device = torch.device("cuda:0")
    calls = []

    class Stream:
        def synchronize(self) -> None:
            calls.append("fenced")

    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: Stream())

    engine._fence_eagle3_feature_staging_reads()

    assert calls == ["fenced"]

    class BrokenStream:
        def synchronize(self) -> None:
            raise RuntimeError("simulated fence failure")

    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: BrokenStream())
    with pytest.raises(RuntimeError, match="failed to fence") as raised:
        engine._fence_eagle3_feature_staging_reads()
    assert raised.value.resources_quiescent is False


def test_eagle3_materialization_event_owns_kv_and_staging_until_complete(
    monkeypatch,
) -> None:
    engine = object.__new__(FlashMTPDraftEngine)
    events = []
    engine.device = torch.device("cuda:0")
    engine.instance_id = "draft-0"
    engine._pending_eagle3_completions = deque()
    engine._publish_materialization_successes = lambda _tasks: events.append("publish")
    engine._publish_failures = lambda _failures: events.append("failure")
    engine._release_persistent_request = lambda _handle: events.append("release")
    engine._drain_deferred_persistent_release = lambda _handle: events.append(
        "drain-release"
    )
    engine._publish_capacity = lambda: events.append("capacity")
    engine._cancelled_ids = set()

    class CompletionEvent:
        ready = False

        def record(self, _stream) -> None:
            events.append("record")

        def query(self) -> bool:
            return self.ready

    completion = CompletionEvent()
    monkeypatch.setattr(torch.cuda, "Event", lambda: completion)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: object())

    class Slots:
        def append(self, _slot) -> None:
            events.append("reuse")

    engine._free_feature_staging_slots = Slots()
    task = SimpleNamespace(
        transfer_id="transfer-0",
        draft_kv_handle="handle-0",
        deadline_ns=2**63 - 1,
    )
    pending = [SimpleNamespace(task=task)]

    engine._enqueue_eagle3_materialization_completion(
        pending, (2,), staging_slot_counts=(1,)
    )
    engine._poll_eagle3_completions()

    assert events == ["record"]
    assert len(engine._pending_eagle3_completions) == 1

    completion.ready = True
    engine._poll_eagle3_completions()

    assert events == ["record", "publish", "reuse", "drain-release", "capacity"]
    assert list(engine._pending_eagle3_completions) == []


def test_eagle3_pending_cancel_releases_unsubmitted_feature_source() -> None:
    engine = object.__new__(FlashMTPDraftEngine)
    task = SimpleNamespace(
        transfer_id="transfer-eagle3-cancel",
        source_read_migration_id="",
        draft_kv_handle="handle-eagle3-cancel",
    )
    engine._persistent = True
    engine._is_feature_backed = True
    engine._pending = deque((task,))
    engine._pending_ids = {task.transfer_id}
    engine._cancelled_ids = {task.transfer_id}
    engine._received_ns_by_transfer = {task.transfer_id: 123}
    released = []
    retired = []
    engine._publish_source_done = released.append
    engine._drain_deferred_persistent_release = retired.append

    engine._drop_cancelled_pending()

    assert list(engine._pending) == []
    assert engine._pending_ids == set()
    assert engine._cancelled_ids == set()
    assert engine._received_ns_by_transfer == {}
    assert released == [task]
    assert retired == [task.draft_kv_handle]


def test_cancelled_transfer_identity_waits_only_for_direct_tombstone_ack() -> None:
    mmap_task = SimpleNamespace(
        transfer_id="mmap-cancel",
        candidate_destination=None,
    )
    direct_task = SimpleNamespace(
        transfer_id="direct-cancel",
        candidate_destination=object(),
    )
    engine = object.__new__(FlashMTPDraftEngine)
    engine._pending_ids = {mmap_task.transfer_id, direct_task.transfer_id}
    engine._cancelled_ids = {mmap_task.transfer_id, direct_task.transfer_id}
    engine._received_ns_by_transfer = {
        mmap_task.transfer_id: 1,
        direct_task.transfer_id: 2,
    }

    engine._retire_cancelled_transfer_identity(mmap_task)
    engine._retire_cancelled_transfer_identity(direct_task)

    assert engine._pending_ids == {direct_task.transfer_id}
    assert engine._cancelled_ids == set()
    assert engine._received_ns_by_transfer == {}


def test_eagle3_token_metadata_batches_ragged_streams_without_reordering() -> None:
    token_chunks = [
        torch.tensor([10, 11]),
        torch.tensor([20]),
        torch.tensor([30, 31, 32]),
    ]
    committed_chunks = [
        torch.tensor([110, 111]),
        torch.tensor([120]),
        torch.tensor([130, 131, 132]),
    ]
    tokens, committed = _batched_eagle3_token_metadata(
        active_indices=[2, 0],
        token_chunks=token_chunks,
        committed_chunks=committed_chunks,
    )
    assert tokens == {2: (30, 31, 32), 0: (10, 11)}
    assert committed == {2: (130, 131, 132), 0: (110, 111)}

    with pytest.raises(ValueError, match="lengths do not match"):
        _batched_eagle3_token_metadata(
            active_indices=[0],
            token_chunks=token_chunks,
            committed_chunks=[torch.tensor([110])],
        )
    assert (
        _eagle3_feature_staging_view_bounds(
            requires_proposal=True, row_ranges=((2, 5), (7, 9))
        )
        is None
    )


def test_eagle3_handoff_keeps_feature_and_committed_token_streams_distinct() -> None:
    worker = object.__new__(EAGLE3DisaggregatedTargetWorkerV2)
    worker.device = torch.device("cpu")
    hidden = [
        torch.randn((2, 6), dtype=torch.bfloat16),
        torch.randn((1, 6), dtype=torch.bfloat16),
        torch.randn((3, 6), dtype=torch.bfloat16),
    ]
    # Feature rows condition on the accepted outputs plus the new bonus, while
    # Draft KV commits the incoming anchor plus accepted draft inputs.
    feature_tokens = [
        torch.tensor([44, 55], dtype=torch.int64),
        torch.tensor([77], dtype=torch.int64),
        torch.tensor([99, 100, 101], dtype=torch.int64),
    ]
    committed_tokens = [
        torch.tensor([33, 44], dtype=torch.int64),
        torch.tensor([66], dtype=torch.int64),
        torch.tensor([88, 99, 100], dtype=torch.int64),
    ]

    handoff = worker._make_handoff(
        bonus_tokens=torch.tensor([55, 77, 101], dtype=torch.int64),
        new_seq_lens=torch.tensor([4, 7, 11], dtype=torch.int64),
        feature_chunks=hidden,
        token_chunks=feature_tokens,
        committed_token_chunks=committed_tokens,
        feature_offsets=[2, 6, 8],
    )

    assert handoff.hidden_states.shape == (3, 0)
    assert handoff.dt_feature_chunks is hidden
    assert handoff.dt_feature_token_chunks is feature_tokens
    assert handoff.dt_committed_token_chunks is committed_tokens
    assert handoff.dt_feature_token_chunks[0].tolist() == [44, 55]
    assert handoff.dt_committed_token_chunks[0].tolist() == [33, 44]

    handoff.filter_batch(torch.tensor([2, 0]), new_indices_cpu=[2, 0])
    assert handoff.hidden_states.shape == (2, 0)
    assert handoff.dt_feature_offsets == [8, 2]
    assert [row.tolist() for row in handoff.dt_feature_token_chunks] == [
        [99, 100, 101],
        [44, 55],
    ]
    assert [row.tolist() for row in handoff.dt_committed_token_chunks] == [
        [88, 99, 100],
        [33, 44],
    ]


@pytest.mark.parametrize(
    "feature_path",
    ("eagle3_verify", "eagle3_prefill", "dflash_prefill"),
)
def test_feature_publication_fences_graph_buffer_reuse(feature_path: str) -> None:
    """Every graph-backed feature handoff fences replay after pool publication."""

    pivot_width = 6
    if feature_path == "eagle3_verify":
        graph_output = torch.arange(
            4 * pivot_width, dtype=torch.float32
        ).to(torch.bfloat16).view(4, pivot_width)
        candidates = torch.tensor([[10, 11, 12, 13]], dtype=torch.int64)
        feature_chunks, token_chunks = split_dflash_committed_rows(
            packed_hidden=graph_output,
            candidates=candidates,
            commit_lens=torch.tensor([2], dtype=torch.int64),
            block_size=4,
        )
        committed_chunks = [candidates[0, :2]]
    else:
        graph_output = torch.arange(
            2 * pivot_width, dtype=torch.float32
        ).to(torch.bfloat16).view(2, pivot_width)
        feature_chunks = list(torch.split(graph_output, [2], dim=0))
        input_chunks = list(
            torch.split(torch.tensor([10, 11], dtype=torch.int64), [2], dim=0)
        )
        if feature_path == "eagle3_prefill":
            token_chunks = [torch.tensor([11, 12], dtype=torch.int64)]
            committed_chunks = input_chunks
        else:
            token_chunks = input_chunks
            committed_chunks = None

    # All three production paths retain an already-contiguous view.  Calling
    # contiguous() in the controller therefore does not establish ownership.
    assert feature_chunks[0].is_contiguous()
    assert feature_chunks[0].contiguous().data_ptr() == graph_output.data_ptr()

    if feature_path.startswith("eagle3"):
        worker = object.__new__(EAGLE3DisaggregatedTargetWorkerV2)
        worker.device = torch.device("cpu")
        handoff = worker._make_handoff(
            bonus_tokens=torch.tensor([12], dtype=torch.int64),
            new_seq_lens=torch.tensor([2], dtype=torch.int64),
            feature_chunks=feature_chunks,
            token_chunks=token_chunks,
            committed_token_chunks=committed_chunks,
            feature_offsets=[0],
        )
    else:
        handoff = DFlashDraftInputV2.create_idle_input(torch.device("cpu"))
        handoff.bonus_tokens = torch.tensor([12], dtype=torch.int64)
        handoff.new_seq_lens = torch.tensor([2], dtype=torch.int64)
        handoff.dt_feature_chunks = feature_chunks
        handoff.dt_feature_token_chunks = token_chunks
        handoff.dt_feature_offsets = [0]

    trace = []
    producer_event = object()

    class ScalarCopyDone:
        def record(self) -> None:
            pass

        def synchronize(self) -> None:
            pass

    class ForwardStream:
        def wait_event(self, event) -> None:
            trace.append(("forward_wait", event))

    class FeatureSourcePool:
        def publish(self, *, hidden_states, **_kwargs):
            # Publication receives the graph-buffer view, so correctness rests
            # on the reverse stream dependency rather than an incidental copy.
            assert hidden_states[0].data_ptr() == graph_output.data_ptr()
            trace.append(("publish", producer_event))
            return SimpleNamespace(
                producer_event=producer_event,
                hidden_produced_ns=1,
                descriptor_published_ns=2,
                pack_strategy="copy-loop",
                pack_chunk_count=len(hidden_states),
                descriptors=(
                    SimpleNamespace(
                        transfer_id="target-0:request-0:0:0",
                        source_slot_id=3,
                        slot_generation=4,
                        source_session_id="target-session",
                        pool_generation="pool-generation",
                        nbytes=int(hidden_states[0].numel() * 2),
                    ),
                ),
            )

        def quarantine(self, **_kwargs) -> None:
            raise AssertionError("successful publication must not be quarantined")

    controller = object.__new__(DTTargetController)
    controller.scheduler = SimpleNamespace(
        forward_stream=ForwardStream(),
        ps=SimpleNamespace(tp_size=1),
        device_module=SimpleNamespace(Event=ScalarCopyDone),
    )
    controller.is_export_rank = True
    controller.is_feature_backed = True
    controller.is_token_only = False
    controller.is_eagle3 = feature_path.startswith("eagle3")
    controller.pivot_width = pivot_width
    controller.block_size = 4
    controller.instance_id = "target-0"
    controller.feature_schema_id = "feature-schema"
    controller._feature_source_pool = FeatureSourcePool()
    controller._next_round = {}
    controller._draft_committed_len = {}
    controller._draft_kv_handle_by_request = {}
    controller._verify_transfer_by_request = {}
    controller._lifecycle_attempt_by_request = {}
    controller._next_lifecycle_attempt = 0
    controller._source_slot_by_transfer = {}
    controller._profile_emit = lambda *_args, **_kwargs: None

    req = SimpleNamespace(
        rid="request-0", finished=lambda: False, is_retracted=False
    )
    batch = SimpleNamespace(reqs=[req])
    batch_result = SimpleNamespace(
        next_draft_input=handoff,
        dt_deferred_dflash_feature_stage=False,
        dt_hidden_produced_ns=1,
    )

    controller.stage_overlap_pivot_copy(batch, batch_result)

    assert trace == [
        ("publish", producer_event),
        ("forward_wait", producer_event),
    ]
    assert batch_result.dt_dflash_feature_publish_strategy == "copy-loop"
    assert batch_result.dt_dflash_feature_publish_chunks == 1
    assert len(batch_result.dt_staged_dflash_features.tasks) == 1


def test_dflash_grouped_suffix_free_flushes_before_reuse_and_cleanup() -> None:
    allocator = TokenToKVPoolAllocator(
        size=5,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=None,
        need_sort=False,
    )
    old_pages = allocator.alloc(5)
    assert old_pages is not None
    req_to_token = torch.zeros((2, 8), dtype=torch.int64)
    req_to_token[1, :5] = old_pages
    req = SimpleNamespace(
        req_pool_idx=1,
        kv=SimpleNamespace(kv_allocated_len=5),
    )
    state = SimpleNamespace(
        request_id="request-0",
        target_instance_id="target-0",
        draft_kv_handle="kv-0",
        last_round_id=0,
        valid_committed_len=2,
        proposal_base_len=2,
        req=req,
    )
    task = SimpleNamespace(
        kind=DraftTaskKind.RECONCILE,
        request_id=state.request_id,
        target_instance_id=state.target_instance_id,
        draft_kv_handle=state.draft_kv_handle,
        round_id=1,
        feature_token_offset=2,
    )
    engine = object.__new__(FlashMTPDraftEngine)
    engine._persistent_requests = {state.draft_kv_handle: state}
    engine._draft_model_runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        token_to_kv_pool_allocator=allocator,
        server_args=SimpleNamespace(page_size=1),
    )

    assert engine._free_dflash_reconcile_suffixes([task, task]) is None
    assert req.kv.kv_allocated_len == 5
    assert allocator.free_pages.numel() == 0
    discarded = engine._free_dflash_reconcile_suffixes([task])

    assert discarded == {state.draft_kv_handle: 3}
    assert req.kv.kv_allocated_len == 2
    replacement = allocator.alloc(2)
    assert replacement is not None
    assert replacement.tolist() == [3, 4]
    req_to_token[1, 2:4] = replacement
    assert allocator.free_pages.tolist() == [5]

    # Emulate failure cleanup after the replacement pages became owned.  The
    # committed prefix plus replacement must free every page exactly once.
    req.kv.kv_allocated_len = 4
    allocator.free(req_to_token[1, : req.kv.kv_allocated_len])
    released = allocator.free_pages.tolist()
    assert sorted(released) == [1, 2, 3, 4, 5]
    assert len(released) == len(set(released))


def test_dflash_grouped_reconcile_reuses_packed_rows_and_cleanup_ownership() -> None:
    allocator = TokenToKVPoolAllocator(
        size=10,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=None,
        need_sort=False,
    )
    old_pages = allocator.alloc(10)
    assert old_pages is not None
    req_to_token = torch.zeros((3, 8), dtype=torch.int32)
    req_to_token[1, :5] = old_pages[:5]
    req_to_token[2, :5] = old_pages[5:]

    def make_req(row: int):
        req = SimpleNamespace(
            req_pool_idx=row,
            kv=SimpleNamespace(kv_allocated_len=5),
            kv_committed_len=0,
            prefix_indices=torch.empty((0,), dtype=torch.int64),
            output_ids=[],
        )
        req._refresh_fill_ids = lambda: None
        req.set_extend_range = lambda start, end: setattr(
            req, "extend_range", (start, end)
        )
        return req

    reqs = [make_req(1), make_req(2)]
    states = [
        SimpleNamespace(
            request_id=f"request-{index}",
            target_instance_id="target-0",
            draft_kv_handle=f"kv-{index}",
            last_round_id=0,
            valid_committed_len=committed_len,
            proposal_base_len=committed_len,
            committed_token_ids=list(range(committed_len)),
            req=req,
        )
        for index, (req, committed_len) in enumerate(zip(reqs, (2, 3)))
    ]
    tasks = [
        SimpleNamespace(
            kind=DraftTaskKind.RECONCILE,
            request_id=state.request_id,
            target_instance_id=state.target_instance_id,
            draft_kv_handle=state.draft_kv_handle,
            round_id=1,
            feature_token_offset=state.valid_committed_len,
            feature_token_count=token_count,
            committed_kv_len=state.valid_committed_len + token_count,
            committed_token_ids=list(range(100 + index, 100 + index + token_count)),
        )
        for index, (state, token_count) in enumerate(zip(states, (1, 2)))
    ]
    engine = object.__new__(FlashMTPDraftEngine)
    engine.device = torch.device("cpu")
    engine.profile = SimpleNamespace(draft_k=7)
    engine.instance_id = "draft-0"
    _init_lifetime_state(engine)
    engine._persistent_requests = {
        state.draft_kv_handle: state for state in states
    }
    req_pool = SimpleNamespace(req_to_token=req_to_token)
    req_pool.free = lambda req: setattr(req, "req_pool_idx", None)
    engine._draft_model_runner = SimpleNamespace(
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=allocator,
        server_args=SimpleNamespace(page_size=1),
    )
    alloc_calls = []
    original_alloc = allocator.alloc

    def counted_alloc(size):
        alloc_calls.append(size)
        return original_alloc(size)

    allocator.alloc = counted_alloc
    phases = [0] * 6
    discarded = engine._free_dflash_reconcile_suffixes(tasks, phases)
    grouped = engine._allocate_dflash_reconcile_rows(tasks, discarded, phases)
    assert grouped is not None
    cache_locs, packed_cache_locs, positions = grouped

    assert alloc_calls == [3]
    assert req_to_token.dtype == torch.int32
    assert packed_cache_locs.dtype == torch.int64
    assert positions.dtype == torch.int64
    assert all(cache_loc.dtype == torch.int64 for cache_loc in cache_locs)
    assert packed_cache_locs.device == req_to_token.device
    assert positions.device == req_to_token.device
    assert positions.tolist() == [2, 3, 4]
    assert torch.cat(cache_locs).tolist() == packed_cache_locs.tolist()
    assert [req.kv.kv_allocated_len for req in reqs] == [3, 5]
    assert req_to_token[1, 2:3].tolist() == cache_locs[0].tolist()
    assert req_to_token[2, 3:5].tolist() == cache_locs[1].tolist()

    for task, cache_loc in zip(tasks, cache_locs, strict=True):
        engine._prepare_dflash_task(
            task,
            phases,
            suffix_already_freed=True,
            discarded_tokens_override=discarded[task.draft_kv_handle],
            refresh_prefix_indices=False,
            cache_loc_override=cache_loc,
            row_already_written=True,
        )
    assert alloc_calls == [3]
    assert [state.valid_committed_len for state in states] == [3, 5]
    assert [req.extend_range for req in reqs] == [(2, 3), (3, 5)]

    for state in states:
        assert engine._release_persistent_request(state.draft_kv_handle)
    released = allocator.free_pages.tolist()
    assert sorted(released) == list(range(1, 11))
    assert len(released) == len(set(released))


def test_dflash_grouped_reconcile_install_failure_frees_only_replacements() -> None:
    allocator = TokenToKVPoolAllocator(
        size=10,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=None,
        need_sort=False,
    )
    old_pages = allocator.alloc(10)
    assert old_pages is not None
    req_to_token = torch.zeros((3, 8), dtype=torch.int32)
    req_to_token[1, :5] = old_pages[:5]
    req_to_token[2, :5] = old_pages[5:]

    def make_req(row: int):
        req = SimpleNamespace(
            req_pool_idx=row,
            kv=SimpleNamespace(kv_allocated_len=5),
            kv_committed_len=0,
            prefix_indices=torch.empty((0,), dtype=torch.int64),
            output_ids=[],
        )
        req._refresh_fill_ids = lambda: None
        req.set_extend_range = lambda start, end: setattr(
            req, "extend_range", (start, end)
        )
        return req

    reqs = [make_req(1), make_req(2)]
    states = [
        SimpleNamespace(
            request_id=f"request-{index}",
            target_instance_id="target-0",
            draft_kv_handle=f"kv-{index}",
            last_round_id=0,
            valid_committed_len=committed_len,
            proposal_base_len=committed_len,
            committed_token_ids=list(range(committed_len)),
            req=req,
        )
        for index, (req, committed_len) in enumerate(zip(reqs, (2, 3)))
    ]
    tasks = [
        SimpleNamespace(
            kind=DraftTaskKind.RECONCILE,
            request_id=state.request_id,
            target_instance_id=state.target_instance_id,
            draft_kv_handle=state.draft_kv_handle,
            round_id=1,
            feature_token_offset=state.valid_committed_len,
            feature_token_count=token_count,
            committed_kv_len=state.valid_committed_len + token_count,
            committed_token_ids=list(range(100, 100 + token_count)),
        )
        for state, token_count in zip(states, (1, 2))
    ]
    engine = object.__new__(FlashMTPDraftEngine)
    engine.device = torch.device("cpu")
    engine.profile = SimpleNamespace(draft_k=7)
    engine.instance_id = "draft-0"
    _init_lifetime_state(engine)
    engine._persistent_requests = {
        state.draft_kv_handle: state for state in states
    }
    req_pool = SimpleNamespace(req_to_token=req_to_token)
    req_pool.free = lambda req: setattr(req, "req_pool_idx", None)
    engine._draft_model_runner = SimpleNamespace(
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=allocator,
        server_args=SimpleNamespace(page_size=1),
    )

    discarded = engine._free_dflash_reconcile_suffixes(tasks)
    assert discarded == {"kv-0": 3, "kv-1": 2}
    # Corrupt the second row handle only after suffix release. This forces the
    # packed page-table transition to fail after replacement allocation.
    reqs[1].req_pool_idx = req_to_token.shape[0]
    with pytest.raises(IndexError, match="out of bounds"):
        engine._allocate_dflash_reconcile_rows(tasks, discarded)

    # The helper must return all replacement pages to the allocator without
    # publishing the longer ownership boundaries. The stable prefixes remain
    # owned by their original request rows and are independently releasable.
    assert [req.kv.kv_allocated_len for req in reqs] == [2, 3]
    reqs[1].req_pool_idx = 2
    for state in states:
        assert engine._release_persistent_request(state.draft_kv_handle)
    released = allocator.free_pages.tolist()
    assert sorted(released) == list(range(1, 11))
    assert len(released) == len(set(released))


def test_eagle3_grouped_reconcile_preserves_prefixes_and_page_ownership() -> None:
    allocator = TokenToKVPoolAllocator(
        size=10,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=None,
        need_sort=False,
    )
    old_pages = allocator.alloc(10)
    assert old_pages is not None
    req_to_token = torch.zeros((3, 8), dtype=torch.int32)
    req_to_token[1, :5] = old_pages[:5]
    req_to_token[2, :5] = old_pages[5:]

    def make_req(row: int, allocated_len: int):
        req = SimpleNamespace(
            req_pool_idx=row,
            kv=SimpleNamespace(kv_allocated_len=allocated_len),
            kv_committed_len=0,
            prefix_indices=torch.empty((0,), dtype=torch.int64),
            output_ids=[],
        )
        req._refresh_fill_ids = lambda: None
        req.set_extend_range = lambda start, end: setattr(
            req, "extend_range", (start, end)
        )
        return req

    reqs = [make_req(1, 5), make_req(2, 5)]
    states = [
        SimpleNamespace(
            request_id=f"request-{index}",
            target_instance_id="target-0",
            draft_kv_handle=f"kv-{index}",
            last_round_id=0,
            valid_committed_len=committed_len,
            proposal_base_len=committed_len,
            committed_token_ids=list(range(committed_len)),
            req=req,
        )
        for index, (req, committed_len) in enumerate(zip(reqs, (2, 3)))
    ]
    deltas = ([101], [201, 202])
    tasks = [
        SimpleNamespace(
            kind=DraftTaskKind.RECONCILE,
            request_id=state.request_id,
            target_instance_id=state.target_instance_id,
            draft_kv_handle=state.draft_kv_handle,
            round_id=1,
            committed_token_ids=delta,
            committed_kv_len=state.valid_committed_len + len(delta),
        )
        for state, delta in zip(states, deltas)
    ]
    engine = object.__new__(FlashMTPDraftEngine)
    engine.instance_id = "draft-0"
    engine._persistent_requests = {
        state.draft_kv_handle: state for state in states
    }
    engine._draft_model_runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        token_to_kv_pool_allocator=allocator,
        server_args=SimpleNamespace(page_size=1),
    )

    prepared, grouped = engine._prepare_eagle3_persistent_tasks(tasks)

    assert grouped is True
    assert prepared == states
    assert reqs[0].prefix_indices.dtype == torch.int64
    assert reqs[0].prefix_indices.tolist() == [1, 2]
    assert reqs[1].prefix_indices.tolist() == [6, 7, 8]
    assert reqs[0].prefix_indices.untyped_storage().data_ptr() == (
        reqs[1].prefix_indices.untyped_storage().data_ptr()
    )
    assert [req.kv.kv_allocated_len for req in reqs] == [2, 3]
    assert [req.kv_committed_len for req in reqs] == [2, 3]
    assert [req.extend_range for req in reqs] == [(2, 3), (3, 5)]
    assert [state.valid_committed_len for state in states] == [3, 5]
    assert allocator.free_pages.tolist() == [3, 4, 5, 9, 10]
    assert len(allocator.free_pages) == len(set(allocator.free_pages.tolist()))


def test_eagle3_grouped_reconcile_failure_cleanup_does_not_double_free(
    monkeypatch,
) -> None:
    allocator = TokenToKVPoolAllocator(
        size=10,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=None,
        need_sort=False,
    )
    old_pages = allocator.alloc(10)
    assert old_pages is not None
    req_to_token = torch.zeros((3, 8), dtype=torch.int32)
    req_to_token[1, :5] = old_pages[:5]
    req_to_token[2, :5] = old_pages[5:]

    def make_req(row: int, allocated_len: int):
        req = SimpleNamespace(
            req_pool_idx=row,
            kv=SimpleNamespace(kv_allocated_len=allocated_len),
            kv_committed_len=0,
            prefix_indices=torch.empty((0,), dtype=torch.int64),
            output_ids=[],
        )
        req._refresh_fill_ids = lambda: None
        req.set_extend_range = lambda start, end: setattr(
            req, "extend_range", (start, end)
        )
        return req

    reqs = [make_req(1, 5), make_req(2, 5)]
    states = [
        SimpleNamespace(
            request_id=f"request-{index}",
            target_instance_id="target-0",
            draft_kv_handle=f"kv-{index}",
            last_round_id=0,
            valid_committed_len=committed_len,
            proposal_base_len=committed_len,
            committed_token_ids=list(range(committed_len)),
            req=req,
        )
        for index, (req, committed_len) in enumerate(zip(reqs, (2, 3)))
    ]
    tasks = [
        SimpleNamespace(
            kind=DraftTaskKind.RECONCILE,
            request_id=state.request_id,
            target_instance_id=state.target_instance_id,
            draft_kv_handle=state.draft_kv_handle,
            round_id=1,
            committed_token_ids=[100 + index],
            committed_kv_len=state.valid_committed_len + 1,
        )
        for index, state in enumerate(states)
    ]
    req_pool = SimpleNamespace(req_to_token=req_to_token)
    req_pool.free = lambda req: setattr(req, "req_pool_idx", None)
    engine = object.__new__(FlashMTPDraftEngine)
    engine.instance_id = "draft-0"
    engine._persistent_requests = {
        state.draft_kv_handle: state for state in states
    }
    engine._draft_model_runner = SimpleNamespace(
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=allocator,
        server_args=SimpleNamespace(page_size=1),
    )
    engine._quarantined_feature_draft_handles = set()
    engine._deferred_persistent_releases = {}
    engine._pending = deque()
    engine._pending_feature_batches = deque()
    engine._pending_dflash_completions = deque()
    engine._migration_source_handles = {}

    monkeypatch.setattr(
        torch,
        "split",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected packed-prefix failure")
        ),
    )
    with pytest.raises(RuntimeError, match="injected packed-prefix failure"):
        engine._prepare_eagle3_persistent_tasks(tasks)

    # The grouped suffix release completed, so cleanup must see only the
    # shortened committed ownership boundaries.
    assert [req.kv.kv_allocated_len for req in reqs] == [2, 3]
    assert allocator.free_pages.tolist() == [3, 4, 5, 9, 10]
    for state in states:
        assert engine._release_persistent_request(state.draft_kv_handle)

    released = allocator.free_pages.tolist()
    assert sorted(released) == list(range(1, 11))
    assert len(released) == len(set(released))
    assert engine._persistent_requests == {}
    assert all(req.kv is None and req.req_pool_idx is None for req in reqs)


def test_late_unquiescent_failure_quarantines_after_result_is_stale() -> None:
    controller = object.__new__(DTTargetController)
    controller.is_dflash = True
    controller.is_feature_backed = True
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


def test_source_read_done_batch_queues_one_reliable_ack() -> None:
    entries = tuple(
        SourceReadDone(
            protocol_version=PROTOCOL_VERSION,
            transfer_id=f"transfer-{index}",
            target_instance_id="target-0",
            draft_instance_id="draft-0",
            source_slot_id=0,
            slot_generation=1,
            completed_ns=1,
        )
        for index in range(3)
    )
    batch = SourceReadDoneBatch(
        protocol_version=PROTOCOL_VERSION,
        batch_id="source-done-batch-0",
        draft_instance_id="draft-0",
        entries=entries,
        sent_ns=1,
    )
    controller = object.__new__(DTTargetController)
    controller._client = SimpleNamespace(recv_nowait=lambda: (batch,))
    controller._draft_progress_ns = {}
    controller._flush_pending_send = lambda: None
    controller._expire_local_rounds = lambda: []
    controller._expire_source_leases = lambda **_kwargs: None
    controller._delivery_outbox = RetryOutbox(retry_interval_ns=1)
    controller._source_done_batch_entries = {}
    controller._maybe_emit_lifecycle_status = lambda: None
    validated = []
    consumed = []
    flushes = []

    def consume(message, *, flush=True, prevalidated=False):
        consumed.append((message, flush, prevalidated))
        controller._delivery_outbox.add(
            "source_read_done", source_read_done_id(message), message
        )

    controller._validate_source_read_done = validated.append
    controller._consume_source_read_done = consume
    controller._flush_reliable_outbox = lambda: flushes.append(True)

    assert controller._poll_export_rank() == []
    assert validated == list(entries)
    assert consumed == [(entry, False, True) for entry in entries]
    assert flushes == [True]
    pending = controller._delivery_outbox.pending()
    assert len(pending) == 1
    operation, message_id, acknowledgement = pending[0]
    assert operation == "source_read_done"
    assert message_id == batch.batch_id
    assert isinstance(acknowledgement, SourceReadDoneBatchAck)
    assert acknowledgement.draft_instance_id == batch.draft_instance_id
    assert acknowledgement.message_ids == tuple(
        source_read_done_id(entry) for entry in entries
    )


def test_source_read_done_batch_prevalidates_before_releasing_any_slot() -> None:
    class FeaturePool:
        def __init__(self) -> None:
            self.released = []

        def validate_read_done(self, *, transfer_id, **_kwargs) -> None:
            if transfer_id == "transfer-1":
                raise ValueError("invalid second lease")

        def read_done(self, *, transfer_id, **_kwargs) -> None:
            self.released.append(transfer_id)

    entries = tuple(
        SourceReadDone(
            protocol_version=PROTOCOL_VERSION,
            transfer_id=f"transfer-{index}",
            target_instance_id="target-0",
            draft_instance_id="draft-0",
            source_slot_id=index,
            slot_generation=1,
            completed_ns=1,
            instance_session_id="draft-session",
            coordinator_session_id="coordinator-session",
            source_session_id="source-session",
            pool_generation="pool-generation",
        )
        for index in range(2)
    )
    controller = object.__new__(DTTargetController)
    controller.instance_id = "target-0"
    controller.is_feature_backed = True
    controller._client = SimpleNamespace(
        coordinator_session_id="coordinator-session"
    )
    controller._feature_source_pool = FeaturePool()
    controller._source_done_applied_ids = set()
    controller._source_slot_by_transfer = {
        entry.transfer_id: (entry.source_slot_id, entry.slot_generation)
        for entry in entries
    }
    controller._source_deadline_by_transfer = {
        entry.transfer_id: 10 for entry in entries
    }
    controller._delivery_outbox = RetryOutbox(retry_interval_ns=1)

    with pytest.raises(ValueError, match="invalid second lease"):
        controller._consume_source_read_done_batch(
            SourceReadDoneBatch(
                protocol_version=PROTOCOL_VERSION,
                batch_id="source-done-batch-invalid",
                draft_instance_id="draft-0",
                entries=entries,
                sent_ns=1,
            )
        )

    assert controller._feature_source_pool.released == []
    assert set(controller._source_slot_by_transfer) == {
        "transfer-0",
        "transfer-1",
    }
    assert not controller._source_done_applied_ids
    assert not controller._delivery_outbox.pending()


def test_source_read_done_batch_ack_rejects_stale_session() -> None:
    engine = object.__new__(FlashMTPDraftEngine)
    engine.instance_id = "draft-0"
    engine._client = SimpleNamespace(
        instance_session_id="draft-session",
        coordinator_session_id="coordinator-session",
    )
    acknowledged = []
    engine._ack_source_done = acknowledged.append
    stale = SourceReadDoneBatchAck(
        protocol_version=PROTOCOL_VERSION,
        batch_id="source-done-batch-0",
        draft_instance_id="draft-0",
        message_ids=("message-0",),
        instance_session_id="stale-draft-session",
        coordinator_session_id="coordinator-session",
    )

    assert not engine._handle_source_done_batch_ack(stale)
    assert acknowledged == []
    current = msgspec.structs.replace(
        stale, instance_session_id="draft-session"
    )
    assert engine._handle_source_done_batch_ack(current)
    assert acknowledged == ["message-0"]


def test_reused_request_id_gets_a_new_lifecycle_attempt() -> None:
    controller = object.__new__(DTTargetController)
    controller._lifecycle_attempt_by_request = {}
    controller._next_lifecycle_attempt = 0

    assert controller._ensure_lifecycle_attempt("request-0") == 0
    controller._lifecycle_attempt_by_request.pop("request-0")
    assert controller._ensure_lifecycle_attempt("request-0") == 1
    assert controller._advance_lifecycle_attempt("request-0") == 2


def test_deferred_dflash_stages_reserve_bounded_source_capacity() -> None:
    controller = object.__new__(DTTargetController)
    controller.is_dflash = True
    controller.is_feature_backed = True
    controller.defer_dflash_feature_stage = True
    controller.is_export_rank = False
    controller._feature_source_free_slots = 2
    controller._deferred_dflash_feature_stages = 0

    def deferred_result():
        return SimpleNamespace(
            dt_deferred_dflash_feature_stage=True,
            next_draft_input=DFlashDraftInputV2.create_idle_input(
                device=torch.device("cpu")
            ),
        )

    first = deferred_result()
    second = deferred_result()
    controller.stage_overlap_pivot_copy(SimpleNamespace(), first)
    assert controller.has_feature_source_capacity()
    controller.stage_overlap_pivot_copy(SimpleNamespace(), second)
    assert not controller.has_feature_source_capacity()

    with pytest.raises(RuntimeError, match="capacity twice"):
        controller.stage_overlap_pivot_copy(SimpleNamespace(), second)

    controller._stage_dflash_feature_copy(
        SimpleNamespace(reqs=[]), first, finalize_deferred=True
    )
    assert controller.has_feature_source_capacity()
    controller._stage_dflash_feature_copy(
        SimpleNamespace(reqs=[]), second, finalize_deferred=True
    )
    assert controller._deferred_dflash_feature_stages == 0


def test_dflash_prefill_reserve_keeps_one_slot_for_admitted_owner() -> None:
    controller = object.__new__(DTTargetController)
    controller.is_feature_backed = True
    controller.is_export_rank = False
    controller._deferred_dflash_feature_stages = 0
    controller._draft_admitted_attempts = {"admitted": 4}
    controller._lifecycle_attempt_by_request = {"admitted": 4}

    controller._feature_source_free_slots = 2
    assert not controller.should_defer_prefill(ready_batch_size=0, chunked_req=None)

    controller._feature_source_free_slots = 1
    assert controller.has_feature_source_capacity()
    assert controller.should_defer_prefill(ready_batch_size=0, chunked_req=None)
    assert not controller.should_defer_prefill(
        ready_batch_size=0,
        chunked_req=SimpleNamespace(rid="admitted"),
    )

    controller._feature_source_free_slots = 0
    assert not controller.has_feature_source_capacity()


def test_terminal_request_retires_late_dflash_control_failure() -> None:
    controller = object.__new__(DTTargetController)
    req = SimpleNamespace(rid="request-0")
    state = SimpleNamespace(req=req, transfer_id="transfer-0")
    controller.instance_id = "target-0"
    controller.is_dflash = True
    controller.is_feature_backed = True
    controller.is_token_only = False
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

    controller._retire_dflash_rounds_for_request(req.rid, reason="request_finished")
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

    controller.cleanup_running_after_oom(retracted_reqs=[req], aborted_reqs=[])

    assert not controller._parked
    assert not controller.pop_failures()
    assert controller._lifecycle_attempt_by_request[req.rid] == 1


def test_detach_parked_requests_clears_already_empty_batch_full_state() -> None:
    controller = object.__new__(DTTargetController)
    batch = SimpleNamespace(reqs=[], batch_is_full=True)

    controller.detach_parked_requests(batch)

    assert batch.batch_is_full is False


def test_empty_prefill_arbitration_profile_deduplicates_state() -> None:
    controller = object.__new__(DTTargetController)
    controller.is_export_rank = True
    controller._profile_active = True
    controller.empty_prefill_arbitration_profile_enabled = True
    controller._feature_source_free_slots = 2
    controller._deferred_dflash_feature_stages = 1
    controller._draft_admitted_attempts = {}
    controller._lifecycle_attempt_by_request = {}
    controller._last_empty_prefill_arbitration_profile = None
    emitted = []
    controller._profile_emit = lambda event, **fields: emitted.append(
        (event, fields)
    )

    kwargs = dict(
        decision_started_ns=1,
        prefill_decision="one_slot_headroom",
        selected="ready",
        waiting_request_ids=("request-0",),
        running_batch_is_full=False,
        empty_full_before_detach=False,
        fullness_cleared_by_detach=False,
        batch_size_before_detach=0,
        batch_size_after_detach=0,
        selected_wave_id=7,
        selected_batch_size=4,
        chunked_req=None,
    )
    controller.profile_empty_prefill_arbitration(**kwargs)
    controller.profile_empty_prefill_arbitration(**kwargs)

    assert len(emitted) == 1
    assert emitted[0][0] == "target_empty_prefill_arbitration"
    assert emitted[0][1]["effective_source_slots"] == 1
    assert emitted[0][1]["waiting_request_ids"] == ("request-0",)
    assert emitted[0][1]["selected_wave_id"] == 7

    controller.profile_empty_prefill_arbitration(
        **{**kwargs, "selected": "idle"}
    )
    assert len(emitted) == 2


@pytest.mark.parametrize("keep_live_request", (False, True))
def test_detach_parked_request_clears_stale_batch_full_state(
    keep_live_request: bool,
) -> None:
    controller = object.__new__(DTTargetController)
    parked_req = SimpleNamespace(
        rid="parked-request",
        is_retracted=False,
        finished=lambda: False,
    )
    live_req = SimpleNamespace(
        rid="live-request",
        is_retracted=False,
        finished=lambda: False,
    )
    controller._parked = {"transfer-0": SimpleNamespace(req=parked_req)}
    controller._target_inflight_counts = {}

    batch = SimpleNamespace(
        reqs=[parked_req] + ([live_req] if keep_live_request else []),
        batch_is_full=True,
        spec_info=None,
    )

    def filter_batch(*, keep_indices):
        batch.reqs = [batch.reqs[index] for index in keep_indices]

    batch.filter_batch = filter_batch

    controller.detach_parked_requests(batch)

    assert batch.reqs == ([live_req] if keep_live_request else [])
    assert batch.batch_is_full is False


def test_remote_failure_retires_one_request_lifecycle_exactly_once() -> None:
    controller = object.__new__(DTTargetController)
    req = SimpleNamespace(rid="request-0")

    def state(transfer_id, *, failure=None):
        return ParkedTargetRound(
            req=req,
            transfer_id=transfer_id,
            round_id=int(transfer_id.rsplit("-", 1)[1]),
            attempt=3,
            bonus_token=1,
            seq_len=2,
            failure=failure,
        )

    first = state("transfer-0", failure="target_round_timeout")
    second = state("transfer-1", failure="target_round_timeout")
    sibling = state("transfer-2")
    pending_task = SimpleNamespace(transfer_id=sibling.transfer_id)
    controller.instance_id = "target-0"
    controller.is_export_rank = True
    controller.is_feature_backed = True
    controller.is_token_only = False
    controller._parked = {
        first.transfer_id: first,
        second.transfer_id: second,
    }
    controller._feature_control_rounds = {sibling.transfer_id: sibling}
    controller._pending_send = deque(
        (
            TargetTaskBatch(
                protocol_version=PROTOCOL_VERSION,
                batch_id="pending-sibling",
                target_instance_id=controller.instance_id,
                entries=(pending_task,),
                sent_ns=1,
            ),
        )
    )
    controller._cancelled_transfers = set()
    controller._source_slot_by_transfer = {}
    controller._source_deadline_by_transfer = {}
    controller._lifecycle_attempt_by_request = {req.rid: 3}
    controller._next_round = {req.rid: 3}
    controller._delivery_outbox = RetryOutbox(retry_interval_ns=1)
    timeout_cancel = CancelTransfers(
        protocol_version=PROTOCOL_VERSION,
        target_instance_id=controller.instance_id,
        transfer_ids=(first.transfer_id,),
        reason="target_round_timeout",
    )
    controller._delivery_outbox.add(
        CANCEL_TRANSFERS,
        cancel_transfers_id(timeout_cancel),
        timeout_cancel,
    )
    controller._cancelled_transfers.add(first.transfer_id)
    controller._ledger = SimpleNamespace(get=lambda _transfer_id: None)
    controller._forget_request_lane = lambda _req: None
    finished = []
    released = []
    controller._finish_verified_request = finished.append
    controller._release_draft_request = released.append
    controller._flush_reliable_outbox = lambda: None

    failures = controller.pop_failures()

    assert failures == [first]
    assert not controller._parked
    assert not controller._feature_control_rounds
    assert not controller._pending_send
    assert finished == [req.rid]
    assert released == [req.rid]
    assert req.rid not in controller._lifecycle_attempt_by_request
    assert not controller.pop_failures()
    assert (
        controller._delivery_outbox.get(
            CANCEL_TRANSFERS, cancel_transfers_id(timeout_cancel)
        )
        == timeout_cancel
    )


def test_stale_attempt_failure_does_not_retire_reused_request() -> None:
    controller = object.__new__(DTTargetController)
    stale_req = SimpleNamespace(rid="request-0")
    current_req = SimpleNamespace(rid="request-0")
    stale = ParkedTargetRound(
        req=stale_req,
        transfer_id="stale-transfer",
        round_id=2,
        attempt=4,
        bonus_token=1,
        seq_len=2,
        failure="late_remote_failure",
    )
    current = ParkedTargetRound(
        req=current_req,
        transfer_id="current-transfer",
        round_id=0,
        attempt=5,
        bonus_token=1,
        seq_len=2,
    )
    controller.is_export_rank = False
    controller._parked = {
        stale.transfer_id: stale,
        current.transfer_id: current,
    }
    controller._feature_control_rounds = {}
    controller._lifecycle_attempt_by_request = {current_req.rid: 5}
    controller._next_round = {current_req.rid: 1}
    controller._forget_request_lane = lambda _req: pytest.fail(
        "stale failure retired current request lane"
    )
    controller._finish_verified_request = lambda _rid: pytest.fail(
        "stale failure retired current verification"
    )
    controller._release_draft_request = lambda _rid: pytest.fail(
        "stale failure released current Draft KV"
    )

    assert not controller.pop_failures()
    assert set(controller._parked) == {current.transfer_id}
    assert controller._parked[current.transfer_id] is current
    assert controller._lifecycle_attempt_by_request[current_req.rid] == 5


def test_stale_attempt_timeout_does_not_release_reused_draft_handle() -> None:
    controller = object.__new__(DTTargetController)
    expired = SimpleNamespace(
        task=SimpleNamespace(
            transfer_id="stale-transfer",
            request_id="request-0",
            attempt=4,
        ),
        draft_instance_id="",
    )
    finished = []
    controller.instance_id = "target-0"
    controller.is_feature_backed = False
    controller.is_token_only = True
    controller.round_timeout_ns = 120_000_000_000
    controller._draft_progress_ns = {}
    controller._ledger = SimpleNamespace(
        expired=lambda **_kwargs: [expired],
        finish=finished.append,
    )
    controller._lifecycle_attempt_by_request = {"request-0": 5}
    controller._cancelled_transfers = set()
    controller._pending_send = deque()
    controller._delivery_outbox = RetryOutbox(retry_interval_ns=1)
    controller._drop_cancelled_unsent = lambda: None
    controller._release_draft_request = lambda _rid: pytest.fail(
        "stale timeout released current Draft handle"
    )
    controller._flush_reliable_outbox = lambda: None

    assert controller._expire_local_rounds() == [
        ("failed", "stale-transfer", "target_round_timeout")
    ]
    assert finished == ["stale-transfer"]
    assert not controller._cancelled_transfers


@pytest.mark.parametrize("is_token_only", (False, True))
def test_timeout_then_pop_preserves_pending_cancel_payload(is_token_only) -> None:
    controller = object.__new__(DTTargetController)
    req = SimpleNamespace(rid="request-0")
    parked = ParkedTargetRound(
        req=req,
        transfer_id="transfer-0",
        round_id=0,
        attempt=2,
        bonus_token=1,
        seq_len=2,
    )
    expired = SimpleNamespace(
        task=SimpleNamespace(
            transfer_id=parked.transfer_id,
            request_id=req.rid,
            attempt=parked.attempt,
        ),
        draft_instance_id="",
    )
    controller.instance_id = "target-0"
    controller.is_feature_backed = not is_token_only
    controller.is_token_only = is_token_only
    controller.is_export_rank = True
    controller.round_timeout_ns = 120_000_000_000
    controller._draft_progress_ns = {}
    controller._parked = {parked.transfer_id: parked}
    controller._feature_control_rounds = {}
    controller._lifecycle_attempt_by_request = {req.rid: parked.attempt}
    controller._next_round = {req.rid: 1}
    controller._cancelled_transfers = set()
    controller._pending_send = deque()
    controller._source_slot_by_transfer = {}
    controller._source_deadline_by_transfer = {}
    controller._delivery_outbox = RetryOutbox(retry_interval_ns=1)
    controller._ledger = SimpleNamespace(
        expired=lambda **_kwargs: [expired],
        finish=lambda _transfer_id: None,
        get=lambda _transfer_id: None,
    )
    controller._quarantine_source_slot = lambda _transfer_id: None
    controller._release_draft_request = lambda _rid: None
    controller._flush_reliable_outbox = lambda: None
    controller._drop_cancelled_unsent = lambda: None
    controller._forget_request_lane = lambda _req: None
    controller._finish_verified_request = lambda _rid: None

    events = controller._expire_local_rounds()
    assert events == [("failed", parked.transfer_id, "target_round_timeout")]
    timeout_cancel = CancelTransfers(
        protocol_version=PROTOCOL_VERSION,
        target_instance_id=controller.instance_id,
        transfer_ids=(parked.transfer_id,),
        reason="target_round_timeout",
    )
    timeout_cancel_id = cancel_transfers_id(timeout_cancel)
    assert (
        controller._delivery_outbox.get(CANCEL_TRANSFERS, timeout_cancel_id)
        == timeout_cancel
    )
    if is_token_only:
        assert not controller._cancelled_transfers

    # poll_results applies the returned event before Scheduler calls pop_failures.
    parked.failure = events[0][2]
    assert controller.pop_failures() == [parked]
    assert (
        controller._delivery_outbox.get(CANCEL_TRANSFERS, timeout_cancel_id)
        == timeout_cancel
    )
    if is_token_only:
        assert not controller._cancelled_transfers


def test_stranded_parked_round_times_out_without_ledger_entry() -> None:
    # A round whose Coordinator copy is still queued locally never entered the
    # ledger; the local round deadline must still fail it terminally.
    controller = object.__new__(DTTargetController)
    req = SimpleNamespace(rid="request-0")
    now_ns = time.monotonic_ns()
    stranded = ParkedTargetRound(
        req=req,
        transfer_id="target-0:request-0:4:0",
        round_id=4,
        attempt=0,
        bonus_token=1,
        seq_len=2,
        deadline_ns=now_ns - 1,
    )
    waiting = ParkedTargetRound(
        req=req,
        transfer_id="target-0:request-0:5:0",
        round_id=5,
        attempt=0,
        bonus_token=1,
        seq_len=2,
        deadline_ns=now_ns + 600_000_000_000,
    )
    unfenced = ParkedTargetRound(
        req=req,
        transfer_id="target-0:request-0:6:0",
        round_id=6,
        attempt=0,
        bonus_token=1,
        seq_len=2,
    )
    controller.instance_id = "target-0"
    controller.is_feature_backed = True
    controller.is_token_only = False
    controller.is_export_rank = True
    controller.round_timeout_ns = 120_000_000_000
    controller._draft_progress_ns = {}
    controller._parked = {
        state.transfer_id: state for state in (stranded, waiting, unfenced)
    }
    controller._feature_control_rounds = {}
    controller._lifecycle_attempt_by_request = {req.rid: 0}
    controller._cancelled_transfers = set()
    controller._pending_send = deque()
    controller._delivery_outbox = RetryOutbox(retry_interval_ns=1)
    controller._ledger = SimpleNamespace(
        expired=lambda **_kwargs: (),
        finish=lambda _transfer_id: pytest.fail(
            "stranded round was never in the ledger"
        ),
        get=lambda _transfer_id: None,
    )
    released = []
    controller._release_draft_request = released.append
    controller._flush_reliable_outbox = lambda: None
    dropped = []
    controller._drop_cancelled_unsent = lambda: dropped.append(
        set(controller._cancelled_transfers)
    )

    events = controller._expire_local_rounds()

    assert events == [("failed", stranded.transfer_id, "target_round_timeout")]
    assert controller._cancelled_transfers == {stranded.transfer_id}
    assert dropped == [{stranded.transfer_id}]
    assert released == [req.rid]
    cancellation = CancelTransfers(
        protocol_version=PROTOCOL_VERSION,
        target_instance_id=controller.instance_id,
        transfer_ids=(stranded.transfer_id,),
        reason="target_round_timeout",
    )
    assert (
        controller._delivery_outbox.get(
            CANCEL_TRANSFERS, cancel_transfers_id(cancellation)
        )
        == cancellation
    )
    assert waiting.failure is None
    assert unfenced.failure is None


def test_unconfirmed_direct_expiry_publishes_failure() -> None:
    # Abandoning a never-confirmed ring task owes the owning Target a
    # terminal failure instead of a silent drop.
    engine = object.__new__(FlashMTPDraftEngine)
    task = SimpleNamespace(
        transfer_id="target-0:request-0:3:1",
        request_id="request-0",
        round_id=3,
        attempt=1,
        target_instance_id="target-0",
    )
    engine.instance_id = "draft-0"
    engine.EntryStatus = EntryStatus
    engine._unconfirmed_direct = {task.transfer_id: 0}
    engine._held_direct_results = {task.transfer_id: object()}
    engine._held_direct_source_done = {task.transfer_id: [object()]}
    engine._pending_ids = {task.transfer_id}
    engine._ring_pending_ids = {task.transfer_id}
    engine._direct_tasks_by_transfer = {task.transfer_id: task}
    engine._ring_record_positions = {task.transfer_id: (object(), 0)}
    engine._ring_retired = {}
    engine._cancelled_ids = set()
    engine._result_sequence = 0
    engine._delivery_outbox = RetryOutbox(retry_interval_ns=1)
    engine._flush_reliable_outbox = lambda: None

    engine._expire_unconfirmed_direct()

    assert not engine._unconfirmed_direct
    assert not engine._held_direct_results
    assert not engine._held_direct_source_done
    assert not engine._pending_ids
    assert not engine._ring_pending_ids
    assert not engine._direct_tasks_by_transfer
    assert not engine._ring_record_positions
    # The drop reported the transfer: a late Coordinator copy must not
    # publish a duplicate failure.
    assert engine._ring_retired[task.transfer_id] is False
    pending = engine._delivery_outbox.pending()
    assert len(pending) == 1
    operation, _batch_id, message = pending[0]
    assert operation == "result"
    assert len(message.entries) == 1
    entry = message.entries[0]
    assert entry.transfer_id == task.transfer_id
    assert entry.request_id == "request-0"
    assert entry.round_id == 3
    assert entry.attempt == 1
    assert entry.target_instance_id == "target-0"
    assert entry.status == EntryStatus.FINAL_ERROR
    assert entry.error_code == "TimeoutError"


def test_dflash_overlap_routes_staged_feature_boundaries_in_fifo_order() -> None:
    controller = object.__new__(DTTargetController)
    controller.is_token_only = False
    controller.is_dflash = True
    controller.is_feature_backed = True
    controller.instance_id = "target-0"
    controller.profile_id = "dflash-profile"
    controller.block_size = 7
    controller.round_timeout_ns = 1_000_000
    controller._draft_committed_len = {"request-0": 20}
    controller._draft_dispatched_len = {}
    controller._draft_kv_handle_by_request = {"request-0": "kv-0"}
    controller.pipeline_lanes = 3
    controller.pipeline_lane_weights = (1, 1, 1)
    controller._lane_by_request = {}
    controller._lane_target_sizes = [0, 0, 0]
    controller._lane_assignment_cursor = 0
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
        previous_transfer_id=None,
        feature_descriptor=descriptor("transfer-0"),
        feature_token_ids=tuple(range(10)),
        committed_token_ids=tuple(range(10)),
        feature_token_offset=0,
        requires_proposal=False,
    )
    second = controller._make_task(
        transfer_id="transfer-1",
        req=req,
        round_id=1,
        attempt=0,
        bonus_token=12,
        previous_transfer_id="transfer-0",
        feature_descriptor=descriptor("transfer-1"),
        feature_token_ids=tuple(range(10, 20)),
        committed_token_ids=tuple(range(10, 20)),
        feature_token_offset=10,
    )

    assert first.committed_kv_len == 10
    assert first.pipeline_lane_id == 0
    assert second.pipeline_lane_id == first.pipeline_lane_id
    assert second.feature_token_offset == first.committed_kv_len
    assert second.committed_kv_len == 20
    assert controller._draft_dispatched_len[req.rid] == 20


def test_weighted_lane_admission_preserves_immutable_request_lane() -> None:
    controller = object.__new__(DTTargetController)
    controller.pipeline_lanes = 3
    controller.pipeline_lane_weights = _parse_pipeline_lane_weights("1,2,1", 3)
    controller._lane_by_request = {}
    controller._lane_target_sizes = [0, 0, 0]
    controller._lane_assignment_cursor = 0
    requests = [SimpleNamespace(rid=f"request-{index}") for index in range(256)]

    assigned = [controller._lane_for_request(req) for req in requests]

    assert [assigned.count(lane_id) for lane_id in range(3)] == [64, 128, 64]
    assert controller._lane_for_request(requests[0]) == assigned[0]


@pytest.mark.parametrize("value", ("1,2", "1,0,1", "one,2,1"))
def test_pipeline_lane_weights_reject_invalid_shapes(value) -> None:
    with pytest.raises(ValueError, match="pipeline_lane_weights"):
        _parse_pipeline_lane_weights(value, 3)


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


def test_overtaking_release_is_acked_after_feature_completion_retires_kv() -> None:
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
    state = SimpleNamespace(
        request_id=release.request_id,
        draft_kv_handle=release.draft_kv_handle,
        req=SimpleNamespace(kv=None, req_pool_idx=None),
    )
    task = SimpleNamespace(draft_kv_handle=release.draft_kv_handle)
    engine._persistent_requests = {release.draft_kv_handle: state}
    engine._pending = deque()
    engine._pending_feature_batches = deque()
    engine._pending_dflash_completions = deque()
    engine._pending_eagle3_completions = deque(
        (SimpleNamespace(pending=(SimpleNamespace(task=task),)),)
    )
    engine._quarantined_feature_draft_handles = set()
    engine._migration_source_handles = {}
    engine._deferred_persistent_releases = {}
    engine._pending_request_release_acks = {release_id: release}
    engine._delivery_outbox = RetryOutbox(retry_interval_ns=1)
    engine._flush_reliable_outbox = lambda: None

    assert not engine._release_persistent_request(
        release.draft_kv_handle, request_id=release.request_id
    )
    assert release.draft_kv_handle in engine._deferred_persistent_releases

    engine._pending_eagle3_completions.clear()
    assert engine._release_persistent_request(
        release.draft_kv_handle, request_id=release.request_id
    )
    acknowledgement = engine._delivery_outbox.get(RELEASE_DRAFT_REQUEST, release_id)
    assert acknowledgement is not None
    assert release.draft_kv_handle not in engine._persistent_requests
    assert release.draft_kv_handle not in engine._deferred_persistent_releases


def test_chunked_feature_dispatch_only_blocks_the_prior_request() -> None:
    controller = object.__new__(DTTargetController)
    blocked_1 = SimpleNamespace(
        request_id="request-0",
        transfer_id="transfer-1",
        round_id=1,
        requires_proposal=True,
    )
    ready_1 = SimpleNamespace(
        request_id="request-1",
        transfer_id="transfer-2",
        round_id=0,
        requires_proposal=True,
    )
    blocked_2 = SimpleNamespace(
        request_id="request-0",
        transfer_id="transfer-3",
        round_id=2,
        requires_proposal=True,
    )
    ready_2 = SimpleNamespace(
        request_id="request-2",
        transfer_id="transfer-4",
        round_id=0,
        requires_proposal=True,
    )
    batches = (
        TargetTaskBatch(
            protocol_version=PROTOCOL_VERSION,
            batch_id="batch-1",
            target_instance_id="target-0",
            entries=(blocked_1, ready_1),
            sent_ns=1,
        ),
        TargetTaskBatch(
            protocol_version=PROTOCOL_VERSION,
            batch_id="batch-2",
            target_instance_id="target-0",
            entries=(blocked_2, ready_2),
            sent_ns=2,
        ),
    )
    controller.instance_id = "target-0"
    controller._batch_sequence = 2
    controller._pending_send = deque(batches)
    controller._last_pending_send_block = None
    controller.whole_batch_task_fifo_ablation = False
    controller._flush_reliable_outbox = lambda: None
    controller._profile_active = False
    controller._profile_emit = lambda *_args, **_kwargs: None
    active = {"request-0": "transfer-0"}
    active_state = SimpleNamespace(
        task=SimpleNamespace(round_id=0, requires_proposal=False),
        phase=SimpleNamespace(value="wait_draft"),
    )
    begun = []
    submitted = []

    def begin(batch) -> None:
        begun.append(batch)
        for task in batch.entries:
            active[task.request_id] = task.transfer_id

    controller._ledger = SimpleNamespace(
        active_transfer_id=lambda request_id: active.get(request_id),
        get=lambda transfer_id: (active_state if transfer_id == "transfer-0" else None),
        begin=begin,
    )
    controller._client = SimpleNamespace(
        submit=lambda value: submitted.append(value) or True
    )
    controller._delivery_outbox = SimpleNamespace(
        add=lambda *_args: None,
        mark_sent=lambda *_args, **_kwargs: None,
    )

    controller._flush_pending_send()
    assert [task.transfer_id for task in submitted[0].entries] == [
        "transfer-2",
        "transfer-4",
    ]
    assert [
        [task.transfer_id for task in batch.entries]
        for batch in controller._pending_send
    ] == [["transfer-1"], ["transfer-3"]]

    active.pop("request-0")
    controller._flush_pending_send()
    assert [task.transfer_id for task in submitted[1].entries] == ["transfer-1"]
    assert [task.transfer_id for task in controller._pending_send[0].entries] == [
        "transfer-3"
    ]

    active.pop("request-0")
    controller._flush_pending_send()
    assert [task.transfer_id for task in submitted[2].entries] == ["transfer-3"]
    assert begun == submitted
    assert not controller._pending_send


def test_whole_batch_fifo_ablation_blocks_independent_request_progress() -> None:
    controller = object.__new__(DTTargetController)
    blocked = SimpleNamespace(
        request_id="request-0",
        transfer_id="transfer-1",
        round_id=1,
        requires_proposal=True,
    )
    independent = SimpleNamespace(
        request_id="request-1",
        transfer_id="transfer-2",
        round_id=0,
        requires_proposal=True,
    )
    batch = SimpleNamespace(
        protocol_version=PROTOCOL_VERSION,
        batch_id="batch-1",
        target_instance_id="target-0",
        entries=(blocked, independent),
        sent_ns=1,
    )
    active_state = SimpleNamespace(
        task=SimpleNamespace(round_id=0, requires_proposal=False),
        phase=SimpleNamespace(value="wait_draft"),
    )
    controller.instance_id = "target-0"
    controller._batch_sequence = 1
    controller._pending_send = deque((batch,))
    controller._last_pending_send_block = None
    controller.whole_batch_task_fifo_ablation = True
    controller._flush_reliable_outbox = lambda: None
    controller._profile_emit = lambda *_args, **_kwargs: None
    controller._ledger = SimpleNamespace(
        active_transfer_id=lambda request_id: (
            "transfer-0" if request_id == "request-0" else None
        ),
        get=lambda transfer_id: active_state,
    )
    submitted = []
    controller._client = SimpleNamespace(
        submit=lambda value: submitted.append(value) or True
    )

    controller._flush_pending_send()

    assert not submitted
    assert controller._pending_send[0] is batch


def test_dflash_migration_waits_for_every_feature_consumer_stage() -> None:
    engine = object.__new__(FlashMTPDraftEngine)
    handle = "draft-kv-0"
    queued_task = SimpleNamespace(draft_kv_handle=handle)
    feature_item = SimpleNamespace(task=queued_task)
    engine._pending = deque((queued_task,))
    engine._pending_feature_batches = deque()
    engine._pending_dflash_completions = deque()
    engine._pending_eagle3_completions = deque()
    assert engine._dflash_handle_has_active_consumer(handle)

    engine._pending.clear()
    engine._pending_feature_batches.append(SimpleNamespace(pending=(feature_item,)))
    assert engine._dflash_handle_has_active_consumer(handle)

    engine._pending_feature_batches.clear()
    engine._pending_dflash_completions.append(SimpleNamespace(pending=(feature_item,)))
    assert engine._dflash_handle_has_active_consumer(handle)

    engine._pending_dflash_completions.clear()
    engine._pending_eagle3_completions.append(
        SimpleNamespace(pending=(feature_item,))
    )
    assert engine._dflash_handle_has_active_consumer(handle)

    engine._pending_eagle3_completions.clear()
    assert not engine._dflash_handle_has_active_consumer(handle)


def test_dflash_pipeline_reserves_one_result_slot_per_submitted_batch() -> None:
    def task(transfer_id: str, *, requires_proposal: bool):
        return SimpleNamespace(
            transfer_id=transfer_id,
            draft_kv_handle=f"kv-{transfer_id}",
            target_instance_id="target-0",
            created_ns=1,
            deadline_ns=2**63 - 1,
            feature_token_count=1,
            source_read_migration_id=None,
            requires_proposal=requires_proposal,
        )

    engine = object.__new__(FlashMTPDraftEngine)
    engine._is_dflash = True
    engine._is_eagle3 = False
    engine._is_feature_backed = True
    engine._persistent = True
    engine._free_result_slots = deque((0, 1))
    engine._free_feature_staging_slots = deque((0, 1))
    reserved_feature = PendingDraftTask(task("feature", requires_proposal=True))
    engine._pending_feature_batches = deque(
        (SimpleNamespace(pending=(reserved_feature,)),)
    )
    engine._pending_dflash_completions = deque(
        (SimpleNamespace(proposal_tasks=(object(),)),)
    )
    engine.batch_ready_threshold = 1
    engine.batch_wait_s = 0
    engine.max_batch_size = 8
    engine.feature_buffer_max_tokens = 8
    engine._cancelled_ids = set()
    engine._received_ns_by_transfer = {}
    leading_materialization = task("materialize", requires_proposal=False)
    waiting_proposal = task("proposal", requires_proposal=True)
    engine._pending = deque((leading_materialization, waiting_proposal))
    engine._pending_ids = {item.transfer_id for item in engine._pending}

    # Both free pool slots are reserved by already submitted proposal batches.
    # Materialization-only work may progress, but the next proposal must wait.
    assert engine._available_dflash_result_slots() == 0
    first = engine._take_batch()
    assert [item.task.transfer_id for item in first] == ["materialize"]
    assert not engine._take_batch()
    assert [item.transfer_id for item in engine._pending] == ["proposal"]

    # Completing one submitted batch releases exactly one logical admission;
    # every proposal row coalesced into the next batch shares that result slot.
    engine._pending_dflash_completions.clear()
    second_proposal = task("proposal-2", requires_proposal=True)
    engine._pending.append(second_proposal)
    engine._pending_ids.add(second_proposal.transfer_id)
    assert engine._available_dflash_result_slots() == 1
    second = engine._take_batch()
    assert [item.task.transfer_id for item in second] == ["proposal", "proposal-2"]


def test_flashmtp_pending_completion_reserves_result_slot() -> None:
    engine = object.__new__(FlashMTPDraftEngine)
    engine._is_feature_backed = False
    engine._persistent = False
    engine._free_result_slots = deque((0, 1))
    engine._pending_feature_batches = deque()
    engine._pending_flashmtp_completions = deque((object(),))
    engine._pending_dflash_completions = deque()

    assert engine._available_dflash_result_slots() == 1


def test_direct_candidate_cancel_publishes_quiescent_host_completion() -> None:
    class ReadyEvent:
        @staticmethod
        def query() -> bool:
            return True

    class CandidateTransport:
        def __init__(self) -> None:
            self.tasks = None
            self.completions = []

        def publish_cancelled(self, tasks):
            self.tasks = list(tasks)
            return SimpleNamespace(
                event=ReadyEvent(),
                transfer_ids=tuple(task.transfer_id for task in tasks),
                submitted_ns=11,
                candidate_bytes=0,
            )

        def publish_completion(self, tasks, *, cancelled):
            self.completions.append((list(tasks), cancelled))

    task = SimpleNamespace(
        transfer_id="direct-cancel",
        request_id="request-0",
        round_id=3,
        attempt=1,
        target_instance_id="target-0",
        candidate_destination=object(),
    )
    sent = []
    capacity_updates = []
    engine = object.__new__(FlashMTPDraftEngine)
    engine.EntryStatus = EntryStatus
    engine.instance_id = "draft-0"
    engine.candidate_transport = "cuda-ipc"
    engine._candidate_push_transport = CandidateTransport()
    engine._pending_candidate_pushes = deque()
    # The cancel may land after INPUT_READY selection removed the task from
    # its queue but before the compute completion owner was linked.
    engine._pending = deque()
    engine._direct_tasks_by_transfer = {task.transfer_id: task}
    engine._pending_feature_batches = deque()
    engine._input_ready_features = deque()
    engine._pending_flashmtp_completions = deque()
    engine._pending_dflash_completions = deque()
    engine._pending_eagle3_completions = deque()
    engine._cancelled_ids = {task.transfer_id}
    engine._pending_ids = {task.transfer_id}
    engine._received_ns_by_transfer = {task.transfer_id: 1}
    engine._fatal_gpu_quarantine_reason = None
    engine._result_sequence = 0
    engine._free_result_slots = deque((7,))
    engine._profile_active = False
    engine._send_result = sent.append
    engine._publish_capacity = lambda: capacity_updates.append(True)

    engine._publish_candidate_cancellations((task.transfer_id,))
    assert engine._candidate_push_transport.tasks == [task]
    assert len(engine._pending_candidate_pushes) == 1
    assert engine._pending_candidate_pushes[0].cancelled
    assert engine._pending_candidate_pushes[0].result_slot is None

    engine._poll_candidate_pushes()
    assert engine._candidate_push_transport.completions == [([task], True)]

    assert not engine._pending_candidate_pushes
    assert list(engine._free_result_slots) == [7]
    assert capacity_updates == [True]
    assert task.transfer_id not in engine._cancelled_ids
    assert task.transfer_id not in engine._direct_tasks_by_transfer
    # Direct publication is only the data-plane half of return ownership.
    # Coordinator ResultBatchAck retires the accepted transfer identity.
    assert engine._pending_ids == {task.transfer_id}
    assert len(sent) == 1
    assert sent[0].entries[0].status == EntryStatus.CANCELLED
    assert sent[0].entries[0].source_read_quiescent
    assert sent[0].buffer is None


def test_direct_candidate_completion_timestamp_precedes_notice_publish(
    monkeypatch,
) -> None:
    class ReadyEvent:
        @staticmethod
        def query() -> bool:
            return True

    publish_observed = []

    class CandidateTransport:
        @staticmethod
        def publish_completion(_tasks, *, cancelled):
            assert not cancelled
            publish_observed.append(dt_draft_engine_module.time.monotonic_ns())

    task = SimpleNamespace(
        transfer_id="direct-profile-order",
        request_id="request-0",
        round_id=3,
        attempt=1,
        target_instance_id="target-0",
    )
    emitted = []
    sent = []
    clock = iter((100, 110, 120, 130))
    monkeypatch.setattr(
        dt_draft_engine_module.time, "monotonic_ns", lambda: next(clock)
    )
    monkeypatch.setattr(
        dt_draft_engine_module,
        "profile_emit",
        lambda event, **fields: emitted.append((event, fields)),
    )

    engine = object.__new__(FlashMTPDraftEngine)
    engine.EntryStatus = EntryStatus
    engine.instance_id = "draft-0"
    engine._candidate_push_transport = CandidateTransport()
    engine._pending_candidate_pushes = deque(
        (
            SimpleNamespace(
                push=SimpleNamespace(
                    event=ReadyEvent(),
                    start_event=None,
                    submitted_ns=50,
                    candidate_bytes=64,
                ),
                tasks=[task],
                result_slot=None,
                cancelled=False,
            ),
        )
    )
    engine._pending = deque()
    engine._pending_feature_batches = deque()
    engine._input_ready_features = deque()
    engine._pending_flashmtp_completions = deque()
    engine._pending_dflash_completions = deque()
    engine._pending_eagle3_completions = deque()
    engine._cancelled_ids = set()
    engine._pending_ids = {task.transfer_id}
    engine._received_ns_by_transfer = {task.transfer_id: 1}
    engine._direct_tasks_by_transfer = {task.transfer_id: task}
    engine._fatal_gpu_quarantine_reason = None
    engine._result_sequence = 0
    engine._free_result_slots = deque()
    engine._profile_active = True
    engine._send_result = sent.append
    engine._publish_capacity = lambda: None

    engine._poll_candidate_pushes()

    observed = dict(emitted)["draft_candidate_producer_completion_observed"]
    completed = dict(emitted)["draft_candidate_push_complete"]
    assert observed["completion_observed_ns"] == 100
    assert publish_observed == [110]
    assert observed["notice_publish_returned_ns"] == 120
    assert completed["completed_ns"] == 100
    assert completed["completion_ns"] == 50
    assert completed["coordinator_result_sent_ns"] == 130
    assert sent[0].sent_ns == 130


def test_target_cuda_observation_emits_one_ordered_wave_without_sync() -> None:
    class FakeEvent:
        def __init__(self, *, ready: bool, elapsed_ms: float) -> None:
            self.ready = ready
            self.elapsed_ms = elapsed_ms
            self.query_calls = 0

        def query(self):
            self.query_calls += 1
            return self.ready

        def elapsed_time(self, _other):
            return self.elapsed_ms

    start = FakeEvent(ready=True, elapsed_ms=0.75)
    end = FakeEvent(ready=True, elapsed_ms=0.0)
    emitted = []
    controller = object.__new__(DTTargetController)
    controller.is_export_rank = True
    controller._profile_active = True
    controller._last_target_verify_end_event = None
    controller._profile_emit = lambda event, **fields: emitted.append(
        (event, fields)
    )
    controller._pending_target_gpu_observations = deque(
        (
            dt_target_module._PendingTargetGpuObservation(
                wave_id=9,
                transfer_ids=("transfer-a",),
                ready_ns=10,
                selection_started_ns=11,
                selection_finished_ns=12,
                submission_started_ns=13,
                submission_returned_ns=14,
                start_event=start,
                end_event=end,
            ),
        )
    )

    controller._poll_target_gpu_observations()

    assert not controller._pending_target_gpu_observations
    assert start.query_calls == 1
    assert end.query_calls == 1
    assert [name for name, _ in emitted] == ["target_verify_cuda_observed"]
    fields = emitted[0][1]
    assert fields["wave_id"] == 9
    assert fields["transfer_ids"] == ("transfer-a",)
    assert fields["cuda_total_ns"] == 750_000
    assert fields["gpu_start_observed_ns"] <= fields["gpu_end_observed_ns"]


def test_direct_tombstone_keeps_cancel_marker_until_feature_pull_retires() -> None:
    class ReadyEvent:
        @staticmethod
        def query() -> bool:
            return True

    class CandidateTransport:
        @staticmethod
        def publish_completion(_tasks, *, cancelled):
            assert cancelled

    task = SimpleNamespace(
        transfer_id="cancel-during-feature-pull",
        request_id="request-0",
        round_id=0,
        attempt=1,
        target_instance_id="target-0",
        candidate_destination=object(),
    )
    item = SimpleNamespace(task=task)
    engine = object.__new__(FlashMTPDraftEngine)
    engine.EntryStatus = EntryStatus
    engine.instance_id = "draft-0"
    engine._candidate_push_transport = CandidateTransport()
    engine._pending_candidate_pushes = deque(
        (
            SimpleNamespace(
                push=SimpleNamespace(event=ReadyEvent(), submitted_ns=1),
                tasks=[task],
                result_slot=None,
                cancelled=True,
            ),
        )
    )
    engine._pending = deque()
    engine._pending_feature_batches = deque(
        (SimpleNamespace(pending=[item]),)
    )
    engine._input_ready_features = deque()
    engine._pending_flashmtp_completions = deque()
    engine._pending_dflash_completions = deque()
    engine._pending_eagle3_completions = deque()
    engine._cancelled_ids = {task.transfer_id}
    engine._pending_ids = {task.transfer_id}
    engine._received_ns_by_transfer = {task.transfer_id: 1}
    engine._direct_tasks_by_transfer = {task.transfer_id: task}
    engine._fatal_gpu_quarantine_reason = None
    engine._result_sequence = 0
    engine._free_result_slots = deque()
    engine._profile_active = False
    engine._send_result = lambda _message: None
    engine._publish_capacity = lambda: None

    engine._poll_candidate_pushes()
    assert task.transfer_id in engine._cancelled_ids
    assert task.transfer_id in engine._pending_ids

    engine._pending_feature_batches.clear()
    engine._retire_cancelled_transfer_identity(task)
    assert task.transfer_id not in engine._cancelled_ids
    # Direct return responsibility is still retired only by ResultBatchAck.
    assert task.transfer_id in engine._pending_ids


def test_result_ack_retires_held_transfer_identity() -> None:
    transfer_id = "held-until-result-ack"
    batch_id = "draft-0:direct-result:9"
    result = SimpleNamespace(
        entries=(SimpleNamespace(transfer_id=transfer_id),)
    )
    outbox = RetryOutbox(retry_interval_ns=1)
    outbox.add("result", batch_id, result)
    capacity_updates = []

    engine = object.__new__(FlashMTPDraftEngine)
    engine._delivery_outbox = outbox
    engine._pending_ids = {transfer_id}
    engine._published_by_batch = {}
    engine._client = SimpleNamespace(
        recv_nowait=lambda: (
            ResultBatchAck(
                protocol_version=PROTOCOL_VERSION,
                batch_id=batch_id,
                accepted_transfer_ids=(transfer_id,),
            ),
        )
    )
    engine._flush_source_done = lambda: None
    engine._poll_migration_transfers = lambda: False
    engine._poll_deferred_migration_gpu_controls = lambda: False
    engine._drop_cancelled_pending = lambda: None
    engine._publish_capacity = lambda: capacity_updates.append(True)

    engine._poll_messages()

    assert engine._pending_ids == set()
    assert outbox.get("result", batch_id) is None
    assert capacity_updates == [True]


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
    _init_p2p_transport_fence_state(transport)

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
    _init_p2p_transport_fence_state(transport)

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
    _init_p2p_transport_fence_state(transport)

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


def test_engine_fatal_quarantine_fences_queued_migration_executor(
    monkeypatch,
) -> None:
    """An engine-fatal latch reaches a queued migration before CUDA access."""

    blocker_started = Event()
    release_blocker = Event()
    operations = []

    class RetainedMappingCache:
        def __init__(self) -> None:
            self.retained_mapping = object()
            self.closed = False

        def map(self, **_kwargs):
            operations.append("map")
            return 17, False

        def close(self) -> None:
            self.closed = True

    class UnusedRuntime:
        @staticmethod
        def copy_batch_async(_segments, _stream) -> None:
            operations.append("copy")

    def hold_executor() -> None:
        blocker_started.set()
        assert release_blocker.wait(timeout=2)

    monkeypatch.setattr(
        dt_kv_p2p.torch.cuda,
        "set_device",
        lambda _device: operations.append("set_device"),
    )
    transport = object.__new__(DraftKvP2pTransport)
    _init_p2p_transport_fence_state(transport)
    transport.device = torch.device("cuda:0")
    transport.device_index = 0
    transport._runtime = UnusedRuntime()
    transport._mapping_cache = RetainedMappingCache()
    transport._copy_stream = object()
    transport._buffers = ()
    transport._executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="test-dt-kv-p2p"
    )
    blocker = transport._executor.submit(hold_executor)
    assert blocker_started.wait(timeout=2)

    source_pool = SimpleNamespace(
        instance_session_id="source-session",
        pool_generation="source-generation",
        allocations=(SimpleNamespace(ipc_handle=b"source-allocation"),),
        buffers=(),
    )
    transfer = SimpleNamespace(
        page_indices=(),
        kv_bytes=0,
        producer_event_handle=b"source-event",
    )
    reservation = SimpleNamespace(page_indices=())
    queued = transport.submit(
        source_draft_instance_id="draft-source",
        source_pool=source_pool,
        transfer=transfer,
        reservation=reservation,
    )
    prepared = SimpleNamespace(
        reservation=reservation,
        source_pool=source_pool,
        transfer_future=queued,
    )
    engine = object.__new__(FlashMTPDraftEngine)
    engine._stop = False
    engine._fatal_gpu_quarantine_reason = None
    engine._feature_pull_transport = None
    engine._p2p_transport = transport
    engine._prepared_migrations = {"migration-queued": prepared}

    try:
        engine._enter_fatal_gpu_quarantine("injected unquiesced proposal failure")
        release_blocker.set()
        blocker.result(timeout=2)
        with pytest.raises(RuntimeError, match="permanent.*quarantine"):
            queued.result(timeout=2)

        assert engine._stop
        assert engine._prepared_migrations["migration-queued"] is prepared
        assert prepared.reservation is reservation
        assert transport._mapping_cache.retained_mapping is not None
        assert not transport._mapping_cache.closed
        assert operations == []
        with pytest.raises(RuntimeError, match="process teardown"):
            transport.close()
        assert not transport._mapping_cache.closed
    finally:
        release_blocker.set()
        transport._executor.shutdown(wait=True, cancel_futures=True)


def test_unquiesced_migration_fences_second_queued_transfer(monkeypatch) -> None:
    """A first callback's failed fence prevents a queued generation remap."""

    first_copy_started = Event()
    release_first_copy = Event()
    operations = []

    class RetainedMappingCache:
        def __init__(self) -> None:
            self.retained_mapping = object()
            self.map_calls = []

        def map(self, *, pool_generation, **_kwargs):
            self.map_calls.append(pool_generation)
            operations.append(("map", pool_generation))
            return 17, False

        def close(self) -> None:
            raise AssertionError("unquiesced mappings must remain retained")

    class FailingAfterEnqueueRuntime:
        @staticmethod
        def copy_batch_async(_segments, _stream) -> None:
            operations.append(("copy", "first"))
            first_copy_started.set()
            assert release_first_copy.wait(timeout=2)
            raise RuntimeError("injected first migration submit failure")

    class FailingCopyStream:
        cuda_stream = 17

        def wait_event(self, _event) -> None:
            operations.append(("wait", "first"))

        def synchronize(self) -> None:
            operations.append(("fence_failed", "first"))
            raise RuntimeError("injected migration fence failure")

    class FakeCudaEvent:
        @classmethod
        def from_ipc_handle(cls, _device_index, handle):
            operations.append(("event", handle))
            return cls()

        def __init__(self, *, enable_timing=False) -> None:
            self.enable_timing = enable_timing

        def record(self, _stream) -> None:
            operations.append(("record", self.enable_timing))

    monkeypatch.setattr(
        dt_kv_p2p.torch.cuda,
        "set_device",
        lambda _device: operations.append(("set_device", "first")),
    )
    monkeypatch.setattr(dt_kv_p2p.torch.cuda, "Event", FakeCudaEvent)
    monkeypatch.setattr(dt_kv_p2p.torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(dt_kv_p2p, "build_copy_segments", lambda **_kwargs: ())

    transport = object.__new__(DraftKvP2pTransport)
    _init_p2p_transport_fence_state(transport)
    transport.device = torch.device("cuda:0")
    transport.device_index = 0
    transport._runtime = FailingAfterEnqueueRuntime()
    transport._mapping_cache = RetainedMappingCache()
    transport._copy_stream = FailingCopyStream()
    transport._buffers = ()
    transport._executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="test-dt-kv-p2p"
    )

    def source_pool(generation: str):
        return SimpleNamespace(
            instance_session_id="source-session",
            pool_generation=generation,
            allocations=(SimpleNamespace(ipc_handle=generation.encode()),),
            buffers=(),
        )

    def transfer(generation: str):
        return SimpleNamespace(
            page_indices=(),
            kv_bytes=0,
            producer_event_handle=f"event-{generation}".encode(),
        )

    first_reservation = SimpleNamespace(page_indices=())
    second_reservation = SimpleNamespace(page_indices=())
    first_source_pool = source_pool("generation-1")
    second_source_pool = source_pool("generation-2")
    first_transfer = transfer("generation-1")
    second_transfer = transfer("generation-2")
    first = transport.submit(
        source_draft_instance_id="draft-source",
        source_pool=first_source_pool,
        transfer=first_transfer,
        reservation=first_reservation,
    )
    assert first_copy_started.wait(timeout=2)
    second = transport.submit(
        source_draft_instance_id="draft-source",
        source_pool=second_source_pool,
        transfer=second_transfer,
        reservation=second_reservation,
    )
    first_owner = SimpleNamespace(
        reservation=first_reservation,
        source_pool=first_source_pool,
        transfer_future=first,
    )
    second_owner = SimpleNamespace(
        reservation=second_reservation,
        source_pool=second_source_pool,
        transfer_future=second,
    )
    engine = object.__new__(FlashMTPDraftEngine)
    engine._prepared_migrations = {
        "migration-first": first_owner,
        "migration-second": second_owner,
    }

    try:
        release_first_copy.set()
        with pytest.raises(TransferQuiescenceError, match="must remain quarantined"):
            first.result(timeout=2)
        with pytest.raises(RuntimeError, match="unquiesced GPU-copy failure"):
            second.result(timeout=2)

        assert transport._mapping_cache.map_calls == ["generation-1"]
        assert sum(item[0] == "set_device" for item in operations) == 1
        assert sum(item[0] == "copy" for item in operations) == 1
        assert transport._mapping_cache.retained_mapping is not None
        assert engine._prepared_migrations["migration-first"] is first_owner
        assert engine._prepared_migrations["migration-second"] is second_owner
        assert first_owner.reservation is first_reservation
        assert second_owner.reservation is second_reservation
        transport.close()
        assert transport._closed
        assert transport._mapping_cache.retained_mapping is not None
    finally:
        release_first_copy.set()
        transport._executor.shutdown(wait=True, cancel_futures=True)


@pytest.mark.parametrize("method", ("dflash", "eagle3"))
def test_shutdown_completion_fence_failure_quarantines_queued_migration(
    monkeypatch,
    method: str,
) -> None:
    """A shutdown fence failure latches quarantine before a queued KV pull."""

    worker_started = Event()
    release_worker = Event()
    operations = []

    class RetainedMappingCache:
        def __init__(self) -> None:
            self.retained_mapping = object()
            self.maps = []
            self.closed = False

        def map(self, **kwargs):
            self.maps.append(kwargs)
            operations.append("map")
            return 17, False

        def close(self) -> None:
            self.closed = True

    class UnusedRuntime:
        @staticmethod
        def copy_batch_async(_segments, _stream) -> None:
            operations.append("copy")

    class FailingCompletionEvent:
        def __init__(self) -> None:
            self.query_calls = 0
            self.synchronize_calls = 0

        def query(self) -> bool:
            self.query_calls += 1
            raise AssertionError("shutdown must synchronize before polling")

        def synchronize(self) -> None:
            self.synchronize_calls += 1
            raise RuntimeError(f"injected {method} shutdown fence failure")

    class FeatureTransport:
        def __init__(self) -> None:
            self.close_calls = 0
            self.quarantine_calls = 0

        def close(self) -> None:
            self.close_calls += 1

        def quarantine(self) -> None:
            self.quarantine_calls += 1

    def hold_executor() -> None:
        worker_started.set()
        assert release_worker.wait(timeout=2)

    monkeypatch.setattr(
        dt_kv_p2p.torch.cuda,
        "set_device",
        lambda _device: operations.append("set_device"),
    )
    transport = object.__new__(DraftKvP2pTransport)
    _init_p2p_transport_fence_state(transport)
    transport.device = torch.device("cuda:0")
    transport.device_index = 0
    transport._runtime = UnusedRuntime()
    transport._mapping_cache = RetainedMappingCache()
    transport._copy_stream = object()
    transport._buffers = ()
    transport._executor = ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="test-dt-kv-p2p"
    )
    blocker = transport._executor.submit(hold_executor)

    try:
        assert worker_started.wait(timeout=2)
        source_pool = SimpleNamespace(
            instance_session_id="source-session",
            pool_generation="source-generation",
            allocations=(SimpleNamespace(ipc_handle=b"source-allocation"),),
            buffers=(),
        )
        transfer = SimpleNamespace(
            page_indices=(),
            kv_bytes=0,
            producer_event_handle=b"source-event",
        )
        reservation = SimpleNamespace(page_indices=())
        queued = transport.submit(
            source_draft_instance_id="draft-source",
            source_pool=source_pool,
            transfer=transfer,
            reservation=reservation,
        )
        prepared = SimpleNamespace(
            reservation=reservation,
            source_pool=source_pool,
            transfer_future=queued,
        )
        task = SimpleNamespace(
            transfer_id=f"transfer-{method}",
            request_id="request-0",
            draft_kv_handle="handle-0",
        )
        completion_event = FailingCompletionEvent()
        completion = SimpleNamespace(
            pending=(PendingDraftTask(task=task),),
            staging_slot=7,
            completion_event=completion_event,
        )
        feature_transport = FeatureTransport()

        engine = object.__new__(FlashMTPDraftEngine)
        engine._is_feature_backed = True
        engine._is_eagle3 = method == "eagle3"
        engine._persistent = True
        engine._stop = True
        engine._fatal_gpu_quarantine_reason = None
        engine._p2p_transport = transport
        engine._feature_pull_transport = feature_transport
        engine._pending = deque()
        engine._pending_ids = set()
        engine._received_ns_by_transfer = {}
        engine._cancelled_ids = set()
        engine._pending_feature_batches = deque()
        engine._pending_dflash_completions = deque(
            (completion,) if method == "dflash" else ()
        )
        engine._pending_eagle3_completions = deque(
            (completion,) if method == "eagle3" else ()
        )
        engine._quarantined_eagle3_completions = []
        engine._quarantined_feature_draft_handles = set()
        engine._free_feature_staging_slots = deque((3,))
        engine._persistent_requests = {"handle-0": object()}
        engine._prepared_migrations = {"migration-queued": prepared}
        engine._deferred_migration_finalizations = {}
        engine.batch_source_read_done = False
        capacity_updates = []
        engine._publish_capacity = lambda: capacity_updates.append("capacity")

        with pytest.raises(RuntimeError, match=f"injected {method} shutdown"):
            engine.close()

        expected_reason = (
            "unquiesced DFlash shutdown completion-event failure"
            if method == "dflash"
            else "unquiesced EAGLE3 shutdown completion-event failure"
        )
        assert engine._fatal_gpu_quarantine_reason == expected_reason
        assert engine._stop
        assert engine._quarantined_feature_draft_handles == {"handle-0"}
        assert list(engine._free_feature_staging_slots) == [3]
        assert "handle-0" in engine._persistent_requests
        assert engine._prepared_migrations["migration-queued"] is prepared
        assert prepared.reservation is reservation
        assert feature_transport.close_calls == 1
        assert feature_transport.quarantine_calls == 1
        assert completion_event.synchronize_calls == 1
        assert completion_event.query_calls == 0
        if method == "dflash":
            assert list(engine._pending_dflash_completions) == [completion]
            assert not engine._quarantined_eagle3_completions
        else:
            assert not engine._pending_eagle3_completions
            assert engine._quarantined_eagle3_completions == [completion]

        release_worker.set()
        blocker.result(timeout=2)
        with pytest.raises(RuntimeError, match="permanent.*quarantine"):
            queued.result(timeout=2)
        assert operations == []
        assert transport._mapping_cache.maps == []
        assert transport._mapping_cache.retained_mapping is not None
        assert not transport._mapping_cache.closed

        # A repeated close fails at the common latch before another event
        # query/synchronization or either transport close can run.
        with pytest.raises(RuntimeError, match="fatal GPU quarantine"):
            engine.close()
        assert capacity_updates == ["capacity", "capacity"]
        assert completion_event.synchronize_calls == 1
        assert completion_event.query_calls == 0
        assert feature_transport.close_calls == 1
        assert engine._prepared_migrations["migration-queued"] is prepared
        assert list(engine._free_feature_staging_slots) == [3]
        assert "handle-0" in engine._persistent_requests
    finally:
        release_worker.set()
        transport._executor.shutdown(wait=True, cancel_futures=True)


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
    engine._p2p_transport = SimpleNamespace(probe_quiescence=lambda: quiescence_probe)
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


def test_shutdown_retains_an_outstanding_exported_source_lease() -> None:
    """Remote-reader ownership keeps source KV alive through repeated close."""

    class Closable:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    class MigrationTransport(Closable):
        def __init__(self) -> None:
            super().__init__()
            self.quarantine_calls = 0

        def quarantine(self) -> None:
            self.quarantine_calls += 1

    class ResultStore(Closable):
        def __init__(self) -> None:
            super().__init__()
            self.unlink_calls = 0

        def unlink(self) -> None:
            self.unlink_calls += 1

    engine = object.__new__(FlashMTPDraftEngine)
    _init_lifetime_state(engine)
    engine._stop = False
    engine._fatal_gpu_quarantine_reason = None
    engine._feature_pull_transport = None
    engine._prepared_migrations = {}
    engine._quarantined_migration_ids = set()
    engine._deferred_migration_finalizations = {}
    producer_event = object()
    source_owner = object()
    engine._migration_source_events = {"migration-source": producer_event}
    engine._migration_source_handles = {"migration-source": "kv-source"}
    engine._persistent_requests = {"kv-source": source_owner}
    transport = MigrationTransport()
    client = Closable()
    result_store = ResultStore()
    engine._p2p_transport = transport
    engine._client = client
    engine._result_store = result_store

    with pytest.raises(RuntimeError, match="Draft migration shutdown") as first:
        engine.close()

    assert "source_events" in engine._fatal_gpu_quarantine_reason
    assert "source_handles" in engine._fatal_gpu_quarantine_reason
    assert transport.quarantine_calls == 1
    assert transport.close_calls == 0
    assert client.close_calls == 0
    assert result_store.close_calls == 0
    assert result_store.unlink_calls == 0
    assert engine._migration_source_events == {
        "migration-source": producer_event
    }
    assert engine._migration_source_handles == {"migration-source": "kv-source"}
    assert engine._persistent_requests == {"kv-source": source_owner}

    with pytest.raises(RuntimeError, match="Draft engine shutdown") as repeated:
        engine.close()

    assert first.value is not repeated.value
    assert transport.quarantine_calls == 1
    assert transport.close_calls == 0
    assert client.close_calls == 0
    assert result_store.close_calls == 0
    assert result_store.unlink_calls == 0
    assert engine._migration_source_events["migration-source"] is producer_event
    assert engine._persistent_requests["kv-source"] is source_owner


def test_shutdown_retains_an_unquiesced_destination_migration() -> None:
    """Shutdown must not discard a reservation with an unconsumed copy result."""

    class Closable:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    class MigrationTransport(Closable):
        def __init__(self) -> None:
            super().__init__()
            self.quarantine_calls = 0

        def quarantine(self) -> None:
            self.quarantine_calls += 1

    class ResultStore(Closable):
        def __init__(self) -> None:
            super().__init__()
            self.unlink_calls = 0

        def unlink(self) -> None:
            self.unlink_calls += 1

    engine = object.__new__(FlashMTPDraftEngine)
    _init_lifetime_state(engine)
    engine._stop = False
    engine._fatal_gpu_quarantine_reason = None
    engine._feature_pull_transport = None
    future = Future()
    future.set_exception(TransferQuiescenceError("GPU completion is unknown"))
    reservation = object()
    req = SimpleNamespace(req_pool_idx=7, kv=object())
    prepared = SimpleNamespace(
        transfer_future=future,
        message=SimpleNamespace(draft_kv_handle="kv-0"),
        reservation=reservation,
        req=req,
    )
    deferred_finalization = object()
    quiescence_probe = Future()
    transport = MigrationTransport()
    client = Closable()
    result_store = ResultStore()
    cleaned = []
    engine._p2p_transport = transport
    engine._prepared_migrations = {"migration-0": prepared}
    engine._quarantined_migration_ids = {"migration-0"}
    engine._deferred_migration_finalizations = {
        "migration-0": deferred_finalization
    }
    engine._migration_quiescence_probes = {"migration-0": quiescence_probe}
    engine._persistent_requests = {}
    engine._migration_source_events = {}
    engine._migration_source_handles = {}
    engine._client = client
    engine._result_store = result_store
    engine._cleanup_partial_migration_request = cleaned.append

    with pytest.raises(RuntimeError, match="Draft migration shutdown"):
        engine.close()

    assert "prepared" in engine._fatal_gpu_quarantine_reason
    assert "quarantined" in engine._fatal_gpu_quarantine_reason
    assert "deferred_finalizations" in engine._fatal_gpu_quarantine_reason
    assert "quiescence_probes" in engine._fatal_gpu_quarantine_reason
    assert transport.quarantine_calls == 1
    assert transport.close_calls == 0
    assert not cleaned
    assert engine._prepared_migrations["migration-0"] is prepared
    assert prepared.transfer_future is future
    assert prepared.reservation is reservation
    assert prepared.req is req
    assert engine._quarantined_migration_ids == {"migration-0"}
    assert (
        engine._deferred_migration_finalizations["migration-0"]
        is deferred_finalization
    )
    assert engine._migration_quiescence_probes["migration-0"] is quiescence_probe
    assert client.close_calls == 0
    assert result_store.close_calls == 0
    assert result_store.unlink_calls == 0

    with pytest.raises(RuntimeError, match="Draft engine shutdown"):
        engine.close()

    assert transport.quarantine_calls == 1
    assert transport.close_calls == 0
    assert not cleaned
    assert engine._prepared_migrations["migration-0"] is prepared
    assert prepared.transfer_future is future
    assert prepared.reservation is reservation
    assert engine._quarantined_migration_ids == {"migration-0"}
    assert client.close_calls == 0
    assert result_store.close_calls == 0
    assert result_store.unlink_calls == 0


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


def test_duplicate_result_ack_does_not_resurrect_pulled_range() -> None:
    engine = object.__new__(FlashMTPDraftEngine)
    engine._result_pool_generation = "result-pool"
    engine._client = SimpleNamespace(instance_session_id="draft-session")
    first_key = ("draft-session", "result-pool", 7, 3, 0, 1)
    second_key = ("draft-session", "result-pool", 7, 3, 1, 1)
    published = PublishedResult(
        batch_id="result-batch",
        slot_id=7,
        generation=3,
        transfer_ids=("transfer-a", "transfer-b"),
        transfer_rows={
            "transfer-a": ("target-0", 0),
            "transfer-b": ("target-1", 1),
        },
        range_transfer_ids={
            first_key: frozenset({"transfer-a"}),
            second_key: frozenset({"transfer-b"}),
        },
    )
    engine._published_by_batch = {published.batch_id: published}
    engine._published_by_range = {
        first_key: published.batch_id,
        second_key: published.batch_id,
    }
    engine._free_result_slots = deque()

    def pull_done(target: str, row: int) -> PullDone:
        return PullDone(
            protocol_version=PROTOCOL_VERSION,
            target_instance_id=target,
            draft_instance_id="draft-0",
            buffer_id=7,
            generation=3,
            row_offset=row,
            row_count=1,
            completed_ns=100,
            draft_session_id="draft-session",
            result_pool_generation="result-pool",
        )

    engine._handle_pull_done(pull_done("target-0", 0), acknowledge=False)
    assert set(published.range_transfer_ids.values()) == {
        frozenset({"transfer-b"})
    }

    # Reliable delivery may repeat the original accepted set after the first
    # Target has already released its rows. Rebuilding must only shrink the
    # live set because that Target's PullDone was already consumed.
    engine._rebuild_accepted_ranges(
        published, {"transfer-a", "transfer-b"}
    )
    assert set(published.range_transfer_ids.values()) == {
        frozenset({"transfer-b"})
    }

    assert engine._handle_pull_done(
        pull_done("target-1", 1), acknowledge=False
    )
    assert not engine._published_by_batch
    assert not engine._published_by_range
    assert list(engine._free_result_slots) == [7]


def test_pull_done_releases_singleton_from_coalesced_target_range() -> None:
    engine = object.__new__(FlashMTPDraftEngine)
    engine._result_pool_generation = "result-pool"
    engine._client = SimpleNamespace(instance_session_id="draft-session")
    combined_key = ("draft-session", "result-pool", 7, 3, 0, 3)
    published = PublishedResult(
        batch_id="result-batch",
        slot_id=7,
        generation=3,
        transfer_ids=("transfer-a", "transfer-b", "transfer-c"),
        transfer_rows={
            "transfer-a": ("target-0", 0),
            "transfer-b": ("target-0", 1),
            "transfer-c": ("target-0", 2),
        },
        range_transfer_ids={
            combined_key: frozenset(
                {"transfer-a", "transfer-b", "transfer-c"}
            )
        },
    )
    engine._published_by_batch = {published.batch_id: published}
    engine._published_by_range = {combined_key: published.batch_id}
    engine._free_result_slots = deque()

    def pull_done(row: int, count: int = 1, target: str = "target-0") -> PullDone:
        return PullDone(
            protocol_version=PROTOCOL_VERSION,
            target_instance_id=target,
            draft_instance_id="draft-0",
            buffer_id=7,
            generation=3,
            row_offset=row,
            row_count=count,
            completed_ns=100,
            draft_session_id="draft-session",
            result_pool_generation="result-pool",
        )

    with pytest.raises(ValueError, match="published Target result rows"):
        engine._handle_pull_done(
            pull_done(0, count=3, target="wrong-target"), acknowledge=False
        )
    assert published.range_transfer_ids == {
        combined_key: frozenset(
            {"transfer-a", "transfer-b", "transfer-c"}
        )
    }

    assert not engine._handle_pull_done(pull_done(1), acknowledge=False)
    leading_key = ("draft-session", "result-pool", 7, 3, 0, 1)
    trailing_key = ("draft-session", "result-pool", 7, 3, 2, 1)
    assert published.range_transfer_ids == {
        leading_key: frozenset({"transfer-a"}),
        trailing_key: frozenset({"transfer-c"}),
    }
    assert engine._published_by_range == {
        leading_key: published.batch_id,
        trailing_key: published.batch_id,
    }

    assert engine._handle_pull_done(pull_done(0, count=3), acknowledge=False)
    assert not engine._published_by_batch
    assert not engine._published_by_range
    assert list(engine._free_result_slots) == [7]


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
