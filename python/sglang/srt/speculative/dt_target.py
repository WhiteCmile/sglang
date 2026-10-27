from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass

import torch
from dt_profile import event_enabled as profile_event_enabled
from dt_profile import explicit_event_enabled as profile_event_explicitly_enabled
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.model_runner import (
    DT_BENCHMARK_MEASUREMENT_RID_PREFIX,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.utils import broadcast_pyobj
from sglang.srt.utils.common import is_pin_memory_available

logger = logging.getLogger(__name__)

# Coordinator failure codes meaning the assigned Draft session is gone.
RING_OWNER_LOSS_CODES = frozenset(
    {"draft_session_replaced", "draft_result_pool_retired", "draft_owner_session_lost"}
)
# Initial control delivery is immediate.  A one-second retry cadence still
# provides multiple attempts inside the round lease while preventing a busy
# scheduler from turning transient ACK latency into a duplicate-message storm.
CONTROL_RETRY_INTERVAL_NS = 1_000_000_000
DT_LANE_MEASUREMENT_MARKER = "DT lane measurement evidence: "
DT_DIRECT_GPU_SOURCE_MARKER = "DT direct GPU source evidence: "
DT_LIFECYCLE_STATUS_MARKER = "DT lifecycle status: "
DT_LIFECYCLE_STATUS_INTERVAL_NS = 250_000_000
EAGLE3_HOST_SPAN_PROFILE_EVENT = "target_eagle3_host_span"


class _PendingTaskPayloadMismatch(ValueError):
    """A queued copy reused a transfer identity with different immutable data."""


def _parse_pipeline_lane_weights(
    value: str | None, pipeline_lanes: int
) -> tuple[int, ...]:
    if value is None:
        return (1,) * pipeline_lanes
    try:
        weights = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError(
            "draft_target_pipeline_lane_weights must contain comma-separated "
            "positive integers"
        ) from exc
    if len(weights) != pipeline_lanes or any(weight <= 0 for weight in weights):
        raise ValueError(
            "draft_target_pipeline_lane_weights must contain one positive "
            "integer per pipeline lane"
        )
    return weights


def _batched_eagle3_token_metadata(
    *,
    active_indices: list[int],
    token_chunks: list[torch.Tensor],
    committed_chunks: list[torch.Tensor],
) -> tuple[dict[int, tuple[int, ...]], dict[int, tuple[int, ...]]]:
    """Copy EAGLE3's two ragged token streams to CPU in two batches."""
    if not active_indices:
        return {}, {}
    lengths = [int(token_chunks[index].numel()) for index in active_indices]
    committed_lengths = [
        int(committed_chunks[index].numel()) for index in active_indices
    ]
    if any(length <= 0 for length in lengths) or committed_lengths != lengths:
        raise ValueError("EAGLE3 token metadata lengths do not match feature rows")
    token_values = (
        torch.cat([token_chunks[index] for index in active_indices])
        .detach()
        .to(device="cpu")
        .tolist()
    )
    committed_values = (
        torch.cat([committed_chunks[index] for index in active_indices])
        .detach()
        .to(device="cpu")
        .tolist()
    )
    tokens_by_index: dict[int, tuple[int, ...]] = {}
    committed_by_index: dict[int, tuple[int, ...]] = {}
    offset = 0
    for index, length in zip(active_indices, lengths, strict=True):
        end = offset + length
        tokens_by_index[index] = tuple(int(value) for value in token_values[offset:end])
        committed_by_index[index] = tuple(
            int(value) for value in committed_values[offset:end]
        )
        offset = end
    if offset != len(token_values) or offset != len(committed_values):
        raise RuntimeError("EAGLE3 batched token metadata split lost rows")
    return tokens_by_index, committed_by_index


def _flashmtp_pivot_preservation_indices(batch: ScheduleBatch) -> list[int]:
    """Rows whose GPU bytes must survive deferred overlap result processing.

    Proposal eligibility is intentionally absent here.  A final prefill chunk
    may launch while an older middle-chunk result still owns the same request;
    those ownership fences are authoritative only when results are processed
    in order.  At this boundary the graph output can be replayed first, so every
    live row needs an independent bounded source lease.
    """

    return [
        index
        for index, req in enumerate(batch.reqs)
        if not req.finished() and not req.is_retracted
    ]


def _flashmtp_pivot_publish_view(
    pivot_hidden: torch.Tensor, preserved_indices: list[int]
) -> tuple[torch.Tensor, bool]:
    """Return contiguous publication rows without gathering an unchanged batch.

    The direct source pool copies the returned rows into its own fenced storage
    before the graph output may be replayed.  When every batch row is live and
    the graph output is already contiguous, an advanced-index gather would
    only materialize the same tensor and can dominate the Target wave.  A
    filtered or non-contiguous selection still takes the owning gather path.
    """

    if (
        len(preserved_indices) == int(pivot_hidden.shape[0])
        and all(index == row for row, index in enumerate(preserved_indices))
        and pivot_hidden.is_contiguous()
    ):
        return pivot_hidden.detach(), False
    return pivot_hidden[preserved_indices].detach().contiguous(), True


def _flashmtp_park_scalar_values(
    *,
    batch_result,
    draft_input: DFlashDraftInputV2,
    active_indices: list[int],
    original_batch_size: int,
) -> tuple[list[int], list[int], bool]:
    """Resolve FlashMTP control scalars, preferring overlap-staged CPU rows."""

    if not active_indices:
        return [], [], False
    staged_bonus = getattr(batch_result, "dt_flashmtp_bonus_tokens_cpu", None)
    staged_seq_lens = getattr(batch_result, "dt_flashmtp_seq_lens_cpu", None)
    if (staged_bonus is None) != (staged_seq_lens is None):
        raise RuntimeError("FlashMTP scalar staging is incomplete")
    if staged_bonus is not None:
        if (
            not isinstance(staged_bonus, torch.Tensor)
            or not isinstance(staged_seq_lens, torch.Tensor)
            or staged_bonus.device.type != "cpu"
            or staged_seq_lens.device.type != "cpu"
            or staged_bonus.ndim != 1
            or staged_seq_lens.ndim != 1
            or int(staged_bonus.numel()) != original_batch_size
            or int(staged_seq_lens.numel()) != original_batch_size
        ):
            raise RuntimeError("FlashMTP staged control scalars changed shape or device")
        bonus_values = staged_bonus
        seq_len_values = staged_seq_lens
    else:
        bonus_values = draft_input.bonus_tokens
        seq_len_values = draft_input.new_seq_lens
    if active_indices != list(range(original_batch_size)):
        bonus_values = bonus_values[active_indices]
        seq_len_values = seq_len_values[active_indices]
    return (
        [int(value) for value in bonus_values.detach().cpu().tolist()],
        [int(value) for value in seq_len_values.detach().cpu().tolist()],
        staged_bonus is not None,
    )


def _ready_batch_h2d(
    cpu_tensor: torch.Tensor,
    *,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Queue a READY-batch host tensor without synchronizing the scheduler."""

    return cpu_tensor.to(
        device=device,
        dtype=dtype,
        non_blocking=cpu_tensor.is_pinned(),
    )


def _ready_candidate_host_tensor(
    candidate_blob: bytes,
    *,
    batch_size: int,
    block_size: int,
    pin_memory: bool,
) -> torch.Tensor:
    """Decode int32 wire candidates into the final int64 host staging dtype."""

    wire_tokens = torch.frombuffer(
        bytearray(candidate_blob), dtype=torch.int32
    ).reshape(batch_size, block_size)
    staged_tokens = torch.empty(
        (batch_size, block_size),
        dtype=torch.int64,
        pin_memory=pin_memory,
    )
    staged_tokens.copy_(wire_tokens)
    return staged_tokens


@dataclass(slots=True)
class ParkedTargetRound:
    req: Req
    transfer_id: str
    round_id: int
    attempt: int
    bonus_token: int
    seq_len: int
    lane_id: int = 0
    candidate_bytes: bytes | None = None
    candidate_completion: object | None = None
    failure: str | None = None
    draft_admitted: bool = False
    # Local round-timeout fence for rounds that have not entered the task
    # ledger yet (their Coordinator copy is still queued or unbuilt).  Zero
    # disables the fence; parking sites stamp it with the round deadline.
    deadline_ns: int = 0
    # Profiling-only provenance. Only the export rank consumes coordinator
    # result batches directly; these fields let its READY event distinguish
    # transport/poll delay from lane assembly without changing the TP payload.
    result_batch_id: str | None = None
    result_received_ns: int = 0


@dataclass(frozen=True, slots=True)
class DeferredFeatureRound:
    request_id: str
    round_id: int
    attempt: int
    draft_kv_handle: str | None
    committed_len: int | None
    previous_transfer_id: str | None


@dataclass(frozen=True, slots=True)
class StagedDFlashTask:
    request_id: str
    transfer_id: str
    round_id: int
    attempt: int
    draft_kv_handle: str
    feature_token_offset: int
    feature_token_count: int
    feature_token_ids: tuple[int, ...]
    committed_token_ids: tuple[int, ...]
    previous_transfer_id: str | None
    bonus_token: int
    seq_len: int


@dataclass(slots=True)
class StagedDFlashFeatureBatch:
    tasks: tuple[StagedDFlashTask, ...]
    descriptors_by_transfer: dict[str, object]
    # Kernel-published (§7.1) waves stage without descriptors or token-id
    # tuples; their Coordinator confirmation tasks are built by the deferred
    # cleanup in _ring_publish_cleanup.
    ring_publish: bool = False


class RingPublishCommittedError(RuntimeError):
    """The ring record went out; the wave may not be re-published.

    Raised only after the publish kernel was launched.  The caller fails the
    wave closed instead of falling back to the Python publication path,
    which would double-publish the same round to the owning Draft.
    """


@dataclass(slots=True)
class RingPublishWave:
    """A kernel-published verify wave awaiting its deferred confirmation.

    The ring record is authoritative for the Draft; the cleanup rebuilds the
    Coordinator-bound copies from the record itself (no GPU sync) and keeps
    the pool/ledger bookkeeping consistent with the Python publication path.
    """

    session_id: str
    wave_seq: int
    slot_id: int
    generation: int
    row_count: int
    ready_event: object
    # 12-field StagedDFlashTask values with empty token-id tuples, in the
    # record's request-table order; aligned with active_slots.
    specs: tuple
    reqs: dict  # rid -> req object at stage time
    active_slots: tuple
    # Filled by _park_dflash_completed_batch: transfer_id ->
    # (requires_proposal, previous_transfer_id).
    proposals: dict


@dataclass(frozen=True, slots=True)
class StagedFlashMTPTask:
    request_id: str
    request_index: int
    source_lease_id: str


@dataclass(slots=True)
class StagedFlashMTPPivotBatch:
    tasks: tuple[StagedFlashMTPTask, ...]
    descriptors_by_transfer: dict[str, object]


@dataclass(slots=True)
class _PendingTargetGpuObservation:
    wave_id: int
    transfer_ids: tuple[str, ...]
    ready_ns: int
    selection_started_ns: int
    selection_finished_ns: int
    submission_started_ns: int
    submission_returned_ns: int
    start_event: object
    end_event: object
    start_observed_ns: int = 0
    # DT_GPU_CLOCK: absolute GPU time of the verify end (dt_gpu_clock).
    clock_slot: int = -1
    clock_done: object = None


class DTTargetController:
    # Object-level construction in existing CPU diagnostics bypasses
    # ``__init__``.  Their established behavior is the mmap A/B baseline.
    candidate_transport = "mmap"
    _candidate_arena = None
    target_phase_profile_enabled = False

    """Owns WAIT_DRAFT requests outside the normal SGLang running batch."""

    def __init__(self, scheduler) -> None:
        try:
            from dt_client import (
                RetryOutbox,
                TargetCoordinatorClient,
                TargetRoundLedger,
            )
            from dt_launcher.topology import default_candidate_slots
            from dt_profile import emit as profile_emit
            from dt_profile import flush as profile_flush
        except ImportError as exc:
            raise RuntimeError(
                "Draft-Target mode requires dt-runtime on PYTHONPATH"
            ) from exc

        self.scheduler = scheduler
        self._profile_emit = profile_emit
        self._profile_flush = profile_flush
        self._eagle3_host_span_startup_flushed = False
        self._eagle3_host_span_flushed_health_ids: set[str] = set()
        self._profile_active = bool(os.environ.get("DT_PROFILE_DIR"))
        self.server_args = scheduler.server_args
        self.instance_id = self.server_args.draft_target_instance_id
        self.profile_id = self.server_args.draft_target_profile_id
        self.feature_schema_id = self.server_args.draft_target_feature_schema_id
        self.block_size = int(self.server_args.speculative_num_draft_tokens)
        self.pivot_width = int(scheduler.draft_worker._pivot_width)
        self.is_token_only = scheduler.spec_algorithm.is_standalone_dt()
        self.is_dflash = scheduler.spec_algorithm.is_dflash()
        self.is_eagle3 = scheduler.spec_algorithm.is_eagle3()
        self.is_feature_backed = self.is_dflash or self.is_eagle3
        self.is_persistent = self.is_token_only or self.is_feature_backed
        self.candidate_transport = str(
            self.server_args.draft_target_candidate_transport
        )
        if self.candidate_transport not in {"mmap", "cuda-ipc"}:
            raise ValueError("candidate transport must be mmap or cuda-ipc")
        self.defer_dflash_feature_stage = bool(
            self.server_args.draft_target_defer_dflash_feature_stage
        )
        if self.defer_dflash_feature_stage and (
            not self.is_dflash or not self.server_args.draft_target_overlap_schedule
        ):
            raise ValueError(
                "deferred DFlash feature staging requires remote DFlash with "
                "Draft-Target overlap scheduling"
            )
        if self.is_feature_backed and not self.server_args.disable_radix_cache:
            raise ValueError(
                "remote feature-backed Draft requires --disable-radix-cache until cached-prefix "
                "hidden states have an authoritative export path"
            )
        self.round_timeout_ns = (
            int(self.server_args.draft_target_round_timeout_ms) * 1_000_000
        )
        self.whole_batch_task_fifo_ablation = (
            os.environ.get("SGLANG_DT_WHOLE_BATCH_TASK_FIFO_ABLATION", "0") == "1"
        )
        self.continuous_batching = not getattr(
            self.server_args, "draft_target_disable_continuous_batching", False
        )
        self.pipeline_lanes = int(self.server_args.draft_target_pipeline_lanes)
        self.pipeline_lane_weights = _parse_pipeline_lane_weights(
            self.server_args.draft_target_pipeline_lane_weights,
            self.pipeline_lanes,
        )
        overlap_result_defer_ms = float(
            self.server_args.draft_target_overlap_result_defer_ms
        )
        if overlap_result_defer_ms < 0:
            raise ValueError(
                "draft_target_overlap_result_defer_ms must be non-negative"
            )
        if (
            overlap_result_defer_ms > 0
            and not self.server_args.draft_target_overlap_schedule
        ):
            raise ValueError(
                "draft_target_overlap_result_defer_ms requires "
                "draft_target_overlap_schedule"
            )
        if self.is_token_only:
            if (
                not self.server_args.draft_target_full_wave
                or self.pipeline_lanes not in {1, 2}
            ):
                raise ValueError(
                    "Draft-Target V2 requires full-wave verification with one or "
                    "two pipeline lanes"
                )
        elif self.is_feature_backed:
            if (
                not self.server_args.draft_target_full_wave
                or self.pipeline_lanes not in {2, 3}
            ):
                raise ValueError(
                    "DFlash Draft-Target requires full-wave verification with "
                    "two or three pipeline lanes"
                )
        elif not self.server_args.draft_target_full_wave or self.pipeline_lanes != 2:
            raise ValueError(
                "Draft-Target V1 requires full-wave verification with two lanes"
            )
        self.overlap_result_defer_ns = int(overlap_result_defer_ms * 1_000_000)
        self.is_export_rank = scheduler.ps.tp_rank == 0
        self.target_phase_profile_enabled = bool(
            self.is_export_rank and self._profile_active
        )
        self.empty_prefill_arbitration_profile_enabled = bool(
            self.is_export_rank and self._profile_active
        )
        self._overlap_fix_observer = None
        if os.environ.get("DT_OVERLAP_FIX_OBSERVER_DIR"):
            from overlap_observer import create

            self._overlap_fix_observer = create(
                self.instance_id, "target", scheduler.ps.tp_rank, scheduler.ps.tp_size
            )
        # This diagnostic performs per-batch timestamping, so unlike ordinary
        # host trace events it must never inherit the empty-filter-means-all
        # behavior.  Only TP0 owns the emitted control-plane chronology.
        self.eagle3_host_span_profile_enabled = bool(
            self.is_eagle3
            and self.is_export_rank
            and profile_event_explicitly_enabled(EAGLE3_HOST_SPAN_PROFILE_EVENT)
        )
        if (
            self.eagle3_host_span_profile_enabled
            and not self.server_args.draft_target_overlap_schedule
        ):
            raise ValueError(
                "target_eagle3_host_span requires Draft-Target overlap scheduling"
            )
        self._parked: dict[str, ParkedTargetRound] = {}
        self._feature_control_rounds: dict[str, ParkedTargetRound] = {}
        self._draft_admitted_attempts: dict[str, int] = {}
        # Overlap scheduling can have more than one chunked-prefill forward for
        # the same request in ``result_queue``.  Track the number of launched
        # Target forwards whose results have not been consumed yet.  A request
        # with a positive count may continue through the chunked-prefill path,
        # but must not re-enter the normal running batch or start a Draft
        # transfer from an older result.
        self._target_inflight_counts: Counter[str] = Counter()
        self._target_inflight_reqs: dict[str, Req] = {}
        self._target_compute_batches_inflight = 0
        self._next_round: dict[str, int] = {}
        # V2 token-only lifecycle.  The Draft owns the physical KV handle; the
        # Target records the authoritative full committed sequence boundary and
        # the transfer currently being verified.
        self._draft_committed_len: dict[str, int] = {}
        # DFlash overlap can stage several Target results before the oldest is
        # turned into a control task.  Keep the furthest staged boundary above
        # separate from the exact FIFO boundary already routed to Draft.
        self._draft_dispatched_len: dict[str, int] = {}
        self._draft_kv_handle_by_request: dict[str, str] = {}
        self._lifecycle_attempt_by_request: dict[str, int] = {}
        self._next_lifecycle_attempt = 0
        self._verify_transfer_by_request: dict[str, str] = {}
        self._flashmtp_pivot_sequence = 0
        self._batch_sequence = 0
        self._pending_send = deque()
        self._last_pending_send_block: tuple[str, str, str] | None = None
        self._delivery_outbox = RetryOutbox[object](
            retry_interval_ns=CONTROL_RETRY_INTERVAL_NS
        )
        self._cancelled_transfers: set[str] = set()
        # FlashMTP graph-output preservation happens before ordered result
        # processing decides whether a row may become a remote proposal.  Keep
        # those unpublished leases distinct from Coordinator-owned transfers:
        # they can be cancelled locally, including during shutdown, until a
        # task batch has been committed to the reliable control path.
        self._staged_flashmtp_source_leases: set[str] = set()
        self._source_slot_by_transfer: dict[str, tuple[int, int]] = {}
        self._source_deadline_by_transfer: dict[str, int] = {}
        self._source_done_applied_ids: set[str] = set()
        self._source_done_batch_entries: dict[str, tuple[str, ...]] = {}
        self._result_stores: dict[str, object] = {}
        self._last_poll_ns = 0
        self._lifecycle_status_sequence = 0
        self._last_lifecycle_status_ns = 0
        self._last_lifecycle_status_sample_ns = 0
        self._last_lifecycle_quiescent: bool | None = None
        # The formal harness needs one successful measured CUDA-IPC event from
        # every endpoint.  Emitting the same proof for every scheduler batch
        # adds synchronous JSON/log I/O to the inference hot path without
        # strengthening that admission check.
        self._direct_gpu_measurement_evidence_emitted = False
        self._direct_gpu_measurement_epochs: dict[
            tuple[str, str, str, str], list[int]
        ] = {}
        self._lane_by_request: dict[str, int] = {}
        self._lane_target_sizes = [0] * self.pipeline_lanes
        self._lane_assignment_cursor = 0
        self._lane_ready_cursor = 0
        self._next_full_wave_id = 0
        self._last_full_wave_completed_ns = 0
        self._last_lane_full_wave_completed_ns = [0] * self.pipeline_lanes
        self._pending_target_gpu_observations: deque[
            _PendingTargetGpuObservation
        ] = deque()
        self._last_target_verify_end_event = None
        self._measurement_lane_counts: dict[tuple[int, str], int] = {}
        self._overlap_launch_ns = 0
        self._overlap_result_ready: bool | None = None
        # Late batch freezing: while a launched batch still has more remaining
        # GPU time than it takes to form and submit the next one, hold the next
        # batch open so READY work and new requests arriving meanwhile join it.
        self._late_freeze = bool(
            self.continuous_batching
            and os.environ.get("DT_TARGET_LATE_FREEZE", "1") == "1"
        )
        self._freeze_margin_ns = int(
            float(os.environ.get("DT_TARGET_FREEZE_MARGIN_MS", "0")) * 1_000_000
        )
        self._freeze_inflight: deque[list] = deque()
        self._freeze_busy_ema_ns: dict[tuple[bool, int], float] = {}
        self._freeze_lead_ema_ns: dict[bool, float] = {}
        self._freeze_last_done_ns = 0
        self.hold_next_batch = False
        # Two-threshold READY batching (design rev 39): launch once READY rows
        # reach B, the wait reaches T, or nothing is still out at a Draft.
        self._ready_batch_wait_ns = int(
            float(os.environ.get("DT_TARGET_BATCH_WAIT_MS", "0")) * 1_000_000
        )
        self._ready_batch_min_size = int(os.environ.get("DT_TARGET_BATCH_MIN_SIZE", "64"))
        self._ready_hold_started_ns = 0
        # Same-host Target -> owner-Draft descriptor rings (dt_direct_ring).
        # They replace the former ZMQ direct dispatch (DT_DIRECT_DISPATCH).
        self._direct_ring = os.environ.get("DT_DIRECT_RING", "1") == "1"
        self._ring_writers: dict[str, object] = {}
        self._ring_outstanding: dict[str, deque] = {}
        self._ring_missing_until_ns: dict[str, int] = {}
        self._ring_encoder = None
        # Ring rounds are authorized by the request's owner lease and their
        # source leases retire only on the ring read proof or a lease
        # revocation (Draft session lost or process gone).  A Coordinator
        # failure result never speaks for a ring copy's source read.
        self._ring_lease = self._direct_ring
        self._ring_unproven: dict[str, str] = {}
        # Bounded record of ring rounds whose read the proof retired; their
        # relayed SourceReadDone is an idempotent repeat.
        self._ring_proven: OrderedDict[str, None] = OrderedDict()
        self._lease_marks: list[tuple[str, str]] = []
        self._ring_revoked_sessions: set[str] = set()
        # Owner lease per request from Coordinator assignments:
        # request_id -> (draft_instance_id, draft_session_id, owner_epoch).
        self._draft_binding: dict[str, tuple[str, str, int]] = {}
        # Ring rounds written but not yet accepted by the Coordinator.
        self._direct_sent: set[str] = set()
        # §7.1 GPU ring publish (DT_RING_PUBLISH=1): eligible verify waves go
        # out as kernel-written ring records; the Coordinator confirmation
        # tasks are built by a deferred cleanup off the park critical path.
        self._ring_publish = os.environ.get("DT_RING_PUBLISH", "1") == "1"
        self._ring_publishers: dict[str, object] = {}  # session -> RingGpuPublisher
        self._ring_publish_pending: deque = deque()
        self._ring_publish_transfers: set[str] = set()
        # Producer events of waves that failed after publication; the records
        # on the ring still reference them, so ownership cannot be dropped.
        self._ring_publish_orphaned_events: deque = deque(maxlen=1024)
        # Sticky publication failure, broadcast to other ranks via poll().
        self._ring_publish_fatal: str | None = None
        # Producer events are per-slot, not per-wave: a slot is never
        # re-leased before its previous read completes, so re-recording the
        # slot's event cannot alias an unconsumed wait.  IPC handles are
        # exported once per slot.
        self._slot_ready_events: dict[int, object] = {}
        self._slot_event_handles: dict[int, bytes] = {}
        self._client = None
        self._ledger = None
        self._feature_source_pool = None
        self._candidate_arena = None
        # A failed direct-GPU-source shutdown is not an ordinary close.  Until
        # every remote read is acknowledged, the pool remains the owner of its
        # exported allocation, producer events, and active leases.  Latch the
        # first failure so a later close cannot retry cleanup against uncertain
        # CUDA work or drop the last strong reference to that ownership.
        self._source_shutdown_quarantine_error: BaseException | None = None
        self._feature_source_free_slots = 0
        self._deferred_dflash_feature_stages = 0
        # data_ptr -> (slot_id, generation, ready_event) for verify waves whose
        # hidden rows were written in place into a reserved source slot.
        self._inplace_hidden: dict[int, tuple[int, int, torch.cuda.Event]] = {}
        self._feature_source_stall_started_ns = 0
        self._last_empty_prefill_arbitration_profile = None
        if self.is_export_rank:
            try:
                self._client = TargetCoordinatorClient(
                    endpoint=self.server_args.draft_target_coordinator_endpoint,
                    instance_id=self.instance_id,
                    data_endpoint="",
                )
                if self.uses_direct_gpu_source:
                    from sglang.srt.speculative.dt_feature_p2p import (
                        TargetFeatureSourcePool,
                    )

                    max_source_rows = (
                        max(
                            int(self.server_args.max_prefill_buffer_tokens()),
                            int(scheduler.max_running_requests) * self.block_size,
                        )
                        if self.is_feature_backed
                        else int(scheduler.max_running_requests)
                    )
                    self._feature_source_pool = TargetFeatureSourcePool(
                        target_instance_id=self.instance_id,
                        source_session_id=self._client.instance_session_id,
                        device=scheduler.device,
                        slot_count=max(2, self.pipeline_lanes * 2),
                        max_tokens=max_source_rows,
                        feature_width=self.pivot_width,
                    )
                    self._profile_emit(
                        "target_feature_pool_registration",
                        target_instance=self.instance_id,
                        source_session_id=self._feature_source_pool.source_session_id,
                        pool_generation=self._feature_source_pool.pool_generation,
                        transport_session=self._feature_source_pool.transport_session,
                    )
                    self._feature_source_free_slots = (
                        self._feature_source_pool.free_slot_count
                    )
                    if self.is_feature_backed and os.environ.get(
                        "DT_INPLACE_DEFERRED_HIDDEN", "1"
                    ) == "1":
                        scheduler.draft_worker.dt_deferred_hidden_destination = (
                            self._reserve_deferred_hidden
                        )
                if self.candidate_transport == "cuda-ipc":
                    from sglang.srt.speculative.dt_candidate_p2p import (
                        TargetCandidateArena,
                    )

                    # Zero derives the launcher default from
                    # max_running_requests; the derivation lives in
                    # dt_launcher.topology (imported above) so sglang does
                    # not restate it.
                    minimum_slots = default_candidate_slots(
                        int(scheduler.max_running_requests)
                    )
                    configured_slots = int(
                        self.server_args.draft_target_candidate_slots
                    )
                    slot_count = (
                        configured_slots if configured_slots > 0 else minimum_slots
                    )
                    self._candidate_arena = TargetCandidateArena(
                        target_instance_id=self.instance_id,
                        target_session_id=self._client.instance_session_id,
                        device=scheduler.device,
                        slot_count=slot_count,
                        row_width=self.block_size,
                        profile_timing=self._profile_active,
                    )
                    self._client.candidate_arena = (
                        self._candidate_arena.registration()
                    )
                self._client.register_target(
                    timeout_s=self.server_args.draft_target_registration_timeout_s
                )
                self._ledger = TargetRoundLedger(target_instance_id=self.instance_id)
            except Exception:
                if self._client is not None:
                    self._client.close()
                    self._client = None
                if self._feature_source_pool is not None:
                    self._feature_source_pool.close()
                    self._feature_source_pool = None
                if self._candidate_arena is not None:
                    self._candidate_arena.close()
                    self._candidate_arena = None
                raise

    @property
    def uses_direct_gpu_source(self) -> bool:
        """Whether proposal input is leased from registered Target VRAM."""

        return self.is_feature_backed or not self.is_token_only

    def _emit_direct_gpu_measurement_evidence_once(
        self, payload: dict[str, object]
    ) -> bool:
        """Log the first successful measured direct-GPU publication."""

        if getattr(self, "_direct_gpu_measurement_evidence_emitted", False):
            return False
        logger.info(
            "%s%s",
            DT_DIRECT_GPU_SOURCE_MARKER,
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )
        self._direct_gpu_measurement_evidence_emitted = True
        return True

    def _record_direct_gpu_measurement_epoch(
        self,
        *,
        measurement_rows: int,
        measurement_requests: int,
        source_session_id: str,
        pool_generation: str,
        transport_session: str,
    ) -> None:
        epochs = getattr(self, "_direct_gpu_measurement_epochs", None)
        if epochs is None:
            epochs = {}
            self._direct_gpu_measurement_epochs = epochs
        key = (
            self.instance_id,
            source_session_id,
            pool_generation,
            transport_session,
        )
        counts = epochs.setdefault(key, [0, 0, 0])
        counts[0] += 1
        counts[1] += measurement_rows
        counts[2] += measurement_requests

    def _direct_gpu_measurement_epoch_evidence(self) -> list[dict[str, object]]:
        epochs = getattr(self, "_direct_gpu_measurement_epochs", {})
        return [
            {
                "target_instance_id": key[0],
                "source_session_id": key[1],
                "pool_generation": key[2],
                "transport_session": key[3],
                "batches": counts[0],
                "measurement_rows": counts[1],
                "measurement_requests": counts[2],
            }
            for key, counts in sorted(epochs.items())
        ]

    def stage_overlap_pivot_copy(self, batch: ScheduleBatch, batch_result) -> None:
        """Stage persistent feature paths; FlashMTP publishes pivots in VRAM."""
        if self.is_feature_backed:
            if getattr(batch_result, "dt_deferred_dflash_feature_stage", False):
                if getattr(batch_result, "dt_dflash_feature_stage_credit", False):
                    raise RuntimeError(
                        "deferred DFlash result acquired source capacity twice"
                    )
                # All ranks reserve a credit even when export staging fails;
                # the ordered result boundary broadcasts failure and releases it.
                batch_result.dt_dflash_feature_stage_credit = True
                self._deferred_dflash_feature_stages += 1
                if self.is_export_rank:
                    from sglang.srt.managers.utils import _async_d2h

                    try:
                        error = getattr(batch_result, "dt_deferred_feature_error", None)
                        if error is not None:
                            raise RuntimeError(error)
                        candidates = batch_result.dt_dflash_candidate_tokens
                        draft_input = batch_result.next_draft_input
                        if candidates is None or not candidates.is_cuda:
                            raise RuntimeError("deferred staging lost GPU token rows")
                        if getattr(batch_result, "dt_deferred_eagle3_feature_stage", False):
                            batch_result.dt_deferred_feature_offsets = tuple(
                                draft_input.dt_feature_offsets
                            )
                            batch_result.dt_deferred_feature_rounds = tuple(
                                DeferredFeatureRound(
                                    req.rid,
                                    self._next_round.get(req.rid, 0),
                                    self._ensure_lifecycle_attempt(req.rid),
                                    self._draft_kv_handle_by_request.get(req.rid),
                                    self._draft_committed_len.get(req.rid),
                                    self._verify_transfer_by_request.get(req.rid),
                                )
                                for req in batch.reqs
                            )
                            batch_result.dt_eagle3_committed_tokens_cpu = _async_d2h(
                                batch_result.dt_eagle3_committed_tokens
                            )
                        batch_result.dt_dflash_candidate_tokens_cpu = _async_d2h(candidates)
                        batch_result.dt_dflash_bonus_tokens_cpu = _async_d2h(
                            draft_input.bonus_tokens
                        )
                        batch_result.dt_dflash_seq_lens_cpu = _async_d2h(
                            draft_input.new_seq_lens
                        )
                        if not getattr(
                            batch_result, "dt_deferred_prefill_feature_stage", False
                        ):
                            # Bootstrap prefill commits every prompt row, so it
                            # has no per-request accept lengths to stage.
                            batch_result.dt_dflash_accept_lens_cpu = _async_d2h(
                                batch_result.accept_lens
                            )
                    except Exception as exc:
                        batch_result.dt_deferred_feature_error = (
                            f"{type(exc).__name__}: {exc}"
                        )
                    # Includes partial D2H submissions on failure. The scheduler
                    # also records copy_done after its ordinary result copies.
                    done = self.scheduler.device_module.Event()
                    done.record()
                    batch_result.dt_deferred_feature_copy_done = done
                return
            self._stage_dflash_feature_copy(batch, batch_result)
            return
        # Token-only AR shares bonus/sequence controls with FlashMTP. Stage
        # them before the next launch so result handling does not copy them
        # from the GPU after independent work has already been submitted.
        self._stage_flashmtp_scalar_copy(batch, batch_result)
        if not self.is_token_only:
            self._stage_flashmtp_pivot_copy(batch, batch_result)

    def _stage_flashmtp_scalar_copy(self, batch: ScheduleBatch, batch_result) -> None:
        """Preserve graph-owned control scalars on the overlap copy stream."""

        staged_bonus = getattr(batch_result, "dt_flashmtp_bonus_tokens_cpu", None)
        staged_seq_lens = getattr(batch_result, "dt_flashmtp_seq_lens_cpu", None)
        if staged_bonus is not None or staged_seq_lens is not None:
            if staged_bonus is None or staged_seq_lens is None:
                raise RuntimeError("FlashMTP scalar staging is incomplete")
            return
        draft_input = getattr(batch_result, "next_draft_input", None)
        if not isinstance(draft_input, DFlashDraftInputV2):
            return
        if not _flashmtp_pivot_preservation_indices(batch):
            return

        from sglang.srt.managers.utils import _async_d2h

        started_ns = time.monotonic_ns() if self._profile_active else 0
        batch_result.dt_flashmtp_bonus_tokens_cpu = _async_d2h(
            draft_input.bonus_tokens
        )
        batch_result.dt_flashmtp_seq_lens_cpu = _async_d2h(
            draft_input.new_seq_lens
        )
        scalar_copy_done = self.scheduler.device_module.Event()
        scalar_copy_done.record()
        self.scheduler.forward_stream.wait_event(scalar_copy_done)
        batch_result.dt_flashmtp_scalar_copy_done = scalar_copy_done
        if self._profile_active:
            batch_result.dt_flashmtp_scalar_stage_submission_ns = (
                time.monotonic_ns() - started_ns
            )

    def _stage_flashmtp_pivot_copy(self, batch: ScheduleBatch, batch_result) -> None:
        """Copy graph-owned FlashMTP pivots into leased Target VRAM.

        Overlap scheduling processes this result only after launching the next
        Target forward.  CUDA Graph outputs may therefore be replayed in place.
        Publish on the scheduler copy stream immediately after the producing
        forward so the registered source slot, rather than the graph output,
        owns the bytes while control routing remains deferred.
        """

        if getattr(batch_result, "dt_staged_flashmtp_pivots", None) is not None:
            return
        draft_input = getattr(batch_result, "next_draft_input", None)
        if not isinstance(draft_input, DFlashDraftInputV2):
            return
        if not batch.reqs:
            batch_result.dt_staged_flashmtp_pivots = StagedFlashMTPPivotBatch(
                tasks=(), descriptors_by_transfer={}
            )
            return

        profile_started_ns = time.monotonic_ns() if self._profile_active else 0
        packet = None
        published = None
        descriptors = {}
        if self.is_export_rank:
            try:
                preserved_indices = _flashmtp_pivot_preservation_indices(batch)
                specs = []
                for index in preserved_indices:
                    req = batch.reqs[index]
                    self._flashmtp_pivot_sequence += 1
                    specs.append(
                        (
                            req.rid,
                            index,
                            f"{self.instance_id}:{req.rid}:pivot:"
                            f"{self._flashmtp_pivot_sequence}",
                        )
                    )
                selection_done_ns = (
                    time.monotonic_ns() if self._profile_active else 0
                )

                if specs:
                    if self._feature_source_pool is None:
                        raise RuntimeError(
                            "FlashMTP direct GPU source pool is unavailable"
                        )
                    pivot_hidden = draft_input.hidden_states
                    if (
                        not isinstance(pivot_hidden, torch.Tensor)
                        or pivot_hidden.device.type != "cuda"
                        or pivot_hidden.dtype != torch.bfloat16
                        or pivot_hidden.ndim != 2
                        or int(pivot_hidden.shape[0]) != len(batch.reqs)
                        or int(pivot_hidden.shape[1]) != self.pivot_width
                    ):
                        raise TypeError(
                            "DT Target FlashMTP pivot must be a batch-aligned CUDA "
                            "BF16 tensor with the profile width"
                        )
                    from sglang.srt.speculative.dt_feature_p2p import (
                        FeatureTransferSlice,
                    )

                    pivot_gpu, materialized_selection = (
                        _flashmtp_pivot_publish_view(
                            pivot_hidden, preserved_indices
                        )
                    )
                    gather_submitted_ns = (
                        time.monotonic_ns() if self._profile_active else 0
                    )
                    publish_started_ns = time.monotonic_ns()
                    published = self._feature_source_pool.publish(
                        hidden_states=pivot_gpu,
                        feature_schema_id=self.feature_schema_id,
                        transfers=tuple(
                            FeatureTransferSlice(
                                transfer_id=values[2],
                                row_offset=row,
                                row_count=1,
                            )
                            for row, values in enumerate(specs)
                        ),
                        hidden_produced_ns=getattr(
                            batch_result, "dt_hidden_produced_ns", publish_started_ns
                        ),
                    )
                    publish_done_ns = (
                        time.monotonic_ns() if self._profile_active else 0
                    )
                    descriptors = {
                        descriptor.transfer_id: descriptor
                        for descriptor in published.descriptors
                    }
                    # The source may be a stable CUDA Graph output buffer.  The
                    # copy stream already waits for the producing forward; add
                    # the reverse edge after the pool copy so the next replay
                    # cannot overwrite the pivot while it is being captured.
                    self.scheduler.forward_stream.wait_event(
                        published.producer_event
                    )
                    reverse_edge_done_ns = (
                        time.monotonic_ns() if self._profile_active else 0
                    )
                    # Establish ownership before profiling or control work can
                    # fail.  Any later uncertain failure quarantines the slot.
                    for descriptor in published.descriptors:
                        self._source_slot_by_transfer[descriptor.transfer_id] = (
                            descriptor.source_slot_id,
                            descriptor.slot_generation,
                        )
                        self._staged_flashmtp_source_leases.add(
                            descriptor.transfer_id
                        )
                    ownership_done_ns = (
                        time.monotonic_ns() if self._profile_active else 0
                    )
                    for descriptor in published.descriptors:
                        self._profile_emit(
                            "target_flashmtp_pivot_publish",
                            target_instance=self.instance_id,
                            transfer_id=descriptor.transfer_id,
                            source_slot_id=descriptor.source_slot_id,
                            slot_generation=descriptor.slot_generation,
                            pivot_bytes=descriptor.nbytes,
                            transport=descriptor.transport,
                        )
                    profile_done_ns = (
                        time.monotonic_ns() if self._profile_active else 0
                    )
                    if self._profile_active:
                        self._profile_emit(
                            "target_flashmtp_pivot_batch_publish",
                            target_instance=self.instance_id,
                            batch_size=len(batch.reqs),
                            preserved_rows=len(specs),
                            materialized_selection=materialized_selection,
                            pivot_bytes=sum(
                                descriptor.nbytes
                                for descriptor in published.descriptors
                            ),
                            selection_ns=selection_done_ns - profile_started_ns,
                            gather_submission_ns=(
                                gather_submitted_ns - selection_done_ns
                            ),
                            pool_publish_ns=publish_done_ns - publish_started_ns,
                            reverse_edge_ns=(
                                reverse_edge_done_ns - publish_done_ns
                            ),
                            ownership_ns=ownership_done_ns - reverse_edge_done_ns,
                            descriptor_profile_ns=(
                                profile_done_ns - ownership_done_ns
                            ),
                            scalar_stage_submission_ns=getattr(
                                batch_result,
                                "dt_flashmtp_scalar_stage_submission_ns",
                                0,
                            ),
                            total_ns=profile_done_ns - profile_started_ns,
                            timing_scope="host_submission",
                            cuda_synchronized=False,
                        )
                    measurement_rows = sum(
                        batch.reqs[index].rid.startswith(
                            DT_BENCHMARK_MEASUREMENT_RID_PREFIX
                        )
                        for index in preserved_indices
                    )
                    if measurement_rows:
                        self._record_direct_gpu_measurement_epoch(
                            measurement_rows=measurement_rows,
                            measurement_requests=measurement_rows,
                            source_session_id=(
                                self._feature_source_pool.source_session_id
                            ),
                            pool_generation=(
                                self._feature_source_pool.pool_generation
                            ),
                            transport_session=(
                                self._feature_source_pool.transport_session
                            ),
                        )
                        if not getattr(
                            self,
                            "_direct_gpu_measurement_evidence_emitted",
                            False,
                        ):
                            self._emit_direct_gpu_measurement_evidence_once(
                                {
                                    "role": "target",
                                    "method": "flashmtp",
                                    "transport": "cuda-ipc",
                                    "measurement_rows": measurement_rows,
                                    "measurement_requests": measurement_rows,
                                    "batch_rows": len(preserved_indices),
                                    "source_session_id": (
                                        self._feature_source_pool.source_session_id
                                    ),
                                    "pool_generation": (
                                        self._feature_source_pool.pool_generation
                                    ),
                                    "transport_session": (
                                        self._feature_source_pool.transport_session
                                    ),
                                }
                            )
                packet = (specs, None)
            except Exception as exc:
                if published is not None:
                    first = published.descriptors[0]
                    self._feature_source_pool.quarantine(
                        source_slot_id=first.source_slot_id,
                        slot_generation=first.slot_generation,
                        source_session_id=first.source_session_id,
                        pool_generation=first.pool_generation,
                    )
                    for descriptor in published.descriptors:
                        self._source_slot_by_transfer.pop(descriptor.transfer_id, None)
                        self._staged_flashmtp_source_leases.discard(
                            descriptor.transfer_id
                        )
                descriptors = {}
                packet = (None, f"{type(exc).__name__}: {exc}")

        try:
            if self.scheduler.ps.tp_size > 1:
                packet = broadcast_pyobj(
                    packet,
                    self.scheduler.tp_group.rank,
                    self.scheduler.tp_cpu_group,
                    src=self.scheduler.tp_group.ranks[0],
                )
        except Exception:
            if self.is_export_rank and published is not None:
                first = published.descriptors[0]
                self._feature_source_pool.quarantine(
                    source_slot_id=first.source_slot_id,
                    slot_generation=first.slot_generation,
                    source_session_id=first.source_session_id,
                    pool_generation=first.pool_generation,
                )
                for descriptor in published.descriptors:
                    self._source_slot_by_transfer.pop(descriptor.transfer_id, None)
                    self._staged_flashmtp_source_leases.discard(
                        descriptor.transfer_id
                    )
            raise
        assert packet is not None
        specs, error = packet
        if error is not None:
            raise RuntimeError(f"FlashMTP pivot publication failed: {error}")

        staged_tasks = []
        for values in specs:
            staged = StagedFlashMTPTask(*values)
            if self.is_export_rank:
                descriptor = descriptors[staged.source_lease_id]
                if self._source_slot_by_transfer.get(staged.source_lease_id) != (
                    descriptor.source_slot_id,
                    descriptor.slot_generation,
                ):
                    raise RuntimeError("FlashMTP staged source lease was lost")
            staged_tasks.append(staged)
        batch_result.dt_staged_flashmtp_pivots = StagedFlashMTPPivotBatch(
            tasks=tuple(staged_tasks), descriptors_by_transfer=descriptors
        )

    def _cancel_staged_flashmtp_pivots(
        self, staged: StagedFlashMTPPivotBatch, batch_result
    ) -> None:
        """Release FlashMTP source leases that never reached control routing."""

        for task in staged.tasks:
            self._parked.pop(task.source_lease_id, None)
            if self.is_export_rank:
                self._release_source_slot(task.source_lease_id)
        batch_result.dt_staged_flashmtp_pivots = None

    def _reserve_deferred_hidden(self, packed_hidden: torch.Tensor):
        """Write a verify wave's hidden rows into a leased source slot (TP0).

        Runs on the forward stream inside the verify, replacing the defensive
        clone.  Returns the slot rows, or ``None`` to fall back to a clone.
        """
        pool = self._feature_source_pool
        if (
            pool is None
            or packed_hidden.ndim != 2
            or packed_hidden.dtype is not pool.dtype
            or int(packed_hidden.shape[1]) != pool.feature_width
            or not packed_hidden.is_contiguous()
        ):
            return None
        reservation = pool.reserve(int(packed_hidden.shape[0]))
        if reservation is None:
            return None
        slot_id, generation, rows = reservation
        try:
            rows.copy_(packed_hidden)
            if getattr(self, "_ring_publish", False):
                events = getattr(self, "_slot_ready_events", None)
                if events is None:
                    events = self._slot_ready_events = {}
                ready_event = events.get(slot_id)
                if ready_event is None:
                    ready_event = events[slot_id] = torch.cuda.Event(
                        interprocess=True
                    )
            else:
                ready_event = torch.cuda.Event(interprocess=True)
            ready_event.record()
        except Exception:
            pool.quarantine_reserved(slot_id, generation)
            raise
        self._inplace_hidden[rows.data_ptr()] = (slot_id, generation, ready_event)
        return rows

    def _release_inplace_hidden(self, batch_result) -> None:
        """Return an in-place reservation that was not published."""
        packed = getattr(batch_result, "dt_dflash_packed_hidden", None)
        inplace = getattr(self, "_inplace_hidden", None)
        if packed is None or not inplace:
            return
        reservation = inplace.pop(packed.data_ptr(), None)
        if reservation is None:
            return
        slot_id, generation, ready_event = reservation
        # The deferred stage already waited for this wave's copies; this only
        # proves the in-place write itself finished before reuse.
        ready_event.synchronize()
        self._feature_source_pool.release_reserved(slot_id, generation)

    def _stage_dflash_feature_copy(
        self, batch: ScheduleBatch, batch_result, *, finalize_deferred: bool = False
    ) -> None:
        if getattr(batch_result, "dt_staged_dflash_features", None) is not None:
            return
        release_credit = bool(
            finalize_deferred
            and getattr(batch_result, "dt_deferred_dflash_feature_stage", False)
        )
        if release_credit and not getattr(
            batch_result, "dt_dflash_feature_stage_credit", False
        ):
            raise RuntimeError("deferred DFlash result has no reserved source capacity")
        try:
            self._stage_dflash_feature_copy_impl(
                batch, batch_result, finalize_deferred=finalize_deferred
            )
        finally:
            if release_credit:
                self._release_inplace_hidden(batch_result)
                for name in (
                    "dt_dflash_packed_hidden", "dt_dflash_candidate_tokens",
                    "dt_dflash_candidate_tokens_cpu", "dt_dflash_bonus_tokens_cpu",
                    "dt_dflash_seq_lens_cpu", "dt_dflash_accept_lens_cpu",
                    "dt_eagle3_committed_tokens", "dt_eagle3_committed_tokens_cpu",
                    "dt_deferred_feature_rounds", "dt_deferred_feature_offsets",
                    "dt_deferred_feature_copy_done",
                    "dt_deferred_prefill_feature_stage",
                    "dt_deferred_prefill_extend_lens",
                ):
                    setattr(batch_result, name, None)
                batch_result.dt_dflash_feature_stage_credit = False
                self._deferred_dflash_feature_stages -= 1
                if self._deferred_dflash_feature_stages < 0:
                    raise RuntimeError(
                        "deferred DFlash source-capacity accounting underflowed"
                    )

    def _stage_dflash_feature_copy_impl(
        self, batch: ScheduleBatch, batch_result, *, finalize_deferred: bool = False
    ) -> None:
        draft_input = getattr(batch_result, "next_draft_input", None)
        if not isinstance(draft_input, DFlashDraftInputV2):
            return
        feature_chunks = draft_input.dt_feature_chunks
        token_chunks = draft_input.dt_feature_token_chunks
        committed_chunks = draft_input.dt_committed_token_chunks
        feature_offsets = draft_input.dt_feature_offsets
        deferred = getattr(batch_result, "dt_deferred_dflash_feature_stage", False)
        phase_started_ns = time.monotonic_ns()
        if deferred and not finalize_deferred:
            return
        if feature_chunks is None and not batch.reqs:
            done = getattr(batch_result, "dt_deferred_feature_copy_done", None)
            if done is not None:
                done.synchronize()
            return

        packet = None
        published = None
        descriptors = {}
        deferred_prefill = deferred and getattr(
            batch_result, "dt_deferred_prefill_feature_stage", False
        )
        # With DT_RING_PUBLISH on, deferred verify waves carry no per-wave
        # broadcast: every rank derives identical commit specs from its own
        # batch state and ledger dicts; only the export rank publishes.
        # Every other wave class keeps the broadcast.  The no-broadcast path
        # is restricted to same-host direct-ring operation, the configuration
        # it was designed and validated for.
        local_verify = (
            deferred
            and not deferred_prefill
            and not self.is_eagle3
            and getattr(self, "_ring_publish", False)
            and self.is_persistent
            and self.candidate_transport == "cuda-ipc"
            and self._direct_ring
            and getattr(batch_result, "accept_lens", None) is not None
            and getattr(draft_input, "bonus_tokens", None) is not None
            and draft_input.bonus_tokens.is_cuda
            and getattr(draft_input, "new_seq_lens", None) is not None
            and draft_input.new_seq_lens.is_cuda
        )
        if (
            deferred
            and not deferred_prefill
            and not local_verify
            and not getattr(self, "_ring_publish_blocked_logged", False)
        ):
            self._ring_publish_blocked_logged = True
            logger.warning(
                "ring publish local_verify blocked: eagle3=%s flag=%s "
                "accept_lens=%s bonus=%s seq_lens=%s",
                self.is_eagle3,
                getattr(self, "_ring_publish", False),
                getattr(getattr(batch_result, "accept_lens", None), "device", None),
                getattr(getattr(draft_input, "bonus_tokens", None), "device", None),
                getattr(getattr(draft_input, "new_seq_lens", None), "device", None),
            )
        export_block = self.is_export_rank
        if local_verify:
            derive_started_ns = time.monotonic_ns()
            try:
                accept_values, bonus_values, seq_len_values = (
                    self._dflash_wave_scalars(batch_result, draft_input, deferred=True)
                )
                counts = accept_values
                _active, staged_meta, specs_local = self._dflash_staged_meta_and_specs(
                    batch, feature_offsets, counts, accept_values, bonus_values,
                    seq_len_values,
                )
            except Exception as exc:
                # The per-wave broadcast is gone, so a derivation failure
                # must publish its own fatality: the other ranks raise it at
                # the next poll instead of committing a poisoned wave.
                if self.is_export_rank:
                    self._ring_publish_fatal = (
                        f"DFLASH deferred staging failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                raise
            batch_result.dt_dflash_feature_metadata_ns = (
                time.monotonic_ns() - derive_started_ns
            )
            if self.is_export_rank:
                try:
                    ring_ok = self._stage_dflash_ring_publish(
                        batch,
                        batch_result,
                        draft_input,
                        feature_chunks,
                        token_chunks,
                        deferred=deferred,
                        staged_meta=staged_meta,
                        accept_values=accept_values,
                        bonus_values=bonus_values,
                        seq_len_values=seq_len_values,
                        commit_specs=specs_local,
                    )
                except RingPublishCommittedError as exc:
                    # The record is already on the ring; a Python retry would
                    # double-publish the wave.  Fail the wave closed instead.
                    ring_ok = None
                    packet = (None, f"RingPublishCommittedError: {exc}", False)
                except Exception:
                    logger.exception(
                        "DT ring publish failed; using the Python publication path"
                    )
                    ring_ok = None
                if ring_ok:
                    packet = (specs_local, None, True)
                    export_block = False
            else:
                packet = (specs_local, None, False)
                export_block = False
        elif (            self.is_export_rank
            and getattr(self, "_ring_publish", False)
            and not deferred
            and not self.is_eagle3
            and feature_chunks is not None
            and getattr(batch_result, "accept_lens", None) is not None
            and batch_result.accept_lens.is_cuda
            and getattr(draft_input, "bonus_tokens", None) is not None
            and draft_input.bonus_tokens.is_cuda
            and getattr(draft_input, "new_seq_lens", None) is not None
            and draft_input.new_seq_lens.is_cuda
        ):
            # Non-deferred configs still stage through the broadcast path.
            accept_values, bonus_values, seq_len_values = self._dflash_wave_scalars(
                batch_result, draft_input, deferred=False
            )
            counts = accept_values
            _active, staged_meta = self._dflash_staged_meta(
                batch, feature_offsets, counts
            )
            specs_local = self._dflash_commit_specs(
                staged_meta, accept_values, bonus_values, seq_len_values
            )
            try:
                ring_ok = self._stage_dflash_ring_publish(
                    batch,
                    batch_result,
                    draft_input,
                    feature_chunks,
                    token_chunks,
                    deferred=deferred,
                    staged_meta=staged_meta,
                    accept_values=accept_values,
                    bonus_values=bonus_values,
                    seq_len_values=seq_len_values,
                    commit_specs=specs_local,
                )
            except RingPublishCommittedError as exc:
                ring_ok = None
                packet = (None, f"RingPublishCommittedError: {exc}", False)
            except Exception:
                logger.exception(
                    "DT ring publish failed; using the Python publication path"
                )
                ring_ok = None
            if ring_ok:
                packet = (specs_local, None, True)
                export_block = False
        if export_block and packet is None and self.is_export_rank:
            try:
                if deferred:
                    error = getattr(batch_result, "dt_deferred_feature_error", None)
                    done = getattr(batch_result, "dt_deferred_feature_copy_done", None)
                    if done is not None:
                        done.synchronize()
                    if error is not None:
                        raise RuntimeError(error)
                    from sglang.srt.speculative.dflash_worker_v2 import (
                        split_dflash_committed_rows,
                    )

                    packed_hidden = batch_result.dt_dflash_packed_hidden
                    candidate_tokens = batch_result.dt_dflash_candidate_tokens_cpu
                    if packed_hidden is None or candidate_tokens is None:
                        raise RuntimeError("deferred result lost feature staging tensors")
                    if packed_hidden.device.type != "cuda" or candidate_tokens.is_cuda:
                        raise RuntimeError("deferred tensors crossed wrong staging boundary")
                    if getattr(
                        batch_result, "dt_deferred_prefill_feature_stage", False
                    ):
                        # Bootstrap prefill commits every prompt row; split the
                        # preserved prompt by its launch-time extend layout
                        # instead of the verify block/accept-lens layout.
                        extend_lens = getattr(
                            batch_result, "dt_deferred_prefill_extend_lens", None
                        )
                        if (
                            extend_lens is None
                            or len(extend_lens) != len(batch.reqs)
                            or int(sum(extend_lens)) != int(packed_hidden.shape[0])
                            or int(candidate_tokens.numel())
                            != int(packed_hidden.shape[0])
                        ):
                            raise RuntimeError(
                                "deferred bootstrap result lost its extend layout"
                            )
                        feature_chunks = list(
                            torch.split(packed_hidden, extend_lens, dim=0)
                        )
                        token_chunks = list(
                            torch.split(candidate_tokens, extend_lens, dim=0)
                        )
                    else:
                        feature_chunks, token_chunks = split_dflash_committed_rows(
                            packed_hidden=packed_hidden,
                            candidates=candidate_tokens,
                            commit_lens=batch_result.dt_dflash_accept_lens_cpu,
                            block_size=self.block_size,
                        )
                    if self.is_eagle3:
                        feature_offsets = batch_result.dt_deferred_feature_offsets
                        committed = batch_result.dt_eagle3_committed_tokens_cpu
                        if committed.is_cuda or committed.shape != candidate_tokens.shape:
                            raise RuntimeError("EAGLE3 committed token schema mismatch")
                        committed_chunks = [
                            committed[index, : int(hidden.shape[0])]
                            for index, hidden in enumerate(feature_chunks)
                        ]
                    # Keep views local: GPU ownership ends after pool publication,
                    # rather than leaking cloned payloads into the next Draft input.
                batch_result.dt_dflash_feature_split_ns = (
                    time.monotonic_ns() - phase_started_ns
                )
                phase_started_ns = time.monotonic_ns()
                if (
                    feature_chunks is None
                    or token_chunks is None
                    or feature_offsets is None
                    or len(feature_chunks) != len(batch.reqs)
                    or len(token_chunks) != len(batch.reqs)
                    or len(feature_offsets) != len(batch.reqs)
                    or (
                        self.is_eagle3
                        and (
                            committed_chunks is None
                            or len(committed_chunks) != len(batch.reqs)
                        )
                    )
                ):
                    raise RuntimeError("DFLASH Target result is missing feature chunks")
                active = [
                    index
                    for index, req in enumerate(batch.reqs)
                    if not req.finished()
                    and not req.is_retracted
                    and int(feature_chunks[index].shape[0]) > 0
                ]
                if deferred:
                    bonus_values = batch_result.dt_dflash_bonus_tokens_cpu.tolist()
                    seq_len_values = batch_result.dt_dflash_seq_lens_cpu.tolist()
                else:
                    bonus_values = draft_input.bonus_tokens.detach().to("cpu").tolist()
                    seq_len_values = (
                        draft_input.new_seq_lens.detach().to("cpu").tolist()
                    )
                eagle_tokens_by_index = {}
                eagle_committed_by_index = {}
                if self.is_eagle3:
                    assert committed_chunks is not None
                    (
                        eagle_tokens_by_index,
                        eagle_committed_by_index,
                    ) = _batched_eagle3_token_metadata(
                        active_indices=active,
                        token_chunks=token_chunks,
                        committed_chunks=committed_chunks,
                    )
                specs = []
                transfers = []
                hidden_rows = []
                row_offset = 0
                for index in active:
                    req = batch.reqs[index]
                    snapshot = None
                    if deferred and self.is_eagle3:
                        snapshots = batch_result.dt_deferred_feature_rounds
                        if len(snapshots) != len(batch.reqs):
                            raise RuntimeError("deferred feature request count changed")
                        snapshot = snapshots[index]
                        current = DeferredFeatureRound(
                            req.rid,
                            self._next_round.get(req.rid, 0),
                            self._lifecycle_attempt_by_request.get(req.rid),
                            self._draft_kv_handle_by_request.get(req.rid),
                            self._draft_committed_len.get(req.rid),
                            self._verify_transfer_by_request.get(req.rid),
                        )
                        if current != snapshot:
                            raise RuntimeError("deferred feature round fencing changed")
                    attempt = (
                        snapshot.attempt if snapshot is not None
                        else self._ensure_lifecycle_attempt(req.rid)
                    )
                    hidden = feature_chunks[index]
                    tokens = token_chunks[index]
                    committed_tokens = (
                        committed_chunks[index] if self.is_eagle3 else tokens
                    )
                    if (
                        hidden.ndim != 2
                        or int(hidden.shape[1]) != self.pivot_width
                        or hidden.dtype != torch.bfloat16
                        or tokens.ndim != 1
                        or int(tokens.numel()) != int(hidden.shape[0])
                        or committed_tokens.ndim != 1
                        or int(committed_tokens.numel()) != int(hidden.shape[0])
                    ):
                        raise RuntimeError(
                            "DFLASH Target feature/token chunk shape mismatch"
                        )
                    offset = int(feature_offsets[index])
                    token_count = int(hidden.shape[0])
                    round_id = (
                        snapshot.round_id if snapshot is not None
                        else self._next_round.get(req.rid, 0)
                    )
                    previous_committed = (
                        snapshot.committed_len if snapshot is not None
                        else self._draft_committed_len.get(req.rid)
                    )
                    if round_id == 0:
                        if offset != 0 or previous_committed is not None:
                            raise RuntimeError(
                                "DFLASH bootstrap must begin at offset zero"
                            )
                        draft_kv_handle = f"{self.instance_id}:{req.rid}:" + str(
                            attempt
                        )
                    else:
                        draft_kv_handle = (
                            snapshot.draft_kv_handle if snapshot is not None
                            else self._draft_kv_handle_by_request.get(req.rid)
                        )
                        if draft_kv_handle is None or previous_committed != offset:
                            raise RuntimeError(
                                "DFLASH feature range has a committed-prefix gap"
                            )
                    transfer_id = f"{self.instance_id}:{req.rid}:{round_id}:{attempt}"
                    if self.is_eagle3:
                        token_ids = eagle_tokens_by_index[index]
                        committed_token_ids = eagle_committed_by_index[index]
                    else:
                        token_ids = tuple(
                            int(value)
                            for value in tokens.detach().to("cpu").tolist()
                        )
                        committed_token_ids = tuple(
                            int(value)
                            for value in committed_tokens.detach().to("cpu").tolist()
                        )
                    specs.append(
                        (
                            req.rid,
                            transfer_id,
                            round_id,
                            attempt,
                            draft_kv_handle,
                            offset,
                            token_count,
                            token_ids,
                            committed_token_ids,
                            (snapshot.previous_transfer_id if snapshot is not None
                             else self._verify_transfer_by_request.get(req.rid)),
                            int(bonus_values[index]),
                            int(seq_len_values[index]),
                        )
                    )
                    if deferred:
                        # Clones were allocated on forward_stream; publication
                        # reads them on schedule_stream after result completion.
                        hidden.record_stream(torch.cuda.current_stream(hidden.device))
                    hidden_rows.append(hidden.contiguous())
                    transfers.append((transfer_id, row_offset, token_count))
                    row_offset += token_count
                batch_result.dt_dflash_feature_metadata_ns = (
                    time.monotonic_ns() - phase_started_ns
                )

                if specs:
                    assert self._feature_source_pool is not None
                    from sglang.srt.speculative.dt_feature_p2p import (
                        FeatureTransferSlice,
                    )

                    phase_started_ns = time.monotonic_ns()
                    # Deferred rows are clones whose completion the host has
                    # already observed. Publishing them on schedule_stream would
                    # queue the copy behind the next verify (WAR barrier), so
                    # the Draft pull would wait for that whole forward.
                    publish_stream = None
                    if (
                        deferred
                        and hidden_rows
                        and torch.cuda.is_available()
                        and all(row.is_contiguous() for row in hidden_rows)
                    ):
                        publish_stream = getattr(self, "_feature_publish_stream", None)
                        if publish_stream is None:
                            publish_stream = torch.cuda.Stream(
                                device=hidden_rows[0].device
                            )
                            self._feature_publish_stream = publish_stream
                    packed = (
                        getattr(batch_result, "dt_dflash_packed_hidden", None)
                        if deferred
                        else None
                    )
                    inplace = getattr(self, "_inplace_hidden", None) or {}
                    reservation = (
                        None if packed is None else inplace.get(packed.data_ptr())
                    )
                    if reservation is not None:
                        # Rows already live in the leased slot; describe them
                        # where the verify wrote them.
                        slot_id, generation, ready_event = reservation
                        row_bytes = self.pivot_width * packed.element_size()
                        published = self._feature_source_pool.publish_reserved(
                            slot_id=slot_id,
                            generation=generation,
                            token_count=int(packed.shape[0]),
                            feature_schema_id=self.feature_schema_id,
                            transfers=tuple(
                                FeatureTransferSlice(
                                    transfer_id=transfer_id,
                                    row_offset=(
                                        row.data_ptr() - packed.data_ptr()
                                    ) // row_bytes,
                                    row_count=row_count,
                                )
                                for (transfer_id, _offset, row_count), row in zip(
                                    transfers, hidden_rows, strict=True
                                )
                            ),
                            producer_event=ready_event,
                            hidden_produced_ns=getattr(
                                batch_result,
                                "dt_hidden_produced_ns",
                                time.monotonic_ns(),
                            ),
                        )
                        inplace.pop(packed.data_ptr(), None)
                    else:
                        published = self._feature_source_pool.publish(
                            stream=publish_stream,
                            hidden_states=tuple(hidden_rows),
                            feature_schema_id=self.feature_schema_id,
                            transfers=tuple(
                                FeatureTransferSlice(
                                    transfer_id=transfer_id,
                                    row_offset=row_offset,
                                    row_count=row_count,
                                )
                                for transfer_id, row_offset, row_count in transfers
                            ),
                            hidden_produced_ns=getattr(
                                batch_result,
                                "dt_hidden_produced_ns",
                                time.monotonic_ns(),
                            ),
                        )
                    batch_result.dt_dflash_feature_publish_strategy = (
                        published.pack_strategy
                    )
                    batch_result.dt_dflash_feature_publish_chunks = (
                        published.pack_chunk_count
                    )
                    # These rows may still alias a stable CUDA Graph output
                    # buffer: ``contiguous()`` preserves an already-contiguous
                    # view.  Scheduler has made the publication copy stream
                    # wait for the producing forward.  Add the reverse edge
                    # after the pool copy so a subsequent graph replay on the
                    # forward stream cannot overwrite EAGLE3/DFlash features
                    # while publication is still reading them.
                    self.scheduler.forward_stream.wait_event(
                        published.producer_event
                    )
                    batch_result.dt_dflash_feature_publish_ns = (
                        time.monotonic_ns() - phase_started_ns
                    )
                    batch_result.dt_dflash_published_descriptor_count = len(
                        published.descriptors
                    )
                    batch_result.dt_dflash_published_feature_rows = row_offset
                    batch_result.dt_dflash_published_feature_bytes = (
                        row_offset * self.pivot_width * 2
                    )
                    batch_result.dt_dflash_descriptor_published_ns = (
                        published.descriptor_published_ns
                    )
                    descriptors = {
                        descriptor.transfer_id: descriptor
                        for descriptor in published.descriptors
                    }
                    # Establish controller ownership before any profiling or
                    # control-plane work can fail.  A later exception can now
                    # deterministically quarantine every undelivered range.
                    phase_started_ns = time.monotonic_ns()
                    for descriptor in published.descriptors:
                        self._source_slot_by_transfer[descriptor.transfer_id] = (
                            descriptor.source_slot_id,
                            descriptor.slot_generation,
                        )
                    profile_publish_batch = profile_event_enabled(
                        "target_feature_publish_batch"
                    )
                    profile_publish_transfer = profile_event_enabled(
                        "target_feature_publish"
                    )
                    if profile_publish_batch or profile_publish_transfer:
                        specs_by_transfer = {
                            values[1]: values for values in specs
                        }
                    if profile_publish_batch:
                        self._profile_emit(
                            "target_feature_publish_batch",
                            target_instance=self.instance_id,
                            transfer_ids=tuple(
                                descriptor.transfer_id
                                for descriptor in published.descriptors
                            ),
                            request_ids=tuple(
                                specs_by_transfer[descriptor.transfer_id][0]
                                for descriptor in published.descriptors
                            ),
                            round_ids=tuple(
                                specs_by_transfer[descriptor.transfer_id][2]
                                for descriptor in published.descriptors
                            ),
                            feature_rows=tuple(
                                specs_by_transfer[descriptor.transfer_id][6]
                                for descriptor in published.descriptors
                            ),
                            publication_strategy=published.pack_strategy,
                            publication_chunk_count=(
                                published.pack_chunk_count
                            ),
                            publication_descriptor_count=len(
                                published.descriptors
                            ),
                            publication_rows=row_offset,
                            publication_bytes=(
                                row_offset * self.pivot_width * 2
                            ),
                            hidden_produced_ns=published.hidden_produced_ns,
                            descriptor_published_ns=(
                                published.descriptor_published_ns
                            ),
                        )
                    if profile_publish_transfer:
                        for descriptor in published.descriptors:
                            spec = specs_by_transfer[descriptor.transfer_id]
                            self._profile_emit(
                                "target_feature_publish",
                                target_instance=self.instance_id,
                                transfer_id=descriptor.transfer_id,
                                request_id=spec[0],
                                round_id=spec[2],
                                feature_rows=spec[6],
                                source_slot_id=descriptor.source_slot_id,
                                slot_generation=descriptor.slot_generation,
                                feature_bytes=descriptor.nbytes,
                                publication_strategy=published.pack_strategy,
                                publication_chunk_count=(
                                    published.pack_chunk_count
                                ),
                                publication_descriptor_count=len(
                                    published.descriptors
                                ),
                                publication_rows=row_offset,
                                publication_bytes=(
                                    row_offset * self.pivot_width * 2
                                ),
                                hidden_produced_ns=published.hidden_produced_ns,
                                descriptor_published_ns=(
                                    published.descriptor_published_ns
                                ),
                            )
                    measurement_specs = [
                        values
                        for values in specs
                        if values[0].startswith(
                            DT_BENCHMARK_MEASUREMENT_RID_PREFIX
                        )
                    ]
                    if measurement_specs:
                        first_descriptor = published.descriptors[0]
                        measurement_rows = sum(
                            int(values[6]) for values in measurement_specs
                        )
                        self._record_direct_gpu_measurement_epoch(
                            measurement_rows=measurement_rows,
                            measurement_requests=len(measurement_specs),
                            source_session_id=first_descriptor.source_session_id,
                            pool_generation=first_descriptor.pool_generation,
                            transport_session=first_descriptor.transport_session,
                        )
                        if not getattr(
                            self,
                            "_direct_gpu_measurement_evidence_emitted",
                            False,
                        ):
                            self._emit_direct_gpu_measurement_evidence_once(
                                {
                                    "role": "target",
                                    "method": (
                                        "eagle3" if self.is_eagle3 else "dflash"
                                    ),
                                    "transport": first_descriptor.transport,
                                    "measurement_rows": measurement_rows,
                                    "measurement_requests": len(measurement_specs),
                                    "batch_rows": row_offset,
                                    "source_session_id": (
                                        first_descriptor.source_session_id
                                    ),
                                    "pool_generation": (
                                        first_descriptor.pool_generation
                                    ),
                                    "transport_session": (
                                        first_descriptor.transport_session
                                    ),
                                }
                            )
                    batch_result.dt_dflash_feature_profile_ns = (
                        time.monotonic_ns() - phase_started_ns
                    )
                packet = (specs, None, False)
            except Exception as exc:
                if published is not None:
                    first_descriptor = published.descriptors[0]
                    self._feature_source_pool.quarantine(
                        source_slot_id=first_descriptor.source_slot_id,
                        slot_generation=first_descriptor.slot_generation,
                        source_session_id=first_descriptor.source_session_id,
                        pool_generation=first_descriptor.pool_generation,
                    )
                    for descriptor in published.descriptors:
                        self._source_slot_by_transfer.pop(descriptor.transfer_id, None)
                    published = None
                packet = (
                    None,
                    f"{type(exc).__name__}: {exc}"
                    + self._feature_source_pool_diagnostics(),
                    False,
                )
                descriptors = {}
        phase_started_ns = time.monotonic_ns()
        try:
            if self.scheduler.ps.tp_size > 1 and not local_verify:
                packet = broadcast_pyobj(
                    packet,
                    self.scheduler.tp_group.rank,
                    self.scheduler.tp_cpu_group,
                    src=self.scheduler.tp_group.ranks[0],
                )
        except Exception:
            if self.is_export_rank and published is not None:
                first_descriptor = published.descriptors[0]
                self._feature_source_pool.quarantine(
                    source_slot_id=first_descriptor.source_slot_id,
                    slot_generation=first_descriptor.slot_generation,
                    source_session_id=first_descriptor.source_session_id,
                    pool_generation=first_descriptor.pool_generation,
                )
                for descriptor in published.descriptors:
                    self._source_slot_by_transfer.pop(descriptor.transfer_id, None)
            raise
        batch_result.dt_dflash_feature_broadcast_ns = (
            time.monotonic_ns() - phase_started_ns
        )
        assert packet is not None
        specs, error, ring_publish = packet
        if error is not None:
            if self.is_export_rank:
                self._ring_publish_fatal = (
                    f"DFLASH feature publication failed: {error}"
                )
            raise RuntimeError(f"DFLASH feature publication failed: {error}")

        phase_started_ns = time.monotonic_ns()
        staged_tasks = []
        for values in specs:
            staged = StagedDFlashTask(*values)
            self._next_round[staged.request_id] = staged.round_id + 1
            self._draft_committed_len[staged.request_id] = (
                staged.feature_token_offset + staged.feature_token_count
            )
            self._draft_kv_handle_by_request[staged.request_id] = staged.draft_kv_handle
            current_attempt = self._lifecycle_attempt_by_request.get(staged.request_id)
            if current_attempt is None:
                self._lifecycle_attempt_by_request[staged.request_id] = staged.attempt
                self._next_lifecycle_attempt = max(
                    self._next_lifecycle_attempt, staged.attempt + 1
                )
            elif current_attempt != staged.attempt:
                raise RuntimeError("DFLASH lifecycle attempt changed during staging")
            if self.is_export_rank and not ring_publish:
                # Ring-published waves register their lease in the deferred
                # cleanup, once the descriptors exist.
                descriptor = descriptors[staged.transfer_id]
                if self._source_slot_by_transfer.get(staged.transfer_id) != (
                    descriptor.source_slot_id,
                    descriptor.slot_generation,
                ):
                    raise RuntimeError("DFLASH staged source lease was lost")
            staged_tasks.append(staged)
        batch_result.dt_staged_dflash_features = StagedDFlashFeatureBatch(
            tasks=tuple(staged_tasks),
            descriptors_by_transfer=descriptors,
            ring_publish=ring_publish,
        )
        batch_result.dt_dflash_feature_commit_ns = (
            time.monotonic_ns() - phase_started_ns
        )

    def park_completed_batch(self, batch: ScheduleBatch, *, batch_result=None) -> None:
        started_ns = time.monotonic_ns()
        batch_size = len(batch.reqs)
        park_succeeded = False
        if self.eagle3_host_span_profile_enabled and batch_result is not None:
            batch_result.dt_eagle3_park_entry_ns = started_ns
        try:
            self._record_target_verify_gpu_breakdown(batch_result, batch_size)
            self._record_eagle3_verify_gpu_breakdown(batch_result, batch_size)
            self._record_measurement_lane_execution(batch, batch_result)
            self._park_completed_batch(batch, batch_result=batch_result)
            park_succeeded = True
            self._record_full_wave_completion(
                batch, batch_result=batch_result, batch_size=batch_size
            )
            if (
                self.eagle3_host_span_profile_enabled
                and batch_result is not None
                and getattr(batch_result, "dt_eagle3_completed_ns", None) is None
            ):
                batch_result.dt_eagle3_completed_ns = time.monotonic_ns()
            if park_succeeded:
                self._record_eagle3_host_span(batch_result, batch_size)
        finally:
            if self.is_export_rank and batch_size:
                self._profile_emit(
                    "target_park_batch",
                    batch_size=batch_size,
                    wave_id=getattr(batch_result, "dt_full_wave_id", None),
                    lane_id=getattr(batch_result, "dt_pipeline_lane_id", None),
                    park_succeeded=park_succeeded,
                    total_ns=time.monotonic_ns() - started_ns,
                    feature_stage_ns=getattr(
                        batch_result, "dt_dflash_feature_stage_ns", 0
                    ),
                    feature_split_ns=getattr(
                        batch_result, "dt_dflash_feature_split_ns", 0
                    ),
                    feature_metadata_ns=getattr(
                        batch_result, "dt_dflash_feature_metadata_ns", 0
                    ),
                    feature_publish_ns=getattr(
                        batch_result, "dt_dflash_feature_publish_ns", 0
                    ),
                    ring_publish_breakdown=getattr(
                        batch_result, "dt_ring_publish_breakdown", None
                    ),
                    feature_publish_strategy=getattr(
                        batch_result,
                        "dt_dflash_feature_publish_strategy",
                        "none",
                    ),
                    feature_publish_chunks=getattr(
                        batch_result, "dt_dflash_feature_publish_chunks", 0
                    ),
                    feature_profile_ns=getattr(
                        batch_result, "dt_dflash_feature_profile_ns", 0
                    ),
                    feature_broadcast_ns=getattr(
                        batch_result, "dt_dflash_feature_broadcast_ns", 0
                    ),
                    feature_commit_ns=getattr(
                        batch_result, "dt_dflash_feature_commit_ns", 0
                    ),
                    published_descriptor_count=getattr(
                        batch_result, "dt_dflash_published_descriptor_count", 0
                    ),
                    published_feature_rows=getattr(
                        batch_result, "dt_dflash_published_feature_rows", 0
                    ),
                    published_feature_bytes=getattr(
                        batch_result, "dt_dflash_published_feature_bytes", 0
                    ),
                    descriptor_published_ns=getattr(
                        batch_result, "dt_dflash_descriptor_published_ns", 0
                    ),
                    request_index_ns=getattr(
                        batch_result, "dt_dflash_request_index_ns", 0
                    ),
                    retire_ns=getattr(batch_result, "dt_dflash_retire_ns", 0),
                    task_build_ns=getattr(batch_result, "dt_dflash_task_build_ns", 0),
                    queue_flush_ns=getattr(batch_result, "dt_dflash_queue_flush_ns", 0),
                    release_ns=getattr(batch_result, "dt_dflash_release_ns", 0),
                    flashmtp_release_inflight_ns=getattr(
                        batch_result, "dt_flashmtp_release_inflight_ns", 0
                    ),
                    flashmtp_retire_ns=getattr(
                        batch_result, "dt_flashmtp_retire_ns", 0
                    ),
                    flashmtp_select_ns=getattr(
                        batch_result, "dt_flashmtp_select_ns", 0
                    ),
                    flashmtp_scalar_copy_ns=getattr(
                        batch_result, "dt_flashmtp_scalar_copy_ns", 0
                    ),
                    flashmtp_task_build_ns=getattr(
                        batch_result, "dt_flashmtp_task_build_ns", 0
                    ),
                    flashmtp_queue_flush_ns=getattr(
                        batch_result, "dt_flashmtp_queue_flush_ns", 0
                    ),
                    flashmtp_finalize_ns=getattr(
                        batch_result, "dt_flashmtp_finalize_ns", 0
                    ),
                    parked=len(self._parked),
                )

    def _record_eagle3_host_span(self, batch_result, batch_size: int) -> None:
        """Emit one TP0 host chronology row for every successful EAGLE3 batch.

        The timestamps are control/enqueue boundaries.  They deliberately do
        not claim to partition CUDA execution; the CUDA corridor is an overlay
        with no absolute host-clock alignment.
        """

        if not self.eagle3_host_span_profile_enabled or batch_result is None:
            return

        wave_id = getattr(batch_result, "dt_full_wave_id", None)
        is_controller_verify = wave_id is not None
        common_names = (
            "selection_returned_ns",
            "run_batch_entry_ns",
            "before_worker_call_ns",
            "worker_entry_ns",
            "worker_return_ns",
            "after_worker_call_ns",
            "copy_submit_started_ns",
            "copy_submit_returned_ns",
            "run_batch_return_ns",
            "result_processing_entry_ns",
            "copy_ready_ns",
            "park_entry_ns",
            "completed_ns",
        )
        common = {
            name: getattr(batch_result, f"dt_eagle3_{name}", None)
            for name in common_names
        }
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in common.values()
        ):
            raise RuntimeError("EAGLE3 host-span batch boundaries are incomplete")
        common_order = tuple(common[name] for name in common_names)
        if any(right < left for left, right in zip(common_order, common_order[1:])):
            raise RuntimeError("EAGLE3 host-span batch boundaries are out of order")

        identity = {
            "record_kind": (
                "controller_verify" if is_controller_verify else "other_batch"
            ),
            "forward_iter": getattr(batch_result, "dt_eagle3_forward_iter", None),
            "scheduler_forward_mode": getattr(
                batch_result, "dt_eagle3_scheduler_forward_mode", None
            ),
            "scheduler_forward_mode_after_worker": getattr(
                batch_result,
                "dt_eagle3_scheduler_forward_mode_after_worker",
                None,
            ),
            "worker_branch": getattr(
                batch_result, "dt_eagle3_worker_branch", None
            ),
            "model_forward_mode": getattr(
                batch_result, "dt_eagle3_model_forward_mode", None
            ),
            "launch_kind": getattr(batch_result, "dt_eagle3_launch_kind", None),
            "batch_size": batch_size,
            "is_extend": getattr(batch_result, "dt_eagle3_is_extend", None),
            "can_run_cuda_graph": bool(batch_result.can_run_cuda_graph),
            "overlap_predecessor_forward_iter": getattr(
                batch_result, "dt_eagle3_overlap_predecessor_forward_iter", None
            ),
            "overlap_predecessor_scheduler_forward_mode": getattr(
                batch_result,
                "dt_eagle3_overlap_predecessor_scheduler_forward_mode",
                None,
            ),
            "overlap_predecessor_launch_kind": getattr(
                batch_result, "dt_eagle3_overlap_predecessor_launch_kind", None
            ),
            **common,
        }
        request_ids = getattr(batch_result, "dt_eagle3_request_ids", None)
        if request_ids is None and getattr(
            batch_result, "dt_eagle3_host_span_active", False
        ):
            raise RuntimeError("EAGLE3 host-span request identity is missing")
        if request_ids is not None:
            if (
                not isinstance(request_ids, tuple)
                or len(request_ids) != batch_size
                or any(not isinstance(rid, str) or not rid for rid in request_ids)
                or len(set(request_ids)) != len(request_ids)
            ):
                raise RuntimeError(
                    "EAGLE3 host-span request identity is invalid"
                )
            identity["request_ids"] = request_ids
        if (
            not isinstance(identity["forward_iter"], int)
            or isinstance(identity["forward_iter"], bool)
            or not isinstance(identity["scheduler_forward_mode"], str)
            or not isinstance(
                identity["scheduler_forward_mode_after_worker"], str
            )
            or not isinstance(identity["worker_branch"], str)
            or not isinstance(identity["model_forward_mode"], str)
            or not isinstance(identity["launch_kind"], str)
            or type(identity["is_extend"]) is not bool
        ):
            raise RuntimeError("EAGLE3 host-span batch identity is incomplete")

        if is_controller_verify:
            verify_names = (
                "ready_build_started_ns",
                "full_wave_ready_ns",
                "post_input_build_ns",
                "pre_run_eagle_verify_ns",
                "post_run_eagle_verify_ns",
                "post_feature_handoff_ns",
            )
            verify = {
                name: getattr(batch_result, f"dt_eagle3_{name}", None)
                for name in verify_names
            }
            if any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in verify.values()
            ):
                raise RuntimeError(
                    "EAGLE3 controller host-span boundaries are incomplete"
                )
            ordered = (
                verify["ready_build_started_ns"],
                verify["full_wave_ready_ns"],
                common["selection_returned_ns"],
                common["run_batch_entry_ns"],
                common["before_worker_call_ns"],
                common["worker_entry_ns"],
                verify["post_input_build_ns"],
                verify["pre_run_eagle_verify_ns"],
                verify["post_run_eagle_verify_ns"],
                verify["post_feature_handoff_ns"],
                common["worker_return_ns"],
                common["after_worker_call_ns"],
                common["copy_submit_started_ns"],
                common["copy_submit_returned_ns"],
                common["run_batch_return_ns"],
                common["result_processing_entry_ns"],
                common["copy_ready_ns"],
                common["park_entry_ns"],
                common["completed_ns"],
            )
            if any(right < left for left, right in zip(ordered, ordered[1:])):
                raise RuntimeError(
                    "EAGLE3 controller host-span boundaries are out of order"
                )
            lane_id = getattr(batch_result, "dt_pipeline_lane_id", None)
            transfer_ids = getattr(
                batch_result, "dt_full_wave_transfer_ids", None
            )
            if (
                identity["scheduler_forward_mode"] != "DECODE"
                or identity["scheduler_forward_mode_after_worker"]
                != "TARGET_VERIFY"
                or identity["worker_branch"] != "remote_eagle3_verify"
                or identity["model_forward_mode"] != "TARGET_VERIFY"
                or identity["launch_kind"] != "controller_verify"
                or not isinstance(lane_id, int)
                or isinstance(lane_id, bool)
                or not isinstance(transfer_ids, tuple)
                or len(transfer_ids) != batch_size
                or any(not isinstance(value, str) or not value for value in transfer_ids)
            ):
                raise RuntimeError("EAGLE3 controller host-span identity is invalid")
            self._profile_emit(
                EAGLE3_HOST_SPAN_PROFILE_EVENT,
                **identity,
                **verify,
                wave_id=wave_id,
                lane_id=lane_id,
                transfer_ids=transfer_ids,
                ready_to_complete_ns=(
                    common["completed_ns"] - verify["full_wave_ready_ns"]
                ),
            )
            if not self._eagle3_host_span_startup_flushed:
                # Expose the first built-in launch-warmup controller row once.
                # The separately RID-qualified health flush below then makes
                # the outer health prefill and all preceding rows visible.
                self._profile_flush()
                self._eagle3_host_span_startup_flushed = True
            self._flush_eagle3_health_host_span(request_ids)
            return

        self._profile_emit(EAGLE3_HOST_SPAN_PROFILE_EVENT, **identity)
        self._flush_eagle3_health_host_span(request_ids)

    def _flush_eagle3_health_host_span(
        self, request_ids: tuple[str, ...] | None
    ) -> None:
        """Expose a completed outer health launch to the startup gate once."""

        if request_ids is None:
            return
        health_ids = {
            rid
            for rid in request_ids
            if rid.startswith(f"{HEALTH_CHECK_RID_PREFIX}_")
        }
        new_health_ids = health_ids - self._eagle3_host_span_flushed_health_ids
        if new_health_ids:
            self._profile_flush()
            self._eagle3_host_span_flushed_health_ids.update(new_health_ids)

    def _record_target_verify_gpu_breakdown(
        self, batch_result, batch_size: int
    ) -> None:
        if not self.is_export_rank or batch_result is None:
            return
        events = getattr(batch_result, "dt_target_verify_gpu_events", None)
        if events is None:
            return
        if events is None or len(events) != 4:
            raise RuntimeError("DFlash Target verify GPU profile is incomplete")
        model_start, model_done, accept_done, finish_done = events
        self._profile_emit(
            "target_verify_gpu_breakdown",
            batch_size=batch_size,
            wave_id=getattr(batch_result, "dt_full_wave_id", None),
            lane_id=getattr(batch_result, "dt_pipeline_lane_id", None),
            model_ns=int(model_start.elapsed_time(model_done) * 1_000_000),
            accept_ns=int(model_done.elapsed_time(accept_done) * 1_000_000),
            finish_ns=int(accept_done.elapsed_time(finish_done) * 1_000_000),
            can_run_cuda_graph=bool(batch_result.can_run_cuda_graph),
        )

    def _record_eagle3_verify_gpu_breakdown(
        self, batch_result, batch_size: int
    ) -> None:
        if not self.is_export_rank or batch_result is None:
            return
        events = getattr(batch_result, "dt_eagle3_verify_gpu_events", None)
        if events is None:
            return
        if len(events) != 5:
            raise RuntimeError("EAGLE3 Target verify GPU profile is incomplete")
        run_start, model_start, model_done, accept_done, feature_done = events
        phase_ns = {
            "prepare_ns": int(run_start.elapsed_time(model_start) * 1_000_000),
            "model_ns": int(model_start.elapsed_time(model_done) * 1_000_000),
            "accept_ns": int(model_done.elapsed_time(accept_done) * 1_000_000),
            "feature_pack_ns": int(
                accept_done.elapsed_time(feature_done) * 1_000_000
            ),
        }
        # Sum the individually rounded stream-corridor phases so the serialized
        # record conserves exactly.  Existing copy_done synchronization has
        # already completed these events before park; do not synchronize here.
        total_ns = sum(phase_ns.values())
        self._profile_emit(
            "target_eagle3_verify_gpu_breakdown",
            batch_size=batch_size,
            wave_id=getattr(batch_result, "dt_full_wave_id", None),
            lane_id=getattr(batch_result, "dt_pipeline_lane_id", None),
            **phase_ns,
            total_ns=total_ns,
            can_run_cuda_graph=bool(batch_result.can_run_cuda_graph),
        )

    def _record_full_wave_completion(
        self, batch: ScheduleBatch, *, batch_result, batch_size: int
    ) -> None:
        """Record a READY_VERIFY wave after committed features are published.

        ``build_ready_batch`` stamps only controller-created verification waves.
        Completion is observed after Target result processing and
        ``_park_completed_batch`` have selected and published the authoritative
        committed hidden states.  The global period describes successive waves
        on this Target; the lane period describes successive completions for
        the same frozen lane.
        """

        wave_id = getattr(
            batch_result,
            "dt_full_wave_id",
            getattr(batch, "dt_full_wave_id", None),
        )
        ready_ns = getattr(
            batch_result,
            "dt_full_wave_ready_ns",
            getattr(batch, "dt_full_wave_ready_ns", None),
        )
        lane_id = getattr(
            batch_result,
            "dt_pipeline_lane_id",
            getattr(batch, "dt_pipeline_lane_id", None),
        )
        if wave_id is None or ready_ns is None or lane_id is None:
            return
        if lane_id != -1 or not self.continuous_batching:
            if lane_id < 0 or lane_id >= self.pipeline_lanes:
                raise RuntimeError("DT full-wave completion has an invalid lane")
        completed_ns = time.monotonic_ns()
        if self.eagle3_host_span_profile_enabled and batch_result is not None:
            batch_result.dt_eagle3_completed_ns = completed_ns
        previous_completed_ns = self._last_full_wave_completed_ns
        previous_lane_completed_ns = (
            self._last_lane_full_wave_completed_ns[lane_id] if lane_id >= 0 else 0
        )
        self._last_full_wave_completed_ns = completed_ns
        if lane_id >= 0:
            self._last_lane_full_wave_completed_ns[lane_id] = completed_ns
        if self.is_export_rank:
            self._profile_emit(
                "target_full_wave_complete",
                wave_id=wave_id,
                lane_id=lane_id,
                batch_size=batch_size,
                ready_ns=ready_ns,
                completed_ns=completed_ns,
                ready_to_complete_ns=completed_ns - ready_ns,
                wave_period_ns=(
                    completed_ns - previous_completed_ns if previous_completed_ns else 0
                ),
                lane_period_ns=(
                    completed_ns - previous_lane_completed_ns
                    if previous_lane_completed_ns
                    else 0
                ),
                parked=len(self._parked),
                feature_source_free_slots=self._feature_source_free_slots,
                deferred_feature_stages=self._deferred_dflash_feature_stages,
            )

    def _record_measurement_lane_execution(
        self, batch: ScheduleBatch, batch_result
    ) -> None:
        if not self.is_export_rank or batch_result is None:
            return
        # Speculative verification keeps the scheduler batch in DECODE mode;
        # only the derived ForwardBatch seen by ModelRunner is TARGET_VERIFY.
        # The event id is attached exclusively to the completed Target verify
        # result, so use it as the authoritative execution discriminator.
        model_runner_event_id = getattr(
            batch_result, "dt_cuda_graph_measurement_event_id", None
        )
        if model_runner_event_id is None:
            return
        measurement_reqs = [
            req
            for req in batch.reqs
            if req.rid.startswith(DT_BENCHMARK_MEASUREMENT_RID_PREFIX)
        ]
        if not measurement_reqs:
            return
        lanes = (
            {getattr(batch, "dt_pipeline_lane_id", -1)}
            if self.continuous_batching
            else {self._lane_by_request.get(req.rid) for req in measurement_reqs}
        )
        if None in lanes or len(lanes) != 1:
            raise RuntimeError(
                "DT measurement verify batch does not have one assigned lane"
            )
        lane_id = lanes.pop()
        execution = "graph" if batch_result.can_run_cuda_graph else "eager"
        key = (lane_id, execution)
        count = self._measurement_lane_counts.get(key, 0) + 1
        self._measurement_lane_counts[key] = count
        if count & (count - 1):
            return
        logger.info(
            "%s%s",
            DT_LANE_MEASUREMENT_MARKER,
            json.dumps(
                {
                    "target_instance": self.instance_id,
                    "lane_id": lane_id,
                    "pipeline_lanes": self.pipeline_lanes,
                    "pipeline_lane_weights": self.pipeline_lane_weights,
                    "scheduling": "continuous" if self.continuous_batching else "fixed_lane",
                    "execution": execution,
                    "fallback_reason": (
                        None if execution == "graph" else "target_verify_not_graphed"
                    ),
                    "batches": count,
                    "batch_measurement_rows": len(measurement_reqs),
                    "batch_size": len(batch.reqs),
                    "model_runner_event_namespace": "model_runner",
                    "model_runner_event_id": model_runner_event_id,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    def _park_completed_batch(self, batch: ScheduleBatch, *, batch_result=None) -> None:
        if not batch.reqs:
            # Every launched forward was registered; its result still retires
            # that ownership even when all its requests left the batch.
            self._release_target_inflight_batch(batch)
            if not getattr(batch_result, "dt_dflash_feature_stage_credit", False):
                self._release_inplace_hidden(batch_result)
            if getattr(batch_result, "dt_dflash_feature_stage_credit", False):
                self._stage_dflash_feature_copy(
                    batch, batch_result, finalize_deferred=True
                )
            staged_flash = getattr(batch_result, "dt_staged_flashmtp_pivots", None)
            if isinstance(staged_flash, StagedFlashMTPPivotBatch):
                self._cancel_staged_flashmtp_pivots(staged_flash, batch_result)
            return
        flashmtp_profile_active = bool(
            getattr(self, "_profile_active", False)
            and not self.is_token_only
            and not self.is_feature_backed
            and batch_result is not None
        )
        flashmtp_phase_started_ns = (
            time.monotonic_ns() if flashmtp_profile_active else 0
        )
        self._release_target_inflight_batch(batch)
        if flashmtp_profile_active:
            batch_result.dt_flashmtp_release_inflight_ns = (
                time.monotonic_ns() - flashmtp_phase_started_ns
            )
        if batch.spec_info is None:
            return
        if not isinstance(batch.spec_info, DFlashDraftInputV2):
            raise TypeError("DT Target can park only DFlashDraftInputV2 batches")

        if self.is_feature_backed:
            feature_stage_started_ns = time.monotonic_ns()
            if getattr(batch_result, "dt_staged_dflash_features", None) is None:
                self._stage_dflash_feature_copy(
                    batch, batch_result, finalize_deferred=True
                )
            batch_result.dt_dflash_feature_stage_ns = (
                time.monotonic_ns() - feature_stage_started_ns
            )
            self._park_dflash_completed_batch(
                batch, batch.spec_info, batch_result=batch_result
            )
            return

        if flashmtp_profile_active:
            flashmtp_phase_started_ns = time.monotonic_ns()
        original_batch_size = len(batch.reqs)
        draft_input = batch.spec_info
        for req in batch.reqs:
            if req.finished() or req.is_retracted:
                self._forget_request_lane(req)
                verified_transfer = self._verify_transfer_by_request.pop(req.rid, None)
                if (
                    self.is_export_rank
                    and verified_transfer is not None
                    and self._ledger.get(verified_transfer) is not None
                ):
                    self._ledger.finish(verified_transfer)
                self._release_draft_request(req.rid)
        for req in batch.reqs:
            if req.finished() or req.is_retracted:
                self._next_round.pop(req.rid, None)
            if req.finished():
                self._lifecycle_attempt_by_request.pop(req.rid, None)
        if flashmtp_profile_active:
            batch_result.dt_flashmtp_retire_ns = (
                time.monotonic_ns() - flashmtp_phase_started_ns
            )
            flashmtp_phase_started_ns = time.monotonic_ns()
        staged_flash = getattr(batch_result, "dt_staged_flashmtp_pivots", None)
        if not self.is_token_only and staged_flash is None:
            # Non-overlap scheduling has no later forward that can replay over
            # this graph output, so publication at the result boundary is safe.
            self._stage_flashmtp_pivot_copy(batch, batch_result)
            staged_flash = getattr(batch_result, "dt_staged_flashmtp_pivots", None)
        if not self.is_token_only and not isinstance(
            staged_flash, StagedFlashMTPPivotBatch
        ):
            raise RuntimeError("FlashMTP Target result has no staged GPU pivot lease")
        # A long prompt can have multiple prefill chunks.  In non-overlap mode
        # the normal result processor decrements ``inflight_middle_chunks``
        # from one to zero before this hook, but ``batch.chunked_req`` still
        # identifies that result as an intermediate chunk.  In overlap mode
        # the counter additionally fences older queued results.  Exporting
        # either kind here would give one request two concurrent Draft rounds.
        chunked_req = getattr(batch, "chunked_req", None)
        if self.is_token_only:
            active_indices = [
                index
                for index, req in enumerate(batch.reqs)
                if not req.finished()
                and not req.is_retracted
                and req is not chunked_req
                and getattr(req, "inflight_middle_chunks", 0) <= 0
                and self._target_inflight_count(req.rid) == 0
            ]
            active_flash_tasks = ()
        else:
            req_by_id = {req.rid: req for req in batch.reqs}
            active_flash = []
            for staged_task in staged_flash.tasks:
                req = req_by_id.get(staged_task.request_id)
                if req is None:
                    self._cancel_staged_flashmtp_pivots(staged_flash, batch_result)
                    raise RuntimeError(
                        "FlashMTP staged request is absent from result batch"
                    )
                if (
                    staged_task.request_index < 0
                    or staged_task.request_index >= len(batch.reqs)
                    or batch.reqs[staged_task.request_index] is not req
                ):
                    self._cancel_staged_flashmtp_pivots(staged_flash, batch_result)
                    raise RuntimeError("FlashMTP staged request index changed")
                eligible = bool(
                    not req.finished()
                    and not req.is_retracted
                    and req is not chunked_req
                    and getattr(req, "inflight_middle_chunks", 0) <= 0
                    and self._target_inflight_count(req.rid) == 0
                )
                if not eligible:
                    if self.is_export_rank:
                        self._release_source_slot(staged_task.source_lease_id)
                    continue
                active_flash.append(staged_task)
            active_flash_tasks = tuple(active_flash)
            active_indices = [task.request_index for task in active_flash_tasks]
        tasks = []
        parked_reqs = []
        lane_ids = [
            self._lane_for_request(batch.reqs[index]) for index in active_indices
        ]
        if flashmtp_profile_active:
            batch_result.dt_flashmtp_select_ns = (
                time.monotonic_ns() - flashmtp_phase_started_ns
            )
            flashmtp_phase_started_ns = time.monotonic_ns()
        (
            bonus_tokens_cpu,
            seq_lens_cpu,
            staged_flashmtp_scalars,
        ) = _flashmtp_park_scalar_values(
            batch_result=batch_result,
            draft_input=draft_input,
            active_indices=active_indices,
            original_batch_size=original_batch_size,
        )
        if staged_flashmtp_scalars:
            batch_result.dt_flashmtp_bonus_tokens_cpu = None
            batch_result.dt_flashmtp_seq_lens_cpu = None
            batch_result.dt_flashmtp_scalar_copy_done = None
        if flashmtp_profile_active:
            batch_result.dt_flashmtp_scalar_copy_ns = (
                time.monotonic_ns() - flashmtp_phase_started_ns
            )
            flashmtp_phase_started_ns = time.monotonic_ns()
        descriptors_by_transfer = (
            {} if self.is_token_only else staged_flash.descriptors_by_transfer
        )
        control_committed = False
        try:
            for active_index, index in enumerate(active_indices):
                req = batch.reqs[index]
                if self.is_token_only:
                    round_id = self._next_round.get(req.rid, 0)
                    attempt = self._ensure_lifecycle_attempt(req.rid)
                    transfer_id = f"{self.instance_id}:{req.rid}:{round_id}:{attempt}"
                    self._next_round[req.rid] = round_id + 1
                else:
                    staged_task = active_flash_tasks[active_index]
                    round_id = self._next_round.get(req.rid, 0)
                    attempt = self._ensure_lifecycle_attempt(req.rid)
                    transfer_id = staged_task.source_lease_id
                    self._next_round[req.rid] = round_id + 1
                previous_transfer_id = None
                if self.is_token_only and round_id > 0:
                    previous_transfer_id = self._verify_transfer_by_request.pop(
                        req.rid, None
                    )
                    if previous_transfer_id is None:
                        raise RuntimeError(
                            "DT reconcile is missing the previous verified transfer"
                        )
                bonus_token = int(bonus_tokens_cpu[active_index])
                seq_len = int(seq_lens_cpu[active_index])
                self._parked[transfer_id] = ParkedTargetRound(
                    req=req,
                    transfer_id=transfer_id,
                    round_id=round_id,
                    attempt=attempt,
                    bonus_token=bonus_token,
                    seq_len=seq_len,
                    lane_id=lane_ids[active_index],
                    deadline_ns=self._new_round_deadline_ns(),
                )
                parked_reqs.append(req)

                if self.is_export_rank:
                    tasks.append(
                        self._make_task(
                            transfer_id=transfer_id,
                            req=req,
                            round_id=round_id,
                            attempt=attempt,
                            bonus_token=bonus_token,
                            previous_transfer_id=previous_transfer_id,
                            feature_descriptor=descriptors_by_transfer.get(transfer_id),
                        )
                    )
            if flashmtp_profile_active:
                batch_result.dt_flashmtp_task_build_ns = (
                    time.monotonic_ns() - flashmtp_phase_started_ns
                )
                flashmtp_phase_started_ns = time.monotonic_ns()
            if tasks:
                self._queue_task_batch(tuple(tasks))
                self._staged_flashmtp_source_leases.difference_update(
                    task.transfer_id for task in tasks
                )
                control_committed = True
                # Source slots are committed at this point, so publish the control
                # message in the same scheduler turn. Waiting for the next poll
                # adds a fixed idle bubble before Draft can start the round.
                self._flush_pending_send()
            if flashmtp_profile_active:
                batch_result.dt_flashmtp_queue_flush_ns = (
                    time.monotonic_ns() - flashmtp_phase_started_ns
                )
                flashmtp_phase_started_ns = time.monotonic_ns()
        except Exception:
            if (
                not self.is_token_only
                and not control_committed
                and isinstance(staged_flash, StagedFlashMTPPivotBatch)
            ):
                self._cancel_staged_flashmtp_pivots(staged_flash, batch_result)
            raise
        if not self.is_token_only:
            batch_result.dt_staged_flashmtp_pivots = None
        if parked_reqs:
            parked_by_lane = Counter(state.lane_id for state in self._parked.values())
            for lane_id, parked_count in parked_by_lane.items():
                self._lane_target_sizes[lane_id] = max(
                    self._lane_target_sizes[lane_id], parked_count
                )
        if parked_reqs:
            if len(parked_reqs) == original_batch_size or getattr(
                batch, "enable_overlap", False
            ):
                # The whole batch crosses into WAIT_DRAFT.  Slicing every
                # ScheduleBatch tensor down to zero only to discard the batch
                # on the next scheduler turn is pure host/GPU bookkeeping.
                # The same applies to a partially parked overlap result
                # snapshot: ScheduleBatch.copy intentionally contains only the
                # fields required by process_batch_result, so it cannot safely
                # run the full tensor filter and is discarded after this hook.
                # Rebinding ownership is enough; later Target forwards or the
                # chunked-prefill pointer own any rows that were not parked.
                batch.reqs = []
            else:
                batch.filter_batch(chunked_req_to_exclude=parked_reqs)
        if flashmtp_profile_active:
            batch_result.dt_flashmtp_finalize_ns = (
                time.monotonic_ns() - flashmtp_phase_started_ns
            )

    def _park_dflash_completed_batch(
        self,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        *,
        batch_result,
    ) -> None:
        """Route staged DFlash features and park only proposal boundaries."""
        original_batch_size = len(batch.reqs)
        staged = getattr(batch_result, "dt_staged_dflash_features", None)
        if not isinstance(staged, StagedDFlashFeatureBatch):
            raise RuntimeError("DFLASH Target result has no staged feature lease")
        phase_started_ns = time.monotonic_ns()
        req_by_id = {req.rid: req for req in batch.reqs}
        batch_result.dt_dflash_request_index_ns = time.monotonic_ns() - phase_started_ns

        phase_started_ns = time.monotonic_ns()
        for req in batch.reqs:
            if req.finished() or req.is_retracted:
                self._retire_dflash_rounds_for_request(
                    req.rid,
                    reason=("kv_retract" if req.is_retracted else "request_finished"),
                )
                self._forget_request_lane(req)
                verified_transfer = self._verify_transfer_by_request.pop(req.rid, None)
                if (
                    self.is_export_rank
                    and verified_transfer is not None
                    and self._ledger.get(verified_transfer) is not None
                ):
                    self._ledger.finish(verified_transfer)
                self._release_draft_request(req.rid)
                self._next_round.pop(req.rid, None)
            if req.finished():
                self._lifecycle_attempt_by_request.pop(req.rid, None)
        batch_result.dt_dflash_retire_ns = time.monotonic_ns() - phase_started_ns

        chunked_req = getattr(batch, "chunked_req", None)
        tasks = []
        parked_reqs = []
        phase_started_ns = time.monotonic_ns()
        for spec in staged.tasks:
            req = req_by_id.get(spec.request_id)
            if req is None:
                raise RuntimeError("DFLASH staged request is absent from result batch")
            if req.finished() or req.is_retracted:
                if self.is_export_rank:
                    if staged.ring_publish:
                        # The wave's descriptors do not exist yet; the
                        # deferred cleanup cancels this transfer's lease
                        # share right after publishing them.
                        self._cancelled_transfers.add(spec.transfer_id)
                    else:
                        descriptor = staged.descriptors_by_transfer[spec.transfer_id]
                        self._feature_source_pool.cancel(
                            transfer_id=spec.transfer_id,
                            source_slot_id=descriptor.source_slot_id,
                            slot_generation=descriptor.slot_generation,
                            source_session_id=descriptor.source_session_id,
                            pool_generation=descriptor.pool_generation,
                        )
                    self._source_slot_by_transfer.pop(spec.transfer_id, None)
                continue
            requires_proposal = bool(
                req is not chunked_req
                and getattr(req, "inflight_middle_chunks", 0) <= 0
                and self._target_inflight_count(req.rid) == 0
            )
            previous_transfer_id = self._verify_transfer_by_request.pop(req.rid, None)
            if previous_transfer_id != spec.previous_transfer_id:
                raise RuntimeError(
                    "DFLASH previous-round identity changed before routing"
                )
            if self.is_export_rank and previous_transfer_id is not None:
                self._ledger.mark_need_reconcile(previous_transfer_id)
            if requires_proposal:
                round_state = ParkedTargetRound(
                    req=req,
                    transfer_id=spec.transfer_id,
                    round_id=spec.round_id,
                    attempt=spec.attempt,
                    bonus_token=spec.bonus_token,
                    seq_len=spec.seq_len,
                    lane_id=self._lane_for_request(req),
                    deadline_ns=self._new_round_deadline_ns(),
                )
                self._parked[spec.transfer_id] = round_state
                parked_reqs.append(req)
            else:
                self._feature_control_rounds[spec.transfer_id] = ParkedTargetRound(
                    req=req,
                    transfer_id=spec.transfer_id,
                    round_id=spec.round_id,
                    attempt=spec.attempt,
                    bonus_token=spec.bonus_token,
                    seq_len=spec.seq_len,
                    lane_id=self._lane_for_request(req),
                    deadline_ns=self._new_round_deadline_ns(),
                )
            if self.is_export_rank:
                if staged.ring_publish:
                    wave = getattr(batch_result, "dt_ring_publish_wave", None)
                    if wave is None:
                        raise RuntimeError("ring-published wave lost its staging state")
                    if not requires_proposal:
                        raise RuntimeError(
                            "ring-published wave changed its proposal boundary"
                        )
                    wave.proposals[spec.transfer_id] = (
                        requires_proposal,
                        previous_transfer_id,
                    )
                else:
                    descriptor = staged.descriptors_by_transfer[spec.transfer_id]
                    tasks.append(
                        self._make_task(
                            transfer_id=spec.transfer_id,
                            req=req,
                            round_id=spec.round_id,
                            attempt=spec.attempt,
                            bonus_token=spec.bonus_token,
                            previous_transfer_id=previous_transfer_id,
                            feature_descriptor=descriptor,
                            feature_token_ids=spec.feature_token_ids,
                            committed_token_ids=spec.committed_token_ids,
                            feature_token_offset=spec.feature_token_offset,
                            requires_proposal=requires_proposal,
                        )
                    )
                if previous_transfer_id is not None:
                    self._ledger.finish(previous_transfer_id)
        batch_result.dt_dflash_task_build_ns = time.monotonic_ns() - phase_started_ns

        phase_started_ns = time.monotonic_ns()
        if tasks:
            self._queue_task_batch(tuple(tasks))
            self._flush_pending_send()
        batch_result.dt_dflash_queue_flush_ns = time.monotonic_ns() - phase_started_ns
        phase_started_ns = time.monotonic_ns()
        batch_result.dt_staged_dflash_features = None
        if parked_reqs:
            parked_by_lane = Counter(state.lane_id for state in self._parked.values())
            for lane_id, parked_count in parked_by_lane.items():
                self._lane_target_sizes[lane_id] = max(
                    self._lane_target_sizes[lane_id], parked_count
                )
            if len(parked_reqs) == original_batch_size or getattr(
                batch, "enable_overlap", False
            ):
                batch.reqs = []
            else:
                batch.filter_batch(chunked_req_to_exclude=parked_reqs)
        batch_result.dt_dflash_release_ns = time.monotonic_ns() - phase_started_ns

    def poll(self, *, overlap_result=None) -> None:
        poll_entered_ns = time.monotonic_ns()
        self._poll_target_gpu_observations()
        observer = getattr(self, "_overlap_fix_observer", None)
        if observer is not None:
            observer.checkpoint(poll_entered_ns)
        previous_poll_ns = self._last_poll_ns
        receive_ns = 0
        packet = None
        if self.is_export_rank:
            receive_start_ns = time.monotonic_ns()
            packet = (
                time.monotonic_ns(),
                self._poll_export_rank(),
                self._query_overlap_result_ready(overlap_result),
                (
                    # A reserved slot belongs to a deferred stage that already
                    # holds a source credit; count it once, as before.
                    # Kernel-published waves pending their flush-time cleanup
                    # hold reservations that belong to no deferred credit, so
                    # they must not count as available capacity; otherwise a
                    # Python-path park can be admitted with zero free slots
                    # and its pool.publish raises.
                    self._feature_source_pool.free_slot_count
                    + self._feature_source_pool.reserved_slot_count
                    - len(getattr(self, "_ring_publish_pending", ()))
                    if self._feature_source_pool is not None
                    else 0
                ),
                self._overlap_launch_ns,
                getattr(self, "_ring_publish_fatal", None),
            )
            packet = (*packet, self._late_freeze_hold(overlap_result, packet[2]))
            receive_ns = time.monotonic_ns() - receive_start_ns
        broadcast_start_ns = time.monotonic_ns()
        if self.scheduler.ps.tp_size > 1:
            packet = broadcast_pyobj(
                packet,
                self.scheduler.tp_group.rank,
                self.scheduler.tp_cpu_group,
                src=self.scheduler.tp_group.ranks[0],
            )
        broadcast_ns = time.monotonic_ns() - broadcast_start_ns
        assert packet is not None
        (
            poll_now_ns,
            events,
            result_ready,
            feature_source_free_slots,
            overlap_launch_ns,
            ring_publish_fatal,
            hold_next_batch,
        ) = packet
        self._last_poll_ns = poll_now_ns
        self._overlap_launch_ns = overlap_launch_ns
        self._overlap_result_ready = result_ready
        self.hold_next_batch = bool(hold_next_batch)
        self._feature_source_free_slots = feature_source_free_slots
        if ring_publish_fatal:
            # A publication failure is fatal on the export rank; the other
            # ranks learn it here rather than through a per-wave broadcast.
            raise RuntimeError(ring_publish_fatal)
        if self.is_export_rank and events:
            poll_finished_ns = time.monotonic_ns()
            self._profile_emit(
                "target_result_poll",
                started_ns=poll_entered_ns,
                finished_ns=poll_finished_ns,
                total_ns=poll_finished_ns - poll_entered_ns,
                poll_gap_ns=(
                    poll_entered_ns - previous_poll_ns if previous_poll_ns else 0
                ),
                receive_ns=receive_ns,
                broadcast_ns=broadcast_ns,
                event_count=len(events),
                ready_count=sum(kind == "ready" for kind, _transfer_id, _ in events),
                direct_ready_count=sum(
                    kind == "ready_direct" for kind, _transfer_id, _ in events
                ),
                accepted_count=sum(
                    kind == "accepted" for kind, _transfer_id, _ in events
                ),
                failed_count=sum(kind == "failed" for kind, _transfer_id, _ in events),
                parked=len(self._parked),
            )
        for kind, transfer_id, payload in events:
            # Successful results prove owner materialization; ACK acceptance
            # alone can also acknowledge a request that ends in FINAL_ERROR.
            if kind in ("ready", "ready_direct", "materialized"):
                admitted_state = self._parked.get(
                    transfer_id
                ) or self._feature_control_rounds.get(transfer_id)
                if (
                    admitted_state is not None
                    and self._lifecycle_attempt_by_request.get(admitted_state.req.rid)
                    == admitted_state.attempt
                ):
                    self._draft_admitted_attempts[admitted_state.req.rid] = (
                        admitted_state.attempt
                    )
            state = self._parked.get(transfer_id)
            if state is None and kind == "materialized":
                self._feature_control_rounds.pop(transfer_id, None)
                continue
            if state is None and kind == "failed":
                state = self._feature_control_rounds.pop(transfer_id, None)
                if state is not None:
                    self._parked[transfer_id] = state
            if state is None:
                continue
            if kind in ("ready", "ready_direct"):
                candidate_payload, bonus_token, seq_len = payload
                if state.bonus_token != bonus_token or state.seq_len != seq_len:
                    raise RuntimeError(
                        "DT Target TP metadata diverged before Draft result: "
                        f"transfer_id={transfer_id}, "
                        f"local_bonus={state.bonus_token}, "
                        f"export_bonus={bonus_token}, "
                        f"local_seq_len={state.seq_len}, "
                        f"export_seq_len={seq_len}"
                    )
                observe_ready = (
                    observer is not None and observer.enabled()
                    and state.candidate_bytes is None
                    and state.candidate_completion is None
                )
                applied_before_ns = time.monotonic_ns() if observe_ready else 0
                if kind == "ready_direct":
                    state.candidate_completion = candidate_payload
                    if self.is_export_rank and self._profile_active:
                        self._profile_emit(
                            "target_direct_tp_apply",
                            transfer_id=transfer_id,
                            request_id=state.req.rid,
                            round_id=state.round_id,
                            completion_observed_ns=candidate_payload.observed_ns,
                            tp_apply_ns=time.monotonic_ns(),
                            broadcast_ns=broadcast_ns,
                            target_compute_batches_inflight=(
                                self._target_compute_batches_inflight
                            ),
                        )
                else:
                    state.candidate_bytes = candidate_payload
                state.draft_admitted = True
                if observe_ready:
                    observer.ready(state, applied_before_ns, time.monotonic_ns())
            elif kind == "accepted":
                state.draft_admitted = True
            else:
                state.failure = payload

        if observer is not None:
            observer.target_decision(self)

    def record_target_batch_submission(
        self,
        batch,
        batch_result,
        *,
        selection_started_ns: int,
        selection_finished_ns: int,
        submission_started_ns: int,
        submission_returned_ns: int,
        result_queue_depth_before: int,
        iteration_started_ns: int,
        receive_started_ns: int,
        receive_finished_ns: int,
        poll_started_ns: int,
        poll_finished_ns: int,
    ) -> None:
        """Record host submission and retain CUDA events for bounded queries."""

        if not getattr(self, "is_export_rank", False) or not getattr(
            self, "_profile_active", False
        ):
            return
        wave_id = getattr(batch, "dt_full_wave_id", None)
        if wave_id is None:
            return
        transfer_ids = tuple(getattr(batch, "dt_full_wave_transfer_ids", ()))
        events = getattr(batch_result, "dt_target_verify_gpu_events", None)
        if events is None or len(events) != 4:
            self._profile_emit(
                "target_verify_cuda_observation_error",
                wave_id=wave_id,
                transfer_ids=transfer_ids,
                reason="missing_events",
            )
        else:
            clock_slot, clock_done = -1, None
            clock = self._verify_gpu_clock()
            if clock is not None:
                clock_slot, clock_done = clock.stamp_after(events[-1])
            self._pending_target_gpu_observations.append(
                _PendingTargetGpuObservation(
                    clock_slot=clock_slot,
                    clock_done=clock_done,
                    wave_id=wave_id,
                    transfer_ids=transfer_ids,
                    ready_ns=getattr(batch, "dt_full_wave_ready_ns", 0),
                    selection_started_ns=selection_started_ns,
                    selection_finished_ns=selection_finished_ns,
                    submission_started_ns=submission_started_ns,
                    submission_returned_ns=submission_returned_ns,
                    start_event=events[0],
                    end_event=events[-1],
                )
            )
        self._profile_emit(
            "target_batch_submit",
            wave_id=wave_id,
            transfer_ids=transfer_ids,
            batch_size=len(batch.reqs),
            ready_ns=getattr(batch, "dt_full_wave_ready_ns", 0),
            selection_started_ns=selection_started_ns,
            selection_finished_ns=selection_finished_ns,
            submission_started_ns=submission_started_ns,
            submission_returned_ns=submission_returned_ns,
            submission_wall_ns=submission_returned_ns - submission_started_ns,
            result_queue_depth_before=result_queue_depth_before,
            target_compute_batches_inflight=self._target_compute_batches_inflight,
            iteration_started_ns=iteration_started_ns,
            receive_started_ns=receive_started_ns,
            receive_finished_ns=receive_finished_ns,
            poll_started_ns=poll_started_ns,
            poll_finished_ns=poll_finished_ns,
        )

    def _verify_gpu_clock(self):
        clock = getattr(self, "_gpu_clock", None)
        if clock is None and os.environ.get("DT_GPU_CLOCK", "0") == "1":
            from sglang.srt.speculative.dt_gpu_clock import GpuClock

            clock = self._gpu_clock = GpuClock(torch.device("cuda", torch.cuda.current_device()))
        return clock

    def _poll_target_gpu_observations(self) -> None:
        """Capture host observations of CUDA boundaries without synchronizing."""

        pending = getattr(self, "_pending_target_gpu_observations", None)
        if (
            not pending
            or not getattr(self, "is_export_rank", False)
            or not getattr(self, "_profile_active", False)
        ):
            return
        retained = deque()
        while pending:
            observation = pending.popleft()
            try:
                if (
                    not observation.start_observed_ns
                    and observation.start_event.query()
                ):
                    observation.start_observed_ns = time.monotonic_ns()
                end_ready = bool(observation.end_event.query())
            except Exception as exc:
                self._profile_emit(
                    "target_verify_cuda_observation_error",
                    wave_id=observation.wave_id,
                    transfer_ids=observation.transfer_ids,
                    reason=type(exc).__name__,
                )
                continue
            if not end_ready or (
                observation.clock_done is not None and not observation.clock_done.query()
            ):
                retained.append(observation)
                continue
            end_observed_ns = time.monotonic_ns()
            if not observation.start_observed_ns:
                observation.start_observed_ns = end_observed_ns
            previous_end_event = getattr(
                self, "_last_target_verify_end_event", None
            )
            previous_verify_end_to_start_ns = (
                int(previous_end_event.elapsed_time(observation.start_event) * 1_000_000)
                if previous_end_event is not None
                else None
            )
            self._profile_emit(
                "target_verify_cuda_observed",
                wave_id=observation.wave_id,
                transfer_ids=observation.transfer_ids,
                ready_ns=observation.ready_ns,
                selection_started_ns=observation.selection_started_ns,
                selection_finished_ns=observation.selection_finished_ns,
                submission_started_ns=observation.submission_started_ns,
                submission_returned_ns=observation.submission_returned_ns,
                gpu_start_observed_ns=observation.start_observed_ns,
                gpu_end_observed_ns=end_observed_ns,
                cuda_total_ns=int(
                    observation.start_event.elapsed_time(observation.end_event)
                    * 1_000_000
                ),
                cuda_previous_verify_end_to_start_ns=(
                    previous_verify_end_to_start_ns
                ),
                gpu_end_clock_ns=(
                    self._gpu_clock.read(observation.clock_slot)
                    if observation.clock_slot >= 0
                    else 0
                ),
            )
            self._last_target_verify_end_event = observation.end_event
        self._pending_target_gpu_observations = retained

    def record_target_old_result_processing(
        self,
        batch,
        *,
        started_ns: int,
        finished_ns: int,
        result_queue_depth_before: int,
    ) -> None:
        if not getattr(self, "is_export_rank", False) or not getattr(
            self, "_profile_active", False
        ):
            return
        self._profile_emit(
            "target_old_result_process",
            wave_id=getattr(batch, "dt_full_wave_id", None),
            transfer_ids=tuple(
                getattr(batch, "dt_full_wave_transfer_ids", ())
            ),
            request_ids=tuple(req.rid for req in batch.reqs),
            started_ns=started_ns,
            finished_ns=finished_ns,
            total_ns=finished_ns - started_ns,
            result_queue_depth_before=result_queue_depth_before,
        )

    def has_feature_source_capacity(self) -> bool:
        has_capacity = not self.uses_direct_gpu_source or self._feature_source_free_slots > (
            self._deferred_dflash_feature_stages
        )
        if self.is_export_rank and self.uses_direct_gpu_source:
            now_ns = time.monotonic_ns()
            if not has_capacity and not self._feature_source_stall_started_ns:
                self._feature_source_stall_started_ns = now_ns
                self._profile_emit(
                    "target_feature_source_stall",
                    phase="start",
                    started_ns=now_ns,
                    free_slots=self._feature_source_free_slots,
                    deferred_feature_stages=self._deferred_dflash_feature_stages,
                    pending_task_batches=len(self._pending_send),
                    parked=len(self._parked),
                )
            elif has_capacity and self._feature_source_stall_started_ns:
                started_ns = self._feature_source_stall_started_ns
                self._feature_source_stall_started_ns = 0
                self._profile_emit(
                    "target_feature_source_stall",
                    phase="end",
                    started_ns=started_ns,
                    observed_ns=now_ns,
                    duration_ns=now_ns - started_ns,
                    free_slots=self._feature_source_free_slots,
                    deferred_feature_stages=self._deferred_dflash_feature_stages,
                    pending_task_batches=len(self._pending_send),
                    parked=len(self._parked),
                )
        return has_capacity

    def profile_empty_prefill_arbitration(
        self,
        *,
        decision_started_ns: int,
        prefill_decision: str,
        selected: str,
        waiting_request_ids: tuple[str, ...],
        running_batch_is_full: bool,
        empty_full_before_detach: bool,
        fullness_cleared_by_detach: bool,
        batch_size_before_detach: int,
        batch_size_after_detach: int,
        selected_wave_id: int | None,
        selected_batch_size: int,
        chunked_req,
    ) -> None:
        """Emit deduplicated host-only evidence for empty-running DT choices."""
        if not self.empty_prefill_arbitration_profile_enabled:
            return
        chunked_request_id = getattr(chunked_req, "rid", None)
        chunked_owner_admitted = bool(
            chunked_request_id is not None
            and self._draft_admitted_attempts.get(chunked_request_id)
            == self._lifecycle_attempt_by_request.get(chunked_request_id, -1)
        )
        effective_source_slots = (
            self._feature_source_free_slots - self._deferred_dflash_feature_stages
        )
        signature = (
            prefill_decision,
            selected,
            waiting_request_ids,
            running_batch_is_full,
            empty_full_before_detach,
            fullness_cleared_by_detach,
            batch_size_before_detach,
            batch_size_after_detach,
            selected_wave_id,
            selected_batch_size,
            effective_source_slots,
            self._deferred_dflash_feature_stages,
            chunked_request_id,
            chunked_owner_admitted,
        )
        if signature == self._last_empty_prefill_arbitration_profile:
            return
        self._last_empty_prefill_arbitration_profile = signature
        observed_ns = time.monotonic_ns()
        self._profile_emit(
            "target_empty_prefill_arbitration",
            decision_started_ns=decision_started_ns,
            observed_ns=observed_ns,
            decision_wall_ns=max(0, observed_ns - decision_started_ns),
            prefill_decision=prefill_decision,
            selected=selected,
            waiting_count=len(waiting_request_ids),
            waiting_request_ids=waiting_request_ids,
            running_batch_is_full=running_batch_is_full,
            empty_full_before_detach=empty_full_before_detach,
            fullness_cleared_by_detach=fullness_cleared_by_detach,
            batch_size_before_detach=batch_size_before_detach,
            batch_size_after_detach=batch_size_after_detach,
            selected_wave_id=selected_wave_id,
            selected_batch_size=selected_batch_size,
            feature_source_free_slots=self._feature_source_free_slots,
            deferred_feature_stages=self._deferred_dflash_feature_stages,
            effective_source_slots=effective_source_slots,
            chunked_request_id=chunked_request_id,
            chunked_owner_admitted=chunked_owner_admitted,
        )

    @staticmethod
    def _query_overlap_result_ready(overlap_result) -> bool | None:
        if overlap_result is None:
            return None
        copy_done = getattr(overlap_result, "copy_done", None)
        # Generation overlap normally installs copy_done. Preserve generic
        # behavior for result types without a completion event rather than
        # deferring them until the hard deadline.
        return True if copy_done is None else bool(copy_done.query())

    def _select_pipeline_lane_ready(
        self, ready: list[ParkedTargetRound]
    ) -> list[ParkedTargetRound]:
        ready_by_lane = [[] for _ in range(self.pipeline_lanes)]
        parked_by_lane = Counter(state.lane_id for state in self._parked.values())
        live_by_lane = Counter(self._lane_by_request.values())
        admitted_by_lane = Counter(
            state.lane_id for state in self._parked.values() if state.draft_admitted
        )
        for state in ready:
            ready_by_lane[state.lane_id].append(state)

        if (
            self.is_export_rank
            and self.is_persistent
            and os.environ.get("DT_PROFILE_DIR")
        ):
            lane_ids_in_cursor_order = tuple(
                (self._lane_ready_cursor + offset) % self.pipeline_lanes
                for offset in range(self.pipeline_lanes)
            )
            complete_lane_ids = tuple(
                lane_id
                for lane_id in lane_ids_in_cursor_order
                if ready_by_lane[lane_id]
                and len(ready_by_lane[lane_id]) == admitted_by_lane[lane_id]
            )
            if complete_lane_ids:
                self._profile_emit(
                    "target_ready_snapshot",
                    cursor=self._lane_ready_cursor,
                    complete_lane_ids=complete_lane_ids,
                    complete_lane_count=len(complete_lane_ids),
                    selected_lane_id=complete_lane_ids[0],
                    aggregate_ready_rows=sum(
                        len(ready_by_lane[lane_id]) for lane_id in complete_lane_ids
                    ),
                    ready_rows_by_lane=tuple(map(len, ready_by_lane)),
                    admitted_rows_by_lane=tuple(
                        admitted_by_lane[lane_id]
                        for lane_id in range(self.pipeline_lanes)
                    ),
                )

        for offset in range(self.pipeline_lanes):
            lane_id = (self._lane_ready_cursor + offset) % self.pipeline_lanes
            lane_ready = ready_by_lane[lane_id]
            if self.is_persistent:
                if lane_ready and len(lane_ready) == admitted_by_lane[lane_id]:
                    self._lane_ready_cursor = (lane_id + 1) % self.pipeline_lanes
                    return lane_ready
                continue
            live_rows = max(parked_by_lane[lane_id], live_by_lane[lane_id])
            if lane_ready and live_rows < self._lane_target_sizes[lane_id]:
                self._lane_target_sizes[lane_id] = live_rows
            if lane_ready and len(lane_ready) == live_rows:
                self._lane_ready_cursor = (lane_id + 1) % self.pipeline_lanes
                return lane_ready
        return []

    def detach_parked_requests(self, batch: ScheduleBatch) -> None:
        """Remove WAIT_DRAFT requests from a stale scheduler batch alias.

        The normal completion path filters them from the batch that just ran.
        SGLang may retain another ScheduleBatch alias with the same Req objects,
        however, so enforce the state boundary again before READY_VERIFY batches
        are injected.  WAIT_DRAFT and READY_VERIFY inputs cannot share a forward.
        """
        if not batch.reqs:
            # PrefillAdder may leave an empty running alias marked full after
            # returning a separate prefill batch. Fullness is a snapshot of
            # the old membership, so an empty alias must never retain it.
            batch.batch_is_full = False
            return
        initial_batch_size = len(batch.reqs)
        parked_req_ids = {id(state.req) for state in self._parked.values()}
        target_inflight_rids = set(self._target_inflight_counts)
        keep_indices = [
            index
            for index, req in enumerate(batch.reqs)
            if not getattr(req, "finished", lambda: False)()
            and not getattr(req, "is_retracted", False)
            and id(req) not in parked_req_ids
            and req.rid not in target_inflight_rids
        ]
        if len(keep_indices) != len(batch.reqs):
            batch.filter_batch(keep_indices=keep_indices)

        if (
            batch.reqs
            and isinstance(batch.spec_info, DFlashDraftInputV2)
            and batch.spec_info.draft_tokens is None
        ):
            logger.warning(
                "Re-parking %d WAIT_DRAFT requests retained by a stale "
                "scheduler batch alias.",
                len(batch.reqs),
            )
            self.park_completed_batch(batch)
            if batch.reqs:
                # park_completed_batch intentionally skips terminal/retracted
                # requests. They may still exist only in this stale alias and
                # must not be merged into a READY_VERIFY batch.
                batch.filter_batch(keep_indices=[])
        if len(batch.reqs) < initial_batch_size:
            # ``batch_is_full`` describes the allocation decision for the old
            # membership. Once parked/inflight owners are detached, retaining
            # it makes get_new_batch_prefill() reject an otherwise empty batch
            # before it examines the waiting queue.
            batch.batch_is_full = False

    def register_overlap_inflight_batch(self, launched_batch: ScheduleBatch) -> None:
        """Register Target ownership immediately after an overlap launch.

        The overlap event loop can process the previous result after launching
        the current forward but before sampling and detaching the current batch.
        A final prefill chunk may contain the same request as that previous
        result.  Registering only at detach time leaves a window where the
        previous result observes no later Target owner and starts Draft early.
        Keep registration next to ``result_queue.append`` so every launched
        forward is visible before any older result can be processed.
        """
        inflight_batches = getattr(self, "_target_compute_batches_inflight", 0)
        if inflight_batches >= 2:
            raise RuntimeError(
                "Target compute invariant violated: more than one executing plus "
                "one submitted batch"
            )
        self._target_compute_batches_inflight = inflight_batches + 1
        if self.is_export_rank:
            self._overlap_launch_ns = time.monotonic_ns()
            self._record_freeze_launch(launched_batch)
        for req in launched_batch.reqs:
            self._target_inflight_counts[req.rid] += 1
            self._target_inflight_reqs[req.rid] = req
        observer = getattr(self, "_overlap_fix_observer", None)
        if observer is not None and observer.enabled():
            observer.submitted(launched_batch)

    def detach_overlap_inflight_batch(
        self, *, running_batch: ScheduleBatch, launched_batch: ScheduleBatch
    ) -> None:
        """Move an already-registered batch out of scheduler ownership.

        The queued result snapshot remains the sole owner after sampling.  The
        Target-forward reference count was registered immediately after launch,
        before an older result could be processed in the same scheduler turn.
        """
        launched_req_ids = {id(req) for req in launched_batch.reqs}
        if not launched_req_ids:
            return

        if running_batch is not launched_batch and any(
            id(req) in launched_req_ids for req in running_batch.reqs
        ):
            raise RuntimeError(
                "DT overlap launched requests unexpectedly alias a distinct "
                "running batch"
            )

        # ScheduleBatch.copy intentionally shares spec_info with the queued
        # result snapshot. Calling filter_batch here would therefore empty the
        # pivot/candidate state that park_completed_batch consumes later.
        # Rebinding only reqs makes this scheduler alias logically empty while
        # leaving the result-queue snapshot and its shared tensors intact.
        launched_batch.reqs = []

    def _target_inflight_count(self, rid: str) -> int:
        return int(self._target_inflight_counts.get(rid, 0))

    def _ensure_lifecycle_attempt(self, request_id: str) -> int:
        attempt = self._lifecycle_attempt_by_request.get(request_id)
        if attempt is not None:
            return attempt
        attempt = self._next_lifecycle_attempt
        self._next_lifecycle_attempt += 1
        self._lifecycle_attempt_by_request[request_id] = attempt
        return attempt

    def _advance_lifecycle_attempt(self, request_id: str) -> int:
        previous = self._lifecycle_attempt_by_request.get(request_id, -1)
        attempt = max(previous + 1, self._next_lifecycle_attempt)
        self._next_lifecycle_attempt = attempt + 1
        self._lifecycle_attempt_by_request[request_id] = attempt
        return attempt

    def _release_target_inflight_batch(self, batch: ScheduleBatch) -> None:
        """Consume one launched Target-forward ownership per result row.

        A request may occur in multiple queued prefill chunks, hence the
        reference count. Missing entries are expected for non-overlap mode.
        """
        if getattr(self, "_target_compute_batches_inflight", 0):
            self._target_compute_batches_inflight -= 1
        for req in batch.reqs:
            count = self._target_inflight_counts.get(req.rid, 0)
            if count <= 0:
                continue
            if count == 1:
                del self._target_inflight_counts[req.rid]
                self._target_inflight_reqs.pop(req.rid, None)
            else:
                self._target_inflight_counts[req.rid] = count - 1

    def assert_overlap_compute_admission(self) -> None:
        if getattr(self, "_target_compute_batches_inflight", 0) >= 2:
            raise RuntimeError(
                "Target compute admission exceeded the one-executing plus "
                "one-submitted bound"
            )

    def has_target_inflight(self) -> bool:
        return bool(self._target_inflight_counts)

    def pop_retractable_for_oom(self) -> list[Req]:
        """Cancel remote waits and detach synchronized Target snapshots."""

        states = [
            *self._parked.values(),
            *self._feature_control_rounds.values(),
        ]
        for state in states:
            self._parked.pop(state.transfer_id, None)
            self._feature_control_rounds.pop(state.transfer_id, None)
            self._forget_request_lane(state.req)
            self._next_round.pop(state.req.rid, None)

        inflight_reqs = list(self._target_inflight_reqs.values())
        self._target_inflight_counts.clear()
        self._target_inflight_reqs.clear()

        if self.is_export_rank and states:
            from dt_protocol import (
                CANCEL_TRANSFERS,
                PROTOCOL_VERSION,
                CancelTransfers,
                cancel_transfers_id,
            )

            transfer_ids = tuple(state.transfer_id for state in states)
            self._cancelled_transfers.update(transfer_ids)
            self._drop_cancelled_unsent()
            if self.is_token_only:
                for transfer_id in transfer_ids:
                    if self._ledger.get(transfer_id) is not None:
                        self._ledger.finish(transfer_id)
                        self._cancelled_transfers.discard(transfer_id)
            cancellation = CancelTransfers(
                protocol_version=PROTOCOL_VERSION,
                target_instance_id=self.instance_id,
                transfer_ids=transfer_ids,
                reason="kv_retract",
            )
            self._delivery_outbox.add(
                CANCEL_TRANSFERS,
                cancel_transfers_id(cancellation),
                cancellation,
            )
            self._flush_reliable_outbox()

        unique: dict[int, Req] = {}
        for req in [*(state.req for state in states), *inflight_reqs]:
            unique[id(req)] = req
        for req in unique.values():
            self._finish_verified_request(req.rid)
            self._release_draft_request(req.rid)
            if req.finished():
                self._lifecycle_attempt_by_request.pop(req.rid, None)
            else:
                self._advance_lifecycle_attempt(req.rid)
        if self.is_export_rank:
            self._flush_reliable_outbox()
        return list(unique.values())

    def cleanup_running_after_oom(
        self,
        *,
        retracted_reqs: list[Req],
        aborted_reqs: list[Req],
    ) -> None:
        """Release V2 Draft lifecycles removed by generic decode retraction."""

        if not self.is_persistent:
            return
        aborted_ids = {id(req) for req in aborted_reqs}
        unique = {id(req): req for req in [*retracted_reqs, *aborted_reqs]}
        affected_rids = {req.rid for req in unique.values()}
        control_states = [
            state
            for state in self._feature_control_rounds.values()
            if state.req.rid in affected_rids
        ]
        parked_states = [
            state for state in self._parked.values() if state.req.rid in affected_rids
        ]
        for state in control_states:
            self._feature_control_rounds.pop(state.transfer_id, None)
        for state in parked_states:
            self._parked.pop(state.transfer_id, None)
        cancelled_states = [*control_states, *parked_states]
        if self.is_export_rank and cancelled_states:
            from dt_protocol import (
                CANCEL_TRANSFERS,
                PROTOCOL_VERSION,
                CancelTransfers,
                cancel_transfers_id,
            )

            transfer_ids = tuple(state.transfer_id for state in cancelled_states)
            self._cancelled_transfers.update(transfer_ids)
            self._drop_cancelled_unsent()
            cancellation = CancelTransfers(
                protocol_version=PROTOCOL_VERSION,
                target_instance_id=self.instance_id,
                transfer_ids=transfer_ids,
                reason="kv_retract",
            )
            self._delivery_outbox.add(
                CANCEL_TRANSFERS,
                cancel_transfers_id(cancellation),
                cancellation,
            )
        for req_id, req in unique.items():
            self._forget_request_lane(req)
            self._next_round.pop(req.rid, None)
            self._finish_verified_request(req.rid)
            self._release_draft_request(req.rid)
            if req_id in aborted_ids:
                self._lifecycle_attempt_by_request.pop(req.rid, None)
            else:
                self._advance_lifecycle_attempt(req.rid)
        if unique and self.is_export_rank:
            self._flush_reliable_outbox()

    @staticmethod
    def _freeze_key(batch: ScheduleBatch) -> tuple[bool, int]:
        if batch.forward_mode.is_extend():
            return True, int(batch.extend_num_tokens or 0).bit_length()
        return False, len(batch.reqs)

    def _record_freeze_launch(self, launched_batch: ScheduleBatch) -> None:
        """Remember a launch so its GPU busy time can be learned (TP0 only)."""
        if not self._late_freeze or not self.scheduler.result_queue:
            return
        result = self.scheduler.result_queue[-1][1]
        if getattr(result, "copy_done", None) is None:
            return
        launch_ns = self._overlap_launch_ns
        key = self._freeze_key(launched_batch)
        if self._last_poll_ns:
            # Poll packet -> launch covers selection, build and submission:
            # the time the next batch needs once its members are frozen.
            # Prefill and decode differ by several milliseconds.
            lead = launch_ns - self._last_poll_ns
            previous = self._freeze_lead_ema_ns.get(key[0])
            self._freeze_lead_ema_ns[key[0]] = (
                lead if previous is None else 0.8 * previous + 0.2 * lead
            )
        self._freeze_inflight.append([result, launch_ns, key, 0])
        while len(self._freeze_inflight) > 4:
            self._freeze_inflight.popleft()

    def _freeze_record(self, result) -> list | None:
        for record in self._freeze_inflight:
            if record[0] is result:
                return record
        return None

    def _late_freeze_hold(self, overlap_result, result_ready) -> bool:
        """Return whether the next batch should stay unfrozen (TP0 decision)."""
        if not self._late_freeze or overlap_result is None:
            return False
        record = self._freeze_record(overlap_result)
        if record is None:
            return False
        now_ns = time.monotonic_ns()
        if result_ready:
            if not record[3]:
                record[3] = now_ns
            return False
        busy_ns = self._freeze_busy_ema_ns.get(record[2])
        if busy_ns is None:
            return False
        start_ns = max(record[1], self._freeze_last_done_ns)
        if now_ns - start_ns > 2 * busy_ns:
            return False
        # The next batch is normally another decode wave.
        lead_ns = self._freeze_lead_ema_ns.get(False)
        if lead_ns is None:
            return False
        lead_ns += self._freeze_margin_ns
        return now_ns < start_ns + busy_ns - lead_ns

    def note_overlap_result_done(self, result) -> None:
        """Learn a batch's GPU busy time when its result is consumed (TP0)."""
        if not self.is_export_rank or not self._late_freeze:
            return
        record = self._freeze_record(result)
        if record is None:
            return
        while self._freeze_inflight and self._freeze_inflight[0] is not record:
            self._freeze_inflight.popleft()
        self._freeze_inflight.popleft()
        copy_done = result.copy_done
        exact = not copy_done.query()
        if exact:
            copy_done.synchronize()
        done_ns = record[3] or time.monotonic_ns()
        exact = exact or bool(record[3])
        busy_ns = done_ns - max(record[1], self._freeze_last_done_ns)
        self._freeze_last_done_ns = done_ns
        previous = self._freeze_busy_ema_ns.get(record[2])
        if previous is None:
            self._freeze_busy_ema_ns[record[2]] = float(busy_ns)
        elif exact or busy_ns < previous:
            # A late observation is only an upper bound on the busy time.
            self._freeze_busy_ema_ns[record[2]] = 0.8 * previous + 0.2 * busy_ns

    def should_defer_overlap_result(
        self, *, next_batch: ScheduleBatch | None, last_batch: ScheduleBatch | None
    ) -> bool:
        """Keep the previous result pending while another lane approaches READY.

        SGLang's generic overlap loop resolves a result immediately when the
        next scheduler turn has no batch. DT can have another full-wave lane
        already executing remotely, so bounded polling can turn that otherwise
        blocking result synchronization into useful producer-consumer overlap.
        ``_last_poll_ns`` and ``_overlap_launch_ns`` come from the TP0 poll
        packet, keeping the defer decision identical on every TP rank.
        """
        if self.continuous_batching:
            # TP0 broadcasts copy_done readiness in poll(). Keep receiving and
            # preparing independent work while this result is on the GPU. Once
            # ready, publish promptly, even if no next batch can be submitted.
            # There is still at most one old result when no batch is launched;
            # the ordinary overlap loop bounds the queue to two on submission.
            return bool(
                next_batch is None
                and last_batch is not None
                and self._overlap_result_ready is False
            )
        elapsed_ns = max(0, self._last_poll_ns - self._overlap_launch_ns)
        should_defer = bool(
            self.overlap_result_defer_ns > 0
            and next_batch is None
            and last_batch is not None
            and self._parked
            and not self._overlap_result_ready
            and elapsed_ns < self.overlap_result_defer_ns
        )
        return should_defer

    def _hold_ready_batch(self, ready_count: int) -> bool:
        """Return whether to keep a small READY set open for returning rows.

        Verify cost grows slowly with batch size, so idling the GPU briefly
        until rows just handed to a Draft come back beats running them as a
        separate small wave.  All inputs are TP-consistent: ``_last_poll_ns``
        comes from the TP0 poll packet and ``_parked`` from its ordered events.
        """
        wait_ns = getattr(self, "_ready_batch_wait_ns", 0)
        if wait_ns <= 0 or ready_count >= getattr(self, "_ready_batch_min_size", 1):
            self._ready_hold_started_ns = 0
            return False
        outstanding = any(
            state.candidate_bytes is None
            and state.candidate_completion is None
            and not state.failure
            and not state.req.finished()
            and not state.req.is_retracted
            for state in self._parked.values()
        )
        if not outstanding:
            self._ready_hold_started_ns = 0
            return False
        now_ns = self._last_poll_ns
        if not self._ready_hold_started_ns:
            self._ready_hold_started_ns = now_ns
        if now_ns - self._ready_hold_started_ns < wait_ns:
            return True
        self._ready_hold_started_ns = 0
        return False

    def build_ready_batch(self) -> ScheduleBatch | None:
        observer = getattr(self, "_overlap_fix_observer", None)
        observe_build = observer is not None and observer.enabled()
        trace_build = bool(self.is_export_rank and self._profile_active)
        build_entry_ns = time.monotonic_ns() if (observe_build or trace_build) else 0
        build_cpu_start_ns = time.thread_time_ns() if observe_build else 0
        ready = [
            state
            for state in self._parked.values()
            if state.candidate_bytes is not None
            or state.candidate_completion is not None
        ]
        if self.continuous_batching:
            # Dict insertion order is shared by TP ranks through poll's ordered
            # control packet. Route affinity remains wire metadata only; it is
            # never a readiness or membership barrier. Decode allocation and
            # OOM/retraction remain owned by Scheduler.update_running_batch.
            ready = [
                state for state in ready
                if state.draft_admitted
                and not state.failure
                and not state.req.finished()
                and not state.req.is_retracted
                and not self._target_inflight_count(state.req.rid)
            ]
            if ready and self._hold_ready_batch(len(ready)):
                return None
            if ready:
                # Reuse the allocator's actual page/reservation estimate;
                # do not retract independent WAIT_DRAFT owners merely because
                # every READY row does not fit together. This is host metadata
                # only, with no GPU tensor construction or Req mutation.
                budget_batch = ScheduleBatch(
                    reqs=[state.req for state in ready],
                    token_to_kv_pool_allocator=self.scheduler.token_to_kv_pool_allocator,
                    tree_cache=self.scheduler.tree_cache,
                    spec_algorithm=self.scheduler.spec_algorithm,
                )
                budget_batch.check_decode_mem()  # existing evictable-cache policy
                available = self.scheduler.token_to_kv_pool_allocator.available_size()
                selected = []
                for index, state in enumerate(ready):
                    needed = budget_batch.new_tokens_required_next_decode([index])
                    if needed <= available:
                        selected.append(state)
                        available -= needed
                        if len(selected) == self.scheduler.max_running_requests:
                            break
                if not selected and not self._target_inflight_counts:
                    # No GPU work can release more capacity. Hand one row to
                    # the existing synchronized OOM/retraction path for progress;
                    # it must pass check_decode_mem again before actual submit.
                    selected = ready[:1]
                ready = selected
        elif ready:
            ready = self._select_pipeline_lane_ready(ready)
        if not ready:
            return None

        started_ns = time.monotonic_ns()

        for state in ready:
            self._parked.pop(state.transfer_id)
            if self.is_persistent:
                existing = self._verify_transfer_by_request.get(state.req.rid)
                if existing is not None and existing != state.transfer_id:
                    raise RuntimeError(
                        "DT request already has a different round in Target verify"
                    )
                self._verify_transfer_by_request[state.req.rid] = state.transfer_id

        reqs = [state.req for state in ready]
        # Reserve membership/order at removal from _parked. Generic decode
        # may still filter/retract for OOM; submission freezes its surviving
        # order. No host control polling intervenes before prepare/submit. The
        # queued ScheduleBatch snapshot and per-request inflight counts retain
        # ownership until its result is processed. -1 means unpartitioned; it
        # is a diagnostic value, never an index into a resource slot array.
        lane_id = -1 if self.continuous_batching else ready[0].lane_id
        if not self.continuous_batching and any(
            state.lane_id != lane_id for state in ready
        ):
            raise RuntimeError("DT ready verification batch spans pipeline lanes")
        device = self.scheduler.device
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.scheduler.req_to_token_pool,
            token_to_kv_pool_allocator=self.scheduler.token_to_kv_pool_allocator,
            tree_cache=self.scheduler.tree_cache,
            model_config=self.scheduler.model_config,
            enable_overlap=False,
            spec_algorithm=self.scheduler.spec_algorithm,
        )
        batch.dt_pipeline_lane_id = lane_id
        initialized_ns = time.monotonic_ns()
        breakdown_active = bool(
            self.is_export_rank and getattr(self, "_profile_active", False)
        )
        pin_ready_inputs = is_pin_memory_available(device)
        batch.req_pool_indices_cpu = torch.tensor(
            [req.req_pool_idx for req in reqs],
            dtype=torch.int64,
            pin_memory=pin_ready_inputs,
        )
        req_pool_cpu_ns = (
            time.monotonic_ns() if breakdown_active else initialized_ns
        )
        batch.req_pool_indices = _ready_batch_h2d(
            batch.req_pool_indices_cpu, device=device
        )
        req_pool_device_ns = (
            time.monotonic_ns() if breakdown_active else req_pool_cpu_ns
        )
        batch.seq_lens_cpu = torch.tensor(
            [state.seq_len for state in ready],
            dtype=torch.int64,
            pin_memory=pin_ready_inputs,
        )
        seq_lens_cpu_ns = (
            time.monotonic_ns() if breakdown_active else req_pool_device_ns
        )
        batch.seq_lens = _ready_batch_h2d(batch.seq_lens_cpu, device=device)
        seq_lens_device_ns = (
            time.monotonic_ns() if breakdown_active else seq_lens_cpu_ns
        )
        batch.orig_seq_lens = batch.seq_lens.to(dtype=torch.int32)
        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
        seq_lens_derived_ns = (
            time.monotonic_ns() if breakdown_active else seq_lens_device_ns
        )
        bonus_tokens_cpu = torch.tensor(
            [state.bonus_token for state in ready],
            dtype=torch.int64,
            pin_memory=pin_ready_inputs,
        )
        bonus_tokens = _ready_batch_h2d(bonus_tokens_cpu, device=device)
        bonus_tokens_ns = (
            time.monotonic_ns() if breakdown_active else seq_lens_derived_ns
        )

        direct_candidates = any(
            state.candidate_completion is not None for state in ready
        )
        if direct_candidates and any(
            state.candidate_completion is None for state in ready
        ):
            raise RuntimeError("one Target batch cannot mix candidate transports")
        if direct_candidates:
            completions = [state.candidate_completion for state in ready]
            if self.is_export_rank:
                if self._candidate_arena is None:
                    raise RuntimeError("direct candidate arena is unavailable")
                wire_tokens = self._candidate_arena.begin_read(completions)
            else:
                wire_tokens = torch.empty(
                    (len(ready), self.block_size),
                    dtype=torch.int32,
                    device=device,
                )
            candidate_cpu_ns = (
                time.monotonic_ns() if breakdown_active else bonus_tokens_ns
            )
            if self.scheduler.ps.tp_size > 1:
                self.scheduler.tp_group.broadcast(wire_tokens, src=0)
            draft_tokens = wire_tokens.to(dtype=torch.int64)
            if self.is_export_rank:
                self._candidate_arena.finish_read(completions)
                if not self.is_persistent:
                    for state in ready:
                        if self._ledger.get(state.transfer_id) is not None:
                            self._ledger.finish(state.transfer_id)
            batch.dt_ready_h2d_sources = (bonus_tokens_cpu,)
        else:
            candidate_blob = b"".join(state.candidate_bytes for state in ready)
            expected_bytes = len(ready) * self.block_size * 4
            if len(candidate_blob) != expected_bytes:
                raise ValueError("DT candidate payload width does not match block size")
            draft_tokens_cpu = _ready_candidate_host_tensor(
                candidate_blob,
                batch_size=len(ready),
                block_size=self.block_size,
                pin_memory=pin_ready_inputs,
            )
            candidate_cpu_ns = (
                time.monotonic_ns() if breakdown_active else bonus_tokens_ns
            )
            draft_tokens = _ready_batch_h2d(draft_tokens_cpu, device=device)
            # req_pool_indices_cpu and seq_lens_cpu already live on ScheduleBatch.
            # Retain the two otherwise-local pinned sources until this batch has
            # launched, rather than depending on host-allocator reuse fencing.
            batch.dt_ready_h2d_sources = (bonus_tokens_cpu, draft_tokens_cpu)
        candidate_device_ns = (
            time.monotonic_ns() if breakdown_active else candidate_cpu_ns
        )
        batch.spec_info = DFlashDraftInputV2(
            topk_p=torch.empty((len(ready), 0), dtype=torch.float32, device=device),
            topk_index=torch.empty((len(ready), 0), dtype=torch.int64, device=device),
            bonus_tokens=bonus_tokens,
            new_seq_lens=batch.seq_lens,
            # The remote EAGLE3 Target verifier consumes only the complete
            # candidate chain.  Its prefill handoff initializes FutureMap's
            # otherwise-unused hidden-state relay with width zero, so keep the
            # READY handoff on that same schema.  The real 3-layer features
            # travel directly through dt_feature_chunks to the remote Draft.
            hidden_states=torch.empty(
                (len(ready), 0 if self.is_eagle3 else self.pivot_width),
                dtype=torch.bfloat16,
                device=device,
            ),
            draft_tokens=draft_tokens,
        )
        tensors_ready_ns = time.monotonic_ns()
        batch.input_ids = None
        self.scheduler.future_map.stash(
            batch.req_pool_indices,
            RelayPayload(
                bonus_tokens=bonus_tokens,
                # FutureMap derives its relay schema from the configured
                # speculative algorithm.  EAGLE3 therefore requires top-k
                # tensors even though this Target receives a complete remote
                # chain and never consumes local Draft logits.
                topk_p=batch.spec_info.topk_p,
                topk_index=batch.spec_info.topk_index,
                hidden_states=batch.spec_info.hidden_states,
            ),
        )
        relay_ready_ns = time.monotonic_ns()
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch, self.scheduler.model_config.vocab_size
        )
        finished_ns = time.monotonic_ns()
        result_received_ns = (
            [state.result_received_ns for state in ready if state.result_received_ns]
            if breakdown_active
            else ()
        )
        result_batch_ids = (
            sorted(
                {
                    state.result_batch_id
                    for state in ready
                    if state.result_batch_id is not None
                }
            )
            if breakdown_active
            else ()
        )
        wave_id = self._next_full_wave_id
        self._next_full_wave_id += 1
        batch.dt_full_wave_id = wave_id
        batch.dt_full_wave_ready_ns = finished_ns
        if trace_build:
            batch.dt_full_wave_transfer_ids = tuple(
                state.transfer_id for state in ready
            )
        if observe_build and observer.enabled(finished_ns):
            observer.build(
                ready, wave_id, build_entry_ns, started_ns, finished_ns,
                time.thread_time_ns() - build_cpu_start_ns,
                scheduling_mode=("continuous" if self.continuous_batching else "fixed_lane"),
                batch_lane_id=lane_id,
            )
        if self.eagle3_host_span_profile_enabled:
            batch.dt_eagle3_ready_build_started_ns = started_ns
            batch.dt_full_wave_transfer_ids = tuple(
                state.transfer_id for state in ready
            )
        if self.is_export_rank:
            self._profile_emit(
                "target_ready_batch",
                wave_id=wave_id,
                lane_id=lane_id,
                batch_size=len(ready),
                transfer_ids=tuple(state.transfer_id for state in ready),
                selection_started_ns=build_entry_ns,
                selection_finished_ns=started_ns,
                build_started_ns=started_ns,
                build_finished_ns=finished_ns,
                result_batch_ids=result_batch_ids,
                oldest_result_age_ns=(
                    finished_ns - min(result_received_ns)
                    if result_received_ns
                    else 0
                ),
                newest_result_age_ns=(
                    finished_ns - max(result_received_ns)
                    if result_received_ns
                    else 0
                ),
                initialize_ns=initialized_ns - started_ns,
                tensors_ns=tensors_ready_ns - initialized_ns,
                req_pool_cpu_ns=req_pool_cpu_ns - initialized_ns,
                req_pool_device_ns=req_pool_device_ns - req_pool_cpu_ns,
                seq_lens_cpu_ns=seq_lens_cpu_ns - req_pool_device_ns,
                seq_lens_device_ns=seq_lens_device_ns - seq_lens_cpu_ns,
                seq_lens_derived_ns=seq_lens_derived_ns - seq_lens_device_ns,
                bonus_tokens_ns=bonus_tokens_ns - seq_lens_derived_ns,
                candidate_cpu_ns=candidate_cpu_ns - bonus_tokens_ns,
                candidate_device_ns=candidate_device_ns - candidate_cpu_ns,
                spec_tensors_ns=tensors_ready_ns - candidate_device_ns,
                relay_ns=relay_ready_ns - tensors_ready_ns,
                sampling_ns=finished_ns - relay_ready_ns,
                total_ns=finished_ns - started_ns,
            )
        return batch

    def should_defer_prefill(
        self, *, ready_batch_size: int, chunked_req: Req | None
    ) -> bool:
        # Bootstrap features may retain slots while Draft owner admission is
        # full. Keep one slot for existing owners to verify or finish a chunk.
        # A locally staged KV handle is not evidence of Draft admission.
        if self.is_feature_backed and (
            self._feature_source_free_slots - self._deferred_dflash_feature_stages
            <= 1
        ):
            if chunked_req is None or self._draft_admitted_attempts.get(
                chunked_req.rid
            ) != self._lifecycle_attempt_by_request.get(chunked_req.rid, -1):
                return True
        # Once a frozen wave is ready, run it before admitting replacement
        # prefills.  Otherwise get_new_batch_prefill wins scheduler priority,
        # the replacement becomes the next frozen wave by itself, and the
        # already-built large verify batch remains stranded in running_batch.
        return chunked_req is None and ready_batch_size > 0

    def pop_failures(self) -> list[ParkedTargetRound]:
        failed = [state for state in self._parked.values() if state.failure]
        if not failed:
            return []

        # Chunked feature prefill can leave several control/proposal rounds for
        # one request in the controller at once.  A failure is terminal for the
        # current request attempt: retire every sibling before the next poll is
        # allowed to dispatch an unsent successor, and report the Req only once
        # to Scheduler so its KV allocation is released exactly once.
        failed_by_lifecycle: dict[tuple[str, int], ParkedTargetRound] = {}
        for state in failed:
            failed_by_lifecycle.setdefault((state.req.rid, state.attempt), state)

        current_failures = []
        for (request_id, attempt), state in failed_by_lifecycle.items():
            is_current = self._lifecycle_attempt_by_request.get(request_id) == attempt
            self._retire_dflash_rounds_for_request(
                request_id,
                reason="remote_draft_failed",
                attempt=attempt,
            )
            if not is_current:
                continue
            self._forget_request_lane(state.req)
            self._next_round.pop(request_id, None)
            self._finish_verified_request(request_id)
            self._release_draft_request(request_id)
            self._lifecycle_attempt_by_request.pop(request_id, None)
            current_failures.append(state)
        if current_failures and self.is_export_rank:
            self._flush_reliable_outbox()
        return current_failures

    def pop_aborted(self, *, abort_all: bool, rid_prefix: str) -> list[Req]:
        from dt_protocol import (
            CANCEL_TRANSFERS,
            PROTOCOL_VERSION,
            CancelTransfers,
            cancel_transfers_id,
        )

        matched = [
            state
            for state in (
                *self._parked.values(),
                *self._feature_control_rounds.values(),
            )
            if abort_all or state.req.rid.startswith(rid_prefix)
        ]
        if not matched:
            return []
        for state in matched:
            self._parked.pop(state.transfer_id, None)
            self._feature_control_rounds.pop(state.transfer_id, None)
            self._forget_request_lane(state.req)
            self._next_round.pop(state.req.rid, None)
            self._finish_verified_request(state.req.rid)
            self._release_draft_request(state.req.rid)
            self._lifecycle_attempt_by_request.pop(state.req.rid, None)
        if self.is_export_rank:
            transfer_ids = {state.transfer_id for state in matched}
            self._cancelled_transfers.update(transfer_ids)
            self._drop_cancelled_unsent()
            for transfer_id in tuple(transfer_ids):
                self._release_candidate_destination(transfer_id)
                if (
                    transfer_id not in self._source_slot_by_transfer
                    and self._ledger.get(transfer_id) is not None
                ):
                    self._ledger.finish(transfer_id)
                    self._cancelled_transfers.discard(transfer_id)
            cancellation = CancelTransfers(
                protocol_version=PROTOCOL_VERSION,
                target_instance_id=self.instance_id,
                transfer_ids=tuple(state.transfer_id for state in matched),
                reason="client_abort",
            )
            self._delivery_outbox.add(
                CANCEL_TRANSFERS,
                cancel_transfers_id(cancellation),
                cancellation,
            )
            self._flush_reliable_outbox()
        return list({id(state.req): state.req for state in matched}.values())

    def _finish_verified_request(self, request_id: str) -> None:
        transfer_id = self._verify_transfer_by_request.pop(request_id, None)
        if (
            self.is_export_rank
            and transfer_id is not None
            and self._ledger.get(transfer_id) is not None
        ):
            self._ledger.finish(transfer_id)

    def __len__(self) -> int:
        return len(self._parked)

    def _lane_for_request(self, req: Req) -> int:
        existing = self._lane_by_request.get(req.rid)
        if existing is not None:
            return existing

        live_counts = Counter(self._lane_by_request.values())
        lanes_with_refill_gap = [
            candidate
            for candidate in range(self.pipeline_lanes)
            if live_counts[candidate] < self._lane_target_sizes[candidate]
        ]
        if lanes_with_refill_gap:
            # Preserve a lane's learned target size across terminal rows.  The
            # most under-filled lane receives the next replacement request.
            lane_id = min(
                lanes_with_refill_gap,
                key=lambda candidate: (
                    live_counts[candidate] - self._lane_target_sizes[candidate],
                    (candidate - self._lane_assignment_cursor) % self.pipeline_lanes,
                ),
            )
        else:
            # Before target sizes are learned, apply the configured admission
            # ratio. Request ownership remains immutable after this choice.
            lane_id = min(
                range(self.pipeline_lanes),
                key=lambda candidate: (
                    live_counts[candidate] / self.pipeline_lane_weights[candidate],
                    (candidate - self._lane_assignment_cursor) % self.pipeline_lanes,
                ),
            )
        self._lane_by_request[req.rid] = lane_id
        req.dt_pipeline_lane_id = lane_id
        self._lane_assignment_cursor = (lane_id + 1) % self.pipeline_lanes
        return lane_id

    def _forget_request_lane(self, req: Req) -> None:
        self._lane_by_request.pop(req.rid, None)
        if hasattr(req, "dt_pipeline_lane_id"):
            del req.dt_pipeline_lane_id

    def close(self) -> None:
        previous_failure = getattr(
            self, "_source_shutdown_quarantine_error", None
        )
        if previous_failure is not None:
            raise RuntimeError(
                "DT Target shutdown remains fenced after an incomplete direct-GPU "
                "source drain; uncertain pool ownership is retained until process "
                "teardown"
            ) from previous_failure

        source_pool = self._feature_source_pool
        source_pool_fenced = False
        shutdown_drained = False
        try:
            if self.is_export_rank and (
                self.is_persistent or self.uses_direct_gpu_source
            ):
                self._drain_token_only_shutdown()
            if source_pool is not None:
                # Controller maps are not the complete ownership record. A
                # publication can fail before descriptors reach this class, or
                # a later exception can remove those mappings after the pool
                # quarantines its slot. Fence admission and inspect the pool's
                # own ownership atomically before dropping its last reference.
                active_leases, quarantined_slots = source_pool.close()
                source_pool_fenced = True
                if active_leases or quarantined_slots:
                    raise RuntimeError(
                        "DT Target shutdown found unresolved direct-GPU source "
                        "pool ownership after controller drain: "
                        f"active_leases={active_leases}, "
                        f"quarantined_slots={quarantined_slots}"
                    )
            self._close_direct_rings()
            if self._candidate_arena is not None:
                active_candidates, quarantined_candidates = (
                    self._candidate_arena.close()
                )
                if active_candidates or quarantined_candidates:
                    raise RuntimeError(
                        "DT Target shutdown found unresolved direct candidate "
                        "ownership: "
                        f"active={active_candidates}, "
                        f"quarantined={quarantined_candidates}"
                    )
            shutdown_drained = True
        except BaseException as exc:
            if source_pool is not None and self.uses_direct_gpu_source:
                # Commit the ownership quarantine before any fallible host-side
                # cleanup. TargetFeatureSourcePool.close() only fences future
                # publication; it deliberately retains storage, events, and
                # active leases. Keep the pool attached to this controller so
                # none of those objects can be reclaimed before process exit.
                self._source_shutdown_quarantine_error = exc
                if not source_pool_fenced:
                    try:
                        source_pool.close()
                    except Exception:
                        logger.exception(
                            "failed to fence DT Target source publication after an "
                            "incomplete shutdown drain; ownership remains retained"
                        )
            raise
        finally:
            for store in self._result_stores.values():
                store.close()
            self._result_stores.clear()
            if self._client is not None:
                self._client.close()
            if shutdown_drained and self._feature_source_pool is not None:
                if not source_pool_fenced:
                    self._feature_source_pool.close()
                self._feature_source_pool = None
            if shutdown_drained:
                self._candidate_arena = None

    def _drain_token_only_shutdown(self) -> None:
        from dt_protocol import (
            CANCEL_TRANSFERS,
            PROTOCOL_VERSION,
            CancelTransfers,
            cancel_transfers_id,
        )

        # These graph-output preservation leases were never submitted to the
        # Coordinator, so no remote reader can exist and no control ACK can
        # retire them.  Return them directly to the bounded source pool before
        # assembling cancellation for Coordinator-owned transfers.
        for transfer_id in tuple(self._staged_flashmtp_source_leases):
            self._release_source_slot(transfer_id)

        # Ring-published waves still awaiting confirmation hold feature-source
        # reservations that were never published (candidate-arena exhaustion
        # holds them, or the kernel record has not landed), so they are absent
        # from _source_slot_by_transfer above.  No confirmation tasks were
        # queued for them, so no remote reader can exist; return each
        # reservation so pool close() finds no RESERVED slots.
        pending_waves = getattr(self, "_ring_publish_pending", None)
        if pending_waves:
            for wave in tuple(pending_waves):
                self._feature_source_pool.release_reserved(
                    wave.slot_id, wave.generation
                )
            pending_waves.clear()

        unsent_transfer_ids = {
            task.transfer_id for batch in self._pending_send for task in batch.entries
        }
        for transfer_id in unsent_transfer_ids:
            self._release_source_slot(transfer_id)
        transfer_ids = tuple(
            dict.fromkeys(
                transfer_id
                for transfer_id in [
                    *self._parked,
                    *self._feature_control_rounds,
                    *self._verify_transfer_by_request.values(),
                    *self._source_slot_by_transfer,
                ]
                if transfer_id not in unsent_transfer_ids
            )
        )

        # No new request work may overtake shutdown control. Task batches that
        # reached the Coordinator are retired by the cancellation below; those
        # still in _pending_send were never submitted and can simply be dropped.
        self._pending_send.clear()
        for operation, message_id, _message in self._delivery_outbox.pending():
            if operation == "task_batch":
                self._delivery_outbox.ack(operation, message_id)
        for transfer_id in transfer_ids:
            if self._ledger.get(transfer_id) is not None:
                self._ledger.finish(transfer_id)
        self._parked.clear()
        self._feature_control_rounds.clear()
        self._verify_transfer_by_request.clear()
        self._lane_by_request.clear()
        self._next_round.clear()
        self._target_inflight_counts.clear()
        self._target_inflight_reqs.clear()
        self._lifecycle_attempt_by_request.clear()

        if transfer_ids:
            cancellation = CancelTransfers(
                protocol_version=PROTOCOL_VERSION,
                target_instance_id=self.instance_id,
                transfer_ids=transfer_ids,
                reason="target_shutdown",
            )
            self._delivery_outbox.add(
                CANCEL_TRANSFERS,
                cancel_transfers_id(cancellation),
                cancellation,
            )
        for request_id in tuple(self._draft_kv_handle_by_request):
            self._release_draft_request(request_id)

        deadline_ns = time.monotonic_ns() + self.round_timeout_ns
        while len(self._delivery_outbox) or (
            self.uses_direct_gpu_source and self._source_slot_by_transfer
        ):
            self._flush_reliable_outbox()
            self._consume_shutdown_acks()
            if not len(self._delivery_outbox) and not (
                self.uses_direct_gpu_source and self._source_slot_by_transfer
            ):
                return
            now_ns = time.monotonic_ns()
            if now_ns >= deadline_ns:
                pending = tuple(
                    (operation, message_id)
                    for operation, message_id, _message in (
                        self._delivery_outbox.pending()
                    )
                )
                raise RuntimeError(
                    "DT Target shutdown cleanup timed out with unacknowledged "
                    f"messages: {pending!r}"
                )
            time.sleep(min(0.01, (deadline_ns - now_ns) / 1_000_000_000))

    def _consume_shutdown_acks(self) -> None:
        from dt_protocol import (
            ControlAck,
            ErrorMessage,
            SourceReadDone,
            SourceReadDoneBatch,
            TargetResultBatch,
            TaskBatchAck,
        )

        for message in self._client.recv_nowait():
            if isinstance(message, ControlAck):
                self._delivery_outbox.ack(message.operation, message.message_id)
                if message.operation == "source_read_done":
                    batch_entries = self._source_done_batch_entries.pop(
                        message.message_id, ()
                    )
                    if batch_entries:
                        self._source_done_applied_ids.difference_update(batch_entries)
                    else:
                        self._source_done_applied_ids.discard(message.message_id)
            elif isinstance(message, TaskBatchAck):
                # Request work has already been retired locally. A late task
                # ACK cannot restart it and needs no retry bookkeeping.
                self._delivery_outbox.ack("task_batch", message.batch_id)
            elif isinstance(message, TargetResultBatch):
                if message.buffer is not None:
                    self._enqueue_pull_done(message)
            elif isinstance(message, SourceReadDone):
                self._consume_source_read_done(message)
            elif isinstance(message, SourceReadDoneBatch):
                self._consume_source_read_done_batch(message)
            elif isinstance(message, ErrorMessage):
                if message.code == "PermissionError":
                    self._delivery_outbox.retry_all_now()
                logger.warning(
                    "Draft Coordinator error during shutdown code=%s "
                    "related_id=%s message=%s",
                    message.code,
                    message.related_id,
                    message.message,
                )
            else:
                raise TypeError(
                    "unexpected Target Coordinator shutdown message "
                    f"{type(message).__name__}"
                )

    def _make_task(
        self,
        *,
        transfer_id: str,
        req: Req,
        round_id: int,
        attempt: int,
        bonus_token: int,
        previous_transfer_id: str | None,
        feature_descriptor=None,
        feature_token_ids: tuple[int, ...] = (),
        committed_token_ids: tuple[int, ...] = (),
        feature_token_offset: int = 0,
        requires_proposal: bool = True,
    ):
        from dt_protocol import DraftTask, DraftTaskKind, SourceRef

        now_ns = time.monotonic_ns()
        if self.is_token_only:
            full_sequence = tuple(req.origin_input_ids) + tuple(req.output_ids)
            if not full_sequence:
                raise ValueError("DT Draft bootstrap requires a non-empty sequence")
            if len(full_sequence) != int(req.kv_committed_len) + 1:
                raise RuntimeError(
                    "DT Target token sequence does not match its committed KV boundary"
                )
            if full_sequence[-1] != bonus_token:
                raise RuntimeError(
                    "DT Target authoritative sequence does not end in the bonus token"
                )
            if round_id == 0:
                if req.rid in self._draft_kv_handle_by_request:
                    raise RuntimeError("DT bootstrap already has a Draft KV handle")
                draft_kv_handle = f"{self.instance_id}:{req.rid}:{attempt}"
                kind = DraftTaskKind.BOOTSTRAP
                committed_token_ids = full_sequence
            else:
                draft_kv_handle = self._draft_kv_handle_by_request.get(req.rid)
                previous_len = self._draft_committed_len.get(req.rid)
                if (
                    draft_kv_handle is None
                    or previous_len is None
                    or previous_transfer_id is None
                ):
                    raise RuntimeError(
                        "DT reconcile is missing the previous committed boundary"
                    )
                if len(full_sequence) <= previous_len:
                    raise RuntimeError(
                        "DT Target verification produced no committed token delta"
                    )
                self._ledger.mark_need_reconcile(previous_transfer_id)
                committed_token_ids = full_sequence[previous_len:]
                kind = DraftTaskKind.RECONCILE
            self._draft_committed_len[req.rid] = len(full_sequence)
            if round_id == 0:
                self._draft_kv_handle_by_request[req.rid] = draft_kv_handle
            task = DraftTask(
                transfer_id=transfer_id,
                request_id=req.rid,
                round_id=round_id,
                attempt=attempt,
                target_instance_id=self.instance_id,
                draft_profile_id=self.profile_id,
                k=self.block_size - 1,
                next_token_id=bonus_token,
                source=None,
                created_ns=now_ns,
                deadline_ns=now_ns + self.round_timeout_ns,
                kind=kind,
                draft_kv_handle=draft_kv_handle,
                committed_token_ids=committed_token_ids,
                committed_kv_len=len(full_sequence),
            )
            if round_id > 0:
                self._ledger.finish(previous_transfer_id)
            return task

        if self.is_feature_backed:
            if feature_descriptor is None:
                raise ValueError("feature-backed task requires a CUDA IPC descriptor")
            if not feature_token_ids or not committed_token_ids:
                raise ValueError("feature-backed task requires token metadata")
            if len(feature_token_ids) != len(committed_token_ids):
                raise ValueError("feature and committed token counts differ")
            committed_kv_len = feature_token_offset + len(feature_token_ids)
            staged_boundary = self._draft_committed_len.get(req.rid)
            dispatched_boundary = self._draft_dispatched_len.get(req.rid, 0)
            if (
                staged_boundary is None
                or staged_boundary < committed_kv_len
                or dispatched_boundary != feature_token_offset
            ):
                raise RuntimeError(
                    "feature task boundary differs from Target committed state"
                )
            draft_kv_handle = self._draft_kv_handle_by_request.get(req.rid)
            if draft_kv_handle is None:
                raise RuntimeError("feature task is missing its persistent KV handle")
            task = DraftTask(
                transfer_id=transfer_id,
                request_id=req.rid,
                round_id=round_id,
                attempt=attempt,
                target_instance_id=self.instance_id,
                draft_profile_id=self.profile_id,
                k=self.block_size - 1,
                next_token_id=bonus_token,
                source=SourceRef(**feature_descriptor.source_ref_kwargs()),
                created_ns=now_ns,
                deadline_ns=now_ns + self.round_timeout_ns,
                kind=(
                    DraftTaskKind.BOOTSTRAP
                    if round_id == 0
                    else DraftTaskKind.RECONCILE
                ),
                draft_kv_handle=draft_kv_handle,
                committed_token_ids=committed_token_ids,
                feature_token_ids=feature_token_ids,
                committed_kv_len=committed_kv_len,
                feature_token_offset=feature_token_offset,
                feature_token_count=len(feature_token_ids),
                requires_proposal=requires_proposal,
                pipeline_lane_id=self._lane_for_request(req),
            )
            self._draft_dispatched_len[req.rid] = committed_kv_len
            return task

        if feature_descriptor is None:
            raise ValueError("FlashMTP task requires a CUDA IPC pivot descriptor")
        expected_bytes = self.pivot_width * 2
        if feature_descriptor.nbytes != expected_bytes:
            raise ValueError(
                "DT Target expected "
                f"{expected_bytes} pivot bytes, got {feature_descriptor.nbytes}"
            )
        if (
            feature_descriptor.transport != "cuda-ipc"
            or feature_descriptor.dtype != "bfloat16"
            or tuple(feature_descriptor.shape) != (1, self.pivot_width)
        ):
            raise ValueError("FlashMTP task requires one fenced CUDA IPC BF16 pivot")
        return DraftTask(
            transfer_id=transfer_id,
            request_id=req.rid,
            round_id=round_id,
            attempt=attempt,
            target_instance_id=self.instance_id,
            draft_profile_id=self.profile_id,
            k=self.block_size - 1,
            next_token_id=bonus_token,
            source=SourceRef(**feature_descriptor.source_ref_kwargs()),
            created_ns=now_ns,
            deadline_ns=now_ns + self.round_timeout_ns,
        )

    def _queue_task_batch(self, tasks: tuple) -> None:
        from dt_protocol import PROTOCOL_VERSION, TargetTaskBatch

        self._batch_sequence += 1
        batch = TargetTaskBatch(
            protocol_version=PROTOCOL_VERSION,
            batch_id=f"{self.instance_id}:tasks:{self._batch_sequence}",
            target_instance_id=self.instance_id,
            entries=tasks,
            sent_ns=time.monotonic_ns(),
        )
        if self.uses_direct_gpu_source:
            for task in tasks:
                if task.source is not None:
                    self._source_deadline_by_transfer[task.transfer_id] = (
                        task.deadline_ns
                    )
        self._pending_send.append(batch)

    def _prune_obsolete_task_retries(self, tasks) -> None:
        """Retire exact local copies; remote leases and ledger owners stay intact."""
        if not getattr(self, "_pending_send", None):
            return
        from msgspec.structs import replace

        obsolete = {task.transfer_id: task for task in tasks}
        if not obsolete:
            return
        retained = deque()
        for batch in self._pending_send:
            entries = []
            for task in batch.entries:
                authoritative = obsolete.get(task.transfer_id)
                if authoritative is None:
                    entries.append(task)
                elif task != authoritative:
                    raise _PendingTaskPayloadMismatch(
                        f"queued transfer {task.transfer_id!r} changed its task payload"
                    )
            if len(entries) == len(batch.entries):
                retained.append(batch)
            elif entries:
                retained.append(replace(batch, entries=tuple(entries)))
        # Commit only after every matching copy passed the full equality check.
        self._pending_send = retained

    def _flush_pending_send(self) -> None:
        from dt_protocol import TargetTaskBatch
        from msgspec.structs import replace

        self._flush_reliable_outbox()
        while self._pending_send:
            dispatchable = []
            selected_by_request = {}
            blocked = None
            candidate_budget = None
            if self.candidate_transport == "cuda-ipc":
                if not self.is_export_rank or self._candidate_arena is None:
                    raise RuntimeError("direct candidate arena is unavailable")
                candidate_budget = self._candidate_arena.free_slot_count
            pending_batches = (
                (self._pending_send[0],)
                if self.whole_batch_task_fifo_ablation
                else self._pending_send
            )
            for pending_batch in pending_batches:
                for task in pending_batch.entries:
                    active_transfer_id = self._ledger.active_transfer_id(
                        task.request_id
                    )
                    queued_predecessor = selected_by_request.get(task.request_id)
                    if queued_predecessor is not None or (
                        active_transfer_id is not None
                        and active_transfer_id != task.transfer_id
                    ):
                        if blocked is None:
                            blocked = (
                                pending_batch,
                                task,
                                active_transfer_id or queued_predecessor.transfer_id,
                            )
                        if self.whole_batch_task_fifo_ablation:
                            dispatchable.clear()
                            break
                        continue
                    if (
                        candidate_budget is not None
                        and task.requires_proposal
                        and task.candidate_destination is None
                    ):
                        if candidate_budget <= 0:
                            continue
                        candidate_budget -= 1
                    dispatchable.append(task)
                    selected_by_request[task.request_id] = task
                if blocked is not None and self.whole_batch_task_fifo_ablation:
                    break

            if blocked is not None:
                # Chunked DFlash prefill produces the next authoritative
                # feature range before the remote Draft necessarily finishes
                # materializing the preceding one.  Preserve FIFO per request,
                # but do not strand independent descriptors in the same or a
                # later pending batch behind that request.
                pending_batch, task, active_transfer_id = blocked
                block_key = (
                    pending_batch.batch_id,
                    task.transfer_id,
                    active_transfer_id,
                )
                if block_key != self._last_pending_send_block:
                    active_state = self._ledger.get(active_transfer_id)
                    self._profile_emit(
                        "target_task_fifo_block",
                        batch_id=pending_batch.batch_id,
                        entries=len(pending_batch.entries),
                        eligible_entries=len(dispatchable),
                        pending_batches=len(self._pending_send),
                        request_id=task.request_id,
                        blocked_transfer_id=task.transfer_id,
                        blocked_round_id=task.round_id,
                        active_transfer_id=active_transfer_id,
                        active_round_id=(
                            active_state.task.round_id
                            if active_state is not None
                            else None
                        ),
                        active_phase=(
                            active_state.phase.value
                            if active_state is not None
                            else None
                        ),
                        active_requires_proposal=(
                            active_state.task.requires_proposal
                            if active_state is not None
                            else None
                        ),
                    )
                    self._last_pending_send_block = block_key

            if not dispatchable:
                break

            reserved = []
            if candidate_budget is not None:
                with_destinations = []
                try:
                    for task in dispatchable:
                        if (
                            not task.requires_proposal
                            or task.candidate_destination is not None
                        ):
                            with_destinations.append(task)
                            continue
                        destination = self._candidate_arena.reserve(task.transfer_id)
                        reserved.append(task.transfer_id)
                        with_destinations.append(
                            replace(task, candidate_destination=destination)
                        )
                except Exception:
                    for transfer_id in reserved:
                        self._candidate_arena.cancel(
                            transfer_id, remote_quiescent=True
                        )
                    raise
                dispatchable = with_destinations
            ring_plan = ()
            if getattr(self, "_direct_ring", False):
                dispatchable, ring_plan = self._plan_direct_ring(dispatchable)

            self._batch_sequence += 1
            dispatch_batch = TargetTaskBatch(
                protocol_version=self._pending_send[0].protocol_version,
                batch_id=f"{self.instance_id}:tasks:{self._batch_sequence}",
                target_instance_id=self.instance_id,
                entries=tuple(dispatchable),
                sent_ns=time.monotonic_ns(),
            )
            try:
                submitted = self._client.submit(dispatch_batch)
            except Exception:
                for transfer_id in reserved:
                    self._candidate_arena.cancel(
                        transfer_id, remote_quiescent=True
                    )
                raise
            if not submitted:
                for transfer_id in reserved:
                    self._candidate_arena.cancel(
                        transfer_id, remote_quiescent=True
                    )
                break
            submitted_ns = time.monotonic_ns()
            if ring_plan:
                self._write_direct_ring(ring_plan)
            if self._profile_active:
                self._profile_emit(
                    "target_task_submit",
                    batch_id=dispatch_batch.batch_id,
                    entries=len(dispatch_batch.entries),
                    transfer_ids=tuple(
                        task.transfer_id for task in dispatch_batch.entries
                    ),
                    proposal_transfer_ids=tuple(
                        task.transfer_id
                        for task in dispatch_batch.entries
                        if task.requires_proposal
                    ),
                    materialization_transfer_ids=tuple(
                        task.transfer_id
                        for task in dispatch_batch.entries
                        if not task.requires_proposal
                    ),
                    feature_rows=tuple(
                        int(task.feature_token_count)
                        for task in dispatch_batch.entries
                    ),
                    submitted_ns=submitted_ns,
                    submit_call_ns=submitted_ns - dispatch_batch.sent_ns,
                    oldest_task_age_ns=(
                        submitted_ns
                        - min(task.created_ns for task in dispatch_batch.entries)
                    ),
                    newest_task_age_ns=(
                        submitted_ns
                        - max(task.created_ns for task in dispatch_batch.entries)
                    ),
                )
            self._ledger.begin(dispatch_batch)
            lease_marks = getattr(self, "_lease_marks", None)
            if lease_marks:
                # A ring copy written before a submit that did not go out is
                # marked once its (re)submitted round is registered.
                retained_marks = []
                for transfer_id, instance_id in lease_marks:
                    if self._ledger.get(transfer_id) is None:
                        retained_marks.append((transfer_id, instance_id))
                    else:
                        self._ledger.mark_lease_dispatched(
                            transfer_id, draft_instance_id=instance_id
                        )
                self._lease_marks = retained_marks
            outbox = self._delivery_outbox
            outbox.add("task_batch", dispatch_batch.batch_id, dispatch_batch)
            outbox.mark_sent("task_batch", dispatch_batch.batch_id, now_ns=submitted_ns)

            dispatched_ids = {task.transfer_id for task in dispatchable}
            retained = deque()
            for pending_batch in self._pending_send:
                entries = tuple(
                    task
                    for task in pending_batch.entries
                    if task.transfer_id not in dispatched_ids
                )
                if entries:
                    retained.append(
                        TargetTaskBatch(
                            protocol_version=pending_batch.protocol_version,
                            batch_id=pending_batch.batch_id,
                            target_instance_id=pending_batch.target_instance_id,
                            entries=entries,
                            sent_ns=pending_batch.sent_ns,
                        )
                    )
            self._pending_send = retained
            if blocked is None:
                self._last_pending_send_block = None

    def _poll_export_rank(self) -> list[tuple[str, str, object]]:
        from dt_protocol import (
            ControlAck,
            ErrorMessage,
            SourceReadDone,
            SourceReadDoneBatch,
            TargetResultBatch,
            TaskBatchAck,
        )

        if getattr(self, "_ring_publish_pending", None):
            # Deferred §7.1 bookkeeping: Coordinator confirmation tasks for
            # kernel-published waves, off the park critical path.
            self._ring_publish_cleanup()
        self._flush_pending_send()

        events: list[tuple[str, str, object]] = []
        if self._candidate_arena is not None:
            # Direct Draft completion is intentionally observed before the
            # independent Coordinator retirement channel.
            self._candidate_arena.poll_completion_notices()
            self._candidate_arena.poll_completion_records()
            if self._ring_outstanding:
                self._poll_direct_ring_reads()
                if self._ring_lease:
                    self._fence_dead_ring_owners()
        for message in self._client.recv_nowait():
            if isinstance(message, TaskBatchAck):
                if not self._ledger.has_submitted_batch(message.batch_id):
                    continue
                self._delivery_outbox.ack("task_batch", message.batch_id)
                retry_tasks = self._ledger.apply_ack(message)
                if self.candidate_transport == "cuda-ipc":
                    if self._candidate_arena is None:
                        raise RuntimeError("direct candidate arena is unavailable")
                    from sglang.srt.speculative.dt_candidate_p2p import (
                        CandidateTransportError,
                    )

                    for assignment in message.assignments:
                        try:
                            self._candidate_arena.authorize(assignment)
                        except CandidateTransportError:
                            if assignment.transfer_id not in self._ring_unproven and (
                                assignment.transfer_id not in self._direct_sent
                            ):
                                raise
                            # The Coordinator named another producer than the
                            # owner lease used for this ring round: the slot is
                            # quarantined (no READY), the round times out.
                            logger.error(
                                "Coordinator assignment for ring round %s differs "
                                "from the owner lease",
                                assignment.transfer_id,
                            )
                        self._remember_draft_binding(assignment)
                    direct_sent = getattr(self, "_direct_sent", None)
                    if direct_sent:
                        direct_sent.difference_update(message.accepted_transfer_ids)
                        direct_sent.difference_update(message.duplicate_transfer_ids)
                self._prune_obsolete_task_retries(
                    state.task
                    for transfer_id in (
                        *message.accepted_transfer_ids,
                        *message.duplicate_transfer_ids,
                    )
                    if (state := self._ledger.get(transfer_id)) is not None
                )
                if self.is_persistent:
                    for transfer_id in (
                        *message.accepted_transfer_ids,
                        *message.duplicate_transfer_ids,
                    ):
                        state = self._ledger.get(transfer_id)
                        if state is not None and state.dispatch_accepted:
                            events.append(("accepted", transfer_id, None))
                active_retries = []
                for task in retry_tasks:
                    if task.transfer_id in self._cancelled_transfers:
                        self._ledger.finish(task.transfer_id)
                        self._retire_unaccepted_transfer(task.transfer_id)
                        self._cancelled_transfers.discard(task.transfer_id)
                    else:
                        active_retries.append(task)
                if active_retries:
                    self._queue_task_batch(tuple(active_retries))
            elif isinstance(message, TargetResultBatch):
                if self._profile_active:
                    consume_started_ns = time.monotonic_ns()
                    result_events = self._consume_result_batch(message)
                    consume_finished_ns = time.monotonic_ns()
                    self._profile_emit(
                        "target_result_receive",
                        batch_id=message.batch_id,
                        draft_instance=message.draft_instance_id,
                        transfer_ids=tuple(
                            entry.transfer_id for entry in message.entries
                        ),
                        ready_transfer_ids=tuple(
                            transfer_id
                            for kind, transfer_id, _payload in result_events
                            if kind == "ready"
                        ),
                        materialized_transfer_ids=tuple(
                            transfer_id
                            for kind, transfer_id, _payload in result_events
                            if kind == "materialized"
                        ),
                        applied_transfer_ids=tuple(
                            transfer_id
                            for _kind, transfer_id, _payload in result_events
                        ),
                        entries=len(message.entries),
                        ready_count=sum(
                            kind == "ready"
                            for kind, _transfer_id, _ in result_events
                        ),
                        failed_count=sum(
                            kind == "failed"
                            for kind, _transfer_id, _ in result_events
                        ),
                        materialized_count=sum(
                            kind == "materialized"
                            for kind, _transfer_id, _ in result_events
                        ),
                        draft_to_target_ns=consume_finished_ns - message.sent_ns,
                        consume_ns=consume_finished_ns - consume_started_ns,
                    )
                else:
                    result_events = self._consume_result_batch(message)
                events.extend(result_events)
            elif isinstance(message, SourceReadDone):
                self._consume_source_read_done(message)
            elif isinstance(message, SourceReadDoneBatch):
                self._consume_source_read_done_batch(message)
            elif isinstance(message, ControlAck):
                self._delivery_outbox.ack(message.operation, message.message_id)
                if message.operation == "source_read_done":
                    batch_entries = self._source_done_batch_entries.pop(
                        message.message_id, ()
                    )
                    if batch_entries:
                        self._source_done_applied_ids.difference_update(batch_entries)
                    else:
                        self._source_done_applied_ids.discard(message.message_id)
            elif isinstance(message, ErrorMessage):
                if message.code == "PermissionError":
                    self._client.register_target(
                        timeout_s=self.server_args.draft_target_registration_timeout_s
                    )
                    self._delivery_outbox.retry_all_now()
                logger.warning(
                    "Draft Coordinator error code=%s related_id=%s message=%s",
                    message.code,
                    message.related_id,
                    message.message,
                )
            else:
                raise TypeError(
                    f"unexpected Target Coordinator message {type(message).__name__}"
                )
        if self._candidate_arena is not None:
            self._candidate_arena.poll_releases()
            self._candidate_arena.poll_cancelled()
            for completion in self._candidate_arena.poll():
                state = self._ledger.get(completion.transfer_id)
                if state is None:
                    continue
                parked = self._parked.get(completion.transfer_id)
                if parked is None:
                    continue
                if parked.candidate_completion is not None:
                    continue
                if completion.root_token != parked.bonus_token:
                    raise ValueError(
                        "direct candidate root does not match Target bonus token"
                    )
                self._ledger.mark_direct_ready(
                    completion.transfer_id,
                    draft_instance_id=completion.draft_instance_id,
                )
                parked.result_received_ns = completion.observed_ns
                if self._profile_active:
                    export_ledger_ready_ns = time.monotonic_ns()
                    self._profile_emit(
                        "target_direct_completion",
                        transfer_id=completion.transfer_id,
                        request_id=parked.req.rid,
                        round_id=parked.round_id,
                        slot_id=completion.slot_id,
                        generation=completion.generation,
                        notice_queued_ns=completion.notice_queued_ns,
                        notice_received_ns=completion.notice_received_ns,
                        metadata_submitted_ns=completion.metadata_submitted_ns,
                        metadata_gpu_ns=completion.metadata_gpu_ns,
                        completion_observed_ns=completion.observed_ns,
                        ready_source=getattr(completion, "ready_source", "notice"),
                        producer_globaltimer_ns=getattr(
                            completion, "producer_globaltimer_ns", 0
                        ),
                        delivery_ns=getattr(completion, "delivery_ns", 0),
                        export_ledger_ready_ns=export_ledger_ready_ns,
                        target_compute_batches_inflight=self._target_compute_batches_inflight,
                    )
                events.append(
                    (
                        "ready_direct",
                        completion.transfer_id,
                        (completion, parked.bonus_token, parked.seq_len),
                    )
                )
        events.extend(self._expire_local_rounds())
        self._expire_source_leases(now_ns=time.monotonic_ns())
        self._maybe_emit_lifecycle_status()
        return events

    def _lifecycle_status(self) -> dict[str, object]:
        ledger = (
            {"rounds": 0, "request_bindings": 0, "submitted_batches": 0}
            if self._ledger is None
            else self._ledger.lifecycle_counts()
        )
        feature_pool = self._feature_source_pool
        active = {
            **{f"ledger_{name}": value for name, value in ledger.items()},
            "parked_rounds": len(self._parked),
            "feature_control_rounds": len(self._feature_control_rounds),
            "target_inflight_forwards": sum(self._target_inflight_counts.values()),
            "target_compute_batches_inflight": self._target_compute_batches_inflight,
            "target_inflight_requests": len(self._target_inflight_reqs),
            "pending_send_batches": len(self._pending_send),
            "pending_send_tasks": sum(
                len(batch.entries) for batch in self._pending_send
            ),
            "delivery_outbox": len(self._delivery_outbox),
            "draft_kv_handles": len(self._draft_kv_handle_by_request),
            "draft_committed_boundaries": len(self._draft_committed_len),
            "draft_dispatched_boundaries": len(self._draft_dispatched_len),
            "lifecycle_attempts": len(self._lifecycle_attempt_by_request),
            "next_round_requests": len(self._next_round),
            "verify_transfers": len(self._verify_transfer_by_request),
            "source_leases": len(self._source_slot_by_transfer),
            "source_deadlines": len(self._source_deadline_by_transfer),
            "source_done_applied": len(self._source_done_applied_ids),
            "source_done_batches": len(self._source_done_batch_entries),
            "cancelled_transfers": len(self._cancelled_transfers),
            "request_lanes": len(self._lane_by_request),
            "early_staging_in_use": 0,
            "deferred_feature_stages": int(self._deferred_dflash_feature_stages),
            "feature_source_leases": (
                0 if feature_pool is None else feature_pool.active_lease_count
            ),
        }
        fatal = {
            "feature_source_quarantined_slots": (
                0 if feature_pool is None else feature_pool.quarantined_slot_count
            )
        }
        resources = {
            "source_slots_free": 0,
            "feature_source_slots_free": (
                0 if feature_pool is None else feature_pool.free_slot_count
            ),
            "feature_source_slots_total": (
                0 if feature_pool is None else feature_pool.slot_count
            ),
            "early_staging_slots_free": 0,
            "early_staging_slots_total": 0,
        }
        return {
            "schema_version": "dt-lifecycle-status/v1",
            "role": "target",
            "instance_id": self.instance_id,
            "pid": os.getpid(),
            "observed_ns": time.monotonic_ns(),
            "sequence": self._lifecycle_status_sequence,
            "instance_session_id": getattr(self._client, "instance_session_id", ""),
            "coordinator_session_id": getattr(
                self._client, "coordinator_session_id", ""
            ),
            "source_pool_generation": (
                "" if feature_pool is None else feature_pool.pool_generation
            ),
            "direct_gpu_measurement_epochs": (
                self._direct_gpu_measurement_epoch_evidence()
            ),
            "active": active,
            "fatal": fatal,
            "resources": resources,
            "terminal": {},
            "quiescent": not any(active.values()) and not any(fatal.values()),
        }

    def _maybe_emit_lifecycle_status(self) -> None:
        now_ns = time.monotonic_ns()
        if (
            now_ns - self._last_lifecycle_status_sample_ns
            < DT_LIFECYCLE_STATUS_INTERVAL_NS
        ):
            return
        self._last_lifecycle_status_sample_ns = now_ns
        status = self._lifecycle_status()
        quiescent = bool(status["quiescent"])
        became_quiescent = (
            quiescent and self._last_lifecycle_quiescent is False
        )
        if (
            quiescent == self._last_lifecycle_quiescent
            and (not quiescent or now_ns - self._last_lifecycle_status_ns < DT_LIFECYCLE_STATUS_INTERVAL_NS)
        ):
            return
        self._lifecycle_status_sequence += 1
        status["sequence"] = self._lifecycle_status_sequence
        status["observed_ns"] = now_ns
        self._last_lifecycle_status_ns = now_ns
        self._last_lifecycle_quiescent = quiescent
        logger.info(
            "%s%s",
            DT_LIFECYCLE_STATUS_MARKER,
            json.dumps(status, sort_keys=True, separators=(",", ":")),
        )
        if became_quiescent and self.eagle3_host_span_profile_enabled:
            # This transition occurs after a workload window has drained, not
            # in its request hot path.  Expose the final buffered host rows so
            # the outer analyzer can join the complete warmup/measured window.
            self._profile_flush()

    def _flush_reliable_outbox(self) -> None:
        from dt_protocol import (
            CANCEL_TRANSFERS,
            PULL_DONE,
            RELEASE_DRAFT_REQUEST,
            SOURCE_READ_DONE,
        )

        now_ns = time.monotonic_ns()
        outbox = self._delivery_outbox
        for operation, message_id, message in outbox.due(now_ns=now_ns):
            if operation == "task_batch":
                sent = self._client.submit(message)
            elif operation == PULL_DONE:
                sent = self._client.pull_done(message)
            elif operation == CANCEL_TRANSFERS:
                sent = self._client.cancel(message)
            elif operation == RELEASE_DRAFT_REQUEST:
                sent = self._client.release_draft_request(message)
            elif operation == SOURCE_READ_DONE:
                sent = self._client.send(message)
            else:
                raise RuntimeError(f"unknown Target delivery operation {operation!r}")
            if not sent:
                break
            outbox.mark_sent(operation, message_id, now_ns=now_ns)

    def _release_draft_request(self, request_id: str) -> None:
        self._draft_admitted_attempts.pop(request_id, None)
        if not self.is_persistent:
            return
        draft_kv_handle = self._draft_kv_handle_by_request.pop(request_id, None)
        self._draft_committed_len.pop(request_id, None)
        self._draft_dispatched_len.pop(request_id, None)
        # The owner lease ends with the request (a reused id starts fresh).
        getattr(self, "_draft_binding", {}).pop(request_id, None)
        if draft_kv_handle is None:
            return
        if not self.is_export_rank:
            return
        from dt_protocol import (
            PROTOCOL_VERSION,
            RELEASE_DRAFT_REQUEST,
            DraftRequestRelease,
            draft_request_release_id,
        )

        message = DraftRequestRelease(
            protocol_version=PROTOCOL_VERSION,
            target_instance_id=self.instance_id,
            request_id=request_id,
            draft_kv_handle=draft_kv_handle,
            released_ns=time.monotonic_ns(),
        )
        self._delivery_outbox.add(
            RELEASE_DRAFT_REQUEST, draft_request_release_id(message), message
        )

    def _retire_dflash_rounds_for_request(
        self, request_id: str, *, reason: str, attempt: int | None = None
    ) -> None:
        """Cancel remote rounds superseded by a local terminal request state.

        Chunked prefill can leave a materialization-only round in flight when a
        later Target chunk finishes or retracts the request.  Detach those
        rounds before a late failure reaches Scheduler cleanup, while retaining
        every published feature lease until SourceReadDone or quarantine.
        """

        states = [
            state
            for state in (
                *self._parked.values(),
                *self._feature_control_rounds.values(),
            )
            if state.req.rid == request_id
            and (attempt is None or state.attempt == attempt)
        ]
        if not states:
            return
        transfer_ids = tuple(dict.fromkeys(state.transfer_id for state in states))
        for transfer_id in transfer_ids:
            self._parked.pop(transfer_id, None)
            self._feature_control_rounds.pop(transfer_id, None)
        if not self.is_export_rank:
            return

        from dt_protocol import (
            CANCEL_TRANSFERS,
            PROTOCOL_VERSION,
            CancelTransfers,
            cancel_transfers_id,
        )

        newly_cancelled = tuple(
            transfer_id
            for transfer_id in transfer_ids
            if transfer_id not in self._cancelled_transfers
        )
        self._cancelled_transfers.update(newly_cancelled)
        self._drop_cancelled_unsent()
        for transfer_id in transfer_ids:
            if (
                transfer_id not in self._source_slot_by_transfer
                and self._ledger.get(transfer_id) is not None
            ):
                self._ledger.finish(transfer_id)
                self._cancelled_transfers.discard(transfer_id)
        if newly_cancelled:
            cancellation = CancelTransfers(
                protocol_version=PROTOCOL_VERSION,
                target_instance_id=self.instance_id,
                transfer_ids=newly_cancelled,
                reason=reason,
            )
            cancellation_id = cancel_transfers_id(cancellation)
            # Token-only timeout cleanup intentionally drops its in-memory
            # cancelled marker immediately, but the reliable timeout message
            # can still be awaiting ACK. Preserve that authoritative payload
            # instead of reusing its ID with a different terminal reason.
            if self._delivery_outbox.get(CANCEL_TRANSFERS, cancellation_id) is None:
                self._delivery_outbox.add(
                    CANCEL_TRANSFERS,
                    cancellation_id,
                    cancellation,
                )
        if self.is_token_only:
            # Token-only transfers own no source lease and late results are
            # stale once their ledger entries are finished.  Do not retain an
            # otherwise-uncollectable cancellation marker after retirement.
            self._cancelled_transfers.difference_update(transfer_ids)

    def _new_round_deadline_ns(self) -> int:
        timeout_ns = getattr(self, "round_timeout_ns", 0)
        return time.monotonic_ns() + timeout_ns if timeout_ns else 0

    def _expire_local_rounds(self) -> list[tuple[str, str, object]]:
        from dt_protocol import (
            CANCEL_TRANSFERS,
            PROTOCOL_VERSION,
            CancelTransfers,
            cancel_transfers_id,
        )

        now_ns = time.monotonic_ns()
        expired = self._ledger.expired(now_ns=now_ns)
        # A round whose Coordinator copy is still queued locally (or, for a
        # kernel-published wave, not even built yet) never entered the ledger,
        # so the ledger cannot time it out.  Its request-visible round started
        # at staging, so the same local deadline applies and fails it without
        # depending on the Coordinator path.
        stranded = [
            state
            for state in (
                *getattr(self, "_parked", {}).values(),
                *getattr(self, "_feature_control_rounds", {}).values(),
            )
            if not state.failure
            and state.deadline_ns
            and state.deadline_ns <= now_ns
            and self._ledger.get(state.transfer_id) is None
        ]
        if not expired and not stranded:
            return []
        transfer_ids = tuple(state.task.transfer_id for state in expired)
        for state in expired:
            transfer_id = state.task.transfer_id
            if self.uses_direct_gpu_source:
                self._quarantine_source_slot(transfer_id)
            self._release_candidate_destination(transfer_id)
            self._ledger.finish(transfer_id)
            if (
                self._lifecycle_attempt_by_request.get(state.task.request_id)
                == state.task.attempt
            ):
                self._release_draft_request(state.task.request_id)
        stranded_ids = tuple(state.transfer_id for state in stranded)
        for state in stranded:
            if (
                self._lifecycle_attempt_by_request.get(state.req.rid)
                == state.attempt
            ):
                self._release_draft_request(state.req.rid)
        self._cancelled_transfers.update(transfer_ids)
        self._cancelled_transfers.update(stranded_ids)
        self._drop_cancelled_unsent()
        if self.is_token_only:
            # Token-only rounds have neither SourceReadDone nor an active
            # ledger entry after timeout.  A late result is therefore stale,
            # so no later event exists (or is needed) to retire this marker.
            self._cancelled_transfers.difference_update(transfer_ids)
            self._cancelled_transfers.difference_update(stranded_ids)
        cancellation = CancelTransfers(
            protocol_version=PROTOCOL_VERSION,
            target_instance_id=self.instance_id,
            transfer_ids=(*transfer_ids, *stranded_ids),
            reason="target_round_timeout",
        )
        self._delivery_outbox.add(
            CANCEL_TRANSFERS,
            cancel_transfers_id(cancellation),
            cancellation,
        )
        self._flush_reliable_outbox()
        return [
            ("failed", transfer_id, "target_round_timeout")
            for transfer_id in (*transfer_ids, *stranded_ids)
        ]

    def _validate_source_read_done(self, message) -> str:
        from dt_protocol import (
            source_read_done_id,
        )

        if message.target_instance_id != self.instance_id:
            raise ValueError("SourceReadDone belongs to another target")
        if (
            not message.draft_instance_id
            or not message.instance_session_id
            or message.coordinator_session_id != self._client.coordinator_session_id
        ):
            raise ValueError(
                "SourceReadDone lacks the current Coordinator/Draft session fence"
            )
        message_id = source_read_done_id(message)
        if message_id in self._source_done_applied_ids:
            return message_id
        if (
            message.transfer_id in getattr(self, "_ring_proven", ())
            and message.transfer_id not in self._source_slot_by_transfer
        ):
            # The ring read proof already retired this lease; the pool's
            # bounded history may no longer remember it.
            return message_id
        source_ref = self._source_slot_by_transfer.get(message.transfer_id)
        if source_ref is not None and source_ref != (
            message.source_slot_id,
            message.slot_generation,
        ):
            raise ValueError(
                "SourceReadDone slot generation does not match the transfer"
            )
        if self.uses_direct_gpu_source:
            if self._feature_source_pool is None:
                raise RuntimeError("direct GPU source pool is unavailable")
            self._feature_source_pool.validate_read_done(
                transfer_id=message.transfer_id,
                source_slot_id=message.source_slot_id,
                slot_generation=message.slot_generation,
                source_session_id=message.source_session_id,
                pool_generation=message.pool_generation,
            )
        return message_id

    def _consume_source_read_done(
        self, message, *, flush: bool = True, prevalidated: bool = False
    ) -> None:
        from dt_protocol import (
            PROTOCOL_VERSION,
            SOURCE_READ_DONE,
            ControlAck,
            source_read_done_id,
        )

        message_id = (
            source_read_done_id(message)
            if prevalidated
            else self._validate_source_read_done(message)
        )
        if message_id in self._source_done_applied_ids:
            if flush:
                self._flush_reliable_outbox()
            return
        ring_retired = (
            message.transfer_id in getattr(self, "_ring_proven", ())
            and message.transfer_id not in self._source_slot_by_transfer
        )
        if ring_retired:
            pass
        elif self.uses_direct_gpu_source:
            assert self._feature_source_pool is not None
            self._feature_source_pool.read_done(
                transfer_id=message.transfer_id,
                source_slot_id=message.source_slot_id,
                slot_generation=message.slot_generation,
                source_session_id=message.source_session_id,
                pool_generation=message.pool_generation,
            )
            self._source_slot_by_transfer.pop(message.transfer_id, None)
            self._source_deadline_by_transfer.pop(message.transfer_id, None)
            staged_leases = getattr(
                self, "_staged_flashmtp_source_leases", None
            )
            if staged_leases is not None:
                staged_leases.discard(message.transfer_id)
        else:
            self._release_source_slot(message.transfer_id)
        if message.transfer_id in self._cancelled_transfers:
            if self._ledger.get(message.transfer_id) is not None:
                self._ledger.finish(message.transfer_id)
            self._cancelled_transfers.discard(message.transfer_id)
        acknowledgement = ControlAck(
            protocol_version=PROTOCOL_VERSION,
            operation=SOURCE_READ_DONE,
            message_id=message_id,
        )
        self._source_done_applied_ids.add(message_id)
        self._delivery_outbox.add(SOURCE_READ_DONE, message_id, acknowledgement)
        if flush:
            self._flush_reliable_outbox()

    def _consume_source_read_done_batch(self, message) -> None:
        from dt_protocol import (
            PROTOCOL_VERSION,
            SOURCE_READ_DONE,
            SourceReadDoneBatchAck,
            source_read_done_id,
        )

        message_ids = tuple(source_read_done_id(entry) for entry in message.entries)
        if not message_ids or len(set(message_ids)) != len(message_ids):
            raise ValueError("SourceReadDoneBatch must contain unique entries")
        profile_source_done = getattr(self, "_profile_active", False)
        if profile_source_done:
            first_applied_transfer_ids = tuple(
                entry.transfer_id
                for entry, message_id in zip(message.entries, message_ids)
                if message_id not in self._source_done_applied_ids
            )
            duplicate_transfer_ids = tuple(
                entry.transfer_id
                for entry, message_id in zip(message.entries, message_ids)
                if message_id in self._source_done_applied_ids
            )
        for entry in message.entries:
            self._validate_source_read_done(entry)
        for entry in message.entries:
            self._consume_source_read_done(
                entry, flush=False, prevalidated=True
            )
            # Replace the per-entry ACK queued by the shared fenced apply path
            # with one reliable batch ACK.
            self._delivery_outbox.ack(SOURCE_READ_DONE, source_read_done_id(entry))
        acknowledgement = SourceReadDoneBatchAck(
            protocol_version=PROTOCOL_VERSION,
            batch_id=message.batch_id,
            draft_instance_id=message.draft_instance_id,
            message_ids=message_ids,
        )
        self._source_done_batch_entries[message.batch_id] = message_ids
        if self._delivery_outbox.get(SOURCE_READ_DONE, message.batch_id) is None:
            self._delivery_outbox.add(
                SOURCE_READ_DONE, message.batch_id, acknowledgement
            )
        else:
            self._delivery_outbox.retry_now(SOURCE_READ_DONE, message.batch_id)
        self._flush_reliable_outbox()
        if profile_source_done:
            self._profile_emit(
                "target_source_done_batch_apply",
                batch_id=message.batch_id,
                draft_instance=message.draft_instance_id,
                transfer_ids=tuple(
                    entry.transfer_id for entry in message.entries
                ),
                first_applied_transfer_ids=first_applied_transfer_ids,
                duplicate_transfer_ids=duplicate_transfer_ids,
                entries=len(message.entries),
            )

    def _expire_source_leases(self, *, now_ns: int) -> None:
        """Quarantine source ranges whose remote read never reached an ACK."""

        for transfer_id, deadline_ns in tuple(
            self._source_deadline_by_transfer.items()
        ):
            if deadline_ns <= now_ns:
                self._quarantine_source_slot(transfer_id)

    def _consume_result_batch(self, batch) -> list[tuple[str, str, object]]:
        try:
            return self._consume_result_batch_body(batch)
        except _PendingTaskPayloadMismatch:
            raise
        except Exception as exc:
            logger.exception(
                "failed to consume DT result buffer; failing its active transfers"
            )
            return self._fail_result_batch(batch, exc)
        finally:
            if batch.buffer is not None:
                self._enqueue_pull_done(batch)

    def _fail_result_batch(self, batch, exc: Exception):
        from dt_protocol import DraftResultEntry, EntryStatus, TargetResultBatch

        failures = []
        for entry in batch.entries:
            state = self._ledger.get(entry.transfer_id)
            if (
                state is None
                or state.phase.value != "wait_draft"
                or entry.request_id != state.task.request_id
                or entry.round_id != state.task.round_id
                or entry.attempt != state.task.attempt
                or entry.target_instance_id != state.task.target_instance_id
            ):
                continue
            failures.append(
                DraftResultEntry(
                    transfer_id=entry.transfer_id,
                    request_id=entry.request_id,
                    round_id=entry.round_id,
                    attempt=entry.attempt,
                    target_instance_id=entry.target_instance_id,
                    status=EntryStatus.FINAL_ERROR,
                    error_code=type(exc).__name__,
                    error_message=str(exc),
                    source_read_quiescent=False,
                )
            )
        if not failures:
            return []
        failure_batch = TargetResultBatch(
            protocol_version=batch.protocol_version,
            batch_id=f"{batch.batch_id}:consume-failure",
            target_instance_id=batch.target_instance_id,
            draft_instance_id=batch.draft_instance_id,
            entries=tuple(failures),
            sent_ns=time.monotonic_ns(),
        )
        disposition = self._ledger.apply_results(failure_batch)
        self._prune_obsolete_task_retries(
            self._ledger.get(transfer_id).task
            for transfer_id in (
                *disposition.ready_transfer_ids,
                *disposition.failed_transfer_ids,
            )
        )
        events = []
        for transfer_id in disposition.failed_transfer_ids:
            if self.uses_direct_gpu_source:
                self._quarantine_source_slot(transfer_id)
            events.append(("failed", transfer_id, str(exc)))
            self._ledger.finish(transfer_id)
        return events

    def _enqueue_pull_done(self, batch) -> None:
        from dt_protocol import (
            PROTOCOL_VERSION,
            PULL_DONE,
            PullDone,
            pull_done_id,
        )

        pull_done = PullDone(
            protocol_version=PROTOCOL_VERSION,
            target_instance_id=self.instance_id,
            draft_instance_id=batch.draft_instance_id,
            buffer_id=batch.buffer.buffer_id,
            generation=batch.buffer.generation,
            row_offset=batch.row_offset,
            row_count=batch.row_count,
            completed_ns=time.monotonic_ns(),
            draft_session_id=batch.buffer.instance_session_id,
            result_pool_generation=batch.buffer.pool_generation,
            instance_session_id=self._client.instance_session_id,
        )
        self._delivery_outbox.add(PULL_DONE, pull_done_id(pull_done), pull_done)
        self._flush_reliable_outbox()

    def _consume_result_batch_body(self, batch) -> list[tuple[str, str, object]]:
        from dt_protocol import EntryStatus
        from dt_transport import MmapSlotStore

        # Lease-safety failures remain actionable after the request result has
        # become stale.  In particular, a Draft can disappear after a timeout
        # result was delivered but before its ordered SourceReadDone reached the
        # Target.  Quarantine first so ledger idempotence cannot hide that late
        # loss of GPU-read quiescence.
        ring_unproven = getattr(self, "_ring_unproven", None) or {}
        if getattr(self, "_ring_lease", False) and getattr(self, "_ring_writers", None):
            batch_session = getattr(batch, "draft_session_id", "")
            lost: dict[str, bool] = {}
            for entry in batch.entries:
                if entry.error_code not in RING_OWNER_LOSS_CODES:
                    continue
                # Coordinator orphan batches may not name a session; the code
                # then refers to the session that holds this ring round.
                session = ring_unproven.get(entry.transfer_id) or batch_session
                if not session or (
                    batch_session and session != batch_session
                ) or session not in self._ring_writers:
                    continue
                lost[session] = lost.get(session, True) and entry.source_read_quiescent
            for draft_session_id, quiescent in lost.items():
                self._revoke_ring_owner(draft_session_id, quiescent=quiescent)
        if self.uses_direct_gpu_source:
            for entry in batch.entries:
                if entry.status == EntryStatus.OK:
                    continue
                if getattr(self, "_ring_lease", False) and entry.transfer_id in ring_unproven:
                    # The Coordinator speaks only for its own copy; the ring
                    # copy's read retires the lease (proof or revocation).
                    continue
                if entry.source_read_quiescent:
                    self._release_source_slot(entry.transfer_id)
                else:
                    self._quarantine_source_slot(entry.transfer_id)

        proposal_ready_ids = {
            entry.transfer_id
            for entry in batch.entries
            if (
                entry.status == EntryStatus.OK
                and (state := self._ledger.get(entry.transfer_id)) is not None
                and state.phase.value == "wait_draft"
                and entry.request_id == state.task.request_id
                and entry.round_id == state.task.round_id
                and entry.attempt == state.task.attempt
                and entry.target_instance_id == state.task.target_instance_id
                and state.task.requires_proposal
                and state.task.candidate_destination is None
            )
        }
        payload = None
        if proposal_ready_ids and batch.buffer is None:
            raise ValueError("ready DT results must carry a published buffer")
        # Target owns every published range until PullDone, including ranges
        # whose entries became stale or cancelled before consumption.
        if proposal_ready_ids and batch.buffer is not None:
            if (
                batch.buffer.draft_instance_id != batch.draft_instance_id
                or not batch.buffer.instance_session_id
                or not batch.buffer.pool_generation
                or batch.draft_session_id != batch.buffer.instance_session_id
                or batch.result_pool_generation != batch.buffer.pool_generation
            ):
                raise ValueError("DT result buffer has an incomplete Draft fence")
            if batch.buffer.transport != "mmap":
                raise ValueError("DT V1 Target supports only mmap result buffers")
            store = self._result_stores.get(batch.buffer.uri)
            if store is None:
                store = MmapSlotStore.open(batch.buffer.uri)
                self._result_stores[batch.buffer.uri] = store
            pull_submitted_ns = time.monotonic_ns()
            payload = store.read(batch.buffer.buffer_id, batch.buffer.generation)
            pull_completed_ns = time.monotonic_ns()
            self._profile_emit(
                "target_candidate_pull",
                draft_instance=batch.draft_instance_id,
                batch_id=batch.batch_id,
                row_count=batch.row_count,
                candidate_bytes=len(payload),
                submitted_ns=pull_submitted_ns,
                completed_ns=pull_completed_ns,
                pull_ns=pull_completed_ns - pull_submitted_ns,
            )
            minimum = (batch.row_offset + batch.row_count) * self.block_size * 4
            if batch.buffer.row_width != self.block_size:
                raise ValueError("DT result row width does not match block size")
            if len(payload) < minimum:
                raise ValueError(
                    "DT result buffer is shorter than its published range: "
                    f"need at least {minimum}, got {len(payload)}"
                )
            row_stride_bytes = self.block_size * 4
            for entry in batch.entries:
                if entry.transfer_id not in proposal_ready_ids:
                    continue
                if not (
                    batch.row_offset
                    <= entry.row_index
                    < batch.row_offset + batch.row_count
                ):
                    raise ValueError(
                        "successful result row is outside the published range"
                    )
                start = entry.row_index * row_stride_bytes
                candidate_bytes = payload[start : start + row_stride_bytes]
                parked = self._parked[entry.transfer_id]
                root = int.from_bytes(
                    candidate_bytes[:4], byteorder="little", signed=True
                )
                if root != parked.bonus_token:
                    raise ValueError(
                        "DT result candidate root does not match bonus token"
                    )

        disposition = self._ledger.apply_results(batch)
        self._prune_obsolete_task_retries(
            self._ledger.get(transfer_id).task
            for transfer_id in (
                *disposition.ready_transfer_ids,
                *disposition.failed_transfer_ids,
            )
        )
        events: list[tuple[str, str, object]] = []

        ready_ids = set(disposition.ready_transfer_ids)
        failed_ids = set(disposition.failed_transfer_ids)
        row_stride_bytes = self.block_size * 4
        for entry in batch.entries:
            transfer_id = entry.transfer_id
            if transfer_id in ready_ids:
                ledger_state = self._ledger.get(transfer_id)
                if ledger_state is None:
                    continue
                if not ledger_state.task.requires_proposal:
                    events.append(("materialized", transfer_id, None))
                    self._ledger.finish(transfer_id)
                    continue
                if transfer_id in self._cancelled_transfers:
                    self._ledger.finish(transfer_id)
                    self._cancelled_transfers.discard(transfer_id)
                    continue
                start = entry.row_index * row_stride_bytes
                candidate_bytes = payload[start : start + row_stride_bytes]
                state = self._parked[transfer_id]
                if self._profile_active:
                    state.result_batch_id = batch.batch_id
                    state.result_received_ns = time.monotonic_ns()
                events.append(
                    (
                        "ready",
                        transfer_id,
                        (candidate_bytes, state.bonus_token, state.seq_len),
                    )
                )
                if not self.is_persistent:
                    self._ledger.finish(transfer_id)
            elif transfer_id in failed_ids:
                if transfer_id in self._cancelled_transfers:
                    self._ledger.finish(transfer_id)
                    self._cancelled_transfers.discard(transfer_id)
                    continue
                events.append(
                    ("failed", transfer_id, entry.error_message or entry.status.value)
                )
                direct_candidate = (
                    self._ledger.get(transfer_id) is not None
                    and self._ledger.get(transfer_id).task.candidate_destination
                    is not None
                )
                self._release_candidate_destination(
                    transfer_id,
                    # Source-read completion proves only that Draft stopped
                    # reading the Target-owned feature source.  It cannot
                    # prove that an attempted direct candidate write stopped.
                    # Direct failures therefore retain/quarantine the row
                    # unless the direct completion endpoint supplies the
                    # authenticated tombstone used by normal cancellation.
                    remote_quiescent=(
                        entry.source_read_quiescent and not direct_candidate
                    ),
                )
                self._ledger.finish(transfer_id)
        return events

    def _drop_cancelled_unsent(self) -> None:
        from dt_protocol import TargetTaskBatch

        retained = deque()
        for batch in self._pending_send:
            entries = tuple(
                task
                for task in batch.entries
                if task.transfer_id not in self._cancelled_transfers
            )
            removed = {
                task.transfer_id
                for task in batch.entries
                if task.transfer_id in self._cancelled_transfers
            }
            for transfer_id in removed:
                self._retire_unaccepted_transfer(transfer_id)
                self._cancelled_transfers.discard(transfer_id)
            if entries:
                retained.append(
                    TargetTaskBatch(
                        protocol_version=batch.protocol_version,
                        batch_id=batch.batch_id,
                        target_instance_id=batch.target_instance_id,
                        entries=entries,
                        sent_ns=batch.sent_ns,
                    )
                )
        self._pending_send = retained

    def _retire_unaccepted_transfer(self, transfer_id: str) -> None:
        """Release a transfer the Coordinator never accepted.

        Without a direct copy no Draft can hold it.  A direct copy may already
        be pulling or pushing, so keep both rows until the Draft's
        authenticated completion or tombstone retires the candidate slot.
        """
        direct_sent = getattr(self, "_direct_sent", None)
        if direct_sent and transfer_id in direct_sent:
            # A ring copy exists: its candidate slot waits for the Draft's
            # completion or tombstone, its source for the ring read proof or
            # an owner revocation.
            direct_sent.discard(transfer_id)
            self._release_candidate_destination(transfer_id, remote_quiescent=False)
            return
        self._release_source_slot(transfer_id)
        self._release_candidate_destination(transfer_id, remote_quiescent=True)

    def _remember_draft_binding(self, assignment) -> None:
        bindings = getattr(self, "_draft_binding", None)
        state = self._ledger.get(assignment.transfer_id)
        if bindings is None or state is None:
            return
        previous = bindings.get(state.task.request_id)
        epoch = int(getattr(assignment, "owner_epoch", 0))
        if previous is not None and epoch and epoch < previous[2]:
            return  # stale assignment from an older owner epoch
        if assignment.draft_session_id in getattr(self, "_ring_revoked_sessions", ()):
            bindings.pop(state.task.request_id, None)
            return
        bindings[state.task.request_id] = (
            assignment.draft_instance_id,
            assignment.draft_session_id,
            epoch,
        )
        if len(bindings) > 65536:
            # Dropping a binding only disables the direct ring for that request.
            for request_id in tuple(bindings)[:16384]:
                del bindings[request_id]

    def _dflash_wave_scalars(self, batch_result, draft_input, *, deferred: bool):
        """Per-request accept/bonus/seq-len ints for the commit specs.

        The export rank reuses the deferred stage's pinned copies; other
        ranks (and the non-deferred path) take one small async D2H batch of
        their own.  Either way the wait lands on work enqueued before the
        forward completed, so it does not gate the park critical path.
        """

        staged = (
            deferred
            and getattr(batch_result, "dt_dflash_accept_lens_cpu", None) is not None
        )
        if staged:
            error = getattr(batch_result, "dt_deferred_feature_error", None)
            done = getattr(batch_result, "dt_deferred_feature_copy_done", None)
            if done is not None:
                done.synchronize()
            if error is not None:
                raise RuntimeError(error)
            accept_cpu = batch_result.dt_dflash_accept_lens_cpu
            bonus_cpu = getattr(batch_result, "dt_dflash_bonus_tokens_cpu", None)
            seq_lens_cpu = getattr(batch_result, "dt_dflash_seq_lens_cpu", None)
            if bonus_cpu is None or seq_lens_cpu is None:
                raise RuntimeError("deferred result lost scalar staging tensors")
        else:
            if deferred:
                error = getattr(batch_result, "dt_deferred_feature_error", None)
                if error is not None:
                    raise RuntimeError(error)
            from sglang.srt.managers.utils import _async_d2h

            accept_lens = getattr(batch_result, "accept_lens", None)
            bonus_tokens = getattr(draft_input, "bonus_tokens", None)
            new_seq_lens = getattr(draft_input, "new_seq_lens", None)
            if (
                accept_lens is None
                or bonus_tokens is None
                or not bonus_tokens.is_cuda
                or new_seq_lens is None
                or not new_seq_lens.is_cuda
            ):
                raise RuntimeError("verify wave lost its GPU scalar tensors")
            if accept_lens.is_cuda:
                accept_cpu = _async_d2h(accept_lens)
            else:
                # Result processing may already have moved the commit counts
                # to host memory; no copy is needed then.
                accept_cpu = accept_lens
            bonus_cpu = _async_d2h(bonus_tokens)
            seq_lens_cpu = _async_d2h(new_seq_lens)
            done = self.scheduler.device_module.Event()
            done.record()
            done.synchronize()
        # Zero-copy numpy views: callers index individual values with int()
        # and stage the counts with one memcpy; building Python lists here
        # costs ~10 µs per array on every park.
        return (
            accept_cpu.numpy(),
            bonus_cpu.numpy(),
            seq_lens_cpu.numpy(),
        )

    def _dflash_staged_meta_and_specs(
        self, batch, feature_offsets, counts, accept_values, bonus_values, seq_len_values
    ):
        """One batch pass producing staged_meta and commit specs together.

        Content-identical to `_dflash_staged_meta` + `_dflash_commit_specs`
        run back to back; the verify park is the only hot caller.
        """

        active = []
        staged_meta = []
        specs = []
        next_round = self._next_round
        committed = self._draft_committed_len
        kv_handles = self._draft_kv_handle_by_request
        verify_transfers = self._verify_transfer_by_request
        for index, req in enumerate(batch.reqs):
            count = int(counts[index])
            if req.finished() or req.is_retracted or count <= 0:
                continue
            offset = int(feature_offsets[index])
            round_id = next_round.get(req.rid, 0)
            previous_committed = committed.get(req.rid)
            attempt = self._ensure_lifecycle_attempt(req.rid)
            if round_id == 0:
                if offset != 0 or previous_committed is not None:
                    raise RuntimeError("DFLASH bootstrap must begin at offset zero")
                draft_kv_handle = f"{self.instance_id}:{req.rid}:" + str(attempt)
            else:
                draft_kv_handle = kv_handles.get(req.rid)
                if draft_kv_handle is None or previous_committed != offset:
                    raise RuntimeError(
                        "DFLASH feature range has a committed-prefix gap"
                    )
            active.append(index)
            staged_meta.append((index, req, offset, round_id, attempt, draft_kv_handle))
            specs.append(
                (
                    req.rid,
                    f"{self.instance_id}:{req.rid}:{round_id}:{attempt}",
                    round_id,
                    attempt,
                    draft_kv_handle,
                    offset,
                    count,
                    (),
                    (),
                    verify_transfers.get(req.rid),
                    int(bonus_values[index]),
                    int(seq_len_values[index]),
                )
            )
        return active, staged_meta, specs

    def _dflash_staged_meta(self, batch, feature_offsets, counts):
        """Active indices and per-request staging scalars, with the Python
        publication path's round/offset fences, derived identically on every
        rank from local batch state and ledger dicts."""

        active = [
            index
            for index, req in enumerate(batch.reqs)
            if not req.finished() and not req.is_retracted and counts[index] > 0
        ]
        staged_meta = []
        for index in active:
            req = batch.reqs[index]
            offset = int(feature_offsets[index])
            round_id = self._next_round.get(req.rid, 0)
            previous_committed = self._draft_committed_len.get(req.rid)
            attempt = self._ensure_lifecycle_attempt(req.rid)
            if round_id == 0:
                if offset != 0 or previous_committed is not None:
                    raise RuntimeError("DFLASH bootstrap must begin at offset zero")
                draft_kv_handle = f"{self.instance_id}:{req.rid}:" + str(attempt)
            else:
                draft_kv_handle = self._draft_kv_handle_by_request.get(req.rid)
                if draft_kv_handle is None or previous_committed != offset:
                    raise RuntimeError(
                        "DFLASH feature range has a committed-prefix gap"
                    )
            staged_meta.append((index, req, offset, round_id, attempt, draft_kv_handle))
        return active, staged_meta

    def _dflash_commit_specs(
        self, staged_meta, accept_values, bonus_values, seq_len_values
    ) -> list:
        """The 12-field commit specs (empty token-id tuples on all ranks)."""

        return [
            (
                req.rid,
                f"{self.instance_id}:{req.rid}:{round_id}:{attempt}",
                round_id,
                attempt,
                draft_kv_handle,
                offset,
                int(accept_values[index]),
                (),
                (),
                self._verify_transfer_by_request.get(req.rid),
                int(bonus_values[index]),
                int(seq_len_values[index]),
            )
            for index, req, offset, round_id, attempt, draft_kv_handle in staged_meta
        ]

    def _ring_publish_publisher(self, draft_session_id: str):
        """Return the GPU publish channel for one serving owner ring, or None."""

        if not getattr(self, "_ring_publish", False):
            return None
        publisher = getattr(self, "_ring_publishers", {}).get(draft_session_id)
        if publisher is not None:
            return publisher
        writer = self._direct_ring_writer(draft_session_id)
        pool = self._feature_source_pool
        if writer is None or pool is None:
            return None
        from sglang.srt.speculative.dt_direct_ring import RingGpuPublisher

        publisher = RingGpuPublisher(
            writer,
            runtime=pool._runtime,
            device=pool._storage.get_device(),
            max_batch_slots=int(self.scheduler.max_running_requests),
            block_size=self.block_size,
            row_bytes=self.pivot_width * 2,
        )
        writers = getattr(self, "_ring_publishers", None)
        if writers is None:
            writers = self._ring_publishers = {}
        writers[draft_session_id] = publisher
        return publisher

    def _stage_dflash_ring_publish(
        self,
        batch: ScheduleBatch,
        batch_result,
        draft_input: DFlashDraftInputV2,
        feature_chunks,
        token_chunks,
        *,
        deferred: bool,
        staged_meta,
        accept_values,
        bonus_values,
        seq_len_values,
        commit_specs,
    ):
        """Publish one verify wave as a kernel-written ring record (§7.1).

        Returns True when the wave went out on the ring, or None when the
        wave is ineligible and the caller must fall back to the Python
        publication path.  Scalars and fences are precomputed by the caller's
        shared derivation (identical on every rank).  The heavy bookkeeping
        (descriptors, Coordinator confirmation tasks, ledger updates) is
        deferred to _ring_publish_cleanup, which rebuilds the per-request
        token data from the ring record itself without a GPU sync.
        """

        from sglang.srt.speculative.dt_direct_ring import V2_MAX_REQ, V2_MAX_TOK

        pool = self._feature_source_pool
        if (
            pool is None
            or self._ledger is None
            or not self.is_persistent
            or self.candidate_transport != "cuda-ipc"
            or not self._direct_ring
            or not getattr(self, "_ring_publish", False)
            or not staged_meta
            or len(staged_meta) > V2_MAX_REQ
        ):
            return None
        active = [index for index, *_ in staged_meta]
        counts = accept_values
        publish_started_ns = time.monotonic_ns()
        _marks = []
        _mark_last = publish_started_ns

        def _mark(name):
            nonlocal _mark_last
            now_ns = time.monotonic_ns()
            _marks.append((name, now_ns - _mark_last))
            _mark_last = now_ns

        packed = None
        candidates_gpu = None
        if deferred:
            candidates_gpu = getattr(batch_result, "dt_dflash_candidate_tokens", None)
            packed = getattr(batch_result, "dt_dflash_packed_hidden", None)
            if (
                candidates_gpu is None
                or not candidates_gpu.is_cuda
                or not candidates_gpu.is_contiguous()
                or packed is None
                or not packed.is_cuda
                or int(packed.shape[0]) != len(batch.reqs) * self.block_size
            ):
                return None
        elif feature_chunks is None:
            return None
        # Eligibility is evaluated at park time: the in-flight/binding state
        # it reads only settles once the previous wave's park has run, so it
        # cannot be precomputed at the deferred stage.  Eligibility and the
        # staging entries are built in ONE batch pass; ineligible waves
        # discard the half-built entries.
        entries = []
        session_id = None
        publisher = None
        chunked_req = getattr(batch, "chunked_req", None)
        candidates_base = (
            int(candidates_gpu.data_ptr()) if deferred else 0
        )
        for index, req, _offset, round_id, attempt, _handle in staged_meta:
            binding = self._draft_binding.get(req.rid)
            if binding is None:
                # Round-0/bootstrap rounds have no owner lease yet; Python path.
                return None
            if session_id is None:
                session_id = binding[1]
                publisher = self._ring_publish_publisher(session_id)
                if publisher is None:
                    return None
            elif binding[1] != session_id:
                # One ring record describes one wave to one owner Draft session.
                return None
            if (
                req is chunked_req
                or getattr(req, "inflight_middle_chunks", 0) > 0
                or self._target_inflight_count(req.rid) != 0
            ):
                # Only proposal-boundary waves are reconstructible Draft-side:
                # the record carries no requires_proposal bit.
                return None
            if not deferred and (
                int(token_chunks[index].numel()) != counts[index]
                or int(feature_chunks[index].shape[0]) != counts[index]
            ):
                raise RuntimeError("DFLASH Target feature/token chunk shape mismatch")
            if deferred:
                token_src = candidates_base + (
                    index * self.block_size * candidates_gpu.element_size()
                )
            else:
                token_src = int(token_chunks[index].data_ptr())
            entries.append(
                (
                    publisher.rid_hash_for(req.rid),
                    round_id,
                    attempt,
                    binding[2],
                    token_src,
                )
            )

        row_count = int(counts[active].sum())
        token_total = row_count
        if row_count > V2_MAX_TOK or publisher.free_entries() <= 0:
            return None
        if (
            deferred
            and packed.data_ptr() in self._inplace_hidden
            and int(packed.shape[0]) * self.pivot_width * 2
            > publisher.writer.max_entry_bytes
        ):
            # The whole packed wave exceeds the Draft landing buffer.
            return None
        # Resolve the slot layout and make the wave's rows visible: either an
        # in-place reservation from the forward (rows already in the slot,
        # block-strided feature offsets) or a fresh reserve plus one pack
        # copy (compacted layout).
        _mark("eligibility")
        offset_stride = 0
        ready_event = None
        if deferred:
            reservation = self._inplace_hidden.pop(packed.data_ptr(), None)
        else:
            reservation = None
        if reservation is not None:
            slot_id, generation, ready_event = reservation
            offset_stride = self.block_size
            row_count = int(packed.shape[0])
        else:
            fresh = pool.reserve(row_count)
            if fresh is None:
                return None
            slot_id, generation, rows = fresh
            try:
                if deferred:
                    block = self.block_size
                    torch.cat(
                        [
                            packed[index * block : index * block + counts[index]]
                            for index in active
                        ],
                        dim=0,
                        out=rows,
                    )
                else:
                    torch.cat(
                        [feature_chunks[index] for index in active], dim=0, out=rows
                    )
                events = getattr(self, "_slot_ready_events", None)
                if events is None:
                    events = self._slot_ready_events = {}
                ready_event = events.get(slot_id)
                if ready_event is None:
                    ready_event = events[slot_id] = torch.cuda.Event(interprocess=True)
                ready_event.record()
            except Exception:
                pool.release_reserved(slot_id, generation)
                raise
        _mark("reserve_pack")
        published_wave = False
        try:
            event_handle = getattr(self, "_slot_event_handles", {}).get(slot_id)
            if event_handle is None:
                event_handle = ready_event.ipc_handle()
                self._slot_event_handles[slot_id] = event_handle
            # Same WAR edge as the Python publication: a later graph replay on
            # the forward stream may not overwrite the rows before the pull.
            self.scheduler.forward_stream.wait_event(ready_event)
            _mark("handle_war")
            publisher.stage_requests(entries)
            _mark("entries")
            source_byte_offset = (
                pool._buffer.allocation_offset + slot_id * pool._slot_nbytes
            )
            # The kernel reads the per-slot accept counts straight from the
            # verify's GPU tensor (same int32 layout as the staged copy); the
            # CPU-staged counts buffer is only a fallback for CPU-side
            # accept_lens, and the kernel_done edge below covers the read.
            accept_lens_gpu = getattr(batch_result, "accept_lens", None)
            if accept_lens_gpu is not None and accept_lens_gpu.is_cuda:
                accept_dev = int(accept_lens_gpu.data_ptr())
            else:
                publisher.stage_counts(counts)
                accept_dev = publisher.counts_dev
            stream_ptr = getattr(self, "_ring_publish_stream_ptr", None)
            if stream_ptr is None:
                stream_ptr = self._ring_publish_stream_ptr = (
                    torch.cuda.current_stream().cuda_stream
                )
            wave_seq = publisher.publish(
                stream=stream_ptr,
                active_slots=[index for index, *_ in staged_meta],
                slot_id=slot_id,
                generation=generation,
                source_byte_offset=source_byte_offset,
                nbytes=row_count * self.pivot_width * 2,
                row_count=row_count,
                event_handle=event_handle,
                accept_dev=accept_dev,
                bonus_dev=int(draft_input.bonus_tokens.data_ptr()),
                seq_lens_dev=int(draft_input.new_seq_lens.data_ptr()),
                offset_stride=offset_stride,
                token_total=token_total,
            )
            published_wave = True
            # The kernel also reads the verify's GPU scalars and candidate
            # tokens; hold the next graph replay off those buffers until it
            # has run (the ready_event edge above only covers the pack).  One
            # event per controller: each park's record tightens the edge the
            # next replay waits on, so reuse cannot weaken the ordering.
            kernel_done = getattr(self, "_ring_publish_done_event", None)
            if kernel_done is None:
                kernel_done = self._ring_publish_done_event = torch.cuda.Event()
            kernel_done.record()
            self.scheduler.forward_stream.wait_event(kernel_done)
            _mark("stage_launch")
        except Exception as exc:
            if published_wave:
                # The record may already be on the ring: the slot must never
                # recycle, the producer event must stay alive for the Draft's
                # wait, and the wave must fail closed rather than re-publish.
                pool.quarantine_reserved(slot_id, generation)
                orphaned = getattr(self, "_ring_publish_orphaned_events", None)
                if orphaned is None:
                    orphaned = self._ring_publish_orphaned_events = deque(maxlen=1024)
                orphaned.append(ready_event)
                raise RingPublishCommittedError(str(exc)) from exc
            pool.release_reserved(slot_id, generation)
            raise

        specs = tuple(commit_specs)
        wave = RingPublishWave(
            session_id=session_id,
            wave_seq=wave_seq,
            slot_id=slot_id,
            generation=generation,
            row_count=row_count,
            ready_event=ready_event,
            specs=specs,
            reqs={req.rid: req for req in batch.reqs},
            active_slots=tuple(index for index, *_ in staged_meta),
            proposals={},
        )
        self._ring_publish_pending.append(wave)
        _mark("wave_build")
        batch_result.dt_ring_publish_wave = wave
        batch_result.dt_ring_publish_breakdown = tuple(_marks)
        batch_result.dt_dflash_feature_publish_ns = (
            time.monotonic_ns() - publish_started_ns
        )
        batch_result.dt_dflash_feature_publish_strategy = "ring-publish"
        batch_result.dt_dflash_feature_publish_chunks = 0
        batch_result.dt_dflash_published_descriptor_count = len(specs)
        batch_result.dt_dflash_published_feature_rows = row_count
        batch_result.dt_dflash_published_feature_bytes = (
            row_count * self.pivot_width * 2
        )
        return True

    def _ring_publish_cleanup(self) -> None:
        """Build Coordinator confirmation tasks for kernel-published waves.

        Runs from _poll_export_rank, off the park critical path.  A wave is
        processed once its record's seq word is visible in the Target's own
        ring mapping (the kernel has landed), so no GPU sync is needed; the
        per-request token ids are read back from the record itself.
        """

        import struct as _struct

        from msgspec.structs import replace

        from sglang.srt.speculative.dt_direct_ring import (
            ENTRY_BYTES,
            V2_HEADER_BYTES,
            V2_MAX_TOK,
            V2_REQ_ENTRY_BYTES,
            V2_TOKEN_OFFSET,
        )
        from sglang.srt.speculative.dt_feature_p2p import FeatureTransferSlice

        pool = self._feature_source_pool
        retained = deque()
        waves = list(self._ring_publish_pending)
        # Waves of one session must confirm in wave order, so once a session's
        # wave is held on slot exhaustion every later wave of that session is
        # held too; waves of other sessions are independent and still fit.
        blocked_sessions = set()
        for wave in waves:
            if wave.session_id in blocked_sessions:
                retained.append(wave)
                continue
            writer = self._ring_writers.get(wave.session_id)
            if writer is None:
                # The owner lease was revoked; the fence machinery retires the
                # wave's rounds and the slot via the quarantine path.
                self._feature_source_pool.quarantine_reserved(
                    wave.slot_id, wave.generation
                )
                continue
            base = (
                writer._ring.entries_offset
                + (wave.wave_seq % writer.capacity) * ENTRY_BYTES
            )
            raw = writer._ring.raw
            if _struct.unpack_from("<Q", raw, base)[0] != wave.wave_seq:
                retained.append(wave)  # kernel has not landed yet; next poll
                continue
            row_count = _struct.unpack_from("<I", raw, base + 48)[0]
            request_count = _struct.unpack_from("<I", raw, base + 56)[0]
            record_reqs = [
                _struct.unpack_from(
                    "<Q6I", raw, base + V2_HEADER_BYTES + i * V2_REQ_ENTRY_BYTES
                )
                for i in range(request_count)
            ]
            record_counts = [rec[4] for rec in record_reqs]
            token_total = sum(record_counts)
            # Feature row offsets come from the record (block-strided for
            # in-place waves); token ids sit at compacted prefix positions.
            row_offsets = [rec[3] for rec in record_reqs]
            tokens = _struct.unpack_from(
                f"<{token_total}I", raw, base + V2_TOKEN_OFFSET
            )
            hidden_produced_ns = _struct.unpack_from("<Q", raw, base + 64)[0]
            specs = wave.specs
            if (
                request_count != len(specs)
                or row_count != wave.row_count
                or token_total > V2_MAX_TOK
            ):
                raise RuntimeError("ring publish record geometry diverged")
            # Reserve each live request's arena destination now and publish
            # it in the ring: the owning Draft's candidate push reads it from
            # the destination region instead of waiting for this cleanup's
            # Coordinator copy to arrive.
            live_positions = [
                position
                for position, spec in enumerate(specs)
                if spec[1] not in self._cancelled_transfers
            ]
            live_set = set(live_positions)
            destinations = {}
            if self._candidate_arena is not None and live_positions:
                if self._candidate_arena.free_slot_count < len(live_positions):
                    # Slot exhaustion must backpressure, never crash: hold the
                    # wave unconfirmed and retry on a later poll.  The Draft
                    # already holds its push for unconfirmed ring tasks, and a
                    # slot is freed only when a verify consumes it, so pending
                    # waves cannot starve the completions that release slots.
                    # Only this session's later waves are held, keeping
                    # confirmations in wave order per session.
                    blocked_sessions.add(wave.session_id)
                    now_ns = time.monotonic_ns()
                    if (
                        now_ns - getattr(self, "_ring_publish_hold_log_ns", 0)
                        > 5_000_000_000
                    ):
                        self._ring_publish_hold_log_ns = now_ns
                        logger.info(
                            "candidate arena exhausted; holding %d "
                            "ring-published wave(s) for session %s until "
                            "verify completions free slots",
                            sum(
                                1
                                for pending in waves
                                if pending.session_id == wave.session_id
                            ),
                            wave.session_id,
                        )
                    retained.append(wave)
                    continue
                try:
                    for position in live_positions:
                        transfer_id = specs[position][1]
                        destinations[transfer_id] = self._candidate_arena.reserve(
                            transfer_id
                        )
                except Exception:
                    for transfer_id in destinations:
                        self._candidate_arena.cancel(
                            transfer_id, remote_quiescent=True
                        )
                    raise
                writer.write_destinations(
                    wave.wave_seq,
                    [
                        (
                            (
                                destinations[spec[1]].slot_id,
                                destinations[spec[1]].generation,
                                destinations[spec[1]].authorization_nonce,
                            )
                            if position in live_set
                            else (0, 0, 0)
                        )
                        for position, spec in enumerate(specs)
                    ],
                    data_buffer_offset=(
                        self._candidate_arena._data_buffer.allocation_offset
                    ),
                    metadata_buffer_offset=(
                        self._candidate_arena._metadata_buffer.allocation_offset
                    ),
                )
            published = pool.publish_reserved(
                slot_id=wave.slot_id,
                generation=wave.generation,
                token_count=wave.row_count,
                feature_schema_id=self.feature_schema_id,
                transfers=tuple(
                    FeatureTransferSlice(
                        transfer_id=spec[1],
                        row_offset=row_offset,
                        row_count=spec[6],
                    )
                    for spec, row_offset in zip(specs, row_offsets, strict=True)
                ),
                producer_event=wave.ready_event,
                hidden_produced_ns=hidden_produced_ns,
            )
            descriptors_by_transfer = {
                descriptor.transfer_id: descriptor
                for descriptor in published.descriptors
            }
            for spec in specs:
                self._source_slot_by_transfer[spec[1]] = (
                    wave.slot_id,
                    wave.generation,
                )
            tasks = []
            token_cursor = 0
            for index, spec in enumerate(specs):
                transfer_id = spec[1]
                count = spec[6]
                token_ids = tuple(tokens[token_cursor : token_cursor + count])
                token_cursor += count
                if int(record_counts[index]) != count:
                    raise RuntimeError("ring publish record token counts diverged")
                if transfer_id in self._cancelled_transfers:
                    pool.cancel(
                        transfer_id=transfer_id,
                        source_slot_id=wave.slot_id,
                        slot_generation=wave.generation,
                        source_session_id=pool.source_session_id,
                        pool_generation=pool.pool_generation,
                    )
                    self._source_slot_by_transfer.pop(transfer_id, None)
                    destination = destinations.get(transfer_id)
                    if destination is not None:
                        self._candidate_arena.cancel(
                            transfer_id, remote_quiescent=True
                        )
                    continue
                requires_proposal, previous_transfer_id = wave.proposals.get(
                    transfer_id, (True, spec[9])
                )
                task = self._make_task(
                    transfer_id=transfer_id,
                    req=wave.reqs[spec[0]],
                    round_id=spec[2],
                    attempt=spec[3],
                    bonus_token=spec[10],
                    previous_transfer_id=previous_transfer_id,
                    feature_descriptor=descriptors_by_transfer[transfer_id],
                    feature_token_ids=token_ids,
                    committed_token_ids=token_ids,
                    feature_token_offset=spec[5],
                    requires_proposal=requires_proposal,
                )
                # Pre-stamped: the record is already on the ring, so the
                # planner only authorizes the owner lease (no second write).
                tasks.append(
                    replace(
                        task,
                        direct_ring_seq=wave.wave_seq,
                        candidate_destination=destinations.get(transfer_id),
                    )
                )
                self._ring_publish_transfers.add(transfer_id)
                self._ring_unproven[transfer_id] = wave.session_id
                binding = self._draft_binding.get(spec[0])
                if binding is not None:
                    self._lease_marks.append((transfer_id, binding[0]))
                self._direct_sent.add(transfer_id)
                self._ring_outstanding[wave.session_id].append(
                    (wave.wave_seq, transfer_id, (wave.slot_id, wave.generation))
                )
            if tasks:
                self._queue_task_batch(tuple(tasks))
                self._flush_pending_send()
            if self._profile_active:
                self._profile_emit(
                    "target_ring_publish",
                    ring_seq=wave.wave_seq,
                    draft_session_id=wave.session_id,
                    publication_strategy="ring-publish",
                    publication_rows=wave.row_count,
                    publication_requests=len(specs),
                    publication_bytes=wave.row_count * self.pivot_width * 2,
                )
        self._ring_publish_pending = retained

    def _direct_ring_writer(self, draft_session_id: str):
        """Return the serving ring to one owner Draft session, or None."""

        from sglang.srt.speculative.dt_direct_ring import (
            STATE_CREATED,
            STATE_SERVING,
            TargetRingWriter,
            ring_path,
        )

        writer = self._ring_writers.get(draft_session_id)
        if writer is None:
            if draft_session_id in self._ring_revoked_sessions:
                return None
            now_ns = time.monotonic_ns()
            if self._ring_missing_until_ns.get(draft_session_id, 0) > now_ns:
                return None
            path = ring_path(draft_session_id, self._client.instance_session_id)
            try:
                writer = TargetRingWriter(path)
            except FileNotFoundError:
                # Not on this host or not created yet: Coordinator path only.
                self._ring_missing_until_ns[draft_session_id] = now_ns + 1_000_000_000
                return None
            self._ring_writers[draft_session_id] = writer
            self._ring_outstanding[draft_session_id] = deque()
        if writer.state == STATE_CREATED:
            pool = self._feature_source_pool
            if pool is not None:
                allocation = pool._allocations[pool._buffer.allocation_index]
                writer.register_source(
                    ipc_handle=allocation.ipc_handle,
                    allocation_length=allocation.length,
                    pool_generation=pool.pool_generation,
                    source_session_id=pool.source_session_id,
                    slot_nbytes=pool._slot_nbytes,
                    slot_count=pool.slot_count,
                    allocation_offset=pool._buffer.allocation_offset,
                )
                writer.source_ipc_handle = allocation.ipc_handle
            else:
                writer.register_source(
                    ipc_handle=bytes(64),
                    allocation_length=0,
                    pool_generation="",
                    source_session_id="",
                )
                writer.source_ipc_handle = b""
        return writer if writer.state == STATE_SERVING else None

    def _plan_direct_ring(self, tasks):
        """Choose owner-known rounds for the ring and stamp their ring seq."""

        from dt_protocol import DraftTaskKind
        from msgspec.structs import replace

        from dt_protocol import DraftAssignment

        from sglang.srt.speculative.dt_candidate_p2p import CandidateTransportError
        from sglang.srt.speculative.dt_direct_ring import MAX_PAYLOAD_BYTES, next_seq

        if not self.is_persistent or self.candidate_transport != "cuda-ipc":
            return tasks, ()
        if self._ring_encoder is None:
            import msgspec

            self._ring_encoder = msgspec.msgpack.Encoder()
        planned_next: dict[str, int] = {}
        planned_count: dict[str, int] = {}
        plan = []
        stamped = []
        for task in tasks:
            if task.direct_ring_seq and task.transfer_id in getattr(
                self, "_ring_publish_transfers", ()
            ):
                # Kernel-published wave: the record is already on the ring,
                # so this Coordinator copy is confirmation-only and must not
                # be planned for a second ring write.  The owner lease still
                # needs its push authorization.
                self._ring_publish_transfers.discard(task.transfer_id)
                rp_binding = self._draft_binding.get(task.request_id)
                if rp_binding is not None and task.candidate_destination is not None:
                    try:
                        self._candidate_arena.authorize(
                            DraftAssignment(
                                transfer_id=task.transfer_id,
                                draft_instance_id=rp_binding[0],
                                draft_session_id=rp_binding[1],
                                authorization_nonce=(
                                    task.candidate_destination.authorization_nonce
                                ),
                                owner_epoch=rp_binding[2],
                            )
                        )
                    except CandidateTransportError:
                        logger.warning(
                            "owner lease cannot authorize %s; confirmation only",
                            task.transfer_id,
                        )
                stamped.append(task)
                continue
            binding = self._draft_binding.get(task.request_id)
            writer = None
            if (
                task.kind == DraftTaskKind.RECONCILE
                and task.candidate_destination is not None
                and not task.direct_ring_seq
                and not task.source_read_migration_id
                and binding is not None
            ):
                writer = self._direct_ring_writer(binding[1])
            source = task.source
            if writer is not None and source is not None and (
                source.ipc_handle != writer.source_ipc_handle
                or source.nbytes > writer.max_entry_bytes
                or len(source.producer_event_handle) != 64
            ):
                writer = None
            if writer is not None:
                count = planned_count.get(binding[1], 0)
                if writer.free_entries() - count <= 0:
                    writer = None
            if writer is None:
                stamped.append(task)
                continue
            seq = planned_next.get(binding[1], writer.next_seq)
            ring_task = replace(task, direct_ring_seq=seq)
            payload = self._ring_encoder.encode(ring_task)
            if len(payload) > MAX_PAYLOAD_BYTES:
                stamped.append(task)
                continue
            instance_id, session_id, epoch = binding
            try:
                # Lease authorization: READY needs no per-round Coordinator
                # assignment; a later assignment must name this producer.
                self._candidate_arena.authorize(
                    DraftAssignment(
                        transfer_id=task.transfer_id,
                        draft_instance_id=instance_id,
                        draft_session_id=session_id,
                        authorization_nonce=task.candidate_destination.authorization_nonce,
                        owner_epoch=epoch,
                    )
                )
            except CandidateTransportError:
                logger.warning(
                    "owner lease cannot authorize %s; using the Coordinator path",
                    task.transfer_id,
                )
                stamped.append(task)
                continue
            planned_next[binding[1]] = next_seq(seq)
            planned_count[binding[1]] = count + 1
            plan.append((binding[1], writer, ring_task, payload))
            stamped.append(ring_task)
        return stamped, tuple(plan)

    def _write_direct_ring(self, plan) -> None:
        """Publish planned ring entries after the Coordinator submit succeeded."""

        for draft_session_id, writer, task, payload in plan:
            # Stamped before the seq store: the Draft may take it immediately.
            written_ns = time.monotonic_ns()
            source = task.source
            if source is None:
                seq = writer.write(payload=payload, has_pull=False)
                pull = None
            else:
                seq = writer.write(
                    payload=payload,
                    has_pull=True,
                    source_offset=source.allocation_offset + source.byte_offset,
                    nbytes=source.nbytes,
                    source_slot_id=source.source_slot_id,
                    slot_generation=source.slot_generation,
                    event_handle=source.producer_event_handle,
                )
                pull = (source.source_slot_id, source.slot_generation)
            if seq != task.direct_ring_seq:
                raise RuntimeError("direct ring sequence diverged from its plan")
            self._ring_unproven[task.transfer_id] = draft_session_id
            # Applied after ledger.begin() registers the round.
            self._lease_marks.append(
                (task.transfer_id, self._draft_binding[task.request_id][0])
            )
            self._direct_sent.add(task.transfer_id)
            self._ring_outstanding[draft_session_id].append(
                (seq, task.transfer_id, pull)
            )
            if self._profile_active:
                self._profile_emit(
                    "target_ring_write",
                    transfer_id=task.transfer_id,
                    request_id=task.request_id,
                    round_id=task.round_id,
                    ring_seq=seq,
                    written_ns=written_ns,
                )

    def _feature_source_pool_diagnostics(self) -> str:
        """Per-slot lease snapshot appended to park publication failures."""

        pool = self._feature_source_pool
        if pool is None:
            return ""
        try:
            parts = []
            for slot_id, slot in enumerate(pool._slots):
                if str(slot.state) == "FeatureSlotState.FREE":
                    continue
                active = sorted(getattr(slot, "active_transfers", None) or ())
                parts.append(
                    f"slot={slot_id} state={slot.state} gen={slot.generation}"
                    f" active={len(active)} {active[:4]}"
                )
            outstanding = {
                session: [seq for seq, _tid, _pull in deque_][:4]
                for session, deque_ in self._ring_outstanding.items()
                if deque_
            }
            return f" | pool[{'; '.join(parts)}] ring_heads={outstanding}"
        except Exception:
            return " | pool diagnostics unavailable"

    def _poll_direct_ring_reads(self) -> None:
        """Retire source leases on the Draft worker's GPU-written read proof.

        The worker writes an entry's INPUT_READY word on its copy stream after
        the hidden copy, so a matching word proves the source range was read;
        no Coordinator relay is needed.  The later relayed SourceReadDone is
        an idempotent repeat.
        """

        pool = self._feature_source_pool
        blocked_since = getattr(self, "_ring_read_blocked_since", None)
        if blocked_since is None:
            blocked_since = self._ring_read_blocked_since = {}
        now_ns = time.monotonic_ns()
        for draft_session_id, outstanding in self._ring_outstanding.items():
            writer = self._ring_writers[draft_session_id]
            while outstanding:
                seq, transfer_id, pull = outstanding[0]
                if not writer.read_proven(seq):
                    blocked_since.setdefault(draft_session_id, now_ns)
                    if now_ns - blocked_since[draft_session_id] > 1_000_000_000:
                        if now_ns - getattr(
                            self, "_ring_read_stall_emit_ns", 0
                        ) > 5_000_000_000:
                            self._ring_read_stall_emit_ns = now_ns
                            self._profile_emit(
                                "target_ring_read_stall",
                                draft_session_id=draft_session_id,
                                head_seq=seq,
                                writer_next_seq=writer.next_seq,
                                ready_word=int(writer._ring.ready[seq % writer.capacity]),
                                outstanding=len(outstanding),
                                blocked_ms=(now_ns - blocked_since[draft_session_id])
                                // 1_000_000,
                            )
                    break
                blocked_since.pop(draft_session_id, None)
                outstanding.popleft()
                self._ring_unproven.pop(transfer_id, None)
                if pull is None or pool is None:
                    continue
                proven = self._ring_proven
                proven[transfer_id] = None
                if len(proven) > 65536:
                    proven.popitem(last=False)
                if self._source_slot_by_transfer.get(transfer_id) != pull:
                    continue
                pool.read_done(
                    transfer_id=transfer_id,
                    source_slot_id=pull[0],
                    slot_generation=pull[1],
                    source_session_id=pool.source_session_id,
                    pool_generation=pool.pool_generation,
                )
                self._source_slot_by_transfer.pop(transfer_id, None)
                self._source_deadline_by_transfer.pop(transfer_id, None)
                self._staged_flashmtp_source_leases.discard(transfer_id)
            if not outstanding:
                blocked_since.pop(draft_session_id, None)

    def _fence_dead_ring_owners(self) -> None:
        """Revoke a ring owner whose process is gone or whose ring is closed.

        Same-host fence: a Draft that exited cannot read Target memory any
        more, so its unproven ring rounds can retire without the Coordinator,
        which never knew of ring copies it did not accept.
        """

        from sglang.srt.speculative.dt_direct_ring import STATE_CLOSED

        now_ns = time.monotonic_ns()
        if now_ns < getattr(self, "_ring_fence_next_ns", 0):
            return
        self._ring_fence_next_ns = now_ns + 200_000_000
        for draft_session_id, outstanding in tuple(self._ring_outstanding.items()):
            if not outstanding:
                continue
            writer = self._ring_writers[draft_session_id]
            gone = writer.state == STATE_CLOSED
            if not gone and writer.owner_pid:
                try:
                    os.kill(writer.owner_pid, 0)
                except ProcessLookupError:
                    gone = True
                except PermissionError:
                    pass
            if gone:
                self._revoke_ring_owner(draft_session_id, quiescent=True)

    def _revoke_ring_owner(self, draft_session_id: str, *, quiescent: bool) -> None:
        """Revoke one Draft session's ring lease and retire its ring rounds."""

        self._ring_revoked_sessions.add(draft_session_id)
        writer = self._ring_writers.pop(draft_session_id, None)
        outstanding = self._ring_outstanding.pop(draft_session_id, deque())
        for _seq, transfer_id, pull in outstanding:
            self._ring_unproven.pop(transfer_id, None)
            if pull is None:
                continue
            if quiescent:
                self._release_source_slot(transfer_id)
            else:
                self._quarantine_source_slot(transfer_id)
        if self._candidate_arena is not None:
            self._candidate_arena.revoke_producer(draft_session_id, quiescent=quiescent)
        for request_id, binding in tuple(self._draft_binding.items()):
            if binding[1] == draft_session_id:
                del self._draft_binding[request_id]
        publisher = getattr(self, "_ring_publishers", {}).pop(draft_session_id, None)
        if publisher is not None:
            publisher.close()
        if writer is not None:
            writer.close(unlink=quiescent)
        logger.warning(
            "revoked direct ring owner session %s (quiescent=%s, ring rounds=%d)",
            draft_session_id,
            quiescent,
            len(outstanding),
        )

    def _close_direct_rings(self) -> None:
        for publisher in getattr(self, "_ring_publishers", {}).values():
            publisher.close()
        self._ring_publishers = {}
        for writer in self._ring_writers.values():
            writer.close()
        self._ring_writers.clear()
        self._ring_outstanding.clear()

    def _release_candidate_destination(
        self, transfer_id: str, *, remote_quiescent: bool = False
    ) -> None:
        if self._candidate_arena is None:
            return
        self._candidate_arena.cancel(
            transfer_id, remote_quiescent=remote_quiescent
        )

    def _release_source_slot(self, transfer_id: str) -> None:
        staged_leases = getattr(self, "_staged_flashmtp_source_leases", None)
        if staged_leases is not None:
            staged_leases.discard(transfer_id)
        self._source_deadline_by_transfer.pop(transfer_id, None)
        source_ref = self._source_slot_by_transfer.pop(transfer_id, None)
        if source_ref is not None:
            source_slot, _generation = source_ref
            if not self.uses_direct_gpu_source or self._feature_source_pool is None:
                raise RuntimeError("source lease has no direct GPU source pool")
            self._feature_source_pool.cancel(
                transfer_id=transfer_id,
                source_slot_id=source_slot,
                slot_generation=_generation,
                source_session_id=self._feature_source_pool.source_session_id,
                pool_generation=self._feature_source_pool.pool_generation,
            )

    def _quarantine_source_slot(self, transfer_id: str) -> None:
        """Fence an uncertain direct GPU read without reusing its source range."""

        self._source_deadline_by_transfer.pop(transfer_id, None)
        # Keep the transfer-to-slot fence while the reader may still exist.
        # A later ordered read-done, or OS-proven Draft process retirement,
        # must be able to resolve this quarantined slot and return it to the
        # bounded source pool safely.
        source_ref = self._source_slot_by_transfer.get(transfer_id)
        if source_ref is None:
            return
        if not self.uses_direct_gpu_source or self._feature_source_pool is None:
            raise RuntimeError("source quarantine requires a direct GPU source pool")
        source_slot, generation = source_ref
        self._feature_source_pool.quarantine(
            source_slot_id=source_slot,
            slot_generation=generation,
            source_session_id=self._feature_source_pool.source_session_id,
            pool_generation=self._feature_source_pool.pool_generation,
        )
        self._profile_emit(
            "target_feature_quarantine",
            target_instance=self.instance_id,
            transfer_id=transfer_id,
            source_slot_id=source_slot,
            slot_generation=generation,
            quarantined_slots=self._feature_source_pool.quarantined_slot_count,
        )
