from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import torch
from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.model_runner import (
    DT_BENCHMARK_MEASUREMENT_RID_PREFIX,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.utils import broadcast_pyobj

logger = logging.getLogger(__name__)
# Initial control delivery is immediate.  A one-second retry cadence still
# provides multiple attempts inside the round lease while preventing a busy
# scheduler from turning transient ACK latency into a duplicate-message storm.
CONTROL_RETRY_INTERVAL_NS = 1_000_000_000
DT_LANE_MEASUREMENT_MARKER = "DT lane measurement evidence: "


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
    failure: str | None = None
    draft_admitted: bool = False


@dataclass(slots=True)
class StagedPinnedPivotBatch:
    """Pinned staging lease filled on the asynchronous result-copy stream."""

    buffer_slot: int
    rows: int
    request_ids: tuple[str, ...]


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
    previous_transfer_id: str | None
    bonus_token: int
    seq_len: int


@dataclass(slots=True)
class StagedDFlashFeatureBatch:
    tasks: tuple[StagedDFlashTask, ...]
    descriptors_by_transfer: dict[str, object]


class DTTargetController:
    """Owns WAIT_DRAFT requests outside the normal SGLang running batch."""

    def __init__(self, scheduler) -> None:
        try:
            from dt_client import (
                RetryOutbox,
                TargetCoordinatorClient,
                TargetRoundLedger,
            )
            from dt_profile import emit as profile_emit
            from dt_transport import MmapSlotStore, cleanup_stale_slot_files
        except ImportError as exc:
            raise RuntimeError(
                "Draft-Target mode requires dt-runtime on PYTHONPATH"
            ) from exc

        self.scheduler = scheduler
        self._profile_emit = profile_emit
        self.server_args = scheduler.server_args
        self.instance_id = self.server_args.draft_target_instance_id
        self.profile_id = self.server_args.draft_target_profile_id
        self.feature_schema_id = self.server_args.draft_target_feature_schema_id
        self.block_size = int(self.server_args.speculative_num_draft_tokens)
        self.pivot_width = int(scheduler.draft_worker._pivot_width)
        self.is_token_only = scheduler.spec_algorithm.is_standalone_dt()
        self.is_dflash = scheduler.spec_algorithm.is_dflash()
        self.is_persistent = self.is_token_only or self.is_dflash
        if self.is_dflash and not self.server_args.disable_radix_cache:
            raise ValueError(
                "remote DFLASH requires --disable-radix-cache until cached-prefix "
                "hidden states have an authoritative export path"
            )
        self.round_timeout_ns = (
            int(self.server_args.draft_target_round_timeout_ms) * 1_000_000
        )
        self.early_pinned_staging = (
            os.environ.get("SGLANG_DT_EARLY_PINNED_STAGING", "0") == "1"
        )
        self.pipeline_lanes = int(self.server_args.draft_target_pipeline_lanes)
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
        elif not self.server_args.draft_target_full_wave or self.pipeline_lanes != 2:
            raise ValueError(
                "Draft-Target V1 requires full-wave verification with two lanes"
            )
        self.overlap_result_defer_ns = int(overlap_result_defer_ms * 1_000_000)
        self.is_export_rank = scheduler.ps.tp_rank == 0
        self._parked: dict[str, ParkedTargetRound] = {}
        self._feature_control_rounds: dict[str, ParkedTargetRound] = {}
        # Overlap scheduling can have more than one chunked-prefill forward for
        # the same request in ``result_queue``.  Track the number of launched
        # Target forwards whose results have not been consumed yet.  A request
        # with a positive count may continue through the chunked-prefill path,
        # but must not re-enter the normal running batch or start a Draft
        # transfer from an older result.
        self._target_inflight_counts: Counter[str] = Counter()
        self._target_inflight_reqs: dict[str, Req] = {}
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
        self._batch_sequence = 0
        self._pending_send = deque()
        self._last_pending_send_block: tuple[str, str, str] | None = None
        self._delivery_outbox = RetryOutbox[object](
            retry_interval_ns=CONTROL_RETRY_INTERVAL_NS
        )
        self._cancelled_transfers: set[str] = set()
        self._free_source_slots = deque(range(scheduler.max_running_requests))
        self._source_slot_by_transfer: dict[str, tuple[int, int]] = {}
        self._source_deadline_by_transfer: dict[str, int] = {}
        self._source_done_applied_ids: set[str] = set()
        self._result_stores: dict[str, object] = {}
        self._last_poll_ns = 0
        self._lane_by_request: dict[str, int] = {}
        self._lane_target_sizes = [0] * self.pipeline_lanes
        self._lane_assignment_cursor = 0
        self._lane_ready_cursor = 0
        self._next_full_wave_id = 0
        self._last_full_wave_completed_ns = 0
        self._last_lane_full_wave_completed_ns = [0] * self.pipeline_lanes
        self._measurement_lane_counts: dict[tuple[int, str], int] = {}
        self._overlap_launch_poll_ns = 0
        self._overlap_result_ready: bool | None = None
        self._client = None
        self._ledger = None
        self._source_store = None
        self._feature_source_pool = None
        self._feature_source_free_slots = 0
        self._feature_source_stall_started_ns = 0
        self._pivot_host_staging = None
        self._early_pivot_host_buffers: list[torch.Tensor] = []
        self._early_pivot_host_free_slots: deque[int] = deque()
        self._early_pivot_host_in_use: set[int] = set()
        if self.is_export_rank:
            if self.pivot_width and not self.is_dflash:
                slot_bytes = (
                    self.pivot_width
                    * torch.tensor([], dtype=torch.bfloat16).element_size()
                )
                safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", self.instance_id)
                data_dir = Path(self.server_args.draft_target_data_dir)
                cleanup_stale_slot_files(data_dir, instance_id=safe_id, kind="pivots")
                source_path = data_dir / f"dt-{safe_id}-{os.getpid()}-pivots.slots"
                self._source_store = MmapSlotStore.create(
                    source_path,
                    slot_count=scheduler.max_running_requests,
                    slot_bytes=slot_bytes,
                )
            try:
                self._client = TargetCoordinatorClient(
                    endpoint=self.server_args.draft_target_coordinator_endpoint,
                    instance_id=self.instance_id,
                    data_endpoint=(
                        self._source_store.uri if self._source_store is not None else ""
                    ),
                )
                if self.is_dflash:
                    from sglang.srt.speculative.dt_feature_p2p import (
                        TargetFeatureSourcePool,
                    )

                    max_feature_tokens = max(
                        int(self.server_args.max_prefill_buffer_tokens()),
                        int(scheduler.max_running_requests) * self.block_size,
                    )
                    self._feature_source_pool = TargetFeatureSourcePool(
                        target_instance_id=self.instance_id,
                        source_session_id=self._client.instance_session_id,
                        device=scheduler.device,
                        slot_count=max(2, self.pipeline_lanes * 2),
                        max_tokens=max_feature_tokens,
                        feature_width=self.pivot_width,
                    )
                    self._feature_source_free_slots = (
                        self._feature_source_pool.free_slot_count
                    )
                self._client.register_target(
                    timeout_s=self.server_args.draft_target_registration_timeout_s
                )
                self._ledger = TargetRoundLedger(target_instance_id=self.instance_id)
                # The source ring bounds the largest wave. Keep one staging
                # buffer for the serial path and two leases for overlap copies.
                if self.pivot_width and not self.is_dflash:
                    self._pivot_host_staging = torch.empty(
                        (scheduler.max_running_requests, self.pivot_width),
                        dtype=torch.bfloat16,
                        device="cpu",
                        pin_memory=True,
                    )
                if self.early_pinned_staging and self.pivot_width and not self.is_dflash:
                    for buffer_slot in range(2):
                        self._early_pivot_host_buffers.append(
                            torch.empty(
                                (scheduler.max_running_requests, self.pivot_width),
                                dtype=torch.bfloat16,
                                device="cpu",
                                pin_memory=True,
                            )
                        )
                        self._early_pivot_host_free_slots.append(buffer_slot)
            except Exception:
                if self._client is not None:
                    self._client.close()
                    self._client = None
                if self._source_store is not None:
                    self._source_store.close()
                    self._source_store.unlink()
                    self._source_store = None
                if self._feature_source_pool is not None:
                    self._feature_source_pool.close()
                    self._feature_source_pool = None
                raise

    def stage_overlap_pivot_copy(self, batch: ScheduleBatch, batch_result) -> None:
        """Copy pivots to a leased pinned buffer on the result-copy stream."""
        if self.is_dflash:
            self._stage_dflash_feature_copy(batch, batch_result)
            return
        if (
            not self.is_export_rank
            or not self.early_pinned_staging
            or getattr(batch_result, "dt_staged_pinned_pivots", None) is not None
        ):
            return
        draft_input = getattr(batch_result, "next_draft_input", None)
        pivot_hidden = getattr(draft_input, "hidden_states", None)
        if pivot_hidden is None or not torch.is_tensor(pivot_hidden):
            return
        if pivot_hidden.ndim != 2 or pivot_hidden.shape[1] != self.pivot_width:
            return
        if pivot_hidden.dtype != torch.bfloat16:
            raise TypeError(
                f"DT Target expected bfloat16 pivot, got {pivot_hidden.dtype}"
            )
        rows = int(pivot_hidden.shape[0])
        if rows != len(batch.reqs):
            raise RuntimeError("DT pivot rows do not match launched batch requests")
        if rows == 0 or not self._early_pivot_host_free_slots:
            return

        buffer_slot = self._early_pivot_host_free_slots.popleft()
        self._early_pivot_host_in_use.add(buffer_slot)
        try:
            self._early_pivot_host_buffers[buffer_slot][:rows].copy_(
                pivot_hidden.detach(), non_blocking=True
            )
        except Exception:
            self._release_early_pinned_staging(buffer_slot)
            raise
        batch_result.dt_staged_pinned_pivots = StagedPinnedPivotBatch(
            buffer_slot=buffer_slot,
            rows=rows,
            request_ids=tuple(req.rid for req in batch.reqs),
        )

    def _stage_dflash_feature_copy(self, batch: ScheduleBatch, batch_result) -> None:
        if getattr(batch_result, "dt_staged_dflash_features", None) is not None:
            return
        draft_input = getattr(batch_result, "next_draft_input", None)
        if not isinstance(draft_input, DFlashDraftInputV2):
            return
        feature_chunks = draft_input.dt_feature_chunks
        token_chunks = draft_input.dt_feature_token_chunks
        feature_offsets = draft_input.dt_feature_offsets
        if feature_chunks is None and not batch.reqs:
            return
        if (
            feature_chunks is None
            or token_chunks is None
            or feature_offsets is None
            or len(feature_chunks) != len(batch.reqs)
            or len(token_chunks) != len(batch.reqs)
            or len(feature_offsets) != len(batch.reqs)
        ):
            raise RuntimeError("DFLASH Target result is missing feature chunks")

        packet = None
        published = None
        descriptors = {}
        if self.is_export_rank:
            try:
                active = [
                    index
                    for index, req in enumerate(batch.reqs)
                    if not req.finished()
                    and not req.is_retracted
                    and int(feature_chunks[index].shape[0]) > 0
                ]
                bonus_values = draft_input.bonus_tokens.detach().to("cpu").tolist()
                seq_len_values = draft_input.new_seq_lens.detach().to("cpu").tolist()
                specs = []
                transfers = []
                hidden_rows = []
                row_offset = 0
                for index in active:
                    req = batch.reqs[index]
                    attempt = self._ensure_lifecycle_attempt(req.rid)
                    hidden = feature_chunks[index]
                    tokens = token_chunks[index]
                    if (
                        hidden.ndim != 2
                        or int(hidden.shape[1]) != self.pivot_width
                        or hidden.dtype != torch.bfloat16
                        or tokens.ndim != 1
                        or int(tokens.numel()) != int(hidden.shape[0])
                    ):
                        raise RuntimeError(
                            "DFLASH Target feature/token chunk shape mismatch"
                        )
                    offset = int(feature_offsets[index])
                    token_count = int(hidden.shape[0])
                    round_id = self._next_round.get(req.rid, 0)
                    previous_committed = self._draft_committed_len.get(req.rid)
                    if round_id == 0:
                        if offset != 0 or previous_committed is not None:
                            raise RuntimeError(
                                "DFLASH bootstrap must begin at offset zero"
                            )
                        draft_kv_handle = f"{self.instance_id}:{req.rid}:" + str(
                            attempt
                        )
                    else:
                        draft_kv_handle = self._draft_kv_handle_by_request.get(
                            req.rid
                        )
                        if draft_kv_handle is None or previous_committed != offset:
                            raise RuntimeError(
                                "DFLASH feature range has a committed-prefix gap"
                            )
                    transfer_id = (
                        f"{self.instance_id}:{req.rid}:{round_id}:{attempt}"
                    )
                    token_ids = tuple(
                        int(value) for value in tokens.detach().to("cpu").tolist()
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
                            self._verify_transfer_by_request.get(req.rid),
                            int(bonus_values[index]),
                            int(seq_len_values[index]),
                        )
                    )
                    hidden_rows.append(hidden.contiguous())
                    transfers.append((transfer_id, row_offset, token_count))
                    row_offset += token_count

                if specs:
                    assert self._feature_source_pool is not None
                    from sglang.srt.speculative.dt_feature_p2p import (
                        FeatureTransferSlice,
                    )

                    published = self._feature_source_pool.publish(
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
                    descriptors = {
                        descriptor.transfer_id: descriptor
                        for descriptor in published.descriptors
                    }
                    # Establish controller ownership before any profiling or
                    # control-plane work can fail.  A later exception can now
                    # deterministically quarantine every undelivered range.
                    for descriptor in published.descriptors:
                        self._source_slot_by_transfer[descriptor.transfer_id] = (
                            descriptor.source_slot_id,
                            descriptor.slot_generation,
                        )
                    for descriptor in published.descriptors:
                        self._profile_emit(
                            "target_feature_publish",
                            target_instance=self.instance_id,
                            transfer_id=descriptor.transfer_id,
                            source_slot_id=descriptor.source_slot_id,
                            slot_generation=descriptor.slot_generation,
                            feature_bytes=descriptor.nbytes,
                            hidden_produced_ns=published.hidden_produced_ns,
                            descriptor_published_ns=(
                                published.descriptor_published_ns
                            ),
                        )
                packet = (specs, None)
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
                        self._source_slot_by_transfer.pop(
                            descriptor.transfer_id, None
                        )
                    published = None
                packet = (None, f"{type(exc).__name__}: {exc}")
                descriptors = {}
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
        assert packet is not None
        specs, error = packet
        if error is not None:
            raise RuntimeError(f"DFLASH feature publication failed: {error}")

        staged_tasks = []
        for values in specs:
            staged = StagedDFlashTask(*values)
            self._next_round[staged.request_id] = staged.round_id + 1
            self._draft_committed_len[staged.request_id] = (
                staged.feature_token_offset + staged.feature_token_count
            )
            self._draft_kv_handle_by_request[staged.request_id] = (
                staged.draft_kv_handle
            )
            current_attempt = self._lifecycle_attempt_by_request.get(
                staged.request_id
            )
            if current_attempt is None:
                self._lifecycle_attempt_by_request[staged.request_id] = (
                    staged.attempt
                )
                self._next_lifecycle_attempt = max(
                    self._next_lifecycle_attempt, staged.attempt + 1
                )
            elif current_attempt != staged.attempt:
                raise RuntimeError("DFLASH lifecycle attempt changed during staging")
            if self.is_export_rank:
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
        )

    def _release_early_pinned_staging(self, buffer_slot: int) -> None:
        if buffer_slot not in self._early_pivot_host_in_use:
            raise RuntimeError(
                f"DT early pinned staging slot {buffer_slot} is not leased"
            )
        self._early_pivot_host_in_use.remove(buffer_slot)
        self._early_pivot_host_free_slots.append(buffer_slot)

    def park_completed_batch(self, batch: ScheduleBatch, *, batch_result=None) -> None:
        started_ns = time.monotonic_ns()
        batch_size = len(batch.reqs)
        staged_pinned_pivots = getattr(batch_result, "dt_staged_pinned_pivots", None)
        try:
            self._record_measurement_lane_execution(batch, batch_result)
            self._park_completed_batch(batch, batch_result=batch_result)
            self._record_full_wave_completion(batch, batch_size=batch_size)
        finally:
            if staged_pinned_pivots is not None:
                if (
                    getattr(batch_result, "dt_staged_pinned_pivots", None)
                    is staged_pinned_pivots
                ):
                    batch_result.dt_staged_pinned_pivots = None
                self._release_early_pinned_staging(staged_pinned_pivots.buffer_slot)
            if self.is_export_rank and batch_size:
                self._profile_emit(
                    "target_park_batch",
                    batch_size=batch_size,
                    total_ns=time.monotonic_ns() - started_ns,
                    parked=len(self._parked),
                )

    def _record_full_wave_completion(
        self, batch: ScheduleBatch, *, batch_size: int
    ) -> None:
        """Record a READY_VERIFY wave after committed features are published.

        ``build_ready_batch`` stamps only controller-created verification waves.
        Completion is observed after Target result processing and
        ``_park_completed_batch`` have selected and published the authoritative
        committed hidden states.  The global period describes successive waves
        on this Target; the lane period describes successive completions for
        the same frozen lane.
        """

        wave_id = getattr(batch, "dt_full_wave_id", None)
        ready_ns = getattr(batch, "dt_full_wave_ready_ns", None)
        lane_id = getattr(batch, "dt_pipeline_lane_id", None)
        if wave_id is None or ready_ns is None or lane_id is None:
            return
        if lane_id < 0 or lane_id >= self.pipeline_lanes:
            raise RuntimeError("DT full-wave completion has an invalid lane")
        completed_ns = time.monotonic_ns()
        previous_completed_ns = self._last_full_wave_completed_ns
        previous_lane_completed_ns = self._last_lane_full_wave_completed_ns[lane_id]
        self._last_full_wave_completed_ns = completed_ns
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
                    completed_ns - previous_completed_ns
                    if previous_completed_ns
                    else 0
                ),
                lane_period_ns=(
                    completed_ns - previous_lane_completed_ns
                    if previous_lane_completed_ns
                    else 0
                ),
                parked=len(self._parked),
                feature_source_free_slots=self._feature_source_free_slots,
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
        lanes = {self._lane_by_request.get(req.rid) for req in measurement_reqs}
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
        staged_pinned_pivots = (
            getattr(batch_result, "dt_staged_pinned_pivots", None)
            if batch_result is not None
            else None
        )
        if not batch.reqs:
            return
        self._release_target_inflight_batch(batch)
        if batch.spec_info is None:
            return
        if not isinstance(batch.spec_info, DFlashDraftInputV2):
            raise TypeError("DT Target can park only DFlashDraftInputV2 batches")

        if self.is_dflash:
            if getattr(batch_result, "dt_staged_dflash_features", None) is None:
                self._stage_dflash_feature_copy(batch, batch_result)
            self._park_dflash_completed_batch(
                batch, batch.spec_info, batch_result=batch_result
            )
            return

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
        # A long prompt can have multiple prefill chunks.  In non-overlap mode
        # the normal result processor decrements ``inflight_middle_chunks``
        # from one to zero before this hook, but ``batch.chunked_req`` still
        # identifies that result as an intermediate chunk.  In overlap mode
        # the counter additionally fences older queued results.  Exporting
        # either kind here would give one request two concurrent Draft rounds.
        chunked_req = getattr(batch, "chunked_req", None)
        active_indices = [
            index
            for index, req in enumerate(batch.reqs)
            if not req.finished()
            and not req.is_retracted
            and req is not chunked_req
            and getattr(req, "inflight_middle_chunks", 0) <= 0
            and self._target_inflight_count(req.rid) == 0
        ]
        tasks = []
        parked_reqs = []
        lane_ids = [
            self._lane_for_request(batch.reqs[index]) for index in active_indices
        ]
        pivot_payloads = None
        source_slots = None
        source_generations = None
        if self.is_export_rank and active_indices and not self.is_token_only:
            if staged_pinned_pivots is not None:
                if staged_pinned_pivots.rows != len(staged_pinned_pivots.request_ids):
                    raise RuntimeError("DT staged pivot row and request counts differ")
                staged_index_by_rid = {
                    rid: index
                    for index, rid in enumerate(staged_pinned_pivots.request_ids)
                }
                if len(staged_index_by_rid) != staged_pinned_pivots.rows:
                    raise RuntimeError(
                        "DT staged pivot request order contains duplicates"
                    )
                try:
                    staged_active_indices = tuple(
                        staged_index_by_rid[batch.reqs[index].rid]
                        for index in active_indices
                    )
                except KeyError as exc:
                    raise RuntimeError(
                        "DT completed request is missing from staged pivot batch"
                    ) from exc
                buffer_slot = staged_pinned_pivots.buffer_slot
                if buffer_slot not in self._early_pivot_host_in_use:
                    raise RuntimeError(
                        f"DT early pinned staging slot {buffer_slot} is not leased"
                    )
                pivot_cpu = self._early_pivot_host_buffers[buffer_slot][
                    : staged_pinned_pivots.rows
                ]
                pivot_rows = pivot_cpu.view(torch.uint8).numpy()
                pivot_payloads = [
                    memoryview(pivot_rows[index]) for index in staged_active_indices
                ]
            else:
                pivot_hidden = draft_input.hidden_states
                if len(active_indices) != original_batch_size:
                    pivot_hidden = pivot_hidden[active_indices]
                if pivot_hidden.dtype != torch.bfloat16:
                    raise TypeError(
                        f"DT Target expected bfloat16 pivot, got {pivot_hidden.dtype}"
                    )
                pivot_gpu = (
                    pivot_hidden.detach().contiguous().view(len(active_indices), -1)
                )
                pivot_cpu = self._pivot_host_staging[: len(active_indices)]
                pivot_cpu.copy_(pivot_gpu, non_blocking=False)
                pivot_rows = pivot_cpu.view(torch.uint8).numpy()
                pivot_payloads = [memoryview(row) for row in pivot_rows]

            acquired_source_slots = []
            try:
                if len(self._free_source_slots) < len(active_indices):
                    raise RuntimeError("DT Target source slot ring is exhausted")
                for _ in active_indices:
                    acquired_source_slots.append(self._free_source_slots.popleft())
                source_slots = tuple(acquired_source_slots)
                source_generations = self._source_store.write_many(
                    tuple(zip(source_slots, pivot_payloads))
                )
            except Exception:
                self._free_source_slots.extendleft(reversed(acquired_source_slots))
                raise

        if active_indices:
            bonus_values = draft_input.bonus_tokens
            seq_len_values = draft_input.new_seq_lens
            if len(active_indices) != original_batch_size:
                bonus_values = bonus_values[active_indices]
                seq_len_values = seq_len_values[active_indices]
            bonus_tokens_cpu = bonus_values.detach().cpu().tolist()
            seq_lens_cpu = seq_len_values.detach().cpu().tolist()
        else:
            bonus_tokens_cpu = []
            seq_lens_cpu = []
        for active_index, index in enumerate(active_indices):
            req = batch.reqs[index]
            round_id = self._next_round.get(req.rid, 0)
            self._next_round[req.rid] = round_id + 1
            attempt = self._ensure_lifecycle_attempt(req.rid)
            transfer_id = f"{self.instance_id}:{req.rid}:{round_id}:{attempt}"
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
            )
            parked_reqs.append(req)

            if self.is_export_rank:
                if self.is_token_only:
                    source_slot = None
                    source_generation = None
                    pivot_nbytes = 0
                else:
                    source_slot = source_slots[active_index]
                    source_generation = source_generations[active_index]
                    self._source_slot_by_transfer[transfer_id] = (
                        source_slot,
                        source_generation,
                    )
                    pivot_nbytes = len(pivot_payloads[active_index])
                tasks.append(
                    self._make_task(
                        transfer_id=transfer_id,
                        req=req,
                        round_id=round_id,
                        attempt=attempt,
                        bonus_token=bonus_token,
                        source_slot=source_slot,
                        source_generation=source_generation,
                        pivot_nbytes=pivot_nbytes,
                        previous_transfer_id=previous_transfer_id,
                    )
                )

        if tasks:
            self._queue_task_batch(tuple(tasks))
            # Source slots are committed at this point, so publish the control
            # message in the same scheduler turn. Waiting for the next poll
            # adds a fixed idle bubble before Draft can start the round.
            self._flush_pending_send()
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
        req_by_id = {req.rid: req for req in batch.reqs}

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

        chunked_req = getattr(batch, "chunked_req", None)
        tasks = []
        parked_reqs = []
        for spec in staged.tasks:
            req = req_by_id.get(spec.request_id)
            if req is None:
                raise RuntimeError("DFLASH staged request is absent from result batch")
            if req.finished() or req.is_retracted:
                if self.is_export_rank:
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
                raise RuntimeError("DFLASH previous-round identity changed before routing")
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
                )
            if self.is_export_rank:
                descriptor = staged.descriptors_by_transfer[spec.transfer_id]
                tasks.append(
                    self._make_task(
                        transfer_id=spec.transfer_id,
                        req=req,
                        round_id=spec.round_id,
                        attempt=spec.attempt,
                        bonus_token=spec.bonus_token,
                        source_slot=None,
                        source_generation=None,
                        pivot_nbytes=descriptor.nbytes,
                        previous_transfer_id=previous_transfer_id,
                        feature_descriptor=descriptor,
                        feature_token_ids=spec.feature_token_ids,
                        feature_token_offset=spec.feature_token_offset,
                        requires_proposal=requires_proposal,
                    )
                )
                if previous_transfer_id is not None:
                    self._ledger.finish(previous_transfer_id)

        if tasks:
            self._queue_task_batch(tuple(tasks))
            self._flush_pending_send()
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

    def poll(self, *, overlap_result=None) -> None:
        poll_entered_ns = time.monotonic_ns()
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
                    self._feature_source_pool.free_slot_count
                    if self._feature_source_pool is not None
                    else 0
                ),
            )
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
        poll_now_ns, events, result_ready, feature_source_free_slots = packet
        self._last_poll_ns = poll_now_ns
        self._overlap_result_ready = result_ready
        self._feature_source_free_slots = feature_source_free_slots
        if self.is_export_rank and events:
            self._profile_emit(
                "target_result_poll",
                poll_gap_ns=(
                    poll_entered_ns - previous_poll_ns if previous_poll_ns else 0
                ),
                receive_ns=receive_ns,
                broadcast_ns=broadcast_ns,
                event_count=len(events),
                ready_count=sum(kind == "ready" for kind, _transfer_id, _ in events),
                accepted_count=sum(
                    kind == "accepted" for kind, _transfer_id, _ in events
                ),
                failed_count=sum(kind == "failed" for kind, _transfer_id, _ in events),
                parked=len(self._parked),
            )
        for kind, transfer_id, payload in events:
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
            if kind == "ready":
                candidate_bytes, bonus_token, seq_len = payload
                if state.bonus_token != bonus_token or state.seq_len != seq_len:
                    raise RuntimeError(
                        "DT Target TP metadata diverged before Draft result: "
                        f"transfer_id={transfer_id}, "
                        f"local_bonus={state.bonus_token}, "
                        f"export_bonus={bonus_token}, "
                        f"local_seq_len={state.seq_len}, "
                        f"export_seq_len={seq_len}"
                    )
                state.candidate_bytes = candidate_bytes
                state.draft_admitted = True
            elif kind == "accepted":
                state.draft_admitted = True
            else:
                state.failure = payload

    def has_feature_source_capacity(self) -> bool:
        has_capacity = not self.is_dflash or self._feature_source_free_slots > 0
        if self.is_export_rank and self.is_dflash:
            now_ns = time.monotonic_ns()
            if not has_capacity and not self._feature_source_stall_started_ns:
                self._feature_source_stall_started_ns = now_ns
                self._profile_emit(
                    "target_feature_source_stall",
                    phase="start",
                    started_ns=now_ns,
                    free_slots=self._feature_source_free_slots,
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
                    pending_task_batches=len(self._pending_send),
                    parked=len(self._parked),
                )
        return has_capacity

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
            return
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
        for req in launched_batch.reqs:
            self._target_inflight_counts[req.rid] += 1
            self._target_inflight_reqs[req.rid] = req

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
        self._overlap_launch_poll_ns = self._last_poll_ns

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
        if not self._target_inflight_counts:
            return
        for req in batch.reqs:
            count = self._target_inflight_counts.get(req.rid, 0)
            if count <= 0:
                continue
            if count == 1:
                del self._target_inflight_counts[req.rid]
                self._target_inflight_reqs.pop(req.rid, None)
            else:
                self._target_inflight_counts[req.rid] = count - 1

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
            state
            for state in self._parked.values()
            if state.req.rid in affected_rids
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

    def should_defer_overlap_result(
        self, *, next_batch: ScheduleBatch | None, last_batch: ScheduleBatch | None
    ) -> bool:
        """Keep the previous result pending while another lane approaches READY.

        SGLang's generic overlap loop resolves a result immediately when the
        next scheduler turn has no batch. DT can have a second full-wave lane
        already executing remotely, so bounded polling can turn that otherwise
        blocking result synchronization into useful producer-consumer overlap.
        ``_last_poll_ns`` and ``_overlap_launch_poll_ns`` come from the TP0 poll
        packet, keeping the defer decision identical on every TP rank.
        """
        elapsed_ns = max(0, self._last_poll_ns - self._overlap_launch_poll_ns)
        should_defer = bool(
            self.overlap_result_defer_ns > 0
            and next_batch is None
            and last_batch is not None
            and self._parked
            and not self._overlap_result_ready
            and elapsed_ns < self.overlap_result_defer_ns
        )
        return should_defer

    def build_ready_batch(self) -> ScheduleBatch | None:
        ready = [
            state
            for state in self._parked.values()
            if state.candidate_bytes is not None
        ]
        if ready:
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
        lane_id = ready[0].lane_id
        if any(state.lane_id != lane_id for state in ready):
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
        batch.req_pool_indices_cpu = torch.tensor(
            [req.req_pool_idx for req in reqs], dtype=torch.int64
        )
        batch.req_pool_indices = batch.req_pool_indices_cpu.to(device=device)
        batch.seq_lens_cpu = torch.tensor(
            [state.seq_len for state in ready], dtype=torch.int64
        )
        batch.seq_lens = batch.seq_lens_cpu.to(device)
        batch.orig_seq_lens = batch.seq_lens.to(dtype=torch.int32)
        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
        bonus_tokens = torch.tensor(
            [state.bonus_token for state in ready],
            dtype=torch.int64,
            device=device,
        )

        candidate_blob = b"".join(state.candidate_bytes for state in ready)
        expected_bytes = len(ready) * self.block_size * 4
        if len(candidate_blob) != expected_bytes:
            raise ValueError("DT candidate payload width does not match block size")
        draft_tokens = torch.frombuffer(
            bytearray(candidate_blob), dtype=torch.int32
        ).reshape(len(ready), self.block_size)
        draft_tokens = draft_tokens.to(device=device, dtype=torch.int64)
        batch.spec_info = DFlashDraftInputV2(
            topk_p=torch.empty((len(ready), 0), dtype=torch.float32, device=device),
            topk_index=torch.empty((len(ready), 0), dtype=torch.int64, device=device),
            bonus_tokens=bonus_tokens,
            new_seq_lens=batch.seq_lens,
            hidden_states=torch.empty(
                (len(ready), self.pivot_width),
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
                hidden_states=batch.spec_info.hidden_states,
            ),
        )
        relay_ready_ns = time.monotonic_ns()
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch, self.scheduler.model_config.vocab_size
        )
        finished_ns = time.monotonic_ns()
        wave_id = self._next_full_wave_id
        self._next_full_wave_id += 1
        batch.dt_full_wave_id = wave_id
        batch.dt_full_wave_ready_ns = finished_ns
        if self.is_export_rank:
            self._profile_emit(
                "target_ready_batch",
                wave_id=wave_id,
                lane_id=lane_id,
                batch_size=len(ready),
                initialize_ns=initialized_ns - started_ns,
                tensors_ns=tensors_ready_ns - initialized_ns,
                relay_ns=relay_ready_ns - tensors_ready_ns,
                sampling_ns=finished_ns - relay_ready_ns,
                total_ns=finished_ns - started_ns,
            )
        return batch

    def should_defer_prefill(self, *, ready_batch_size: int) -> bool:
        # Once a frozen wave is ready, run it before admitting replacement
        # prefills.  Otherwise get_new_batch_prefill wins scheduler priority,
        # the replacement becomes the next frozen wave by itself, and the
        # already-built large verify batch remains stranded in running_batch.
        return ready_batch_size > 0

    def pop_failures(self) -> list[ParkedTargetRound]:
        failed = [state for state in self._parked.values() if state.failure]
        for state in failed:
            self._parked.pop(state.transfer_id)
            self._forget_request_lane(state.req)
            self._next_round.pop(state.req.rid, None)
            self._finish_verified_request(state.req.rid)
            self._release_draft_request(state.req.rid)
            self._lifecycle_attempt_by_request.pop(state.req.rid, None)
        if failed and self.is_export_rank:
            self._flush_reliable_outbox()
        return failed

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
            # Before target sizes are learned, balance persistent lanes.
            lane_id = min(
                range(self.pipeline_lanes),
                key=lambda candidate: (
                    live_counts[candidate],
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
        try:
            if self.is_export_rank and self.is_persistent:
                self._drain_token_only_shutdown()
        finally:
            for store in self._result_stores.values():
                store.close()
            self._result_stores.clear()
            if self._client is not None:
                self._client.close()
            if self._feature_source_pool is not None:
                self._feature_source_pool.close()
                self._feature_source_pool = None
            if self._source_store is not None:
                self._source_store.close()
                self._source_store.unlink()

    def _drain_token_only_shutdown(self) -> None:
        from dt_protocol import (
            CANCEL_TRANSFERS,
            PROTOCOL_VERSION,
            CancelTransfers,
            cancel_transfers_id,
        )

        unsent_transfer_ids = {
            task.transfer_id
            for batch in self._pending_send
            for task in batch.entries
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
            self.is_dflash and self._source_slot_by_transfer
        ):
            self._flush_reliable_outbox()
            self._consume_shutdown_acks()
            if not len(self._delivery_outbox) and not (
                self.is_dflash and self._source_slot_by_transfer
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
                for entry in message.entries:
                    self._consume_source_read_done(entry)
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
        source_slot: int | None,
        source_generation: int | None,
        pivot_nbytes: int,
        previous_transfer_id: str | None,
        feature_descriptor=None,
        feature_token_ids: tuple[int, ...] = (),
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

        if self.is_dflash:
            if feature_descriptor is None:
                raise ValueError("DFLASH task requires a CUDA IPC feature descriptor")
            if not feature_token_ids:
                raise ValueError("DFLASH task requires feature-row token metadata")
            committed_kv_len = feature_token_offset + len(feature_token_ids)
            staged_boundary = self._draft_committed_len.get(req.rid)
            dispatched_boundary = self._draft_dispatched_len.get(req.rid, 0)
            if (
                staged_boundary is None
                or staged_boundary < committed_kv_len
                or dispatched_boundary != feature_token_offset
            ):
                raise RuntimeError(
                    "DFLASH task boundary differs from Target committed feature state"
                )
            draft_kv_handle = self._draft_kv_handle_by_request.get(req.rid)
            if draft_kv_handle is None:
                raise RuntimeError("DFLASH task is missing its persistent KV handle")
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
                committed_token_ids=feature_token_ids,
                committed_kv_len=committed_kv_len,
                feature_token_offset=feature_token_offset,
                feature_token_count=len(feature_token_ids),
                requires_proposal=requires_proposal,
            )
            self._draft_dispatched_len[req.rid] = committed_kv_len
            return task

        expected_bytes = self.pivot_width * 2
        if pivot_nbytes != expected_bytes:
            raise ValueError(
                f"DT Target expected {expected_bytes} pivot bytes, got {pivot_nbytes}"
            )
        assert source_slot is not None and source_generation is not None
        assert self._source_store is not None
        return DraftTask(
            transfer_id=transfer_id,
            request_id=req.rid,
            round_id=round_id,
            attempt=attempt,
            target_instance_id=self.instance_id,
            draft_profile_id=self.profile_id,
            k=self.block_size - 1,
            next_token_id=bonus_token,
            source=SourceRef(
                target_instance_id=self.instance_id,
                source_device_id=self.scheduler.ps.gpu_id,
                source_slot_id=source_slot,
                slot_generation=source_generation,
                transfer_id=transfer_id,
                feature_schema_id=self.feature_schema_id,
                nbytes=pivot_nbytes,
                transport="mmap",
                uri=self._source_store.uri,
            ),
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
        if self.is_dflash:
            for task in tasks:
                if task.source is not None:
                    self._source_deadline_by_transfer[task.transfer_id] = (
                        task.deadline_ns
                    )
        self._pending_send.append(batch)

    def _flush_pending_send(self) -> None:
        self._flush_reliable_outbox()
        while self._pending_send:
            batch = self._pending_send[0]
            blocked = next(
                (
                    (task, active)
                    for task in batch.entries
                    if (
                        active := self._ledger.active_transfer_id(task.request_id)
                    )
                    is not None
                    and active != task.transfer_id
                ),
                None,
            )
            if blocked is not None:
                # Chunked DFlash prefill produces the next authoritative
                # feature range before the remote Draft necessarily finishes
                # materializing the preceding one.  Preserve FIFO here so the
                # same request never has two live descriptors and the Draft
                # sees a contiguous committed prefix.  The fixed source ring
                # provides bounded backpressure while this batch waits.
                task, active_transfer_id = blocked
                block_key = (
                    batch.batch_id,
                    task.transfer_id,
                    active_transfer_id,
                )
                if block_key != self._last_pending_send_block:
                    active_state = self._ledger.get(active_transfer_id)
                    self._profile_emit(
                        "target_task_fifo_block",
                        batch_id=batch.batch_id,
                        entries=len(batch.entries),
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
                break
            if not self._client.submit(batch):
                break
            submitted_ns = time.monotonic_ns()
            self._ledger.begin(batch)
            outbox = self._delivery_outbox
            outbox.add("task_batch", batch.batch_id, batch)
            outbox.mark_sent("task_batch", batch.batch_id, now_ns=submitted_ns)
            self._pending_send.popleft()
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

        self._flush_pending_send()

        events: list[tuple[str, str, object]] = []
        for message in self._client.recv_nowait():
            if isinstance(message, TaskBatchAck):
                if not self._ledger.has_submitted_batch(message.batch_id):
                    continue
                self._delivery_outbox.ack("task_batch", message.batch_id)
                retry_tasks = self._ledger.apply_ack(message)
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
                        self._release_source_slot(task.transfer_id)
                        self._cancelled_transfers.discard(task.transfer_id)
                    else:
                        active_retries.append(task)
                if active_retries:
                    self._queue_task_batch(tuple(active_retries))
            elif isinstance(message, TargetResultBatch):
                events.extend(self._consume_result_batch(message))
            elif isinstance(message, SourceReadDone):
                self._consume_source_read_done(message)
            elif isinstance(message, SourceReadDoneBatch):
                for entry in message.entries:
                    self._consume_source_read_done(entry)
            elif isinstance(message, ControlAck):
                self._delivery_outbox.ack(message.operation, message.message_id)
                if message.operation == "source_read_done":
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
        events.extend(self._expire_local_rounds())
        self._expire_source_leases(now_ns=time.monotonic_ns())
        return events

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
        if not self.is_persistent:
            return
        draft_kv_handle = self._draft_kv_handle_by_request.pop(request_id, None)
        self._draft_committed_len.pop(request_id, None)
        self._draft_dispatched_len.pop(request_id, None)
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
        self, request_id: str, *, reason: str
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

        self._cancelled_transfers.update(transfer_ids)
        self._drop_cancelled_unsent()
        for transfer_id in transfer_ids:
            if (
                transfer_id not in self._source_slot_by_transfer
                and self._ledger.get(transfer_id) is not None
            ):
                self._ledger.finish(transfer_id)
                self._cancelled_transfers.discard(transfer_id)
        cancellation = CancelTransfers(
            protocol_version=PROTOCOL_VERSION,
            target_instance_id=self.instance_id,
            transfer_ids=transfer_ids,
            reason=reason,
        )
        self._delivery_outbox.add(
            CANCEL_TRANSFERS,
            cancel_transfers_id(cancellation),
            cancellation,
        )

    def _expire_local_rounds(self) -> list[tuple[str, str, object]]:
        from dt_protocol import (
            CANCEL_TRANSFERS,
            PROTOCOL_VERSION,
            CancelTransfers,
            cancel_transfers_id,
        )

        expired = self._ledger.expired(now_ns=time.monotonic_ns())
        if not expired:
            return []
        transfer_ids = tuple(state.task.transfer_id for state in expired)
        for state in expired:
            transfer_id = state.task.transfer_id
            if self.is_dflash:
                self._quarantine_source_slot(transfer_id)
            self._ledger.finish(transfer_id)
            self._release_draft_request(state.task.request_id)
        self._cancelled_transfers.update(transfer_ids)
        self._drop_cancelled_unsent()
        if self.is_token_only:
            # Token-only rounds have neither SourceReadDone nor an active
            # ledger entry after timeout.  A late result is therefore stale,
            # so no later event exists (or is needed) to retire this marker.
            self._cancelled_transfers.difference_update(transfer_ids)
        cancellation = CancelTransfers(
            protocol_version=PROTOCOL_VERSION,
            target_instance_id=self.instance_id,
            transfer_ids=transfer_ids,
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
            for transfer_id in transfer_ids
        ]

    def _consume_source_read_done(self, message) -> None:
        from dt_protocol import (
            PROTOCOL_VERSION,
            SOURCE_READ_DONE,
            ControlAck,
            source_read_done_id,
        )

        if message.target_instance_id != self.instance_id:
            raise ValueError("SourceReadDone belongs to another target")
        if (
            not message.draft_instance_id
            or not message.instance_session_id
            or message.coordinator_session_id
            != self._client.coordinator_session_id
        ):
            raise ValueError(
                "SourceReadDone lacks the current Coordinator/Draft session fence"
            )
        message_id = source_read_done_id(message)
        if message_id in self._source_done_applied_ids:
            self._flush_reliable_outbox()
            return
        source_ref = self._source_slot_by_transfer.get(message.transfer_id)
        if source_ref is not None and source_ref != (
            message.source_slot_id,
            message.slot_generation,
        ):
            raise ValueError(
                "SourceReadDone slot generation does not match the transfer"
            )
        if self.is_dflash:
            if self._feature_source_pool is None:
                raise RuntimeError("DFLASH feature source pool is unavailable")
            self._feature_source_pool.read_done(
                transfer_id=message.transfer_id,
                source_slot_id=message.source_slot_id,
                slot_generation=message.slot_generation,
                source_session_id=message.source_session_id,
                pool_generation=message.pool_generation,
            )
            self._source_slot_by_transfer.pop(message.transfer_id, None)
            self._source_deadline_by_transfer.pop(message.transfer_id, None)
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
        self._flush_reliable_outbox()

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
        events = []
        for transfer_id in disposition.failed_transfer_ids:
            if self.is_dflash:
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
        if self.is_dflash:
            for entry in batch.entries:
                if entry.status == EntryStatus.OK:
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
                self._release_source_slot(transfer_id)
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

    def _release_source_slot(self, transfer_id: str) -> None:
        self._source_deadline_by_transfer.pop(transfer_id, None)
        source_ref = self._source_slot_by_transfer.pop(transfer_id, None)
        if source_ref is not None:
            source_slot, _generation = source_ref
            if self.is_dflash:
                assert self._feature_source_pool is not None
                self._feature_source_pool.cancel(
                    transfer_id=transfer_id,
                    source_slot_id=source_slot,
                    slot_generation=_generation,
                    source_session_id=self._feature_source_pool.source_session_id,
                    pool_generation=self._feature_source_pool.pool_generation,
                )
            else:
                self._free_source_slots.append(source_slot)

    def _quarantine_source_slot(self, transfer_id: str) -> None:
        """Fence an uncertain DFlash read without ever reusing its GPU range."""

        self._source_deadline_by_transfer.pop(transfer_id, None)
        # Keep the transfer-to-slot fence while the reader may still exist.
        # A later ordered read-done, or OS-proven Draft process retirement,
        # must be able to resolve this quarantined slot and return it to the
        # bounded source pool safely.
        source_ref = self._source_slot_by_transfer.get(transfer_id)
        if source_ref is None:
            return
        if not self.is_dflash or self._feature_source_pool is None:
            raise RuntimeError("feature quarantine requires a DFlash source pool")
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
