from __future__ import annotations

import ctypes
import logging
import os
import re
import struct
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from dt_profile import emit as profile_emit
from sglang.srt.managers.overlap_utils import RelayPayload
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.flashmtp_shared import (
    adaptive_verify_enabled,
    pinned_verify_metadata_enabled,
    ragged_verify_enabled,
    validate_component_dependencies,
)
from sglang.srt.utils import broadcast_pyobj
from sglang.srt.utils.common import is_pin_memory_available

logger = logging.getLogger(__name__)
CONTROL_RETRY_INTERVAL_NS = 50_000_000


@dataclass(slots=True)
class ParkedTargetRound:
    req: Req
    transfer_id: str
    round_id: int
    attempt: int
    bonus_token: int
    seq_len: int
    cohort_id: int = 0
    cohort_size: int = 1
    cohort_phase: str = "unknown"
    parked_ns: int = 0
    result_ready_ns: int = 0
    lane_id: int = 0
    candidate_bytes: bytes | None = None
    verify_len: int | None = None
    padding_priority: int | None = None
    failure: str | None = None


@dataclass(slots=True)
class StagedPivotBatch:
    """Direct-mmap slots filled on SGLang's asynchronous result-copy stream."""

    source_slots: tuple[int, ...]
    source_generations: tuple[int, ...]
    request_ids: tuple[str, ...]
    enqueued_ns: int


@dataclass(slots=True)
class StagedPinnedPivotBatch:
    """Pinned staging lease filled on the asynchronous result-copy stream."""

    buffer_slot: int
    rows: int
    request_ids: tuple[str, ...]
    enqueued_ns: int


@dataclass(slots=True)
class ReadyMetadataSlot:
    """Pinned/device buffers kept alive through one Target verify result."""

    req_pool_indices_cpu: torch.Tensor
    req_pool_indices: torch.Tensor
    seq_lens_cpu: torch.Tensor
    seq_lens: torch.Tensor
    bonus_tokens_cpu: torch.Tensor
    bonus_tokens: torch.Tensor
    draft_tokens_cpu: torch.Tensor
    draft_tokens: torch.Tensor
    verify_lens_cpu: torch.Tensor
    padding_priorities_cpu: torch.Tensor


class DTTargetController:
    """Owns WAIT_DRAFT requests outside the normal SGLang running batch."""

    def __init__(self, scheduler) -> None:
        try:
            from dt_client import (
                RetryOutbox,
                TargetCoordinatorClient,
                TargetRoundLedger,
            )
            from dt_transport import MmapSlotStore, cleanup_stale_slot_files
        except ImportError as exc:
            raise RuntimeError(
                "Draft-Target mode requires dt-runtime on PYTHONPATH"
            ) from exc

        self.scheduler = scheduler
        self.server_args = scheduler.server_args
        self.instance_id = self.server_args.draft_target_instance_id
        self.profile_id = self.server_args.draft_target_profile_id
        self.feature_schema_id = self.server_args.draft_target_feature_schema_id
        self.block_size = int(self.server_args.speculative_num_draft_tokens)
        validate_component_dependencies()
        adaptive_component = adaptive_verify_enabled()
        adaptive_margin = os.environ.get("SGLANG_DT_ADAPTIVE_VERIFY_MARGIN")
        self.ragged_verify = ragged_verify_enabled() or adaptive_margin is not None
        self.pinned_verify_metadata = (
            pinned_verify_metadata_enabled() or adaptive_margin is not None
        )
        self.adaptive_verify_margin = (
            0.0
            if adaptive_component
            else (float(adaptive_margin) if adaptive_margin is not None else None)
        )
        if self.adaptive_verify_margin is not None and self.adaptive_verify_margin < 0:
            raise ValueError("SGLANG_DT_ADAPTIVE_VERIFY_MARGIN must be non-negative")
        self.adaptive_verify_min_target_batch = int(
            os.environ.get(
                "SGLANG_FLASHMTP_ADAPTIVE_MIN_BATCH",
                "24" if adaptive_component else os.environ.get(
                    "SGLANG_DT_ADAPTIVE_VERIFY_MIN_TARGET_BATCH", "1"
                ),
            )
        )
        if self.adaptive_verify_min_target_batch <= 0:
            raise ValueError(
                "SGLANG_DT_ADAPTIVE_VERIFY_MIN_TARGET_BATCH must be positive"
            )
        self.adaptive_padding_priority = (
            os.environ.get("SGLANG_DT_ADAPTIVE_PADDING_PRIORITY", "0") == "1"
        )
        self.reuse_result_metadata = (
            os.environ.get("SGLANG_DT_REUSE_RESULT_METADATA", "0") == "1"
        )
        self.reuse_ready_metadata = (
            os.environ.get("SGLANG_DT_REUSE_READY_METADATA", "0") == "1"
        )
        if self.adaptive_padding_priority and self.adaptive_verify_margin is None:
            raise ValueError(
                "SGLANG_DT_ADAPTIVE_PADDING_PRIORITY requires "
                "SGLANG_DT_ADAPTIVE_VERIFY_MARGIN"
            )
        if self.adaptive_padding_priority and self.block_size != 8:
            raise ValueError(
                "SGLANG_DT_ADAPTIVE_PADDING_PRIORITY is calibrated only for "
                "block_size=8"
            )
        self.result_row_width = (
            self.block_size
            + int(self.adaptive_verify_margin is not None)
            + int(self.adaptive_padding_priority)
        )
        self.pivot_width = int(scheduler.draft_worker._pivot_width)
        self.round_timeout_ns = (
            int(self.server_args.draft_target_round_timeout_ms) * 1_000_000
        )
        ready_batch_wait_ms = float(self.server_args.draft_target_ready_batch_wait_ms)
        self.ready_batch_threshold = int(
            self.server_args.draft_target_ready_batch_threshold
        )
        prefill_batch_wait_ms = float(
            self.server_args.draft_target_prefill_batch_wait_ms
        )
        self.prefill_batch_threshold = int(
            self.server_args.draft_target_prefill_batch_threshold
        )
        self.prefill_flush_on_gpu_idle = (
            os.environ.get("SGLANG_DT_TARGET_PREFILL_FLUSH_ON_GPU_IDLE", "0") == "1"
        )
        prefill_flush_launch_age_ms = float(
            os.environ.get("SGLANG_DT_TARGET_PREFILL_FLUSH_LAUNCH_AGE_MS", "0")
        )
        self.direct_pinned_mmap_export = bool(
            getattr(
                self.server_args,
                "draft_target_direct_pinned_mmap_export",
                False,
            )
        )
        self.early_pinned_staging = (
            os.environ.get("SGLANG_DT_EARLY_PINNED_STAGING", "0") == "1"
        )
        if self.early_pinned_staging and self.direct_pinned_mmap_export:
            raise ValueError(
                "SGLANG_DT_EARLY_PINNED_STAGING cannot be combined with "
                "draft_target_direct_pinned_mmap_export"
            )
        self.full_wave = bool(
            getattr(self.server_args, "draft_target_full_wave", False)
        )
        self.pipeline_lanes = int(
            getattr(self.server_args, "draft_target_pipeline_lanes", 1)
        )
        lane_weights_arg = getattr(
            self.server_args, "draft_target_pipeline_lane_weights", None
        )
        self.pipeline_lane_slack = int(
            getattr(self.server_args, "draft_target_pipeline_lane_slack", 0)
        )
        self.pipeline_idle_slack = int(
            getattr(self.server_args, "draft_target_pipeline_idle_slack", 0)
        )
        self.pipeline_cross_lane_fill_threshold = int(
            getattr(
                self.server_args,
                "draft_target_pipeline_cross_lane_fill_threshold",
                0,
            )
        )
        self.pipeline_lane_collapse_threshold = int(
            getattr(
                self.server_args,
                "draft_target_pipeline_lane_collapse_threshold",
                0,
            )
        )
        self.pipeline_lane_tail_merge_threshold = int(
            getattr(
                self.server_args,
                "draft_target_pipeline_lane_tail_merge_threshold",
                0,
            )
        )
        self.pipeline_pooled_ready_min_rows = int(
            os.environ.get("SGLANG_DT_TARGET_PIPELINE_POOLED_MIN_ROWS", "0")
        )
        self.pipeline_pooled_ready_idle_only = (
            os.environ.get("SGLANG_DT_TARGET_PIPELINE_POOLED_IDLE_ONLY", "0") == "1"
        )
        full_wave_refill_wait_ms = float(
            getattr(self.server_args, "draft_target_full_wave_refill_wait_ms", 0.0)
        )
        overlap_result_defer_ms = float(
            getattr(self.server_args, "draft_target_overlap_result_defer_ms", 0.0)
        )
        if ready_batch_wait_ms < 0:
            raise ValueError("draft_target_ready_batch_wait_ms must be non-negative")
        if self.ready_batch_threshold <= 0:
            raise ValueError("draft_target_ready_batch_threshold must be positive")
        if prefill_batch_wait_ms < 0:
            raise ValueError("draft_target_prefill_batch_wait_ms must be non-negative")
        if prefill_flush_launch_age_ms < 0:
            raise ValueError(
                "SGLANG_DT_TARGET_PREFILL_FLUSH_LAUNCH_AGE_MS must be "
                "non-negative"
            )
        if self.prefill_batch_threshold <= 0:
            raise ValueError("draft_target_prefill_batch_threshold must be positive")
        if full_wave_refill_wait_ms < 0:
            raise ValueError(
                "draft_target_full_wave_refill_wait_ms must be non-negative"
            )
        if overlap_result_defer_ms < 0:
            raise ValueError(
                "draft_target_overlap_result_defer_ms must be non-negative"
            )
        if overlap_result_defer_ms > 0 and not getattr(
            self.server_args, "draft_target_overlap_schedule", False
        ):
            raise ValueError(
                "draft_target_overlap_result_defer_ms requires "
                "draft_target_overlap_schedule"
            )
        if self.pipeline_lanes <= 0:
            raise ValueError("draft_target_pipeline_lanes must be positive")
        if self.pipeline_lanes > 1 and not self.full_wave:
            raise ValueError(
                "draft_target_pipeline_lanes greater than one requires "
                "draft_target_full_wave"
            )
        if self.pipeline_lane_slack < 0:
            raise ValueError("draft_target_pipeline_lane_slack must be non-negative")
        if self.pipeline_lane_slack > 0 and self.pipeline_lanes <= 1:
            raise ValueError(
                "draft_target_pipeline_lane_slack requires more than one pipeline lane"
            )
        if self.pipeline_idle_slack < 0:
            raise ValueError(
                "draft_target_pipeline_idle_slack must be non-negative"
            )
        if self.pipeline_idle_slack > 0:
            if self.pipeline_lanes <= 1:
                raise ValueError(
                    "draft_target_pipeline_idle_slack requires more than one "
                    "pipeline lane"
                )
            if self.pipeline_idle_slack < self.pipeline_lane_slack:
                raise ValueError(
                    "draft_target_pipeline_idle_slack must be at least the "
                    "normal lane slack"
                )
            if not getattr(
                self.server_args, "draft_target_overlap_schedule", False
            ):
                raise ValueError(
                    "draft_target_pipeline_idle_slack requires overlap schedule"
                )
        if self.pipeline_cross_lane_fill_threshold < 0:
            raise ValueError(
                "draft_target_pipeline_cross_lane_fill_threshold must be non-negative"
            )
        if self.pipeline_cross_lane_fill_threshold > 0 and self.pipeline_lanes <= 1:
            raise ValueError(
                "draft_target_pipeline_cross_lane_fill_threshold requires more "
                "than one pipeline lane"
            )
        if self.pipeline_lane_collapse_threshold < 0:
            raise ValueError(
                "draft_target_pipeline_lane_collapse_threshold must be non-negative"
            )
        if self.pipeline_lane_collapse_threshold > 0 and self.pipeline_lanes <= 1:
            raise ValueError(
                "draft_target_pipeline_lane_collapse_threshold requires more "
                "than one pipeline lane"
            )
        if self.pipeline_lane_tail_merge_threshold < 0:
            raise ValueError(
                "draft_target_pipeline_lane_tail_merge_threshold must be non-negative"
            )
        if self.pipeline_lane_tail_merge_threshold > 0 and self.pipeline_lanes <= 1:
            raise ValueError(
                "draft_target_pipeline_lane_tail_merge_threshold requires "
                "more than one pipeline lane"
            )
        if self.pipeline_pooled_ready_min_rows < 0:
            raise ValueError(
                "SGLANG_DT_TARGET_PIPELINE_POOLED_MIN_ROWS must be non-negative"
            )
        if self.pipeline_pooled_ready_min_rows > 0 and self.pipeline_lanes <= 1:
            raise ValueError(
                "SGLANG_DT_TARGET_PIPELINE_POOLED_MIN_ROWS requires more than "
                "one pipeline lane"
            )
        if lane_weights_arg is None:
            self._pipeline_lane_weights = (1,) * self.pipeline_lanes
        else:
            try:
                self._pipeline_lane_weights = tuple(
                    int(value.strip()) for value in lane_weights_arg.split(",")
                )
            except ValueError as exc:
                raise ValueError(
                    "draft_target_pipeline_lane_weights must contain "
                    "comma-separated positive integers"
                ) from exc
            if len(self._pipeline_lane_weights) != self.pipeline_lanes or any(
                weight <= 0 for weight in self._pipeline_lane_weights
            ):
                raise ValueError(
                    "draft_target_pipeline_lane_weights must contain one "
                    "positive integer per pipeline lane"
                )
        self.ready_batch_wait_ns = int(ready_batch_wait_ms * 1_000_000)
        self.prefill_batch_wait_ns = int(prefill_batch_wait_ms * 1_000_000)
        self.prefill_flush_launch_age_ns = int(
            prefill_flush_launch_age_ms * 1_000_000
        )
        self.full_wave_refill_wait_ns = int(full_wave_refill_wait_ms * 1_000_000)
        self.overlap_result_defer_ns = int(overlap_result_defer_ms * 1_000_000)
        self.is_export_rank = scheduler.ps.tp_rank == 0
        self.tensor_empty_poll_broadcast = (
            os.environ.get("SGLANG_DT_TENSOR_EMPTY_POLL_BROADCAST", "0") == "1"
            and scheduler.ps.tp_size > 1
        )
        self._poll_control = (
            torch.empty(3, dtype=torch.int64)
            if self.tensor_empty_poll_broadcast
            else None
        )
        self._profile_poll_stages = bool(os.environ.get("DT_PROFILE_DIR")) and bool(
            self.is_export_rank
        )
        self._profile_poll_calls = 0
        self._profile_poll_empty_calls = 0
        self._profile_poll_total_ns = 0
        self._profile_poll_export_recv_ns = 0
        self._profile_poll_overlap_query_ns = 0
        self._profile_poll_broadcast_ns = 0
        self._profile_poll_apply_ns = 0
        self._parked: dict[str, ParkedTargetRound] = {}
        # Overlap scheduling can have more than one chunked-prefill forward for
        # the same request in ``result_queue``.  Track the number of launched
        # Target forwards whose results have not been consumed yet.  A request
        # with a positive count may continue through the chunked-prefill path,
        # but must not re-enter the normal running batch or start a Draft
        # transfer from an older result.
        self._target_inflight_counts: Counter[str] = Counter()
        self._target_inflight_reqs: dict[str, Req] = {}
        self._next_round: dict[str, int] = {}
        self._cohort_sequence = 0
        self._batch_sequence = 0
        self._pending_send = deque()
        self._delivery_outbox = RetryOutbox[object](
            retry_interval_ns=CONTROL_RETRY_INTERVAL_NS
        )
        self._cancelled_transfers: set[str] = set()
        self._free_source_slots = deque(range(scheduler.max_running_requests))
        self._source_slot_by_transfer: dict[str, tuple[int, int]] = {}
        self._result_stores: dict[str, object] = {}
        self._last_poll_ns = 0
        self._prefill_wait_started_ns = 0
        self._full_wave_target_size = 0
        self._full_wave_refill_started_ns = 0
        self._full_wave_members: set[str] = set()
        self._lane_by_request: dict[str, int] = {}
        self._lane_target_sizes = [0] * self.pipeline_lanes
        self._lane_refill_started_ns = [0] * self.pipeline_lanes
        self._lane_assignment_cursor = 0
        self._lane_ready_cursor = 0
        self._selected_pipeline_lane = -1
        self._selected_pipeline_lane_lagging_rows = 0
        self._selected_pipeline_lane_slack_used = 0
        self._selected_pipeline_cross_lane_rows = 0
        self._pipeline_lanes_collapsed = False
        self._pipeline_lanes_tail_merged = False
        self._pipeline_lane_tail_merge_armed = False
        self._pipeline_lanes_pooled = False
        self._overlap_launch_poll_ns = 0
        self._overlap_defer_active = False
        self._overlap_result_ready: bool | None = None
        self._client = None
        self._ledger = None
        self._source_store = None
        self._pivot_host_staging = None
        self._early_pivot_host_buffers: list[torch.Tensor] = []
        self._early_pivot_host_free_slots: deque[int] = deque()
        self._early_pivot_host_in_use: set[int] = set()
        self._direct_pivot_mapping_view = None
        self._direct_pivot_mapping_address = None
        self._direct_pivot_slot_views = []
        self._direct_pivot_slot_tensors = []
        self._direct_pivot_registered = False
        self._ready_metadata_slots: list[ReadyMetadataSlot] = []
        self._ready_metadata_free_slots: deque[int] = deque()
        if self.reuse_ready_metadata:
            self._initialize_ready_metadata_slots()
        if self.is_export_rank:
            slot_bytes = (
                self.pivot_width * torch.tensor([], dtype=torch.bfloat16).element_size()
            )
            safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", self.instance_id)
            data_dir = Path(self.server_args.draft_target_data_dir)
            cleanup_stale_slot_files(
                data_dir, instance_id=safe_id, kind="pivots"
            )
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
                    data_endpoint=self._source_store.uri,
                )
                self._client.register_target(
                    timeout_s=self.server_args.draft_target_registration_timeout_s
                )
                self._ledger = TargetRoundLedger(target_instance_id=self.instance_id)
                if self.direct_pinned_mmap_export:
                    self._initialize_direct_pinned_export()
                else:
                    # The DT source-slot ring bounds the largest possible wave.
                    # Keep the original synchronous staging buffer permanently
                    # available; H11 optionally adds two leased early-copy
                    # buffers, for a hard total of three pinned pivot buffers.
                    self._pivot_host_staging = torch.empty(
                        (scheduler.max_running_requests, self.pivot_width),
                        dtype=torch.bfloat16,
                        device="cpu",
                        pin_memory=True,
                    )
                    if self.early_pinned_staging:
                        for buffer_slot in range(2):
                            self._early_pivot_host_buffers.append(
                                torch.empty(
                                    (
                                        scheduler.max_running_requests,
                                        self.pivot_width,
                                    ),
                                    dtype=torch.bfloat16,
                                    device="cpu",
                                    pin_memory=True,
                                )
                            )
                            self._early_pivot_host_free_slots.append(buffer_slot)
            except Exception:
                self._close_direct_pinned_export()
                if self._client is not None:
                    self._client.close()
                    self._client = None
                self._source_store.close()
                self._source_store.unlink()
                self._source_store = None
                raise

    def stage_overlap_pivot_copy(
        self, batch: ScheduleBatch, batch_result
    ) -> None:
        """Start pivot D2H as soon as the Target forward has been submitted.

        The overlap scheduler calls this method from its dedicated result-copy
        stream after making that stream wait for the forward stream.  Filling
        either a pinned staging lease or registered mmap slots here lets the
        next Target forward overlap the pivot D2H.  ``copy_done`` is recorded
        after this method returns, so the normal result-processing fence proves
        the selected host destination is ready before publish.
        """
        if (
            getattr(self, "reuse_result_metadata", False)
            and getattr(batch_result, "dt_input_seq_lens_cpu", None) is None
        ):
            # DFLASH prepare_for_decode uses a reusable CPU planning buffer for
            # seq_lens_cpu.  An overlap result can outlive the next scheduler
            # iteration, so retain a private input-length snapshot while the
            # result-copy stream is being submitted.  next_token_ids and
            # accept_lens are already copied to pinned CPU storage by
            # GenerationBatchResult.copy_to_cpu; together these three CPU
            # tensors are sufficient to recover the next Draft metadata without
            # another synchronous GPU -> CPU transfer in park_completed_batch.
            mixed_prefill_bs = int(getattr(batch, "dt_mixed_prefill_bs", 0) or 0)
            if mixed_prefill_bs:
                # mix_with_dt_verify rewrites the top-level verify seq_lens_cpu
                # to prefix + verify_window so the packed EXTEND attention
                # metadata is correct.  Result reconstruction instead needs
                # each verify row's pre-forward prefix length.  Preserve the
                # two logical snapshots rather than cloning the rewritten
                # packed tensor.
                prefill_batch = getattr(batch, "dt_mixed_prefill_batch", None)
                verify_batch = getattr(batch, "dt_mixed_verify_batch", None)
                prefill_seq_lens_cpu = getattr(
                    prefill_batch, "seq_lens_cpu", None
                )
                verify_seq_lens_cpu = getattr(verify_batch, "seq_lens_cpu", None)
                if (
                    torch.is_tensor(prefill_seq_lens_cpu)
                    and prefill_seq_lens_cpu.device.type == "cpu"
                    and int(prefill_seq_lens_cpu.numel()) == mixed_prefill_bs
                    and torch.is_tensor(verify_seq_lens_cpu)
                    and verify_seq_lens_cpu.device.type == "cpu"
                    and int(verify_seq_lens_cpu.numel())
                    == len(batch.reqs) - mixed_prefill_bs
                ):
                    seq_lens_cpu = torch.cat(
                        [prefill_seq_lens_cpu, verify_seq_lens_cpu]
                    )
                else:
                    seq_lens_cpu = None
            else:
                seq_lens_cpu = getattr(batch, "seq_lens_cpu", None)
            if (
                torch.is_tensor(seq_lens_cpu)
                and seq_lens_cpu.device.type == "cpu"
                and int(seq_lens_cpu.numel()) == len(batch.reqs)
            ):
                batch_result.dt_input_seq_lens_cpu = seq_lens_cpu.detach().clone()

        if (
            self.is_export_rank
            and self.early_pinned_staging
            and getattr(batch_result, "dt_staged_pinned_pivots", None) is None
        ):
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
                raise RuntimeError(
                    "DT pivot rows do not match launched batch requests"
                )
            if rows == 0 or not self._early_pivot_host_free_slots:
                return

            buffer_slot = self._early_pivot_host_free_slots.popleft()
            self._early_pivot_host_in_use.add(buffer_slot)
            enqueue_start_ns = time.monotonic_ns()
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
                enqueued_ns=enqueue_start_ns,
            )
            return

        if (
            not self.is_export_rank
            or not self.direct_pinned_mmap_export
            or not self._direct_pivot_slot_tensors
            or getattr(batch_result, "dt_staged_pivots", None) is not None
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
        if rows == 0 or len(self._free_source_slots) < rows:
            return
        pivot_nbytes = self.pivot_width * pivot_hidden.element_size()
        source_slots = tuple(self._free_source_slots.popleft() for _ in range(rows))
        try:
            source_generations = self._source_store.reserve_write_many(
                tuple((slot, pivot_nbytes) for slot in source_slots)
            )
            enqueue_start_ns = time.monotonic_ns()
            for row, slot in enumerate(source_slots):
                self._direct_pivot_slot_tensors[slot].copy_(
                    pivot_hidden[row], non_blocking=True
                )
        except Exception:
            # A reserved generation is private until it is committed.  Publish
            # it before returning the slot to the free ring so the next reserve
            # can advance from a valid even generation.
            generations = locals().get("source_generations")
            if generations is not None:
                self._source_store.commit_write_many(
                    tuple(zip(source_slots, generations))
                )
            self._free_source_slots.extendleft(reversed(source_slots))
            raise

        batch_result.dt_staged_pivots = StagedPivotBatch(
            source_slots=source_slots,
            source_generations=source_generations,
            request_ids=tuple(req.rid for req in batch.reqs),
            enqueued_ns=enqueue_start_ns,
        )
        profile_emit(
            "target_pivot_batch_enqueue",
            instance_id=self.instance_id,
            rows=rows,
            nbytes=rows * pivot_nbytes,
            enqueue_ns=time.monotonic_ns() - enqueue_start_ns,
            direct_mmap=True,
            early_copy_stream=True,
        )

    def _release_early_pinned_staging(self, buffer_slot: int) -> None:
        if buffer_slot not in self._early_pivot_host_in_use:
            raise RuntimeError(
                f"DT early pinned staging slot {buffer_slot} is not leased"
            )
        self._early_pivot_host_in_use.remove(buffer_slot)
        self._early_pivot_host_free_slots.append(buffer_slot)

    def _metadata_from_cpu_result(
        self,
        batch: ScheduleBatch,
        batch_result,
        active_indices: list[int],
        original_batch_size: int,
    ) -> tuple[list[int], list[int]] | None:
        """Recover Draft root tokens and sequence lengths from existing D2H.

        The overlap result processor already waits for next_token_ids and
        accept_lens copied by GenerationBatchResult.copy_to_cpu.  Reusing them
        avoids copying next_draft_input.bonus_tokens and new_seq_lens from the
        GPU a second time.  Return None for unsupported/synthetic batches so the
        established direct metadata copy remains the correctness fallback.
        """

        if (
            not getattr(self, "reuse_result_metadata", False)
            or batch_result is None
        ):
            return None
        if not active_indices:
            return ([], [])

        input_seq_lens = getattr(batch_result, "dt_input_seq_lens_cpu", None)
        next_token_ids = getattr(batch_result, "next_token_ids", None)
        if (
            not torch.is_tensor(input_seq_lens)
            or input_seq_lens.device.type != "cpu"
            or int(input_seq_lens.numel()) != original_batch_size
            or not torch.is_tensor(next_token_ids)
            or next_token_ids.device.type != "cpu"
        ):
            return None

        forward_mode = getattr(batch, "forward_mode", None)
        mixed_prefill_bs = int(getattr(batch, "dt_mixed_prefill_bs", 0) or 0)
        if mixed_prefill_bs:
            accept_lens = getattr(batch_result, "accept_lens", None)
            stride = int(
                getattr(batch_result, "speculative_num_draft_tokens", 0)
                or self.block_size
            )
            if (
                mixed_prefill_bs >= original_batch_size
                or not torch.is_tensor(accept_lens)
                or accept_lens.device.type != "cpu"
                or int(accept_lens.numel()) != original_batch_size
                or int(next_token_ids.numel())
                < original_batch_size * stride
            ):
                return None
            commit_lens = accept_lens.to(dtype=torch.int64).reshape(-1)
            if (
                bool(torch.any(commit_lens < 1))
                or bool(torch.any(commit_lens > stride))
                or bool(torch.any(commit_lens[:mixed_prefill_bs] != 1))
            ):
                return None
            row_ids = torch.tensor(active_indices, dtype=torch.int64)
            selected_commit_lens = commit_lens[row_ids]
            token_offsets = row_ids * stride + selected_commit_lens - 1
            bonus_tokens = next_token_ids.reshape(-1)[token_offsets]
            seq_lens = input_seq_lens.reshape(-1)[row_ids].clone()
            verify_rows = row_ids >= mixed_prefill_bs
            seq_lens[verify_rows] += selected_commit_lens[verify_rows]
        elif forward_mode is not None and forward_mode.is_decode():
            accept_lens = getattr(batch_result, "accept_lens", None)
            stride = int(
                getattr(batch_result, "speculative_num_draft_tokens", 0)
                or self.block_size
            )
            if (
                not torch.is_tensor(accept_lens)
                or accept_lens.device.type != "cpu"
                or int(accept_lens.numel()) != original_batch_size
                or int(next_token_ids.numel())
                < original_batch_size * stride
            ):
                return None
            commit_lens = accept_lens.to(dtype=torch.int64).reshape(-1)
            if bool(torch.any(commit_lens < 1)) or bool(
                torch.any(commit_lens > stride)
            ):
                return None
            row_ids = torch.tensor(active_indices, dtype=torch.int64)
            selected_commit_lens = commit_lens[row_ids]
            token_offsets = row_ids * stride + selected_commit_lens - 1
            bonus_tokens = next_token_ids.reshape(-1)[token_offsets]
            seq_lens = input_seq_lens.reshape(-1)[row_ids] + selected_commit_lens
        elif forward_mode is not None and forward_mode.is_extend():
            if int(next_token_ids.numel()) != original_batch_size:
                return None
            row_ids = torch.tensor(active_indices, dtype=torch.int64)
            bonus_tokens = next_token_ids.reshape(-1)[row_ids]
            # Prefill samples the root token but has committed only the prompt
            # prefix to Target KV, matching next_draft_input.new_seq_lens.
            seq_lens = input_seq_lens.reshape(-1)[row_ids]
        else:
            return None

        return (
            [int(value) for value in bonus_tokens.tolist()],
            [int(value) for value in seq_lens.tolist()],
        )

    def park_completed_batch(self, batch: ScheduleBatch, *, batch_result=None) -> None:
        staged_pinned_pivots = getattr(
            batch_result, "dt_staged_pinned_pivots", None
        )
        try:
            self._park_completed_batch(batch, batch_result=batch_result)
        finally:
            if staged_pinned_pivots is not None:
                if (
                    getattr(batch_result, "dt_staged_pinned_pivots", None)
                    is staged_pinned_pivots
                ):
                    batch_result.dt_staged_pinned_pivots = None
                self._release_early_pinned_staging(
                    staged_pinned_pivots.buffer_slot
                )

    def _park_completed_batch(self, batch: ScheduleBatch, *, batch_result=None) -> None:
        self._release_ready_metadata_slot(batch)
        staged_pivots = (
            getattr(batch_result, "dt_staged_pivots", None)
            if batch_result is not None
            else None
        )
        staged_pinned_pivots = (
            getattr(batch_result, "dt_staged_pinned_pivots", None)
            if batch_result is not None
            else None
        )
        if staged_pivots is not None:
            # ``process_batch_result`` synchronizes ``copy_done`` before this
            # hook.  Commit every reserved row, including terminal rows that
            # will be recycled without a Draft task.
            staged_commit_start_ns = time.monotonic_ns()
            self._source_store.commit_write_many(
                tuple(
                    zip(
                        staged_pivots.source_slots,
                        staged_pivots.source_generations,
                    )
                )
            )
            staged_commit_ns = time.monotonic_ns() - staged_commit_start_ns
            batch_result.dt_staged_pivots = None
        if not batch.reqs:
            if staged_pivots is not None:
                self._free_source_slots.extend(staged_pivots.source_slots)
            return
        self._release_target_inflight_batch(batch)
        if batch.spec_info is None:
            return
        if not isinstance(batch.spec_info, DFlashDraftInputV2):
            raise TypeError("DT Target can park only DFlashDraftInputV2 batches")

        park_start_ns = time.monotonic_ns()
        original_batch_size = len(batch.reqs)
        draft_input = batch.spec_info
        if getattr(self, "pipeline_lanes", 1) > 1:
            for req in batch.reqs:
                if req.finished() or req.is_retracted:
                    self._forget_request_lane(req)
        for req in batch.reqs:
            if req.finished() or req.is_retracted:
                getattr(self, "_next_round", {}).pop(req.rid, None)
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
        if active_indices:
            self._cohort_sequence += 1
        cohort_id = self._cohort_sequence
        cohort_size = len(active_indices)
        tasks = []
        parked_reqs = []
        round_ids = [
            self._next_round.get(batch.reqs[index].rid, 0) for index in active_indices
        ]
        lane_ids = [
            self._lane_for_request(batch.reqs[index]) for index in active_indices
        ]
        pipeline_lane = (
            lane_ids[0]
            if lane_ids and all(value == lane_ids[0] for value in lane_ids)
            else -1
        )
        cohort_phase = (
            "empty" if not round_ids else "prefill" if max(round_ids) == 0 else "verify"
        )
        pivot_payloads = None
        source_slots = None
        source_generations = None
        if staged_pivots is not None:
            staged_index_by_rid = {
                rid: index for index, rid in enumerate(staged_pivots.request_ids)
            }
            try:
                staged_active_indices = tuple(
                    staged_index_by_rid[batch.reqs[index].rid]
                    for index in active_indices
                )
            except KeyError as exc:
                raise RuntimeError(
                    "DT completed request is missing from staged pivot batch"
                ) from exc
            active_index_set = set(staged_active_indices)
            source_slots = tuple(
                staged_pivots.source_slots[index] for index in staged_active_indices
            )
            source_generations = tuple(
                staged_pivots.source_generations[index]
                for index in staged_active_indices
            )
            self._free_source_slots.extend(
                staged_pivots.source_slots[index]
                for index in range(len(staged_pivots.source_slots))
                if index not in active_index_set
            )
            profile_emit(
                "target_pivot_batch_copy",
                instance_id=self.instance_id,
                rows=len(active_indices),
                nbytes=len(active_indices) * self.pivot_width * 2,
                gpu_to_cpu_ns=0,
                completion_observed_ns=(
                    time.monotonic_ns() - staged_pivots.enqueued_ns
                ),
                prepare_views_ns=0,
                direct_mmap=True,
                async_mmap=True,
                early_copy_stream=True,
            )
            profile_emit(
                "target_pivot_batch_publish",
                instance_id=self.instance_id,
                rows=len(active_indices),
                nbytes=len(active_indices) * self.pivot_width * 2,
                mmap_write_ns=staged_commit_ns,
                direct_mmap=True,
                async_mmap=True,
                early_copy_stream=True,
            )
        if getattr(self, "is_export_rank", False) and active_indices:
            if staged_pivots is None:
                direct_export = bool(
                    staged_pinned_pivots is None
                    and getattr(self, "direct_pinned_mmap_export", False)
                    and self._direct_pivot_slot_tensors
                )
                if staged_pinned_pivots is not None:
                    if (
                        staged_pinned_pivots.rows
                        != len(staged_pinned_pivots.request_ids)
                    ):
                        raise RuntimeError(
                            "DT staged pivot row and request counts differ"
                        )
                    staged_index_by_rid = {
                        rid: index
                        for index, rid in enumerate(
                            staged_pinned_pivots.request_ids
                        )
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
                    prepare_views_start_ns = time.monotonic_ns()
                    pivot_cpu = self._early_pivot_host_buffers[buffer_slot][
                        : staged_pinned_pivots.rows
                    ]
                    pivot_rows = pivot_cpu.view(torch.uint8).numpy()
                    pivot_payloads = [
                        memoryview(pivot_rows[index])
                        for index in staged_active_indices
                    ]
                    copy_end_ns = time.monotonic_ns()
                    pivot_copy_nbytes = (
                        staged_pinned_pivots.rows * self.pivot_width * 2
                    )
                    pivot_publish_nbytes = (
                        len(active_indices) * self.pivot_width * 2
                    )
                    prepare_views_ns = copy_end_ns - prepare_views_start_ns
                else:
                    pivot_hidden = draft_input.hidden_states
                    if len(active_indices) != original_batch_size:
                        pivot_hidden = pivot_hidden[active_indices]
                    if pivot_hidden.dtype != torch.bfloat16:
                        raise TypeError(
                            f"DT Target expected bfloat16 pivot, got {pivot_hidden.dtype}"
                        )
                    copy_start_ns = time.monotonic_ns()
                    pivot_gpu = pivot_hidden.detach().contiguous().view(
                        len(active_indices), -1
                    )
                    if not direct_export:
                        pivot_host_staging = getattr(
                            self, "_pivot_host_staging", None
                        )
                        if pivot_host_staging is None:
                            # Keep construction-through-``__new__`` test doubles
                            # and CPU transports usable without a pinned allocator.
                            pivot_cpu = pivot_gpu.cpu()
                        else:
                            pivot_cpu = pivot_host_staging[: len(active_indices)]
                            pivot_cpu.copy_(pivot_gpu, non_blocking=False)
                        copy_end_ns = time.monotonic_ns()
                        pivot_bytes = pivot_cpu.view(torch.uint8)
                        pivot_rows = pivot_bytes.numpy()
                        pivot_payloads = [memoryview(row) for row in pivot_rows]
                        pivot_copy_nbytes = pivot_bytes.numel()
                        pivot_publish_nbytes = pivot_bytes.numel()
                        prepare_views_ns = time.monotonic_ns() - copy_end_ns
            else:
                direct_export = True
            if direct_export and staged_pivots is None:
                pivot_nbytes = (
                    self.pivot_width
                    * torch.tensor([], dtype=torch.bfloat16).element_size()
                )
                acquired_source_slots = []
                try:
                    if len(self._free_source_slots) < len(active_indices):
                        raise RuntimeError(
                            "DT Target source slot ring is exhausted"
                        )
                    for _ in active_indices:
                        acquired_source_slots.append(
                            self._free_source_slots.popleft()
                        )
                    source_slots = tuple(acquired_source_slots)
                    source_generations = self._source_store.reserve_write_many(
                        tuple((slot, pivot_nbytes) for slot in source_slots)
                    )
                    for row, slot in enumerate(source_slots):
                        self._direct_pivot_slot_tensors[slot].copy_(
                            pivot_gpu[row], non_blocking=True
                        )
                    torch.get_device_module(
                        pivot_gpu.device
                    ).current_stream().synchronize()
                    copy_end_ns = time.monotonic_ns()
                    commit_start_ns = copy_end_ns
                    self._source_store.commit_write_many(
                        tuple(zip(source_slots, source_generations))
                    )
                except Exception:
                    self._free_source_slots.extendleft(
                        reversed(acquired_source_slots)
                    )
                    raise
                profile_emit(
                    "target_pivot_batch_copy",
                    instance_id=self.instance_id,
                    rows=len(active_indices),
                    nbytes=len(active_indices) * pivot_nbytes,
                    gpu_to_cpu_ns=copy_end_ns - copy_start_ns,
                    prepare_views_ns=0,
                    direct_mmap=True,
                )
                profile_emit(
                    "target_pivot_batch_publish",
                    instance_id=self.instance_id,
                    rows=len(active_indices),
                    nbytes=len(active_indices) * pivot_nbytes,
                    mmap_write_ns=time.monotonic_ns() - commit_start_ns,
                    direct_mmap=True,
                )
            elif not direct_export:
                profile_emit(
                    "target_pivot_batch_copy",
                    instance_id=self.instance_id,
                    rows=len(active_indices),
                    nbytes=pivot_copy_nbytes,
                    gpu_to_cpu_ns=(
                        0
                        if staged_pinned_pivots is not None
                        else copy_end_ns - copy_start_ns
                    ),
                    completion_observed_ns=(
                        copy_end_ns - staged_pinned_pivots.enqueued_ns
                        if staged_pinned_pivots is not None
                        else 0
                    ),
                    prepare_views_ns=prepare_views_ns,
                    direct_mmap=False,
                    early_copy_stream=staged_pinned_pivots is not None,
                )
                write_start_ns = time.monotonic_ns()
                acquired_source_slots = []
                try:
                    if len(self._free_source_slots) < len(active_indices):
                        raise RuntimeError(
                            "DT Target source slot ring is exhausted"
                        )
                    for _ in active_indices:
                        acquired_source_slots.append(
                            self._free_source_slots.popleft()
                        )
                    source_slots = tuple(acquired_source_slots)
                    source_generations = self._source_store.write_many(
                        tuple(zip(source_slots, pivot_payloads))
                    )
                except Exception:
                    self._free_source_slots.extendleft(
                        reversed(acquired_source_slots)
                    )
                    raise
                profile_emit(
                    "target_pivot_batch_publish",
                    instance_id=self.instance_id,
                    rows=len(active_indices),
                    nbytes=pivot_publish_nbytes,
                    mmap_write_ns=time.monotonic_ns() - write_start_ns,
                    direct_mmap=False,
                )
        metadata_copy_start_ns = time.monotonic_ns()
        reused_metadata = self._metadata_from_cpu_result(
            batch,
            batch_result,
            active_indices,
            original_batch_size,
        )
        if reused_metadata is not None:
            bonus_tokens_cpu, seq_lens_cpu = reused_metadata
            metadata_source = "result_cpu"
        elif active_indices:
            bonus_values = draft_input.bonus_tokens
            seq_len_values = draft_input.new_seq_lens
            if len(active_indices) != original_batch_size:
                bonus_values = bonus_values[active_indices]
                seq_len_values = seq_len_values[active_indices]
            # Two batched transfers replace two CUDA ``item()`` calls per row.
            bonus_tokens_cpu = bonus_values.detach().cpu().tolist()
            seq_lens_cpu = seq_len_values.detach().cpu().tolist()
            metadata_source = "draft_input_d2h"
        else:
            bonus_tokens_cpu = []
            seq_lens_cpu = []
            metadata_source = "empty"
        metadata_copy_end_ns = time.monotonic_ns()
        task_build_start_ns = time.monotonic_ns()
        for active_index, index in enumerate(active_indices):
            req = batch.reqs[index]
            round_id = self._next_round.get(req.rid, 0)
            self._next_round[req.rid] = round_id + 1
            attempt = 0
            transfer_id = f"{self.instance_id}:{req.rid}:{round_id}:{attempt}"
            bonus_token = int(bonus_tokens_cpu[active_index])
            seq_len = int(seq_lens_cpu[active_index])
            self._parked[transfer_id] = ParkedTargetRound(
                req=req,
                transfer_id=transfer_id,
                round_id=round_id,
                attempt=attempt,
                bonus_token=bonus_token,
                seq_len=seq_len,
                cohort_id=cohort_id,
                cohort_size=cohort_size,
                cohort_phase=cohort_phase,
                parked_ns=park_start_ns,
                lane_id=lane_ids[active_index],
            )
            parked_reqs.append(req)

            if getattr(self, "is_export_rank", False):
                source_slot = source_slots[active_index]
                source_generation = source_generations[active_index]
                self._source_slot_by_transfer[transfer_id] = (
                    source_slot,
                    source_generation,
                )
                tasks.append(
                    self._make_task(
                        transfer_id=transfer_id,
                        req=req,
                        round_id=round_id,
                        attempt=attempt,
                        bonus_token=bonus_token,
                        source_slot=source_slot,
                        source_generation=source_generation,
                        pivot_nbytes=(
                            self.pivot_width * 2
                            if pivot_payloads is None
                            else len(pivot_payloads[active_index])
                        ),
                    )
                )

        task_build_end_ns = time.monotonic_ns()
        if tasks:
            self._queue_task_batch(tuple(tasks))
            # Source slots are committed at this point, so publish the control
            # message in the same scheduler turn. Waiting for the next poll
            # adds a fixed idle bubble before Draft can start the round.
            self._flush_pending_send()
        task_queue_end_ns = time.monotonic_ns()
        if getattr(self, "full_wave", False) and parked_reqs:
            if getattr(self, "pipeline_lanes", 1) > 1:
                parked_by_lane = Counter(
                    state.lane_id for state in self._parked.values()
                )
                for lane_id, parked_count in parked_by_lane.items():
                    self._lane_target_sizes[lane_id] = max(
                        self._lane_target_sizes[lane_id], parked_count
                    )
            else:
                self._full_wave_target_size = max(
                    getattr(self, "_full_wave_target_size", 0), len(self._parked)
                )
                # Freeze the current wave at its first export.  Replacement
                # prefills that arrive while this wave is remote stay parked,
                # but join only after the current members verify and re-export.
                # This overlaps refill work without either delaying the current
                # wave or launching a singleton verify.
                if not self._full_wave_members:
                    self._full_wave_members.update(self._parked)
        wave_update_end_ns = time.monotonic_ns()
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
        detach_end_ns = time.monotonic_ns()
        if getattr(self, "is_export_rank", False):
            launch_ts = getattr(batch, "launch_ts", 0.0) or 0.0
            profile_emit(
                "target_park_batch",
                instance_id=self.instance_id,
                phase=cohort_phase,
                cohort_id=cohort_id,
                cohort_size=cohort_size,
                pipeline_lane=pipeline_lane,
                round_id_min=min(round_ids, default=-1),
                round_id_max=max(round_ids, default=-1),
                original_batch_size=original_batch_size,
                parked_count=len(parked_reqs),
                target_step_before_export_ns=(
                    park_start_ns - int(launch_ts * 1_000_000_000) if launch_ts else 0
                ),
                metadata_copy_ns=metadata_copy_end_ns - metadata_copy_start_ns,
                metadata_source=metadata_source,
                task_build_ns=task_build_end_ns - task_build_start_ns,
                task_queue_ns=task_queue_end_ns - task_build_end_ns,
                wave_update_ns=wave_update_end_ns - task_queue_end_ns,
                detach_ns=detach_end_ns - wave_update_end_ns,
                park_and_export_ns=time.monotonic_ns() - park_start_ns,
                parked_rounds=len(self._parked),
            )

    def poll(self, *, overlap_result=None) -> None:
        poll_start_ns = time.monotonic_ns()
        profile_poll_stages = getattr(self, "_profile_poll_stages", False)
        packet = None
        if self.is_export_rank:
            events = self._poll_export_rank()
            export_recv_done_ns = (
                time.monotonic_ns() if profile_poll_stages else poll_start_ns
            )
            result_ready = self._query_overlap_result_ready(overlap_result)
            overlap_query_done_ns = (
                time.monotonic_ns() if profile_poll_stages else export_recv_done_ns
            )
            packet = (time.monotonic_ns(), events, result_ready)
        else:
            export_recv_done_ns = poll_start_ns
            overlap_query_done_ns = poll_start_ns
        if self.scheduler.ps.tp_size > 1:
            if getattr(self, "tensor_empty_poll_broadcast", False):
                control = self._poll_control
                assert control is not None
                if self.is_export_rank:
                    control[0] = packet[0]
                    control[1] = self._encode_poll_result_ready(packet[2])
                    control[2] = len(packet[1])
                torch.distributed.broadcast(
                    control,
                    src=self.scheduler.tp_group.ranks[0],
                    group=self.scheduler.tp_cpu_group,
                )
                poll_now_ns, encoded_result_ready, event_count = control.tolist()
                if event_count:
                    packet = broadcast_pyobj(
                        packet,
                        self.scheduler.tp_group.rank,
                        self.scheduler.tp_cpu_group,
                        src=self.scheduler.tp_group.ranks[0],
                    )
                else:
                    packet = (
                        poll_now_ns,
                        (),
                        self._decode_poll_result_ready(encoded_result_ready),
                    )
            else:
                packet = broadcast_pyobj(
                    packet,
                    self.scheduler.tp_group.rank,
                    self.scheduler.tp_cpu_group,
                    src=self.scheduler.tp_group.ranks[0],
                )
        broadcast_done_ns = (
            time.monotonic_ns() if profile_poll_stages else overlap_query_done_ns
        )
        assert packet is not None
        poll_now_ns, events, result_ready = packet
        self._last_poll_ns = poll_now_ns
        self._overlap_result_ready = result_ready
        for kind, transfer_id, payload in events:
            state = self._parked.get(transfer_id)
            if state is None:
                continue
            if kind == "ready":
                if isinstance(payload, tuple) and len(payload) == 5:
                    (
                        candidate_bytes,
                        verify_len,
                        padding_priority,
                        bonus_token,
                        seq_len,
                    ) = payload
                    if (
                        state.bonus_token != bonus_token
                        or state.seq_len != seq_len
                    ):
                        raise RuntimeError(
                            "DT Target TP metadata diverged before Draft result: "
                            f"transfer_id={transfer_id}, "
                            f"local_bonus={state.bonus_token}, "
                            f"export_bonus={bonus_token}, "
                            f"local_seq_len={state.seq_len}, "
                            f"export_seq_len={seq_len}"
                        )
                    state.candidate_bytes = candidate_bytes
                    state.verify_len = verify_len
                    state.padding_priority = padding_priority
                elif isinstance(payload, tuple) and len(payload) == 4:
                    candidate_bytes, verify_len, bonus_token, seq_len = payload
                    if (
                        state.bonus_token != bonus_token
                        or state.seq_len != seq_len
                    ):
                        raise RuntimeError(
                            "DT Target TP metadata diverged before Draft result: "
                            f"transfer_id={transfer_id}, "
                            f"local_bonus={state.bonus_token}, "
                            f"export_bonus={bonus_token}, "
                            f"local_seq_len={state.seq_len}, "
                            f"export_seq_len={seq_len}"
                        )
                    state.candidate_bytes = candidate_bytes
                    state.verify_len = verify_len
                elif isinstance(payload, tuple):
                    state.candidate_bytes, state.verify_len = payload
                else:
                    state.candidate_bytes = payload
                    state.verify_len = getattr(self, "block_size", None)
                state.result_ready_ns = poll_now_ns
            else:
                state.failure = payload
        apply_done_ns = time.monotonic_ns()
        if profile_poll_stages:
            self._profile_poll_calls += 1
            self._profile_poll_empty_calls += int(not events)
            self._profile_poll_total_ns += apply_done_ns - poll_start_ns
            self._profile_poll_export_recv_ns += (
                export_recv_done_ns - poll_start_ns
            )
            self._profile_poll_overlap_query_ns += (
                overlap_query_done_ns - export_recv_done_ns
            )
            self._profile_poll_broadcast_ns += (
                broadcast_done_ns - overlap_query_done_ns
            )
            self._profile_poll_apply_ns += apply_done_ns - broadcast_done_ns
        if self.is_export_rank and events:
            profile_fields = {
                "instance_id": self.instance_id,
                "event_count": len(events),
                "ready_count": sum(kind == "ready" for kind, _, _ in events),
                "poll_ns": apply_done_ns - poll_start_ns,
                "parked_rounds": len(self._parked),
            }
            if profile_poll_stages:
                profile_fields.update(
                    poll_calls_since_event=self._profile_poll_calls,
                    empty_poll_calls_since_event=self._profile_poll_empty_calls,
                    poll_accumulated_ns=self._profile_poll_total_ns,
                    export_recv_accumulated_ns=(
                        self._profile_poll_export_recv_ns
                    ),
                    overlap_query_accumulated_ns=(
                        self._profile_poll_overlap_query_ns
                    ),
                    broadcast_accumulated_ns=self._profile_poll_broadcast_ns,
                    apply_accumulated_ns=self._profile_poll_apply_ns,
                )
                self._profile_poll_calls = 0
                self._profile_poll_empty_calls = 0
                self._profile_poll_total_ns = 0
                self._profile_poll_export_recv_ns = 0
                self._profile_poll_overlap_query_ns = 0
                self._profile_poll_broadcast_ns = 0
                self._profile_poll_apply_ns = 0
            profile_emit("target_poll_events", **profile_fields)

    @staticmethod
    def _query_overlap_result_ready(overlap_result) -> bool | None:
        if overlap_result is None:
            return None
        copy_done = getattr(overlap_result, "copy_done", None)
        # Generation overlap normally installs copy_done. Preserve generic
        # behavior for result types without a completion event rather than
        # deferring them until the hard deadline.
        return True if copy_done is None else bool(copy_done.query())

    @staticmethod
    def _encode_poll_result_ready(result_ready: bool | None) -> int:
        return -1 if result_ready is None else int(result_ready)

    @staticmethod
    def _decode_poll_result_ready(encoded: int) -> bool | None:
        if encoded == -1:
            return None
        if encoded not in (0, 1):
            raise ValueError(f"invalid encoded poll result readiness: {encoded}")
        return bool(encoded)

    def install_profile_device_timer(self) -> None:
        """Add GPU phase durations to DT traces without synchronizing CUDA."""
        if not self.is_export_rank:
            return
        timer = self.scheduler.metrics_reporter.forward_pass_device_timer
        if timer is None:
            return

        def report_target_gpu_phase(*, t: float, category: str, **_fields) -> None:
            profile_emit(
                "target_gpu_phase",
                instance_id=self.instance_id,
                phase=category,
                gpu_ns=round(t * 1_000_000_000),
            )

        def report_target_gpu_gap(
            *,
            t: float,
            previous_category: str,
            category: str,
            **_fields,
        ) -> None:
            profile_emit(
                "target_gpu_gap",
                instance_id=self.instance_id,
                previous_phase=previous_category,
                phase=category,
                gpu_gap_ns=round(t * 1_000_000_000),
            )

        timer.add_reporter(report_target_gpu_phase)
        timer.add_gap_reporter(report_target_gpu_gap)

    def _initialize_ready_metadata_slots(self) -> None:
        device = self.scheduler.device
        if not is_pin_memory_available(device):
            logger.warning(
                "SGLANG_DT_REUSE_READY_METADATA requested without pinned-memory "
                "support; using the allocation path"
            )
            self.reuse_ready_metadata = False
            return
        slot_count = int(os.environ.get("SGLANG_DT_READY_METADATA_SLOTS", "4"))
        if slot_count <= 0:
            raise ValueError("SGLANG_DT_READY_METADATA_SLOTS must be positive")
        max_rows = self.scheduler.max_running_requests
        for _ in range(slot_count):
            self._ready_metadata_slots.append(
                ReadyMetadataSlot(
                    req_pool_indices_cpu=torch.empty(
                        max_rows, dtype=torch.int64, pin_memory=True
                    ),
                    req_pool_indices=torch.empty(
                        max_rows, dtype=torch.int64, device=device
                    ),
                    seq_lens_cpu=torch.empty(
                        max_rows, dtype=torch.int64, pin_memory=True
                    ),
                    seq_lens=torch.empty(max_rows, dtype=torch.int64, device=device),
                    bonus_tokens_cpu=torch.empty(
                        max_rows, dtype=torch.int64, pin_memory=True
                    ),
                    bonus_tokens=torch.empty(
                        max_rows, dtype=torch.int64, device=device
                    ),
                    draft_tokens_cpu=torch.empty(
                        (max_rows, self.block_size),
                        dtype=torch.int32,
                        pin_memory=True,
                    ),
                    draft_tokens=torch.empty(
                        (max_rows, self.block_size),
                        dtype=torch.int64,
                        device=device,
                    ),
                    verify_lens_cpu=torch.empty(
                        max_rows,
                        dtype=torch.int32,
                        pin_memory=self.pinned_verify_metadata,
                    ),
                    padding_priorities_cpu=torch.empty(
                        max_rows, dtype=torch.int32, pin_memory=True
                    ),
                )
            )
        self._ready_metadata_free_slots.extend(range(slot_count))

    def _acquire_ready_metadata_slot(
        self, row_count: int
    ) -> tuple[int | None, ReadyMetadataSlot | None]:
        free_slots = getattr(self, "_ready_metadata_free_slots", ())
        if not getattr(self, "reuse_ready_metadata", False) or not free_slots:
            return None, None
        slot_id = free_slots.popleft()
        slot = self._ready_metadata_slots[slot_id]
        if row_count > slot.req_pool_indices_cpu.numel():
            free_slots.appendleft(slot_id)
            return None, None
        return slot_id, slot

    def _release_ready_metadata_slot(self, batch: ScheduleBatch) -> None:
        slot_ids = getattr(batch, "dt_ready_metadata_slots", ())
        if not slot_ids:
            return
        batch.dt_ready_metadata_slots = ()
        free_slots = self._ready_metadata_free_slots
        for slot_id in slot_ids:
            if slot_id not in free_slots:
                free_slots.append(slot_id)

    def _ready_batch_should_flush(self, ready: list[ParkedTargetRound]) -> bool:
        if len(ready) >= self.ready_batch_threshold:
            outstanding = len(self._parked) - len(ready)
            completion_slack = max(1, self.ready_batch_threshold // 8)
            # A nearly-complete wave is worth holding for the bounded window:
            # otherwise the last few Draft results arrive during the large
            # Target forward and force a separate singleton verify launch.
            if outstanding == 0 or outstanding > completion_slack:
                return True
        # A coalescing delay only has useful work to collect while enough
        # remote rounds remain outstanding. Flush the low-concurrency tail
        # immediately instead of paying the full window when no threshold-sized
        # verify batch can be formed.
        if len(self._parked) < self.ready_batch_threshold:
            return True
        if self.ready_batch_wait_ns == 0:
            return True
        ready_since_ns = min(
            (state.result_ready_ns for state in ready if state.result_ready_ns),
            default=0,
        )
        if ready_since_ns == 0:
            return True
        return self._last_poll_ns - ready_since_ns >= self.ready_batch_wait_ns

    def _select_full_wave_ready(
        self, ready: list[ParkedTargetRound]
    ) -> list[ParkedTargetRound]:
        members = getattr(self, "_full_wave_members", None)
        if members is None:
            members = self._full_wave_members = set()
        members.intersection_update(self._parked)
        if not members and self._parked:
            members.update(self._parked)
        if not members:
            return []

        ready_by_transfer = {state.transfer_id: state for state in ready}
        if any(transfer_id not in ready_by_transfer for transfer_id in members):
            return []
        return [
            state
            for transfer_id, state in self._parked.items()
            if transfer_id in members
        ]

    def _select_pipeline_lane_ready(
        self, ready: list[ParkedTargetRound]
    ) -> list[ParkedTargetRound]:
        lane_count = self.pipeline_lanes
        ready_by_lane: dict[int, list[ParkedTargetRound]] = {
            lane_id: [] for lane_id in range(lane_count)
        }
        parked_count_by_lane = [0] * lane_count
        for state in self._parked.values():
            parked_count_by_lane[state.lane_id] += 1
        for state in ready:
            ready_by_lane[state.lane_id].append(state)

        # Include rows currently executing on Target, not just remote parked
        # rows.  After a partial lane launch those rows remain live while a
        # laggard may become ready.  Treating the laggard as a complete
        # one-row lane would enqueue a singleton behind the large forward and
        # destroy the phase alignment we are trying to recover.
        live_count_by_lane = Counter(getattr(self, "_lane_by_request", {}).values())
        total_live_count = sum(live_count_by_lane.values())
        if total_live_count == 0:
            total_live_count = sum(parked_count_by_lane)
        collapse_threshold = getattr(self, "pipeline_lane_collapse_threshold", 0)
        if collapse_threshold > 0 and 0 < total_live_count <= collapse_threshold:
            if len(ready) != total_live_count:
                return []
            # At drain-tail concurrency, keeping two independently shrinking
            # lanes halves the useful Target batch without providing enough
            # remote work to hide another large forward.  Once every live row
            # is remotely ready, reunite them for one verify wave.  Persistent
            # assignments are retained, so the normal two-lane policy resumes
            # automatically if replacement admissions raise concurrency.
            self._lane_refill_started_ns = [0] * lane_count
            self._selected_pipeline_lane = -2
            self._pipeline_lanes_collapsed = True
            return ready

        tail_merge_threshold = getattr(self, "pipeline_lane_tail_merge_threshold", 0)
        tail_merge_armed = getattr(self, "_pipeline_lane_tail_merge_armed", False)
        if tail_merge_threshold > 0:
            if total_live_count > tail_merge_threshold:
                # A low live-row count also occurs while a benchmark is
                # ramping up.  Arm only after observing the steady-state high
                # watermark so startup cannot be mistaken for a drain tail.
                tail_merge_armed = True
            elif total_live_count == 0:
                tail_merge_armed = False
            self._pipeline_lane_tail_merge_armed = tail_merge_armed
        if (
            tail_merge_threshold > 0
            and tail_merge_armed
            and 0 < total_live_count <= tail_merge_threshold
        ):
            complete_lane_ids = [
                lane_id
                for lane_id in range(lane_count)
                if ready_by_lane[lane_id]
                and len(ready_by_lane[lane_id]) == live_count_by_lane[lane_id]
            ]
            if complete_lane_ids:
                # A full collapse idles Target until every lane returns.  At
                # the drain tail, instead preserve the earliest complete-lane
                # launch and opportunistically absorb ready rows from the
                # other lane.  The laggards remain parked and naturally join a
                # later wave, so this never waits longer than strict lanes.
                selected_lane = min(
                    complete_lane_ids,
                    key=lambda lane_id: (
                        (lane_id - self._lane_ready_cursor) % lane_count
                    ),
                )
                self._lane_refill_started_ns = [0] * lane_count
                self._lane_ready_cursor = (selected_lane + 1) % lane_count
                self._selected_pipeline_lane = -3
                self._pipeline_lanes_tail_merged = True
                return ready

        for offset in range(lane_count):
            lane_id = (self._lane_ready_cursor + offset) % lane_count
            lane_ready = ready_by_lane[lane_id]
            parked_count = parked_count_by_lane[lane_id]
            live_count = max(parked_count, live_count_by_lane[lane_id])
            if not lane_ready:
                self._lane_refill_started_ns[lane_id] = 0
                continue

            target_size = max(self._lane_target_sizes[lane_id], live_count)
            if live_count < target_size:
                if self.full_wave_refill_wait_ns:
                    refill_started_ns = self._lane_refill_started_ns[lane_id]
                    if refill_started_ns == 0:
                        self._lane_refill_started_ns[lane_id] = self._last_poll_ns
                        continue
                    if (
                        self._last_poll_ns - refill_started_ns
                        < self.full_wave_refill_wait_ns
                    ):
                        continue
                self._lane_target_sizes[lane_id] = live_count
                target_size = live_count

            lagging_rows = max(0, live_count - len(lane_ready))
            lane_slack = getattr(self, "pipeline_lane_slack", 0)
            if getattr(self, "_overlap_result_ready", None) is True:
                lane_slack = max(
                    lane_slack,
                    getattr(self, "pipeline_idle_slack", 0),
                )
            if lagging_rows > lane_slack or len(lane_ready) < max(
                1, target_size - lane_slack
            ):
                continue

            self._lane_refill_started_ns[lane_id] = 0
            self._lane_ready_cursor = (lane_id + 1) % lane_count
            self._selected_pipeline_lane = lane_id
            self._selected_pipeline_lane_lagging_rows = lagging_rows
            self._selected_pipeline_lane_slack_used = lane_slack
            fill_threshold = getattr(
                self, "pipeline_cross_lane_fill_threshold", 0
            )
            if fill_threshold <= len(lane_ready):
                return lane_ready

            donor_rows = sorted(
                (
                    state
                    for donor_lane, donor_ready in ready_by_lane.items()
                    if donor_lane != lane_id
                    for state in donor_ready
                ),
                key=lambda state: (state.result_ready_ns, state.transfer_id),
            )
            borrowed = donor_rows[: fill_threshold - len(lane_ready)]
            if lagging_rows and borrowed:
                # Repair a nearly-complete lane without changing persistent
                # lane sizes.  Moving only the ready donor rows makes the
                # selected lane grow and the donor shrink on every straggler,
                # eventually recreating the asymmetric-lane throughput loss.
                # Swap an equal number of selected-lane laggards back to the
                # corresponding donor lanes so this launch is full while the
                # next waves remain balanced.
                borrowed = borrowed[:lagging_rows]
                ready_rids = {state.req.rid for state in lane_ready}
                parked_laggards = {
                    state.req.rid: state
                    for state in self._parked.values()
                    if state.lane_id == lane_id and state.candidate_bytes is None
                }
                # ``live_count`` includes requests whose previous Target forward
                # has not reached ``park_completed_batch`` yet.  Such a row has
                # persistent lane ownership but no ParkedTargetRound to mutate.
                # Swap its ownership in ``_lane_by_request`` now; when the
                # in-flight result parks, _lane_for_request observes the new
                # donor lane.  This keeps lane sizes balanced without assuming
                # every laggard is already in the remote queue.
                laggard_rids = sorted(
                    (
                        rid
                        for rid, assigned_lane in self._lane_by_request.items()
                        if assigned_lane == lane_id
                        and rid not in ready_rids
                    ),
                    key=lambda rid: (
                        0 if rid in parked_laggards else 1,
                        getattr(parked_laggards.get(rid), "parked_ns", 0),
                        rid,
                    ),
                )[: len(borrowed)]
                # A stale out-of-tree/test mapping must not turn an otherwise
                # valid partial-lane launch into a process-wide crash.  Borrow
                # only rows for which an equal ownership swap is possible.
                borrowed = borrowed[: len(laggard_rids)]
                for donor, laggard_rid in zip(
                    borrowed, laggard_rids, strict=True
                ):
                    donor_lane = donor.lane_id
                    donor.lane_id = lane_id
                    laggard = parked_laggards.get(laggard_rid)
                    if laggard is not None:
                        laggard.lane_id = donor_lane
                    self._lane_by_request[donor.req.rid] = lane_id
                    self._lane_by_request[laggard_rid] = donor_lane
            else:
                for state in borrowed:
                    state.lane_id = lane_id
                    self._lane_by_request[state.req.rid] = lane_id
            self._selected_pipeline_cross_lane_rows = len(borrowed)
            return lane_ready + borrowed

        pooled_min_rows = getattr(self, "pipeline_pooled_ready_min_rows", 0)
        pooled_idle_only = getattr(self, "pipeline_pooled_ready_idle_only", False)
        pooled_can_launch = not pooled_idle_only or (
            getattr(self, "_overlap_result_ready", None) is True
        )
        if (
            pooled_min_rows > 0
            and len(ready) >= pooled_min_rows
            and pooled_can_launch
        ):
            # Both persistent lanes are incomplete, but together they already
            # contain enough ready work for a full (or nearly full) graph.  A
            # strict lane boundary would idle Target until one lane's last
            # straggler returns.  Pool only as a fallback after the normal
            # lane-first checks above, and keep persistent ownership unchanged
            # so subsequent Draft waves retain their established balance.
            pooled_size = min(len(ready), self.ready_batch_threshold)
            pooled = sorted(
                ready,
                key=lambda state: (
                    state.result_ready_ns,
                    state.transfer_id,
                ),
            )[:pooled_size]
            self._lane_refill_started_ns = [0] * lane_count
            self._selected_pipeline_lane = -4
            self._selected_pipeline_cross_lane_rows = sum(
                1 for state in pooled if state.lane_id != pooled[0].lane_id
            )
            self._pipeline_lanes_pooled = True
            return pooled
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
        target_inflight_rids = set(
            getattr(self, "_target_inflight_counts", {}).keys()
        )
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
            if getattr(self, "is_export_rank", False):
                profile_emit(
                    "target_stale_repark",
                    instance_id=self.instance_id,
                    batch_size=len(batch.reqs),
                    parked_rounds=len(self._parked),
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
        target_inflight_counts = getattr(self, "_target_inflight_counts", None)
        if target_inflight_counts is None:
            # Preserve lightweight construction through ``__new__`` in unit
            # tests and out-of-tree integrations.
            target_inflight_counts = self._target_inflight_counts = Counter()
        for req in launched_batch.reqs:
            target_inflight_counts[req.rid] += 1
            target_inflight_reqs = getattr(self, "_target_inflight_reqs", None)
            if target_inflight_reqs is None:
                target_inflight_reqs = self._target_inflight_reqs = {}
            target_inflight_reqs[req.rid] = req

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

        if self.is_export_rank:
            target_inflight_counts = getattr(
                self, "_target_inflight_counts", Counter()
            )
            profile_emit(
                "target_overlap_inflight_detach",
                instance_id=self.instance_id,
                batch_size=len(launched_req_ids),
                running_after=running_batch.batch_size(),
                inflight_requests=len(target_inflight_counts),
                inflight_forwards=sum(target_inflight_counts.values()),
            )

    def _target_inflight_count(self, rid: str) -> int:
        return int(getattr(self, "_target_inflight_counts", {}).get(rid, 0))

    def _release_target_inflight_batch(self, batch: ScheduleBatch) -> None:
        """Consume one launched Target-forward ownership per result row.

        A request may occur in multiple queued prefill chunks, hence the
        reference count.  Missing entries are expected for non-overlap mode
        and for synthetic batches used by tests.
        """
        target_inflight_counts = getattr(self, "_target_inflight_counts", None)
        if not target_inflight_counts:
            return
        for req in batch.reqs:
            count = target_inflight_counts.get(req.rid, 0)
            if count <= 0:
                continue
            if count == 1:
                del target_inflight_counts[req.rid]
                getattr(self, "_target_inflight_reqs", {}).pop(req.rid, None)
            else:
                target_inflight_counts[req.rid] = count - 1

    def has_target_inflight(self) -> bool:
        return bool(getattr(self, "_target_inflight_counts", None))

    def pop_retractable_for_oom(self) -> list[Req]:
        """Cancel remote waits and detach synchronized Target snapshots."""

        states = list(self._parked.values())
        for state in states:
            self._parked.pop(state.transfer_id, None)
            full_wave_members = getattr(self, "_full_wave_members", None)
            if full_wave_members is not None:
                full_wave_members.discard(state.transfer_id)
            self._forget_request_lane(state.req)
            getattr(self, "_next_round", {}).pop(state.req.rid, None)

        inflight_reqs = list(
            getattr(self, "_target_inflight_reqs", {}).values()
        )
        getattr(self, "_target_inflight_counts", {}).clear()
        getattr(self, "_target_inflight_reqs", {}).clear()

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
            cancellation = CancelTransfers(
                protocol_version=PROTOCOL_VERSION,
                target_instance_id=self.instance_id,
                transfer_ids=transfer_ids,
                reason="kv_retract",
            )
            self._reliable_outbox().add(
                CANCEL_TRANSFERS,
                cancel_transfers_id(cancellation),
                cancellation,
            )
            self._flush_reliable_outbox()

        unique: dict[int, Req] = {}
        for req in [*(state.req for state in states), *inflight_reqs]:
            unique[id(req)] = req
        return list(unique.values())

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
        reason = "defer_disabled"
        should_defer = False
        elapsed_ns = max(0, self._last_poll_ns - self._overlap_launch_poll_ns)
        if self.overlap_result_defer_ns <= 0:
            reason = "defer_disabled"
        elif next_batch is not None:
            reason = "next_batch_ready"
        elif last_batch is None:
            reason = "no_last_batch"
        elif not self._parked:
            reason = "no_remote_round"
        elif self._overlap_result_ready:
            # The previous Target forward and its async result copy have
            # completed. Process it immediately so the next remote Draft round
            # starts now; waiting longer cannot create useful GPU overlap.
            reason = "result_ready"
        elif elapsed_ns >= self.overlap_result_defer_ns:
            reason = "deadline"
        else:
            reason = "remote_round_pending"
            should_defer = True

        if should_defer and not self._overlap_defer_active:
            self._overlap_defer_active = True
            if self.is_export_rank:
                profile_emit(
                    "target_overlap_result_defer_start",
                    instance_id=self.instance_id,
                    elapsed_ns=elapsed_ns,
                    parked_rounds=len(self._parked),
                )
        elif not should_defer and self._overlap_defer_active:
            self._overlap_defer_active = False
            if self.is_export_rank:
                profile_emit(
                    "target_overlap_result_defer_end",
                    instance_id=self.instance_id,
                    elapsed_ns=elapsed_ns,
                    parked_rounds=len(self._parked),
                    reason=reason,
                )
        return should_defer

    def build_ready_batch(self) -> ScheduleBatch | None:
        build_start_ns = time.monotonic_ns()
        profile_build_stages = bool(os.environ.get("DT_PROFILE_DIR")) and bool(
            getattr(self, "is_export_rank", False)
        )
        ready = [
            state
            for state in self._parked.values()
            if state.candidate_bytes is not None
        ]
        self._selected_pipeline_lane = -1
        self._selected_pipeline_lane_lagging_rows = 0
        self._selected_pipeline_lane_slack_used = 0
        self._selected_pipeline_cross_lane_rows = 0
        self._pipeline_lanes_collapsed = False
        self._pipeline_lanes_tail_merged = False
        self._pipeline_lanes_pooled = False
        if (
            ready
            and getattr(self, "full_wave", False)
            and getattr(self, "pipeline_lanes", 1) > 1
        ):
            ready = self._select_pipeline_lane_ready(ready)
        elif ready and getattr(self, "full_wave", False):
            ready = self._select_full_wave_ready(ready)
        elif ready and not self._ready_batch_should_flush(ready):
            ready = []
        if not ready:
            return None
        select_end_ns = (
            time.monotonic_ns() if profile_build_stages else build_start_ns
        )
        remaining_by_cohort = Counter(
            state.cohort_id for state in self._parked.values()
        )
        ready_by_cohort = Counter(state.cohort_id for state in ready)
        original_by_cohort = {
            state.cohort_id: state.cohort_size for state in self._parked.values()
        }
        whole_cohort_ids = {
            cohort_id
            for cohort_id, ready_count in ready_by_cohort.items()
            if ready_count == remaining_by_cohort[cohort_id]
            and remaining_by_cohort[cohort_id] == original_by_cohort[cohort_id]
        }
        partial_cohort_ids = {
            cohort_id
            for cohort_id, ready_count in ready_by_cohort.items()
            if ready_count < remaining_by_cohort[cohort_id]
        }
        residual_cohort_ids = {
            cohort_id
            for cohort_id in ready_by_cohort
            if remaining_by_cohort[cohort_id] < original_by_cohort[cohort_id]
        }
        for state in ready:
            self._parked.pop(state.transfer_id)
            full_wave_members = getattr(self, "_full_wave_members", None)
            if full_wave_members is not None:
                full_wave_members.discard(state.transfer_id)
        cohort_end_ns = (
            time.monotonic_ns() if profile_build_stages else select_end_ns
        )

        reqs = [state.req for state in ready]
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
        batch_init_end_ns = (
            time.monotonic_ns() if profile_build_stages else cohort_end_ns
        )
        req_pool_indices = [req.req_pool_idx for req in reqs]
        metadata_slot_id, metadata_slot = self._acquire_ready_metadata_slot(
            len(ready)
        )
        batch.dt_ready_metadata_slots = (
            (metadata_slot_id,) if metadata_slot_id is not None else ()
        )
        ready_metadata_source = "slot" if metadata_slot is not None else "allocate"
        req_list_end_ns = (
            time.monotonic_ns() if profile_build_stages else batch_init_end_ns
        )
        req_cpu_ns = 0
        req_device_ns = 0
        if metadata_slot is None:
            batch.req_pool_indices_cpu = torch.tensor(
                req_pool_indices, dtype=torch.int64
            )
            batch.req_pool_indices = batch.req_pool_indices_cpu.to(device=device)
        else:
            batch.req_pool_indices_cpu = metadata_slot.req_pool_indices_cpu[
                : len(ready)
            ]
            batch.req_pool_indices_cpu.numpy()[:] = req_pool_indices
            batch.req_pool_indices = metadata_slot.req_pool_indices[: len(ready)]
            batch.req_pool_indices.copy_(
                batch.req_pool_indices_cpu, non_blocking=True
            )
        req_indices_end_ns = (
            time.monotonic_ns() if profile_build_stages else batch_init_end_ns
        )
        seq_lens_values = [state.seq_len for state in ready]
        if metadata_slot is None:
            seq_lens_cpu = torch.tensor(seq_lens_values, dtype=torch.int64)
        else:
            seq_lens_cpu = metadata_slot.seq_lens_cpu[: len(ready)]
            seq_lens_cpu.numpy()[:] = seq_lens_values
        batch.seq_lens_cpu = seq_lens_cpu
        if metadata_slot is None:
            batch.seq_lens = seq_lens_cpu.to(device)
        else:
            batch.seq_lens = metadata_slot.seq_lens[: len(ready)]
            batch.seq_lens.copy_(seq_lens_cpu, non_blocking=True)
        batch.orig_seq_lens = batch.seq_lens.to(dtype=torch.int32)
        batch.seq_lens_sum = int(seq_lens_cpu.sum())
        seq_meta_end_ns = (
            time.monotonic_ns() if profile_build_stages else req_indices_end_ns
        )
        bonus_token_values = [state.bonus_token for state in ready]
        if metadata_slot is None:
            bonus_tokens = torch.tensor(
                bonus_token_values,
                dtype=torch.int64,
                device=device,
            )
        else:
            bonus_tokens_cpu = metadata_slot.bonus_tokens_cpu[: len(ready)]
            bonus_tokens_cpu.numpy()[:] = bonus_token_values
            bonus_tokens = metadata_slot.bonus_tokens[: len(ready)]
            bonus_tokens.copy_(bonus_tokens_cpu, non_blocking=True)
        bonus_end_ns = (
            time.monotonic_ns() if profile_build_stages else seq_meta_end_ns
        )
        row_bytes = self.block_size * 4
        candidate_blob = b"".join(state.candidate_bytes for state in ready)
        if len(candidate_blob) != len(ready) * row_bytes:
            raise ValueError("DT candidate payload width does not match block size")
        if metadata_slot is None:
            draft_tokens = torch.frombuffer(
                bytearray(candidate_blob), dtype=torch.int32
            ).reshape(len(ready), self.block_size)
        else:
            draft_tokens = metadata_slot.draft_tokens_cpu[: len(ready)]
            draft_tokens.numpy()[:] = np.frombuffer(
                candidate_blob, dtype=np.int32
            ).reshape(len(ready), self.block_size)
        candidate_host_end_ns = (
            time.monotonic_ns() if profile_build_stages else bonus_end_ns
        )
        if metadata_slot is None:
            draft_tokens = draft_tokens.to(device=device, dtype=torch.int64)
        else:
            draft_tokens_device = metadata_slot.draft_tokens[: len(ready)]
            draft_tokens_device.copy_(draft_tokens, non_blocking=True)
            draft_tokens = draft_tokens_device
        candidate_device_end_ns = (
            time.monotonic_ns() if profile_build_stages else candidate_host_end_ns
        )
        verify_lens_cpu = None
        padding_priorities_cpu = None
        if (
            getattr(self, "ragged_verify", False)
            or getattr(self, "adaptive_verify_margin", None) is not None
        ):
            min_adaptive_batch = getattr(
                self, "adaptive_verify_min_target_batch", 1
            )
            if (
                getattr(self, "adaptive_verify_margin", None) is None
                or len(ready) < min_adaptive_batch
            ):
                verify_lens = [self.block_size] * len(ready)
            else:
                verify_lens = [
                    state.verify_len
                    if state.verify_len is not None
                    else self.block_size
                    for state in ready
                ]
            if metadata_slot is None:
                verify_lens_cpu = torch.tensor(
                    verify_lens,
                    dtype=torch.int32,
                    pin_memory=(
                        getattr(
                            self,
                            "pinned_verify_metadata",
                            is_pin_memory_available(device),
                        )
                        and is_pin_memory_available(device)
                    ),
                )
            else:
                verify_lens_cpu = metadata_slot.verify_lens_cpu[: len(ready)]
                verify_lens_cpu.numpy()[:] = verify_lens
            if getattr(self, "adaptive_padding_priority", False):
                padding_priorities = []
                for state in ready:
                    if state.padding_priority is None:
                        raise RuntimeError(
                            "DT adaptive padding priority is missing from a "
                            "ready result row"
                        )
                    padding_priorities.append(state.padding_priority)
                if metadata_slot is None:
                    padding_priorities_cpu = torch.tensor(
                        padding_priorities,
                        dtype=torch.int32,
                        pin_memory=is_pin_memory_available(device),
                    )
                else:
                    padding_priorities_cpu = (
                        metadata_slot.padding_priorities_cpu[: len(ready)]
                    )
                    padding_priorities_cpu.numpy()[:] = padding_priorities
        adaptive_end_ns = (
            time.monotonic_ns() if profile_build_stages else candidate_device_end_ns
        )
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
            verify_lens_cpu=verify_lens_cpu,
            padding_priorities_cpu=padding_priorities_cpu,
        )
        batch.input_ids = None
        spec_alloc_end_ns = (
            time.monotonic_ns() if profile_build_stages else adaptive_end_ns
        )
        self.scheduler.future_map.stash(
            batch.req_pool_indices,
            RelayPayload(
                bonus_tokens=bonus_tokens,
                # FutureMap keeps a hidden-state relay for FLASHMTP even
                # though the disaggregated verify consumes remote candidates.
                # Seed its per-request rows so overlap resolve can preserve
                # the normal spec-v2 buffer invariants.
                hidden_states=batch.spec_info.hidden_states,
            ),
        )
        future_stash_end_ns = (
            time.monotonic_ns() if profile_build_stages else spec_alloc_end_ns
        )
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch, self.scheduler.model_config.vocab_size
        )
        sampling_end_ns = time.monotonic_ns()
        if getattr(self, "is_export_rank", False):
            waits = [build_start_ns - state.parked_ns for state in ready]
            ready_to_build = [
                build_start_ns - state.result_ready_ns
                for state in ready
                if state.result_ready_ns
            ]
            profile_emit(
                "target_ready_batch_build",
                instance_id=self.instance_id,
                batch_size=len(ready),
                build_ns=sampling_end_ns - build_start_ns,
                select_ns=select_end_ns - build_start_ns,
                cohort_ns=cohort_end_ns - select_end_ns,
                batch_init_ns=batch_init_end_ns - cohort_end_ns,
                req_list_ns=req_list_end_ns - batch_init_end_ns,
                req_cpu_ns=req_cpu_ns,
                req_device_ns=req_device_ns,
                ready_metadata_source=ready_metadata_source,
                req_indices_ns=req_indices_end_ns - batch_init_end_ns,
                seq_meta_ns=seq_meta_end_ns - req_indices_end_ns,
                bonus_ns=bonus_end_ns - seq_meta_end_ns,
                candidate_host_ns=candidate_host_end_ns - bonus_end_ns,
                candidate_device_ns=(
                    candidate_device_end_ns - candidate_host_end_ns
                ),
                adaptive_host_ns=adaptive_end_ns - candidate_device_end_ns,
                spec_alloc_ns=spec_alloc_end_ns - adaptive_end_ns,
                future_stash_ns=future_stash_end_ns - spec_alloc_end_ns,
                sampling_info_ns=sampling_end_ns - future_stash_end_ns,
                round_wait_mean_ns=sum(waits) // len(waits),
                round_wait_max_ns=max(waits),
                result_to_build_mean_ns=(
                    sum(ready_to_build) // len(ready_to_build) if ready_to_build else 0
                ),
                result_to_build_min_ns=min(ready_to_build, default=0),
                result_to_build_max_ns=max(ready_to_build, default=0),
                cohort_count=len(ready_by_cohort),
                whole_cohort_count=len(whole_cohort_ids),
                whole_cohort_rows=sum(
                    ready_by_cohort[cohort_id] for cohort_id in whole_cohort_ids
                ),
                partial_cohort_count=len(partial_cohort_ids),
                partial_cohort_rows=sum(
                    ready_by_cohort[cohort_id] for cohort_id in partial_cohort_ids
                ),
                partial_outstanding_rows=sum(
                    remaining_by_cohort[cohort_id] - ready_by_cohort[cohort_id]
                    for cohort_id in partial_cohort_ids
                ),
                residual_cohort_count=len(residual_cohort_ids),
                residual_cohort_rows=sum(
                    ready_by_cohort[cohort_id] for cohort_id in residual_cohort_ids
                ),
                parked_rounds=len(self._parked),
                full_wave=getattr(self, "full_wave", False),
                full_wave_target_size=getattr(self, "_full_wave_target_size", 0),
                full_wave_members=len(getattr(self, "_full_wave_members", ())),
                deferred_wave_rows=len(self._parked),
                pipeline_lanes=getattr(self, "pipeline_lanes", 1),
                pipeline_lane=self._selected_pipeline_lane,
                pipeline_lane_lagging_rows=getattr(
                    self, "_selected_pipeline_lane_lagging_rows", 0
                ),
                pipeline_lane_slack_used=getattr(
                    self, "_selected_pipeline_lane_slack_used", 0
                ),
                pipeline_cross_lane_rows=getattr(
                    self, "_selected_pipeline_cross_lane_rows", 0
                ),
                pipeline_lanes_collapsed=getattr(
                    self, "_pipeline_lanes_collapsed", False
                ),
                pipeline_lanes_tail_merged=getattr(
                    self, "_pipeline_lanes_tail_merged", False
                ),
                pipeline_lane_tail_merge_armed=getattr(
                    self, "_pipeline_lane_tail_merge_armed", False
                ),
                pipeline_lanes_pooled=getattr(
                    self, "_pipeline_lanes_pooled", False
                ),
                pipeline_pooled_ready_min_rows=getattr(
                    self, "pipeline_pooled_ready_min_rows", 0
                ),
                pipeline_pooled_ready_idle_only=getattr(
                    self, "pipeline_pooled_ready_idle_only", False
                ),
                pipeline_lane_weights=getattr(self, "_pipeline_lane_weights", (1,)),
            )
        return batch

    def record_scheduler_plan(
        self,
        *,
        ready_batch_size: int,
        prefill_batch_size: int,
        running_batch_size: int,
        waiting_queue_size: int,
        selected_mode: str,
        selected_batch_size: int,
    ) -> None:
        if not self.is_export_rank:
            return
        if (
            ready_batch_size == 0
            and prefill_batch_size == 0
            and selected_batch_size == 0
        ):
            return
        profile_emit(
            "target_scheduler_plan",
            instance_id=self.instance_id,
            ready_batch_size=ready_batch_size,
            prefill_batch_size=prefill_batch_size,
            running_batch_size=running_batch_size,
            waiting_queue_size=waiting_queue_size,
            parked_rounds=len(self._parked),
            selected_mode=selected_mode,
            selected_batch_size=selected_batch_size,
        )

    def should_defer_prefill(
        self, *, waiting_queue_size: int, ready_batch_size: int
    ) -> bool:
        # Once a frozen wave is ready, run it before admitting replacement
        # prefills.  Otherwise get_new_batch_prefill wins scheduler priority,
        # the replacement becomes the next frozen wave by itself, and the
        # already-built large verify batch remains stranded in running_batch.
        if ready_batch_size > 0:
            self._prefill_wait_started_ns = 0
            return True

        active_verify_rows = len(self._parked) + ready_batch_size
        flush_on_gpu_idle = getattr(self, "prefill_flush_on_gpu_idle", False) and (
            getattr(self, "_overlap_result_ready", None) is True
        )
        launch_age_ns = max(
            0,
            self._last_poll_ns - getattr(self, "_overlap_launch_poll_ns", 0),
        )
        flush_launch_age_ns = getattr(self, "prefill_flush_launch_age_ns", 0)
        flush_on_launch_age = (
            flush_launch_age_ns > 0
            and getattr(self, "_overlap_launch_poll_ns", 0) > 0
            and launch_age_ns >= flush_launch_age_ns
        )
        if (
            waiting_queue_size <= 0
            or waiting_queue_size >= self.prefill_batch_threshold
            or active_verify_rows == 0
            or self.prefill_batch_wait_ns == 0
            or flush_on_gpu_idle
            or flush_on_launch_age
        ):
            if self._prefill_wait_started_ns and self.is_export_rank:
                profile_emit(
                    "target_prefill_defer_end",
                    instance_id=self.instance_id,
                    waiting_queue_size=waiting_queue_size,
                    active_verify_rows=active_verify_rows,
                    waited_ns=self._last_poll_ns - self._prefill_wait_started_ns,
                    reason=(
                        "gpu_idle"
                        if flush_on_gpu_idle
                        else "launch_age"
                        if flush_on_launch_age
                        else "threshold_or_no_active_work"
                    ),
                    launch_age_ns=launch_age_ns,
                )
            self._prefill_wait_started_ns = 0
            return False

        if self._prefill_wait_started_ns == 0:
            self._prefill_wait_started_ns = self._last_poll_ns
            if self.is_export_rank:
                profile_emit(
                    "target_prefill_defer_start",
                    instance_id=self.instance_id,
                    waiting_queue_size=waiting_queue_size,
                    active_verify_rows=active_verify_rows,
                    flush_on_gpu_idle=getattr(
                        self, "prefill_flush_on_gpu_idle", False
                    ),
                    flush_launch_age_ns=flush_launch_age_ns,
                    launch_age_ns=launch_age_ns,
                )
            return True

        if (
            self._last_poll_ns - self._prefill_wait_started_ns
            < self.prefill_batch_wait_ns
        ):
            return True

        if self.is_export_rank:
            profile_emit(
                "target_prefill_defer_end",
                instance_id=self.instance_id,
                waiting_queue_size=waiting_queue_size,
                active_verify_rows=active_verify_rows,
                waited_ns=self._last_poll_ns - self._prefill_wait_started_ns,
                reason="deadline",
                launch_age_ns=launch_age_ns,
            )
        self._prefill_wait_started_ns = 0
        return False

    def pop_failures(self) -> list[ParkedTargetRound]:
        failed = [state for state in self._parked.values() if state.failure]
        for state in failed:
            self._parked.pop(state.transfer_id)
            full_wave_members = getattr(self, "_full_wave_members", None)
            if full_wave_members is not None:
                full_wave_members.discard(state.transfer_id)
            self._forget_request_lane(state.req)
            getattr(self, "_next_round", {}).pop(state.req.rid, None)
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
            for state in self._parked.values()
            if abort_all or state.req.rid.startswith(rid_prefix)
        ]
        if not matched:
            return []
        for state in matched:
            self._parked.pop(state.transfer_id, None)
            full_wave_members = getattr(self, "_full_wave_members", None)
            if full_wave_members is not None:
                full_wave_members.discard(state.transfer_id)
            self._forget_request_lane(state.req)
            getattr(self, "_next_round", {}).pop(state.req.rid, None)
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
            self._reliable_outbox().add(
                CANCEL_TRANSFERS,
                cancel_transfers_id(cancellation),
                cancellation,
            )
            self._flush_reliable_outbox()
        return [state.req for state in matched]

    def __len__(self) -> int:
        return len(self._parked)

    def _lane_for_request(self, req: Req) -> int:
        if getattr(self, "pipeline_lanes", 1) <= 1:
            return 0
        existing = self._lane_by_request.get(req.rid)
        if existing is not None:
            return existing

        live_counts = Counter(self._lane_by_request.values())
        lane_weights = getattr(
            self, "_pipeline_lane_weights", (1,) * self.pipeline_lanes
        )
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
            # Before target sizes are learned (and after all refill gaps are
            # closed), weighted normalized load creates persistent asymmetric
            # lanes without hard-coding a concurrency-specific batch size.
            lane_id = min(
                range(self.pipeline_lanes),
                key=lambda candidate: (
                    live_counts[candidate] / lane_weights[candidate],
                    (candidate - self._lane_assignment_cursor) % self.pipeline_lanes,
                ),
            )
        self._lane_by_request[req.rid] = lane_id
        self._lane_assignment_cursor = (lane_id + 1) % self.pipeline_lanes
        return lane_id

    def _forget_request_lane(self, req: Req) -> None:
        lane_by_request = getattr(self, "_lane_by_request", None)
        if lane_by_request is not None:
            lane_by_request.pop(req.rid, None)
            if not lane_by_request:
                # A completed workload must not carry its drain state into the
                # startup ramp of the next workload on the same server.
                self._pipeline_lane_tail_merge_armed = False

    def close(self) -> None:
        for store in self._result_stores.values():
            store.close()
        self._result_stores.clear()
        if self._client is not None:
            self._client.close()
        if self._source_store is not None:
            self._close_direct_pinned_export()
            self._source_store.close()
            self._source_store.unlink()

    def _initialize_direct_pinned_export(self) -> None:
        if self._source_store is None:
            raise RuntimeError("DT source store must exist before CUDA registration")
        mapping_view = self._source_store.writable_mapping_view()
        try:
            address = ctypes.addressof(ctypes.c_char.from_buffer(mapping_view))
            cudart = torch.cuda.cudart()
            result = cudart.cudaHostRegister(address, len(mapping_view), 0)
            if result != cudart.cudaError.success:
                raise RuntimeError(
                    f"cudaHostRegister failed for the DT source-slot mmap: {result}"
                )
            self._direct_pivot_mapping_view = mapping_view
            self._direct_pivot_mapping_address = address
            self._direct_pivot_registered = True
            for slot in range(self._source_store.slot_count):
                slot_view = self._source_store.writable_payload_view(slot)
                self._direct_pivot_slot_views.append(slot_view)
                self._direct_pivot_slot_tensors.append(
                    torch.frombuffer(
                        slot_view,
                        dtype=torch.bfloat16,
                        count=self.pivot_width,
                    )
                )
        except Exception:
            if self._direct_pivot_registered:
                self._close_direct_pinned_export()
            else:
                mapping_view.release()
            raise

    def _close_direct_pinned_export(self) -> None:
        if getattr(self, "_direct_pivot_registered", False):
            result = torch.cuda.cudart().cudaHostUnregister(
                self._direct_pivot_mapping_address
            )
            if result != torch.cuda.cudart().cudaError.success:
                logger.warning(
                    "cudaHostUnregister failed for DT source mmap: %s", result
                )
            self._direct_pivot_registered = False
        getattr(self, "_direct_pivot_slot_tensors", []).clear()
        for slot_view in getattr(self, "_direct_pivot_slot_views", []):
            slot_view.release()
        getattr(self, "_direct_pivot_slot_views", []).clear()
        if getattr(self, "_direct_pivot_mapping_view", None) is not None:
            self._direct_pivot_mapping_view.release()
            self._direct_pivot_mapping_view = None
        self._direct_pivot_mapping_address = None

    def _make_task(
        self,
        *,
        transfer_id: str,
        req: Req,
        round_id: int,
        attempt: int,
        bonus_token: int,
        source_slot: int,
        source_generation: int,
        pivot_nbytes: int,
    ):
        from dt_protocol import DraftTask, SourceRef

        expected_bytes = self.pivot_width * 2
        if pivot_nbytes != expected_bytes:
            raise ValueError(
                f"DT Target expected {expected_bytes} pivot bytes, got {pivot_nbytes}"
            )
        now_ns = time.monotonic_ns()
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
        self._pending_send.append(batch)

    def _flush_pending_send(self) -> None:
        self._flush_reliable_outbox()
        while self._pending_send:
            batch = self._pending_send[0]
            if not self._client.submit(batch):
                break
            submitted_ns = time.monotonic_ns()
            self._ledger.begin(batch)
            outbox = self._reliable_outbox()
            outbox.add("task_batch", batch.batch_id, batch)
            outbox.mark_sent(
                "task_batch", batch.batch_id, now_ns=submitted_ns
            )
            self._pending_send.popleft()
            profile_emit(
                "target_task_batch_sent",
                instance_id=self.instance_id,
                batch_id=batch.batch_id,
                entries=len(batch.entries),
                local_queue_ns=submitted_ns - batch.sent_ns,
                pending_batches=len(self._pending_send),
            )

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
                self._reliable_outbox().ack("task_batch", message.batch_id)
                ack_received_ns = time.monotonic_ns()
                retry_tasks = self._ledger.apply_ack(message)
                profile_emit(
                    "target_task_ack",
                    instance_id=self.instance_id,
                    batch_id=message.batch_id,
                    accepted=len(message.accepted_transfer_ids),
                    duplicate=len(message.duplicate_transfer_ids),
                    backpressured=len(message.backpressured_transfer_ids),
                    received_ns=ack_received_ns,
                )
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
                received_ns, source_age_ns = self._consume_source_read_done(message)
                profile_emit(
                    "target_source_read_done",
                    instance_id=self.instance_id,
                    transfer_id=message.transfer_id,
                    source_read_from_created_ns=source_age_ns,
                    control_return_ns=received_ns - message.completed_ns,
                )
            elif isinstance(message, SourceReadDoneBatch):
                received_ns = time.monotonic_ns()
                source_ages = []
                for entry in message.entries:
                    _, source_age_ns = self._consume_source_read_done(
                        entry, received_ns=received_ns
                    )
                    source_ages.append(source_age_ns)
                profile_emit(
                    "target_source_read_done_batch",
                    instance_id=self.instance_id,
                    batch_id=message.batch_id,
                    entries=len(message.entries),
                    source_read_from_created_mean_ns=(
                        sum(source_ages) // len(source_ages) if source_ages else 0
                    ),
                    coordinator_to_target_ns=received_ns - message.sent_ns,
                )
            elif isinstance(message, ControlAck):
                self._reliable_outbox().ack(message.operation, message.message_id)
            elif isinstance(message, ErrorMessage):
                if message.code == "PermissionError":
                    self._client.register_target(
                        timeout_s=self.server_args.draft_target_registration_timeout_s
                    )
                    self._reliable_outbox().retry_all_now()
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
        return events

    def _reliable_outbox(self):
        outbox = getattr(self, "_delivery_outbox", None)
        if outbox is None:
            from dt_client import RetryOutbox

            outbox = self._delivery_outbox = RetryOutbox[object](
                retry_interval_ns=CONTROL_RETRY_INTERVAL_NS
            )
        return outbox

    def _flush_reliable_outbox(self) -> None:
        from dt_protocol import CANCEL_TRANSFERS, PULL_DONE

        now_ns = time.monotonic_ns()
        outbox = self._reliable_outbox()
        for operation, message_id, message in outbox.due(now_ns=now_ns):
            if operation == "task_batch":
                sent = self._client.submit(message)
            elif operation == PULL_DONE:
                sent = self._client.pull_done(message)
            elif operation == CANCEL_TRANSFERS:
                sent = self._client.cancel(message)
            else:
                raise RuntimeError(
                    f"unknown Target delivery operation {operation!r}"
                )
            if not sent:
                break
            outbox.mark_sent(operation, message_id, now_ns=now_ns)

    def _expire_local_rounds(self) -> list[tuple[str, str, object]]:
        if not hasattr(self, "round_timeout_ns"):
            return []
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
        for transfer_id in transfer_ids:
            self._ledger.finish(transfer_id)
        self._cancelled_transfers.update(transfer_ids)
        cancellation = CancelTransfers(
            protocol_version=PROTOCOL_VERSION,
            target_instance_id=self.instance_id,
            transfer_ids=transfer_ids,
            reason="target_round_timeout",
        )
        self._reliable_outbox().add(
            CANCEL_TRANSFERS,
            cancel_transfers_id(cancellation),
            cancellation,
        )
        self._flush_reliable_outbox()
        return [
            ("failed", transfer_id, "target_round_timeout")
            for transfer_id in transfer_ids
        ]

    def _consume_source_read_done(
        self, message, *, received_ns: int | None = None
    ) -> tuple[int, int]:
        received_ns = time.monotonic_ns() if received_ns is None else received_ns
        if message.target_instance_id != self.instance_id:
            raise ValueError("SourceReadDone belongs to another target")
        source_ref = self._source_slot_by_transfer.get(message.transfer_id)
        if source_ref is not None and source_ref != (
            message.source_slot_id,
            message.slot_generation,
        ):
            raise ValueError(
                "SourceReadDone slot generation does not match the transfer"
            )
        self._release_source_slot(message.transfer_id)
        round_state = self._ledger.get(message.transfer_id)
        source_age_ns = (
            message.completed_ns - round_state.task.created_ns
            if round_state is not None
            else 0
        )
        if message.transfer_id in self._cancelled_transfers:
            if self._ledger.get(message.transfer_id) is not None:
                self._ledger.finish(message.transfer_id)
            self._cancelled_transfers.discard(message.transfer_id)
        return received_ns, source_age_ns

    def _consume_result_batch(self, batch) -> list[tuple[str, str, object]]:
        cleanup_failed_after_error = False
        try:
            return self._consume_result_batch_body(batch)
        except BaseException:
            cleanup_failed_after_error = True
            raise
        finally:
            if batch.buffer is not None:
                try:
                    self._enqueue_pull_done(batch)
                except BaseException:
                    if cleanup_failed_after_error:
                        logger.exception(
                            "failed to enqueue PullDone while preserving result "
                            "batch error"
                        )
                    else:
                        raise

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
        )
        self._reliable_outbox().add(PULL_DONE, pull_done_id(pull_done), pull_done)
        self._flush_reliable_outbox()

    def _consume_result_batch_body(self, batch) -> list[tuple[str, str, object]]:
        from dt_transport import MmapSlotStore

        receive_start_ns = time.monotonic_ns()
        disposition = self._ledger.apply_results(batch)
        events: list[tuple[str, str, object]] = []
        payload = None
        read_start_ns = time.monotonic_ns()
        if disposition.ready_transfer_ids and batch.buffer is None:
            raise ValueError("ready DT results must carry a published buffer")
        # A published result buffer is owned by Target until PullDone, even if
        # local timeout/cancel processing made every entry stale before this
        # batch was consumed.  Read and validate the published range whenever
        # a buffer is present so ownership release is independent of the
        # ledger's ready classification.
        if batch.buffer is not None:
            if batch.buffer.transport != "mmap":
                raise ValueError("DT V1 Target supports only mmap result buffers")
            store = self._result_stores.get(batch.buffer.uri)
            if store is None:
                store = MmapSlotStore.open(batch.buffer.uri)
                self._result_stores[batch.buffer.uri] = store
            payload = store.read(batch.buffer.buffer_id, batch.buffer.generation)
            minimum = (batch.row_offset + batch.row_count) * batch.buffer.row_width * 4
            if len(payload) < minimum:
                raise ValueError(
                    "DT result buffer is shorter than its published range: "
                    f"need at least {minimum}, got {len(payload)}"
                )
        read_end_ns = time.monotonic_ns()

        ready_ids = set(disposition.ready_transfer_ids)
        failed_ids = set(disposition.failed_transfer_ids)
        expected_row_width = getattr(self, "result_row_width", self.block_size)
        row_stride_bytes = expected_row_width * 4
        candidate_width_bytes = self.block_size * 4
        for entry in batch.entries:
            if entry.transfer_id in ready_ids:
                if entry.transfer_id in self._cancelled_transfers:
                    self._ledger.finish(entry.transfer_id)
                    self._cancelled_transfers.discard(entry.transfer_id)
                    continue
                if batch.buffer.row_width != expected_row_width:
                    raise ValueError("DT result row width does not match block size")
                start = entry.row_index * row_stride_bytes
                candidate_bytes = payload[start : start + candidate_width_bytes]
                root = struct.unpack_from("<i", candidate_bytes)[0]
                state = self._parked[entry.transfer_id]
                if root != state.bonus_token:
                    raise ValueError(
                        "DT result candidate root does not match bonus token"
                    )
                verify_len = self.block_size
                if expected_row_width > self.block_size:
                    verify_len = struct.unpack_from(
                        "<i", payload, start + candidate_width_bytes
                    )[0]
                    if not 1 <= verify_len <= self.block_size:
                        raise ValueError(
                            f"DT adaptive verify length {verify_len} is outside "
                            f"[1, {self.block_size}]"
                        )
                padding_priority = None
                if getattr(self, "adaptive_padding_priority", False):
                    padding_priority = struct.unpack_from(
                        "<i", payload, start + candidate_width_bytes + 4
                    )[0]
                # TP0 is the transport authority: it validated the candidate
                # root against the exact metadata used to launch Draft. Carry
                # that round metadata with the broadcast so slower TP ranks
                # cannot pair the result with stale overlap-relay values.
                if padding_priority is None:
                    ready_payload = (
                        candidate_bytes,
                        verify_len,
                        state.bonus_token,
                        state.seq_len,
                    )
                else:
                    ready_payload = (
                        candidate_bytes,
                        verify_len,
                        padding_priority,
                        state.bonus_token,
                        state.seq_len,
                    )
                events.append(("ready", entry.transfer_id, ready_payload))
                self._ledger.finish(entry.transfer_id)
            elif entry.transfer_id in failed_ids:
                if entry.transfer_id in self._cancelled_transfers:
                    self._ledger.finish(entry.transfer_id)
                    self._cancelled_transfers.discard(entry.transfer_id)
                    continue
                error = entry.error_message or entry.status.value
                events.append(("failed", entry.transfer_id, error))
                self._ledger.finish(entry.transfer_id)

        waits = [
            receive_start_ns - self._parked[transfer_id].parked_ns
            for transfer_id in disposition.ready_transfer_ids
            if transfer_id in self._parked
        ]
        profile_emit(
            "target_result_pull",
            instance_id=self.instance_id,
            batch_id=batch.batch_id,
            ready_count=len(disposition.ready_transfer_ids),
            failed_count=len(disposition.failed_transfer_ids),
            coordinator_to_target_ns=receive_start_ns - batch.sent_ns,
            mmap_read_ns=read_end_ns - read_start_ns,
            payload_bytes=len(payload) if payload is not None else 0,
            round_wait_mean_ns=sum(waits) // len(waits) if waits else 0,
            round_wait_max_ns=max(waits) if waits else 0,
            consume_ns=time.monotonic_ns() - receive_start_ns,
        )
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
        source_ref = self._source_slot_by_transfer.pop(transfer_id, None)
        if source_ref is not None:
            source_slot, _generation = source_ref
            self._free_source_slots.append(source_slot)
