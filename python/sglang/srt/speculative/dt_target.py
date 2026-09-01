from __future__ import annotations

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
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.utils import broadcast_pyobj

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
    lane_id: int = 0
    candidate_bytes: bytes | None = None
    failure: str | None = None


@dataclass(slots=True)
class StagedPinnedPivotBatch:
    """Pinned staging lease filled on the asynchronous result-copy stream."""

    buffer_slot: int
    rows: int
    request_ids: tuple[str, ...]


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
        self.pivot_width = int(scheduler.draft_worker._pivot_width)
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
        if not self.server_args.draft_target_full_wave or self.pipeline_lanes != 2:
            raise ValueError(
                "Draft-Target V1 requires full-wave verification with two lanes"
            )
        self.overlap_result_defer_ns = int(overlap_result_defer_ms * 1_000_000)
        self.is_export_rank = scheduler.ps.tp_rank == 0
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
        self._lane_by_request: dict[str, int] = {}
        self._lane_target_sizes = [0] * self.pipeline_lanes
        self._lane_assignment_cursor = 0
        self._lane_ready_cursor = 0
        self._overlap_launch_poll_ns = 0
        self._overlap_result_ready: bool | None = None
        self._client = None
        self._ledger = None
        self._source_store = None
        self._pivot_host_staging = None
        self._early_pivot_host_buffers: list[torch.Tensor] = []
        self._early_pivot_host_free_slots: deque[int] = deque()
        self._early_pivot_host_in_use: set[int] = set()
        if self.is_export_rank:
            slot_bytes = (
                self.pivot_width * torch.tensor([], dtype=torch.bfloat16).element_size()
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
                    data_endpoint=self._source_store.uri,
                )
                self._client.register_target(
                    timeout_s=self.server_args.draft_target_registration_timeout_s
                )
                self._ledger = TargetRoundLedger(target_instance_id=self.instance_id)
                # The source ring bounds the largest wave. Keep one staging
                # buffer for the serial path and two leases for overlap copies.
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
                self._source_store.close()
                self._source_store.unlink()
                self._source_store = None
                raise

    def stage_overlap_pivot_copy(self, batch: ScheduleBatch, batch_result) -> None:
        """Copy pivots to a leased pinned buffer on the result-copy stream."""
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

    def _release_early_pinned_staging(self, buffer_slot: int) -> None:
        if buffer_slot not in self._early_pivot_host_in_use:
            raise RuntimeError(
                f"DT early pinned staging slot {buffer_slot} is not leased"
            )
        self._early_pivot_host_in_use.remove(buffer_slot)
        self._early_pivot_host_free_slots.append(buffer_slot)

    def park_completed_batch(self, batch: ScheduleBatch, *, batch_result=None) -> None:
        staged_pinned_pivots = getattr(batch_result, "dt_staged_pinned_pivots", None)
        try:
            self._park_completed_batch(batch, batch_result=batch_result)
        finally:
            if staged_pinned_pivots is not None:
                if (
                    getattr(batch_result, "dt_staged_pinned_pivots", None)
                    is staged_pinned_pivots
                ):
                    batch_result.dt_staged_pinned_pivots = None
                self._release_early_pinned_staging(staged_pinned_pivots.buffer_slot)

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

        original_batch_size = len(batch.reqs)
        draft_input = batch.spec_info
        for req in batch.reqs:
            if req.finished() or req.is_retracted:
                self._forget_request_lane(req)
        for req in batch.reqs:
            if req.finished() or req.is_retracted:
                self._next_round.pop(req.rid, None)
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
        if self.is_export_rank and active_indices:
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
                lane_id=lane_ids[active_index],
            )
            parked_reqs.append(req)

            if self.is_export_rank:
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
                        pivot_nbytes=len(pivot_payloads[active_index]),
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

    def poll(self, *, overlap_result=None) -> None:
        packet = None
        if self.is_export_rank:
            packet = (
                time.monotonic_ns(),
                self._poll_export_rank(),
                self._query_overlap_result_ready(overlap_result),
            )
        if self.scheduler.ps.tp_size > 1:
            packet = broadcast_pyobj(
                packet,
                self.scheduler.tp_group.rank,
                self.scheduler.tp_cpu_group,
                src=self.scheduler.tp_group.ranks[0],
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
            else:
                state.failure = payload

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
        for state in ready:
            ready_by_lane[state.lane_id].append(state)

        for offset in range(self.pipeline_lanes):
            lane_id = (self._lane_ready_cursor + offset) % self.pipeline_lanes
            lane_ready = ready_by_lane[lane_id]
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

        states = list(self._parked.values())
        for state in states:
            self._parked.pop(state.transfer_id, None)
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

        for state in ready:
            self._parked.pop(state.transfer_id)

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
        batch.input_ids = None
        self.scheduler.future_map.stash(
            batch.req_pool_indices,
            RelayPayload(
                bonus_tokens=bonus_tokens,
                hidden_states=batch.spec_info.hidden_states,
            ),
        )
        batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
            batch, self.scheduler.model_config.vocab_size
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
            self._forget_request_lane(state.req)
            self._next_round.pop(state.req.rid, None)
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
        return [state.req for state in matched]

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
        self._lane_assignment_cursor = (lane_id + 1) % self.pipeline_lanes
        return lane_id

    def _forget_request_lane(self, req: Req) -> None:
        self._lane_by_request.pop(req.rid, None)

    def close(self) -> None:
        for store in self._result_stores.values():
            store.close()
        self._result_stores.clear()
        if self._client is not None:
            self._client.close()
        if self._source_store is not None:
            self._source_store.close()
            self._source_store.unlink()

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
            outbox = self._delivery_outbox
            outbox.add("task_batch", batch.batch_id, batch)
            outbox.mark_sent("task_batch", batch.batch_id, now_ns=submitted_ns)
            self._pending_send.popleft()

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
        return events

    def _flush_reliable_outbox(self) -> None:
        from dt_protocol import CANCEL_TRANSFERS, PULL_DONE

        now_ns = time.monotonic_ns()
        outbox = self._delivery_outbox
        for operation, message_id, message in outbox.due(now_ns=now_ns):
            if operation == "task_batch":
                sent = self._client.submit(message)
            elif operation == PULL_DONE:
                sent = self._client.pull_done(message)
            elif operation == CANCEL_TRANSFERS:
                sent = self._client.cancel(message)
            else:
                raise RuntimeError(f"unknown Target delivery operation {operation!r}")
            if not sent:
                break
            outbox.mark_sent(operation, message_id, now_ns=now_ns)

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
        for transfer_id in transfer_ids:
            self._ledger.finish(transfer_id)
        self._cancelled_transfers.update(transfer_ids)
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
        if message.transfer_id in self._cancelled_transfers:
            if self._ledger.get(message.transfer_id) is not None:
                self._ledger.finish(message.transfer_id)
            self._cancelled_transfers.discard(message.transfer_id)

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
        self._delivery_outbox.add(PULL_DONE, pull_done_id(pull_done), pull_done)
        self._flush_reliable_outbox()

    def _consume_result_batch_body(self, batch) -> list[tuple[str, str, object]]:
        from dt_transport import MmapSlotStore

        disposition = self._ledger.apply_results(batch)
        events: list[tuple[str, str, object]] = []
        payload = None
        if disposition.ready_transfer_ids and batch.buffer is None:
            raise ValueError("ready DT results must carry a published buffer")
        # Target owns every published range until PullDone, including ranges
        # whose entries became stale or cancelled before consumption.
        if batch.buffer is not None:
            if batch.buffer.transport != "mmap":
                raise ValueError("DT V1 Target supports only mmap result buffers")
            store = self._result_stores.get(batch.buffer.uri)
            if store is None:
                store = MmapSlotStore.open(batch.buffer.uri)
                self._result_stores[batch.buffer.uri] = store
            payload = store.read(batch.buffer.buffer_id, batch.buffer.generation)
            minimum = (batch.row_offset + batch.row_count) * self.block_size * 4
            if batch.buffer.row_width != self.block_size:
                raise ValueError("DT result row width does not match block size")
            if len(payload) < minimum:
                raise ValueError(
                    "DT result buffer is shorter than its published range: "
                    f"need at least {minimum}, got {len(payload)}"
                )

        ready_ids = set(disposition.ready_transfer_ids)
        failed_ids = set(disposition.failed_transfer_ids)
        row_stride_bytes = self.block_size * 4
        for entry in batch.entries:
            transfer_id = entry.transfer_id
            if transfer_id in ready_ids:
                if transfer_id in self._cancelled_transfers:
                    self._ledger.finish(transfer_id)
                    self._cancelled_transfers.discard(transfer_id)
                    continue
                start = entry.row_index * row_stride_bytes
                candidate_bytes = payload[start : start + row_stride_bytes]
                state = self._parked[transfer_id]
                root = int.from_bytes(
                    candidate_bytes[:4], byteorder="little", signed=True
                )
                if root != state.bonus_token:
                    raise ValueError(
                        "DT result candidate root does not match bonus token"
                    )
                events.append(
                    (
                        "ready",
                        transfer_id,
                        (candidate_bytes, state.bonus_token, state.seq_len),
                    )
                )
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
        source_ref = self._source_slot_by_transfer.pop(transfer_id, None)
        if source_ref is not None:
            source_slot, _generation = source_ref
            self._free_source_slots.append(source_slot)
