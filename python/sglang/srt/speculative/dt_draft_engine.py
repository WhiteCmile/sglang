from __future__ import annotations

import argparse
import bisect
import contextlib
import enum
import json
import logging
import os
import re
import signal
import time
import uuid
from array import array
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import Future
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import msgspec
import numpy as np
import torch
import torch.nn.functional as F
from dt_client import RetryOutbox
from dt_profile import cuda_sync_enabled
from dt_profile import emit as profile_emit
from dt_profile import enabled as profile_enabled
from dt_profile import event_enabled as profile_event_enabled
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import bootstrap
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo, ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.model_executor.model_runner import (
    DT_BENCHMARK_MEASUREMENT_RID_PREFIX,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    compute_position,
    compute_position_torch,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.draft_worker_common import build_draft_tp_worker
from sglang.srt.speculative.flashmtp_utils import parse_flashmtp_draft_config
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)

IDLE_POLL_S = 0.0005
# Idle-loop poll interval while a DFlash/EAGLE3/FlashMTP completion or a
# direct candidate push is still in flight on the GPU: long enough to keep
# the CPU cost bounded, short enough not to delay the Draft done -> push ->
# Target READY path (issue #149).
INFLIGHT_POLL_S = 0.00005
DIRECT_PUSH_SPIN_NS = 200_000
# Token-count buckets for the CUDA-graphed DFlash KV materializer.
DFLASH_MATERIALIZE_GRAPH_BUCKETS = (8, 16, 32, 64, 96, 128, 192, 256, 384, 512)

# Packed per-row DFlash submit metadata (issue #358): one pinned host block
# plus one device block per staging slot, uploaded with a single H2D per
# batch and consumed by the reconcile gather, the top-up page-table write,
# and the proposal block preparation.  This field order is the single
# source for the host pack and every device-side slice.
_DFLASH_META_FIELD_NAMES = (
    "req_pool_idx",  # request row in req_to_token
    "feature_offset",  # first newly committed position (gather start)
    "committed_len",  # committed prefix end = proposal prefix length
    "bonus_token",  # Target-provided root of the proposal block
    "allocated_len",  # kv_allocated_len before this batch's top-up
)
(
    _DFLASH_META_REQ_POOL_IDX,
    _DFLASH_META_FEATURE_OFFSET,
    _DFLASH_META_COMMITTED_LEN,
    _DFLASH_META_BONUS_TOKEN,
    _DFLASH_META_ALLOCATED_LEN,
) = range(len(_DFLASH_META_FIELD_NAMES))
_DFLASH_META_FIELDS = len(_DFLASH_META_FIELD_NAMES)


class _GraphedDFlashMaterializer:
    """CUDA-graph replay of the DFlash target-hidden -> draft-KV writer.

    The eager writer launches the projection, fused norm/RoPE and one KV write
    per layer from Python on every Draft batch.  Each token-count bucket is
    captured once; padded rows write KV slot 0, which the allocator reserves for
    padded tokens.  Inputs are copied into static buffers on the same stream,
    so a later batch never overwrites them before an earlier replay ran.
    Larger batches (prefill chunks) use the eager writer unchanged.
    """

    def __init__(self, materializer, *, width: int, dtype, device, buckets) -> None:
        self._materializer = materializer
        self._buckets = tuple(sorted(buckets))
        top = self._buckets[-1]
        self._hidden = torch.zeros((top, width), dtype=dtype, device=device)
        self._cache_loc = torch.zeros((top,), dtype=torch.int64, device=device)
        self._positions = torch.zeros((top,), dtype=torch.int64, device=device)
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}

    def capture(self) -> None:
        pool = torch.cuda.graph_pool_handle()
        # Largest first, so the writer's growable workspace reaches its final
        # size before any capture records its address.
        for bucket in reversed(self._buckets):
            self._eager(bucket)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                self._eager(bucket)
            self._graphs[bucket] = graph
        torch.cuda.synchronize()

    def _eager(self, rows: int) -> None:
        self._materializer.append(
            target_hidden=self._hidden[:rows],
            cache_loc=self._cache_loc[:rows],
            positions=self._positions[:rows],
        )

    def append(self, *, target_hidden, cache_loc, positions) -> None:
        rows = int(target_hidden.shape[0])
        bucket = next((size for size in self._buckets if size >= rows), None)
        if bucket is None or bucket not in self._graphs or rows == 0:
            self._materializer.append(
                target_hidden=target_hidden, cache_loc=cache_loc, positions=positions
            )
            return
        self._hidden[:rows].copy_(target_hidden)
        self._cache_loc[:rows].copy_(cache_loc)
        self._positions[:rows].copy_(positions)
        if rows < bucket:
            self._cache_loc[rows:bucket].zero_()
            self._positions[rows:bucket].zero_()
        self._graphs[bucket].replay()

    def append_rows(self, *, source, rows_index, cache_loc, positions) -> None:
        """An all-native-ring batch: gather its landing-buffer rows straight
        into the static hidden buffer (one index_select; the caller's per-row
        views, torch.cat and the copy_ it needed are all skipped)."""

        rows = int(rows_index.shape[0])
        bucket = next((size for size in self._buckets if size >= rows), None)
        if bucket is None or bucket not in self._graphs or rows == 0:
            self._materializer.append(
                target_hidden=source.index_select(0, rows_index),
                cache_loc=cache_loc,
                positions=positions,
            )
            return
        torch.index_select(source, 0, rows_index, out=self._hidden[:rows])
        self._cache_loc[:rows].copy_(cache_loc)
        self._positions[:rows].copy_(positions)
        if rows < bucket:
            self._hidden[rows:bucket].zero_()
            self._cache_loc[rows:bucket].zero_()
            self._positions[rows:bucket].zero_()
        self._graphs[bucket].replay()
# A direct dispatch never confirmed by the Coordinator is dropped from local
# accounting this long after its task deadline, once its result is held.
UNCONFIRMED_DIRECT_GRACE_NS = 1_000_000_000


def _select_candidate_rows(rows: torch.Tensor, indices: list[int]) -> torch.Tensor:
    """Select ordered candidate rows without blocking the serving loop.

    Indexing a CUDA tensor with a Python list uploads the index from pageable
    memory, which waits for all work already queued on the stream, i.e. the
    next compute batch. Keep the common all-active case copy-free and stage
    partial indices through pinned memory.
    """
    if len(indices) == int(rows.shape[0]) and indices == list(range(len(indices))):
        return rows
    if not rows.is_cuda:
        return rows[indices].contiguous()
    index_cpu = torch.tensor(indices, dtype=torch.int64, pin_memory=True)
    index = index_cpu.to(rows.device, non_blocking=True)
    return rows.index_select(0, index)
# Queue candidate pushes behind the proposal at launch and let the Target
# observe READY from GPU-written host records (dt_candidate_p2p).
GPU_READY_RECORD = os.environ.get("DT_GPU_READY_RECORD", "1") == "1"
PUSH_COMPLETION_RECORD = os.environ.get("DT_PUSH_COMPLETION_RECORD", "1") == "1"
DIRECT_RING = os.environ.get("DT_DIRECT_RING", "1") == "1"
# Draft redesign D11 (issue #280): the redesign switches below, D6's
# DT_DRAFT_INGRESS (dt_direct_ring) and D3/D4's DT_RING_CONFIRMATION
# (dt_protocol) default to the new path.  Each later DFlash stage requires
# the earlier ones (checked in __init__), so the legacy loop is selected by
# setting them back together: DT_DRAFT_SCHEDULER= DT_DRAFT_SEND_LOOP=0
# DT_DRAFT_INGRESS= DT_DRAFT_CONTROL_LOOP=0 DT_RING_CONFIRMATION=1
# DT_DRAFT_ADMISSION= DT_DRAFT_SLOT_DECOUPLE=0 (and DT_DRAFT_GC= for the old
# heap; DT_COORDINATOR_GC= in the Coordinator).
# Draft redesign D1 (issue #270): DFlash scratch is per batch, not per staging
# slot, so a row is never held behind the batch that owns its landing buffer
# (no compute-busy, no submit-time seal; the ring worker seals on overflow).
DRAFT_SLOT_DECOUPLE = os.environ.get("DT_DRAFT_SLOT_DECOUPLE", "1") == "1"
# Draft redesign D5 (issue #274): a native thread owns the Coordinator socket
# (dt_control.DraftControlLoop, issue #288) and the DFlash loop's idle tail
# waits on the native wake word (dt_wake) for the same poll interval, so a
# Coordinator message ends the wait at once.
DRAFT_CONTROL_LOOP = os.environ.get("DT_DRAFT_CONTROL_LOOP", "1") == "1"
# Draft redesign D8 (issue #277, design §7): the native publish worker also
# watches each submitted batch's completion event, so a batch stops counting
# against the depth gate at its GPU done rather than when this loop reaches
# the completion poll.  It pushes the batch's landing releases to ingress
# itself and hands the scheduler a DONE record (CUDA elapsed included) to
# absorb.
DRAFT_SEND_LOOP = os.environ.get("DT_DRAFT_SEND_LOOP", "1") == "1"
# Draft redesign D10 (issue #279, design §5.1-5.2): DT_DRAFT_SCHEDULER=v2 runs
# DFlash on a wake-driven state machine (_run_scheduler_v2) instead of run()'s
# staged loop: the depth gate reads frozen_seq - done_seq, and with one batch
# in flight the next one freezes late, when the running batch's estimated GPU
# end minus the estimated CPU work of the next is reached.  Other methods keep
# run().
DRAFT_SCHEDULER_V2 = os.environ.get("DT_DRAFT_SCHEDULER", "v2") == "v2"
# Subtracted from the late-freeze deadline (design §5.2; 0 as the Target's).
DRAFT_LATE_FREEZE_MARGIN_NS = (
    int(os.environ.get("DT_DRAFT_LATE_FREEZE_MARGIN_US", "0")) * 1000
)
# One executing plus one submitted Draft compute batch.
DRAFT_COMPUTE_DEPTH = 2
RING_RETIRED_LIMIT = 65536
# cuda-ipc proposal admission bound.  The direct path carries no result-slot
# pool, so this explicit credit replaces the admission accounting the pool
# used to provide (issue #42); 64 is the validated dev operating point.
DEFAULT_DIRECT_PROPOSAL_ADMISSION_LIMIT = 64
# Draft redesign D2 (§5.3): DT_DRAFT_ADMISSION=bound drops that credit.  The
# batch bound (max_batch_size rows, feature_buffer_max_tokens) plus the KV and
# req-row fit at ring admission are then the only proposal admission limits.
DRAFT_ADMISSION_BOUND = os.environ.get("DT_DRAFT_ADMISSION", "bound") == "bound"
# Draft redesign D7b (issue #295): DT_DRAFT_GC=managed freezes the Draft
# process's warm heap.  A full (generation-2) collection scans the whole heap
# with the scheduler loop stopped (0.36-0.54 s at r56); ~0.93 M of its objects
# exist before serving starts (imported code) and serving adds only ~10 k.
DRAFT_GC_MANAGED = os.environ.get("DT_DRAFT_GC", "managed") == "managed"


def _resolve_direct_proposal_admission_limit() -> int:
    """Resolve the cuda-ipc proposal admission bound.

    Unset reproduces the validated 64; DT_DIRECT_PROPOSAL_ADMISSION_LIMIT
    overrides it for experiments without touching launch argv.
    """

    env_limit = os.environ.get("DT_DIRECT_PROPOSAL_ADMISSION_LIMIT")
    limit = int(env_limit) if env_limit else DEFAULT_DIRECT_PROPOSAL_ADMISSION_LIMIT
    if limit <= 0:
        raise ValueError("direct_proposal_admission_limit must be positive")
    return limit


# Provisional deadline for a §7.1 ring-record task; its Coordinator
# confirmation copy carries the authoritative one.
# READY entry fields the DRAIN loop reads (dt_ring_worker.cpp ReadyEntry).
_READY_ROW_FIELDS = (
    "seq", "ring_id", "record_seq", "position", "req_handle", "req_gen", "round_id",
    "attempt", "seq_len", "token_count", "bonus_token", "landing_k", "landing_gen",
    "row_first", "token_at", "ready_ns",
)
_RING_RECORD_TIMEOUT_NS = (
    int(os.environ.get("DT_RING_RECORD_TIMEOUT_S", "30")) * 1_000_000_000
)
# Why FREEZE left a READY row waiting (issue #292).  A row's time between two
# evaluations is charged to the reason of the earlier one: "unscanned" is
# READY publish -> first pass, "compute_depth" any part of it (after the drain)
# the scheduler spent with DRAFT_COMPUTE_DEPTH batches in flight, when no
# FREEZE pass runs, and "late_freeze" (D10) any part it spent holding the
# next freeze for the late-freeze deadline (design §5.2).  The charges of a
# row sum to its hold, publish -> frozen.
_ReadySkip = enum.IntEnum(
    "_ReadySkip",
    "unscanned compute_depth owner_inflight owner_selected proposal_mode "
    "batch_rows batch_tokens req_rows kv late_freeze",
    start=0,
)
_READY_SKIP_REASONS = tuple(reason.name for reason in _ReadySkip)
_READY_DROP_OUTCOMES = ("released", "cancelled", "expired")
# Long holds (publish -> frozen or dropped): counted untraced above the first,
# one draft_ready_skip record each above the second under DT_PROFILE_DIR.
_READY_LONG_HOLD_NS = 100_000_000
_READY_SKIP_RECORD_NS = 20_000_000
# Initial control delivery is immediate.  Retrying every scheduler tick-scale
# interval amplifies delayed ACKs under full-wave load; keep retries reliable
# but slow enough that the Coordinator can drain the original transaction.
CONTROL_RETRY_INTERVAL_NS = 1_000_000_000
DT_DRAFT_GRAPH_MEASUREMENT_MARKER = "DT Draft graph measurement evidence: "
DT_DIRECT_GPU_SOURCE_MARKER = "DT direct GPU source evidence: "
DT_LIFECYCLE_STATUS_MARKER = "DT lifecycle status: "
DT_LIFECYCLE_STATUS_INTERVAL_NS = 250_000_000
DT_DIRECT_GPU_REQUEST_EVIDENCE_IDLE_NS = 750_000_000
# Draft-boundness stats line (issue #174): throttled so the production path
# pays at most one log line per interval while the queue is busy.
DT_DRAFT_STATS_INTERVAL_NS = 2_000_000_000
# Shadow late-freeze estimator (design §5.2; D0 only logs it, issue #269):
# weight of the newest batch in the per-bucket EMA.
BATCH_ESTIMATE_EMA_WEIGHT = 0.125
# Parts timed inside a submit stage (issue #306, profile only): the FREEZE
# DraftTask build and KV/req-row fit per frozen row, the proposal metadata's
# three H2D enqueues, and the D9 hand-off's destination-region reads.
SUBMIT_SUB_STAGES = (
    "freeze_task_build", "freeze_fit", "metadata_h2d", "handoff_region_read"
)


class DraftBatchEstimator:
    """G(n)/C(n) of the late-freeze design (§5.2): logged under D0, read by
    the v2 scheduler's late freeze (D10, ``predict``).

    G is a DFlash batch's GPU span (materialize_start -> proposal_done, CUDA
    elapsed) and C the scheduler CPU work that formed and launched it
    (selection start -> hand-off end).  Each is an EMA per power-of-two size
    bucket, keyed both by rows and by feature tokens (design Q3); every
    completed batch reports the predictions made before it, then updates.
    """

    def __init__(self) -> None:
        self._ema: dict[tuple[str, int, str], float] = {}

    @staticmethod
    def bucket(size: int) -> int:
        return max(size, 1).bit_length()  # sizes [2**(b-1), 2**b)

    def predict(self, key: str, size: int, name: str) -> float | None:
        """The EMA of ``name`` ("g" or "c") for the ``key`` bucket of ``size``."""

        return self._ema.get((key, self.bucket(size), name))

    def observe(self, *, rows: int, tokens: int, gpu_ns: int, cpu_ns: int) -> dict:
        fields = {}
        for key, size in (("rows", rows), ("tokens", tokens)):
            bucket = self.bucket(size)
            fields[f"{key}_bucket"] = bucket
            for name, actual in (("g", gpu_ns), ("c", cpu_ns)):
                predicted = self._ema.get((key, bucket, name))
                fields[f"{name}_pred_by_{key}_ns"] = (
                    None if predicted is None else int(predicted)
                )
                self._ema[(key, bucket, name)] = (
                    actual
                    if predicted is None
                    else predicted + BATCH_ESTIMATE_EMA_WEIGHT * (actual - predicted)
                )
        return fields


def _prepare_eagle3_extend_batch(batch: ScheduleBatch) -> bool:
    """Prepare a text-only persistent EAGLE3 reconcile without dead token input.

    The feature-aligned IDs are installed by the caller. Keep the generic KV
    allocator, sampling state and request accounting; only skip rebuilding a
    committed-ID tensor that this adapter never consumes. Unsupported request
    types fall back before any state is changed.
    """
    from sglang.srt.managers.schedule_batch import compute_extend_logprob_start_len
    from sglang.srt.mem_cache.allocation import alloc_for_extend
    from sglang.srt.runtime_context import get_server_args
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
    from sglang.srt.utils.common import is_pin_memory_available

    server_args = get_server_args()
    reqs = batch.reqs
    if (
        not reqs
        or not batch.spec_algorithm.is_eagle3()
        or batch.tree_cache.supports_swa()
        or batch.tree_cache.supports_mamba()
        or batch.is_dllm()
        or batch.enable_overlap
        or batch.return_logprob
        or batch.model_config.is_encoder_decoder
        or server_args.enable_mamba_extra_buffer()
        or any(
            req.input_embeds is not None
            or req.positional_embed_overrides is not None
            or req.multimodal_inputs is not None
            or req.mamba_pool_idx is not None
            for req in reqs
        )
    ):
        return False

    prefix_lens = [len(req.prefix_indices) for req in reqs]
    seq_lens = [req.extend_range.end for req in reqs]
    extend_lens = [req.extend_range.length for req in reqs]
    for req, prefix_len, seq_len in zip(reqs, prefix_lens, seq_lens):
        # Generic preparation derives token counts by slicing the history.
        # Validate the same boundaries explicitly when using metadata alone.
        if (
            prefix_len != req.extend_range.start
            or seq_len < prefix_len
            or seq_len > len(req.full_untruncated_fill_ids)
        ):
            raise ValueError("EAGLE3 extend range does not match committed history")
    extend_logprob_start_lens = [
        compute_extend_logprob_start_len(
            logprob_start_len=req.logprob_start_len,
            prefix_len=prefix_len,
            extend_len=extend_len,
            full_untruncated_fill_len=len(req.full_untruncated_fill_ids),
        )
        for req, prefix_len, extend_len in zip(reqs, prefix_lens, extend_lens)
    ]
    pin = is_pin_memory_available(batch.device)
    batch.forward_mode = ForwardMode.EXTEND
    batch.prefix_lens = prefix_lens
    batch.extend_lens = extend_lens
    batch.extend_num_tokens = sum(extend_lens)
    batch.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, pin_memory=pin).to(
        batch.device, non_blocking=True
    )
    batch.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
    orig_seq_lens = torch.tensor(
        [max(end, len(req.origin_input_ids)) for req, end in zip(reqs, seq_lens)],
        dtype=torch.int32,
        pin_memory=pin,
    ).to(batch.device, non_blocking=True)
    out_cache_loc, req_pool_indices, req_pool_indices_cpu = alloc_for_extend(batch)

    # Keep this accounting in lockstep with ScheduleBatch.prepare_for_extend.
    for req, seq_len, prefix_len in zip(reqs, seq_lens, prefix_lens):
        req.extend_batch_idx += 1
        req.kv_committed_len = seq_len
        if not req.retracted_stain:
            req.cached_tokens += prefix_len - req.already_computed
            if not req._cache_breakdown_computed:
                host_total = req.host_hit_length
                storage_portion = min(host_total, req.storage_hit_length)
                req.cached_tokens_device = max(0, prefix_len - host_total)
                req.cached_tokens_host = host_total - storage_portion
                req.cached_tokens_storage = storage_portion
                req._cache_breakdown_computed = True
            req.already_computed = seq_len
        req.is_retracted = False

    batch.input_ids = None
    batch.prefill_input_ids_cpu = None
    batch.req_pool_indices = req_pool_indices
    batch.req_pool_indices_cpu = req_pool_indices_cpu
    batch.orig_seq_lens = orig_seq_lens
    batch.out_cache_loc = out_cache_loc
    batch.input_embeds = None
    batch.replace_embeds = None
    batch.replace_positions = None
    batch.multimodal_inputs = [None] * len(reqs)
    batch.seq_lens_sum = sum(seq_lens)
    batch.extend_logprob_start_lens = extend_logprob_start_lens
    batch.extend_input_logprob_token_ids = None
    batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
        batch, batch.model_config.vocab_size
    )
    return True


def _eagle3_feature_staging_view_bounds(
    *,
    requires_proposal: bool,
    row_ranges: tuple[tuple[int, int], ...],
) -> tuple[int, int] | None:
    """Return safe direct-view bounds for a proposal batch's staging rows.

    Proposal completion copies candidates to CPU before the staging slot is
    released, which fences the same CUDA stream that consumed the feature
    rows.  Materialization-only batches lack that fence and must retain their
    owned concatenation even when the source ranges happen to be adjacent.
    """
    if not requires_proposal or not row_ranges:
        return None
    if any(
        row_ranges[index][1] != row_ranges[index + 1][0]
        for index in range(len(row_ranges) - 1)
    ):
        return None
    return row_ranges[0][0], row_ranges[-1][1]


@dataclass(slots=True)
class PendingDraftTask:
    task: object
    pending_before: int = 0
    pending_after: int = 0
    taken_ns: int = 0
    oldest_queue_age_ns: int = 0
    batch_feature_token_count: int = 0
    stop_reason: str = "unrecorded"
    cap_limited: bool = False


@dataclass(slots=True)
class PublishedResult:
    batch_id: str
    slot_id: int
    generation: int
    transfer_ids: tuple[str, ...]
    transfer_rows: dict[str, tuple[str, int]]
    range_transfer_ids: dict[tuple[str, str, int, int, int, int], frozenset[str]]


@dataclass(slots=True)
class CapturedDraftBatch:
    graph: object
    pivots: torch.Tensor
    bonus_tokens: torch.Tensor
    block_ids: torch.Tensor
    candidates: torch.Tensor


@dataclass(slots=True)
class PersistentDraftRequest:
    """One Draft-owned request slot and its valid committed KV boundary."""

    req: Req
    target_instance_id: str
    request_id: str
    draft_kv_handle: str
    valid_committed_len: int
    last_round_id: int
    proposal_base_len: int
    committed_token_ids: list[int]
    # Native ingress: (gen << 32) | cell of the request's rid-ledger
    # registration at insert (0: none), so the index can drop exactly this
    # registration even if the rid was registered again meanwhile.
    cell_key: int = 0


@dataclass(slots=True)
class PreparedDraftMigration:
    message: object
    req: Req
    reservation: object
    prepared_ns: int
    import_message: object | None = None
    transfer_future: object | None = None
    scheduler_submit_ns: int = 0


@dataclass(slots=True)
class DeferredMigrationGpuControl:
    message: object
    deferred_ns: int


@dataclass(slots=True)
class PendingFeatureBatch:
    batch_id: str
    pending: list[PendingDraftTask]
    staging_slot: int
    row_ranges: tuple[tuple[int, int], ...]
    submitted_ns: int
    futures: tuple[Future, ...] = ()
    # Compatibility for lifecycle fixtures and old snapshots. New serving
    # code always supplies one Future per request through ``futures``.
    future: Future | None = None

    def __post_init__(self) -> None:
        if self.future is not None:
            if self.futures:
                raise ValueError("feature batch cannot set future and futures")
            self.futures = (self.future,)


@dataclass(slots=True)
class InputReadyFeature:
    pending: PendingDraftTask
    staging_slot: int
    row_range: tuple[int, int]
    ready_ns: int
    # W7b mark, emitted only in the profile: passed over while its staging
    # slot was compute-busy (never set under DT_DRAFT_SLOT_DECOUPLE).
    slot_busy_blocked: bool = False
    # DT_DRAFT_INGRESS=native: the READY seq of a READY-table row (0: none).
    ready_seq: int = 0


@dataclass(slots=True)
class PendingDFlashCompletion:
    pending: list[PendingDraftTask]
    staging_slots: tuple[int, ...]
    staging_slot_counts: tuple[int, ...]
    completion_event: torch.cuda.Event
    materialize_start_event: torch.cuda.Event
    materialize_done_event: torch.cuda.Event
    proposal_done_event: torch.cuda.Event
    proposal_tasks: tuple[object, ...]
    candidates: torch.Tensor | None
    candidate_rows_cpu: torch.Tensor | None
    launch_started_ns: int
    tasks_prepared_ns: int
    proposal_metadata_ready_ns: int
    materialize_submitted_ns: int
    launched_ns: int
    proposal_execution: str | None
    proposal_fallback_reason: str | None
    reconcile_prepare_phase_ns: tuple[int, ...]
    prior_unfinished_proposal_count: int
    prior_candidate_copy_pending_count: int
    # DT_DRAFT_INGRESS=native: (ring id, record seq) of each READY-table row
    # with a landing placement; their placements are released at retire.
    ready_rows: tuple[tuple[int, int], ...] = ()
    # Profile-only D0 stamps (issue #269): selection start, membership fixed,
    # hand-off end (launch-time push queued), and the DT_GPU_CLOCK stamp of
    # proposal_done (slot -1: none).
    select_started_ns: int = 0
    frozen_ns: int = 0
    handoff_ns: int = 0
    clock_slot: int = -1
    clock_done: torch.cuda.Event | None = None
    # DT_DRAFT_SLOT_DECOUPLE: owns per-batch scratch batch_seq % depth (-1: off).
    batch_seq: int = -1
    # Profile-only submit stage marks and nested parts (issue #306).
    submit_marks: dict[str, int] | None = None
    submit_sub_ns: list[int] | None = None


@dataclass(slots=True)
class PendingEagle3Completion:
    """Existing persistent completion owner, shared by EAGLE3 and AR."""

    pending: list[PendingDraftTask]
    staging_slots: tuple[int, ...]
    staging_slot_counts: tuple[int, ...]
    completion_event: torch.cuda.Event
    submitted_ns: int
    candidates: torch.Tensor | None = None
    candidate_rows_cpu: torch.Tensor | None = None
    result_slot: int | None = None
    # True while the completion owns one direct proposal-admission credit
    # (cuda-ipc standalone path; acquired at batch submit).
    admission_credit: bool = False


@dataclass(slots=True)
class PendingFlashMTPCompletion:
    """Own FlashMTP candidate bytes until same-stream D2H completes."""

    pending: list[PendingDraftTask]
    staging_slot: int
    completion_event: torch.cuda.Event
    bonus_h2d_start_event: torch.cuda.Event | None
    bonus_h2d_done_event: torch.cuda.Event | None
    proposal_start_event: torch.cuda.Event | None
    proposal_done_event: torch.cuda.Event | None
    candidate_rows_cpu: torch.Tensor | None
    execute_started_ns: int
    inputs_ready_ns: int
    submitted_ns: int
    bonus_pack_ns: int
    bonus_h2d_submit_ns: int
    bonus_device: str
    execution: str
    measurement_rows: int
    prior_pending_flashmtp_completions: int
    # Legacy mmap-only unit fixtures predate direct GPU publication.  Keeping
    # the GPU snapshot optional at the tail preserves their construction while
    # production direct completions always populate it.
    candidates: torch.Tensor | None = None


@dataclass(slots=True)
class PendingCandidatePush:
    push: object
    tasks: list[object]
    result_slot: int | None
    candidates: torch.Tensor | None
    cancelled: bool = False
    # False for a dropped ring copy whose round was already reported.
    coordinator_result: bool = True
    # True while the push owns one direct proposal-admission credit, returned
    # when the GPU push completion retires it.  Launch-time early pushes and
    # cancellation tombstones hold no credit.
    admission_credit: bool = False


STANDALONE_DRAFT_DECODE_GRAPH_BATCH_SIZES = (
    1,
    2,
    4,
    8,
    12,
    16,
    24,
    32,
    40,
    48,
    56,
    64,
    72,
    80,
    88,
    96,
    104,
    112,
    120,
    128,
)
STANDALONE_DRAFT_PREFILL_GRAPH_BATCH_SIZES = (
    1,
    2,
    3,
    4,
    8,
    12,
    16,
    20,
    24,
    28,
    32,
    48,
    64,
    80,
    96,
    112,
    128,
    144,
    160,
    176,
    192,
    208,
    224,
    240,
    256,
    288,
    320,
    352,
    384,
    416,
    448,
    480,
    512,
    576,
    640,
    704,
    768,
    832,
    896,
    960,
    1024,
    1280,
    1536,
    1792,
    2048,
    2304,
    2560,
    2816,
    3072,
    3328,
    3584,
    3840,
    4096,
    4608,
    5120,
    5632,
    6144,
    6656,
    7168,
    7680,
    8192,
)
DRAFT_PREFILL_GRAPH_MAX_TOKENS = STANDALONE_DRAFT_PREFILL_GRAPH_BATCH_SIZES[-1]


def standalone_draft_decode_graph_batch_sizes(
    max_batch_size: int,
) -> tuple[int, ...]:
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    sizes = [
        size
        for size in STANDALONE_DRAFT_DECODE_GRAPH_BATCH_SIZES
        if size <= max_batch_size
    ]
    if max_batch_size not in sizes:
        sizes.append(max_batch_size)
    return tuple(sizes)


def draft_cuda_graph_batch_sizes(max_batch_size: int) -> tuple[int, ...]:
    """Return low-padding capture buckets for standalone FlashMTP batches."""
    if max_batch_size <= 0:
        raise ValueError("max_batch_size must be positive")
    sizes = (
        list(range(1, 9))
        + list(range(10, 33, 2))
        + list(range(36, 65, 4))
        + list(range(72, max_batch_size + 1, 8))
    )
    sizes = [size for size in sizes if size <= max_batch_size]
    if max_batch_size not in sizes:
        sizes.append(max_batch_size)
    return tuple(sorted(set(sizes)))


class StandaloneDraftCudaGraphRunner:
    """CUDA Graph replay for the fixed-width standalone FlashMTP forward."""

    def __init__(
        self,
        *,
        draft_model,
        token_embedding,
        device: torch.device,
        pivot_width: int,
        draft_block_len: int,
        block_size: int,
        mask_token_id: int,
        local_positions: torch.Tensor,
        max_batch_size: int,
        target_lm_head=None,
    ) -> None:
        self.draft_model = draft_model
        self.token_embedding = token_embedding
        self.device = device
        self.pivot_width = int(pivot_width)
        self.draft_block_len = int(draft_block_len)
        self.block_size = int(block_size)
        self.mask_token_id = int(mask_token_id)
        self.local_positions = local_positions
        self.target_lm_head = target_lm_head
        self.batch_sizes = draft_cuda_graph_batch_sizes(max_batch_size)
        self._captured: dict[int, CapturedDraftBatch] = {}
        self._replay_evidence_emitted = False

    def _sample_proposals(
        self, *, block_hidden: torch.Tensor, bonus_tokens: torch.Tensor
    ) -> torch.Tensor:
        if self.draft_model.markov_head is not None:
            return self.draft_model.sample_draft_tokens(
                block_hidden=block_hidden,
                first_prev_token_ids=bonus_tokens,
            )
        if self.target_lm_head is None:
            raise RuntimeError(
                "legacy FlashMTP requires the Target LM head for proposal sampling"
            )
        prediction_hidden = self.draft_model.prediction_hidden(block_hidden)
        logits = F.linear(prediction_hidden, self.target_lm_head.weight)
        return torch.argmax(logits, dim=-1)

    def capture(self) -> None:
        pool = torch.cuda.graph_pool_handle()
        capture_stream = torch.cuda.Stream(device=self.device)
        capture_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(capture_stream), torch.inference_mode():
            # Warm up lazy kernels and allocator paths outside graph capture.
            self._run_once(self.batch_sizes[-1])
        torch.cuda.current_stream(self.device).wait_stream(capture_stream)
        torch.cuda.synchronize(self.device)

        for batch_size in reversed(self.batch_sizes):
            pivots = torch.zeros(
                (batch_size, self.pivot_width),
                dtype=torch.bfloat16,
                device=self.device,
            )
            bonus_tokens = torch.zeros(
                (batch_size,), dtype=torch.int64, device=self.device
            )
            block_ids = torch.full(
                (batch_size, self.draft_block_len),
                self.mask_token_id,
                dtype=torch.int64,
                device=self.device,
            )
            positions = self.local_positions.expand(batch_size, -1)
            candidates = torch.empty(
                (batch_size, self.block_size),
                dtype=torch.int32,
                device=self.device,
            )
            forward_batch = SimpleNamespace(
                batch_size=batch_size,
                spec_info=SimpleNamespace(hidden_states=pivots),
            )

            run_graph = partial(
                self._run_graph,
                batch_size=batch_size,
                bonus_tokens=bonus_tokens,
                block_ids=block_ids,
                positions=positions,
                candidates=candidates,
                forward_batch=forward_batch,
            )

            # One shape-specific warmup catches library workspace setup before
            # capture while keeping the steady-state path allocation-free.
            with torch.cuda.stream(capture_stream), torch.inference_mode():
                run_graph()
            torch.cuda.current_stream(self.device).wait_stream(capture_stream)
            torch.cuda.synchronize(self.device)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=capture_stream):
                run_graph()
            self._captured[batch_size] = CapturedDraftBatch(
                graph=graph,
                pivots=pivots,
                bonus_tokens=bonus_tokens,
                block_ids=block_ids,
                candidates=candidates,
            )
        torch.cuda.synchronize(self.device)

    def _run_graph(
        self,
        *,
        batch_size: int,
        bonus_tokens: torch.Tensor,
        block_ids: torch.Tensor,
        positions: torch.Tensor,
        candidates: torch.Tensor,
        forward_batch: SimpleNamespace,
    ) -> None:
        block_ids[:, 0].copy_(bonus_tokens)
        input_embeds = self.token_embedding(block_ids)
        output = self.draft_model(
            input_ids=block_ids.reshape(-1),
            positions=positions.reshape(-1),
            forward_batch=forward_batch,
            input_embeds=input_embeds.reshape(batch_size * self.draft_block_len, -1),
        )
        block_hidden = output.hidden_states.reshape(
            batch_size, self.draft_block_len, -1
        )
        proposals = self._sample_proposals(
            block_hidden=block_hidden, bonus_tokens=bonus_tokens
        )
        candidates[:, 0].copy_(bonus_tokens)
        candidates[:, 1:].copy_(proposals)

    def _run_once(self, batch_size: int) -> torch.Tensor:
        pivots = torch.zeros(
            (batch_size, self.pivot_width),
            dtype=torch.bfloat16,
            device=self.device,
        )
        bonus_tokens = torch.zeros((batch_size,), dtype=torch.int64, device=self.device)
        block_ids = torch.full(
            (batch_size, self.draft_block_len),
            self.mask_token_id,
            dtype=torch.int64,
            device=self.device,
        )
        block_ids[:, 0].copy_(bonus_tokens)
        input_embeds = self.token_embedding(block_ids)
        positions = self.local_positions.expand(batch_size, -1)
        forward_batch = SimpleNamespace(
            batch_size=batch_size,
            spec_info=SimpleNamespace(hidden_states=pivots),
        )
        output = self.draft_model(
            input_ids=block_ids.reshape(-1),
            positions=positions.reshape(-1),
            forward_batch=forward_batch,
            input_embeds=input_embeds.reshape(batch_size * self.draft_block_len, -1),
        )
        block_hidden = output.hidden_states.reshape(
            batch_size, self.draft_block_len, -1
        )
        return self._sample_proposals(
            block_hidden=block_hidden, bonus_tokens=bonus_tokens
        )

    def replay(
        self, *, pivots: torch.Tensor, bonus_tokens: torch.Tensor
    ) -> torch.Tensor:
        batch_size = int(pivots.shape[0])
        bucket_index = bisect.bisect_left(self.batch_sizes, batch_size)
        if bucket_index >= len(self.batch_sizes):
            raise ValueError(
                f"batch size {batch_size} exceeds captured maximum "
                f"{self.batch_sizes[-1]}"
            )
        padded_batch_size = self.batch_sizes[bucket_index]
        captured = self._captured[padded_batch_size]
        captured.pivots[:batch_size].copy_(pivots)
        captured.bonus_tokens[:batch_size].copy_(bonus_tokens)
        captured.graph.replay()
        if not self._replay_evidence_emitted:
            self._replay_evidence_emitted = True
            logger.info(
                "Standalone Draft CUDA graph replay observed: mode=DRAFT_VERIFY "
                "batch_size=%d padded_batch_size=%d",
                batch_size,
                padded_batch_size,
            )
        return captured.candidates[:batch_size]


def target_major(tasks: list[PendingDraftTask]) -> list[PendingDraftTask]:
    """Keep every target's result rows contiguous for Coordinator fanout."""
    return sorted(
        tasks,
        key=lambda pending: (
            pending.task.target_instance_id,
            pending.task.created_ns,
            pending.task.transfer_id,
        ),
    )


class FlashMTPDraftEngine:
    """One-GPU DT engine with the existing independent continuous batch queue."""

    # DT_DRAFT_SLOT_DECOUPLE (DFlash only) and the sequence of its next
    # per-batch scratch owner; class defaults keep __new__ fixtures legacy.
    _slot_decouple = False
    _dflash_batch_seq = 0
    # DT_DRAFT_CONTROL_LOOP (DFlash only): the scheduler wake word, or None
    # for the legacy socket polling and idle sleep.
    _wake = None
    # DT_DRAFT_INGRESS=native (DFlash only): rows come from the native READY
    # queue (design §5.1 DRAIN/FREEZE), not from mailbox takes.
    _native_ingress = False
    _ready_pending = ()
    # DT_DRAFT_SCHEDULER=v2 (DFlash only, D10): the GPU done stamp of the
    # last absorbed DONE record, and this pass's wait (profile stamps).
    _scheduler_v2 = False
    _dflash_last_done_ns = 0
    _scheduler_wait_ns = (0, 0)

    # Preserve the historical transport for partially constructed test and
    # diagnostic instances that intentionally bypass ``__init__``.
    candidate_transport = "mmap"
    # Per-round Coordinator confirmation of ring rounds (design §8.2).  The
    # class default is the legacy mode for instances that bypass
    # ``__init__``; ``__init__`` reads ``DT_RING_CONFIRMATION`` through
    # dt_protocol.  Confirmation-free (False), a ring round is born confirmed:
    # it is never held, deduped or expired against a Coordinator copy, sends
    # the Coordinator no result, and only a cross-host round sends a
    # SourceReadDone (the INPUT_READY word is the same-host read proof).
    _ring_confirmation = True

    def __init__(
        self,
        *,
        coordinator_endpoint: str,
        instance_id: str,
        profile_path: str,
        gpu_id: int,
        nccl_port: int,
        max_pending_tasks: int,
        max_batch_size: int,
        result_slot_count: int,
        data_dir: str,
        mem_fraction_static: float,
        registration_timeout_s: float,
        disable_cuda_graph: bool = False,
        enable_deterministic_inference: bool = False,
        batch_source_read_done: bool = False,
        pinned_pivot_staging: bool = False,
        migration_backend: str = "disabled",
        feature_buffer_max_tokens: int = 8192,
        feature_buffer_slot_count: int = 2,
        dflash_submit_pipeline: bool = False,
        continuous_batching: bool = True,
        candidate_transport: str = "mmap",
        dt_host_addr: str = "",
    ) -> None:
        try:
            from dt_client import DraftCoordinatorClient
            from dt_protocol import EntryStatus
            from dt_protocol.profile import load_draft_profile
            from dt_transport import MmapSlotStore, cleanup_stale_slot_files
        except ImportError as exc:
            raise RuntimeError(
                "Draft-Target mode requires dt-runtime on PYTHONPATH"
            ) from exc

        if max_pending_tasks <= 0 or max_batch_size <= 0:
            raise ValueError("queue and batch capacities must be positive")
        if result_slot_count <= 0:
            raise ValueError("result_slot_count must be positive")
        if migration_backend not in {"disabled", "cuda-ipc"}:
            raise ValueError("migration_backend must be 'disabled' or 'cuda-ipc'")
        if migration_backend == "cuda-ipc" and os.environ.get(
            "CUDA_MPS_PIPE_DIRECTORY"
        ):
            raise RuntimeError(
                "DT CUDA IPC requires the registered Draft process to own its "
                "CUDA context directly; CUDA MPS is unsupported"
            )
        if feature_buffer_max_tokens <= 0 or feature_buffer_slot_count <= 0:
            raise ValueError("feature buffer dimensions must be positive")
        if candidate_transport not in {"mmap", "cuda-ipc"}:
            raise ValueError("candidate_transport must be 'mmap' or 'cuda-ipc'")

        self.EntryStatus = EntryStatus
        self.instance_id = instance_id
        self._profile_active = profile_enabled()
        self._last_input_ready_profile_state = None
        self._batch_estimator = (
            DraftBatchEstimator()
            if profile_event_enabled("draft_batch_estimate")
            else None
        )
        # Absolute GPU end of each DFlash batch (dt_gpu_clock), profile only.
        self._gpu_clock = None
        self._gpu_clock_enabled = (
            self._profile_active and os.environ.get("DT_GPU_CLOCK", "0") == "1"
        )
        self._input_ready_select_ns = (0, 0)
        # Profile-only stage marks of the batch being selected and submitted
        # (issue #306): stage name -> monotonic ns at the stage's end, from
        # selection start; None with profiling off.  _submit_sub_ns times
        # the SUBMIT_SUB_STAGES parts nested inside a stage.
        self._submit_marks = None
        self._submit_sub_ns = None
        self.profile = load_draft_profile(profile_path)
        if self.profile.method not in {
            "flashmtp",
            "standalone",
            "dflash",
            "eagle3",
        }:
            raise ValueError(
                "DT Draft engine supports FlashMTP V1, STANDALONE V2, "
                "DFlash V2, and EAGLE3 V2"
            )
        self._is_dflash = self.profile.method == "dflash"
        self._is_eagle3 = self.profile.method == "eagle3"
        self._is_feature_backed = self._is_dflash or self._is_eagle3
        self._slot_decouple = DRAFT_SLOT_DECOUPLE and self._is_dflash
        self._persistent = self.profile.method in {
            "standalone",
            "dflash",
            "eagle3",
        }
        self.gpu_id = gpu_id
        self.max_pending_tasks = max_pending_tasks
        self.max_batch_size = max_batch_size
        self.continuous_batching = bool(continuous_batching)
        self.batch_source_read_done = batch_source_read_done
        self.pinned_pivot_staging = pinned_pivot_staging
        self.enable_deterministic_inference = bool(enable_deterministic_inference)
        self.migration_backend = migration_backend
        self.feature_buffer_max_tokens = int(feature_buffer_max_tokens)
        self.feature_buffer_slot_count = int(feature_buffer_slot_count)
        self.candidate_transport = candidate_transport
        # This Draft's own host address (--dt-host-addr).  A Target arena
        # registration whose host_addr is set and differs is cross-host: the
        # direct ring is skipped (the Target falls back to the Coordinator
        # relay) and the two data legs use the RDMA backend (dt_rdma).
        self._dt_host_addr = dt_host_addr
        self._remote_targets: dict[str, tuple[str, int]] = {}
        self.dflash_submit_pipeline = bool(
            dflash_submit_pipeline or (getattr(self, "continuous_batching", False) and self._is_dflash)
        )
        if dflash_submit_pipeline and not self._is_dflash:
            raise ValueError("DFLASH submit pipelining requires a DFlash profile")
        self._result_row_width = self.profile.block_size
        self._pending: deque[object] = deque()
        self._pending_ids: set[str] = set()
        self._ring_pending_ids: set[str] = set()
        self._received_ns_by_transfer: dict[str, int] = {}
        self._cancelled_ids: set[str] = set()
        self._persistent_requests: dict[str, PersistentDraftRequest] = {}
        # Native-ingress hot path: the same states indexed by ledger cell key
        # ((gen << 32) | cell) so READY-table rows are evaluated without
        # building their f-string handle.  Values are (state, attempt); a row
        # naming another attempt of the request is not this state's row (the
        # old string-key check misses it the same way).
        self._persistent_by_cell: dict[int, tuple[PersistentDraftRequest, int]] = {}
        self._migration_roles: dict[str, str] = {}
        self._migration_decisions: dict[str, bool] = {}
        self._migration_release_requests: dict[str, bool] = {}
        self._migration_cleanup_ids: dict[str, str] = {}
        self._migration_release_source_leases: dict[str, bool] = {}
        self._migration_release_feature_transfers: dict[str, str] = {}
        self._migration_release_feature_sources: dict[str, bool] = {}
        self._migration_destination_copy_quiescent: dict[str, bool] = {}
        self._migration_source_leases_released: dict[str, bool] = {}
        self._migration_feature_sources_released: dict[str, bool] = {}
        self._prepared_migrations: dict[str, PreparedDraftMigration] = {}
        self._deferred_migration_gpu_controls: dict[
            tuple[type, str], DeferredMigrationGpuControl
        ] = {}
        self._migration_source_events: dict[str, torch.cuda.Event] = {}
        self._migration_source_handles: dict[str, str] = {}
        self._deferred_persistent_releases: dict[str, str] = {}
        # The same deferred releases by ledger cell key, so the native
        # FREEZE answers its released check without building the row's
        # handle (a deferred release is outstanding for ~one batch lifetime
        # at serving churn, so the check runs on most freeze passes).
        self._deferred_release_cells: set[int] = set()
        self._pending_request_release_acks: dict[str, object] = {}
        self._deferred_migration_finalizations: dict[str, object] = {}
        self._deferred_feature_source_tasks: dict[tuple[str, str], object] = {}
        self._feature_source_inflight: set[tuple[str, str]] = set()
        self._feature_source_quiescent: set[tuple[str, str]] = set()
        self._feature_source_unquiescent: set[tuple[str, str]] = set()
        self._quarantined_feature_draft_handles: set[str] = set()
        self._migration_quiescence_probes: dict[str, Future[bool]] = {}
        self._migration_quiescence_retry_ns: dict[str, int] = {}
        self._p2p_transport = None
        self._kv_pool = None
        self._last_batch_size = 0
        self._last_queue_delay_ns = 0
        # Draft-boundness evidence (issue #174): GPU time of the last completed
        # DFlash proposal batch, read by the throttled stats line below.
        self._last_step_ns = 0
        self._last_stats_log_ns = 0
        # Loop-stage accounting (issue #222): cumulative scheduler-thread wall
        # time and call count per named loop stage, sampled into the throttled
        # stats line so a load point decomposes the Draft loop without
        # tracing.  Stage names are fixed at their call sites.
        self._loop_iters = 0
        self._loop_wall_ns = 0
        self._loop_stage_ns: dict[str, int] = defaultdict(int)
        self._loop_stage_calls: dict[str, int] = defaultdict(int)
        self._data_dir = Path(data_dir).absolute()
        self._published_by_batch: dict[str, PublishedResult] = {}
        self._published_by_range: dict[tuple[str, str, int, int, int, int], str] = {}
        self._delivery_outbox = RetryOutbox[object](
            retry_interval_ns=CONTROL_RETRY_INTERVAL_NS
        )
        self._source_done_pending: deque[object] = deque()
        self._source_done_batch_by_entry: dict[str, str] = {}
        self._source_done_batch_remaining: dict[str, set[str]] = {}
        self._pull_done_applied_ids: set[str] = set()
        self._free_result_slots = deque(range(result_slot_count))
        self.direct_proposal_admission_limit = (
            _resolve_direct_proposal_admission_limit()
        )
        # Credits held by published-but-incomplete cuda-ipc waves.  This is an
        # independent counter, not derived from any deque: every acquire must
        # be balanced by exactly one release (push completion, held-push drop,
        # or exception).
        self._direct_proposal_admission_inflight = 0
        self._result_sequence = 0
        self._result_pool_generation = uuid.uuid4().hex
        self._source_done_sequence = 0
        self._stop = False
        # Ordinary stop requests still run the bounded shutdown/drain path.
        # This latch is stronger: an unquiesced CUDA operation means stream
        # ordering and resource ownership can no longer be proven, so no later
        # GPU query, allocation, export, finalization, or release is allowed.
        self._fatal_gpu_quarantine_reason: str | None = None
        self._lifecycle_status_sequence = 0
        self._last_lifecycle_status_ns = 0
        self._last_lifecycle_status_sample_ns = 0
        self._last_lifecycle_quiescent: bool | None = None
        # One successful measured CUDA-IPC pull is sufficient to admit this
        # endpoint.  Avoid repeating synchronous evidence logging on every
        # Draft batch in the measured hot path.
        self._direct_gpu_measurement_evidence_emitted = False
        self._direct_gpu_measurement_epochs: dict[
            tuple[str, str, str, str], list[int]
        ] = {}
        self._direct_gpu_measurement_requests: dict[
            tuple[str, str, str, str, str], list[int]
        ] = {}
        self._last_direct_gpu_measurement_ns = 0
        self._graph_measurement_counts: dict[tuple[str, str], list[int]] = {}
        self._quarantined_migration_ids: set[str] = set()
        self._feature_pull_transport = None
        self._feature_staging_buffers: list[torch.Tensor] = []
        self._free_feature_staging_slots: deque[int] = deque()
        self._pending_feature_batches: deque[PendingFeatureBatch] = deque()
        self._input_ready_features: deque[InputReadyFeature] = deque()
        self._feature_staging_refcounts: dict[int, int] = {}
        self._feature_staging_compute_busy: set[int] = set()
        self._pending_dflash_completions: deque[PendingDFlashCompletion] = deque()
        self._quarantined_dflash_completions: list[PendingDFlashCompletion] = []
        self._pending_eagle3_completions: deque[PendingEagle3Completion] = deque()
        self._quarantined_eagle3_completions: list[PendingEagle3Completion] = []
        self._pending_flashmtp_completions: deque[PendingFlashMTPCompletion] = deque()
        self._quarantined_flashmtp_completions: list[
            PendingFlashMTPCompletion
        ] = []
        self._dflash_metadata_host_buffers: list[torch.Tensor] = []
        self._dflash_metadata_buffers: list[torch.Tensor] = []
        self._proposal_bonus_tokens_host_buffers: list[torch.Tensor] = []
        self._proposal_bonus_tokens_buffers: list[torch.Tensor] = []
        self._candidate_result_host_buffers: list[torch.Tensor] = []
        self._candidate_result_copy_stream = None
        self._candidate_push_transport = None
        self._pending_candidate_pushes: deque[PendingCandidatePush] = deque()
        # Retain each accepted direct destination across the small transient
        # gaps between READY queues and completion queues.  Cancellation can
        # arrive in any of those gaps and must still publish an authenticated
        # tombstone for the Target-owned generation.
        self._direct_tasks_by_transfer: dict[str, object] = {}
        # Transfers whose candidate push was queued at launch, behind the
        # proposal on the push stream (DT_GPU_READY_RECORD or
        # DT_PUSH_COMPLETION_RECORD).  Their completion handler must not push,
        # expire, or tombstone them again.
        self._early_pushed_ids: set[str] = set()
        # Same-host Target -> Draft descriptor rings (DT_DIRECT_RING); created
        # on the first candidate arena registration.
        self._ring_worker = None
        self._ring_endpoints: dict[str, object] = {}
        self._ring_targets: dict[int, str] = {}
        # Cross-host task-record rings (issue #182), keyed by Target session;
        # served by the same native ring worker as the local rings (#198).
        self._remote_ring_endpoints: dict[str, object] = {}
        # Sessions whose ring leg failed to connect: never retried (the
        # Coordinator relay remains their path).
        self._remote_ring_failed: set[str] = set()
        # Attached ring source registrations: ring_id -> (ipc_handle, length).
        self._ring_source_registration: dict[int, tuple[bytes, int]] = {}
        self._ring_admission: deque[tuple[object, int, tuple[int, int]]] = deque()
        self._ring_waiting: dict[str, object] = {}
        self._ring_task_decoder = None
        self._landing_slot_base = -1
        self._ring_landing = None
        self._landing_taken: list[int] = []
        # Landing frees waiting for the GPU work that last read the buffer.
        self._landing_free_pending: deque[tuple[int, torch.cuda.Event]] = deque()
        # Ring copies that must never execute: cancelled/released before the
        # copy arrived, or abandoned after its Coordinator copy timed out.
        self._ring_abandoned: set[str] = set()
        # Bounded record of dropped/expired ring rounds; their late
        # Coordinator copies get no second result.
        self._ring_retired: OrderedDict[str, bool] = OrderedDict()

        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", instance_id)
        self._result_store = None
        if self.candidate_transport == "mmap":
            cleanup_stale_slot_files(
                Path(data_dir), instance_id=safe_id, kind="results"
            )
            result_path = (
                Path(data_dir).absolute()
                / f"dt-{safe_id}-{os.getpid()}-results.slots"
            )
            self._result_store = MmapSlotStore.create(
                result_path,
                slot_count=result_slot_count,
                slot_bytes=max_batch_size * self._result_row_width * 4,
            )
        self._received_tasks = 0
        # Tasks received directly from their Target before the Coordinator's
        # confirming copy: transfer_id -> deadline_ns.  Their Coordinator-bound
        # retirement and source-read messages are held until confirmation.
        self._unconfirmed_direct: dict[str, int] = {}
        self._held_direct_results: dict[str, object] = {}
        self._held_direct_source_done: dict[str, list] = {}
        # §7.1 record waves whose candidate push waits for every task's
        # Coordinator confirmation: (tasks, candidates, result_slot).
        self._held_ring_pushes: list = []
        # §7.1 kernel-published records carry request-id hashes only; this
        # ledger maps them back.  Populated at bootstrap admission; only
        # needed (and built) when kernel publication is enabled.
        from sglang.srt.speculative.dt_direct_ring import (
            RING_PUBLISH,
            RING_PUBLISH_AT_LAUNCH,
            NativeRidLedger,
        )

        self._rid_ledger = NativeRidLedger() if RING_PUBLISH else None
        # Ring tombstones (zero destination triples for rows that ended in
        # their wave) exist only in launch-published records: park staging
        # runs after result processing and never publishes finished rows.
        # Gate the held-push tombstone scan on the flag so the default Draft
        # loop does no per-held-request shared-memory scanning.
        self._ring_publish_at_launch = RING_PUBLISH_AT_LAUNCH
        # Kernel-published record transfer -> (endpoint, request-table
        # position), for arena destination resolution from the ring region.
        self._ring_record_positions: dict = {}
        self._ring_dest_region = os.environ.get("DT_RING_DEST_REGION", "1") == "1"
        from dt_protocol import ring_confirmation_enabled

        self._ring_confirmation = ring_confirmation_enabled()
        if not self._ring_confirmation and not self._ring_dest_region:
            # Confirmation-free, the region is a v2 row's only destination.
            raise ValueError("DT_RING_CONFIRMATION=0 requires DT_RING_DEST_REGION=1")
        # Draft redesign D7: the READY-table consumer builds on D1 (no
        # compute-busy), D2 (no credit) and D4 (no confirmation) and has no
        # path for their legacy modes.
        from sglang.srt.speculative.dt_direct_ring import DRAFT_INGRESS_NATIVE

        self._native_ingress = DRAFT_INGRESS_NATIVE and self._is_dflash
        if self._native_ingress and (
            not self._slot_decouple
            or not DRAFT_ADMISSION_BOUND
            or self._ring_confirmation
        ):
            raise ValueError(
                "DT_DRAFT_INGRESS=native requires DT_DRAFT_SLOT_DECOUPLE=1, "
                "DT_DRAFT_ADMISSION=bound and DT_RING_CONFIRMATION=0"
            )
        # Draft redesign D8: the send loop extends the publish worker, hands
        # its DONE wake to the control loop's wake word, and produces the
        # landing releases the native ingress consumes.
        self._send_loop = DRAFT_SEND_LOOP and self._is_dflash
        if self._send_loop and not (
            self._native_ingress
            and DRAFT_CONTROL_LOOP
            and self.candidate_transport == "cuda-ipc"
        ):
            raise ValueError(
                "DT_DRAFT_SEND_LOOP=1 requires DT_DRAFT_INGRESS=native, "
                "DT_DRAFT_CONTROL_LOOP=1 and the cuda-ipc candidate transport"
            )
        # Draft redesign D10: the v2 scheduler gates on the send loop's
        # done_seq, waits on its DONE wake, and times its late freeze with
        # the G(n)/C(n) estimator (always on under the flag).
        self._scheduler_v2 = DRAFT_SCHEDULER_V2 and self._is_dflash
        if self._scheduler_v2:
            if not self._send_loop:
                raise ValueError("DT_DRAFT_SCHEDULER=v2 requires DT_DRAFT_SEND_LOOP=1")
            if self._batch_estimator is None:
                self._batch_estimator = DraftBatchEstimator()
        # The publish worker once it is the send loop (None until the ring
        # worker exists); the fence events of the release-only descriptors
        # still outstanding, in submit order, live in _ready_release_fenced.
        self._send_loop_worker = None
        # D9: the same worker as the source of PUSH_DONE records; unlike
        # _send_loop_worker it outlives the ring worker (shutdown drains
        # pushes after the rings close).
        self._push_done_worker = None
        # D9: rows handed to the send loop for a late or cross-host push, by
        # done seq: [tasks by descriptor row (None once retired), the batch's
        # candidates (the loop reads them), rows not yet retired, groups].
        self._send_loop_rows: dict[int, list] = {}
        # D9: send-loop pushes with cross-host rows whose WRITEs are posted
        # but not yet acknowledged (design §7.2 step 5: complete = push
        # event + RDMA CQ): [groups, issued_ns, candidates, tasks].
        self._send_loop_remote: list[list] = []
        # READY table (scheduler-owned, native ingress): entries by READY seq
        # modulo the queue capacity (created with the ring worker), the
        # unselected rows in arrival order (READY seq plus the row's ledger
        # cell key; the string identity is built lazily), the decoded
        # V1_PAYLOAD tasks (until Target item T4), per-record landing
        # placements
        # (ring id, record seq) -> [rows not yet released, k, gen], and
        # ready_consumed_seq with the consumed seqs past it (#276 decision 2:
        # reclaim per entry, the counter advances over the contiguous prefix).
        self._ready_table = None
        self._ready_pending: list[tuple] = []
        self._ready_tasks: dict[int, object] = {}
        self._ready_records: dict[tuple[int, int], list[int]] = {}
        self._ready_consumed = 0
        self._ready_done: set[int] = set()
        # Placements of rows dropped without a batch: (event, keys), released
        # once the current stream passed their copy events.
        self._ready_release_fenced: deque[tuple[torch.cuda.Event, tuple]] = deque()
        # READY residence (publish -> consumed): the longest one, and a
        # cumulative histogram of floor(4 * log2 ns) for its p99.
        self._ready_held_max_ns = 0
        self._ready_held_hist = np.zeros(256, dtype=np.int64)
        # FREEZE skip reasons (#292), maintained only under the profile
        # (production keeps _ready_drops/_ready_frozen_batches): per waiting
        # row (by READY seq) its [reason, last evaluation ns, gate mark,
        # passes, first reason, fit at the last fit skip, held ns per
        # reason]; cumulative counters for the lifecycle status; compute-depth
        # gate time while rows waited.
        self._ready_skip_state: dict[int, list] = {}
        reasons = len(_READY_SKIP_REASONS)
        self._ready_skips = [0] * reasons
        self._ready_skip_held_ns = [0] * reasons
        self._ready_long_held_ns = [0] * reasons
        self._ready_long_rows = [0] * reasons
        self._ready_longest: dict[str, object] | None = None
        self._ready_drops = [0] * len(_READY_DROP_OUTCOMES)
        self._ready_passes = 0
        self._ready_frozen_batches = 0
        self._ready_gate_ns = 0
        self._ready_gated_since_ns = 0
        # D10: the same for the v2 scheduler's late-freeze holds.
        self._ready_late_ns = 0
        self._ready_late_since_ns = 0
        # Held results released by confirmations, sent as one batch per turn.
        self._confirmed_held_results: list = []
        self._confirmed_held_sequence = 0
        # Bounded window of accepted transfer ids: a direct copy that arrives
        # after its Coordinator copy (even after retirement) must not re-run.
        self._seen_transfers: set[str] = set()
        self._seen_transfer_order: deque[str] = deque()
        # Pure cache of rid -> FNV hash for the per-row ring-record
        # cross-check; cleared when it grows past the seen-transfer window.
        self._rid_hash_cache: dict[str, int] = {}
        self._capacity_sequence = 0
        self._client = None
        try:
            self._client = DraftCoordinatorClient(
                endpoint=coordinator_endpoint,
                instance_id=instance_id,
                draft_profile_ids=(self.profile.profile_id,),
                max_pending_tasks=max_pending_tasks,
                data_endpoint=(
                    "" if self._result_store is None else self._result_store.uri
                ),
                result_pool_generation=self._result_pool_generation,
            )
            self._initialize_model(
                nccl_port=nccl_port,
                mem_fraction_static=mem_fraction_static,
                disable_cuda_graph=disable_cuda_graph,
            )
            self._initialize_direct_gpu_source_transport()
            if self.candidate_transport == "cuda-ipc":
                from sglang.srt.speculative.dt_candidate_p2p import (
                    DraftCandidatePushTransport,
                )

                self._candidate_push_transport = DraftCandidatePushTransport(
                    draft_instance_id=self.instance_id,
                    draft_session_id=self._client.instance_session_id,
                    device=self.device,
                    profile_timing=self._profile_active,
                    host_addr=self._dt_host_addr,
                )
            if (
                self.candidate_transport == "mmap"
                and getattr(self, "continuous_batching", False)
                and self.profile.method in {"eagle3", "standalone"}
            ):
                # EAGLE3 uses its feature slots; AR reserves an existing result
                # slot. Both retain pinned storage until the completion event.
                self._candidate_result_host_buffers = [
                    torch.empty(
                        (self.max_batch_size, self.profile.block_size),
                        dtype=torch.int32, device="cpu", pin_memory=True,
                    )
                    for _ in range(
                        self.feature_buffer_slot_count if self._is_eagle3 else result_slot_count
                    )
                ]
            self._initialize_migration_transport()
            self._initialize_pivot_staging()
            self._client.pool = self._kv_pool
            self._client.register_draft(timeout_s=registration_timeout_s)
            self._publish_capacity()
            if DRAFT_CONTROL_LOOP and self._is_dflash:
                from sglang.srt.speculative.dt_control import DraftControlLoop
                from sglang.srt.speculative.dt_wake import DraftWake

                self._wake = DraftWake()
                self._client = DraftControlLoop(self._client, self._wake)
        except Exception:
            if self._candidate_push_transport is not None:
                self._candidate_push_transport.close()
                self._candidate_push_transport = None
            if self._feature_pull_transport is not None:
                self._feature_pull_transport.close()
                self._feature_pull_transport = None
            if self._p2p_transport is not None:
                self._p2p_transport.close()
                self._p2p_transport = None
            if self._client is not None:
                self._client.close()
                self._client = None
            if self._result_store is not None:
                self._result_store.close()
                self._result_store.unlink()
            raise

    def _initialize_migration_transport(self) -> None:
        if self.migration_backend == "disabled":
            return
        if not self._persistent:
            raise RuntimeError("CUDA IPC migration requires a persistent Draft")
        self._require_migration_kv_layout()
        registration_started_ns = time.monotonic_ns()
        try:
            from sglang.srt.speculative.dt_kv_p2p import DraftKvP2pTransport

            self._p2p_transport = DraftKvP2pTransport(
                model_runner=self._draft_model_runner,
                device=self.device,
            )
            self._kv_pool = self._p2p_transport.pool_ref(
                instance_session_id=self._client.instance_session_id,
                profile_id=self.profile.profile_id,
            )
        except Exception as exc:
            raise RuntimeError(
                "explicitly requested CUDA IPC Draft KV migration is unavailable"
            ) from exc
        logger.info(
            "Registered complete Draft KV pool for CUDA IPC migration: "
            "generation=%s allocations=%d buffers=%d",
            self._kv_pool.pool_generation,
            len(self._kv_pool.allocations),
            len(self._kv_pool.buffers),
        )
        profile_emit(
            "migration_pool_registration",
            draft_instance=self.instance_id,
            pool_generation=self._kv_pool.pool_generation,
            allocation_count=len(self._kv_pool.allocations),
            buffer_count=len(self._kv_pool.buffers),
            registration_ns=time.monotonic_ns() - registration_started_ns,
        )

    def _initialize_pivot_staging(self) -> None:
        """Retain legacy attributes without allocating a host pivot data plane."""

        self._pivot_host_staging = None
        self._pivot_host_views = ()
        if self._persistent or not self.pinned_pivot_staging:
            return
        logger.info(
            "Ignoring legacy pinned FlashMTP pivot staging: direct CUDA IPC is required"
        )

    @property
    def _uses_direct_gpu_source(self) -> bool:
        return self._is_feature_backed or not self._persistent

    def _emit_direct_gpu_measurement_evidence_once(
        self, payload: dict[str, object]
    ) -> bool:
        """Log the first successful measured direct-GPU pull."""

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
        target_instance_id: str,
        source_session_id: str,
        pool_generation: str,
        transport_session: str,
        measurement_rows: int,
        measurement_requests: int,
        measurement_proposal_requests: int,
    ) -> None:
        epochs = getattr(self, "_direct_gpu_measurement_epochs", None)
        if epochs is None:
            epochs = {}
            self._direct_gpu_measurement_epochs = epochs
        key = (
            target_instance_id,
            source_session_id,
            pool_generation,
            transport_session,
        )
        counts = epochs.setdefault(key, [0, 0, 0, 0])
        counts[0] += 1
        counts[1] += measurement_rows
        counts[2] += measurement_requests
        counts[3] += measurement_proposal_requests

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
                "measurement_proposal_requests": counts[3],
            }
            for key, counts in sorted(epochs.items())
        ]

    def _record_direct_gpu_measurement_request(
        self,
        *,
        target_instance_id: str,
        source_session_id: str,
        pool_generation: str,
        transport_session: str,
        request_id: str,
        measurement_rows: int,
        measurement_requests: int,
        measurement_proposal_requests: int,
    ) -> None:
        requests = getattr(self, "_direct_gpu_measurement_requests", None)
        if requests is None:
            requests = {}
            self._direct_gpu_measurement_requests = requests
        key = (
            target_instance_id,
            source_session_id,
            pool_generation,
            transport_session,
            request_id,
        )
        counts = requests.setdefault(key, [0, 0, 0])
        counts[0] += measurement_rows
        counts[1] += measurement_requests
        counts[2] += measurement_proposal_requests
        self._last_direct_gpu_measurement_ns = time.monotonic_ns()

    def _direct_gpu_measurement_request_evidence(self) -> list[dict[str, object]]:
        requests = getattr(self, "_direct_gpu_measurement_requests", {})
        return [
            {
                "target_instance_id": key[0],
                "source_session_id": key[1],
                "pool_generation": key[2],
                "transport_session": key[3],
                "request_id": key[4],
                "measurement_rows": counts[0],
                "measurement_requests": counts[1],
                "measurement_proposal_requests": counts[2],
            }
            for key, counts in sorted(requests.items())
        ]

    def _initialize_direct_gpu_source_transport(self) -> None:
        """Allocate the Draft-local destination ring for direct Target GPU pulls."""

        if not self._uses_direct_gpu_source or self._feature_pull_transport is not None:
            return
        from sglang.srt.speculative.dt_feature_p2p import DraftFeaturePullTransport

        self._feature_pull_transport = DraftFeaturePullTransport(
            device=self.device,
            max_workers=self.max_batch_size,
            profile_timing=self._profile_active,
            host_addr=self._dt_host_addr,
        )
        # FlashMTP transfers exactly one pivot row per request. Unlike DFlash
        # feature ranges, its staging bound is the Draft batch size, not the
        # multi-token feature-buffer limit.
        rows = self.max_batch_size
        self._feature_staging_buffers = [
            torch.empty(
                (rows, self._pivot_width),
                dtype=torch.bfloat16,
                device=self.device,
            )
            for _ in range(self.feature_buffer_slot_count)
        ]
        self._free_feature_staging_slots.extend(range(self.feature_buffer_slot_count))
        if not self._persistent:
            self._initialize_flashmtp_bonus_token_staging()
            if self.candidate_transport == "mmap":
                self._candidate_result_host_buffers = [
                    torch.empty(
                        (self.max_batch_size, self.profile.block_size),
                        dtype=torch.int32,
                        device="cpu",
                        pin_memory=True,
                    )
                    for _ in range(self.feature_buffer_slot_count)
                ]
        logger.info(
            "DT FlashMTP direct GPU pivot transport ready: instance=%s "
            "transport=cuda-ipc slots=%d rows=%d width=%d",
            self.instance_id,
            self.feature_buffer_slot_count,
            rows,
            self._pivot_width,
        )

    def _initialize_flashmtp_bonus_token_staging(self) -> None:
        """Allocate one slot-owned pinned/Device bonus-token pair per pivot slot."""

        self._proposal_bonus_tokens_host_buffers = [
            torch.empty(
                self.max_batch_size,
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(self.feature_buffer_slot_count)
        ]
        self._proposal_bonus_tokens_buffers = [
            torch.empty(
                self.max_batch_size,
                dtype=torch.int64,
                device=self.device,
            )
            for _ in range(self.feature_buffer_slot_count)
        ]

    def _initialize_model(
        self,
        *,
        nccl_port: int,
        mem_fraction_static: float,
        disable_cuda_graph: bool,
    ) -> None:
        if self._is_dflash:
            self._initialize_dflash_model(
                nccl_port=nccl_port,
                mem_fraction_static=mem_fraction_static,
                disable_cuda_graph=disable_cuda_graph,
            )
            return
        if self._is_eagle3:
            self._initialize_eagle3_model(
                nccl_port=nccl_port,
                mem_fraction_static=mem_fraction_static,
                disable_cuda_graph=disable_cuda_graph,
            )
            return
        if self._persistent:
            self._initialize_standalone_model(
                nccl_port=nccl_port,
                mem_fraction_static=mem_fraction_static,
                disable_cuda_graph=disable_cuda_graph,
            )
            return
        server_args_kwargs = {
            # The remote process owns the DFlash checkpoint and its KV pool as
            # a primary worker. There is no co-located Target pool to inherit.
            "model_path": self.profile.draft_model_path,
            "tokenizer_path": self.profile.target_model_path,
            "skip_tokenizer_init": True,
            "trust_remote_code": True,
            "device": "cuda",
            "dtype": self.profile.draft_dtype,
            "tp_size": 1,
            "pp_size": 1,
            "mem_fraction_static": mem_fraction_static,
            "speculative_algorithm": "FLASHMTP",
            "speculative_draft_model_path": self.profile.draft_model_path,
            "speculative_num_draft_tokens": self.profile.block_size,
            "disable_overlap_schedule": True,
            "max_running_requests": self.max_batch_size,
        }
        if self.enable_deterministic_inference:
            server_args_kwargs["enable_deterministic_inference"] = True
        server_args = ServerArgs(**server_args_kwargs)
        get_context().set_server_args(server_args)
        target_model_config = ModelConfig.from_server_args(server_args)
        parallel_state = ParallelState.trivial(gpu_id=self.gpu_id)
        # An in-process speculative draft worker reuses the target worker's
        # process groups. This standalone engine has no target worker, so seed
        # the same trivial groups before constructing TpModelWorker with
        # is_draft_worker=True.
        torch.get_device_module(server_args.device).set_device(self.gpu_id)
        bootstrap.init_torch_distributed(
            server_args=server_args,
            model_config=target_model_config,
            device=server_args.device,
            ps=parallel_state,
            dist_port=nccl_port,
            is_draft_worker=False,
            local_omp_cpuid=None,
        )
        bundle = build_draft_tp_worker(
            server_args=server_args,
            gpu_id=self.gpu_id,
            ps=parallel_state,
            nccl_port=nccl_port,
            target_model_config=target_model_config,
            algo_label="DT-FLASHMTP",
        )
        self._draft_worker = bundle.draft_worker
        self._draft_model_runner = bundle.draft_model_runner
        self._draft_model = bundle.draft_model
        self.device = self._draft_model_runner.device
        self._draft_model.eval()

        flash_config = parse_flashmtp_draft_config(
            draft_hf_config=self._draft_model_runner.model_config.hf_config
        )
        self._draft_block_len = int(flash_config.draft_block_len)
        self._num_proposals = int(flash_config.num_proposed_tokens)
        self._mask_token_id = int(flash_config.mask_token_id)
        self._pivot_width = (
            len(flash_config.target_layer_ids)
            * self.profile.feature_schema.feature_width
        )
        self._local_positions = torch.arange(
            1,
            self._draft_block_len + 1,
            device=self.device,
            dtype=torch.int64,
        )
        self._token_embedding = self._load_target_embedding(
            self.profile.target_model_path
        )
        self._target_lm_head = (
            None
            if self._draft_model.markov_head is not None
            else self._load_target_lm_head(self.profile.target_model_path)
        )
        self._draft_cuda_graph = None
        if not disable_cuda_graph:
            capture_start = time.monotonic()
            graph_runner = StandaloneDraftCudaGraphRunner(
                draft_model=self._draft_model,
                token_embedding=self._token_embedding,
                device=self.device,
                pivot_width=self._pivot_width,
                draft_block_len=self._draft_block_len,
                block_size=self.profile.block_size,
                mask_token_id=self._mask_token_id,
                local_positions=self._local_positions,
                max_batch_size=self.max_batch_size,
                target_lm_head=self._target_lm_head,
            )
            graph_runner.capture()
            self._draft_cuda_graph = graph_runner
            logger.info(
                "Captured standalone Draft CUDA graphs: batch_sizes=%s elapsed=%.2fs",
                graph_runner.batch_sizes,
                time.monotonic() - capture_start,
            )
        logger.info(
            "DT Draft model ready: instance=%s profile=%s block=%d batch=%d "
            "embedding_shape=%s cuda_graph=%s",
            self.instance_id,
            self.profile.profile_id,
            self.profile.block_size,
            self.max_batch_size,
            tuple(self._token_embedding.weight.shape),
            self._draft_cuda_graph is not None,
        )
    def _initialize_dflash_model(
        self,
        *,
        nccl_port: int,
        mem_fraction_static: float,
        disable_cuda_graph: bool,
    ) -> None:
        from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config
        from sglang.srt.speculative.dflash_worker_v2 import (
            DFlashKvMaterializer,
            _DflashDraftSampler,
        )
        from sglang.srt.speculative.draft_worker_common import (
            build_block_pos_offsets,
            make_draft_sampler_capture_hook,
            make_draft_block_spec_info,
        )
        from sglang.srt.speculative.dt_feature_p2p import DraftFeaturePullTransport

        server_args_kwargs = {
            # Preserve the ordinary DFLASH target/draft ServerArgs contract so
            # ModelRunner captures the native block-width TARGET_VERIFY graph.
            # TpModelWorker still loads only the Draft checkpoint below.
            "model_path": self.profile.target_model_path,
            "speculative_algorithm": "DFLASH",
            "speculative_draft_model_path": self.profile.draft_model_path,
            "speculative_dflash_block_size": self.profile.block_size,
            "tokenizer_path": self.profile.target_model_path,
            "skip_tokenizer_init": True,
            "trust_remote_code": True,
            "device": "cuda",
            "dtype": self.profile.draft_dtype,
            "tp_size": 1,
            "pp_size": 1,
            "mem_fraction_static": mem_fraction_static,
            "page_size": 1,
            "speculative_num_draft_tokens": self.profile.block_size,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "disable_cuda_graph": disable_cuda_graph,
            "cuda_graph_config": (
                {
                    "decode": {"backend": "disabled"},
                    "prefill": {"backend": "disabled"},
                }
                if disable_cuda_graph
                else {
                    "decode": {
                        "backend": "full",
                        "max_bs": self.max_batch_size,
                    },
                    "prefill": {"backend": "disabled"},
                }
            ),
            "max_running_requests": self.max_pending_tasks,
        }
        if self.enable_deterministic_inference:
            server_args_kwargs["enable_deterministic_inference"] = True
        server_args = ServerArgs(**server_args_kwargs)
        get_context().set_server_args(server_args)
        target_model_config = ModelConfig.from_server_args(server_args)
        parallel_state = ParallelState.trivial(gpu_id=self.gpu_id)
        torch.get_device_module(server_args.device).set_device(self.gpu_id)
        # A co-located speculative child inherits the Target's initialized
        # groups. This standalone process has no Target worker, so establish
        # the equivalent trivial groups before constructing the Draft child.
        bootstrap.init_torch_distributed(
            server_args=server_args,
            model_config=target_model_config,
            device=server_args.device,
            ps=parallel_state,
            dist_port=nccl_port,
            is_draft_worker=False,
            local_omp_cpuid=None,
        )
        self._draft_worker = TpModelWorker(
            server_args=server_args,
            gpu_id=self.gpu_id,
            ps=parallel_state,
            nccl_port=nccl_port,
            is_draft_worker=True,
        )
        self._draft_model_runner = self._draft_worker.model_runner
        self._draft_model = self._draft_model_runner.model
        self.device = self._draft_model_runner.device
        self._draft_model.eval()
        # A normal speculative child receives its Target's resolved pool
        # dimensions. The remote Draft owns an independent KV pool, so resolve
        # the same config from its own post-load memory budget first.
        self._draft_model_runner.init_kv_cache_configurator()
        draft_pool_config = (
            self._draft_model_runner.kv_cache_configurator._resolve_memory_pool_config(
                self._draft_model_runner.pre_model_load_memory
            )
        )
        self._draft_worker.alloc_memory_pool(memory_pool_config=draft_pool_config)
        self.max_pending_tasks = min(
            self.max_pending_tasks,
            int(self._draft_model_runner.max_running_requests),
        )
        self._client.max_pending_tasks = self.max_pending_tasks
        self._draft_worker.init_attention_backends()

        config = parse_dflash_draft_config(
            draft_hf_config=self._draft_model_runner.model_config.hf_config
        )
        self._mask_token_id = int(config.mask_token_id)
        self._pivot_width = int(self._draft_model.fc.in_features)
        if self._pivot_width != (
            self.profile.feature_schema.num_feature_layers
            * self.profile.feature_schema.feature_width
        ):
            raise ValueError(
                "DFLASH profile feature width does not match the checkpoint"
            )
        self._block_pos_offsets = build_block_pos_offsets(
            length=self.profile.block_size, device=self.device
        )
        self._draft_block_spec_info = make_draft_block_spec_info(
            draft_token_num=self.profile.block_size, device=self.device
        )
        self._token_embedding = self._load_target_embedding(
            self.profile.target_model_path
        )
        self._target_lm_head = self._load_target_lm_head(self.profile.target_model_path)
        self._dflash_sampler = _DflashDraftSampler(
            weight=self._target_lm_head.weight,
            block_size=self.profile.block_size,
            num_org=self.profile.vocab_size,
            org_vocab_start=0,
            max_bs=self.max_batch_size,
            sample_from_anchor=config.sample_from_anchor,
        )
        self._dflash_materializer = DFlashKvMaterializer(
            draft_model_runner=self._draft_model_runner,
            draft_model=self._draft_model,
            ps=parallel_state,
            block_size=self.profile.block_size,
            context_len=self._draft_model_runner.model_config.context_len,
        )
        self._tree_cache = ChunkCache(
            CacheInitParams(
                disable=True,
                req_to_token_pool=self._draft_model_runner.req_to_token_pool,
                token_to_kv_pool_allocator=(
                    self._draft_model_runner.token_to_kv_pool_allocator
                ),
                page_size=1,
            )
        )
        self._feature_pull_transport = DraftFeaturePullTransport(
            device=self.device,
            max_workers=self.max_batch_size,
            profile_timing=self._profile_active,
            host_addr=self._dt_host_addr,
        )
        self._feature_staging_buffers = [
            torch.empty(
                (self.feature_buffer_max_tokens, self._pivot_width),
                dtype=torch.bfloat16,
                device=self.device,
            )
            for _ in range(self.feature_buffer_slot_count)
        ]
        self._free_feature_staging_slots.extend(range(self.feature_buffer_slot_count))
        # Proposal metadata and pinned candidate rows: one set per staging
        # slot, or per in-flight batch under DT_DRAFT_SLOT_DECOUPLE.
        scratch_count = (
            DRAFT_COMPUTE_DEPTH
            if self._slot_decouple
            else self.feature_buffer_slot_count
        )
        self._dflash_metadata_host_buffers = [
            torch.empty(
                (_DFLASH_META_FIELDS, self.max_batch_size),
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(scratch_count)
        ]
        # Landing-row gather indices for an all-native-ring batch: pinned so
        # the H2D is asynchronous; per scratch slot so a batch's copy is done
        # before the slot's next batch rewrites it (the metadata host buffer
        # above has the same lifetime).
        self._dflash_gather_index_host_buffers = [
            torch.empty(
                (self.feature_buffer_max_tokens,),
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(scratch_count)
        ]
        self._dflash_metadata_buffers = [
            torch.empty(
                (_DFLASH_META_FIELDS, self.max_batch_size),
                dtype=torch.int64,
                device=self.device,
            )
            for _ in range(scratch_count)
        ]
        if self.dflash_submit_pipeline:
            self._candidate_result_host_buffers = [
                torch.empty(
                    (self.max_batch_size, self.profile.block_size),
                    dtype=torch.int32,
                    device="cpu",
                    pin_memory=True,
                )
                for _ in range(scratch_count)
            ]
        if self.dflash_submit_pipeline:
            self._candidate_result_copy_stream = torch.cuda.Stream(device=self.device)
        if not disable_cuda_graph:
            self._draft_model_runner.capture_tail_hooks.append(
                make_draft_sampler_capture_hook(self._dflash_sampler)
            )
        self._draft_worker.init_cuda_graphs(
            capture_decode_cuda_graph=not disable_cuda_graph
        )
        self._draft_cuda_graph = self._draft_model_runner.decode_cuda_graph_runner
        if not disable_cuda_graph and self._is_dflash and self._uses_direct_gpu_source:
            self._capture_dflash_materializer_graphs()
        logger.info(
            "DT DFLASH Draft ready: instance=%s block=%d feature_width=%d "
            "feature_slots=%d feature_tokens=%d cuda_graph=%s submit_pipeline=%s",
            self.instance_id,
            self.profile.block_size,
            self._pivot_width,
            self.feature_buffer_slot_count,
            self.feature_buffer_max_tokens,
            self._draft_cuda_graph is not None,
            self.dflash_submit_pipeline,
        )

    def _capture_dflash_materializer_graphs(self) -> None:
        """Replay the target-hidden -> draft-KV writer as one CUDA graph per bucket.

        The eager writer launches a projection, the fused norm/RoPE kernel and
        one KV write per layer from Python on every Draft batch
        (~0.5 ms of scheduler-thread launch overhead per batch).  The graphed
        writer copies the three inputs into static buffers and replays a
        captured graph instead; rows beyond the largest bucket (capped well
        past the measured p99 materialization size) keep the eager writer.
        """

        materializer = self._dflash_materializer
        helper = getattr(materializer, "_fused_kv_helper", None)
        if not getattr(materializer, "_use_fused_kv_materialize", False) or (
            helper is None
        ):
            logger.info("DT DFLASH materializer graphs skipped: fused writer off")
            return
        staging_dtype = self._feature_staging_buffers[0].dtype
        # Captured graphs bake the helper's workspace addresses; grow it once
        # to the largest possible materialization batch so a later eager
        # fallback (rows beyond the top bucket) can never reallocate the
        # workspace under the captured graphs.
        helper._ensure_workspace(self.feature_buffer_max_tokens, staging_dtype)
        top_bucket = min(self.feature_buffer_max_tokens, self.max_batch_size * 2)
        self._dflash_materializer = _GraphedDFlashMaterializer(
            materializer,
            width=self._pivot_width,
            dtype=staging_dtype,
            device=self.device,
            buckets=draft_cuda_graph_batch_sizes(top_bucket),
        )
        self._dflash_materializer.capture()
        logger.info(
            "DT DFLASH materializer graphs captured: buckets=%d top=%d",
            len(self._dflash_materializer._graphs),
            top_bucket,
        )

    def _initialize_standalone_model(
        self,
        *,
        nccl_port: int,
        mem_fraction_static: float,
        disable_cuda_graph: bool,
    ) -> None:
        # The external Draft is an independent causal-LM worker, not an
        # in-process speculative sub-worker.  Giving it the Draft checkpoint
        # as its primary model lets the normal worker path size and own its KV
        # pools; proposal construction is driven explicitly below.
        server_args_kwargs = {
            "model_path": self.profile.draft_model_path,
            "tokenizer_path": self.profile.target_model_path,
            "skip_tokenizer_init": True,
            "trust_remote_code": True,
            "device": "cuda",
            "dtype": self.profile.draft_dtype,
            "tp_size": 1,
            "pp_size": 1,
            "mem_fraction_static": mem_fraction_static,
            "page_size": 1,
            "disable_overlap_schedule": True,
            # The persistent Draft builds ordinary SGLang ScheduleBatch inputs,
            # so its extend/decode forwards can reuse the production graph
            # runners. Freeze the H800 production backends and bounds; SGLang
            # resolves the exact bucket lists from this source revision.
            "disable_cuda_graph": disable_cuda_graph,
            "cuda_graph_config": {
                "decode": {
                    "backend": "full",
                    "max_bs": self.max_batch_size,
                    "bs": list(
                        standalone_draft_decode_graph_batch_sizes(self.max_batch_size)
                    ),
                },
                "prefill": {
                    "backend": "breakable",
                    "max_bs": 8192,
                    "bs": list(STANDALONE_DRAFT_PREFILL_GRAPH_BATCH_SIZES),
                },
            },
            "max_running_requests": self.max_pending_tasks,
        }
        if self.enable_deterministic_inference:
            server_args_kwargs["enable_deterministic_inference"] = True
        server_args = ServerArgs(**server_args_kwargs)
        get_context().set_server_args(server_args)
        parallel_state = ParallelState.trivial(gpu_id=self.gpu_id)
        torch.get_device_module(server_args.device).set_device(self.gpu_id)
        self._draft_worker = TpModelWorker(
            server_args=server_args,
            gpu_id=self.gpu_id,
            ps=parallel_state,
            nccl_port=nccl_port,
            is_draft_worker=False,
        )
        self._draft_model_runner = self._draft_worker.model_runner
        self._draft_model = self._draft_model_runner.model
        self.device = self._draft_model_runner.device
        self._draft_worker.alloc_memory_pool()
        self.max_pending_tasks = min(
            self.max_pending_tasks,
            int(self._draft_model_runner.max_running_requests),
        )
        self._client.max_pending_tasks = self.max_pending_tasks
        self._draft_worker.init_attention_backends()
        # This installs the eager runner objects when graphs are disabled.
        self._draft_worker.init_cuda_graphs()
        if (
            int(self._draft_model_runner.model_config.vocab_size)
            != self.profile.vocab_size
        ):
            raise ValueError(
                "STANDALONE Draft vocabulary does not match the configured profile"
            )
        self._tree_cache = ChunkCache(
            CacheInitParams(
                disable=True,
                req_to_token_pool=self._draft_model_runner.req_to_token_pool,
                token_to_kv_pool_allocator=(
                    self._draft_model_runner.token_to_kv_pool_allocator
                ),
                page_size=self._draft_model_runner.server_args.page_size,
            )
        )
        self._pivot_width = 0
        self._draft_cuda_graph = None
        logger.info(
            "DT STANDALONE Draft ready: instance=%s profile=%s block=%d "
            "max_requests=%d cuda_graph=%s resolved_cuda_graph=%s",
            self.instance_id,
            self.profile.profile_id,
            self.profile.block_size,
            self.max_pending_tasks,
            not disable_cuda_graph,
            server_args.cuda_graph_config,
        )

    def _initialize_eagle3_model(
        self,
        *,
        nccl_port: int,
        mem_fraction_static: float,
        disable_cuda_graph: bool,
    ) -> None:
        """Initialize a Target-free TP1 EAGLE3 worker with persistent Draft KV."""

        from sglang.srt.speculative.dt_feature_p2p import DraftFeaturePullTransport
        from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker

        if self.profile.draft_k != self.profile.block_size - 1:
            raise ValueError("EAGLE3 draft_k must equal block_size - 1")
        server_args = ServerArgs(
            model_path=self.profile.target_model_path,
            speculative_algorithm="EAGLE3",
            speculative_draft_model_path=self.profile.draft_model_path,
            speculative_num_steps=self.profile.draft_k,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=self.profile.block_size,
            speculative_draft_attention_backend=(
                self.profile.draft_attention_backend
            ),
            tokenizer_path=self.profile.target_model_path,
            skip_tokenizer_init=True,
            trust_remote_code=True,
            device="cuda",
            dtype=self.profile.draft_dtype,
            tp_size=1,
            pp_size=1,
            mem_fraction_static=mem_fraction_static,
            page_size=1,
            disable_overlap_schedule=True,
            disable_radix_cache=True,
            disable_cuda_graph=disable_cuda_graph,
            cuda_graph_config=(
                {
                    "decode": {"backend": "disabled"},
                    "prefill": {"backend": "disabled"},
                }
                if disable_cuda_graph
                else {
                    "decode": {
                        "backend": "full",
                        "max_bs": self.max_batch_size,
                    },
                    "prefill": {
                        "backend": "breakable",
                        "max_bs": 8192,
                        "bs": list(STANDALONE_DRAFT_PREFILL_GRAPH_BATCH_SIZES),
                    },
                }
            ),
            max_running_requests=self.max_pending_tasks,
            enable_deterministic_inference=self.enable_deterministic_inference,
        )
        get_context().set_server_args(server_args)
        target_model_config = ModelConfig.from_server_args(server_args)
        parallel_state = ParallelState.trivial(gpu_id=self.gpu_id)
        torch.get_device_module(server_args.device).set_device(self.gpu_id)
        bootstrap.init_torch_distributed(
            server_args=server_args,
            model_config=target_model_config,
            device=server_args.device,
            ps=parallel_state,
            dist_port=nccl_port,
            is_draft_worker=False,
            local_omp_cpuid=None,
        )

        self.device = torch.device("cuda", self.gpu_id)
        target_embedding = self._load_target_embedding(
            self.profile.target_model_path
        ).weight

        class _TargetWeights:
            def get_embed_and_head(self):
                return target_embedding, None

        target_proxy = SimpleNamespace(
            device=server_args.device,
            model_config=target_model_config,
            model_runner=SimpleNamespace(
                model_config=target_model_config,
                model=_TargetWeights(),
            ),
        )
        self._eagle_draft_worker = EagleDraftWorker(
            server_args,
            self.gpu_id,
            parallel_state,
            nccl_port,
            target_proxy,
        )
        self._draft_worker = self._eagle_draft_worker.draft_worker
        self._draft_model_runner = self._eagle_draft_worker.draft_runner
        self._draft_model = self._draft_model_runner.model
        self.device = self._draft_model_runner.device
        self._draft_model.eval()
        self._draft_model_runner.init_kv_cache_configurator()
        draft_pool_config = (
            self._draft_model_runner.kv_cache_configurator._resolve_memory_pool_config(
                self._draft_model_runner.pre_model_load_memory
            )
        )
        self._eagle_draft_worker.alloc_memory_pool(
            memory_pool_config=draft_pool_config
        )
        self.max_pending_tasks = min(
            self.max_pending_tasks,
            int(self._draft_model_runner.max_running_requests),
        )
        self._client.max_pending_tasks = self.max_pending_tasks
        self._eagle_draft_worker.init_attention_backends()
        self._eagle_draft_worker.init_cuda_graphs()

        self._pivot_width = (
            self.profile.feature_schema.num_feature_layers
            * self.profile.feature_schema.feature_width
        )
        draft_fc = getattr(self._draft_model, "fc", None)
        if draft_fc is not None and int(draft_fc.in_features) != self._pivot_width:
            raise ValueError(
                "EAGLE3 profile feature width does not match the checkpoint"
            )
        self._tree_cache = ChunkCache(
            CacheInitParams(
                disable=True,
                req_to_token_pool=self._draft_model_runner.req_to_token_pool,
                token_to_kv_pool_allocator=(
                    self._draft_model_runner.token_to_kv_pool_allocator
                ),
                page_size=1,
            )
        )
        self._feature_pull_transport = DraftFeaturePullTransport(
            device=self.device,
            max_workers=self.max_batch_size,
            profile_timing=self._profile_active,
            host_addr=self._dt_host_addr,
        )
        self._feature_staging_buffers = [
            torch.empty(
                (self.feature_buffer_max_tokens, self._pivot_width),
                dtype=torch.bfloat16,
                device=self.device,
            )
            for _ in range(self.feature_buffer_slot_count)
        ]
        self._free_feature_staging_slots.extend(range(self.feature_buffer_slot_count))
        self._draft_cuda_graph = self._eagle_draft_worker.cuda_graph_runner
        logger.info(
            "DT EAGLE3 Draft ready: instance=%s profile=%s steps=%d "
            "feature_width=%d max_requests=%d cuda_graph=%s",
            self.instance_id,
            self.profile.profile_id,
            self.profile.draft_k,
            self._pivot_width,
            self.max_pending_tasks,
            self._draft_cuda_graph is not None,
        )

    def _load_target_embedding(self, model_path: str) -> torch.nn.Embedding:
        from safetensors import safe_open

        root = Path(model_path)
        index_path = root / "model.safetensors.index.json"
        tensor_name = "model.embed_tokens.weight"
        if index_path.exists():
            with index_path.open() as file:
                weight_map = json.load(file)["weight_map"]
            shard_name = weight_map.get(tensor_name)
            if shard_name is None:
                raise KeyError(f"target checkpoint has no {tensor_name!r}")
            tensor_path = root / shard_name
        else:
            tensor_path = root / "model.safetensors"
        with safe_open(tensor_path, framework="pt", device="cpu") as file:
            weight = file.get_tensor(tensor_name)
        if weight.dtype != torch.bfloat16:
            weight = weight.to(torch.bfloat16)
        embedding = torch.nn.Embedding.from_pretrained(weight, freeze=True)
        return embedding.to(self.device)

    def _load_target_lm_head(self, model_path: str) -> torch.nn.Linear:
        from safetensors import safe_open

        root = Path(model_path)
        index_path = root / "model.safetensors.index.json"
        tensor_name = "lm_head.weight"
        if index_path.exists():
            with index_path.open() as file:
                weight_map = json.load(file)["weight_map"]
            shard_name = weight_map.get(tensor_name)
            if shard_name is None:
                raise KeyError(f"target checkpoint has no {tensor_name!r}")
            tensor_path = root / shard_name
        else:
            tensor_path = root / "model.safetensors"
        with safe_open(tensor_path, framework="pt", device="cpu") as file:
            weight = file.get_tensor(tensor_name)
        if weight.dtype != torch.bfloat16:
            weight = weight.to(torch.bfloat16)
        head = torch.nn.Linear(
            weight.shape[1], weight.shape[0], bias=False, device="meta"
        )
        head.weight = torch.nn.Parameter(weight, requires_grad=False)
        return head.to(self.device)

    def _loop_stage(self, name: str, fn, *args):
        """Time one scheduler-loop stage into the cumulative stats buckets."""
        started_ns = time.monotonic_ns()
        result = fn(*args)
        self._loop_stage_ns[name] += time.monotonic_ns() - started_ns
        self._loop_stage_calls[name] += 1
        return result

    def _push_transport_progress(self) -> None:
        transport = self._candidate_push_transport
        transport.progress_completions()
        if self._profile_active:
            for acknowledgement in transport.pop_acknowledgements():
                profile_emit(
                    "draft_candidate_notice_ack",
                    draft_instance=self.instance_id,
                    **acknowledgement,
                )

    def run(self) -> None:
        logger.info("DT Draft engine serving")
        if self._scheduler_v2:
            self._run_scheduler_v2()
            return
        observer = None
        if os.environ.get("DT_OVERLAP_FIX_OBSERVER_DIR"):
            from overlap_observer import create

            observer = create(self.instance_id, "draft", 0, 1)
        self._overlap_fix_observer = observer
        wake = self._wake
        while not self._stop:
            iter_started_ns = time.monotonic_ns()
            if observer is not None:
                observer.checkpoint()
            # Read before draining: a set after this read ends the idle wait.
            wake_epoch = wake.epoch() if wake is not None else 0
            self._loop_stage("poll_messages", self._poll_messages)
            if self._send_loop:
                # ABSORB before the drain and the gate (design §5.1): a batch
                # the send loop finished stops counting here, not after this
                # turn's submit.
                self._absorb_dflash_done()
                # D9: push completions arrive as PUSH_DONE records, so no
                # push event is polled on this path.
                self._absorb_push_done()
            if self._candidate_push_transport is not None:
                self._loop_stage("push_progress", self._push_transport_progress)
            if self._stop:
                break
            # Native ingress drains READY once, right before the freeze.
            rings = not self._native_ingress and (
                self._ring_worker is not None or self._remote_ring_endpoints
            )
            if rings:
                self._loop_stage("poll_rings_head", self._poll_direct_rings)
            if self._ring_confirmation:
                self._loop_stage("flush_confirmed", self._flush_confirmed_results)
            if self._uses_direct_gpu_source:
                self._loop_stage("poll_feature_batches", self._poll_feature_batches)
                if self._stop:
                    break
                # Input preparation is independent of final compute
                # membership. Fill at most one free bounded arena per turn.
                preparation = self._loop_stage("take_batch", self._take_batch)
                if preparation:
                    self._loop_stage(
                        "execute_batch", self._execute_batch, preparation
                    )
                    # A pull that completed during the take/submit above is
                    # promoted here, at the next point the loop already
                    # passes, instead of after the tail and the next
                    # iteration's head (issue #195). Observing completion
                    # late leaves its admission credit reserved and, under
                    # load, costs a full busy-stretch of this loop thread.
                    self._loop_stage(
                        "poll_feature_batches", self._poll_feature_batches
                    )
                if self._native_ingress:
                    self._loop_stage("ready_drain", self._drain_ready)
                elif rings:
                    self._loop_stage("poll_rings_mid", self._poll_direct_rings)
                ready = self._loop_stage(
                    "take_input_ready", self._take_input_ready_batch
                )
                if ready:
                    self._loop_stage(
                        "host_submit", self._execute_input_ready_batch, ready
                    )
                    # The host submit above is the longest stage of this
                    # loop; a pull that completed during it promotes here
                    # rather than waiting out the tail and the next head.
                    self._loop_stage(
                        "poll_feature_batches", self._poll_feature_batches
                    )
            elif self._draft_compute_batches_inflight() < DRAFT_COMPUTE_DEPTH:
                batch = self._loop_stage("take_batch", self._take_batch)
                if batch:
                    self._loop_stage("execute_batch", self._execute_batch, batch)

            # Publish/retire old results after attempting the next READY
            # submission. This bounded pass on every turn also prevents an
            # endless input stream from starving retirement.
            if getattr(self, "_pending_eagle3_completions", ()):
                self._loop_stage("poll_eagle3", self._poll_eagle3_completions)
            if self._uses_direct_gpu_source:
                self._loop_stage("poll_flashmtp", self._poll_flashmtp_completions)
                if self._is_feature_backed and not self._send_loop:
                    # Split internally into dflash_query/publish/retire.
                    self._poll_dflash_completions()
            if self._pending_candidate_pushes and self._push_done_worker is None:
                self._loop_stage("poll_candidate_pushes", self._poll_candidate_pushes)
            if rings:
                self._loop_stage("poll_rings_tail", self._poll_direct_rings)
            if self._held_ring_pushes and not self._ring_confirmation:
                self._loop_stage(
                    "resolve_held_pushes", self._resolve_held_ring_destinations
                )
            elif self._held_ring_pushes and self._ring_publish_at_launch:
                self._loop_stage(
                    "retire_tombstoned", self._retire_tombstoned_ring_pushes
                )

            if (
                not self._pending
                and not self._input_ready_features
                and not self._pending_feature_batches
                and not self._ring_admission
                and not self._ready_pending
            ):
                if (
                    (self._pending_dflash_completions and not self._send_loop)
                    or self._pending_eagle3_completions
                    or self._pending_flashmtp_completions
                    or (self._pending_candidate_pushes and self._push_done_worker is None)
                ):
                    # A completion or direct candidate push is still in flight
                    # on the GPU; observe it soon instead of sleeping the full
                    # idle poll (issue #149).  The send loop's DONE wake ends
                    # the wait instead, so its batches and (D9) its pushes
                    # are not polled (§4.4).
                    idle_s = INFLIGHT_POLL_S
                else:
                    idle_s = IDLE_POLL_S
                if wake is None:
                    self._loop_stage("idle_sleep", time.sleep, idle_s)
                else:
                    self._loop_stage("idle_sleep", wake.wait, wake_epoch, idle_s)
            self._loop_stage("lifecycle", self._maybe_emit_lifecycle_status)
            self._loop_wall_ns += time.monotonic_ns() - iter_started_ns
            self._loop_iters += 1

    def _run_scheduler_v2(self) -> None:
        """The DFlash scheduler state machine (design §5.1, D10).

        One pass is ABSORB -> DRAIN -> GATE -> [LATE] -> FREEZE -> CPU WORK
        -> LAUNCH -> HAND-OFF, then WAIT:

        - ABSORB: the send loop's DONE and PUSH_DONE records, the control
          queue, and round-0 relayed pulls (run()'s mechanics);
        - DRAIN: the native READY queue into the READY table;
        - GATE/LATE: ``_dflash_gate``;
        - FREEZE .. HAND-OFF: the same take and submit as run(); the batch
          bound (§5.3) is the FREEZE fit of D2/D7;
        - WAIT: on the wake word (GIL released) for a READY, DONE or CONTROL
          bit, the late-freeze deadline, or the poll cap; never after a
          launch, since the next batch may freeze at once.  The cap is
          run()'s idle poll, shortened for work nothing wakes (relayed
          pulls, cross-host CQs, held pushes).
        """

        if not self._uses_direct_gpu_source:
            raise RuntimeError("DT_DRAFT_SCHEDULER=v2 requires the direct GPU source")
        wake = self._wake
        while not self._stop:
            iter_started_ns = time.monotonic_ns()
            # Read before draining: a set after this read ends the wait.
            wake_epoch = wake.epoch()
            # ABSORB in §5.1's order: the DONE records first (the -1).
            self._absorb_dflash_done()
            self._absorb_push_done()
            self._loop_stage("poll_messages", self._poll_messages)
            if self._candidate_push_transport is not None:
                self._loop_stage("push_progress", self._push_transport_progress)
            if self._stop:
                break
            self._loop_stage("poll_feature_batches", self._poll_feature_batches)
            if self._stop:
                break
            preparation = self._loop_stage("take_batch", self._take_batch)
            if preparation:
                self._loop_stage("execute_batch", self._execute_batch, preparation)
                self._loop_stage("poll_feature_batches", self._poll_feature_batches)
            self._loop_stage("ready_drain", self._drain_ready)
            if self._held_ring_pushes:
                self._loop_stage(
                    "resolve_held_pushes", self._resolve_held_ring_destinations
                )
            now_ns = time.monotonic_ns()
            deadline_ns = self._loop_stage("gate", self._dflash_gate, now_ns)
            if deadline_ns == 0:
                launched, frozen_ns = self._dflash_batch_seq, time.monotonic_ns()
                ready = self._loop_stage("take_input_ready", self._take_input_ready_batch)
                if ready:
                    self._loop_stage("host_submit", self._execute_input_ready_batch, ready)
                if self._dflash_batch_seq != launched:
                    completion = self._pending_dflash_completions[-1]
                    if not completion.handoff_ns:  # profiling stamps it itself
                        # C(n) is freeze -> hand-off end (§5.2).
                        completion.select_started_ns = frozen_ns
                        completion.handoff_ns = time.monotonic_ns()
                    self._loop_stage("lifecycle", self._maybe_emit_lifecycle_status)
                    self._loop_wall_ns += time.monotonic_ns() - iter_started_ns
                    self._loop_iters += 1
                    continue
            elif self._ready_pending:
                # The rows wait out the gate or the late freeze (#292).
                if deadline_ns is None:
                    self._ready_gated_since_ns = now_ns
                else:
                    self._ready_late_since_ns = now_ns
            self._loop_stage("lifecycle", self._maybe_emit_lifecycle_status)
            timeout_s = (
                INFLIGHT_POLL_S
                if self._pending
                or self._pending_feature_batches
                or self._send_loop_remote
                or self._held_ring_pushes
                else IDLE_POLL_S
            )
            if deadline_ns:
                timeout_s = min(timeout_s, max(deadline_ns - time.monotonic_ns(), 0) / 1e9)
            wait_started_ns = time.monotonic_ns()
            self._loop_stage("idle_sleep", wake.wait, wake_epoch, timeout_s)
            self._scheduler_wait_ns = (wait_started_ns, time.monotonic_ns())
            self._loop_wall_ns += time.monotonic_ns() - iter_started_ns
            self._loop_iters += 1

    def _dflash_gate(self, now_ns: int) -> int | None:
        """GATE and LATE of the v2 scheduler (design §5.1-5.2).

        In-flight is frozen_seq - done_seq: the send loop's counter moves at
        the GPU done, so a batch that finished since ABSORB is absorbed here
        (its record precedes the counter), and its owners and scratch are
        free for this freeze.  Returns None to wait for a DONE (DRAFT_COMPUTE_
        DEPTH in flight, or nothing READY), 0 to freeze now, or the
        late-freeze deadline (monotonic ns): with one batch running, the next
        freezes at its estimated GPU end minus the estimated CPU work of the
        next batch minus the margin.  It freezes at once without a G or C
        estimate for the buckets, or once the running batch has run past
        twice its estimate."""

        if self._fatal_gpu_quarantine_active() or not (
            self._ready_pending or self._input_ready_features
        ):
            return None
        worker = self._send_loop_worker
        in_flight = len(self._pending_dflash_completions)
        if worker is not None:
            in_flight = self._dflash_batch_seq - worker.done_seq()
            if in_flight < len(self._pending_dflash_completions):
                self._absorb_dflash_done()
            # The loop publishes a DONE record before it stores done_seq, so
            # ABSORB may already hold a batch the counter still counts: its
            # GPU work is done, and the deque is the in-flight set.
            in_flight = min(in_flight, len(self._pending_dflash_completions))
        if in_flight >= DRAFT_COMPUTE_DEPTH:
            return None
        if in_flight == 0:
            return 0
        running = self._pending_dflash_completions[0]
        estimator = self._batch_estimator
        gpu_ns = estimator.predict(
            "tokens", sum(item.task.feature_token_count for item in running.pending), "g"
        )
        ready_tokens = sum(row[4] for row in self._ready_pending) + sum(
            entry.row_range[1] - entry.row_range[0] for entry in self._input_ready_features
        )
        cpu_ns = estimator.predict(
            "tokens", min(ready_tokens, self.feature_buffer_max_tokens), "c"
        )
        if gpu_ns is None or cpu_ns is None:
            return 0
        # Its GPU start: its hand-off, or the GPU done of the batch it was
        # queued behind.
        start_ns = max(running.handoff_ns, self._dflash_last_done_ns)
        deadline_ns = int(start_ns + gpu_ns - cpu_ns) - DRAFT_LATE_FREEZE_MARGIN_NS
        if now_ns >= deadline_ns or now_ns - start_ns > 2 * gpu_ns:
            return 0
        return deadline_ns

    @contextlib.contextmanager
    def _direct_push_scope(self, ready_event, *sources: torch.Tensor):
        """Select and push completed candidates on the push stream.

        ``ready_event`` is the producer completion the caller has already
        observed. Afterwards spin briefly for the tiny push so its READY notice
        leaves now instead of one Draft loop iteration (one batch launch) later.
        """
        transport = getattr(self, "_candidate_push_transport", None)
        stream = getattr(transport, "push_stream", None)
        if self.candidate_transport != "cuda-ipc" or stream is None:
            yield
            return
        stream.wait_event(ready_event)
        for source in sources:
            source.record_stream(stream)
        before = len(self._pending_candidate_pushes)
        with torch.cuda.stream(stream):
            yield
        if self._push_done_worker is not None:
            return  # D9: the push's PUSH_DONE record retires it; no spin
        if len(self._pending_candidate_pushes) > before:
            event = self._pending_candidate_pushes[-1].push.event
            deadline_ns = time.monotonic_ns() + DIRECT_PUSH_SPIN_NS
            while not event.query() and time.monotonic_ns() < deadline_ns:
                pass
            self._poll_candidate_pushes()

    def _queue_early_candidate_push(
        self, tasks, candidates: torch.Tensor | None, ready_event, indices=None
    ) -> None:
        """Queue the candidate push behind the proposal at launch time.

        The push stream waits on ``ready_event`` on the GPU, then copies the
        rows into the Target arena and writes metadata and the host READY or
        completion record.  The Target sees READY without this loop reaching
        the completion poll; that poll and the push poll still do the
        Coordinator result, notice, KV and staging retirement.
        """

        if not GPU_READY_RECORD and not PUSH_COMPLETION_RECORD:
            return
        if candidates is None:
            return
        transport = getattr(self, "_candidate_push_transport", None)
        if self.candidate_transport != "cuda-ipc" or transport is None:
            return
        tasks = list(tasks)
        if indices is None:  # D9 passes the rows it pre-queues
            indices = [
                index
                for index, task in enumerate(tasks)
                if task.candidate_destination is not None
                and task.transfer_id not in self._cancelled_ids
            ]
        selected = [tasks[index] for index in indices]
        if not selected or not (
            transport.has_ready_records(selected)
            or transport.has_completion_records(selected)
        ):
            return
        stream = transport.push_stream
        stream.wait_event(ready_event)
        candidates.record_stream(stream)
        try:
            with torch.cuda.stream(stream):
                rows = candidates[: len(tasks)]
                if len(indices) != len(tasks):
                    rows = _select_candidate_rows(rows, indices)
                rows = rows.contiguous()
                push = transport.submit(tasks=selected, candidates=rows)
        except Exception:
            # Some READY records may already be queued (the Target can act on
            # them) while others are not; no failure result may follow those.
            # Fence and fail stop: Draft exit revokes the owner lease.
            logger.critical(
                "launch-time candidate push failed; entering fatal GPU quarantine",
                exc_info=True,
            )
            try:
                stream.synchronize()
            except Exception:
                pass
            self._early_pushed_ids.update(task.transfer_id for task in selected)
            self._enter_fatal_gpu_quarantine("launch-time candidate push failure")
            return
        self._pending_candidate_pushes.append(
            PendingCandidatePush(
                push=push, tasks=selected, result_slot=None, candidates=rows
            )
        )
        self._early_pushed_ids.update(task.transfer_id for task in selected)
        if self._profile_active:
            profile_emit(
                "draft_candidate_push_submit",
                draft_instance=self.instance_id,
                transfer_ids=tuple(task.transfer_id for task in selected),
                entries=len(selected),
                candidate_bytes=push.candidate_bytes,
                cold_mapping_count=push.cold_mapping_count,
                submitted_ns=push.submitted_ns,
                ticket=push.ticket,
                queued_at_launch=True,
            )

    def _ensure_ring_worker(self):
        """The native ring worker and its landing buffers, created once."""

        from sglang.srt.speculative.dt_direct_ring import DraftRingWorker

        if self._ring_worker is None:
            if self._is_feature_backed:
                count = int(os.environ.get("DT_DIRECT_RING_LANDING", "4"))
                landing = torch.empty(
                    (count, self.feature_buffer_max_tokens, self._pivot_width),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                base = len(self._feature_staging_buffers)
                self._landing_slot_base = base
                self._feature_staging_buffers.extend(landing[index] for index in range(count))
                self._ring_landing = landing
                # Every per-staging-slot scratch list (proposal metadata, pinned
                # candidate rows, ...) gets entries for the landing slots too;
                # under DT_DRAFT_SLOT_DECOUPLE they are per batch instead.
                for name, value in list(vars(self).items()):
                    if (
                        self._slot_decouple
                        or name == "_feature_staging_buffers"
                        or not name.endswith("_buffers")
                        or not isinstance(value, list)
                        or len(value) != base
                        or not all(isinstance(item, torch.Tensor) for item in value)
                    ):
                        continue
                    template = value[0]
                    value.extend(
                        torch.empty(
                            template.shape,
                            dtype=template.dtype,
                            device=template.device,
                            pin_memory=template.device.type == "cpu"
                            and template.is_pinned(),
                        )
                        for _ in range(count)
                    )
                row_bytes = self._pivot_width * landing.element_size()
            else:
                # Token-only rounds carry no hidden; the worker still needs a
                # (never used) landing buffer.
                count = 1
                landing = torch.empty((1, 1, 1), dtype=torch.bfloat16, device=self.device)
                row_bytes = landing.element_size()
            self._landing_taken = [0] * count
            self._ring_worker = DraftRingWorker(
                device=self.gpu_id,
                cpu=int(os.environ.get("DT_DIRECT_RING_CPU", "-1")),
                landing=landing,
                row_bytes=row_bytes,
                # DFlash only; other methods keep the legacy ingress.
                ingress=self._native_ingress,
                wake=self._wake,
            )
            if self._native_ingress:
                capacity = self._ring_worker.ready_capacity
                self._ready_table = np.zeros(capacity, dtype=self._ring_worker.ready_dtype)
            if self._send_loop:
                # The send loop produces this worker's landing releases, so it
                # can only start once the worker exists (design §4.3: one
                # producer on that queue).
                self._send_loop_worker = (
                    self._candidate_push_transport.enable_send_loop(
                        self._wake, self._ring_worker
                    )
                )
                if self._send_loop_worker is None:
                    raise RuntimeError(
                        "DT_DRAFT_SEND_LOOP=1 requires DT_PUBLISH_WORKER=1"
                    )
                self._push_done_worker = self._send_loop_worker
        return self._ring_worker

    def _open_direct_ring(self, registration) -> None:
        """Create this Draft's ring for one Target session (same host only)."""

        from sglang.srt.speculative.dt_direct_ring import (
            RING_CAPACITY,
            DraftRingEndpoint,
            ring_path,
        )

        if (
            not self._persistent
            or self.candidate_transport != "cuda-ipc"
            or self._candidate_push_transport is None
            or registration.target_session_id in self._ring_endpoints
        ):
            return
        self._ensure_ring_worker()
        ring_id = len(self._ring_targets) + 1
        endpoint = DraftRingEndpoint(
            path=ring_path(
                self._candidate_push_transport.draft_session_id,
                registration.target_session_id,
            ),
            ring_id=ring_id,
            target_session_id=registration.target_session_id,
            capacity=RING_CAPACITY,
            max_entry_bytes=(
                self._ring_worker.landing_bytes if self._is_feature_backed else 0
            ),
            runtime=self._candidate_push_transport._runtime,
        )
        self._ring_endpoints[registration.target_session_id] = endpoint
        self._ring_targets[ring_id] = registration.target_instance_id

    def _open_remote_ring(self, registration) -> None:
        """Create the RDMA task-record ring for one cross-host Target.

        The native ring worker serves it like a same-host ring (issue #198):
        it READs each record's hidden rows into the shared landing buffers,
        gated on the Target's ready records, so the row enters through
        ``_poll_direct_rings`` -> ``_admit_ring_tasks``.  Relayed rounds to
        the same Target share the ring's pull leg on the Python pull path,
        which also keeps it, draining its ready records itself, if the ring
        cannot be served.
        """

        from sglang.srt.speculative.dt_rdma import DraftRdmaPuller, DraftRemoteRing

        if (
            not self._persistent
            or self.candidate_transport != "cuda-ipc"
            or self._candidate_push_transport is None
            or registration.target_session_id in self._remote_ring_endpoints
            or registration.target_session_id in self._remote_ring_failed
        ):
            return
        worker = self._ensure_ring_worker()
        ring_id = len(self._ring_targets) + 1
        device = self._canonical_draft_device(self.device)
        puller = None
        if self._is_feature_backed:
            puller = DraftRdmaPuller(
                host=registration.host_addr,
                port=int(registration.data_port),
                device=device,
                draft_instance_id=self.instance_id,
                draft_session_id=self._client.instance_session_id,
                native_ready=True,
            )
        try:
            endpoint = DraftRemoteRing(
                host=registration.host_addr,
                port=int(registration.data_port),
                device=device,
                draft_instance_id=self.instance_id,
                draft_session_id=self._client.instance_session_id,
                target_instance_id=registration.target_instance_id,
                target_session_id=registration.target_session_id,
                ring_id=ring_id,
                landing=worker.landing if self._is_feature_backed else None,
                max_entry_bytes=worker.landing_bytes if self._is_feature_backed else 0,
                puller=puller,
            )
            try:
                worker.attach_remote(endpoint)
            except Exception:
                endpoint.close()
                raise
        except Exception:
            if puller is not None:
                # Closing the pull leg would make the Target fence the whole
                # session; the relay keeps it with Python draining it.
                puller.native_ready = False
                self._feature_pull_transport.adopt_remote_puller(
                    registration.target_instance_id, puller
                )
            raise
        if puller is not None:
            self._feature_pull_transport.adopt_remote_puller(
                registration.target_instance_id, puller
            )
        self._remote_ring_endpoints[registration.target_session_id] = endpoint
        self._ring_targets[ring_id] = registration.target_instance_id
        logger.info(
            "remote task-record ring open to Target %s session %s (ring %d)",
            registration.target_instance_id,
            registration.target_session_id,
            ring_id,
        )

    def _poll_direct_rings(self) -> None:
        """Attach registered rings, take worker-completed entries, admit them."""

        from dt_protocol import DraftTask, DraftTaskKind
        from sglang.srt.speculative.dt_direct_ring import RingRecordV2, rid_hash

        worker = self._ring_worker
        if worker is not None:
            worker.check()
            if worker.ingress:
                worker.drain_ready()
        decoder = self._ring_task_decoder
        if decoder is None:
            decoder = self._ring_task_decoder = msgspec.msgpack.Decoder(DraftTask)
        for endpoint in self._remote_ring_endpoints.values():
            # The worker validated and pulled every record it published; a
            # failed leg stops it (worker.check above raises).
            for _ in range(256):
                entry = endpoint.take_ready()
                if entry is None:
                    break
                task = decoder.decode(entry.payload)
                self._accept_ring_task(endpoint, entry, task, DraftTaskKind)
        for endpoint in self._ring_endpoints.values():
            if not endpoint.attached:
                self._attach_direct_ring(worker, endpoint)
                continue
            for _ in range(256):
                entry = endpoint.take_ready()
                if entry is None:
                    break
                if isinstance(entry, RingRecordV2):
                    self._accept_ring_record(endpoint, entry)
                    continue
                task = decoder.decode(entry.payload)
                self._accept_ring_task(endpoint, entry, task, DraftTaskKind)
        admit_started_ns = time.monotonic_ns()
        self._admit_ring_tasks()
        self._loop_stage_ns["rings_admit"] += time.monotonic_ns() - admit_started_ns
        self._collect_rid_evictions()
        if self._ring_waiting:
            now_ns = time.monotonic_ns()
            for transfer_id, task in tuple(self._ring_waiting.items()):
                if now_ns >= task.deadline_ns + UNCONFIRMED_DIRECT_GRACE_NS:
                    # The worker may still be stalled on landing capacity: the
                    # ring copy can still read the source and push, so this
                    # result proves nothing; drop that copy when it arrives.
                    del self._ring_waiting[transfer_id]
                    self._received_tasks += 1  # Coordinator copy retires here
                    self._ring_abandoned.add(transfer_id)
                    error = TimeoutError("direct ring copy did not arrive in time")
                    error.resources_quiescent = False
                    self._publish_failures([(task, error)])
        landing_started_ns = time.monotonic_ns()
        pending_frees = self._landing_free_pending
        while pending_frees and pending_frees[0][1].query():
            index, _event = pending_frees.popleft()
            taken = self._landing_taken[index]
            if taken and worker.try_free(index, taken) == 1:
                self._landing_taken[index] = 0
        for index, taken in enumerate(self._landing_taken):
            if taken and self._feature_staging_refcounts.get(
                self._landing_slot_base + index, 0
            ) == 0 and not any(item[0] == index for item in pending_frees):
                self._free_landing(index)
        self._loop_stage_ns["rings_landing"] += (
            time.monotonic_ns() - landing_started_ns
        )

    def _attach_direct_ring(self, worker, endpoint) -> None:
        """Attach a same-host ring once its Target registered its source."""

        registration = endpoint.registration()
        if registration is None:
            return
        handle, length, pool_generation, source_session_id = registration
        self._ring_source_registration[endpoint.ring_id] = (handle, length)
        source_base = 0
        if self._is_feature_backed:
            if not length:
                raise RuntimeError("direct ring Target registered no source")
            source_base, _warm = self._feature_pull_transport._mapping_cache.map(
                peer_instance_id=self._ring_targets[endpoint.ring_id],
                peer_session_id=source_session_id,
                pool_generation=pool_generation,
                ipc_handle=handle,
            )
        worker.attach(
            endpoint,
            source_base=source_base,
            source_length=length,
            ledger=(
                self._rid_ledger._handle
                if self._rid_ledger is not None
                else None
            ),
            target_instance_id=self._ring_targets[endpoint.ring_id],
        )
        endpoint.mark_serving(
            pool_generation=pool_generation,
            source_session_id=source_session_id,
        )

    def _check_ring_task(self, endpoint, entry, task, task_kinds) -> None:
        """The ring entry and its decoded v1 task name the same round."""

        if (
            task.direct_ring_seq != entry.seq
            or task.target_instance_id != self._ring_targets[endpoint.ring_id]
            or task.kind != task_kinds.RECONCILE
            or task.candidate_destination is None
            or task.candidate_destination.target_session_id != endpoint.target_session_id
        ):
            raise ValueError("direct ring entry does not match its task")
        if entry.has_pull:
            source = task.source
            if (
                source is None
                or source.source_slot_id != entry.source_slot_id
                or source.slot_generation != entry.slot_generation
                or source.nbytes != entry.nbytes
                or source.allocation_offset + source.byte_offset != entry.source_offset
                or source.source_session_id != endpoint.source_session_id
                or source.pool_generation != endpoint.pool_generation
                or entry.landing_index < 0
            ):
                raise ValueError("direct ring pull does not match its task source")
        elif task.source is not None:
            raise ValueError("direct ring task source was not pulled")

    def _accept_ring_task(self, endpoint, entry, task, task_kinds) -> None:
        self._check_ring_task(endpoint, entry, task, task_kinds)
        transfer_id = task.transfer_id
        if transfer_id in self._pending_ids:
            raise ValueError("direct ring task is already known")
        self._validate_task(task)
        abandoned = transfer_id in self._ring_abandoned
        if abandoned or transfer_id in self._cancelled_ids:
            self._drop_ring_task(task, entry, report=not abandoned)
            return
        received_ns = time.monotonic_ns()
        self._direct_tasks_by_transfer[transfer_id] = task
        if self._ring_confirmation:
            self._unconfirmed_direct[transfer_id] = task.deadline_ns
            self._seen_transfers.add(transfer_id)
            self._seen_transfer_order.append(transfer_id)
            if len(self._seen_transfer_order) > 65536:
                self._seen_transfers.discard(self._seen_transfer_order.popleft())
        self._pending_ids.add(transfer_id)
        self._ring_pending_ids.add(transfer_id)
        self._received_ns_by_transfer[transfer_id] = received_ns
        if entry.has_pull:
            slot = self._landing_slot_base + entry.landing_index
            row_bytes = self._ring_worker.row_bytes
            first = entry.landing_offset // row_bytes
            rows = (first, first + entry.nbytes // row_bytes)
            if rows[1] - rows[0] != int(task.feature_token_count):
                raise ValueError("direct ring pull row count mismatch")
            refcounts = self._feature_staging_refcounts
            refcounts[slot] = refcounts.get(slot, 0) + 1
            self._landing_taken[entry.landing_index] += 1
            # The worker's INPUT_READY write followed the copy on its stream.
            self._publish_ring_source_done(task)
            self._ring_admission.append((task, slot, rows))
        else:
            self._pending.append(task)
        profile_emit(
            "draft_ring_take",
            draft_instance=self.instance_id,
            transfer_id=transfer_id,
            request_id=task.request_id,
            round_id=task.round_id,
            ring_id=endpoint.ring_id,
            ring_seq=entry.seq,
            has_pull=entry.has_pull,
            taken_ns=received_ns,
        )
        waiting = self._ring_waiting.pop(transfer_id, None)
        if waiting is not None:
            self._confirm_direct_task(waiting)

    def _drop_ring_task(self, task, entry, *, report: bool) -> None:
        """Consume a ring copy that must not run (cancelled/released/abandoned).

        The worker already read its source.  Publish the authenticated
        candidate tombstone so the Target can reuse the slot; ``report`` also
        sends the Coordinator its CANCELLED result (an abandoned round already
        got its failure).
        """

        transfer_id = task.transfer_id
        self._ring_abandoned.discard(transfer_id)
        waiting = self._ring_waiting.pop(transfer_id, None)
        if waiting is not None:
            self._received_tasks += 1  # its Coordinator copy retires here
        if entry.has_pull:
            # The worker placed it in a landing buffer: count it as released
            # (no staging reference) so try_free's placed/released match holds.
            # Native ingress: the caller releases the placement.
            if not self._native_ingress:
                self._landing_taken[entry.landing_index] += 1
            self._publish_ring_source_done(waiting or task)
        transport = self._candidate_push_transport
        if transport is not None and task.candidate_destination is not None:
            push = transport.publish_cancelled([task])
            self._pending_candidate_pushes.append(
                PendingCandidatePush(
                    push=push,
                    tasks=[task],
                    result_slot=None,
                    candidates=None,
                    cancelled=True,
                    coordinator_result=report and self._ring_confirmation,
                )
            )
        self._cancelled_ids.discard(transfer_id)
        self._received_ns_by_transfer.pop(transfer_id, None)
        if waiting is None and self._ring_confirmation:
            # A later Coordinator copy is answered by this drop.
            self._remember_ring_retired(transfer_id, failed=False)
        self._drain_deferred_persistent_release(task.draft_kv_handle)
        profile_emit(
            "draft_ring_drop",
            draft_instance=self.instance_id,
            transfer_id=transfer_id,
            ring_seq=entry.seq,
            reported=report,
        )

    def _accept_ring_record(self, endpoint, record) -> None:
        """Admit one kernel-published verify wave (§7.1) as per-request tasks.

        The record carries no candidate destinations; each reconstructed task
        executes from the ring data and adopts its destination from the later
        Coordinator confirmation copy.
        """

        from dt_protocol import DraftTask, DraftTaskKind, SourceRef

        dropped_mask = getattr(record, "dropped_mask", 0)
        if record.landing_index < 0:
            if dropped_mask:
                # Fully dropped wave: every request was released before the
                # worker served the record; consumed without a pull, so
                # there is nothing to admit.
                return
            raise ValueError("ring record has no landing placement")
        ledger = getattr(self, "_rid_ledger", None)
        if ledger is None:
            raise RuntimeError("ring record rid ledger is unavailable")
        from sglang.srt.speculative.dt_direct_ring import rid_hash

        registration = self._ring_source_registration.get(endpoint.ring_id)
        if registration is None:
            raise RuntimeError("ring record arrived before source registration")
        ipc_handle, allocation_length = registration
        target_instance_id = self._ring_targets[endpoint.ring_id]
        row_bytes = record.row_bytes
        slot = self._landing_slot_base + record.landing_index
        # The worker placed the whole wave as one landing entry; claim it once
        # so try_free's placed/released match survives dropped requests.  The
        # staging refcount below stays per request.
        self._landing_taken[record.landing_index] += 1
        created_ns = time.monotonic_ns()
        # Token ids sit at compacted prefix positions in the record's token
        # array; token_offset is the feature row offset (block-strided for
        # in-place waves), not the token array position.
        token_prefix = 0
        for position, req in enumerate(record.requests):
            if dropped_mask >> position & 1:
                # Released before the worker served the record: consumed
                # without a task (the ring never stalls behind it).
                token_prefix += req.token_count
                continue
            if (
                req.round_id < 1
                or req.token_count <= 0
                or req.seq_len < req.token_count
                or req.token_offset + req.token_count > record.row_count
            ):
                raise ValueError("ring record request scalars are invalid")
            # The native worker resolved the rid into the mailbox before
            # sealing the record; cross-check it against the record's hash.
            rid = req.rid
            if not rid:
                raise ValueError("ring record references an unknown request")
            if len(self._rid_hash_cache) > 65536:
                self._rid_hash_cache.clear()
            cached_hash = self._rid_hash_cache.get(rid)
            if cached_hash is None:
                cached_hash = rid_hash(rid)
                self._rid_hash_cache[rid] = cached_hash
            if cached_hash != req.rid_hash:
                raise ValueError("ring record references an unknown request")
            transfer_id = (
                f"{target_instance_id}:{rid}:{req.round_id}:{req.attempt}"
            )
            ids = record.token_ids[token_prefix : token_prefix + req.token_count]
            token_prefix += req.token_count
            source = SourceRef(
                target_instance_id=target_instance_id,
                # Provisional device id; the fence fields that matter (session,
                # generation, handles, offsets) are real.
                source_device_id=0,
                source_slot_id=record.source_slot_id,
                slot_generation=record.slot_generation,
                transfer_id=transfer_id,
                feature_schema_id=self.profile.feature_schema.schema_id,
                nbytes=req.token_count * row_bytes,
                byte_offset=req.token_offset * row_bytes,
                transport="cuda-ipc",
                source_session_id=endpoint.source_session_id,
                pool_generation=endpoint.pool_generation,
                allocation_length=allocation_length,
                allocation_offset=record.source_byte_offset,
                ipc_handle=ipc_handle,
                dtype="bfloat16",
                shape=(req.token_count, self._pivot_width),
                producer_event_handle=record.producer_event_handle,
            )
            task = DraftTask(
                transfer_id=transfer_id,
                request_id=rid,
                round_id=req.round_id,
                attempt=req.attempt,
                target_instance_id=target_instance_id,
                draft_profile_id=self.profile.profile_id,
                k=record.block_size - 1,
                next_token_id=req.bonus_token,
                source=source,
                created_ns=created_ns,
                deadline_ns=created_ns + _RING_RECORD_TIMEOUT_NS,
                kind=DraftTaskKind.RECONCILE,
                draft_kv_handle=f"{target_instance_id}:{rid}:{req.attempt}",
                committed_token_ids=ids,
                feature_token_ids=ids,
                committed_kv_len=req.seq_len,
                feature_token_offset=req.seq_len - req.token_count,
                feature_token_count=req.token_count,
                requires_proposal=True,
                direct_ring_seq=record.seq,
                candidate_destination=None,
            )
            if transfer_id in self._pending_ids:
                raise ValueError("direct ring task is already known")
            self._validate_task(task)
            abandoned = transfer_id in self._ring_abandoned
            if abandoned or transfer_id in self._cancelled_ids:
                self._drop_ring_record_task(record, task, report=not abandoned)
                continue
            if self._ring_publish_at_launch:
                # Scan the destination region while this record's ring
                # generation is still observable.  The Target's cleanup
                # normally trails the record by about one scheduler
                # iteration, but when it already ran (a slow consumer), a
                # tombstoned row — one that ended in its wave, with no
                # Coordinator task — is retired here, before candidate
                # compute creates the wave's held push; by then the ring may
                # have wrapped and the tombstone would be unrecoverable.
                # A live triple is latched onto the task: it is the same
                # destination the publish path would resolve later, observed
                # at the only moment its generation is provably current.
                resolved = endpoint.destination(record.seq, position)
                if resolved is not None:
                    if (resolved[0], resolved[1], resolved[2]) == (0, 0, 0):
                        self._drop_ring_record_task(record, task, report=True)
                        continue
                    destination = self._ring_destination_ref(task, resolved)
                    if destination is not None:
                        from msgspec.structs import replace

                        task = replace(task, candidate_destination=destination)
            received_ns = time.monotonic_ns()
            self._direct_tasks_by_transfer[transfer_id] = task
            if self._ring_confirmation:
                self._unconfirmed_direct[transfer_id] = task.deadline_ns
                self._seen_transfers.add(transfer_id)
                self._seen_transfer_order.append(transfer_id)
                if len(self._seen_transfer_order) > 65536:
                    self._seen_transfers.discard(
                        self._seen_transfer_order.popleft()
                    )
            self._pending_ids.add(transfer_id)
            self._ring_pending_ids.add(transfer_id)
            self._received_ns_by_transfer[transfer_id] = received_ns
            self._ring_record_positions[transfer_id] = (endpoint, position)
            first = record.landing_offset // row_bytes + req.token_offset
            rows = (first, first + req.token_count)
            refcounts = self._feature_staging_refcounts
            refcounts[slot] = refcounts.get(slot, 0) + 1
            # The worker's INPUT_READY write followed the copy on its stream.
            self._publish_ring_source_done(task)
            self._ring_admission.append((task, slot, rows))
            profile_emit(
                "draft_ring_take",
                draft_instance=self.instance_id,
                transfer_id=transfer_id,
                request_id=task.request_id,
                round_id=task.round_id,
                ring_id=endpoint.ring_id,
                ring_seq=record.seq,
                has_pull=True,
                taken_ns=received_ns,
            )
            waiting = self._ring_waiting.pop(transfer_id, None)
            if waiting is not None:
                self._confirm_direct_task(waiting)

    def _drop_ring_record_task(self, record, task, *, report: bool) -> None:
        """Consume one record-reconstructed task that must not run.

        Mirrors ``_drop_ring_task`` for the §7.1 record path.  The wave's
        landing entry was claimed once per record, so no landing accounting
        happens here; the worker already read the source, so the Target still
        gets its authenticated completion (and a cancelled tombstone when the
        waiting Coordinator copy supplies a destination).
        """

        transfer_id = task.transfer_id
        self._ring_abandoned.discard(transfer_id)
        self._ring_record_positions.pop(transfer_id, None)
        waiting = self._ring_waiting.pop(transfer_id, None)
        if waiting is not None:
            self._received_tasks += 1  # its Coordinator copy retires here
        self._publish_ring_source_done(waiting or task)
        tombstone = waiting if waiting is not None else task
        transport = self._candidate_push_transport
        if transport is not None and tombstone.candidate_destination is not None:
            push = transport.publish_cancelled([tombstone])
            self._pending_candidate_pushes.append(
                PendingCandidatePush(
                    push=push,
                    tasks=[tombstone],
                    result_slot=None,
                    candidates=None,
                    cancelled=True,
                    coordinator_result=report and self._ring_confirmation,
                )
            )
        self._cancelled_ids.discard(transfer_id)
        self._received_ns_by_transfer.pop(transfer_id, None)
        if waiting is None and self._ring_confirmation:
            # A later Coordinator copy is answered by this drop.
            self._remember_ring_retired(transfer_id, failed=False)
        self._drain_deferred_persistent_release(task.draft_kv_handle)
        profile_emit(
            "draft_ring_drop",
            draft_instance=self.instance_id,
            transfer_id=transfer_id,
            ring_seq=record.seq,
            reported=report,
        )

    def _publish_ring_source_done(self, task) -> None:
        """Send a taken ring round's read proof where the Target needs it.

        The worker writes the entry's INPUT_READY word after the copy on its
        stream, and that word is the same-host Target's read proof
        (``dt_target._poll_direct_ring_reads``).  Confirmation-free, only a
        cross-host round, whose remote ring has no early read proof, relays a
        SourceReadDone, at take (design §8.3 duty 1).
        """

        if self._ring_confirmation or task.target_instance_id in self._remote_targets:
            self._publish_source_done(task)

    def _remember_ring_retired(self, transfer_id: str, *, failed: bool) -> None:
        retired = self._ring_retired
        retired[transfer_id] = failed
        while len(retired) > RING_RETIRED_LIMIT:
            retired.popitem(last=False)

    def _collect_rid_evictions(self) -> None:
        """Evict release-deferred rids once their owner rings caught up."""

        horizon = getattr(self, "_rid_evict_horizon", None)
        if not horizon:
            return
        endpoints = getattr(self, "_ring_endpoints", {})
        ledger = getattr(self, "_rid_ledger", None)
        for rid, thresholds in tuple(horizon.items()):
            if all(
                ring_id not in endpoints or endpoints[ring_id].next_seq >= threshold
                for ring_id, threshold in thresholds.items()
            ):
                del horizon[rid]
                if ledger is not None:
                    ledger.evict(rid)

    def _admit_ring_tasks(self) -> None:
        """Move pulled ring tasks into INPUT_READY under persistent admission."""

        queue = self._ring_admission
        if not queue:
            return
        available_rows, available_kv = self._unreserved_allocation(
            self._input_ready_features
        )
        now_ns = time.monotonic_ns()
        from dt_protocol import DraftTaskKind

        # Why the FIFO stopped draining this pass (issue #269, #267 §8.1):
        # "drained" unless one of the two head-of-line breaks below fires.
        stop_reason = "drained"
        while queue:
            task, slot, rows = queue[0]
            if (
                task.kind is DraftTaskKind.RECONCILE
                and task.round_id >= 1
                and task.draft_kv_handle not in self._persistent_requests
            ):
                # A later round of a request the release already retired:
                # the wave was published before the release landed, so the
                # record task must retire like a cancellation, never run.
                self._cancelled_ids.add(task.transfer_id)
                profile_emit(
                    "draft_ring_released_drop",
                    draft_instance=self.instance_id,
                    transfer_id=task.transfer_id,
                    request_id=task.request_id,
                    round_id=task.round_id,
                )
            if task.transfer_id in self._cancelled_ids or now_ns >= task.deadline_ns:
                queue.popleft()
                cancelled = task.transfer_id in self._cancelled_ids
                if cancelled:
                    self._retire_cancelled_transfer_identity(task)
                    self._drain_deferred_persistent_release(task.draft_kv_handle)
                else:
                    self._release_persistent_request(task.draft_kv_handle)
                    self._publish_failures(
                        [(task, TimeoutError("direct ring task deadline expired"))]
                    )
                self._release_feature_staging_entries(slot, 1)
                continue
            row_cost, kv_cost = self._persistent_task_allocation(task)
            if row_cost > available_rows or kv_cost > available_kv:
                stop_reason = "kv_rows" if row_cost > available_rows else "kv_tokens"
                break
            if task.requires_proposal and self._proposal_admission_blocked():
                stop_reason = "credit"
                break
            queue.popleft()
            available_rows -= row_cost
            available_kv -= max(kv_cost, 0)
            self._received_ns_by_transfer.pop(task.transfer_id, None)
            self._input_ready_features.append(
                InputReadyFeature(
                    pending=PendingDraftTask(task=task, taken_ns=now_ns),
                    staging_slot=slot,
                    row_range=rows,
                    ready_ns=now_ns,
                )
            )
            if self._profile_active:
                # Per-round READY on the direct path (issue #269; the pull
                # path emits the same event in _poll_feature_batches).
                profile_emit(
                    "draft_input_ready",
                    draft_instance=self.instance_id,
                    transfer_id=task.transfer_id,
                    request_id=task.request_id,
                    round_id=task.round_id,
                    staging_slot=slot,
                    ready_ns=now_ns,
                    direct=True,
                )
        if self._profile_active:
            self._profile_ring_admission_state(
                observed_ns=now_ns,
                reason=stop_reason,
                queue_len=len(queue),
                head_transfer_id=queue[0][0].transfer_id if queue else "",
                available_rows=available_rows,
                available_kv=available_kv,
            )

    def _unreserved_allocation(self, input_ready) -> tuple[int, int]:
        """Free req rows and KV pages minus what admitted work will take:
        pending relayed pulls and the ``input_ready`` entries."""

        available_rows = self._draft_model_runner.req_to_token_pool.available_size()
        available_kv = self._draft_model_runner.token_to_kv_pool_allocator.available_size()
        for owner in self._pending_feature_batches:
            for item in owner.pending:
                rows, tokens = self._persistent_task_allocation(item.task)
                available_rows -= rows
                available_kv -= max(tokens, 0)
        for entry in input_ready:
            rows, tokens = self._persistent_task_allocation(entry.pending.task)
            available_rows -= rows
            available_kv -= max(tokens, 0)
        return available_rows, available_kv

    def _drain_ready(self) -> None:
        """DRAIN (design §5.1): attach newly registered rings, then move
        every published READY entry into the READY table.  DROPPED entries
        retire here (as the legacy take skips dropped requests); a
        V1_PAYLOAD entry is decoded into its row (until Target item T4)."""

        if self._ready_gated_since_ns:  # the last take stopped at the gate
            self._ready_gate_ns += time.monotonic_ns() - self._ready_gated_since_ns
            self._ready_gated_since_ns = 0
        if self._ready_late_since_ns:  # the v2 scheduler held a late freeze
            self._ready_late_ns += time.monotonic_ns() - self._ready_late_since_ns
            self._ready_late_since_ns = 0
        worker = self._ring_worker
        if worker is None:
            return
        worker.check()
        if self._send_loop_worker is None:
            # With the send loop the fence is a descriptor: its DONE record
            # retires the event in ABSORB instead.
            fenced = self._ready_release_fenced
            while fenced and fenced[0][0].query():
                self._release_ready_rows(fenced.popleft()[1])
        for endpoint in self._ring_endpoints.values():
            if not endpoint.attached:
                self._attach_direct_ring(worker, endpoint)
        entries = worker.drain_ready()
        if not len(entries):
            return
        kinds = worker.ready_kinds
        self._ready_table[(entries["seq"] - 1) % len(self._ready_table)] = entries
        drained_ns = time.monotonic_ns()
        kind = entries["kind"] & ~np.uint32(kinds["DEST_KNOWN"])
        marks = (self._ready_gate_ns, self._ready_frozen_batches, self._ready_late_ns)
        done, dropped = [], []
        for kind, (
            seq, ring_id, record_seq, position, cell, gen, round_id, attempt,
            seq_len, count, bonus, k, landing_gen, row_first, token_at, ready_ns,
        ) in zip(kind.tolist(), zip(*(entries[f].tolist() for f in _READY_ROW_FIELDS))):
            if kind != kinds["ROW"] and kind != kinds["V1_PAYLOAD"]:
                done.append(seq)  # DROPPED: its request was released
                continue
            key = (ring_id, record_seq)
            if k >= 0:
                record = self._ready_records.get(key)
                if record is None:
                    self._ready_records[key] = [1, k, landing_gen]
                else:
                    record[0] += 1
            if kind == kinds["ROW"]:
                # Native row: its identity is the rid-ledger cell key; the
                # f-string handle/transfer_id are built lazily
                # (_ready_row_identity) when a row freezes or retires.
                ident = (gen << 32) | cell
                fields = (
                    ring_id, record_seq, position, round_id, attempt, seq_len,
                    token_at, bonus, ready_ns, cell, gen,
                )
                deadline_ns, requires_proposal = ready_ns + _RING_RECORD_TIMEOUT_NS, True
            else:
                task = self._accept_ready_v1(worker, seq, ring_id, record_seq, k)
                if task is None:
                    done.append(seq)
                    dropped.append(key)
                    continue
                ident = (task.draft_kv_handle, task.transfer_id)
                fields = (
                    ring_id, record_seq, position, round_id, attempt, seq_len,
                    token_at, bonus, ready_ns, task.request_id,
                )
                deadline_ns, requires_proposal = task.deadline_ns, task.requires_proposal
                count = task.feature_token_count
            self._ready_pending.append((
                seq, ident, deadline_ns, requires_proposal, count, k,
                row_first, drained_ns, fields, marks,
            ))
        if dropped:
            self._release_ready_rows(dropped, fence_seqs=done)
        if done:
            self._ready_consume(done)

    def _accept_ready_v1(self, worker, seq, ring_id, record_seq, k):
        """Decode a V1_PAYLOAD entry's task for its READY-table row; None
        when the round was cancelled first (it then retires here, as the
        legacy take does)."""

        from dt_protocol import DraftTask, DraftTaskKind

        if self._ring_task_decoder is None:
            self._ring_task_decoder = msgspec.msgpack.Decoder(DraftTask)
        endpoint = worker.endpoint(ring_id)
        entry = endpoint.v1_entry(record_seq)
        task = self._ring_task_decoder.decode(entry.payload)
        self._check_ring_task(endpoint, entry, task, DraftTaskKind)
        if not entry.has_pull or k < 0:
            raise ValueError("direct ring task source was not pulled")
        if entry.nbytes // worker.row_bytes != int(task.feature_token_count):
            raise ValueError("direct ring pull row count mismatch")
        if task.transfer_id in self._cancelled_ids:
            self._drop_ring_task(task, entry, report=True)
            return None
        # A cross-host round relays its read proof at take (design §8.3).
        self._publish_ring_source_done(task)
        self._direct_tasks_by_transfer[task.transfer_id] = task
        self._ready_tasks[seq] = task
        return task

    def _ready_row_identity(self, row):
        """(handle, transfer_id, rid) of a READY-table row, built lazily.

        Native rows keep only their rid-ledger cell key through DRAIN and
        FREEZE; the strings are needed by the retire/publish/profile paths
        and by the DraftTask a frozen row gets.  V1 rows carry the decoded
        task's identity."""

        ident, fields = row[1], row[8]
        if type(ident) is tuple:
            return ident[0], ident[1], fields[9]
        target = self._ring_targets[fields[0]]
        rid = self._rid_ledger.rid_of(fields[9], fields[10])
        handle = f"{target}:{rid}:{fields[4]}"
        return handle, f"{target}:{rid}:{fields[3]}:{fields[4]}", rid

    def _freeze_ready_rows(
        self, selected, busy_handles, proposal_mode, token_count, now_ns
    ) -> None:
        """FREEZE over the READY table (design §5.1): append to ``selected``,
        in arrival order, every row that fits (owner not busy, same proposal
        mode, batch bound, KV and req-row fit).  A row that does not fit stays
        READY; there is no head-of-line break.  Rows of released or cancelled
        requests and expired rows retire here.  Only a frozen row gets a
        DraftTask: the retire and push paths still take one until D8/D9.

        Native rows are evaluated by their ledger cell key; the f-string
        handle/transfer_id and the DraftTask are built lazily, only for a
        row that freezes or retires (and under the profile).  The #292 skip
        accounting runs only under the profile; production keeps the drop
        and frozen-batch counters."""

        if not self._ready_pending:
            return
        worker = self._ring_worker
        available_rows, available_kv = self._unreserved_allocation(
            (*selected, *self._input_ready_features)
        )
        profile = self._profile_active
        sub_ns = self._submit_sub_ns if profile else None  # SUBMIT_SUB_STAGES
        retained, done, dropped = [], [], []
        by_cell = self._persistent_by_cell
        cancelled_ids = self._cancelled_ids
        deferred = self._deferred_persistent_releases
        deferred_cells = self._deferred_release_cells
        pass_handles = set()

        if profile:
            # Skip accounting (#292).  A waiting row's state is [reason, last
            # evaluation ns, gate mark, passes, first reason, fit, held ns per
            # reason, late-freeze mark]; a row frozen at its first pass gets
            # none.
            states, gate_ns = self._ready_skip_state, self._ready_gate_ns
            late_ns = self._ready_late_ns
            total, skips = self._ready_skip_held_ns, self._ready_skips
            unscanned, gate = int(_ReadySkip.unscanned), int(_ReadySkip.compute_depth)
            late = int(_ReadySkip.late_freeze)
            self._ready_passes += 1

            def charge(held, last, waited, gate_mark, late_mark):
                # The gate and late-freeze time since the marks, then the rest
                # to the reason of the earlier evaluation.
                gated = min(gate_ns - gate_mark, waited)
                held_late = min(late_ns - late_mark, waited - gated)
                for index, value in (
                    (last, waited - gated - held_late), (gate, gated), (late, held_late)
                ):
                    held[index] += value
                    total[index] += value

            def skip(row, reason, fit=None):
                state = states.get(row[0])
                if state is None:
                    held = [0] * len(total)
                    charge(held, unscanned, now_ns - row[8][8], row[9][0], row[9][2])
                    states[row[0]] = [reason, now_ns, gate_ns, 1, reason, fit, held, late_ns]
                else:
                    charge(state[6], state[0], now_ns - state[1], state[2], state[7])
                    state[0], state[1], state[2], state[7] = reason, now_ns, gate_ns, late_ns
                    state[3] += 1
                    if fit is not None:
                        state[5] = fit
                skips[reason] += 1
                retained.append(row)

            def finish(row, outcome):
                fields, marks = row[8], row[9]
                state = states.pop(row[0], None)
                if state is None:
                    held = [0] * len(total)
                    charge(held, unscanned, now_ns - fields[8], marks[0], marks[2])
                    last = unscanned
                else:
                    held, last = state[6], state[0]
                    charge(held, last, now_ns - state[1], state[2], state[7])
                if outcome != "frozen":
                    self._ready_drops[_READY_DROP_OUTCOMES.index(outcome)] += 1
                held_ns = now_ns - fields[8]
                if held_ns < _READY_SKIP_RECORD_NS:
                    return
                summary = {
                    "outcome": outcome,
                    "held_ns": held_ns,
                    "passes": 0 if state is None else state[3],
                    "first_reason": None if state is None else _READY_SKIP_REASONS[state[4]],
                    "last_reason": None if state is None else _READY_SKIP_REASONS[state[0]],
                    "reasons_ns": {n: v for n, v in zip(_READY_SKIP_REASONS, held) if v},
                    "batches": self._ready_frozen_batches - marks[1],
                    "fit": None if state is None else state[5],
                }
                if held_ns >= _READY_LONG_HOLD_NS:
                    for index, value in enumerate(held):
                        self._ready_long_held_ns[index] += value
                    self._ready_long_rows[held.index(max(held))] += 1
                    if self._ready_longest is None or held_ns > self._ready_longest["held_ns"]:
                        self._ready_longest = summary
                _, transfer_id, rid = self._ready_row_identity(row)
                profile_emit(
                    "draft_ready_skip",
                    draft_instance=self.instance_id,
                    transfer_id=transfer_id,
                    request_id=rid,
                    round_id=fields[3],
                    ready_ns=fields[8],
                    drained_ns=row[7],
                    finished_ns=now_ns,
                    long=held_ns >= _READY_LONG_HOLD_NS,
                    **summary,
                )

        else:

            def skip(row, reason, fit=None):
                retained.append(row)

            def finish(row, outcome):
                if outcome != "frozen":
                    self._ready_drops[_READY_DROP_OUTCOMES.index(outcome)] += 1

        for row in self._ready_pending:
            (
                seq, ident, deadline_ns, requires_proposal, count, k,
                row_first, drained_ns, fields, _marks,
            ) = row
            ring_id, record_seq, position, round_id = fields[:4]
            task = self._ready_tasks.get(seq)
            # The row's request is gone, or its release waits only on an
            # in-flight batch: this later round must not run (as the legacy
            # admission's released drop).  Native rows answer from the cell
            # index and the deferred-release cell set; the string identity
            # is built only while a cancel is outstanding (and for a row
            # that freezes or retires).
            if type(ident) is int:
                entry = by_cell.get(ident)
                released = (
                    entry is None
                    or entry[1] != fields[4]
                    or ident in deferred_cells
                )
                state = None if released else entry[0]
            else:
                state = None
                released = (
                    ident[0] not in self._persistent_requests
                    or ident[0] in deferred
                )
            if released:
                cancelled = True
            elif cancelled_ids:
                cancelled = self._ready_row_identity(row)[1] in cancelled_ids
            else:
                cancelled = False
            if cancelled or now_ns >= deadline_ns:
                handle, transfer_id, rid = self._ready_row_identity(row)
                if task is None and not cancelled:
                    task = self._ready_row_task(handle, transfer_id, deadline_ns, count, fields, rid)
                if cancelled:
                    if task is not None:
                        self._retire_cancelled_transfer_identity(task)
                    self._cancelled_ids.discard(transfer_id)
                    self._drain_deferred_persistent_release(handle)
                    profile_emit(
                        "draft_ring_released_drop",
                        draft_instance=self.instance_id,
                        transfer_id=transfer_id,
                        round_id=round_id,
                    )
                else:
                    # A ring identity: _send_result keeps the failure local.
                    self._pending_ids.add(transfer_id)
                    self._ring_pending_ids.add(transfer_id)
                    self._release_persistent_request(handle)
                    self._publish_failures(
                        [(task, TimeoutError("direct ring task deadline expired"))]
                    )
                dropped.append((ring_id, record_seq))
                done.append(seq)
                finish(row, "released" if released else "cancelled" if cancelled else "expired")
                continue
            # Owner-busy: native rows answer through the state the released
            # check already fetched (same handle as the row's).
            if type(ident) is int:
                busy = state.draft_kv_handle in busy_handles
            else:
                busy = ident[0] in busy_handles
            if busy:
                if type(ident) is int:
                    owner_selected = state.draft_kv_handle in pass_handles
                else:
                    owner_selected = ident[0] in pass_handles
                skip(row, _ReadySkip.owner_selected if owner_selected
                     else _ReadySkip.owner_inflight)
                continue
            if len(selected) >= self.max_batch_size:
                skip(row, _ReadySkip.batch_rows)
                continue
            if proposal_mode is None:
                proposal_mode = requires_proposal
            if requires_proposal != proposal_mode:
                skip(row, _ReadySkip.proposal_mode)
                continue
            if token_count + count > self.feature_buffer_max_tokens:
                skip(row, _ReadySkip.batch_tokens)
                continue
            if profile:
                part_ns = time.monotonic_ns()
            if type(ident) is int:
                # _persistent_task_allocation of the row's (not yet built)
                # task: a native row always requires a proposal, so the fit
                # reserves one block over the committed prefix.
                allocated = (
                    int(state.req.kv.kv_allocated_len)
                    if state.req.kv is not None
                    else 0
                )
                row_cost = int(state.req.req_pool_idx is None)
                kv_cost = fields[5] + self.profile.block_size - allocated
            else:
                if task is None:
                    handle, transfer_id, rid = self._ready_row_identity(row)
                    task = self._ready_row_task(handle, transfer_id, deadline_ns, count, fields, rid)
                row_cost, kv_cost = self._persistent_task_allocation(task)
            if profile:
                sub_ns[1] += time.monotonic_ns() - part_ns
            if row_cost > available_rows or kv_cost > available_kv:
                fit = (available_rows, available_kv, row_cost, kv_cost)
                skip(row, _ReadySkip.req_rows if row_cost > available_rows
                     else _ReadySkip.kv, fit)
                continue
            available_rows -= row_cost
            available_kv -= max(kv_cost, 0)
            token_count += count
            if type(ident) is int:
                handle, transfer_id, rid = self._ready_row_identity(row)
                if profile:
                    part_ns = time.monotonic_ns()
                task = self._ready_row_task(handle, transfer_id, deadline_ns, count, fields, rid)
                if profile:
                    sub_ns[0] += time.monotonic_ns() - part_ns
            else:
                handle, transfer_id = ident
            busy_handles.add(handle)
            pass_handles.add(handle)
            finish(row, "frozen")
            self._pending_ids.add(transfer_id)
            self._ring_pending_ids.add(transfer_id)
            if seq not in self._ready_tasks:
                self._ring_record_positions[transfer_id] = (
                    worker.endpoint(ring_id),
                    position,
                )
            selected.append(
                InputReadyFeature(
                    pending=PendingDraftTask(task=task, taken_ns=drained_ns),
                    staging_slot=self._landing_slot_base + k,
                    row_range=(row_first, row_first + count),
                    ready_ns=drained_ns,
                    ready_seq=seq,
                )
            )
            if profile:
                # Native ingress has no admission queue: the drain is both
                # the take and READY of the D0 chain.
                profile_emit(
                    "draft_ring_take",
                    draft_instance=self.instance_id,
                    transfer_id=transfer_id,
                    request_id=task.request_id,
                    round_id=round_id,
                    ring_id=ring_id,
                    ring_seq=record_seq,
                    has_pull=True,
                    taken_ns=drained_ns,
                )
                profile_emit(
                    "draft_input_ready",
                    draft_instance=self.instance_id,
                    transfer_id=transfer_id,
                    request_id=task.request_id,
                    round_id=round_id,
                    staging_slot=self._landing_slot_base + k,
                    ready_ns=drained_ns,
                    direct=True,
                )
        self._ready_pending = retained
        if selected:
            self._ready_frozen_batches += 1
        if done:
            self._release_ready_rows(dropped, fence_seqs=done)
            self._ready_consume(done)

    def _ready_freeze_status(self, observed_ns: int) -> dict[str, object]:
        """Cumulative FREEZE skip counters (#292) for ``rings.ready``: row
        passes skipped and row-ms charged per reason (rows still waiting are
        charged up to their last pass), the same for rows held at least
        _READY_LONG_HOLD_NS plus their count by largest reason, drops, and
        the reason holding the oldest waiting row (arrival order) and the
        breakdown of the longest hold so far."""

        def by_reason(values, ms=False):
            return {
                name: value / 1e6 if ms else value
                for name, value in zip(_READY_SKIP_REASONS, values)
                if value
            }

        oldest = None
        if self._ready_pending:
            row = self._ready_pending[0]
            state = self._ready_skip_state.get(row[0])
            oldest = {
                "held_ms": (observed_ns - row[8][8]) / 1e6,
                "reason": _READY_SKIP_REASONS[0 if state is None else state[0]],
                "passes": 0 if state is None else state[3],
            }
        return {
            "long_hold_ns": _READY_LONG_HOLD_NS,
            "passes": self._ready_passes,
            "frozen_batches": self._ready_frozen_batches,
            "gate_ms": self._ready_gate_ns / 1e6,
            "late_freeze_ms": self._ready_late_ns / 1e6,
            "skips": by_reason(self._ready_skips),
            "held_ms": by_reason(self._ready_skip_held_ns, ms=True),
            "long_held_ms": by_reason(self._ready_long_held_ns, ms=True),
            "long_rows": by_reason(self._ready_long_rows),
            "drops": dict(zip(_READY_DROP_OUTCOMES, self._ready_drops)),
            "oldest": oldest,
            "longest": self._ready_longest,
        }

    def _ready_row_task(self, handle, transfer_id, deadline_ns, count, fields, rid):
        """The DraftTask of one natively parsed v2 row: the fields the legacy
        take built from the mailbox, without a SourceRef (the row is already
        in its landing buffer) and without _validate_task (ingress applied
        the record's field checks).  Built lazily: only a frozen or retiring
        row needs one (FREEZE evaluates rows by their cell key)."""

        from dt_protocol import DraftTask, DraftTaskKind

        ring_id, record_seq, _, round_id, attempt, seq_len, token_at, bonus, ready_ns = fields[:9]
        tokens = self._ring_worker.ready_tokens
        at = token_at % len(tokens)
        ids = tuple(tokens[at : at + count].tolist())
        return DraftTask(
            transfer_id=transfer_id,
            request_id=rid,
            round_id=round_id,
            attempt=attempt,
            target_instance_id=self._ring_targets[ring_id],
            draft_profile_id=self.profile.profile_id,
            k=self.profile.draft_k,
            next_token_id=bonus,
            source=None,
            created_ns=ready_ns,
            deadline_ns=deadline_ns,
            kind=DraftTaskKind.RECONCILE,
            draft_kv_handle=handle,
            committed_token_ids=ids,
            feature_token_ids=ids,
            committed_kv_len=seq_len,
            feature_token_offset=seq_len - count,
            feature_token_count=count,
            requires_proposal=True,
            direct_ring_seq=record_seq,
            candidate_destination=None,
        )

    def _ready_consume(self, seqs) -> None:
        """Mark READY entries consumed and pass ready_consumed_seq over the
        contiguous consumed prefix (#276 decisions 1-2).  A submitted row is
        consumed only after its cuStreamWaitEvent was issued (§6.5)."""

        table = self._ready_table
        held = time.monotonic_ns() - table["ready_ns"][
            (np.asarray(seqs, dtype=np.int64) - 1) % len(table)
        ].astype(np.int64)
        self._ready_held_max_ns = max(self._ready_held_max_ns, int(held.max()))
        np.add.at(
            self._ready_held_hist,
            (4 * np.log2(np.maximum(held, 1))).astype(np.int64),
            1,
        )
        done = self._ready_done
        for seq in seqs:
            done.add(seq)
            self._ready_tasks.pop(seq, None)
        consumed = self._ready_consumed
        while consumed + 1 in done:
            consumed += 1
            done.remove(consumed)
        if consumed != self._ready_consumed:
            self._ready_consumed = consumed
            if self._ring_worker is not None:  # shutdown: already destroyed
                self._ring_worker.ready_consume(consumed)

    def _submit_batch_descriptor(self, done_seq, event, releases, timing=(), push=()) -> None:
        """Hand one descriptor to the send loop (design §5.1 HAND-OFF (ii)).

        ``done_seq`` 0 is a release-only fence: it carries no timing events
        and its DONE record only retires the fence event.  ``event`` and the
        timing events are recorded on the one compute stream, so descriptors
        complete in the order they are submitted.  ``push`` is the D9 push
        section from _hand_off_pushes (empty: no rows)."""

        descriptor = array(
            "Q",
            (
                done_seq,
                event.cuda_event,
                *(timing or (0, 0, 0)),
                len(releases),
            ),
        )
        for k, gen in releases:
            descriptor.extend((k, gen, 1))
        descriptor.extend(push or (0, 0, 0))
        self._send_loop_worker.submit_batch(descriptor)

    def _hand_off_pushes(self, done_seq, proposal_tasks, candidates):
        """HAND-OFF (i) of D9 (design §5.1, §7.3-7.4): sort a batch's
        proposal rows by how their candidates reach the Target.

        - destination known, same host: pre-queued behind the proposal now
          (``_queue_early_candidate_push``).  A v2 row's destination is read
          from the region here, so a row whose Target already wrote it is
          pre-queued rather than pushed late;
        - same-host v2 row, destination not in the region yet: LATE_LOCAL,
          resolved by the send loop at the GPU done (held until it appears);
        - cross-host: CROSS_HOST, copied and posted by the send loop at the
          GPU done.

        Returns (descriptor push section, the proposal tasks with the
        destinations read here, pre-queue indices).  Every row
        handed to the loop counts as pushed at launch for the retire path, and
        its PUSH_DONE record retires it (``_absorb_push_done``)."""

        from msgspec.structs import replace

        transport = self._candidate_push_transport
        row_bytes = int(candidates.shape[1]) * candidates.element_size()
        base = candidates.data_ptr()
        prequeue, rows, owned, cross = [], [], [], []
        proposal_tasks = list(proposal_tasks)
        sub_ns = self._submit_sub_ns if self._profile_active else None
        for index, task in enumerate(proposal_tasks):
            if task.transfer_id in self._cancelled_ids:
                continue  # as the pre-queue skips it: the retire path owns it
            destination = task.candidate_destination
            if destination is None and task.direct_ring_seq:
                if sub_ns is not None:
                    read_ns = time.monotonic_ns()
                destination = self._resolve_ring_destination(task)
                if sub_ns is not None:
                    sub_ns[3] += time.monotonic_ns() - read_ns
                if destination is not None:
                    task = replace(task, candidate_destination=destination)
                    proposal_tasks[index] = task
            source = base + index * row_bytes
            if destination is None:
                location = self._ring_record_positions.get(task.transfer_id)
                row = (
                    None
                    if location is None
                    else transport.late_push_row(
                        task, source, row_bytes,
                        location[0].destination_entry(task.direct_ring_seq), location[1],
                    )
                )
                if row is not None:
                    rows += row
                    owned.append(task)
            elif destination.target_instance_id in transport._remote:
                cross.append((task, source, row_bytes))
            else:
                prequeue.append(index)
        post_fn = post_arg = 0
        groups = ()
        if cross:
            words, post_fn, post_arg, groups = transport.cross_host_rows(cross)
            rows += words
            owned += [task for task, _source, _bytes in cross]
        if owned:
            self._send_loop_rows[done_seq] = [owned, candidates, len(owned), groups]
            self._early_pushed_ids.update(task.transfer_id for task in owned)
        return (len(owned), post_fn, post_arg, *rows), proposal_tasks, prequeue

    def _absorb_dflash_done(self) -> None:
        """ABSORB (design §5.1): take the send loop's DONE records.

        Phase one pops every completed batch off the in-flight deque, so the
        depth gate is open as soon as the DONE wake reaches this thread;
        phase two does their Python bookkeeping.  A record is published
        before ``done_seq`` and the wake, so every batch the counter covers
        has its record here."""

        worker = self._send_loop_worker
        if worker is None or self._fatal_gpu_quarantine_active():
            return
        stage_started_ns = time.monotonic_ns()
        records = worker.drain_done()
        if not len(records):
            return
        retired = []
        for seq, gpu_done_ns, materialize_ns, proposal_ns, error in records.tolist():
            if not seq:
                # A release-only fence: the send loop has let go of its event.
                self._ready_release_fenced.popleft()
                continue
            completion = self._pending_dflash_completions.popleft()
            if completion.batch_seq + 1 != seq:
                raise RuntimeError(
                    "DFLASH DONE record is not the oldest in-flight batch: "
                    f"record={seq} batch={completion.batch_seq + 1}"
                )
            self._dflash_last_done_ns = gpu_done_ns
            retired.append((completion, gpu_done_ns, materialize_ns, proposal_ns, error))
        absorbed_ns = time.monotonic_ns()
        self._loop_stage_ns["dflash_absorb"] += absorbed_ns - stage_started_ns
        self._loop_stage_calls["dflash_absorb"] += len(retired)
        failed = False
        for completion, gpu_done_ns, materialize_ns, proposal_ns, error in retired:
            if self._profile_active:
                profile_emit(
                    "draft_dflash_done_absorb",
                    draft_instance=self.instance_id,
                    done_seq=completion.batch_seq + 1,
                    gpu_done_ns=gpu_done_ns,
                    absorb_started_ns=stage_started_ns,
                    absorbed_ns=absorbed_ns,
                    # v2: the scheduler's last wait (start, return), so a
                    # done that arrived while it waited can be told apart
                    # from one that arrived during its other work.
                    wait_ns=self._scheduler_wait_ns,
                )
            if failed or error:
                # The send loop could not prove this batch's GPU writes have
                # stopped (design §7.5): keep every allocation it touched.
                # Phase one already popped every record's batch, so the ones
                # behind the failure are quarantined too rather than dropped:
                # nothing else holds them, and a later batch's GPU state is
                # no more certain than the failed one's.
                failed = True
                tasks = [item.task for item in completion.pending]
                self._quarantined_feature_draft_handles.update(
                    task.draft_kv_handle for task in tasks
                )
                self._quarantined_dflash_completions.append(completion)
                logger.critical(
                    "quarantined DFlash completion after a send-loop failure "
                    "(driver rc=%s): transfers=%s",
                    error,
                    [task.transfer_id for task in tasks],
                )
                self._enter_fatal_gpu_quarantine("unquiesced DFlash send-loop failure")
                exc = RuntimeError(f"DFLASH send loop failed with rc={error}")
                exc.resources_quiescent = False
                self._publish_failures([(task, exc) for task in tasks])
                continue
            self._retire_dflash_completion(
                completion,
                completed_ns=gpu_done_ns,
                gpu_materialization_ns=materialize_ns,
                proposal_gpu_ns=proposal_ns,
                stage_started_ns=absorbed_ns,
            )
        if retired:
            self._publish_capacity()

    def _absorb_push_done(self) -> None:
        """ABSORB of D9 (design §5.1, §7.2 step 5): take the send loop's
        PUSH_DONE records instead of polling push events.

        A job record retires the oldest queued push: the publish worker runs
        jobs in order on one stream and reports them in that order.  A row
        record retires one late or cross-host row of a batch: pushed rows as
        a completed push, a row the Target tombstoned or whose destination
        never appeared as a cancelled identity (as the legacy held release
        does).  The cross-host legs' completion queues stay with this thread
        and are reaped here, without waiting: a batch's pushed rows with a
        cross-host group retire only once every group's WRITEs are
        acknowledged (§7.2 step 5), which is their push_complete."""

        worker = self._push_done_worker
        if worker is None or self._fatal_gpu_quarantine_active():
            return
        records = worker.drain_push_done()
        try:
            if self._send_loop_rows or self._send_loop_remote or self._pending_candidate_pushes:
                self._candidate_push_transport.reap_remote()
            retired = self._retire_acknowledged_remote()
        except Exception:
            logger.critical("cross-host candidate push failed", exc_info=True)
            self._enter_fatal_gpu_quarantine("unquiesced cross-host candidate push failure")
            return
        if not len(records):
            if retired:
                self._publish_capacity()
            return
        stage_started_ns = time.monotonic_ns()
        pushed: dict[tuple[int, int], list] = {}
        for kind, ident, row, outcome, issued_ns, done_ns in records.tolist():
            if kind == worker.PUSH_DONE_JOB:
                pending = self._pending_candidate_pushes.popleft()
                if pending.push.ticket != ident:
                    raise RuntimeError(
                        f"PUSH_DONE record {ident} is not the oldest push {pending.push.ticket}"
                    )
                tasks = pending.tasks
            else:
                entry = self._send_loop_rows[ident]
                task, entry[0][row] = entry[0][row], None
                entry[2] -= 1
                if not entry[2]:
                    del self._send_loop_rows[ident]
                tasks = [task]
            if outcome == worker.FAILED:
                exc = RuntimeError("send-loop candidate push failed")
                exc.resources_quiescent = False
                self._enter_fatal_gpu_quarantine("unquiesced send-loop candidate push failure")
                self._publish_failures([(task, exc) for task in tasks])
                return
            if kind == worker.PUSH_DONE_JOB:
                self._retire_candidate_push(pending, done_ns)
            elif outcome == worker.PUSHED:
                group = pushed.setdefault((ident, issued_ns), [done_ns, entry[1], [], entry[3]])
                group[0] = max(group[0], done_ns)
                group[2].append(task)
            else:  # CANCELLED by the Target, or EXPIRED without a destination
                if outcome == worker.EXPIRED:
                    logger.warning(
                        "late push row has no destination at its deadline: %s",
                        task.transfer_id,
                    )
                self._retire_cancelled_transfer_identity(task)
        for (_seq, issued_ns), (done_ns, candidates, tasks, groups) in pushed.items():
            if groups:
                # Posted, not yet acknowledged: retired by the CQ (above).
                self._send_loop_remote.append([groups, issued_ns, candidates, tasks])
            else:
                self._retire_send_loop_push(issued_ns, done_ns, candidates, tasks)
        try:
            self._retire_acknowledged_remote()
        except Exception:
            logger.critical("cross-host candidate push failed", exc_info=True)
            self._enter_fatal_gpu_quarantine("unquiesced cross-host candidate push failure")
            return
        self._loop_stage_ns["push_absorb"] += time.monotonic_ns() - stage_started_ns
        self._loop_stage_calls["push_absorb"] += len(records)
        self._publish_capacity()

    def _retire_acknowledged_remote(self) -> int:
        """Retire the send-loop pushes whose cross-host groups are all
        acknowledged; a failed group raises (fail stop, as a failed push
        event does on the legacy path)."""

        done, waiting = [], []
        for item in self._send_loop_remote:
            (done if all(group.poll() for group in item[0]) else waiting).append(item)
        if done:
            observed_ns = time.monotonic_ns()
            self._send_loop_remote = waiting
            for _groups, issued_ns, candidates, tasks in done:
                self._retire_send_loop_push(issued_ns, observed_ns, candidates, tasks)
        return len(done)

    def _retire_send_loop_push(self, issued_ns, done_ns, candidates, tasks) -> None:
        """Retire rows the send loop pushed (one late job of one batch)."""

        from sglang.srt.speculative.dt_candidate_p2p import CandidatePush

        push = CandidatePush(
            event=None,
            transfer_ids=tuple(task.transfer_id for task in tasks),
            submitted_ns=issued_ns,
            candidate_bytes=len(tasks) * candidates.shape[1] * candidates.element_size(),
            cold_mapping_count=0,
        )
        if self._profile_active:
            # The send loop issued these at the GPU done (D0 chain:
            # push_submit is that issue, push_complete its observation).
            profile_emit(
                "draft_candidate_push_submit",
                draft_instance=self.instance_id,
                transfer_ids=push.transfer_ids,
                entries=len(tasks),
                candidate_bytes=push.candidate_bytes,
                cold_mapping_count=0,
                submitted_ns=issued_ns,
                ticket=0,
                send_loop=True,
            )
        self._retire_candidate_push(
            PendingCandidatePush(push=push, tasks=tasks, result_slot=None, candidates=None),
            done_ns,
            notice=False,
        )

    def _ready_row_keys(self, seqs) -> tuple[tuple[int, int], ...]:
        """(ring id, record seq) of each READY row, for its landing release."""

        rows = self._ready_table[(np.asarray(seqs, dtype=np.int64) - 1) % len(self._ready_table)]
        return tuple(zip(rows["ring_id"].tolist(), rows["record_seq"].tolist()))

    def _wait_ready_copies(self, seqs) -> None:
        """The current stream waits on the copy event of every READY row
        (one cuStreamWaitEvent per distinct event; cross-host rows have
        none), before the batch reads their landing rows (§5.1, §6.5)."""

        worker = self._ring_worker
        rows = self._ready_table[(np.asarray(seqs, dtype=np.int64) - 1) % len(self._ready_table)]
        no_event = worker.ready_kinds["NO_EVENT"]
        index = rows["copy_event_idx"]
        with_event = index != no_event
        if with_event.any():
            # Dedup the (idx, gen) pairs with numpy (the wait is
            # order-free); the old per-row Python set of tuples cost ~100 µs
            # per batch (#353 copy_waits).
            pairs = np.stack(
                (index[with_event].astype(np.int64), rows["copy_event_gen"][with_event].astype(np.int64)),
                axis=1,
            )
            unique = np.unique(pairs, axis=0)
            worker.wait_copy_events(
                torch.cuda.current_stream(self.device).cuda_stream,
                unique[:, 0].astype(np.uint32),
                unique[:, 1].astype(np.uint32),
            )

    def _resolve_ready_releases(self, keys) -> tuple[tuple[int, int], ...]:
        """Drop one row from each named record and return the ``(k, gen)`` of
        the records whose last row went: their landing placement is now free.

        A record is placed once by ingress and released once, with its last
        row, however its rows are split across batches."""

        records = self._ready_records
        freed = []
        for key in keys:
            record = records[key]
            record[0] -= 1
            if not record[0]:
                del records[key]
                freed.append((record[1], record[2]))
        return tuple(freed)

    def _release_ready_rows(self, keys, fence_seqs=()) -> None:
        """Release READY rows' landing placements.  Rows released without a
        batch that waited on their copies pass ``fence_seqs`` (their READY
        seqs, not yet consumed): READY precedes the copy's completion (§6.5),
        so their release waits until the current stream has passed those
        copies.

        Under the D8 send loop the fence is a release-only descriptor, so
        every landing release reaches ingress from the one producer thread
        the queue allows (§4.3); otherwise the scheduler pushes them itself,
        and polls the fence events in DRAIN."""

        worker = self._ring_worker
        if fence_seqs and worker is not None:
            if self._send_loop_worker is not None:
                releases = self._resolve_ready_releases(keys)
                if not releases:
                    return  # these records still have live rows
                self._wait_ready_copies(fence_seqs)
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(self.device))
                # Keep the event alive until its seq-0 DONE record proves the
                # send loop is done with it (ABSORB pops this entry then).
                self._ready_release_fenced.append((event, ()))
                self._submit_batch_descriptor(0, event, releases)
                return
            self._wait_ready_copies(fence_seqs)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            self._ready_release_fenced.append((event, tuple(keys)))
            return
        for k, gen in self._resolve_ready_releases(keys):
            if worker is not None:  # shutdown: the worker already drained
                worker.landing_release(k, gen, 1)

    def _profile_ring_admission_state(
        self,
        *,
        observed_ns: int,
        reason: str,
        queue_len: int,
        head_transfer_id: str,
        available_rows: int,
        available_kv: int,
    ) -> None:
        """Emit only transitions of the ring-admission gate (#267 §8.1).

        The FIFO wait between ``draft_ring_take`` and the ``ready_ns`` of
        ``draft_input_ready`` is the largest per-row leg (#267 §2), but the
        gate that holds it had no event of its own.  Same transition-only
        shape as ``_profile_input_ready_state``: one event per change of
        (reason, head), i.e. at most about the selection rate.
        """

        available_credit = (
            None if DRAFT_ADMISSION_BOUND
            else self._available_direct_proposal_admission()
        )
        signature = (reason, head_transfer_id)
        if signature == getattr(self, "_last_ring_admission_profile_state", None):
            return
        self._last_ring_admission_profile_state = signature
        profile_emit(
            "draft_ring_admission_state",
            draft_instance=self.instance_id,
            observed_ns=observed_ns,
            reason=reason,
            queue_len=queue_len,
            head_transfer_id=head_transfer_id,
            available_rows=available_rows,
            available_kv=available_kv,
            available_credit=available_credit,
        )

    def _free_landing(self, index: int) -> None:
        taken = self._landing_taken[index]
        worker = self._ring_worker
        if worker is None:
            # Shutdown: the worker is gone and its stream drained.
            self._landing_taken[index] = 0
            return
        if not taken:
            return
        if self._slot_decouple:
            # DFlash releases a row only after its batch's completion event,
            # so no GPU read is left.  The worker frees only a buffer it sealed
            # on overflow; until then this is retried every ring pass.
            if worker.try_free(index, taken) == 1:
                self._landing_taken[index] = 0
            return
        # The last consumer may have been launched just now (synchronous
        # EAGLE3 publish): return the buffer only after that GPU read.
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        self._landing_free_pending.append((index, event))

    def _close_direct_rings(self) -> None:
        """Stop the ring worker (drains its copies) and retire unadmitted work.

        Endpoints stay SERVING; ``_close_direct_ring_endpoints`` marks them
        CLOSED only after every candidate push has drained, because the
        Target treats CLOSED as proof that no further push can land.
        """

        worker = getattr(self, "_ring_worker", None)
        if worker is None:
            return
        if self._send_loop_worker is not None:
            # Every outstanding descriptor still owes this worker its landing
            # releases; the publish worker outlives it (it is closed with the
            # push transport), so it must hold none before it is destroyed.
            deadline_ns = time.monotonic_ns() + _RING_RECORD_TIMEOUT_NS
            while self._send_loop_worker.batch_stats()[1]:
                if time.monotonic_ns() > deadline_ns:
                    raise RuntimeError("send loop still holds batch descriptors")
                time.sleep(0)
            # Every batch has published its DONE record by now.  Absorb them
            # before the handle goes: afterwards nothing can retire them, and
            # the shutdown drain would spin on a deque it cannot shorten.
            self._absorb_dflash_done()
            self._send_loop_worker = None
        self._ring_worker = None
        rc = worker.destroy()
        if rc:
            raise RuntimeError(f"direct ring worker stream drain failed: {rc}")
        self._landing_free_pending.clear()
        # Unfrozen READY rows hold no request state; their placements died
        # with the worker.
        self._ready_pending.clear()
        self._ready_skip_state.clear()
        self._ready_release_fenced.clear()
        queue = self._ring_admission
        while queue:
            task, slot, _rows = queue.popleft()
            self._cancelled_ids.add(task.transfer_id)
            self._release_persistent_request(task.draft_kv_handle)
            refcounts = self._feature_staging_refcounts
            remaining = refcounts.get(slot, 0) - 1
            if remaining > 0:
                refcounts[slot] = remaining
            else:
                refcounts.pop(slot, None)

    def _close_direct_ring_endpoints(self) -> None:
        for endpoint in getattr(self, "_remote_ring_endpoints", {}).values():
            endpoint.close()
        self._remote_ring_endpoints.clear()
        for endpoint in getattr(self, "_ring_endpoints", {}).values():
            endpoint.close()
        self._ring_endpoints.clear()

    def _take_early_pushed(self, tasks) -> set[str]:
        """Remove and return the transfers whose push was queued at launch."""

        early = getattr(self, "_early_pushed_ids", None)
        if not early:
            return set()
        taken = {task.transfer_id for task in tasks if task.transfer_id in early}
        early.difference_update(taken)
        return taken

    def _draft_compute_batches_inflight(self) -> int:
        count = (
            len(self._pending_flashmtp_completions)
            + len(self._pending_dflash_completions)
            + len(self._pending_eagle3_completions)
        )
        if count > DRAFT_COMPUTE_DEPTH:
            raise RuntimeError(
                "Draft compute invariant violated: more than one executing plus "
                "one submitted batch"
            )
        return count

    def stop(self) -> None:
        self._stop = True

    def _fatal_gpu_quarantine_active(self) -> bool:
        return bool(getattr(self, "_fatal_gpu_quarantine_reason", None))

    def _enter_fatal_gpu_quarantine(self, reason: str) -> None:
        """Stop GPU work permanently after a failed quiescence fence.

        This is deliberately distinct from ``stop()``.  A normal stop drains
        known-complete work and releases owned resources.  Fatal quarantine
        preserves every uncertain allocation and lease for CUDA context/process
        teardown instead.
        """

        if not reason:
            raise ValueError("fatal GPU quarantine requires a reason")
        previous = getattr(self, "_fatal_gpu_quarantine_reason", None)
        if previous is None:
            self._fatal_gpu_quarantine_reason = reason
        # Commit the engine state before touching either independently
        # scheduled copy worker.  Transport notification is deliberately
        # non-destructive, but it is still fallible; neither a notification nor
        # logging failure may reopen ordinary serving or shutdown.
        self._stop = True
        transports = (
            ("migration", getattr(self, "_p2p_transport", None)),
            ("feature-pull", getattr(self, "_feature_pull_transport", None)),
        )
        for transport_name, transport in transports:
            quarantine_transport = (
                None if transport is None else getattr(transport, "quarantine", None)
            )
            if quarantine_transport is None:
                continue
            try:
                quarantine_transport()
            except Exception:
                logger.exception(
                    "failed to notify %s transport of fatal GPU quarantine; "
                    "engine quarantine remains latched",
                    transport_name,
                )
        if previous is None:
            logger.critical(
                "entered fatal GPU quarantine; all later GPU work is disabled: %s",
                reason,
            )
        elif previous != reason:
            logger.critical(
                "additional fatal GPU quarantine condition ignored after %s: %s",
                previous,
                reason,
            )

    def _raise_if_fatal_gpu_quarantine(self, operation: str) -> None:
        reason = getattr(self, "_fatal_gpu_quarantine_reason", None)
        if reason is None:
            return
        raise RuntimeError(
            f"fatal GPU quarantine prevents {operation}; uncertain GPU resources "
            f"must remain owned until process teardown: {reason}"
        )

    def _lifecycle_status(self) -> dict[str, object]:
        req_pool = getattr(self._draft_model_runner, "req_to_token_pool", None)
        allocator = getattr(
            self._draft_model_runner, "token_to_kv_pool_allocator", None
        )
        active = {
            "pending_tasks": len(self._pending),
            "pending_task_ids": len(self._pending_ids),
            "received_timestamps": len(self._received_ns_by_transfer),
            "persistent_requests": len(self._persistent_requests),
            "published_result_batches": len(self._published_by_batch),
            "published_result_ranges": len(self._published_by_range),
            "delivery_outbox": len(self._delivery_outbox),
            "source_done_pending": len(self._source_done_pending),
            "source_done_batch_entries": len(self._source_done_batch_by_entry),
            "source_done_batches": len(self._source_done_batch_remaining),
            "pending_feature_batches": len(self._pending_feature_batches),
            "input_ready_features": len(self._input_ready_features),
            "feature_staging_owned": len(self._feature_staging_refcounts),
            "draft_compute_batches_inflight": self._draft_compute_batches_inflight(),
            "pending_flashmtp_completions": len(
                getattr(self, "_pending_flashmtp_completions", ())
            ),
            "pending_dflash_completions": len(self._pending_dflash_completions),
            "pending_eagle3_completions": len(self._pending_eagle3_completions),
            "pending_candidate_pushes": len(self._pending_candidate_pushes),
            "pending_candidate_completion_notices": (
                0
                if getattr(self, "_candidate_push_transport", None) is None
                else self._candidate_push_transport.pending_completion_count
            ),
            "prepared_migrations": len(self._prepared_migrations),
            "deferred_migration_gpu_controls": len(
                self._deferred_migration_gpu_controls
            ),
            "migration_source_events": len(self._migration_source_events),
            "migration_source_handles": len(self._migration_source_handles),
            "deferred_persistent_releases": len(self._deferred_persistent_releases),
            "pending_request_release_acks": len(self._pending_request_release_acks),
            "deferred_migration_finalizations": len(
                self._deferred_migration_finalizations
            ),
            "deferred_feature_source_tasks": len(
                self._deferred_feature_source_tasks
            ),
            "feature_source_inflight": len(self._feature_source_inflight),
            "feature_source_quiescent": len(self._feature_source_quiescent),
            "migration_quiescence_probes": len(self._migration_quiescence_probes),
            "migration_quiescence_retries": len(
                self._migration_quiescence_retry_ns
            ),
        }
        fatal = {
            "feature_source_unquiescent": len(self._feature_source_unquiescent),
            "quarantined_feature_draft_handles": len(
                self._quarantined_feature_draft_handles
            ),
            "quarantined_flashmtp_completions": len(
                getattr(self, "_quarantined_flashmtp_completions", ())
            ),
            "quarantined_dflash_completions": len(
                self._quarantined_dflash_completions
            ),
            "quarantined_eagle3_completions": len(
                self._quarantined_eagle3_completions
            ),
            "quarantined_migrations": len(self._quarantined_migration_ids),
        }
        resources = {
            "free_request_rows": (
                0 if req_pool is None else int(req_pool.available_size())
            ),
            "free_kv_pages": (
                0 if allocator is None else int(allocator.available_size())
            ),
            "free_result_slots": len(self._free_result_slots),
            "free_feature_staging_slots": len(self._free_feature_staging_slots),
            "feature_staging_slots_total": len(self._feature_staging_buffers),
        }
        observed_ns = time.monotonic_ns()
        quiescent = not any(active.values()) and not any(fatal.values())
        rings = {
            "remote": {
                session_id: {
                    "records_taken": endpoint.records_taken,
                    "next_seq": endpoint.next_seq,
                }
                for session_id, endpoint in self._remote_ring_endpoints.items()
            },
        }
        if self._native_ingress:
            # READY residence, publish -> consumed (#276 decision 2); the
            # histogram is cumulative, bucket b covers [2**(b/4), 2**((b+1)/4)) ns.
            hist = self._ready_held_hist
            total = int(hist.sum())
            p99 = int(np.searchsorted(np.cumsum(hist), 0.99 * total)) if total else 0
            rings["ready"] = {
                "rows": len(self._ready_pending),
                "consumed_seq": self._ready_consumed,
                "held_max_ns": self._ready_held_max_ns,
                "held_p99_ns": int(2 ** ((p99 + 1) / 4)) if total else 0,
                "held_hist": {int(b): int(hist[b]) for b in np.flatnonzero(hist)},
                "freeze": self._ready_freeze_status(observed_ns),
            }
        last_measurement_ns = getattr(self, "_last_direct_gpu_measurement_ns", 0)
        request_evidence_ready = bool(
            quiescent
            and last_measurement_ns
            and observed_ns - last_measurement_ns
            >= DT_DIRECT_GPU_REQUEST_EVIDENCE_IDLE_NS
        )
        return {
            "schema_version": "dt-lifecycle-status/v1",
            "role": "draft",
            "instance_id": self.instance_id,
            "pid": os.getpid(),
            "observed_ns": observed_ns,
            "sequence": self._lifecycle_status_sequence,
            "instance_session_id": self._client.instance_session_id,
            "coordinator_session_id": self._client.coordinator_session_id,
            "result_pool_generation": self._result_pool_generation,
            "kv_pool_generation": (
                "" if self._kv_pool is None else self._kv_pool.pool_generation
            ),
            "direct_gpu_measurement_epochs": (
                self._direct_gpu_measurement_epoch_evidence()
            ),
            "direct_gpu_measurement_requests": (
                self._direct_gpu_measurement_request_evidence()
                if request_evidence_ready
                else []
            ),
            "active": active,
            "rings": rings,
            "fatal": fatal,
            "resources": resources,
            "terminal": {
                "cancelled_transfers": len(self._cancelled_ids),
                "completed_pull_done": len(self._pull_done_applied_ids),
                "migration_roles": len(self._migration_roles),
                "migration_decisions": len(self._migration_decisions),
                "source_store_count": 0,
            },
            "quiescent": quiescent,
        }

    def _maybe_emit_lifecycle_status(self) -> None:
        now_ns = time.monotonic_ns()
        if (
            now_ns - self._last_lifecycle_status_sample_ns
            < DT_LIFECYCLE_STATUS_INTERVAL_NS
        ):
            return
        self._last_lifecycle_status_sample_ns = now_ns
        if self._profile_active:
            self._dump_native_probes()
        # Capacity sends are nonblocking. Refresh cumulative credit evidence so
        # a locally backpressured final snapshot cannot strand all admission.
        self._publish_capacity()
        status = self._lifecycle_status()
        quiescent = bool(status["quiescent"])
        if (
            quiescent == self._last_lifecycle_quiescent
            and (
                not quiescent
                or now_ns - self._last_lifecycle_status_ns
                < DT_LIFECYCLE_STATUS_INTERVAL_NS
            )
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

    def _dump_native_probes(self) -> None:
        """Drain the native workers' D0 stamps into the profile (issue #269)."""

        worker = self._ring_worker
        if worker is not None and worker.probe is not None:
            worker.probe.dump()
        transport = self._candidate_push_transport
        if transport is not None:
            transport.dump_native_probe()

    def _quiesce_direct_gpu_source_shutdown(self) -> None:
        # Fence new coordinator admission before retiring already-dispatched
        # source leases.
        self._publish_capacity()
        # Publishing unhealthy capacity is control-only.  Everything below can
        # fence, query, close, or release CUDA-owned state and is forbidden once
        # an earlier fence failed.
        self._raise_if_fatal_gpu_quarantine("direct-GPU-source shutdown")
        queued = tuple(self._pending)
        for task in queued:
            self._cancelled_ids.add(task.transfer_id)
            if self._uses_direct_gpu_source and not task.source_read_migration_id:
                self._publish_source_done(task)
            self._pending_ids.discard(task.transfer_id)
            getattr(self, "_ring_pending_ids", set()).discard(task.transfer_id)
            self._received_ns_by_transfer.pop(task.transfer_id, None)
        self._pending.clear()
        for batch in self._pending_feature_batches:
            self._cancelled_ids.update(item.task.transfer_id for item in batch.pending)
        for entry in getattr(self, "_input_ready_features", ()):
            self._cancelled_ids.add(entry.pending.task.transfer_id)
        for completion in getattr(self, "_pending_flashmtp_completions", ()):
            self._cancelled_ids.update(
                item.task.transfer_id for item in completion.pending
            )
        for completion in self._pending_dflash_completions:
            self._cancelled_ids.update(
                item.task.transfer_id for item in completion.pending
            )
        for completion in self._pending_eagle3_completions:
            self._cancelled_ids.update(
                item.task.transfer_id for item in completion.pending
            )
        self._close_direct_rings()
        if self._feature_pull_transport is not None:
            self._feature_pull_transport.close()
        input_ready = getattr(self, "_input_ready_features", deque())
        while input_ready:
            entry = input_ready.popleft()
            if self._persistent:
                self._release_persistent_request(entry.pending.task.draft_kv_handle)
            self._release_feature_staging_entries(entry.staging_slot, 1)
        if getattr(self, "_quarantined_eagle3_completions", ()):
            raise RuntimeError(
                "feature-backed Draft shutdown requires process teardown after "
                "an EAGLE3 completion fence failure"
            )
        if getattr(self, "_quarantined_flashmtp_completions", ()):
            raise RuntimeError(
                "FlashMTP Draft shutdown requires process teardown after a "
                "candidate completion fence failure"
            )
        # Drain every already-copied batch and its source read-done lifecycle.
        # Serial mode preserves its single-completion ordering; pipeline mode
        # retains the same per-slot ownership used by the serving loop.
        while (
            self._pending_feature_batches
            or getattr(self, "_pending_flashmtp_completions", ())
            or self._pending_dflash_completions
            or self._pending_eagle3_completions
            or getattr(self, "_pending_candidate_pushes", ())
            or getattr(self, "_send_loop_rows", None)
            or getattr(self, "_send_loop_remote", None)
        ):
            self._poll_feature_batches()
            self._raise_if_fatal_gpu_quarantine("shutdown feature-batch drain")
            if getattr(self, "_pending_flashmtp_completions", ()):
                completion = self._pending_flashmtp_completions[0]
                try:
                    completion.completion_event.synchronize()
                except Exception:
                    self._enter_fatal_gpu_quarantine(
                        "unquiesced FlashMTP shutdown completion-event failure"
                    )
                    self._pending_flashmtp_completions.popleft()
                    self._quarantined_flashmtp_completions.append(completion)
                    raise
                self._poll_flashmtp_completions()
                self._raise_if_fatal_gpu_quarantine(
                    "shutdown FlashMTP completion drain"
                )
            if self._pending_dflash_completions:
                completion = self._pending_dflash_completions[0]
                try:
                    completion.completion_event.synchronize()
                except Exception:
                    # The event failed to prove that model-stream writes have
                    # stopped.  Latch both executor fences before inspecting
                    # or reporting task state; the completion stays at the
                    # head of the queue and retains its staging/KV ownership.
                    self._enter_fatal_gpu_quarantine(
                        "unquiesced DFlash shutdown completion-event failure"
                    )
                    self._quarantined_feature_draft_handles.update(
                        item.task.draft_kv_handle for item in completion.pending
                    )
                    raise
                if self._send_loop:
                    # The send loop observes the done event and records it;
                    # this turn absorbs whatever has arrived, and the outer
                    # loop comes back until the deque is empty.
                    self._absorb_dflash_done()
                else:
                    self._poll_dflash_completions()
                self._raise_if_fatal_gpu_quarantine(
                    "shutdown DFlash completion drain"
                )
            if self._pending_eagle3_completions:
                completion = self._pending_eagle3_completions[0]
                try:
                    completion.completion_event.synchronize()
                except Exception:
                    # As with the normal EAGLE3 query-failure path, move the
                    # uncertain completion to permanent quarantine.  Enter the
                    # common fatal state first so a concurrent migration
                    # callback cannot reach CUDA while ownership is recorded.
                    self._enter_fatal_gpu_quarantine(
                        "unquiesced EAGLE3 shutdown completion-event failure"
                    )
                    self._pending_eagle3_completions.popleft()
                    self._quarantined_eagle3_completions.append(completion)
                    self._quarantined_feature_draft_handles.update(
                        item.task.draft_kv_handle for item in completion.pending
                    )
                    raise
                self._poll_eagle3_completions()
                self._raise_if_fatal_gpu_quarantine(
                    "shutdown EAGLE3 completion drain"
                )
                if self._quarantined_eagle3_completions:
                    raise RuntimeError(
                        "feature-backed Draft shutdown requires process teardown "
                        "after an EAGLE3 completion fence failure"
                    )
            if getattr(self, "_push_done_worker", None) is not None and not (
                self._pending_dflash_completions
            ):
                # D9: every batch is absorbed, so no row can join the held
                # list; give up the held ones and take the records.
                self._push_done_worker.expire_held()
                self._absorb_push_done()
                self._raise_if_fatal_gpu_quarantine(
                    "shutdown send-loop push drain"
                )
                time.sleep(0)
            elif getattr(self, "_pending_candidate_pushes", ()):
                push = self._pending_candidate_pushes[0]
                try:
                    push.push.event.synchronize()
                except Exception:
                    self._enter_fatal_gpu_quarantine(
                        "unquiesced direct candidate shutdown push failure"
                    )
                    raise
                self._poll_candidate_pushes()
                self._raise_if_fatal_gpu_quarantine(
                    "shutdown direct candidate push drain"
                )
        if getattr(self, "_candidate_push_transport", None) is not None:
            completion_deadline_ns = time.monotonic_ns() + 10_000_000_000
            while self._candidate_push_transport.pending_completion_count:
                self._candidate_push_transport.progress_completions()
                if time.monotonic_ns() >= completion_deadline_ns:
                    raise RuntimeError(
                        "Draft shutdown timed out waiting for direct completion ACKs"
                    )
                time.sleep(IDLE_POLL_S)
        while self._poll_deferred_migration_gpu_controls():
            pass
        for deferred_key, task in tuple(self._deferred_feature_source_tasks.items()):
            if deferred_key in self._feature_source_unquiescent:
                continue
            self._publish_source_done(task)
            self._deferred_feature_source_tasks.pop(deferred_key, None)
            self._feature_source_inflight.discard(deferred_key)
            self._feature_source_quiescent.discard(deferred_key)
        self._flush_source_done()
        deadline_ns = time.monotonic_ns() + 2_000_000_000
        while any(
            operation == "source_read_done"
            for operation, _message_id, _message in self._delivery_outbox.pending()
        ):
            # Reuse the normal receiver so result leases and concurrent
            # lifecycle controls continue to make progress.  Any dispatch that
            # was already in flight is rejected locally by immediately
            # acknowledging its unread source descriptor.
            self._poll_messages()
            for task in tuple(self._pending):
                self._cancelled_ids.add(task.transfer_id)
                if not task.source_read_migration_id:
                    self._publish_source_done(task)
                self._pending_ids.discard(task.transfer_id)
                getattr(self, "_ring_pending_ids", set()).discard(task.transfer_id)
                self._received_ns_by_transfer.pop(task.transfer_id, None)
            self._pending.clear()
            self._flush_source_done()
            if time.monotonic_ns() >= deadline_ns:
                raise RuntimeError(
                    "direct GPU source shutdown timed out waiting for read-done ACKs"
                )
            time.sleep(IDLE_POLL_S)
        self._feature_pull_transport = None
        if (
            self._feature_source_unquiescent
            or self._quarantined_feature_draft_handles
        ):
            raise RuntimeError(
                "feature-backed Draft shutdown requires process teardown after "
                "an unquiesced GPU operation"
            )

    def _fence_unresolved_migration_shutdown(self) -> None:
        """Fail closed while any Draft-KV migration ownership is unresolved.

        A prepared destination remains authoritative until completion and
        finalization retire its reservation.  A source event/handle remains a
        remote-reader lease over the persistent request's backing KV.  Closing
        the local executor proves neither lifecycle, so shutdown must retain
        all of these objects for process teardown instead of clearing maps or
        releasing request rows/pages.
        """

        ownership = {
            "prepared": tuple(sorted(getattr(self, "_prepared_migrations", {}))),
            "source_events": tuple(
                sorted(getattr(self, "_migration_source_events", {}))
            ),
            "source_handles": tuple(
                sorted(getattr(self, "_migration_source_handles", {}))
            ),
            "quarantined": tuple(
                sorted(getattr(self, "_quarantined_migration_ids", ()))
            ),
            "deferred_finalizations": tuple(
                sorted(getattr(self, "_deferred_migration_finalizations", {}))
            ),
            "quiescence_probes": tuple(
                sorted(getattr(self, "_migration_quiescence_probes", {}))
            ),
        }
        unresolved = {name: ids for name, ids in ownership.items() if ids}
        if not unresolved:
            return
        reason = (
            "unresolved Draft KV migration ownership during shutdown: "
            + json.dumps(unresolved, sort_keys=True, separators=(",", ":"))
        )
        # Latch and fence both independently scheduled GPU transports before
        # raising or reporting. No map, event, handle, request, reservation, or
        # backing allocation is modified by this transition.
        self._enter_fatal_gpu_quarantine(reason)
        self._raise_if_fatal_gpu_quarantine("Draft migration shutdown")

    def close(self) -> None:
        if self._uses_direct_gpu_source or getattr(self, "continuous_batching", False):
            self._quiesce_direct_gpu_source_shutdown()
        self._close_direct_rings()
        # Direct-source shutdown checks this immediately after its control-only
        # capacity publication.  Keep the same fail-closed guarantee for any
        # future/non-feature path that enters the common fatal state: neither
        # migration transport shutdown nor request cleanup may touch uncertain
        # GPU-owned resources.
        self._raise_if_fatal_gpu_quarantine("Draft engine shutdown")
        self._fence_unresolved_migration_shutdown()
        if self._p2p_transport is not None:
            self._p2p_transport.close()
        if self._candidate_push_transport is not None:
            self._candidate_push_transport.close()
        # Push stream drained above: CLOSED now means no further pushes.
        self._close_direct_ring_endpoints()
        for draft_kv_handle in tuple(self._persistent_requests):
            self._release_persistent_request(draft_kv_handle)
        self._client.close()
        if self._wake is not None:
            self._wake.close()
        if self._result_store is not None:
            self._result_store.close()
            self._result_store.unlink()

    def _poll_messages(self) -> None:
        from dt_protocol import (
            CancelTransfers,
            ControlAck,
            DraftDispatchBatch,
            DraftKvMigrationExport,
            DraftKvMigrationFinalize,
            DraftKvMigrationImport,
            DraftKvMigrationPrepare,
            DraftRequestRelease,
            ErrorMessage,
            PullDone,
            ResultBatchAck,
            SourceReadDoneBatchAck,
        )

        capacity_changed = False
        self._flush_source_done()
        if getattr(self, "_unconfirmed_direct", None):
            self._expire_unconfirmed_direct()
        capacity_changed = self._poll_migration_transfers() or capacity_changed
        for message in self._client.recv_nowait():
            if isinstance(message, DraftDispatchBatch):
                capacity_changed = self._accept_dispatch(message) or capacity_changed
            elif isinstance(message, CancelTransfers):
                self._publish_candidate_cancellations(message.transfer_ids)
                published_ids = {
                    transfer_id
                    for published in self._published_by_batch.values()
                    for transfer_id in published.transfer_ids
                }
                self._cancelled_ids.update(
                    transfer_id
                    for transfer_id in message.transfer_ids
                    if transfer_id not in published_ids
                )
                # A §7.1 push held for confirmation of a now-cancelled wave is
                # dropped here; releasable ones leave on the next confirm.
                self._release_held_ring_pushes()
                capacity_changed = True
            elif isinstance(message, DraftRequestRelease):
                if message.draft_instance_id != self.instance_id:
                    raise ValueError("DraftRequestRelease belongs to another Draft")
                if (
                    message.instance_session_id != self._client.instance_session_id
                    or message.coordinator_session_id
                    != self._client.coordinator_session_id
                ):
                    logger.warning(
                        "ignoring stale DraftRequestRelease: request=%s handle=%s",
                        message.request_id,
                        message.draft_kv_handle,
                    )
                    continue
                from dt_protocol import draft_request_release_id

                release_id = draft_request_release_id(message)
                previous = self._pending_request_release_acks.get(release_id)
                if previous is not None and previous != message:
                    raise ValueError(
                        "DraftRequestRelease conflicts with a pending release"
                    )
                self._pending_request_release_acks[release_id] = message
                # A shutdown/abort release can overtake local execution of an
                # already-dispatched task.  Cancel matching queued work before
                # freeing the row so bootstrap cannot recreate it afterward.
                self._cancelled_ids.update(
                    task.transfer_id
                    for task in self._pending
                    if task.draft_kv_handle == message.draft_kv_handle
                    and task.request_id == message.request_id
                )
                self._cancelled_ids.update(
                    item.task.transfer_id
                    for batch in self._pending_feature_batches
                    for item in batch.pending
                    if item.task.draft_kv_handle == message.draft_kv_handle
                    and item.task.request_id == message.request_id
                )
                self._cancelled_ids.update(
                    entry.pending.task.transfer_id
                    for entry in getattr(self, "_input_ready_features", ())
                    if entry.pending.task.draft_kv_handle
                    == message.draft_kv_handle
                    and entry.pending.task.request_id == message.request_id
                )
                self._cancelled_ids.update(
                    item.task.transfer_id
                    for batch in self._pending_dflash_completions
                    for item in batch.pending
                    if item.task.draft_kv_handle == message.draft_kv_handle
                    and item.task.request_id == message.request_id
                )
                self._cancelled_ids.update(
                    item.task.transfer_id
                    for batch in self._pending_eagle3_completions
                    for item in batch.pending
                    if item.task.draft_kv_handle == message.draft_kv_handle
                    and item.task.request_id == message.request_id
                )
                # Ring rounds: admitted copies hold the handle; waiting
                # Coordinator copies make their ring copy drop on arrival.
                self._cancelled_ids.update(
                    queued_task.transfer_id
                    for queued_task in (
                        *(item[0] for item in getattr(self, "_ring_admission", ())),
                        *getattr(self, "_ring_waiting", {}).values(),
                    )
                    if queued_task.draft_kv_handle == message.draft_kv_handle
                    and queued_task.request_id == message.request_id
                )
                self._release_persistent_request(
                    message.draft_kv_handle, request_id=message.request_id
                )
                self._ack_applied_request_releases(message.draft_kv_handle)
                capacity_changed = True
            elif isinstance(message, DraftKvMigrationPrepare):
                self._handle_migration_prepare(message)
                capacity_changed = True
            elif isinstance(message, DraftKvMigrationExport):
                self._handle_migration_export(message)
                capacity_changed = True
            elif isinstance(message, DraftKvMigrationImport):
                self._handle_migration_import(message)
                capacity_changed = True
            elif isinstance(message, DraftKvMigrationFinalize):
                self._handle_migration_finalize(message)
                capacity_changed = True
            elif isinstance(message, ResultBatchAck):
                pending_result = self._delivery_outbox.get(
                    "result", message.batch_id
                )
                self._delivery_outbox.ack("result", message.batch_id)
                if pending_result is not None:
                    for entry in pending_result.entries:
                        self._pending_ids.discard(entry.transfer_id)
                        getattr(self, "_ring_pending_ids", set()).discard(entry.transfer_id)
                    capacity_changed = True
                published = self._published_by_batch.get(message.batch_id)
                if published is not None:
                    accepted = set(message.accepted_transfer_ids)
                    if not accepted:
                        self._release_result(published)
                        capacity_changed = True
                    else:
                        self._rebuild_accepted_ranges(published, accepted)
                        if not published.range_transfer_ids:
                            self._release_result(published)
                            capacity_changed = True
            elif isinstance(message, PullDone):
                capacity_changed = self._handle_pull_done(message) or capacity_changed
            elif isinstance(message, ControlAck):
                if message.operation == "source_read_done":
                    self._ack_source_done(message.message_id)
                else:
                    self._delivery_outbox.ack(message.operation, message.message_id)
                    if message.operation == "pull_done":
                        self._pull_done_applied_ids.discard(message.message_id)
            elif isinstance(message, SourceReadDoneBatchAck):
                self._handle_source_done_batch_ack(message)
            elif isinstance(message, ErrorMessage):
                if message.code == "PermissionError":
                    self._client.register_draft(timeout_s=2.0)
                    self._delivery_outbox.retry_all_now()
                if (
                    message.code == "COORDINATOR_BACKPRESSURE"
                    and self._delivery_outbox.get("result", message.related_id)
                    is not None
                ):
                    self._delivery_outbox.retry_now("result", message.related_id)
                logger.warning(
                    "Draft Coordinator error code=%s related_id=%s message=%s",
                    message.code,
                    message.related_id,
                    message.message,
                )
            else:
                raise TypeError(
                    f"unexpected Draft Coordinator message {type(message).__name__}"
                )
        capacity_changed = (
            self._poll_deferred_migration_gpu_controls() or capacity_changed
        )
        if capacity_changed:
            self._drop_cancelled_pending()
            self._publish_capacity()

    @staticmethod
    def _ring_record_task_matches(direct, task) -> bool:
        """Identity/fence equality of a §7.1 record task and its confirmation."""

        if (
            direct.transfer_id != task.transfer_id
            or direct.request_id != task.request_id
            or direct.round_id != task.round_id
            or direct.attempt != task.attempt
            or direct.target_instance_id != task.target_instance_id
            or direct.draft_kv_handle != task.draft_kv_handle
            or direct.kind != task.kind
            or direct.requires_proposal != task.requires_proposal
            or direct.next_token_id != task.next_token_id
            or direct.feature_token_offset != task.feature_token_offset
            or direct.feature_token_count != task.feature_token_count
            or direct.committed_kv_len != task.committed_kv_len
            or direct.committed_token_ids != task.committed_token_ids
            or direct.feature_token_ids != task.feature_token_ids
        ):
            return False
        source, other = direct.source, task.source
        if source is None or other is None:
            return source is None and other is None
        # A wave split across several records names each request's rows
        # relative to its record's subrange (allocation_offset at the subrange
        # start), while the Coordinator descriptor is slot-absolute; the
        # absolute source range is the identity.
        return (
            source.source_slot_id == other.source_slot_id
            and source.slot_generation == other.slot_generation
            and source.nbytes == other.nbytes
            and source.byte_offset + source.allocation_offset
            == other.byte_offset + other.allocation_offset
            and source.source_session_id == other.source_session_id
            and source.pool_generation == other.pool_generation
        )

    def _confirm_direct_task(self, task) -> None:
        """The Coordinator accepted a directly received task: release its holds."""
        direct = self._direct_tasks_by_transfer.get(task.transfer_id)
        if direct is not None and direct != task:
            if not (
                direct.direct_ring_seq
                and self._ring_record_task_matches(direct, task)
            ):
                raise ValueError("Coordinator task differs from its direct dispatch")
            # A §7.1 ring-record task lacks its scheduling metadata, and with
            # DT_RING_PUBLISH_AT_LAUNCH=1 its destination was already latched
            # at ring take; adopt the Coordinator copy for both.
            self._direct_tasks_by_transfer[task.transfer_id] = task
        self._unconfirmed_direct.pop(task.transfer_id, None)
        self._ring_record_positions.pop(task.transfer_id, None)
        self._received_tasks += 1
        self._release_held_ring_pushes()
        for message in self._held_direct_source_done.pop(task.transfer_id, ()):
            self._queue_source_done(message)
        held = self._held_direct_results.pop(task.transfer_id, None)
        if held is not None:
            self._confirmed_held_results.append(held)

    def _flush_confirmed_results(self) -> None:
        """Send the results released by Coordinator confirmations as one batch.

        A ring round usually completes before its Coordinator copy arrives, so
        its result is held per entry.  Sending each one on confirmation
        multiplied result, ACK and Target routing traffic by the batch size and
        saturated the Coordinator under load.
        """
        held = getattr(self, "_confirmed_held_results", None)
        if not held:
            return
        self._confirmed_held_results = []
        from msgspec.structs import replace

        groups: dict[tuple[str, str], list] = {}
        for message in held:
            key = (message.instance_session_id, message.coordinator_session_id)
            groups.setdefault(key, []).append(message)
        for (instance_session_id, _), messages in groups.items():
            self._confirmed_held_sequence += 1
            self._send_result(
                replace(
                    messages[0],
                    batch_id=(
                        f"{self.instance_id}:held:{instance_session_id}:"
                        f"{self._confirmed_held_sequence}"
                    ),
                    entries=tuple(entry for m in messages for entry in m.entries),
                    sent_ns=min(m.sent_ns for m in messages),
                )
            )

    def _expire_unconfirmed_direct(self) -> None:
        """Fail never-confirmed direct work after its deadline.

        A held result must not outlive its ring task: without the Coordinator
        confirmation the Target can never apply it, so abandoning the task
        owes the owning Target an authenticated terminal outcome.  Publish the
        same failure the ring path already sends for a retired-then-dispatched
        task.  Where the Coordinator never accepted its copy this failure is
        stale there, and the Target's own round deadline retires the round.

        A row of a §7.1 held candidate push has no result yet and never will
        once its record deadline passed with neither the ring destination
        nor the Coordinator confirmation arriving (its ring generation may
        already be gone).  Retire it as cancelled so the held push cannot
        outlive the ring data it waits on; the release path retires the
        identity.  A genuinely live row is later answered by the Target's own
        round deadline, and a late Coordinator copy is dropped as a duplicate
        by the seen-set.
        """

        if not self._unconfirmed_direct:
            return
        now_ns = time.monotonic_ns()
        if now_ns < getattr(self, "_next_unconfirmed_expiry_ns", 0):
            # Deadlines are task creation time plus a fixed timeout, so
            # nothing admitted after the last empty scan can expire earlier
            # than the cached minimum; skip the walk until then (#150).
            return
        held_push_ids = {
            task.transfer_id
            for tasks, _candidates, _slot in getattr(self, "_held_ring_pushes", ())
            for task in tasks
        }
        expired_held_push = False
        expired_any = False
        for transfer_id, deadline_ns in tuple(self._unconfirmed_direct.items()):
            if now_ns < deadline_ns + UNCONFIRMED_DIRECT_GRACE_NS:
                continue
            expired_any = True
            if transfer_id in held_push_ids:
                self._unconfirmed_direct.pop(transfer_id)
                self._held_direct_source_done.pop(transfer_id, None)
                self._cancelled_ids.add(transfer_id)
                expired_held_push = True
                continue
            if transfer_id not in self._held_direct_results:
                continue
            self._unconfirmed_direct.pop(transfer_id)
            self._held_direct_results.pop(transfer_id)
            self._held_direct_source_done.pop(transfer_id, None)
            self._pending_ids.discard(transfer_id)
            getattr(self, "_ring_pending_ids", set()).discard(transfer_id)
            task = self._direct_tasks_by_transfer.pop(transfer_id, None)
            self._ring_record_positions.pop(transfer_id, None)
            destination = getattr(task, "candidate_destination", None)
            if (
                task is not None
                and destination is not None
                and self._candidate_push_transport is not None
                and destination.target_instance_id
                in getattr(self, "_remote_targets", ())
            ):
                # The Coordinator never accepted this round, so neither a
                # SourceReadDone nor the failure below can reach the owning
                # Target through it: the router drops both as stale for an
                # unknown transfer.  Publish the authenticated candidate
                # tombstone over the RDMA push leg instead; the Target's
                # arena validates it against the destination's generation
                # and nonce, and the controller retires the round's source
                # lease from it — an explicit read proof that needs no
                # Coordinator confirmation.
                push = self._candidate_push_transport.publish_cancelled([task])
                self._pending_candidate_pushes.append(
                    PendingCandidatePush(
                        push=push,
                        tasks=[task],
                        result_slot=None,
                        candidates=None,
                        cancelled=True,
                        # The failure publication below already reports the
                        # round; the Coordinator never knew this transfer.
                        coordinator_result=False,
                    )
                )
            logger.warning(
                "dropping direct Draft task never confirmed by Coordinator: %s",
                transfer_id,
            )
            if task is not None:
                self._publish_failures(
                    [
                        (
                            task,
                            TimeoutError(
                                "direct Draft task never confirmed by Coordinator"
                            ),
                        )
                    ]
                )
            if hasattr(self, "_ring_retired"):
                # The drop just reported this transfer; a later Coordinator
                # copy must not publish a duplicate failure.
                self._remember_ring_retired(transfer_id, failed=False)
        if expired_any:
            # An expiry fired (or an entry was skipped as not yet
            # result-bearing): rescan on the next poll; neighbours may be
            # near due.
            self._next_unconfirmed_expiry_ns = 0
        else:
            pending = [deadline_ns for deadline_ns in self._unconfirmed_direct.values()]
            self._next_unconfirmed_expiry_ns = (
                min(pending) + UNCONFIRMED_DIRECT_GRACE_NS if pending else 0
            )
        if expired_held_push:
            self._release_held_ring_pushes()

    def _accept_dispatch(self, message) -> bool:
        received_ns = time.monotonic_ns()
        accepted = False
        candidate_registrations = getattr(message, "candidate_registrations", ())
        if candidate_registrations:
            if self.candidate_transport != "cuda-ipc":
                raise ValueError(
                    "mmap Draft received direct candidate arena registration"
                )
            if self._candidate_push_transport is None:
                raise RuntimeError("direct candidate push transport is unavailable")
            for registration in candidate_registrations:
                self._candidate_push_transport.register(registration)
                reg_host = getattr(registration, "host_addr", "") or ""
                if reg_host and reg_host != self._dt_host_addr:
                    # Cross-host Target (issue #164): the data legs use the
                    # RDMA backend (#165); under DT_DIRECT_RING the 4096-byte
                    # task record rides a remote direct ring (issue #182)
                    # instead of the Coordinator relay, and until the ring leg
                    # connects the Target's planner falls back to the relay,
                    # as before.
                    self._remote_targets[registration.target_instance_id] = (
                        reg_host,
                        int(getattr(registration, "data_port", 0) or 0),
                    )
                    if DIRECT_RING:
                        # The Target only plans ring records under
                        # DT_DIRECT_RING; without it this ring would be
                        # allocated but never written.
                        try:
                            self._open_remote_ring(registration)
                        except Exception:
                            self._remote_ring_failed.add(registration.target_session_id)
                            logger.exception(
                                "remote task-record ring to Target %s unavailable; "
                                "the Coordinator relay remains the path",
                                registration.target_instance_id,
                            )
                elif DIRECT_RING:
                    self._open_direct_ring(registration)
            if self._remote_targets and self._feature_pull_transport is not None:
                self._feature_pull_transport.set_remote_targets(
                    self._remote_targets,
                    draft_instance_id=self.instance_id,
                    draft_session_id=self._client.instance_session_id,
                )
        unconfirmed = getattr(self, "_unconfirmed_direct", None)
        if unconfirmed is None:
            unconfirmed = self._unconfirmed_direct = {}
        # The seen window dedups a ring round's two deliveries; confirmation-free
        # there is only the ring one.
        seen = (
            getattr(self, "_seen_transfers", None) if self._ring_confirmation else None
        )
        # Gated-queue depth for the capacity check below: only non-ring
        # identities count.  Direct-ring rounds — copies waiting for their
        # ring data, admitted or held rounds — are dispatched by the
        # Coordinator without the capacity gate (the Target pipelines rounds
        # per request, so a ring confirmation must be accepted in order and
        # can never be back-pressured) and are bounded by owned KV and by
        # execution admission, not by the Coordinator queue this check
        # protects.  None of the counted collections changes while this
        # message is processed (only self._pending does, one appended
        # non-ring entry per accepted task below), so compute the count once
        # instead of per entry.
        ring_ids = getattr(self, "_ring_pending_ids", None) or frozenset()
        held_gated = sum(
            1 for pending in self._pending if pending.transfer_id not in ring_ids
        )
        for batch in self._pending_feature_batches:
            held_gated += sum(
                1
                for item in batch.pending
                if item.task.transfer_id not in ring_ids
            )
        held_gated += sum(
            1
            for entry in getattr(self, "_input_ready_features", ())
            if entry.pending.task.transfer_id not in ring_ids
        )
        for completion_queue in (
            getattr(self, "_pending_flashmtp_completions", ()),
            self._pending_dflash_completions,
            self._pending_eagle3_completions,
        ):
            for batch in completion_queue:
                held_gated += sum(
                    1
                    for item in batch.pending
                    if item.task.transfer_id not in ring_ids
                )
        for task in message.entries:
            if task.transfer_id in self._pending_ids:
                if task.transfer_id in unconfirmed:
                    self._confirm_direct_task(task)
                    accepted = True
                continue
            if getattr(task, "direct_ring_seq", 0):
                if not self._ring_confirmation:
                    # The Coordinator absorbs ring copies (design §8.2): one
                    # here means the processes of this launch disagree on
                    # the mode.
                    raise ValueError(
                        "Coordinator forwarded a ring round to a "
                        "confirmation-free Draft"
                    )
                # The ring copy (and its hidden pull) is authoritative for
                # execution; wait for it and confirm then.  Never pull again:
                # the Target may already have retired the source on the ring
                # worker's completion proof.
                retired = getattr(self, "_ring_retired", {}).pop(task.transfer_id, None)
                if retired is not None:
                    # Received and retired at once: the Coordinator dispatched it.
                    self._received_tasks += 1
                    if retired:
                        # Its ring copy ran and was retired unconfirmed.
                        self._publish_failures(
                            [(task, TimeoutError("ring task retired before confirmation"))]
                        )
                    # Otherwise the drop already reported this transfer.
                    accepted = True
                    continue
                if seen is not None and task.transfer_id in seen:
                    # Duplicate delivery of a copy already confirmed.
                    accepted = True
                    continue
                waiting = self._ring_waiting
                if task.transfer_id in waiting and waiting[task.transfer_id] != task:
                    raise ValueError("Coordinator ring task changed while waiting")
                waiting[task.transfer_id] = task
                accepted = True
                continue
            if held_gated >= self.max_pending_tasks:
                raise RuntimeError(
                    "Coordinator exceeded advertised Draft queue capacity"
                )
            self._validate_task(task)
            if task.source_read_migration_id:
                deferred_key = (
                    task.source_read_migration_id,
                    task.transfer_id,
                )
                previous = self._deferred_feature_source_tasks.get(deferred_key)
                if previous is not None and previous != task:
                    raise ValueError("migration feature lease changed before execution")
                self._deferred_feature_source_tasks[deferred_key] = task
            self._pending.append(task)
            held_gated += 1
            if task.candidate_destination is not None:
                self._direct_tasks_by_transfer[task.transfer_id] = task
            self._received_tasks += 1
            if seen is not None:
                seen.add(task.transfer_id)
                self._seen_transfer_order.append(task.transfer_id)
                if len(self._seen_transfer_order) > 65536:
                    seen.discard(self._seen_transfer_order.popleft())
            self._pending_ids.add(task.transfer_id)
            self._received_ns_by_transfer[task.transfer_id] = received_ns
            accepted = True
        return accepted

    def _persistent_task_allocation(self, task) -> tuple[int, int]:
        """Request rows and net page-1 KV needed by this existing model path."""
        state = self._persistent_requests.get(task.draft_kv_handle)
        allocated = (
            int(state.req.kv.kv_allocated_len)
            if state is not None and state.req.kv is not None else 0
        )
        rows = int(state is None or state.req.req_pool_idx is None)
        proposal_tokens = 0
        if task.requires_proposal or not self._is_feature_backed:
            if self._is_eagle3:
                from sglang.srt.mem_cache.allocation_sizing import get_alloc_reserve_per_decode

                proposal_tokens = get_alloc_reserve_per_decode(
                    self._draft_model_runner.server_args
                )
            elif self._is_dflash:
                proposal_tokens = self.profile.block_size
            else:
                # Extend samples the first AR token; only k-1 decode inputs
                # are subsequently written into persistent KV.
                proposal_tokens = max(self.profile.draft_k - 1, 0)
        return rows, int(task.committed_kv_len) + proposal_tokens - allocated

    def _take_batch(self) -> list[PendingDraftTask]:
        observer = getattr(self, "_overlap_fix_observer", None)
        if observer is not None:
            observer.draft_decision(self, "admission")
        if not self._pending:
            return []
        if self._uses_direct_gpu_source and not self._free_feature_staging_slots:
            return []
        if not self._is_feature_backed and self._proposal_wave_buffer_exhausted():
            return []

        queued: list[PendingDraftTask] = []
        take_ns = time.monotonic_ns()
        trace_selection = profile_event_enabled("draft_ready_select")
        selected_received = {} if trace_selection else None
        profile_eagle3_batch = self._is_eagle3 and profile_event_enabled(
            "draft_eagle3_batch"
        )
        pending_before = len(self._pending) if profile_eagle3_batch else 0
        queue_delays = []
        feature_token_count = 0
        persistent_prefill_token_count = 0
        reserves_result_slot = False
        result_slot_available = not self._proposal_admission_blocked()
        # The cuda-ipc credit is a hard per-request bound: each proposal task
        # queued below claims one of the remaining units, so a pull wave stops
        # at the limit instead of reserving one unit and then expanding past
        # it when its feature batch promotes into INPUT_READY entries.
        admission_units_left = (
            self._available_direct_proposal_admission()
            if self.candidate_transport == "cuda-ipc" and not DRAFT_ADMISSION_BOUND
            else 0
        )

        def proposal_blocked(task) -> bool:
            if not task.requires_proposal:
                return False
            if self.candidate_transport == "cuda-ipc":
                return not DRAFT_ADMISSION_BOUND and admission_units_left <= 0
            return not reserves_result_slot and not result_slot_available

        stop_reason = "queue_exhausted"
        # Examine each currently queued row once. Incompatible or temporarily
        # blocked rows retain their relative FIFO order for the next round.
        deferred = []
        busy_handles = {
            item.task.draft_kv_handle
            for owner in (
                *self._pending_feature_batches,
                *(SimpleNamespace(pending=[entry.pending]) for entry in getattr(self, "_input_ready_features", ())),
                *self._pending_dflash_completions,
                *self._pending_eagle3_completions,
            )
            for item in owner.pending
        } if getattr(self, "continuous_batching", False) and self._persistent else set()
        selected_handles = set()
        if getattr(self, "continuous_batching", False) and self._persistent:
            available_rows = self._draft_model_runner.req_to_token_pool.available_size()
            available_kv = self._draft_model_runner.token_to_kv_pool_allocator.available_size()
            # Pulled/preparing batches own a reservation before allocation;
            # submitted GPU batches already appear in allocator availability.
            for owner in self._pending_feature_batches:
                for item in owner.pending:
                    rows, tokens = self._persistent_task_allocation(item.task)
                    available_rows -= rows
                    available_kv -= max(tokens, 0)
            for entry in getattr(self, "_input_ready_features", ()):
                rows, tokens = self._persistent_task_allocation(entry.pending.task)
                available_rows -= rows
                available_kv -= max(tokens, 0)
        while self._pending and len(queued) < self.max_batch_size:
            next_task = self._pending[0]
            if getattr(self, "continuous_batching", False) and self._persistent and (
                next_task.draft_kv_handle in busy_handles
                or next_task.draft_kv_handle in selected_handles
            ):
                deferred.append(self._pending.popleft())
                continue
            if getattr(self, "continuous_batching", False) and next_task.transfer_id in self._cancelled_ids:
                task = self._pending.popleft()
                self._retire_cancelled_transfer_identity(task)
                if self._uses_direct_gpu_source and not task.source_read_migration_id:
                    self._publish_source_done(task)
                if self._persistent:
                    self._drain_deferred_persistent_release(task.draft_kv_handle)
                continue
            if getattr(self, "continuous_batching", False) and take_ns >= next_task.deadline_ns:
                task = self._pending.popleft()
                self._received_ns_by_transfer.pop(task.transfer_id, None)
                if self._uses_direct_gpu_source and not task.source_read_migration_id:
                    self._publish_source_done(task)
                if self._persistent:
                    self._release_persistent_request(task.draft_kv_handle)
                self._publish_failures([(task, TimeoutError("Draft task deadline expired"))])
                continue
            if getattr(self, "continuous_batching", False) and self._persistent:
                row_cost, kv_cost = self._persistent_task_allocation(next_task)
                if row_cost > available_rows or kv_cost > available_kv:
                    deferred.append(self._pending.popleft())
                    stop_reason = "persistent_capacity"
                    continue
            if self._uses_direct_gpu_source:
                if (
                    (self._is_eagle3 or (getattr(self, "continuous_batching", False) and self._is_dflash))
                    and queued
                    and bool(next_task.requires_proposal)
                    != bool(queued[0].task.requires_proposal)
                ):
                    if profile_eagle3_batch:
                        stop_reason = "proposal_materialization_boundary"
                    if getattr(self, "continuous_batching", False):
                        deferred.append(self._pending.popleft())
                        continue
                    break
                if proposal_blocked(next_task):
                    if profile_eagle3_batch:
                        stop_reason = "result_slot_unavailable"
                    if getattr(self, "continuous_batching", False):
                        deferred.append(self._pending.popleft())
                        continue
                    break
                next_feature_tokens = (
                    int(next_task.feature_token_count)
                    if self._is_feature_backed
                    else int(next_task.source.shape[0])
                )
                if (
                    feature_token_count + next_feature_tokens
                    > self.feature_buffer_max_tokens
                ):
                    if profile_eagle3_batch:
                        stop_reason = "feature_token_capacity"
                    if getattr(self, "continuous_batching", False):
                        deferred.append(self._pending.popleft())
                        continue
                    break
            elif self._persistent:
                next_prefill_tokens = len(next_task.committed_token_ids)
                if (
                    queued
                    and persistent_prefill_token_count + next_prefill_tokens
                    > DRAFT_PREFILL_GRAPH_MAX_TOKENS
                ):
                    if getattr(self, "continuous_batching", False):
                        deferred.append(self._pending.popleft())
                        continue
                    break
            task = self._pending.popleft()
            received_ns = self._received_ns_by_transfer.pop(task.transfer_id, None)
            if received_ns is not None:
                queue_delays.append(take_ns - received_ns)
            if task.transfer_id in self._cancelled_ids:
                self._retire_cancelled_transfer_identity(task)
                if self._uses_direct_gpu_source and not task.source_read_migration_id:
                    self._publish_source_done(task)
                continue
            if self._uses_direct_gpu_source and take_ns >= task.deadline_ns:
                if not task.source_read_migration_id:
                    self._publish_source_done(task)
                if self._persistent:
                    self._release_persistent_request(task.draft_kv_handle)
                self._publish_failures(
                    [(task, TimeoutError("direct GPU source task deadline expired"))]
                )
                continue
            if self._uses_direct_gpu_source:
                feature_token_count += (
                    int(task.feature_token_count)
                    if self._is_feature_backed
                    else int(task.source.shape[0])
                )
            if self._persistent and not self._is_feature_backed:
                persistent_prefill_token_count += len(task.committed_token_ids)
            if trace_selection:
                selected_received[task.transfer_id] = received_ns
            queued.append(
                PendingDraftTask(task=task)
            )
            if task.requires_proposal:
                reserves_result_slot = True
                admission_units_left -= 1
            if self._persistent:
                selected_handles.add(task.draft_kv_handle)
                if getattr(self, "continuous_batching", False):
                    available_rows -= row_cost
                    available_kv -= kv_cost
        self._pending.extendleft(reversed(deferred))
        pending_after = len(self._pending)
        cap_limited = False
        if (
            profile_eagle3_batch
            and len(queued) == self.max_batch_size
            and self._pending
        ):
            next_task = self._pending[0]
            compatible = not (
                self._is_eagle3
                and queued
                and bool(next_task.requires_proposal)
                != bool(queued[0].task.requires_proposal)
            )
            result_available = not (
                self._uses_direct_gpu_source and proposal_blocked(next_task)
            )
            next_feature_tokens = (
                int(next_task.feature_token_count)
                if self._is_feature_backed
                else (
                    int(next_task.source.shape[0])
                    if self._uses_direct_gpu_source
                    else 0
                )
            )
            feature_available = not (
                self._uses_direct_gpu_source
                and feature_token_count + next_feature_tokens
                > self.feature_buffer_max_tokens
            )
            cap_limited = compatible and result_available and feature_available
            if cap_limited:
                stop_reason = "max_batch_size"
            elif not compatible:
                stop_reason = "proposal_materialization_boundary"
            elif not result_available:
                stop_reason = "result_slot_unavailable"
            elif not feature_available:
                stop_reason = "feature_token_capacity"
        if profile_eagle3_batch:
            oldest_queue_age_ns = max(queue_delays, default=0)
            for item in queued:
                item.pending_before = pending_before
                item.pending_after = pending_after
                item.taken_ns = take_ns
                item.oldest_queue_age_ns = oldest_queue_age_ns
                item.batch_feature_token_count = feature_token_count
                item.stop_reason = stop_reason
                item.cap_limited = cap_limited
        if queue_delays:
            self._last_queue_delay_ns = max(queue_delays)

        # The result publisher needs rows grouped by Target before assigning
        # Draft-local destination and result rows.
        queued = target_major(queued)
        if trace_selection and queued:
            profile_emit(
                "draft_ready_select",
                draft_instance=self.instance_id,
                started_ns=take_ns,
                finished_ns=time.monotonic_ns(),
                # This is input admission. Feature-backed rows become compute
                # eligible only after their existing pull-completion event.
                requires_feature_copy=self._uses_direct_gpu_source,
                persistent_kv=self._persistent,
                rows=tuple(dict(
                    transfer_id=item.task.transfer_id,
                    request_id=item.task.request_id,
                    round_id=item.task.round_id,
                    received_ns=selected_received[item.task.transfer_id],
                ) for item in queued),
            )
        if self._uses_direct_gpu_source:
            return queued
        if self._persistent:
            return queued
        raise RuntimeError("nonpersistent Draft task lacks direct GPU source transport")

    def _dflash_admission_reserved(self) -> int:
        """Submitted-but-unpublished proposal work holding an admission unit.

        Ring admission, INPUT_READY proposal entries, and pending
        feature/FlashMTP/DFlash/EAGLE3 completions each reserve one unit until
        their wave publishes (the same terms the result-pool accounting used).
        The cuda-ipc credit is a hard per-request bound, so its pending
        feature batches reserve one unit per proposal task — promotion into
        INPUT_READY then swaps units 1:1 and cannot expand past the limit.
        The mmap pool keeps the per-batch reservation it always had.
        """

        if self.candidate_transport == "cuda-ipc":
            pending_feature = sum(
                bool(item.task.requires_proposal)
                for batch in self._pending_feature_batches
                for item in batch.pending
            )
        else:
            pending_feature = sum(
                any(item.task.requires_proposal for item in batch.pending)
                for batch in self._pending_feature_batches
            )
        return pending_feature + sum(
            bool(entry.pending.task.requires_proposal)
            for entry in getattr(self, "_input_ready_features", ())
        ) + len(
            getattr(self, "_pending_flashmtp_completions", ())
        ) + sum(
            bool(completion.proposal_tasks)
            for completion in self._pending_dflash_completions
        ) + sum(
            getattr(completion, "candidates", None) is not None
            for completion in getattr(self, "_pending_eagle3_completions", ())
        )

    def _available_dflash_result_slots(self) -> int:
        """Count result slots after bounded reservations by submitted work."""
        if not self._uses_direct_gpu_source:
            return len(self._free_result_slots)
        return max(
            0, len(self._free_result_slots) - self._dflash_admission_reserved()
        )

    def _acquire_direct_proposal_admission(self) -> None:
        """Hold one cuda-ipc proposal-admission credit for a published wave."""
        if DRAFT_ADMISSION_BOUND:
            return  # no credit lifecycle under the batch bound
        self._direct_proposal_admission_inflight += 1

    def _release_direct_proposal_admission(self) -> None:
        """Return one wave's credit; every holder must release exactly once."""
        if DRAFT_ADMISSION_BOUND:
            return
        if self._direct_proposal_admission_inflight <= 0:
            raise RuntimeError("direct proposal admission credit underflow")
        self._direct_proposal_admission_inflight -= 1

    def _available_direct_proposal_admission(self) -> int:
        """cuda-ipc proposal-admission credits left for scheduling decisions.

        Independent replacement for the admission bound the result-slot pool
        used to give the direct path (issue #42): the configured limit minus
        credits held by published-but-incomplete waves minus the same
        submitted-work reservations the pool accounting subtracted.  The base
        is an explicit counter, never a deque length.
        """

        available = (
            self.direct_proposal_admission_limit
            - self._direct_proposal_admission_inflight
        )
        if self._uses_direct_gpu_source:
            available -= self._dflash_admission_reserved()
        return max(0, available)

    def _proposal_admission_blocked(self) -> bool:
        """Whether one more proposal wave is admission-blocked right now."""
        if self.candidate_transport == "cuda-ipc":
            return (
                not DRAFT_ADMISSION_BOUND
                and self._available_direct_proposal_admission() <= 0
            )
        return self._available_dflash_result_slots() <= 0

    def _proposal_wave_buffer_exhausted(self) -> bool:
        """Raw per-wave publish buffer exhaustion (no reservation accounting)."""
        if self.candidate_transport == "cuda-ipc":
            return (
                not DRAFT_ADMISSION_BOUND
                and self._direct_proposal_admission_inflight
                >= self.direct_proposal_admission_limit
            )
        return not self._free_result_slots

    @torch.inference_mode()
    def _execute_batch(self, pending: list[PendingDraftTask]) -> None:
        self._last_batch_size = len(pending)
        if self._uses_direct_gpu_source:
            self._submit_dflash_feature_batch(pending)
            return
        if self._persistent:
            self._execute_standalone_batch(pending)
            return
        raise RuntimeError("FlashMTP proposal execution requires a pulled GPU pivot")

    def _canonical_draft_device(self, device: object) -> torch.device:
        resolved = torch.device(device)
        if resolved.type == "cuda" and resolved.index is None:
            return torch.device("cuda", self.gpu_id)
        return resolved

    def _require_flashmtp_pivot_layout(
        self, pivots: torch.Tensor, *, batch_size: int
    ) -> torch.device:
        if not self._feature_staging_buffers:
            raise RuntimeError("FlashMTP GPU pivot staging is not initialized")
        staging_device = self._canonical_draft_device(
            self._feature_staging_buffers[0].device
        )
        configured_device = self._canonical_draft_device(self.device)
        if staging_device.type != "cuda" or staging_device.index is None:
            raise RuntimeError(
                "FlashMTP pivot staging must use an explicitly indexed CUDA device"
            )
        if configured_device != staging_device or any(
            self._canonical_draft_device(staging.device) != staging_device
            for staging in self._feature_staging_buffers[1:]
        ):
            raise RuntimeError("FlashMTP GPU pivot staging device is inconsistent")
        if (
            self._canonical_draft_device(pivots.device) != staging_device
            or pivots.dtype != torch.bfloat16
            or tuple(pivots.shape) != (batch_size, self._pivot_width)
        ):
            raise ValueError("FlashMTP pulled pivots have an invalid GPU layout")
        return staging_device

    def _stage_flashmtp_bonus_tokens(
        self,
        tasks: list[object],
        *,
        staging_slot: int,
        execution_device: torch.device,
    ) -> tuple[
        torch.Tensor,
        torch.cuda.Event | None,
        torch.cuda.Event | None,
        int,
        int,
    ]:
        """Pack roots into the slot-owned pinned buffer and enqueue H2D."""

        if (
            len(self._proposal_bonus_tokens_host_buffers)
            != len(self._feature_staging_buffers)
            or len(self._proposal_bonus_tokens_buffers)
            != len(self._feature_staging_buffers)
            or not 0 <= staging_slot < len(self._feature_staging_buffers)
        ):
            raise RuntimeError("FlashMTP bonus-token staging slots are inconsistent")
        batch_size = len(tasks)
        if not 0 < batch_size <= self.max_batch_size:
            raise ValueError("FlashMTP bonus-token batch size is invalid")
        host_buffer = self._proposal_bonus_tokens_host_buffers[staging_slot]
        device_buffer = self._proposal_bonus_tokens_buffers[staging_slot]
        if (
            host_buffer.dtype != torch.int64
            or tuple(host_buffer.shape) != (self.max_batch_size,)
            or torch.device(host_buffer.device).type != "cpu"
            or not host_buffer.is_pinned()
        ):
            raise RuntimeError(
                "FlashMTP bonus-token host staging must be pinned CPU int64"
            )
        if (
            device_buffer.dtype != torch.int64
            or tuple(device_buffer.shape) != (self.max_batch_size,)
            or self._canonical_draft_device(device_buffer.device) != execution_device
        ):
            raise RuntimeError(
                "FlashMTP bonus-token device staging has an invalid layout"
            )

        host_view = host_buffer[:batch_size]
        device_view = device_buffer[:batch_size]
        profile = self._profile_active
        pack_started_ns = time.monotonic_ns() if profile else 0
        host_view.copy_(
            torch.tensor(
                [task.next_token_id for task in tasks],
                dtype=torch.int64,
                device="cpu",
            )
        )
        bonus_pack_ns = time.monotonic_ns() - pack_started_ns if profile else 0

        stream = torch.cuda.current_stream(self.device)
        h2d_start_event = None
        h2d_done_event = None
        if profile:
            h2d_start_event = torch.cuda.Event(enable_timing=True)
            h2d_done_event = torch.cuda.Event(enable_timing=True)
            h2d_start_event.record(stream)
        submit_started_ns = time.monotonic_ns() if profile else 0
        device_view.copy_(host_view, non_blocking=True)
        bonus_h2d_submit_ns = (
            time.monotonic_ns() - submit_started_ns if profile else 0
        )
        if h2d_done_event is not None:
            h2d_done_event.record(stream)
        return (
            device_view,
            h2d_start_event,
            h2d_done_event,
            bonus_pack_ns,
            bonus_h2d_submit_ns,
        )

    @torch.inference_mode()
    def _execute_flashmtp_batch(
        self,
        pending: list[PendingDraftTask],
        pivots: torch.Tensor,
        *,
        staging_slot: int,
    ) -> None:
        tasks = [item.task for item in pending]
        batch_size = len(tasks)
        profile = self._profile_active
        execute_started_ns = time.monotonic_ns() if profile else 0
        execution_device = self._require_flashmtp_pivot_layout(
            pivots, batch_size=batch_size
        )
        prior_pending_flashmtp_completions = len(
            self._pending_flashmtp_completions
        )
        (
            bonus_tokens,
            bonus_h2d_start_event,
            bonus_h2d_done_event,
            bonus_pack_ns,
            bonus_h2d_submit_ns,
        ) = self._stage_flashmtp_bonus_tokens(
            tasks,
            staging_slot=staging_slot,
            execution_device=execution_device,
        )
        inputs_ready_ns = time.monotonic_ns() if profile else 0
        proposal_start_event = None
        proposal_done_event = None
        if profile:
            proposal_start_event = torch.cuda.Event(enable_timing=True)
            proposal_done_event = torch.cuda.Event(enable_timing=True)
            proposal_start_event.record(torch.cuda.current_stream(self.device))
        if self._draft_cuda_graph is not None:
            candidates = self._draft_cuda_graph.replay(
                pivots=pivots,
                bonus_tokens=bonus_tokens,
            )
            flashmtp_execution = "graph"
        else:
            block_ids = torch.full(
                (batch_size, self._draft_block_len),
                self._mask_token_id,
                dtype=torch.int64,
                device=execution_device,
            )
            block_ids[:, 0].copy_(bonus_tokens)
            input_embeds = self._token_embedding(block_ids)
            positions = self._local_positions.expand(batch_size, -1)
            forward_batch = SimpleNamespace(
                batch_size=batch_size,
                spec_info=SimpleNamespace(hidden_states=pivots),
            )
            output = self._draft_model(
                input_ids=block_ids.reshape(-1),
                positions=positions.reshape(-1),
                forward_batch=forward_batch,
                input_embeds=input_embeds.reshape(
                    batch_size * self._draft_block_len, -1
                ),
            )
            if output.hidden_states is None:
                raise RuntimeError("FlashMTP Draft forward returned no hidden states")
            block_hidden = output.hidden_states.reshape(
                batch_size, self._draft_block_len, -1
            )
            if self._draft_model.markov_head is not None:
                proposals = self._draft_model.sample_draft_tokens(
                    block_hidden=block_hidden,
                    first_prev_token_ids=bonus_tokens,
                )
            else:
                if self._target_lm_head is None:
                    raise RuntimeError(
                        "legacy FlashMTP requires the Target LM head for proposal sampling"
                    )
                prediction_hidden = self._draft_model.prediction_hidden(block_hidden)
                proposals = torch.argmax(
                    F.linear(prediction_hidden, self._target_lm_head.weight), dim=-1
                )
            if tuple(proposals.shape) != (batch_size, self._num_proposals):
                raise RuntimeError(
                    f"FlashMTP proposal shape mismatch: got {tuple(proposals.shape)}"
                )
            candidates = torch.empty(
                (batch_size, self.profile.block_size),
                dtype=torch.int32,
                device=execution_device,
            )
            candidates[:, 0].copy_(bonus_tokens)
            candidates[:, 1:].copy_(proposals)
            flashmtp_execution = "eager"
        if proposal_done_event is not None:
            proposal_done_event.record(torch.cuda.current_stream(self.device))
        measurement_rows = sum(
            task.request_id.startswith(DT_BENCHMARK_MEASUREMENT_RID_PREFIX)
            for task in tasks
        )
        if measurement_rows:
            self._record_graph_measurement(
                path="flashmtp_proposal",
                execution=flashmtp_execution,
                measurement_rows=measurement_rows,
                measurement_batch_rows=measurement_rows,
                batch_size=batch_size,
                graph_replays=int(flashmtp_execution == "graph"),
                expected_graph_replays=1,
                fallback_reason=(
                    None
                    if flashmtp_execution == "graph"
                    else "flashmtp_cuda_graph_unavailable"
                ),
            )
        candidate_rows_cpu = None
        if self.candidate_transport == "mmap":
            if not 0 <= staging_slot < len(self._candidate_result_host_buffers):
                raise RuntimeError("FlashMTP candidate staging slot is unavailable")
            candidate_rows_cpu = self._candidate_result_host_buffers[staging_slot][
                :batch_size
            ]
            candidate_rows_cpu.copy_(candidates, non_blocking=True)
        else:
            # CUDA graph proposal outputs use stable addresses and are replayed
            # in place. Snapshot into this bounded completion's GPU ownership
            # on the same stream before a later batch can overwrite them.
            candidates = candidates.clone()
        completion_event = torch.cuda.Event(enable_timing=profile)
        completion_event.record(torch.cuda.current_stream(self.device))
        submitted_ns = time.monotonic_ns()
        self._pending_flashmtp_completions.append(
            PendingFlashMTPCompletion(
                pending=pending,
                staging_slot=staging_slot,
                completion_event=completion_event,
                bonus_h2d_start_event=bonus_h2d_start_event,
                bonus_h2d_done_event=bonus_h2d_done_event,
                proposal_start_event=proposal_start_event,
                proposal_done_event=proposal_done_event,
                candidates=candidates,
                candidate_rows_cpu=candidate_rows_cpu,
                execute_started_ns=execute_started_ns,
                inputs_ready_ns=inputs_ready_ns,
                submitted_ns=submitted_ns,
                bonus_pack_ns=bonus_pack_ns,
                bonus_h2d_submit_ns=bonus_h2d_submit_ns,
                bonus_device=str(bonus_tokens.device),
                execution=flashmtp_execution,
                measurement_rows=measurement_rows,
                prior_pending_flashmtp_completions=(
                    prior_pending_flashmtp_completions
                ),
            )
        )
        if self.candidate_transport == "cuda-ipc":
            self._queue_early_candidate_push(
                [item.task for item in pending], candidates, completion_event
            )

    def _poll_flashmtp_completions(self) -> None:
        """Publish same-stream candidate copies only after their event retires."""

        if self._fatal_gpu_quarantine_active():
            return
        capacity_changed = False
        while self._pending_flashmtp_completions:
            completion = self._pending_flashmtp_completions[0]
            try:
                ready = completion.completion_event.query()
            except Exception as exc:
                # A failed query cannot prove that either the graph output read
                # or the pinned D2H has stopped.  Retain the staging slot and
                # host buffer until process teardown and prohibit later CUDA.
                self._pending_flashmtp_completions.popleft()
                self._quarantined_flashmtp_completions.append(completion)
                exc.resources_quiescent = False
                logger.critical(
                    "quarantined FlashMTP candidate completion after event "
                    "query failure: transfers=%s",
                    [item.task.transfer_id for item in completion.pending],
                    exc_info=True,
                )
                self._enter_fatal_gpu_quarantine(
                    "unquiesced FlashMTP completion-event failure"
                )
                self._publish_failures(
                    [(item.task, exc) for item in completion.pending]
                )
                self._publish_capacity()
                break
            if not ready:
                break

            self._pending_flashmtp_completions.popleft()
            tasks = [item.task for item in completion.pending]
            completed_ns = time.monotonic_ns()
            # A push queued at launch already published these candidates.
            early = self._take_early_pushed(tasks)
            expired = [
                task
                for task in tasks
                if task.transfer_id not in self._cancelled_ids
                and task.transfer_id not in early
                and completed_ns >= task.deadline_ns
            ]
            expired_ids = {task.transfer_id for task in expired}
            active_indices = [
                index
                for index, task in enumerate(tasks)
                if task.transfer_id not in self._cancelled_ids
                and task.transfer_id not in expired_ids
                and task.transfer_id not in early
            ]
            cancelled = [
                task for task in tasks if task.transfer_id in self._cancelled_ids
            ]

            if self._profile_active:
                if (
                    completion.bonus_h2d_start_event is None
                    or completion.bonus_h2d_done_event is None
                    or completion.proposal_start_event is None
                    or completion.proposal_done_event is None
                ):
                    raise RuntimeError(
                        "profiled FlashMTP completion lost GPU timing events"
                    )
                profile_emit(
                    "draft_flashmtp_proposal",
                    draft_instance=self.instance_id,
                    execute_started_ns=completion.execute_started_ns,
                    inputs_ready_ns=completion.inputs_ready_ns,
                    proposal_enqueued_ns=completion.submitted_ns,
                    batch_size=len(tasks),
                    transfer_ids=tuple(task.transfer_id for task in tasks),
                    execution=completion.execution,
                    measurement_rows=completion.measurement_rows,
                    input_prepare_ns=(
                        completion.inputs_ready_ns - completion.execute_started_ns
                    ),
                    proposal_to_host_ns=(
                        completed_ns - completion.inputs_ready_ns
                    ),
                    total_ns=completed_ns - completion.execute_started_ns,
                    candidate_bytes=(
                        len(tasks) * self.profile.block_size * 4
                    ),
                    proposal_gpu_ns=int(
                        completion.proposal_start_event.elapsed_time(
                            completion.proposal_done_event
                        )
                        * 1_000_000
                    ),
                    candidate_d2h_gpu_ns=(
                        int(
                            completion.proposal_done_event.elapsed_time(
                                completion.completion_event
                            )
                            * 1_000_000
                        )
                        if self.candidate_transport == "mmap"
                        else 0
                    ),
                    completion_poll_ns=completed_ns - completion.submitted_ns,
                    asynchronous_completion=True,
                    bonus_stage_async=True,
                    bonus_tokens_pinned_h2d=True,
                    bonus_staging_slot=completion.staging_slot,
                    bonus_token_rows=len(tasks),
                    bonus_token_bytes=len(tasks) * 8,
                    bonus_host_pinned=True,
                    bonus_host_dtype=str(torch.int64),
                    bonus_device_dtype=str(torch.int64),
                    bonus_device=completion.bonus_device,
                    bonus_pack_ns=completion.bonus_pack_ns,
                    bonus_h2d_submit_ns=completion.bonus_h2d_submit_ns,
                    bonus_h2d_gpu_ns=int(
                        completion.bonus_h2d_start_event.elapsed_time(
                            completion.bonus_h2d_done_event
                        )
                        * 1_000_000
                    ),
                    prior_pending_flashmtp_completions=(
                        completion.prior_pending_flashmtp_completions
                    ),
                    **(
                        {
                            "candidate_stage_gpu_ns": int(
                                completion.proposal_done_event.elapsed_time(
                                    completion.completion_event
                                )
                                * 1_000_000
                            ),
                            "candidate_stage_kind": "gpu_snapshot",
                        }
                        if self.candidate_transport == "cuda-ipc"
                        else {}
                    ),
                )

            if active_indices:
                source_rows = (
                    completion.candidates
                    if self.candidate_transport == "cuda-ipc"
                    else completion.candidate_rows_cpu
                )
                if source_rows is None:
                    raise RuntimeError("FlashMTP candidate rows are unavailable")
                with self._direct_push_scope(completion.completion_event, source_rows):
                    candidate_rows = _select_candidate_rows(source_rows, active_indices)
                    self._publish_successes(
                        [tasks[index] for index in active_indices],
                        (
                            candidate_rows
                            if self.candidate_transport == "cuda-ipc"
                            else candidate_rows.contiguous().numpy().tobytes()
                        ),
                    )
            for task in cancelled:
                self._retire_cancelled_transfer_identity(task)
            if expired:
                self._publish_failures(
                    [
                        (
                            task,
                            TimeoutError(
                                "FlashMTP candidate completed after its deadline"
                            ),
                        )
                        for task in expired
                    ]
                )

            self._complete_feature_compute(
                (completion.staging_slot,), (len(completion.pending),)
            )
            capacity_changed = True

        if capacity_changed:
            self._publish_capacity()

    def _submit_dflash_feature_batch(self, pending: list[PendingDraftTask]) -> None:
        from sglang.srt.speculative.dt_feature_p2p import FeatureSourceDescriptor

        if self._feature_pull_transport is None:
            raise RuntimeError("direct GPU source transport is not initialized")
        if not self._free_feature_staging_slots:
            raise RuntimeError("direct GPU staging admission was violated")
        staging_slot = self._free_feature_staging_slots.popleft()
        staging = self._feature_staging_buffers[staging_slot]
        sources = []
        destinations = []
        futures = []
        row_ranges = []
        row_offset = 0
        submitted_ns = time.monotonic_ns()
        batch_id = uuid.uuid4().hex
        for item in pending:
            task = item.task
            source = task.source
            row_count = (
                int(task.feature_token_count)
                if self._is_feature_backed
                else int(source.shape[0])
            )
            row_end = row_offset + row_count
            descriptor = FeatureSourceDescriptor(
                target_instance_id=source.target_instance_id,
                source_device_id=source.source_device_id,
                source_slot_id=source.source_slot_id,
                slot_generation=source.slot_generation,
                transfer_id=source.transfer_id,
                feature_schema_id=source.feature_schema_id,
                nbytes=source.nbytes,
                byte_offset=source.byte_offset,
                transport=source.transport,
                source_session_id=source.source_session_id,
                pool_generation=source.pool_generation,
                transport_session=source.transport_session,
                device_uuid=source.device_uuid,
                allocation_length=source.allocation_length,
                allocation_offset=source.allocation_offset,
                ipc_handle=source.ipc_handle,
                dtype=source.dtype,
                shape=tuple(source.shape),
                producer_event_handle=source.producer_event_handle,
            )
            sources.append(descriptor)
            destinations.append(staging[row_offset:row_end])
            if task.source_read_migration_id:
                self._feature_source_inflight.add(
                    (task.source_read_migration_id, task.transfer_id)
                )
            row_ranges.append((row_offset, row_end))
            row_offset = row_end
            profile_emit(
                "draft_feature_pull_submit",
                draft_instance=self.instance_id,
                feature_batch_id=batch_id,
                transfer_id=task.transfer_id,
                request_id=task.request_id,
                round_id=task.round_id,
                feature_bytes=source.nbytes,
                submitted_ns=submitted_ns,
            )
        # Completion/admission is request-granular. The transport may still
        # serialize these copies on its bounded stream, but one slow entry no
        # longer freezes compute membership chosen for the other entries.
        # Consecutive rows of one cross-host Target go down as one batch so
        # the pull worker covers their READs with a single GPU-visibility
        # fence instead of a per-row fence + landing copy (#196).
        futures: list = [None] * len(sources)
        remote_run: list[int] = []
        remote_run_target: str | None = None

        def flush_remote_run() -> None:
            nonlocal remote_run_target
            if not remote_run:
                return
            run_futures = self._feature_pull_transport.submit_remote_batch(
                sources=tuple(sources[index] for index in remote_run),
                destinations=tuple(destinations[index] for index in remote_run),
            )
            for index, future in zip(remote_run, run_futures, strict=True):
                futures[index] = future
            remote_run.clear()
            remote_run_target = None

        transport = self._feature_pull_transport
        for index, (source, destination) in enumerate(
            zip(sources, destinations, strict=True)
        ):
            if transport.is_remote_target(source.target_instance_id):
                if remote_run_target != source.target_instance_id:
                    flush_remote_run()
                    remote_run_target = source.target_instance_id
                remote_run.append(index)
                continue
            flush_remote_run()
            try:
                futures[index] = transport.submit(
                    source=source,
                    destination=destination,
                )
            except Exception as exc:
                failed = Future()
                failed.set_exception(exc)
                futures[index] = failed
        flush_remote_run()
        profile_emit(
            "draft_feature_pull_batch_submit",
            draft_instance=self.instance_id,
            feature_batch_id=batch_id,
            segment_count=len(sources),
            peer_count=len({source.target_instance_id for source in sources}),
            feature_bytes=sum(source.nbytes for source in sources),
            submitted_ns=submitted_ns,
        )
        self._pending_feature_batches.append(
            PendingFeatureBatch(
                batch_id=batch_id,
                pending=pending,
                staging_slot=staging_slot,
                row_ranges=tuple(row_ranges),
                futures=tuple(futures),
                submitted_ns=submitted_ns,
            )
        )
        if staging_slot in self._feature_staging_refcounts:
            raise RuntimeError("feature staging slot was admitted twice")
        self._feature_staging_refcounts[staging_slot] = len(pending)
        self._publish_capacity()

    def _release_feature_staging_entries(self, staging_slot: int, count: int) -> None:
        if count <= 0:
            raise ValueError("feature staging release count must be positive")
        if self._native_ingress and 0 <= self._landing_slot_base <= staging_slot:
            return  # landing placements are released per record (ready_rows)
        refcounts = getattr(self, "_feature_staging_refcounts", None)
        if refcounts is None:
            # Narrow unit fixtures predating request-granular ownership create
            # the engine with __new__. Production initialization always owns
            # the explicit refcount map.
            self._free_feature_staging_slots.append(staging_slot)
            return
        remaining = refcounts.get(staging_slot)
        if remaining is None or count > remaining:
            raise RuntimeError("feature staging ownership underflow")
        remaining -= count
        if remaining:
            self._feature_staging_refcounts[staging_slot] = remaining
            return
        if staging_slot in self._feature_staging_compute_busy:
            raise RuntimeError("feature staging released while compute still owns it")
        del self._feature_staging_refcounts[staging_slot]
        if 0 <= getattr(self, "_landing_slot_base", -1) <= staging_slot:
            # Ring landing buffers return to the ring worker, not the pull pool.
            self._free_landing(staging_slot - self._landing_slot_base)
            return
        self._free_feature_staging_slots.append(staging_slot)

    def _complete_feature_compute(
        self,
        staging_slots: tuple[int, ...],
        staging_slot_counts: tuple[int, ...],
    ) -> None:
        busy = (
            None
            if self._slot_decouple
            else getattr(self, "_feature_staging_compute_busy", None)
        )
        for staging_slot, count in zip(
            staging_slots, staging_slot_counts, strict=True
        ):
            if busy is not None:
                if staging_slot not in busy:
                    raise RuntimeError(
                        "feature compute completion has no submitted owner"
                    )
                busy.remove(staging_slot)
            self._release_feature_staging_entries(staging_slot, count)

    def _poll_feature_batches(self) -> None:
        """Promote each completed pull independently into INPUT_READY."""

        if self._fatal_gpu_quarantine_active():
            return
        transport = getattr(self, "_feature_pull_transport", None)
        scan_needed = getattr(transport, "scan_needed", None)
        scan = transport is None or scan_needed is None or scan_needed()
        retained = deque()
        capacity_changed = False
        while scan and self._pending_feature_batches:
            feature_batch = self._pending_feature_batches.popleft()
            unfinished_pending = []
            unfinished_ranges = []
            unfinished_futures = []
            entries = tuple(zip(
                feature_batch.pending,
                feature_batch.row_ranges,
                feature_batch.futures,
                strict=True,
            ))
            for entry_index, (item, row_range, future) in enumerate(entries):
                if not future.done():
                    unfinished_pending.append(item)
                    unfinished_ranges.append(row_range)
                    unfinished_futures.append(future)
                    continue
                task = item.task
                deferred_key = (
                    (task.source_read_migration_id, task.transfer_id)
                    if task.source_read_migration_id
                    else None
                )
                resources_quiescent = True
                try:
                    metrics = future.result()
                except Exception as exc:
                    resources_quiescent = bool(
                        getattr(exc, "resources_quiescent", True)
                    )
                    if deferred_key is not None:
                        self._feature_source_inflight.discard(deferred_key)
                        (self._feature_source_quiescent if resources_quiescent else self._feature_source_unquiescent).add(deferred_key)
                    elif not resources_quiescent:
                        self._feature_source_unquiescent.add(
                            (task.source.source_session_id, task.transfer_id)
                        )
                    if resources_quiescent:
                        if self._persistent:
                            self._release_persistent_request(task.draft_kv_handle)
                        self._release_feature_staging_entries(
                            feature_batch.staging_slot, 1
                        )
                    else:
                        if self._persistent:
                            self._quarantined_feature_draft_handles.add(
                                task.draft_kv_handle
                            )
                        self._enter_fatal_gpu_quarantine(
                            "unquiesced feature-copy failure"
                        )
                    self._publish_failures([(task, exc)])
                    capacity_changed = True
                    if not resources_quiescent:
                        remaining_entries = entries[entry_index + 1 :]
                        if remaining_entries:
                            retained.append(
                                PendingFeatureBatch(
                                    batch_id=feature_batch.batch_id,
                                    pending=[entry[0] for entry in remaining_entries],
                                    staging_slot=feature_batch.staging_slot,
                                    row_ranges=tuple(
                                        entry[1] for entry in remaining_entries
                                    ),
                                    futures=tuple(
                                        entry[2] for entry in remaining_entries
                                    ),
                                    submitted_ns=feature_batch.submitted_ns,
                                )
                            )
                        retained.extend(self._pending_feature_batches)
                        self._pending_feature_batches = retained
                        if capacity_changed:
                            self._publish_capacity()
                        return
                    continue

                completed_ns = time.monotonic_ns()
                profile_emit(
                    "draft_feature_pull_complete",
                    draft_instance=self.instance_id,
                    feature_batch_id=feature_batch.batch_id,
                    transfer_id=task.transfer_id,
                    request_id=task.request_id,
                    round_id=task.round_id,
                    controller_submitted_ns=feature_batch.submitted_ns,
                    worker_started_ns=getattr(metrics, "worker_started_ns", 0),
                    mapping_done_ns=getattr(metrics, "mapping_done_ns", 0),
                    transport_submitted_ns=metrics.submitted_ns,
                    driver_submit_started_ns=getattr(
                        metrics, "driver_submit_started_ns", 0
                    ),
                    driver_submit_done_ns=getattr(
                        metrics, "driver_submit_done_ns", 0
                    ),
                    cuda_pre_copy_ns=getattr(metrics, "cuda_pre_copy_ns", 0),
                    cuda_copy_ns=metrics.gpu_copy_ns,
                    gpu_start_observed_ns=getattr(
                        metrics, "gpu_start_observed_ns", 0
                    ),
                    gpu_completed_ns=metrics.gpu_completed_ns,
                    owner_observed_ns=completed_ns,
                    owner_poll_delay_ns=completed_ns - metrics.gpu_completed_ns,
                    feature_bytes=metrics.feature_bytes,
                    target_instance=task.source.target_instance_id,
                    source_session_id=task.source.source_session_id,
                    pool_generation=task.source.pool_generation,
                    transport_session=task.source.transport_session,
                )
                if task.request_id.startswith(DT_BENCHMARK_MEASUREMENT_RID_PREFIX):
                    source = task.source
                    self._record_direct_gpu_measurement_request(
                        target_instance_id=source.target_instance_id,
                        source_session_id=source.source_session_id,
                        pool_generation=source.pool_generation,
                        transport_session=source.transport_session,
                        request_id=task.request_id,
                        measurement_rows=int(source.shape[0]),
                        measurement_requests=1,
                        measurement_proposal_requests=int(task.requires_proposal),
                    )
                    self._record_direct_gpu_measurement_epoch(
                        target_instance_id=source.target_instance_id,
                        source_session_id=source.source_session_id,
                        pool_generation=source.pool_generation,
                        transport_session=source.transport_session,
                        measurement_rows=int(source.shape[0]),
                        measurement_requests=1,
                        measurement_proposal_requests=int(task.requires_proposal),
                    )
                if deferred_key is not None:
                    self._feature_source_inflight.discard(deferred_key)
                    self._feature_source_quiescent.add(deferred_key)
                    previous = self._deferred_feature_source_tasks.get(deferred_key)
                    if previous is not None and previous != task:
                        raise ValueError("deferred feature transfer changed payload")
                    self._deferred_feature_source_tasks[deferred_key] = task
                else:
                    self._publish_source_done(task)

                cancelled = task.transfer_id in self._cancelled_ids
                expired = not cancelled and completed_ns >= task.deadline_ns
                if cancelled or expired:
                    if cancelled:
                        self._retire_cancelled_transfer_identity(task)
                    if self._persistent:
                        self._release_persistent_request(task.draft_kv_handle)
                    if expired:
                        self._publish_failures(
                            [(task, TimeoutError(
                                "direct GPU source pull completed after its deadline"
                            ))]
                        )
                    self._release_feature_staging_entries(
                        feature_batch.staging_slot, 1
                    )
                    capacity_changed = True
                    continue

                self._input_ready_features.append(
                    InputReadyFeature(
                        pending=item,
                        staging_slot=feature_batch.staging_slot,
                        row_range=row_range,
                        ready_ns=completed_ns,
                    )
                )
                profile_emit(
                    "draft_input_ready",
                    draft_instance=self.instance_id,
                    transfer_id=task.transfer_id,
                    request_id=task.request_id,
                    round_id=task.round_id,
                    staging_slot=feature_batch.staging_slot,
                    ready_ns=completed_ns,
                )
                capacity_changed = True
            if unfinished_pending:
                feature_batch.pending = unfinished_pending
                feature_batch.row_ranges = tuple(unfinished_ranges)
                feature_batch.futures = tuple(unfinished_futures)
                retained.append(feature_batch)
        if scan:
            self._pending_feature_batches = retained
        if self.batch_source_read_done:
            self._flush_source_done()
        self._resume_feature_finalizations()
        if capacity_changed:
            self._publish_capacity()

    def _take_input_ready_batch(self) -> list[InputReadyFeature]:
        """Choose final compute membership only from request-ready inputs."""

        profile_active = getattr(self, "_profile_active", False)
        selection_started_ns = time.monotonic_ns() if profile_active else 0
        if profile_active:
            self._submit_marks = {"select_started": selection_started_ns}
            self._submit_sub_ns = [0] * len(SUBMIT_SUB_STAGES)
        inflight_before = self._draft_compute_batches_inflight()
        waiting_before = tuple(
            entry.pending.task.transfer_id for entry in self._input_ready_features
        )
        if (
            not (self._input_ready_features or self._ready_pending)
            or inflight_before >= DRAFT_COMPUTE_DEPTH
        ):
            if self._ready_pending:  # READY rows wait out the gate (#292)
                self._ready_gated_since_ns = time.monotonic_ns()
            self._profile_input_ready_state(
                observed_ns=selection_started_ns,
                waiting_transfer_ids=waiting_before,
                reason=("compute_depth" if waiting_before else "empty"),
                inflight=inflight_before,
            )
            return []
        busy_handles = {
            item.task.draft_kv_handle
            for owner in (
                *self._pending_dflash_completions,
                *self._pending_eagle3_completions,
            )
            for item in owner.pending
        }
        selected = []
        retained = deque()
        proposal_mode = None
        token_count = 0
        # Persistent compute (DFlash/EAGLE3) merges every non-busy slot into
        # one batch; FlashMTP keeps the single-slot staging read, so its
        # selection stays pinned to the first eligible slot.
        single_slot = None
        now_ns = time.monotonic_ns()
        while self._input_ready_features:
            entry = self._input_ready_features.popleft()
            task = entry.pending.task
            slot_busy = entry.staging_slot in self._feature_staging_compute_busy
            if slot_busy or len(selected) >= self.max_batch_size:
                entry.slot_busy_blocked |= slot_busy
                retained.append(entry)
                continue
            if self._persistent and task.draft_kv_handle in busy_handles:
                retained.append(entry)
                continue
            if not self._persistent:
                if single_slot is None:
                    single_slot = entry.staging_slot
                elif entry.staging_slot != single_slot:
                    retained.append(entry)
                    continue
            if task.transfer_id in self._cancelled_ids or now_ns >= task.deadline_ns:
                cancelled = task.transfer_id in self._cancelled_ids
                if cancelled:
                    self._retire_cancelled_transfer_identity(task)
                if self._persistent:
                    self._release_persistent_request(task.draft_kv_handle)
                if not cancelled:
                    self._publish_failures(
                        [(task, TimeoutError("INPUT_READY task deadline expired"))]
                    )
                self._release_feature_staging_entries(entry.staging_slot, 1)
                continue
            if proposal_mode is None:
                proposal_mode = bool(task.requires_proposal)
            if bool(task.requires_proposal) != proposal_mode:
                retained.append(entry)
                continue
            rows = entry.row_range[1] - entry.row_range[0]
            if token_count + rows > self.feature_buffer_max_tokens:
                retained.append(entry)
                continue
            selected.append(entry)
            token_count += rows
            if self._persistent:
                busy_handles.add(task.draft_kv_handle)
        self._input_ready_features = retained
        if profile_active:
            self._submit_marks["select_scan"] = time.monotonic_ns()
        if self._native_ingress:
            self._freeze_ready_rows(
                selected, busy_handles, proposal_mode, token_count, now_ns
            )
        selection_finished_ns = time.monotonic_ns() if profile_active else 0
        if profile_active:
            self._submit_marks["freeze"] = selection_finished_ns
        if not selected:
            self._profile_input_ready_state(
                observed_ns=selection_finished_ns,
                waiting_transfer_ids=tuple(
                    entry.pending.task.transfer_id for entry in retained
                ),
                reason="slot_or_owner_busy",
                inflight=inflight_before,
            )
            return []
        if profile_active:
            profile_emit(
                "draft_input_ready_select",
                draft_instance=self.instance_id,
                started_ns=selection_started_ns,
                finished_ns=selection_finished_ns,
                inflight_before=inflight_before,
                waiting_before=len(waiting_before),
                waiting_after=len(retained),
                staging_slots=tuple(
                    dict.fromkeys(entry.staging_slot for entry in selected)
                ),
                rows=tuple(
                    dict(
                        transfer_id=entry.pending.task.transfer_id,
                        request_id=entry.pending.task.request_id,
                        round_id=entry.pending.task.round_id,
                        ready_ns=entry.ready_ns,
                    )
                    for entry in selected
                ),
            )
        self._profile_input_ready_state(
            observed_ns=selection_finished_ns,
            waiting_transfer_ids=tuple(
                entry.pending.task.transfer_id for entry in retained
            ),
            reason="selected",
            inflight=inflight_before,
        )
        self._input_ready_select_ns = (selection_started_ns, selection_finished_ns)
        if profile_active:  # the selection's own profile records above
            self._submit_marks["select_profile"] = time.monotonic_ns()
        return selected

    def _profile_input_ready_state(
        self,
        *,
        observed_ns: int,
        waiting_transfer_ids: tuple[str, ...],
        reason: str,
        inflight: int,
    ) -> None:
        """Emit only transitions in request-granular Draft compute admission."""

        if not getattr(self, "_profile_active", False):
            return
        signature = (waiting_transfer_ids, reason, inflight)
        if signature == getattr(self, "_last_input_ready_profile_state", None):
            return
        self._last_input_ready_profile_state = signature
        profile_emit(
            "draft_input_ready_state",
            draft_instance=self.instance_id,
            observed_ns=observed_ns,
            waiting_transfer_ids=waiting_transfer_ids,
            reason=reason,
            inflight=inflight,
        )

    def _execute_input_ready_batch(self, ready: list[InputReadyFeature]) -> None:
        if not ready:
            return
        profile_active = getattr(self, "_profile_active", False)
        submit_started_ns = time.monotonic_ns() if profile_active else 0
        marks = self._submit_marks if profile_active else None
        if marks is not None:
            marks["select_return"] = submit_started_ns
        inflight_before = self._draft_compute_batches_inflight()
        entry_slots = tuple(entry.staging_slot for entry in ready)
        staging_slots = tuple(dict.fromkeys(entry_slots))
        staging_slot_counts = tuple(
            entry_slots.count(slot) for slot in staging_slots
        )
        if not self._slot_decouple:
            if any(
                slot in self._feature_staging_compute_busy for slot in staging_slots
            ):
                raise RuntimeError(
                    "feature staging slot already has submitted compute"
                )
            self._feature_staging_compute_busy.update(staging_slots)
            landing_slot_base = getattr(self, "_landing_slot_base", -1)
            for slot in staging_slots:
                if 0 <= landing_slot_base <= slot:
                    # New ring arrivals land in another buffer instead of
                    # waiting behind this batch's compute ownership.
                    self._ring_worker.seal(slot - landing_slot_base)
        pending = [entry.pending for entry in ready]
        row_ranges = tuple(entry.row_range for entry in ready)
        ready_seqs = tuple(entry.ready_seq for entry in ready if entry.ready_seq)
        try:
            if not self._persistent:
                if len(staging_slots) != 1:
                    raise RuntimeError(
                        "one FlashMTP compute batch cannot span feature staging slots"
                    )
                staging_slot = staging_slots[0]
                staging = self._feature_staging_buffers[staging_slot]
                rows = [staging[start:end] for start, end in row_ranges]
                if any(int(row.shape[0]) != 1 for row in rows):
                    raise RuntimeError(
                        "FlashMTP direct GPU source must provide one pivot row"
                    )
                pivots = torch.cat(rows, dim=0) if len(rows) > 1 else rows[0]
                self._execute_flashmtp_batch(
                    pending, pivots, staging_slot=staging_slot
                )
            elif self._is_eagle3:
                self._execute_eagle3_feature_batch(
                    pending=pending,
                    staging_slots=entry_slots,
                    row_ranges=row_ranges,
                )
            else:
                self._launch_dflash_materialization(
                    pending=pending,
                    staging_slots=entry_slots,
                    row_ranges=row_ranges,
                    ready_seqs=ready_seqs,
                )
            if profile_active:
                submit_returned_ns = time.monotonic_ns()
                marks["submit_tail"] = submit_returned_ns
                select_started_ns, frozen_ns = self._input_ready_select_ns
                if self._persistent and not self._is_eagle3:  # the DFlash launch
                    completion = self._pending_dflash_completions[-1]
                    completion.select_started_ns = select_started_ns
                    completion.frozen_ns = frozen_ns
                    completion.handoff_ns = submit_returned_ns
                profile_emit(
                    "draft_input_ready_submit",
                    draft_instance=self.instance_id,
                    transfer_ids=tuple(
                        entry.pending.task.transfer_id for entry in ready
                    ),
                    request_ids=tuple(
                        entry.pending.task.request_id for entry in ready
                    ),
                    round_ids=tuple(
                        entry.pending.task.round_id for entry in ready
                    ),
                    slot_busy_blocked=tuple(
                        entry.pending.task.transfer_id
                        for entry in ready
                        if entry.slot_busy_blocked
                    ),
                    staging_slots=staging_slots,
                    select_started_ns=select_started_ns,
                    frozen_ns=frozen_ns,
                    submitted_started_ns=submit_started_ns,
                    submitted_returned_ns=submit_returned_ns,
                    submit_wall_ns=submit_returned_ns - submit_started_ns,
                    inflight_before=inflight_before,
                    inflight_after=self._draft_compute_batches_inflight(),
                )
        except Exception as exc:
            logger.exception("failed to execute INPUT_READY Draft batch")
            resources_quiescent = bool(getattr(exc, "resources_quiescent", True))
            if resources_quiescent:
                try:
                    torch.cuda.current_stream(self.device).synchronize()
                except Exception:
                    resources_quiescent = False
            tasks = [item.task for item in pending]
            if resources_quiescent:
                for slot, count in zip(
                    staging_slots, staging_slot_counts, strict=True
                ):
                    self._feature_staging_compute_busy.discard(slot)
                    self._release_feature_staging_entries(slot, count)
                if ready_seqs:
                    self._release_ready_rows(
                        self._ready_row_keys(ready_seqs), fence_seqs=ready_seqs
                    )
                if self._persistent:
                    for handle in {task.draft_kv_handle for task in tasks}:
                        self._release_persistent_request(handle)
            else:
                if self._persistent:
                    self._quarantined_feature_draft_handles.update(
                        task.draft_kv_handle for task in tasks
                    )
                self._enter_fatal_gpu_quarantine(
                    "unquiesced feature proposal/materialization failure"
                )
            self._publish_failures([(task, exc) for task in tasks])
        finally:
            if ready_seqs:
                # After the submit issued its copy-event waits (or failed).
                if marks is not None:  # past the hand-off: C has ended
                    marks["submit_profile"] = time.monotonic_ns()
                self._ready_consume(ready_seqs)
                if marks is not None:
                    marks["ready_consume"] = time.monotonic_ns()

    def _execute_eagle3_feature_batch(
        self,
        *,
        pending: list[PendingDraftTask],
        staging_slots: tuple[int, ...],
        row_ranges: tuple[tuple[int, int], ...],
    ) -> None:
        """Reconcile authoritative Target features, then run remote EAGLE3."""

        from sglang.srt.speculative.eagle_worker_common import prepare_for_draft

        tasks = [item.task for item in pending]
        if not tasks:
            raise ValueError("EAGLE3 feature batch must not be empty")
        if len({bool(task.requires_proposal) for task in tasks}) != 1:
            raise RuntimeError("EAGLE3 feature batch mixed proposal boundaries")
        distinct_slots = tuple(dict.fromkeys(staging_slots))
        staging_slot_counts = tuple(
            staging_slots.count(slot) for slot in distinct_slots
        )

        batch_started_ns = time.monotonic_ns()
        states, grouped_reconcile = self._prepare_eagle3_persistent_tasks(tasks)
        batch = ScheduleBatch.init_new(
            reqs=[state.req for state in states],
            req_to_token_pool=self._draft_model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=(
                self._draft_model_runner.token_to_kv_pool_allocator
            ),
            tree_cache=self._tree_cache,
            model_config=self._draft_model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.EAGLE3,
        )
        if not grouped_reconcile or not _prepare_eagle3_extend_batch(batch):
            batch.prepare_for_extend()
        batch_prepared_ns = time.monotonic_ns()
        # Target sends the exact token aligned with each auxiliary-hidden row
        # (the output-token stream used by stock EAGLE draft-extend), while the
        # request itself retains the authoritative committed input tokens.
        # Consume that stream directly. Reconstructing a per-request preimage
        # only for the stock helper to rotate it back adds a host loop and a GPU
        # kernel to every remote round without changing the model input.
        feature_input_ids: list[int] = []
        tail_token_ids: list[int] = []
        for task in tasks:
            feature_ids = list(task.feature_token_ids)
            if not feature_ids:
                raise ValueError("EAGLE3 feature token stream must not be empty")
            feature_input_ids.extend(int(token_id) for token_id in feature_ids)
            tail_token_ids.append(int(feature_ids[-1]))
            if task.requires_proposal and int(feature_ids[-1]) != int(
                task.next_token_id
            ):
                raise ValueError(
                    "EAGLE3 proposal root differs from the feature-aligned tail"
                )
        # Keep the existing ablation paths intact. Continuous batches must not
        # host-wait for the previous batch when uploading independent inputs.
        # Pinned tensor lifetime is tracked by PyTorch's async-copy allocator;
        # the existing bounded feature slots still limit in-flight batches.
        async_inputs = getattr(self, "continuous_batching", False)
        input_device = "cpu" if async_inputs else self.device
        batch.input_ids = torch.tensor(
            feature_input_ids,
            dtype=torch.int64,
            device=input_device,
            pin_memory=async_inputs,
        ).to(self.device, non_blocking=async_inputs)
        # EAGLE3 consumes the authoritative feature-aligned stream above.
        # The committed-ID buffer from generic extend preparation has no
        # consumer here; do not upload it or let deferred resolution replace
        # the feature-aligned inputs.
        batch.prefill_input_ids_cpu = None
        staging_view_bounds = None
        if len(distinct_slots) == 1:
            # The direct staging view only exists within one slot's buffer;
            # cross-slot entries always materialize through the concatenation.
            staging_view_bounds = _eagle3_feature_staging_view_bounds(
                requires_proposal=bool(tasks[0].requires_proposal),
                row_ranges=row_ranges,
            )
        if staging_view_bounds is None:
            target_hidden = torch.cat(
                [
                    self._feature_staging_buffers[slot][start:end]
                    for slot, (start, end) in zip(
                        staging_slots, row_ranges, strict=True
                    )
                ],
                dim=0,
            )
        else:
            staging = self._feature_staging_buffers[distinct_slots[0]]
            target_hidden = staging[
                staging_view_bounds[0] : staging_view_bounds[1]
            ]
        if int(target_hidden.shape[0]) != int(batch.extend_num_tokens):
            raise RuntimeError(
                "EAGLE3 Target feature rows do not match the committed delta"
            )
        feature_tail_tokens = torch.tensor(
            tail_token_ids,
            dtype=torch.int64,
            device=input_device,
            pin_memory=async_inputs,
        ).to(self.device, non_blocking=async_inputs)
        inputs_ready_ns = time.monotonic_ns()
        draft_input = self._eagle_draft_worker._draft_extend_for_prefill(
            batch,
            target_hidden,
            feature_tail_tokens,
            input_ids_are_feature_aligned=True,
        )
        extend_done_ns = time.monotonic_ns()
        extend_execution = (
            self._draft_model_runner._last_cuda_graph_measurement_execution
            or "missing"
        )
        extend_fallback_reason = (
            self._draft_model_runner._last_cuda_graph_measurement_fallback_reason
            or (
                "model_runner_measurement_event_missing"
                if extend_execution == "missing"
                else None
            )
        )
        extend_event_id = self._draft_model_runner._last_cuda_graph_measurement_event_id
        measurement_tasks = [
            task
            for task in tasks
            if task.request_id.startswith(DT_BENCHMARK_MEASUREMENT_RID_PREFIX)
        ]
        if measurement_tasks:
            measurement_bootstrap = sum(
                task.round_id == 0 for task in measurement_tasks
            )
            measurement_reconcile = len(measurement_tasks) - measurement_bootstrap
            if measurement_bootstrap:
                self._record_graph_measurement(
                    path="eagle3_bootstrap_extend",
                    execution=extend_execution,
                    measurement_rows=measurement_bootstrap,
                    measurement_batch_rows=len(measurement_tasks),
                    batch_size=len(tasks),
                    model_runner_event_id=extend_event_id,
                    fallback_reason=extend_fallback_reason,
                )
            if measurement_reconcile:
                self._record_graph_measurement(
                    path="eagle3_reconcile_extend",
                    execution=extend_execution,
                    measurement_rows=measurement_reconcile,
                    measurement_batch_rows=len(measurement_tasks),
                    batch_size=len(tasks),
                    model_runner_event_id=extend_event_id,
                    fallback_reason=extend_fallback_reason,
                )

        for state in states:
            state.req.kv_committed_len = state.valid_committed_len
            state.proposal_base_len = state.valid_committed_len

        if not tasks[0].requires_proposal:
            if profile_event_enabled("draft_eagle3_batch"):
                batch_finished_ns = time.monotonic_ns()
                profile_emit(
                    "draft_eagle3_batch",
                    draft_instance=self.instance_id,
                    batch_size=len(tasks),
                    transfer_ids=tuple(task.transfer_id for task in tasks),
                    measurement_rows=sum(
                        task.request_id.startswith(
                            DT_BENCHMARK_MEASUREMENT_RID_PREFIX
                        )
                        for task in tasks
                    ),
                    proposal_rows=0,
                    materialization_only_rows=len(tasks),
                    feature_tokens=int(batch.extend_num_tokens),
                    requires_proposal=False,
                    feature_staging_view=staging_view_bounds is not None,
                    grouped_reconcile=grouped_reconcile,
                    extend_execution=extend_execution,
                    extend_fallback_reason=extend_fallback_reason,
                    proposal_execution=None,
                    proposal_fallback_reason=None,
                    pending_before=pending[0].pending_before,
                    pending_after=pending[0].pending_after,
                    taken_ns=pending[0].taken_ns,
                    oldest_queue_age_ns=pending[0].oldest_queue_age_ns,
                    max_batch_size=self.max_batch_size,
                    batch_feature_token_count=(
                        pending[0].batch_feature_token_count
                    ),
                    stop_reason=pending[0].stop_reason,
                    cap_limited=pending[0].cap_limited,
                    execute_started_ns=batch_started_ns,
                    proposal_enqueued_ns=0,
                    candidate_host_ready_ns=0,
                    stream_fence_completed_ns=0,
                    batch_started_ns=batch_started_ns,
                    batch_finished_ns=batch_finished_ns,
                    state_batch_prepare_ns=(
                        batch_prepared_ns - batch_started_ns
                    ),
                    input_pack_ns=inputs_ready_ns - batch_prepared_ns,
                    extend_ns=extend_done_ns - inputs_ready_ns,
                    post_extend_host_ns=batch_finished_ns - extend_done_ns,
                    proposal_prepare_ns=0,
                    proposal_execute_ns=0,
                    candidate_d2h_ns=0,
                    candidate_completion_wait_ns=0,
                    candidate_completion_wait_semantics="not_applicable",
                    stream_fence_ns=0,
                    total_ns=batch_finished_ns - batch_started_ns,
                )
            self._enqueue_eagle3_materialization_completion(
                pending,
                distinct_slots,
                staging_slot_counts=staging_slot_counts,
            )
            return

        batch.spec_info = draft_input
        # The feature-aligned tail was checked against next_token_id above.
        # Extend does not mutate it; reuse avoids an H2D host fence after extend.
        bonus_tokens = (
            feature_tail_tokens
            if async_inputs
            else torch.tensor(
                [task.next_token_id for task in tasks],
                dtype=torch.int64,
                device=self.device,
            )
        )
        # The co-located scheduler crosses an iteration boundary between draft
        # extend and proposal decode.  The remote engine performs both within
        # one task, so it must make the same ScheduleBatch mode transition here;
        # calling the allocation helper alone leaves ForwardMode.EXTEND and the
        # logits processor then indexes a one-row decode output as a prefill.
        batch.prepare_for_decode()
        forward_batch, can_cuda_graph = prepare_for_draft(
            draft_input,
            self._draft_model_runner.req_to_token_pool,
            batch,
            self._eagle_draft_worker.cuda_graph_runner,
            self._draft_model_runner,
            self._eagle_draft_worker.topk,
            self._eagle_draft_worker.speculative_num_steps,
        )
        proposal_prepared_ns = time.monotonic_ns()
        if can_cuda_graph:
            _parents, _scores, proposals, _draft_probs = (
                self._eagle_draft_worker.cuda_graph_runner.execute(forward_batch)
            )
            if not self.__dict__.get("_eagle3_graph_replay_evidence_emitted", False):
                self._eagle3_graph_replay_evidence_emitted = True
                logger.info(
                    "DT EAGLE3 Draft CUDA graph replay observed: "
                    "mode=DRAFT_DECODE batch_size=%d",
                    len(tasks),
                )
            execution = "graph"
            fallback_reason = None
        else:
            if self._eagle_draft_worker.speculative_num_steps > 1:
                self._eagle_draft_worker.draft_attn_backend.init_forward_metadata(
                    forward_batch
                )
                forward_batch.mark_forward_metadata_ready()
            _parents, _scores, proposals, _draft_probs = (
                self._eagle_draft_worker.draft_forward(forward_batch)
            )
            execution = "eager"
            fallback_reason = "draft_decode_runner_rejected"
        proposal_done_ns = time.monotonic_ns()

        candidates = torch.cat(
            [bonus_tokens.view(-1, 1), proposals.to(torch.int64)], dim=1
        ).to(torch.int32)
        expected_shape = (len(tasks), self.profile.block_size)
        if tuple(candidates.shape) != expected_shape:
            raise RuntimeError(
                "EAGLE3 proposal shape mismatch: "
                f"expected {expected_shape}, got {tuple(candidates.shape)}"
            )
        measurement_rows = sum(
            task.request_id.startswith(DT_BENCHMARK_MEASUREMENT_RID_PREFIX)
            for task in tasks
        )
        if measurement_rows:
            self._record_graph_measurement(
                path="eagle3_proposal_decode",
                execution=execution,
                measurement_rows=measurement_rows,
                measurement_batch_rows=measurement_rows,
                batch_size=len(tasks),
                fallback_reason=fallback_reason,
        )
        if getattr(self, "continuous_batching", False):
            # cat/to above already snapshots graph-owned proposals on the
            # forward stream. The slot owns candidates, optional pinned rows,
            # and feature reads until its completion event.
            profile_emit(
                "draft_eagle3_batch",
                draft_instance=self.instance_id,
                batch_size=len(tasks),
                transfer_ids=tuple(task.transfer_id for task in tasks),
                execute_started_ns=batch_started_ns,
                batch_finished_ns=time.monotonic_ns(),
                proposal_enqueued_ns=proposal_done_ns,
                state_batch_prepare_ns=batch_prepared_ns - batch_started_ns,
                input_pack_ns=inputs_ready_ns - batch_prepared_ns,
                extend_ns=extend_done_ns - inputs_ready_ns,
                proposal_prepare_ns=proposal_prepared_ns - extend_done_ns,
                proposal_execute_ns=proposal_done_ns - proposal_prepared_ns,
                taken_ns=pending[0].taken_ns,
                oldest_queue_age_ns=pending[0].oldest_queue_age_ns,
                asynchronous_completion=True,
                candidate_completion_wait_semantics=(
                    "deferred_gpu_snapshot_event"
                    if self.candidate_transport == "cuda-ipc"
                    else "deferred_pinned_D2H_event"
                ),
            )
            self._enqueue_eagle3_materialization_completion(
                pending,
                distinct_slots,
                staging_slot_counts=staging_slot_counts,
                candidates=candidates,
            )
            return
        if self.candidate_transport == "cuda-ipc":
            self._publish_successes(tasks, candidates.contiguous())
            self._complete_feature_compute(distinct_slots, staging_slot_counts)
            return
        candidate_bytes = candidates.cpu().contiguous().numpy().tobytes()
        candidate_ready_ns = time.monotonic_ns()
        self._fence_eagle3_feature_staging_reads()
        if profile_event_enabled("draft_eagle3_batch"):
            stream_fence_completed_ns = time.monotonic_ns()
            profile_emit(
                "draft_eagle3_batch",
                draft_instance=self.instance_id,
                batch_size=len(tasks),
                transfer_ids=tuple(task.transfer_id for task in tasks),
                measurement_rows=measurement_rows,
                proposal_rows=len(tasks),
                materialization_only_rows=0,
                feature_tokens=int(batch.extend_num_tokens),
                requires_proposal=True,
                feature_staging_view=staging_view_bounds is not None,
                grouped_reconcile=grouped_reconcile,
                extend_execution=extend_execution,
                extend_fallback_reason=extend_fallback_reason,
                proposal_execution=execution,
                proposal_fallback_reason=fallback_reason,
                pending_before=pending[0].pending_before,
                pending_after=pending[0].pending_after,
                taken_ns=pending[0].taken_ns,
                oldest_queue_age_ns=pending[0].oldest_queue_age_ns,
                max_batch_size=self.max_batch_size,
                batch_feature_token_count=(
                    pending[0].batch_feature_token_count
                ),
                stop_reason=pending[0].stop_reason,
                cap_limited=pending[0].cap_limited,
                execute_started_ns=batch_started_ns,
                proposal_enqueued_ns=proposal_done_ns,
                candidate_host_ready_ns=candidate_ready_ns,
                stream_fence_completed_ns=stream_fence_completed_ns,
                batch_started_ns=batch_started_ns,
                batch_finished_ns=stream_fence_completed_ns,
                state_batch_prepare_ns=batch_prepared_ns - batch_started_ns,
                input_pack_ns=inputs_ready_ns - batch_prepared_ns,
                extend_ns=extend_done_ns - inputs_ready_ns,
                post_extend_host_ns=0,
                proposal_prepare_ns=proposal_prepared_ns - extend_done_ns,
                proposal_execute_ns=proposal_done_ns - proposal_prepared_ns,
                candidate_d2h_ns=candidate_ready_ns - proposal_done_ns,
                candidate_completion_wait_ns=(
                    candidate_ready_ns - proposal_done_ns
                ),
                candidate_completion_wait_semantics=(
                    "blocking_cpu_materialization_after_same_stream_proposal; "
                    "not isolated D2H"
                ),
                stream_fence_ns=(
                    stream_fence_completed_ns - candidate_ready_ns
                ),
                total_ns=stream_fence_completed_ns - batch_started_ns,
            )
        completed_ns = time.monotonic_ns()
        expired = [task for task in tasks if completed_ns >= task.deadline_ns]
        if expired:
            for task in expired:
                self._release_persistent_request(task.draft_kv_handle)
            self._publish_failures(
                [
                    (
                        task,
                        TimeoutError("EAGLE3 proposal completed after its deadline"),
                    )
                    for task in expired
                ]
            )
            active_indices = [
                index for index, task in enumerate(tasks) if task not in expired
            ]
            if active_indices:
                active_rows = candidates[active_indices].cpu().contiguous()
                self._publish_successes(
                    [tasks[index] for index in active_indices],
                    active_rows.numpy().tobytes(),
                )
        else:
            self._publish_successes(tasks, candidate_bytes)
        self._complete_feature_compute(distinct_slots, staging_slot_counts)

    def _fence_eagle3_feature_staging_reads(self) -> None:
        """Fence every consumer before the independent pull stream can overwrite."""

        try:
            torch.cuda.current_stream(self.device).synchronize()
        except Exception as exc:
            error = RuntimeError(
                "failed to fence EAGLE3 feature staging before slot reuse"
            )
            error.resources_quiescent = False
            raise error from exc

    def _enqueue_eagle3_materialization_completion(
        self,
        pending: list[PendingDraftTask],
        staging_slots: tuple[int, ...] | None,
        *,
        staging_slot_counts: tuple[int, ...] = (),
        candidates: torch.Tensor | None = None,
        result_slot: int | None = None,
        admission_credit: bool = False,
    ) -> None:
        """Keep persistent KV and slot buffers owned until GPU/D2H completion.

        EAGLE3 retains its feature slots. AR has no feature staging and instead
        reserves a result slot (mmap) or proposal-admission credit (cuda-ipc),
        using the same completion/release machinery.
        """
        host_rows = None
        if candidates is not None and self.candidate_transport == "mmap":
            buffer_slot = staging_slots[0] if staging_slots else result_slot
            if buffer_slot is None:
                raise RuntimeError("persistent proposal has no bounded buffer slot")
            host_rows = self._candidate_result_host_buffers[buffer_slot][:len(pending)]
        completion = PendingEagle3Completion(
            pending=pending,
            staging_slots=tuple(staging_slots or ()),
            staging_slot_counts=staging_slot_counts,
            completion_event=torch.cuda.Event(),
            submitted_ns=time.monotonic_ns(),
            candidates=candidates,
            candidate_rows_cpu=host_rows,
            result_slot=result_slot,
            admission_credit=admission_credit,
        )
        try:
            if host_rows is not None:
                host_rows.copy_(candidates, non_blocking=True)
            completion.completion_event.record(torch.cuda.current_stream(self.device))
        except Exception as exc:
            try:
                torch.cuda.current_stream(self.device).synchronize()
            except Exception:
                self._quarantined_eagle3_completions.append(completion)
                self._quarantined_feature_draft_handles.update(
                    item.task.draft_kv_handle for item in pending
                )
                self._enter_fatal_gpu_quarantine("unquiesced persistent candidate copy")
                exc.resources_quiescent = False
            raise
        self._pending_eagle3_completions.append(completion)
        if candidates is not None and self.candidate_transport == "cuda-ipc":
            self._queue_early_candidate_push(
                [item.task for item in pending],
                candidates,
                completion.completion_event,
            )

    def _poll_eagle3_completions(self) -> None:
        if self._fatal_gpu_quarantine_active():
            return
        capacity_changed = False
        while self._pending_eagle3_completions:
            completion = self._pending_eagle3_completions[0]
            try:
                ready = completion.completion_event.query()
            except Exception as exc:
                self._pending_eagle3_completions.popleft()
                self._quarantined_eagle3_completions.append(completion)
                tasks = [item.task for item in completion.pending]
                self._quarantined_feature_draft_handles.update(
                    task.draft_kv_handle for task in tasks
                )
                exc.resources_quiescent = False
                logger.critical(
                    "quarantined EAGLE3 staging and KV after completion-event "
                    "query failure: transfers=%s",
                    [task.transfer_id for task in tasks],
                    exc_info=True,
                )
                self._enter_fatal_gpu_quarantine(
                    "unquiesced EAGLE3 completion-event failure"
                )
                # Publication may raise.  Fatal quarantine and resource
                # ownership must already be permanent before attempting it.
                self._publish_failures([(task, exc) for task in tasks])
                self._publish_capacity()
                break
            if not ready:
                break
            self._pending_eagle3_completions.popleft()
            tasks = [item.task for item in completion.pending]
            completed_ns = time.monotonic_ns()
            # A push queued at launch already published these candidates.
            early = self._take_early_pushed(tasks)
            expired = [
                task
                for task in tasks
                if task.transfer_id not in self._cancelled_ids
                and task.transfer_id not in early
                and completed_ns >= task.deadline_ns
            ]
            expired_ids = {task.transfer_id for task in expired}
            active = [
                task
                for task in tasks
                if task.transfer_id not in self._cancelled_ids
                and task.transfer_id not in expired_ids
            ]
            cancelled = [
                task for task in tasks if task.transfer_id in self._cancelled_ids
            ]
            candidate_rows_cpu = getattr(completion, "candidate_rows_cpu", None)
            result_slot = getattr(completion, "result_slot", None)
            admission_credit = getattr(completion, "admission_credit", False)
            push_active = [task for task in active if task.transfer_id not in early]
            if active and not push_active and completion.candidates is not None:
                if result_slot is not None:
                    self._free_result_slots.append(result_slot)
                if admission_credit:
                    # Every live task was early-pushed at launch; no publish
                    # will claim the completion's credit.
                    self._release_direct_proposal_admission()
            elif active:
                if completion.candidates is None:
                    self._publish_materialization_successes(active)
                else:
                    active_ids = {task.transfer_id for task in push_active}
                    indices = [
                        index for index, task in enumerate(tasks)
                        if task.transfer_id in active_ids
                    ]
                    source_rows = (
                        completion.candidates
                        if self.candidate_transport == "cuda-ipc"
                        else candidate_rows_cpu
                    )
                    if source_rows is None:
                        raise RuntimeError("persistent candidate rows are unavailable")
                    with self._direct_push_scope(
                        completion.completion_event, source_rows
                    ):
                        rows = _select_candidate_rows(source_rows, indices)
                        self._publish_successes(
                            push_active,
                            (
                                rows
                                if self.candidate_transport == "cuda-ipc"
                                else rows.numpy().tobytes()
                            ),
                            result_slot=result_slot,
                            admission_credit=admission_credit,
                        )
            elif result_slot is not None:
                self._free_result_slots.append(result_slot)
            elif admission_credit:
                self._release_direct_proposal_admission()
            for task in cancelled:
                self._retire_cancelled_transfer_identity(task)
                self._release_persistent_request(task.draft_kv_handle)
            for task in expired:
                self._release_persistent_request(task.draft_kv_handle)
            if expired:
                self._publish_failures(
                    [
                        (
                            task,
                            TimeoutError(
                                "persistent Draft work completed after its deadline"
                            ),
                        )
                        for task in expired
                    ]
                )
            completion_event_name = (
                "draft_persistent_proposal_complete"
                if completion.candidates is not None
                else "draft_eagle3_materialization_complete"
            )
            if profile_event_enabled(completion_event_name):
                profile_emit(
                    completion_event_name,
                    draft_instance=self.instance_id,
                    batch_size=len(tasks),
                    transfer_ids=tuple(task.transfer_id for task in tasks),
                    measurement_rows=sum(
                        task.request_id.startswith(
                            DT_BENCHMARK_MEASUREMENT_RID_PREFIX
                        )
                        for task in tasks
                    ),
                    active_task_count=len(active),
                    cancelled_task_count=len(cancelled),
                    expired_task_count=len(expired),
                    submitted_ns=completion.submitted_ns,
                    gpu_completed_ns=completed_ns,
                )
            if completion.staging_slots:
                self._complete_feature_compute(
                    completion.staging_slots, completion.staging_slot_counts
                )
            for handle in {task.draft_kv_handle for task in tasks}:
                self._drain_deferred_persistent_release(handle)
            capacity_changed = True
        if capacity_changed:
            self._publish_capacity()

    def _launch_dflash_materialization(
        self,
        *,
        pending: list[PendingDraftTask],
        staging_slots: tuple[int, ...],
        row_ranges: tuple[tuple[int, int], ...],
        ready_seqs: tuple[int, ...] = (),
    ) -> None:
        if self._pending_dflash_completions and not self.dflash_submit_pipeline:
            raise RuntimeError(
                "DFLASH materialization submission must wait for the prior "
                "completion event"
            )
        # staging_slots is per entry, parallel to row_ranges.  Temporary H2D
        # staging (proposal metadata, candidate rows) is indexed by slot; any
        # slot this batch owns is valid, so use the first.  Under
        # DT_DRAFT_SLOT_DECOUPLE it is indexed by batch_seq % depth instead:
        # the depth gate and FIFO retirement free that scratch first.
        slots_used = tuple(dict.fromkeys(staging_slots))
        staging_slot_counts = tuple(
            staging_slots.count(slot) for slot in slots_used
        )
        batch_seq = -1
        metadata_slot = slots_used[0]
        if self._slot_decouple:
            batch_seq = self._dflash_batch_seq
            metadata_slot = batch_seq % DRAFT_COMPUTE_DEPTH
            if any(
                completion.batch_seq % DRAFT_COMPUTE_DEPTH == metadata_slot
                for completion in self._pending_dflash_completions
            ):
                raise RuntimeError("DFLASH per-batch scratch is still owned")
        prior_unfinished_proposal_count = 0
        prior_candidate_copy_pending_count = 0
        for completion in self._pending_dflash_completions:
            if not completion.proposal_done_event.query():
                prior_unfinished_proposal_count += 1
            elif not completion.completion_event.query():
                prior_candidate_copy_pending_count += 1
        launch_started_ns = time.monotonic_ns()
        marks = self._submit_marks if getattr(self, "_profile_active", False) else None
        if marks is not None:
            marks["submit_entry"] = launch_started_ns
        tasks = [item.task for item in pending]
        if getattr(self, "continuous_batching", False) and len({bool(task.requires_proposal) for task in tasks}) > 1:
            raise RuntimeError("DFLASH frozen batch mixed proposal boundaries")
        from dt_protocol import DraftTaskKind

        reconcile_prepare_phase_ns = [0] * 6 if profile_enabled() else None
        if (
            tasks
            and all(task.kind == DraftTaskKind.RECONCILE for task in tasks)
            and len({task.draft_kv_handle for task in tasks}) == len(tasks)
        ):
            states, metadata, total_tokens = self._reconcile_dflash_rows(
                tasks,
                metadata_host=self._dflash_metadata_host_buffers[metadata_slot],
                metadata_device=self._dflash_metadata_buffers[metadata_slot],
                reconcile_prepare_phase_ns=reconcile_prepare_phase_ns,
                submit_sub_ns=self._submit_sub_ns if marks is not None else None,
            )
            (
                req_pool_indices,
                feature_offsets,
                committed_lens,
                bonus_tokens,
                allocated_lens,
            ) = metadata
            token_counts = committed_lens - feature_offsets
            if self._canonical_draft_device(self.device).type == "cuda":
                positions, _ = compute_position(
                    self._draft_model_runner.server_args.attention_backend,
                    feature_offsets,
                    token_counts,
                    total_tokens,
                )
            else:
                positions, _ = compute_position_torch(feature_offsets, token_counts)
            # The committed range already maps to pages each row owns, so the
            # materializer's cache_loc is one gather from the page table.
            # output_size pins the result length: without it repeat_interleave
            # syncs the submit thread on the token-count sum.
            packed_cache_locs = self._draft_model_runner.req_to_token_pool.req_to_token[
                torch.repeat_interleave(
                    req_pool_indices, token_counts, output_size=total_tokens
                ),
                positions,
            ].to(torch.int64)
        else:
            states = []
            cache_locs = []
            position_parts = []
            for task in tasks:
                state, cache_loc, position = self._prepare_dflash_task(
                    task, reconcile_prepare_phase_ns
                )
                states.append(state)
                cache_locs.append(cache_loc)
                position_parts.append(position)
            packed_cache_locs = torch.cat(cache_locs, dim=0)
            positions = torch.cat(position_parts, dim=0)
            metadata = None
        tasks_prepared_ns = time.monotonic_ns()
        if marks is not None:
            marks["task_prepare"] = tasks_prepared_ns
        if ready_seqs:
            # Before the first read of the landing rows (the gather below).
            self._wait_ready_copies(ready_seqs)
        if marks is not None:
            marks["copy_waits"] = time.monotonic_ns()
        contiguous = len(slots_used) == 1 and all(
            row_ranges[index][1] == row_ranges[index + 1][0]
            for index in range(len(row_ranges) - 1)
        )
        gather_source = None
        if (
            not contiguous
            and self._ring_landing is not None
            and isinstance(self._dflash_materializer, _GraphedDFlashMaterializer)
            and all(slot >= self._landing_slot_base for slot in slots_used)
        ):
            # All-native-ring batch: gather the rows from the landing buffer
            # straight into the graphed materializer's static hidden buffer
            # (one index_select) instead of per-row views + torch.cat + the
            # copy_ into that buffer it would need.  Mixed/relayed batches
            # keep the staging views.  The index is one vectorized fill of
            # the pinned per-slot buffer (arange(total) - repeat(run starts)
            # + repeat(firsts)); a per-run np.arange loop costs more host
            # time than the cat it replaces.
            host = self._dflash_gather_index_host_buffers[metadata_slot]
            counts = np.fromiter(
                (end - start for _, (start, end) in zip(staging_slots, row_ranges, strict=True)),
                dtype=np.int64,
                count=len(row_ranges),
            )
            firsts = np.fromiter(
                (
                    (slot - self._landing_slot_base) * self.feature_buffer_max_tokens + start
                    for slot, (start, _) in zip(staging_slots, row_ranges, strict=True)
                ),
                dtype=np.int64,
                count=len(row_ranges),
            )
            total = int(counts.sum())
            host_np = host.numpy()
            host_np[:total] = (
                np.arange(total, dtype=np.int64)
                - np.repeat(np.cumsum(counts) - counts, counts)
                + np.repeat(firsts, counts)
            )
            gather_index = host[:total].to(self.device, non_blocking=True)
            gather_source = self._ring_landing.view(-1, self._ring_landing.shape[-1])
            target_hidden = None
        elif contiguous:
            # The contiguous-range view only exists within one slot's buffer.
            target_hidden = self._feature_staging_buffers[slots_used[0]][
                row_ranges[0][0] : row_ranges[-1][1]
            ]
        else:
            hidden_rows = [
                self._feature_staging_buffers[slot][start:end]
                for slot, (start, end) in zip(staging_slots, row_ranges, strict=True)
            ]
            target_hidden = torch.cat(hidden_rows, dim=0)
        materialize_start = torch.cuda.Event(enable_timing=True)
        materialize_done = torch.cuda.Event(enable_timing=True)
        proposal_done = torch.cuda.Event(enable_timing=True)
        proposal_indices = [
            index for index, task in enumerate(tasks) if task.requires_proposal
        ]
        proposal_tasks = tuple(tasks[index] for index in proposal_indices)
        proposal_states = [states[index] for index in proposal_indices]
        proposal_req_pool_indices = None
        proposal_prefix_lens = None
        proposal_alloc_starts = None
        proposal_bonus_tokens = None
        if marks is not None:
            marks["hidden_view"] = time.monotonic_ns()
        if proposal_indices:
            if metadata is not None and len(proposal_indices) == len(tasks):
                # The single packed upload already carries every proposal
                # input; only the tail-start view is derived here.
                proposal_req_pool_indices = req_pool_indices
                proposal_prefix_lens = committed_lens
                proposal_alloc_starts = torch.maximum(allocated_lens, committed_lens)
                proposal_bonus_tokens = bonus_tokens
            else:
                proposal_req_pool_indices = torch.tensor(
                    [state.req.req_pool_idx for state in proposal_states],
                    dtype=torch.int64,
                    device=self.device,
                )
                proposal_prefix_lens = torch.tensor(
                    [state.valid_committed_len for state in proposal_states],
                    dtype=torch.int64,
                    device=self.device,
                )
                proposal_alloc_starts = torch.tensor(
                    [state.req.kv.kv_allocated_len for state in proposal_states],
                    dtype=torch.int64,
                    device=self.device,
                )
                proposal_bonus_tokens = torch.tensor(
                    [task.next_token_id for task in proposal_tasks],
                    dtype=torch.int64,
                    device=self.device,
                )
        proposal_metadata_ready_ns = time.monotonic_ns()
        if marks is not None:
            marks["metadata"] = proposal_metadata_ready_ns

        materialize_start.record(torch.cuda.current_stream(self.device))
        if gather_source is not None:
            self._dflash_materializer.append_rows(
                source=gather_source,
                rows_index=gather_index,
                cache_loc=packed_cache_locs,
                positions=positions,
            )
        else:
            self._dflash_materializer.append(
                target_hidden=target_hidden,
                cache_loc=packed_cache_locs,
                positions=positions,
            )
        materialize_done.record(torch.cuda.current_stream(self.device))
        materialize_submitted_ns = time.monotonic_ns()
        if marks is not None:
            marks["materialize_launch"] = materialize_submitted_ns

        candidates = None
        proposal_execution = None
        proposal_fallback_reason = None
        if proposal_indices:
            (
                candidates,
                proposal_execution,
                proposal_fallback_reason,
            ) = self._run_dflash_proposal(
                states=proposal_states,
                req_pool_indices=proposal_req_pool_indices,
                prefix_lens=proposal_prefix_lens,
                alloc_starts=proposal_alloc_starts,
                bonus_tokens=proposal_bonus_tokens,
            )
        proposal_done.record(torch.cuda.current_stream(self.device))
        if marks is not None:
            marks["proposal_tail"] = time.monotonic_ns()
        clock_slot, clock_done = -1, None
        if self._gpu_clock_enabled:
            if self._gpu_clock is None:
                from sglang.srt.speculative.dt_gpu_clock import GpuClock

                # self.device may carry no index; GpuClock needs one (as T).
                self._gpu_clock = GpuClock(
                    torch.device("cuda", torch.cuda.current_device())
                )
            clock_slot, clock_done = self._gpu_clock.stamp_after(proposal_done)
        candidate_rows_cpu = None
        copy_stream = None
        if (
            candidates is not None
            and self.dflash_submit_pipeline
            and self.candidate_transport == "mmap"
        ):
            copy_stream = self._candidate_result_copy_stream
            if copy_stream is None:
                raise RuntimeError("DFLASH candidate copy stream is unavailable")
            candidate_rows_cpu = self._candidate_result_host_buffers[metadata_slot][
                : len(proposal_tasks)
            ]
        completion = PendingDFlashCompletion(
            pending=pending,
            staging_slots=slots_used,
            staging_slot_counts=staging_slot_counts,
            # The candidate-copy path replaces this with its own event after
            # ownership is registered and before submitting work to that stream.
            completion_event=proposal_done,
            materialize_start_event=materialize_start,
            materialize_done_event=materialize_done,
            proposal_done_event=proposal_done,
            proposal_tasks=proposal_tasks,
            candidates=candidates,
            candidate_rows_cpu=candidate_rows_cpu,
            launch_started_ns=launch_started_ns,
            tasks_prepared_ns=tasks_prepared_ns,
            proposal_metadata_ready_ns=proposal_metadata_ready_ns,
            materialize_submitted_ns=materialize_submitted_ns,
            launched_ns=0,
            proposal_execution=proposal_execution,
            proposal_fallback_reason=proposal_fallback_reason,
            reconcile_prepare_phase_ns=tuple(reconcile_prepare_phase_ns or ()),
            prior_unfinished_proposal_count=prior_unfinished_proposal_count,
            prior_candidate_copy_pending_count=prior_candidate_copy_pending_count,
            clock_slot=clock_slot,
            clock_done=clock_done,
            batch_seq=batch_seq,
            ready_rows=self._ready_row_keys(ready_seqs) if ready_seqs else (),
        )
        self._register_dflash_completion(
            completion=completion,
            copy_stream=copy_stream,
        )
        if marks is not None:
            marks["register"] = completion.launched_ns
            completion.submit_marks = marks
            completion.submit_sub_ns = self._submit_sub_ns
        if self._slot_decouple:
            # A failed launch is quiesced (or fatal) and keeps its sequence.
            self._dflash_batch_seq = batch_seq + 1
        if self._send_loop:
            if self._send_loop_worker is None:
                # The loop starts with the ring worker, at the first Target
                # registration; no batch exists before one.
                raise RuntimeError("DFLASH batch submitted before the send loop")
            # HAND-OFF (design §5.1): frozen_seq is batch_seq + 1, and the
            # send loop pushes this batch's landing releases once its GPU
            # work is done, and its late and cross-host rows (D9).  The batch
            # is registered and launched: if the hand-off fails there is no
            # watcher for GPU work that is already running, so this is fatal
            # and nothing it touched may be reused.
            try:
                releases = self._resolve_ready_releases(completion.ready_rows)
                completion.ready_rows = ()
                if marks is not None:
                    marks["handoff_releases"] = time.monotonic_ns()
                push, push_tasks, prequeue = (), proposal_tasks, None
                if copy_stream is None and proposal_tasks and candidates is not None:
                    push, push_tasks, prequeue = self._hand_off_pushes(
                        batch_seq + 1, proposal_tasks, candidates
                    )
                if marks is not None:
                    marks["handoff_classify"] = time.monotonic_ns()
                self._submit_batch_descriptor(
                    batch_seq + 1,
                    completion.completion_event,
                    releases,
                    timing=(
                        materialize_start.cuda_event,
                        materialize_done.cuda_event,
                        proposal_done.cuda_event,
                    ),
                    push=push,
                )
                if marks is not None:
                    marks["handoff_descriptor"] = time.monotonic_ns()
            except Exception as exc:
                self._pending_dflash_completions.remove(completion)
                self._quarantined_dflash_completions.append(completion)
                self._quarantined_feature_draft_handles.update(
                    item.task.draft_kv_handle for item in completion.pending
                )
                exc.resources_quiescent = False
                self._enter_fatal_gpu_quarantine("DFLASH batch hand-off failed")
                raise
            if prequeue:
                self._queue_early_candidate_push(
                    push_tasks, candidates, completion.completion_event, prequeue
                )
            if marks is not None:
                marks["handoff_prequeue"] = time.monotonic_ns()
        elif copy_stream is None and proposal_tasks:
            self._queue_early_candidate_push(
                proposal_tasks, candidates, completion.completion_event
            )
        stage_ns = self._loop_stage_ns
        stage_ns["submit_task_prepare"] += tasks_prepared_ns - launch_started_ns
        stage_ns["submit_metadata"] += proposal_metadata_ready_ns - tasks_prepared_ns
        stage_ns["submit_launch"] += (
            completion.launched_ns - proposal_metadata_ready_ns
        )
        self._loop_stage_calls["submit_task_prepare"] += 1

    def _register_dflash_completion(
        self,
        *,
        completion: PendingDFlashCompletion,
        copy_stream,
    ) -> None:
        """Own a DFlash completion before an asynchronous candidate copy.

        A failed event record does not prove that the preceding D2H copy was
        rejected.  Registering first ensures that a failed recovery fence can
        permanently retain the device candidates, timing/completion events, and
        pinned destination buffer independently of exception tracebacks.
        """

        self._pending_dflash_completions.append(completion)

        def remove_pending_owner() -> None:
            # This method runs on the single Draft scheduler thread, but use
            # identity rather than dataclass equality because tensor fields do
            # not have scalar equality semantics.
            self._pending_dflash_completions = deque(
                item
                for item in self._pending_dflash_completions
                if item is not completion
            )

        try:
            if copy_stream is not None:
                if (
                    completion.candidates is None
                    or completion.candidate_rows_cpu is None
                ):
                    raise RuntimeError(
                        "DFLASH candidate copy ownership is incomplete"
                    )
                completion.completion_event = torch.cuda.Event()
                with torch.cuda.stream(copy_stream):
                    copy_stream.wait_event(completion.proposal_done_event)
                    completion.candidate_rows_cpu.copy_(
                        completion.candidates, non_blocking=True
                    )
                    completion.completion_event.record(copy_stream)
            completion.launched_ns = time.monotonic_ns()
        except Exception as exc:
            # Registered ownership may be retired only after every stream that
            # could reference it is proven quiescent.
            if copy_stream is not None:
                try:
                    copy_stream.synchronize()
                except Exception as sync_exc:
                    # Move ownership before latching or reporting.  The fatal
                    # path never queries these events or releases their tensors;
                    # repeated close calls stop at the common latch.
                    self._quarantined_dflash_completions.append(completion)
                    remove_pending_owner()
                    exc.resources_quiescent = False
                    sync_exc.resources_quiescent = False
                    self._enter_fatal_gpu_quarantine(
                        "unquiesced DFlash candidate-copy failure"
                    )
                    raise sync_exc from exc
            remove_pending_owner()
            raise

    def _reconcile_dflash_rows(
        self,
        tasks,
        *,
        metadata_host: torch.Tensor,
        metadata_device: torch.Tensor,
        reconcile_prepare_phase_ns: list[int] | None = None,
        submit_sub_ns: list[int] | None = None,
    ) -> tuple[list, tuple[torch.Tensor, ...], int]:
        """Advance one grouped DFlash reconcile wave without moving pages.

        A proposal leaves each row's lookahead block mapped at
        [valid, valid + block_size), so the next reconcile's committed range
        [offset, committed) already sits on pages the row owns: no free, no
        allocation, and no page-table write in the common case.  Only a row
        whose committed boundary outgrew its allocation (bootstrap's first
        reconcile, rounds without a proposal) tops up the shortfall.  The
        per-row metadata is packed once and uploaded with a single H2D that
        also serves the materialize gather and the proposal block.

        Returns (states, metadata_fields, total_feature_tokens), where
        metadata_fields are the device-side per-field rows ordered by
        _DFLASH_META_*.
        """

        phase_started_ns = (
            time.monotonic_ns() if reconcile_prepare_phase_ns is not None else 0
        )
        states = []
        req_pool_indices = []
        feature_offsets = []
        committed_lens = []
        bonus_tokens = []
        allocated_lens = []
        topup_total = 0
        total_tokens = 0
        for task in tasks:
            state = self._persistent_requests.get(task.draft_kv_handle)
            if state is None:
                raise ValueError(
                    "DFLASH reconcile KV handle does not exist: "
                    f"handle={task.draft_kv_handle!r} rid={task.request_id!r} "
                    f"round={task.round_id} transfer={task.transfer_id!r}"
                )
            if (
                state.request_id != task.request_id
                or state.target_instance_id != task.target_instance_id
            ):
                raise ValueError("DFLASH reconcile identity does not match KV owner")
            if task.round_id != state.last_round_id + 1:
                raise ValueError("DFLASH reconcile round is not the next request round")
            if task.feature_token_offset != state.valid_committed_len:
                raise ValueError("DFLASH reconcile has a committed-prefix gap")
            if state.proposal_base_len != state.valid_committed_len:
                raise RuntimeError(
                    "DFLASH proposal base diverged from committed boundary"
                )
            req = state.req
            if req.kv is None or req.req_pool_idx is None:
                raise RuntimeError("DFLASH reconcile lost its persistent KV handle")
            allocated_len = req.kv.kv_allocated_len
            if allocated_len < state.valid_committed_len:
                raise RuntimeError(
                    "DFLASH allocated KV is shorter than its committed prefix"
                )
            token_count = int(task.feature_token_count)
            if (
                token_count <= 0
                or task.committed_kv_len - task.feature_token_offset != token_count
            ):
                raise ValueError("DFLASH reconcile token range mismatch")
            states.append(state)
            req_pool_indices.append(req.req_pool_idx)
            feature_offsets.append(task.feature_token_offset)
            committed_lens.append(task.committed_kv_len)
            bonus_tokens.append(task.next_token_id)
            allocated_lens.append(allocated_len)
            topup_total += max(0, task.committed_kv_len - allocated_len)
            total_tokens += token_count
        if reconcile_prepare_phase_ns is not None:
            phase_completed_ns = time.monotonic_ns()
            reconcile_prepare_phase_ns[0] += phase_completed_ns - phase_started_ns
            phase_started_ns = phase_completed_ns
        allocator = self._draft_model_runner.token_to_kv_pool_allocator
        topup_pages = None
        if topup_total:
            topup_pages = allocator.alloc(topup_total)
            if topup_pages is None:
                raise RuntimeError("DFLASH reconcile KV-page OOM")
        if reconcile_prepare_phase_ns is not None:
            phase_completed_ns = time.monotonic_ns()
            reconcile_prepare_phase_ns[2] += phase_completed_ns - phase_started_ns
            phase_started_ns = phase_completed_ns
        row_count = len(tasks)
        try:
            metadata_host[:, :row_count].copy_(
                torch.tensor(
                    [
                        req_pool_indices,
                        feature_offsets,
                        committed_lens,
                        bonus_tokens,
                        allocated_lens,
                    ],
                    dtype=torch.int64,
                )
            )
            h2d_started_ns = (
                time.monotonic_ns() if submit_sub_ns is not None else 0
            )
            # Upload the whole packed block in one H2D; only the first
            # row_count columns are read below, and the slot's ownership
            # retains the buffers until this batch completes.
            metadata_device.copy_(metadata_host, non_blocking=True)
            if submit_sub_ns is not None:
                submit_sub_ns[2] += time.monotonic_ns() - h2d_started_ns
            if topup_pages is not None:
                committed = metadata_device[_DFLASH_META_COMMITTED_LEN][:row_count]
                self._install_dflash_row_pages(
                    metadata_device[_DFLASH_META_REQ_POOL_IDX][:row_count],
                    torch.minimum(
                        metadata_device[_DFLASH_META_ALLOCATED_LEN][:row_count],
                        committed,
                    ),
                    committed,
                    topup_pages,
                    row_count,
                )
        except Exception:
            if topup_pages is not None:
                allocator.free(topup_pages)
            raise
        if reconcile_prepare_phase_ns is not None:
            phase_completed_ns = time.monotonic_ns()
            reconcile_prepare_phase_ns[3] += phase_completed_ns - phase_started_ns
            phase_started_ns = phase_completed_ns
        # The top-up write and any later cleanup use the same CUDA stream.
        # Publish all CPU ownership boundaries only after the write is queued;
        # from here, outer failure cleanup owns every topped-up page.
        for state, task, allocated_len in zip(states, tasks, allocated_lens):
            state.req.kv.kv_allocated_len = max(allocated_len, task.committed_kv_len)
            state.req.kv_committed_len = task.committed_kv_len
            discarded = max(0, allocated_len - task.committed_kv_len)
            state.valid_committed_len = task.committed_kv_len
            state.proposal_base_len = task.committed_kv_len
            state.last_round_id = task.round_id
            state.committed_token_ids.extend(task.committed_token_ids)
            profile_emit(
                "draft_reconciliation",
                draft_instance=self.instance_id,
                request_id=state.request_id,
                round_id=task.round_id,
                discarded_tokens=discarded,
                materialized_tokens=int(task.feature_token_count),
                committed_kv_len=task.committed_kv_len,
            )
        if reconcile_prepare_phase_ns is not None:
            reconcile_prepare_phase_ns[5] += time.monotonic_ns() - phase_started_ns
        grouped_batches = (
            self.__dict__.get("_dflash_grouped_reconcile_batches", 0) + 1
        )
        grouped_rows = self.__dict__.get("_dflash_grouped_reconcile_rows", 0) + len(
            tasks
        )
        self._dflash_grouped_reconcile_batches = grouped_batches
        self._dflash_grouped_reconcile_rows = grouped_rows
        if grouped_batches & (grouped_batches - 1) == 0:
            logger.info(
                "DT DFLASH grouped reconcile observed: batches=%d rows=%d",
                grouped_batches,
                grouped_rows,
            )
        metadata_fields = tuple(
            metadata_device[field][:row_count] for field in range(_DFLASH_META_FIELDS)
        )
        return states, metadata_fields, total_tokens

    def _install_dflash_row_pages(
        self,
        req_pool_indices: torch.Tensor,
        start_offsets: torch.Tensor,
        end_offsets: torch.Tensor,
        pages: torch.Tensor,
        batch_size: int,
    ) -> None:
        """Write packed KV pages into per-row [start, end) page-table ranges."""

        req_to_token = self._draft_model_runner.req_to_token_pool.req_to_token
        if self._canonical_draft_device(self.device).type == "cuda":
            from sglang.srt.mem_cache.allocation import (
                assign_req_to_token_pool_func,
            )

            assign_req_to_token_pool_func(
                req_pool_indices,
                req_to_token,
                start_offsets,
                end_offsets,
                pages,
                batch_size,
            )
            return
        offset = 0
        for row, start, end in zip(
            req_pool_indices.tolist(),
            start_offsets.tolist(),
            end_offsets.tolist(),
            strict=True,
        ):
            length = end - start
            if length > 0:
                req_to_token[row, start:end] = pages[offset : offset + length].to(
                    req_to_token.dtype
                )
            offset += length

    def _prepare_dflash_task(
        self,
        task,
        reconcile_prepare_phase_ns: list[int] | None = None,
    ) -> tuple[PersistentDraftRequest, torch.Tensor, torch.Tensor]:
        """Prepare one DFlash row outside the grouped reconcile wave.

        Handles bootstrap rows and reconcile rows in mixed batches.  Returns
        (state, cache_loc, positions) for the materializer.  Reconcile rows
        follow the same pages-stay-put ownership as the grouped wave: no
        free, and a page-table write only for a genuine shortfall.
        """

        from dt_protocol import DraftTaskKind

        req_pool = self._draft_model_runner.req_to_token_pool
        allocator = self._draft_model_runner.token_to_kv_pool_allocator
        token_count = int(task.feature_token_count)
        if task.kind == DraftTaskKind.BOOTSTRAP:
            if task.draft_kv_handle in self._persistent_requests:
                raise ValueError("DFLASH bootstrap KV handle already exists")
            req = Req(
                rid=task.request_id,
                origin_input_text="",
                origin_input_ids=array("q", task.committed_token_ids),
                sampling_params=SamplingParams(
                    temperature=0, max_new_tokens=self.profile.draft_k
                ),
            )
            req.full_untruncated_fill_ids = req.origin_input_ids
            req.logprob_start_len = -1
            req.set_extend_range(0, task.committed_kv_len)
            if req_pool.alloc([req]) is None:
                raise RuntimeError("DFLASH request-row OOM")
            cache_loc = allocator.alloc(token_count)
            if cache_loc is None:
                req_pool.free(req)
                raise RuntimeError("DFLASH KV-page OOM")
            req.kv = ReqKvInfo(
                kv_allocated_len=task.committed_kv_len,
                swa_evicted_seqlen=0,
            )
            req.kv_committed_len = task.committed_kv_len
            req_pool.req_to_token[req.req_pool_idx, : task.committed_kv_len] = (
                cache_loc
            )
            state = PersistentDraftRequest(
                req=req,
                target_instance_id=task.target_instance_id,
                request_id=task.request_id,
                draft_kv_handle=task.draft_kv_handle,
                valid_committed_len=task.committed_kv_len,
                last_round_id=task.round_id,
                proposal_base_len=task.committed_kv_len,
                committed_token_ids=list(task.committed_token_ids),
            )
            ledger = getattr(self, "_rid_ledger", None)
            if ledger is not None:
                # §7.1 ring records reference requests by rid hash.
                ledger.register(task.request_id)
            self._persistent_requests[task.draft_kv_handle] = state
            self._persistent_index_add(state)
            positions = torch.arange(
                task.committed_kv_len, dtype=torch.int64, device=self.device
            )
            return state, cache_loc.to(torch.int64), positions

        phase_started_ns = (
            time.monotonic_ns() if reconcile_prepare_phase_ns is not None else 0
        )
        state = self._persistent_requests.get(task.draft_kv_handle)
        if state is None:
            raise ValueError("DFLASH reconcile KV handle does not exist")
        if (
            state.request_id != task.request_id
            or state.target_instance_id != task.target_instance_id
        ):
            raise ValueError("DFLASH reconcile identity does not match KV owner")
        if task.round_id != state.last_round_id + 1:
            raise ValueError("DFLASH reconcile round is not the next request round")
        if task.feature_token_offset != state.valid_committed_len:
            raise ValueError("DFLASH reconcile has a committed-prefix gap")
        if state.proposal_base_len != state.valid_committed_len:
            raise RuntimeError("DFLASH proposal base diverged from committed boundary")
        req = state.req
        if req.kv is None or req.req_pool_idx is None:
            raise RuntimeError("DFLASH reconcile lost its persistent KV handle")
        allocated_len = req.kv.kv_allocated_len
        if allocated_len < state.valid_committed_len:
            raise RuntimeError(
                "DFLASH allocated KV is shorter than its committed prefix"
            )
        if (
            token_count <= 0
            or task.committed_kv_len - task.feature_token_offset != token_count
        ):
            raise ValueError("DFLASH reconcile token range mismatch")
        if reconcile_prepare_phase_ns is not None:
            phase_completed_ns = time.monotonic_ns()
            reconcile_prepare_phase_ns[0] += phase_completed_ns - phase_started_ns
            phase_started_ns = phase_completed_ns
        topup = task.committed_kv_len - allocated_len
        if topup > 0:
            topup_pages = allocator.alloc(topup)
            if topup_pages is None:
                raise RuntimeError("DFLASH reconcile KV-page OOM")
            try:
                req_pool.req_to_token[
                    req.req_pool_idx, allocated_len : task.committed_kv_len
                ] = topup_pages.to(req_pool.req_to_token.dtype)
            except Exception:
                allocator.free(topup_pages)
                raise
            req.kv.kv_allocated_len = task.committed_kv_len
        if reconcile_prepare_phase_ns is not None:
            phase_completed_ns = time.monotonic_ns()
            reconcile_prepare_phase_ns[2] += phase_completed_ns - phase_started_ns
            phase_started_ns = phase_completed_ns
        req.kv_committed_len = task.committed_kv_len
        discarded_tokens = max(0, allocated_len - task.committed_kv_len)
        state.valid_committed_len = task.committed_kv_len
        state.proposal_base_len = task.committed_kv_len
        state.last_round_id = task.round_id
        state.committed_token_ids.extend(task.committed_token_ids)
        profile_emit(
            "draft_reconciliation",
            draft_instance=self.instance_id,
            request_id=state.request_id,
            round_id=task.round_id,
            discarded_tokens=discarded_tokens,
            materialized_tokens=token_count,
            committed_kv_len=task.committed_kv_len,
        )
        if reconcile_prepare_phase_ns is not None:
            reconcile_prepare_phase_ns[5] += time.monotonic_ns() - phase_started_ns
        cache_loc = req_pool.req_to_token[
            req.req_pool_idx, task.feature_token_offset : task.committed_kv_len
        ].to(torch.int64)
        positions = torch.arange(
            task.feature_token_offset,
            task.committed_kv_len,
            dtype=torch.int64,
            device=self.device,
        )
        return state, cache_loc, positions

    def _run_dflash_proposal(
        self,
        *,
        states: list[PersistentDraftRequest],
        req_pool_indices: torch.Tensor,
        prefix_lens: torch.Tensor,
        alloc_starts: torch.Tensor,
        bonus_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, str, str | None]:
        batch_size = len(states)
        block_size = int(self.profile.block_size)
        req_pool = self._draft_model_runner.req_to_token_pool
        allocator = self._draft_model_runner.token_to_kv_pool_allocator
        marks = self._submit_marks if getattr(self, "_profile_active", False) else None
        # Each row's block head still maps the previous round's lookahead
        # pages, which this forward overwrites; only the tail beyond the
        # current allocation takes fresh pages, in one batched allocation.
        tail_total = 0
        for state in states:
            tail = (
                state.valid_committed_len
                + block_size
                - state.req.kv.kv_allocated_len
            )
            if tail < 0:
                raise RuntimeError("DFLASH proposal allocation exceeds its block")
            tail_total += tail
        # Committed lengths are already authoritative CPU state, so
        # seq_lens_cpu never needs a GPU round-trip.
        seq_lens_cpu = torch.tensor(
            [state.valid_committed_len + block_size for state in states],
            dtype=torch.int32,
            device="cpu",
        )
        tail_pages = allocator.alloc(tail_total) if tail_total else None
        if tail_total and tail_pages is None:
            raise RuntimeError("DFLASH proposal KV-page OOM")
        pages_installed = False
        try:
            if tail_pages is not None:
                self._install_dflash_row_pages(
                    req_pool_indices,
                    alloc_starts,
                    prefix_lens + block_size,
                    tail_pages,
                    batch_size,
                )
            for state in states:
                end = state.valid_committed_len + block_size
                state.req.kv.kv_allocated_len = end
                state.req.kv_committed_len = state.valid_committed_len
                state.proposal_base_len = state.valid_committed_len
            pages_installed = True
            if marks is not None:
                marks["proposal_kv"] = time.monotonic_ns()

            if self._canonical_draft_device(self.device).type == "cuda":
                from sglang.kernels.ops.speculative.dflash import (
                    _prepare_dflash_draft_block_unchecked,
                )

                block_ids = torch.empty(
                    (batch_size, block_size), dtype=torch.int64, device=self.device
                )
                positions = torch.empty(
                    (batch_size, block_size), dtype=torch.int64, device=self.device
                )
                block_cache_locs = torch.empty(
                    (batch_size, block_size), dtype=torch.int64, device=self.device
                )
                _prepare_dflash_draft_block_unchecked(
                    bonus_tokens=bonus_tokens,
                    prefix_lens=prefix_lens,
                    req_pool_indices=req_pool_indices,
                    req_to_token=req_pool.req_to_token,
                    block_ids_out=block_ids,
                    positions_out=positions,
                    cache_loc_out=block_cache_locs,
                    mask_token_id=self._mask_token_id,
                )
            else:
                block_ids = torch.full(
                    (batch_size, block_size),
                    self._mask_token_id,
                    dtype=torch.int64,
                    device=self.device,
                )
                block_ids[:, 0].copy_(bonus_tokens)
                positions = prefix_lens.unsqueeze(1) + self._block_pos_offsets
                block_cache_locs = req_pool.req_to_token[
                    req_pool_indices.unsqueeze(1).expand_as(positions), positions
                ].to(torch.int64)
            out_cache_loc = block_cache_locs.reshape(-1)
            input_embeds = self._token_embedding(block_ids).reshape(
                batch_size * block_size, -1
            )
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.TARGET_VERIFY,
                batch_size=batch_size,
                input_ids=block_ids.reshape(-1),
                req_pool_indices=req_pool_indices,
                seq_lens=prefix_lens,
                out_cache_loc=out_cache_loc,
                seq_lens_sum=int(seq_lens_cpu.sum()),
                seq_lens_cpu=seq_lens_cpu,
                positions=positions.reshape(-1),
                input_embeds=input_embeds,
                spec_algorithm=SpeculativeAlgorithm.DFLASH,
                spec_info=self._draft_block_spec_info,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                device_timer_category="draft_proposal",
                rids=[state.request_id for state in states],
            )
            if marks is not None:
                marks["proposal_inputs"] = time.monotonic_ns()
            draft_out = self._draft_model_runner.forward(forward_batch)
            proposal_execution = "graph" if draft_out.can_run_graph else "eager"
            proposal_fallback_reason = (
                None
                if draft_out.can_run_graph
                else self._draft_model_runner._cuda_graph_fallback_reason(forward_batch)
            )
            if not draft_out.can_run_graph:
                hidden_states = draft_out.logits_output.hidden_states
                if hidden_states is None:
                    raise RuntimeError("DFLASH Draft forward returned no hidden states")
                self._dflash_sampler(hidden_states)
            if marks is not None:
                marks["proposal_forward"] = time.monotonic_ns()
            proposals = self._dflash_sampler.out[: batch_size * (block_size - 1)].view(
                batch_size, block_size - 1
            )
            measurement_rows = sum(
                state.request_id.startswith(DT_BENCHMARK_MEASUREMENT_RID_PREFIX)
                for state in states
            )
            if measurement_rows:
                self._record_graph_measurement(
                    path="dflash_proposal",
                    execution=proposal_execution,
                    measurement_rows=measurement_rows,
                    measurement_batch_rows=measurement_rows,
                    batch_size=batch_size,
                    model_runner_event_id=(
                        self._draft_model_runner._last_cuda_graph_measurement_event_id
                    ),
                    graph_replays=int(draft_out.can_run_graph),
                    expected_graph_replays=1,
                    fallback_reason=proposal_fallback_reason,
                )
            candidates = torch.empty(
                (batch_size, block_size), dtype=torch.int32, device=self.device
            )
            candidates[:, 0].copy_(bonus_tokens)
            candidates[:, 1:].copy_(proposals)
            return candidates, proposal_execution, proposal_fallback_reason
        except Exception as exc:
            # Fence exceptional GPU work before any touched page can be
            # reused; if that fence itself fails, report the resource as
            # unquiesced so the caller quarantines every affected request
            # instead.
            try:
                torch.cuda.current_stream(self.device).synchronize()
            except Exception as sync_exc:
                # Propagate the failed recovery fence, not the original
                # proposal error.  Both exceptions carry the ownership marker
                # so every caller in the chain must retain the touched Draft-KV
                # pages and staging range.  Chaining preserves the proposal
                # failure that required this otherwise-fallible fence.
                setattr(exc, "resources_quiescent", False)
                setattr(sync_exc, "resources_quiescent", False)
                raise sync_exc from exc
            if not pages_installed and tail_pages is not None:
                # The tail pages never entered the page table; return them
                # directly.  Once installed, the rows own them and outer
                # failure cleanup frees [0, kv_allocated_len) as usual.
                allocator.free(tail_pages)
            raise

    def _profile_dflash_batch(self, completion, tasks, gpu_ns: int, estimate) -> None:
        """D0 per-batch stamps (issue #269): absolute GPU end, the G(n)/C(n)
        estimate against the measured values."""

        gpu_end_clock_ns = 0
        if completion.clock_done is not None and completion.clock_done.query():
            gpu_end_clock_ns = self._gpu_clock.read(completion.clock_slot)
        profile_emit(
            "draft_batch_estimate",
            draft_instance=self.instance_id,
            transfer_ids=tuple(task.transfer_id for task in tasks),
            rows=len(tasks),
            tokens=sum(task.feature_token_count for task in tasks),
            proposal_rows=len(completion.proposal_tasks),
            g_ns=gpu_ns,
            c_ns=(
                completion.handoff_ns - completion.select_started_ns
                if completion.handoff_ns
                else None
            ),
            select_started_ns=completion.select_started_ns,
            frozen_ns=completion.frozen_ns,
            submit_started_ns=completion.launch_started_ns,
            launched_ns=completion.launched_ns,
            handoff_ns=completion.handoff_ns,
            gpu_end_clock_ns=gpu_end_clock_ns,
            # D8: joins this batch to its draft_dflash_done_absorb record, so
            # the per-hop table can span GPU clock end -> scheduler -1.
            done_seq=completion.batch_seq + 1,
            # #306: the stage split of C as [stage, end ns] in execution
            # order (C ends at submit_tail), its nested parts, and the
            # reconcile phases.
            submit_marks=(
                None
                if completion.submit_marks is None
                else list(completion.submit_marks.items())
            ),
            submit_sub_ns=(
                None
                if completion.submit_sub_ns is None
                else dict(zip(SUBMIT_SUB_STAGES, completion.submit_sub_ns))
            ),
            reconcile_phase_ns=completion.reconcile_prepare_phase_ns,
            **estimate,
        )

    def _poll_dflash_completions(self) -> None:
        if self._fatal_gpu_quarantine_active():
            return
        capacity_changed = False
        while self._pending_dflash_completions:
            stage_started_ns = time.monotonic_ns()
            completion = self._pending_dflash_completions[0]
            try:
                ready = completion.completion_event.query()
            except Exception as exc:
                # Leave the completion in place: it owns the staging slot and
                # every touched Draft-KV row until CUDA context teardown.  A
                # failed event query cannot prove that proposal/materialization
                # writes have stopped, so neither shutdown nor another poll may
                # touch this completion again.
                tasks = [item.task for item in completion.pending]
                self._quarantined_feature_draft_handles.update(
                    task.draft_kv_handle for task in tasks
                )
                exc.resources_quiescent = False
                logger.critical(
                    "quarantined DFlash completion, staging, and KV after "
                    "completion-event query failure: transfers=%s",
                    [task.transfer_id for task in tasks],
                    exc_info=True,
                )
                self._enter_fatal_gpu_quarantine(
                    "unquiesced DFlash completion-event failure"
                )
                # As above, reporting is best-effort and may itself raise; the
                # fatal latch and retained ownership must not depend on it.
                self._publish_failures([(task, exc) for task in tasks])
                self._publish_capacity()
                break
            if not ready:
                break
            self._pending_dflash_completions.popleft()
            completed_ns = time.monotonic_ns()
            gpu_materialization_ns = int(
                completion.materialize_start_event.elapsed_time(
                    completion.materialize_done_event
                )
                * 1_000_000
            )
            proposal_gpu_ns = int(
                completion.materialize_done_event.elapsed_time(
                    completion.proposal_done_event
                )
                * 1_000_000
            )
            self._retire_dflash_completion(
                completion,
                completed_ns=completed_ns,
                gpu_materialization_ns=gpu_materialization_ns,
                proposal_gpu_ns=proposal_gpu_ns,
                stage_started_ns=stage_started_ns,
            )
            capacity_changed = True
        if capacity_changed:
            self._publish_capacity()

    def _retire_dflash_completion(
        self,
        completion,
        *,
        completed_ns: int,
        gpu_materialization_ns: int,
        proposal_gpu_ns: int,
        stage_started_ns: int,
    ) -> None:
        """Retire one DFlash batch whose GPU work is done.

        The Python-state bookkeeping of design §5.1 ABSORB, shared by the
        legacy completion-event poll and the D8 send loop, which observes
        the GPU done natively and passes its stamp and CUDA elapsed in.
        """

        tasks = [item.task for item in completion.pending]
        # A push queued at launch already published these candidates.
        early = self._take_early_pushed(tasks)
        expired_tasks = [
            task
            for task in tasks
            if task.transfer_id not in self._cancelled_ids
            and task.transfer_id not in early
            and completed_ns >= task.deadline_ns
        ]
        expired_ids = {task.transfer_id for task in expired_tasks}
        active_tasks = [
            task
            for task in tasks
            if task.transfer_id not in self._cancelled_ids
            and task.transfer_id not in expired_ids
        ]
        cancelled_tasks = [
            task for task in tasks if task.transfer_id in self._cancelled_ids
        ]
        materialize_only = [
            task for task in active_tasks if not task.requires_proposal
        ]
        active_proposal_indices = [
            index
            for index, task in enumerate(completion.proposal_tasks)
            if task.transfer_id not in self._cancelled_ids
            and task.transfer_id not in expired_ids
            and task.transfer_id not in early
        ]
        # The step's GPU service time spans both timed phases: candidate
        # feature materialization (materialize_start -> materialize_done)
        # and proposal execution (materialize_done -> proposal_done).
        self._last_step_ns = gpu_materialization_ns + proposal_gpu_ns
        stage_query_ns = time.monotonic_ns()
        self._loop_stage_ns["dflash_query"] += stage_query_ns - stage_started_ns
        self._loop_stage_calls["dflash_query"] += 1
        estimate = {}
        if self._batch_estimator is not None and completion.handoff_ns:
            estimate = self._batch_estimator.observe(
                rows=len(tasks),
                tokens=sum(task.feature_token_count for task in tasks),
                gpu_ns=gpu_materialization_ns + proposal_gpu_ns,
                cpu_ns=completion.handoff_ns - completion.select_started_ns,
            )
        if self._profile_active:
            self._profile_dflash_batch(
                completion, tasks, gpu_materialization_ns + proposal_gpu_ns, estimate
            )
        profile_emit(
            "draft_dflash_batch_complete",
            draft_instance=self.instance_id,
            transfer_ids=tuple(task.transfer_id for task in tasks),
            execute_started_ns=completion.launch_started_ns,
            tasks_prepared_ns=completion.tasks_prepared_ns,
            metadata_ready_ns=completion.proposal_metadata_ready_ns,
            materialize_submitted_ns=completion.materialize_submitted_ns,
            batch_size=len(tasks),
            active_task_count=len(active_tasks),
            cancelled_task_count=len(cancelled_tasks),
            expired_task_count=len(expired_tasks),
            proposal_count=len(completion.proposal_tasks),
            active_proposal_count=len(active_proposal_indices),
            materialize_only_count=len(materialize_only),
            bootstrap_count=sum(task.round_id == 0 for task in tasks),
            reconcile_count=sum(task.round_id > 0 for task in tasks),
            materialized_tokens=sum(task.feature_token_count for task in tasks),
            submitted_ns=completion.launched_ns,
            gpu_completed_ns=completed_ns,
            host_task_prepare_ns=(
                completion.tasks_prepared_ns - completion.launch_started_ns
            ),
            host_proposal_metadata_ns=(
                completion.proposal_metadata_ready_ns - completion.tasks_prepared_ns
            ),
            host_materialize_submit_ns=(
                completion.materialize_submitted_ns
                - completion.proposal_metadata_ready_ns
            ),
            host_proposal_submit_ns=(
                completion.launched_ns - completion.materialize_submitted_ns
            ),
            host_total_submit_ns=(
                completion.launched_ns - completion.launch_started_ns
            ),
            gpu_materialization_ns=gpu_materialization_ns,
            proposal_gpu_ns=proposal_gpu_ns,
            proposal_execution=completion.proposal_execution,
            proposal_fallback_reason=completion.proposal_fallback_reason,
            prior_unfinished_proposal_count=(
                completion.prior_unfinished_proposal_count
            ),
            prior_candidate_copy_pending_count=(
                completion.prior_candidate_copy_pending_count
            ),
            host_reconcile_validate_ns=(
                completion.reconcile_prepare_phase_ns[0]
                if completion.reconcile_prepare_phase_ns
                else 0
            ),
            host_reconcile_free_ns=(
                completion.reconcile_prepare_phase_ns[1]
                if completion.reconcile_prepare_phase_ns
                else 0
            ),
            host_reconcile_alloc_ns=(
                completion.reconcile_prepare_phase_ns[2]
                if completion.reconcile_prepare_phase_ns
                else 0
            ),
            host_reconcile_row_write_ns=(
                completion.reconcile_prepare_phase_ns[3]
                if completion.reconcile_prepare_phase_ns
                else 0
            ),
            host_reconcile_prefix_refresh_ns=(
                completion.reconcile_prepare_phase_ns[4]
                if completion.reconcile_prepare_phase_ns
                else 0
            ),
            host_reconcile_state_refresh_ns=(
                completion.reconcile_prepare_phase_ns[5]
                if completion.reconcile_prepare_phase_ns
                else 0
            ),
        )
        if materialize_only:
            self._publish_materialization_successes(materialize_only)
        if active_proposal_indices:
            push_scope = (
                self._direct_push_scope(
                    completion.completion_event, completion.candidates
                )
                if self.candidate_transport == "cuda-ipc"
                and completion.candidates is not None
                else contextlib.nullcontext()
            )
            with push_scope:
                if self.candidate_transport == "cuda-ipc":
                    if completion.candidates is None:
                        raise RuntimeError("DFLASH GPU candidates are unavailable")
                    candidate_rows = _select_candidate_rows(
                        completion.candidates, active_proposal_indices
                    )
                    candidate_payload = candidate_rows
                elif completion.candidate_rows_cpu is not None:
                    candidate_rows = completion.candidate_rows_cpu[
                        active_proposal_indices
                    ]
                    candidate_payload = candidate_rows.contiguous().numpy().tobytes()
                else:
                    candidate_rows = completion.candidates[
                        active_proposal_indices
                    ].cpu()
                    candidate_payload = candidate_rows.contiguous().numpy().tobytes()
                self._publish_successes(
                    [
                        completion.proposal_tasks[index]
                        for index in active_proposal_indices
                    ],
                    candidate_payload,
                )
        stage_publish_ns = time.monotonic_ns()
        self._loop_stage_ns["dflash_publish"] += stage_publish_ns - stage_query_ns
        for task in active_tasks:
            profile_emit(
                "draft_kv_materialization",
                draft_instance=self.instance_id,
                transfer_id=task.transfer_id,
                request_id=task.request_id,
                round_id=task.round_id,
                committed_kv_len=task.committed_kv_len,
                materialized_tokens=task.feature_token_count,
                gpu_materialization_ns=gpu_materialization_ns,
                proposal_gpu_ns=proposal_gpu_ns,
                completed_ns=time.monotonic_ns(),
            )
            if task.requires_proposal:
                profile_emit(
                    "draft_proposal",
                    draft_instance=self.instance_id,
                    transfer_id=task.transfer_id,
                    request_id=task.request_id,
                    round_id=task.round_id,
                    submitted_ns=completion.launched_ns,
                    gpu_completed_ns=completed_ns,
                    proposal_gpu_ns=proposal_gpu_ns,
                )
        for task in cancelled_tasks:
            self._retire_cancelled_transfer_identity(task)
            self._release_persistent_request(task.draft_kv_handle)
        for task in expired_tasks:
            self._release_persistent_request(task.draft_kv_handle)
        if expired_tasks:
            self._publish_failures(
                [
                    (
                        task,
                        TimeoutError(
                            "DFLASH materialization completed after its deadline"
                        ),
                    )
                    for task in expired_tasks
                ]
            )
        self._complete_feature_compute(
            completion.staging_slots, completion.staging_slot_counts
        )
        if completion.ready_rows:
            self._release_ready_rows(completion.ready_rows)
        for handle in {task.draft_kv_handle for task in tasks}:
            self._drain_deferred_persistent_release(handle)
        self._loop_stage_ns["dflash_retire"] += (
            time.monotonic_ns() - stage_publish_ns
        )

    def _execute_standalone_batch(self, pending: list[PendingDraftTask]) -> None:
        batch_start_ns = time.monotonic_ns()
        sync_profile = profile_enabled() and cuda_sync_enabled()
        # _take_batch already fixed Target-major order; all AR steps use it.
        tasks = [item.task for item in pending]
        states: list[PersistentDraftRequest] = []
        prepare_done_ns = batch_start_ns
        extend_done_ns = batch_start_ns
        decode_done_ns: list[int] = []
        extend_cuda_graph = False
        extend_graph_replays = 0
        expected_extend_graph_replays = 1
        decode_graph_replays = 0
        result_slot = None
        admission_credit = False
        try:
            if getattr(self, "continuous_batching", False):
                if self.candidate_transport == "mmap":
                    # The mmap completion path indexes its pinned host result
                    # buffer by this slot; the direct path carries no pooled id.
                    result_slot = self._free_result_slots.popleft()
                else:
                    # The direct path holds a proposal-admission credit for the
                    # whole submit→publish→push lifetime instead (issue #42).
                    self._acquire_direct_proposal_admission()
                    admission_credit = True
            states = [self._prepare_persistent_task(task) for task in tasks]
            oversized_bootstrap = (
                len(tasks) == 1
                and tasks[0].round_id == 0
                and len(tasks[0].committed_token_ids)
                > DRAFT_PREFILL_GRAPH_MAX_TOKENS
            )
            prepare_done_ns = time.monotonic_ns()
            if oversized_bootstrap:
                task = tasks[0]
                state = states[0]
                if len(task.committed_token_ids) != task.committed_kv_len:
                    raise RuntimeError(
                        "STANDALONE bootstrap committed tokens do not cover its KV"
                    )
                expected_extend_graph_replays = (
                    len(task.committed_token_ids)
                    + DRAFT_PREFILL_GRAPH_MAX_TOKENS
                    - 1
                ) // DRAFT_PREFILL_GRAPH_MAX_TOKENS
                for chunk_start in range(
                    0,
                    len(task.committed_token_ids),
                    DRAFT_PREFILL_GRAPH_MAX_TOKENS,
                ):
                    chunk_end = min(
                        chunk_start + DRAFT_PREFILL_GRAPH_MAX_TOKENS,
                        len(task.committed_token_ids),
                    )
                    if chunk_start:
                        req = state.req
                        req.kv_committed_len = chunk_start
                        req.prefix_indices = (
                            self._draft_model_runner.req_to_token_pool.req_to_token[
                                req.req_pool_idx, :chunk_start
                            ].to(torch.int64)
                        )
                    state.req.set_extend_range(chunk_start, chunk_end)
                    batch = ScheduleBatch.init_new(
                        reqs=[state.req],
                        req_to_token_pool=self._draft_model_runner.req_to_token_pool,
                        token_to_kv_pool_allocator=(
                            self._draft_model_runner.token_to_kv_pool_allocator
                        ),
                        tree_cache=self._tree_cache,
                        model_config=self._draft_model_runner.model_config,
                        enable_overlap=False,
                        spec_algorithm=SpeculativeAlgorithm.NONE,
                    )
                    batch.prepare_for_extend()
                    self._resolve_prefill_inputs(batch)
                    extend_result = self._draft_worker.forward_batch_generation(batch)
                    next_tokens = extend_result.next_token_ids
                    extend_graph_replays += int(
                        extend_result.can_run_cuda_graph
                    )
                extend_cuda_graph = (
                    extend_graph_replays == expected_extend_graph_replays
                )
                logger.info(
                    "DT STANDALONE chunked bootstrap: request=%s tokens=%d "
                    "chunks=%d graph_replays=%d",
                    task.request_id,
                    len(task.committed_token_ids),
                    expected_extend_graph_replays,
                    extend_graph_replays,
                )
            else:
                batch = ScheduleBatch.init_new(
                    reqs=[state.req for state in states],
                    req_to_token_pool=self._draft_model_runner.req_to_token_pool,
                    token_to_kv_pool_allocator=(
                        self._draft_model_runner.token_to_kv_pool_allocator
                    ),
                    tree_cache=self._tree_cache,
                    model_config=self._draft_model_runner.model_config,
                    enable_overlap=False,
                    spec_algorithm=SpeculativeAlgorithm.NONE,
                )
                batch.prepare_for_extend()
                self._resolve_prefill_inputs(batch)
                extend_result = self._draft_worker.forward_batch_generation(batch)
                next_tokens = extend_result.next_token_ids
                extend_graph_replays = int(extend_result.can_run_cuda_graph)
                extend_cuda_graph = bool(extend_result.can_run_cuda_graph)
            extend_measurement_event_id = (
                self._draft_model_runner._last_cuda_graph_measurement_event_id
            )
            if sync_profile:
                torch.get_device_module(self.device).synchronize()
            extend_done_ns = time.monotonic_ns()
            proposals = [next_tokens]
            proposal_measurement_event_id = None
            for _ in range(1, self.profile.draft_k):
                batch.input_ids = next_tokens.to(torch.int64)
                batch.prepare_for_decode()
                decode_result = self._draft_worker.forward_batch_generation(batch)
                next_tokens = decode_result.next_token_ids
                decode_graph_replays += int(decode_result.can_run_cuda_graph)
                proposal_measurement_event_id = (
                    self._draft_model_runner._last_cuda_graph_measurement_event_id
                )
                if sync_profile:
                    torch.get_device_module(self.device).synchronize()
                decode_done_ns.append(time.monotonic_ns())
                proposals.append(next_tokens)

            if (
                decode_graph_replays == max(self.profile.draft_k - 1, 0)
                and not self.__dict__.get(
                    "_standalone_graph_replay_evidence_emitted", False
                )
            ):
                self._standalone_graph_replay_evidence_emitted = True
                logger.info(
                    "DT STANDALONE Draft CUDA graph replay observed: "
                    "mode=PROPOSAL_DECODE batch_size=%d replays=%d",
                    len(tasks),
                    decode_graph_replays,
                )

            measurement_tasks = [
                task
                for task in tasks
                if task.request_id.startswith(DT_BENCHMARK_MEASUREMENT_RID_PREFIX)
            ]
            measurement_reconcile = sum(task.round_id > 0 for task in measurement_tasks)
            measurement_bootstrap = len(measurement_tasks) - measurement_reconcile
            if extend_graph_replays == expected_extend_graph_replays:
                extend_execution = "graph"
            elif extend_graph_replays == 0:
                extend_execution = "eager"
            else:
                extend_execution = "mixed"
            if measurement_bootstrap:
                self._record_graph_measurement(
                    path="bootstrap_prefill",
                    execution=extend_execution,
                    measurement_rows=measurement_bootstrap,
                    measurement_batch_rows=len(measurement_tasks),
                    batch_size=len(tasks),
                    model_runner_event_id=extend_measurement_event_id,
                    graph_replays=extend_graph_replays,
                    expected_graph_replays=expected_extend_graph_replays,
                )
            if measurement_reconcile:
                self._record_graph_measurement(
                    path="reconcile_prefill",
                    execution=extend_execution,
                    measurement_rows=measurement_reconcile,
                    measurement_batch_rows=len(measurement_tasks),
                    batch_size=len(tasks),
                    model_runner_event_id=extend_measurement_event_id,
                    graph_replays=extend_graph_replays,
                    expected_graph_replays=expected_extend_graph_replays,
                )
            if measurement_tasks:
                expected_decode_replays = max(self.profile.draft_k - 1, 0)
                if decode_graph_replays == expected_decode_replays:
                    decode_execution = "graph"
                elif decode_graph_replays == 0:
                    decode_execution = "eager"
                else:
                    decode_execution = "mixed"
                self._record_graph_measurement(
                    path="proposal_decode",
                    execution=decode_execution,
                    measurement_rows=len(measurement_tasks),
                    measurement_batch_rows=len(measurement_tasks),
                    batch_size=len(tasks),
                    model_runner_event_id=proposal_measurement_event_id,
                    graph_replays=decode_graph_replays,
                    expected_graph_replays=expected_decode_replays,
                )

            # Root IDs are already host-owned. Do not wait for the preceding
            # extend/decode work just to upload them before the candidate copy.
            # Existing result slots bound inflight batches; PyTorch tracks the
            # pinned source lifetime through the same-stream asynchronous copy.
            async_roots = getattr(self, "continuous_batching", False)
            roots = torch.tensor(
                [task.committed_token_ids[-1] for task in tasks],
                dtype=torch.int64,
                device="cpu" if async_roots else self.device,
                pin_memory=async_roots,
            ).to(self.device, non_blocking=async_roots)
            candidates = torch.stack([roots, *proposals], dim=1).to(torch.int32)
            if tuple(candidates.shape) != (len(tasks), self.profile.block_size):
                raise RuntimeError(
                    "STANDALONE proposal shape does not match profile block size"
                )
            for task, state in zip(tasks, states):
                # Ordinary decode accounts proposal inputs as committed.  They
                # remain physically allocated until reconciliation, but the
                # stable logical boundary is exactly the Target commit.
                state.req.kv_committed_len = state.valid_committed_len
                state.proposal_base_len = state.valid_committed_len
                logger.debug(
                    "DT Draft proposal ready: request=%s round=%d committed=%d "
                    "allocated=%d",
                    state.request_id,
                    state.last_round_id,
                    state.valid_committed_len,
                    state.req.kv.kv_allocated_len,
                )
                profile_emit(
                    "draft_proposal",
                    draft_instance=self.instance_id,
                    transfer_id=task.transfer_id,
                    request_id=state.request_id,
                    round_id=state.last_round_id,
                    committed_kv_len=state.valid_committed_len,
                    proposal_tokens=self.profile.draft_k,
                )
            if getattr(self, "continuous_batching", False):
                self._enqueue_eagle3_materialization_completion(
                    pending,
                    None,
                    candidates=candidates,
                    result_slot=result_slot,
                    admission_credit=admission_credit,
                )
                result_slot = None  # ownership moved to the completion
                admission_credit = False
            else:
                candidate_bytes = None
                if self.candidate_transport == "mmap":
                    candidate_bytes = candidates.cpu().contiguous().numpy().tobytes()
        except Exception as exc:
            logger.exception("failed DT STANDALONE batch")
            resources_quiescent = bool(getattr(exc, "resources_quiescent", True))
            if resources_quiescent:
                try:
                    torch.cuda.current_stream(self.device).synchronize()
                except Exception:
                    resources_quiescent = False
            if resources_quiescent:
                for task in tasks:
                    self._release_persistent_request(task.draft_kv_handle)
                if result_slot is not None:
                    self._free_result_slots.append(result_slot)
                if admission_credit:
                    self._release_direct_proposal_admission()
            else:
                self._quarantined_feature_draft_handles.update(
                    task.draft_kv_handle for task in tasks
                )
                self._enter_fatal_gpu_quarantine("unquiesced AR proposal failure")
            self._publish_failures([(task, exc) for task in tasks])
        else:
            if not getattr(self, "continuous_batching", False):
                self._publish_successes(
                    tasks,
                    (
                        candidates
                        if self.candidate_transport == "cuda-ipc"
                        else candidate_bytes
                    ),
                )
        finished_ns = time.monotonic_ns()
        decode_durations_ns = []
        previous_ns = extend_done_ns
        for done_ns in decode_done_ns:
            decode_durations_ns.append(done_ns - previous_ns)
            previous_ns = done_ns
        if profile_event_enabled("draft_standalone_batch"):
            profile_emit(
                "draft_standalone_batch",
                draft_instance=self.instance_id,
                transfer_ids=tuple(task.transfer_id for task in tasks),
                execute_started_ns=batch_start_ns,
                prepare_finished_ns=prepare_done_ns,
                proposal_enqueued_ns=decode_done_ns[-1] if decode_done_ns else extend_done_ns,
                finished_ns=finished_ns,
                asynchronous_completion=getattr(self, "continuous_batching", False),
                batch_size=len(tasks),
                bootstrap_count=sum(task.round_id == 0 for task in tasks),
                reconcile_count=sum(task.round_id > 0 for task in tasks),
                extend_cuda_graph=extend_cuda_graph,
                extend_graph_replays=extend_graph_replays,
                expected_extend_graph_replays=expected_extend_graph_replays,
                decode_graph_replays=decode_graph_replays,
                prepare_ns=prepare_done_ns - batch_start_ns,
                extend_ns=extend_done_ns - prepare_done_ns,
                decode_ns=decode_durations_ns,
                publish_ns=finished_ns - previous_ns,
                total_ns=finished_ns - batch_start_ns,
            )
        self._publish_capacity()

    def _record_graph_measurement(
        self,
        *,
        path: str,
        execution: str,
        measurement_rows: int,
        measurement_batch_rows: int,
        batch_size: int,
        model_runner_event_id: int | None = None,
        graph_replays: int | None = None,
        expected_graph_replays: int | None = None,
        fallback_reason: str | None = None,
    ) -> None:
        key = (path, execution)
        counts = self._graph_measurement_counts.setdefault(key, [0, 0])
        counts[0] += 1
        counts[1] += measurement_rows
        if execution == "graph" and counts[0] & (counts[0] - 1):
            return
        payload = {
            "draft_instance": self.instance_id,
            "path": path,
            "execution": execution,
            "fallback_reason": (
                None
                if execution == "graph"
                else fallback_reason or "model_runner_reported"
            ),
            "batches": counts[0],
            "measurement_rows": counts[1],
            "measurement_batch_rows": measurement_batch_rows,
            "batch_size": batch_size,
            "model_runner_event_namespace": "model_runner",
            "model_runner_event_id": model_runner_event_id,
            "graph_replays": graph_replays,
            "expected_graph_replays": expected_graph_replays,
        }
        logger.info(
            "%s%s",
            DT_DRAFT_GRAPH_MEASUREMENT_MARKER,
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )

    @staticmethod
    def _resolve_prefill_inputs(batch: ScheduleBatch) -> None:
        if batch.input_ids is None and batch.prefill_input_ids_cpu is not None:
            batch.input_ids = batch.prefill_input_ids_cpu.to(
                batch.device, non_blocking=True
            )
            batch.prefill_input_ids_cpu = None

    def _prepare_persistent_task(self, task) -> PersistentDraftRequest:
        from dt_protocol import DraftTaskKind

        if task.kind == DraftTaskKind.BOOTSTRAP:
            if task.draft_kv_handle in self._persistent_requests:
                raise ValueError("Draft bootstrap KV handle already exists")
            req = Req(
                rid=task.request_id,
                origin_input_text="",
                origin_input_ids=array("q", task.committed_token_ids),
                sampling_params=SamplingParams(
                    temperature=0, max_new_tokens=self.profile.draft_k
                ),
            )
            req.full_untruncated_fill_ids = req.origin_input_ids
            req.logprob_start_len = -1
            req.set_extend_range(0, task.committed_kv_len)
            state = PersistentDraftRequest(
                req=req,
                target_instance_id=task.target_instance_id,
                request_id=task.request_id,
                draft_kv_handle=task.draft_kv_handle,
                valid_committed_len=task.committed_kv_len,
                last_round_id=task.round_id,
                proposal_base_len=task.committed_kv_len,
                committed_token_ids=list(task.committed_token_ids),
            )
            ledger = getattr(self, "_rid_ledger", None)
            if ledger is not None:
                # §7.1 ring records reference requests by rid hash.
                ledger.register(task.request_id)
            self._persistent_requests[task.draft_kv_handle] = state
            self._persistent_index_add(state)
            return state

        state = self._persistent_requests.get(task.draft_kv_handle)
        if state is None:
            raise ValueError("Draft reconcile KV handle does not exist")
        if (
            state.request_id != task.request_id
            or state.target_instance_id != task.target_instance_id
        ):
            raise ValueError("Draft reconcile request identity does not match KV owner")
        if task.round_id != state.last_round_id + 1:
            raise ValueError("Draft reconcile round is not the next request round")
        if state.proposal_base_len != state.valid_committed_len:
            raise RuntimeError("Draft proposal base diverged from committed boundary")
        if not task.committed_token_ids:
            raise ValueError("Draft reconcile requires a non-empty committed delta")
        expected_len = state.valid_committed_len + len(task.committed_token_ids)
        if task.committed_kv_len != expected_len:
            raise ValueError("Draft reconcile delta does not end at committed length")

        req = state.req
        if req.kv is None or req.req_pool_idx is None:
            raise RuntimeError("Draft reconcile lost its persistent KV handle")
        old_allocated_len = req.kv.kv_allocated_len
        if old_allocated_len < state.valid_committed_len:
            raise RuntimeError("Draft allocated KV is shorter than committed prefix")
        discarded_tokens = old_allocated_len - state.valid_committed_len
        self._free_persistent_kv_suffix(
            req, state.valid_committed_len, old_allocated_len
        )
        logger.debug(
            "DT Draft reconcile: request=%s round=%d stable=%d discarded=%d "
            "refill=%d committed=%d",
            state.request_id,
            task.round_id,
            state.valid_committed_len,
            discarded_tokens,
            len(task.committed_token_ids),
            task.committed_kv_len,
        )
        req.kv.kv_allocated_len = state.valid_committed_len
        req.kv_committed_len = state.valid_committed_len
        req.prefix_indices = self._draft_model_runner.req_to_token_pool.req_to_token[
            req.req_pool_idx, : state.valid_committed_len
        ].to(torch.int64)
        req.output_ids.extend(task.committed_token_ids)
        req._refresh_fill_ids()
        req.set_extend_range(state.valid_committed_len, task.committed_kv_len)
        state.valid_committed_len = task.committed_kv_len
        state.proposal_base_len = task.committed_kv_len
        state.last_round_id = task.round_id
        state.committed_token_ids.extend(task.committed_token_ids)
        profile_emit(
            "draft_reconciliation",
            draft_instance=self.instance_id,
            request_id=state.request_id,
            round_id=task.round_id,
            discarded_tokens=discarded_tokens,
            refilled_tokens=len(task.committed_token_ids),
            committed_kv_len=task.committed_kv_len,
        )
        return state

    def _prepare_eagle3_persistent_tasks(
        self, tasks: list[object]
    ) -> tuple[list[PersistentDraftRequest], bool]:
        """Batch the GPU bookkeeping shared by an EAGLE3 reconcile wave.

        ``_prepare_persistent_task`` intentionally remains the general,
        per-request path used by STANDALONE and by mixed bootstrap/reconcile
        batches.  A normal EAGLE3 steady-state batch, however, contains unique
        reconcile handles.  Freeing every rejected proposal suffix as one
        allocator group and packing all committed-prefix rows with one dtype
        conversion avoids two GPU launches per request without changing any
        request's persistent KV ownership.
        """

        from dt_protocol import DraftTaskKind

        if not tasks or any(task.kind != DraftTaskKind.RECONCILE for task in tasks):
            return [self._prepare_persistent_task(task) for task in tasks], False
        handles = [task.draft_kv_handle for task in tasks]
        if len(set(handles)) != len(handles):
            return [self._prepare_persistent_task(task) for task in tasks], False

        validated: list[tuple[object, PersistentDraftRequest, Req, int, int]] = []
        for task in tasks:
            state = self._persistent_requests.get(task.draft_kv_handle)
            if state is None:
                raise ValueError("Draft reconcile KV handle does not exist")
            if (
                state.request_id != task.request_id
                or state.target_instance_id != task.target_instance_id
            ):
                raise ValueError(
                    "Draft reconcile request identity does not match KV owner"
                )
            if task.round_id != state.last_round_id + 1:
                raise ValueError("Draft reconcile round is not the next request round")
            if state.proposal_base_len != state.valid_committed_len:
                raise RuntimeError(
                    "Draft proposal base diverged from committed boundary"
                )
            if not task.committed_token_ids:
                raise ValueError("Draft reconcile requires a non-empty committed delta")
            expected_len = state.valid_committed_len + len(task.committed_token_ids)
            if task.committed_kv_len != expected_len:
                raise ValueError(
                    "Draft reconcile delta does not end at committed length"
                )
            req = state.req
            if req.kv is None or req.req_pool_idx is None:
                raise RuntimeError("Draft reconcile lost its persistent KV handle")
            old_allocated_len = req.kv.kv_allocated_len
            if old_allocated_len < state.valid_committed_len:
                raise RuntimeError(
                    "Draft allocated KV is shorter than committed prefix"
                )
            validated.append(
                (
                    task,
                    state,
                    req,
                    state.valid_committed_len,
                    old_allocated_len,
                )
            )

        allocator = self._draft_model_runner.token_to_kv_pool_allocator
        allocator.free_group_begin()
        try:
            for _task, _state, req, committed_len, allocated_len in validated:
                self._free_persistent_kv_suffix(req, committed_len, allocated_len)
            allocator.free_group_end()
        except Exception as exc:
            # A failed grouped allocator update cannot prove which suffix pages
            # remain owned.  The outer feature-batch guard must quarantine the
            # affected rows rather than attempting a second free.
            setattr(exc, "resources_quiescent", False)
            raise

        # Publish the shortened ownership boundary before allocating the packed
        # prefix view.  If that allocation fails, outer cleanup must free only
        # the still-owned committed pages, never the suffixes released above.
        for _task, _state, req, committed_len, _allocated_len in validated:
            req.kv.kv_allocated_len = committed_len

        req_to_token = self._draft_model_runner.req_to_token_pool.req_to_token
        prefix_lengths = [item[3] for item in validated]
        packed_prefix_indices = torch.cat(
            [
                req_to_token[req.req_pool_idx, :committed_len]
                for _task, _state, req, committed_len, _allocated_len in validated
            ],
            dim=0,
        ).to(torch.int64)
        prefix_views = torch.split(packed_prefix_indices, prefix_lengths)

        states = []
        for (
            task,
            state,
            req,
            committed_len,
            old_allocated_len,
        ), prefix_indices in zip(validated, prefix_views, strict=True):
            discarded_tokens = old_allocated_len - committed_len
            logger.debug(
                "DT Draft grouped reconcile: request=%s round=%d stable=%d "
                "discarded=%d refill=%d committed=%d",
                state.request_id,
                task.round_id,
                committed_len,
                discarded_tokens,
                len(task.committed_token_ids),
                task.committed_kv_len,
            )
            req.kv_committed_len = committed_len
            req.prefix_indices = prefix_indices
            req.output_ids.extend(task.committed_token_ids)
            req._refresh_fill_ids()
            req.set_extend_range(committed_len, task.committed_kv_len)
            state.valid_committed_len = task.committed_kv_len
            state.proposal_base_len = task.committed_kv_len
            state.last_round_id = task.round_id
            state.committed_token_ids.extend(task.committed_token_ids)
            profile_emit(
                "draft_reconciliation",
                draft_instance=self.instance_id,
                request_id=state.request_id,
                round_id=task.round_id,
                discarded_tokens=discarded_tokens,
                refilled_tokens=len(task.committed_token_ids),
                committed_kv_len=task.committed_kv_len,
                grouped=True,
            )
            states.append(state)
        grouped_batches = self.__dict__.get("_eagle3_grouped_reconcile_batches", 0) + 1
        grouped_rows = self.__dict__.get("_eagle3_grouped_reconcile_rows", 0) + len(
            tasks
        )
        self._eagle3_grouped_reconcile_batches = grouped_batches
        self._eagle3_grouped_reconcile_rows = grouped_rows
        if grouped_batches & (grouped_batches - 1) == 0:
            logger.info(
                "DT EAGLE3 grouped reconcile observed: batches=%d rows=%d",
                grouped_batches,
                grouped_rows,
            )
        return states, True

    @staticmethod
    def _migration_status_key(migration_id: str, phase) -> str:
        return f"{migration_id}:{phase.value}"

    def _publish_migration_status(self, status) -> None:
        key = self._migration_status_key(status.migration_id, status.phase)
        self._delivery_outbox.add("migration_status", key, status)
        self._flush_reliable_outbox()

    def _require_p2p_transport(self):
        if self.migration_backend != "cuda-ipc" or self._p2p_transport is None:
            raise RuntimeError(
                "CUDA IPC migration control received without --migration-backend "
                "cuda-ipc"
            )
        return self._p2p_transport

    @staticmethod
    def _migration_gpu_control_key(message) -> tuple[type, str]:
        return type(message), message.migration_id

    def _defer_migration_gpu_control(self, message) -> None:
        controls = getattr(self, "_deferred_migration_gpu_controls", None)
        if controls is None:
            controls = {}
            self._deferred_migration_gpu_controls = controls
        key = self._migration_gpu_control_key(message)
        previous = controls.get(key)
        if previous is not None:
            if previous.message != message:
                raise ValueError(
                    "migration GPU control id was reused with another payload"
                )
            return
        deferred_ns = time.monotonic_ns()
        controls[key] = DeferredMigrationGpuControl(
            message=message,
            deferred_ns=deferred_ns,
        )
        profile_emit(
            "migration_gpu_control_deferred",
            migration_id=message.migration_id,
            draft_instance=self.instance_id,
            control=type(message).__name__,
            deferred_ns=deferred_ns,
            pending_dflash_batches=len(
                getattr(self, "_pending_dflash_completions", ())
            ),
            pending_flashmtp_batches=len(
                getattr(self, "_pending_flashmtp_completions", ())
            ),
            pending_eagle3_batches=len(
                getattr(self, "_pending_eagle3_completions", ())
            ),
        )

    def _has_pending_feature_gpu_completions(self) -> bool:
        return bool(
            getattr(self, "_pending_flashmtp_completions", ())
            or getattr(self, "_pending_dflash_completions", ())
            or getattr(self, "_pending_eagle3_completions", ())
        )

    def _poll_deferred_migration_gpu_controls(self) -> bool:
        if self._fatal_gpu_quarantine_active():
            return False
        controls = getattr(self, "_deferred_migration_gpu_controls", None)
        if not controls or self._has_pending_feature_gpu_completions():
            return False
        key = next(iter(controls))
        deferred = controls.pop(key)
        message = deferred.message
        from dt_protocol import DraftKvMigrationExport, DraftKvMigrationPrepare

        resumed_ns = time.monotonic_ns()
        profile_emit(
            "migration_gpu_control_resumed",
            migration_id=message.migration_id,
            draft_instance=self.instance_id,
            control=type(message).__name__,
            deferred_ns=deferred.deferred_ns,
            resumed_ns=resumed_ns,
            wait_ns=resumed_ns - deferred.deferred_ns,
        )
        if isinstance(message, DraftKvMigrationPrepare):
            self._handle_migration_prepare(message)
        elif isinstance(message, DraftKvMigrationExport):
            self._handle_migration_export(message)
        else:  # pragma: no cover - the enqueue boundary is closed above
            raise TypeError(
                f"unexpected deferred migration control {type(message).__name__}"
            )
        return True

    def _handle_migration_prepare(self, message) -> None:
        from dt_protocol import (
            PROTOCOL_VERSION,
            DraftKvMigrationPhase,
            DraftKvMigrationReservation,
            DraftKvMigrationStatus,
            EntryStatus,
        )

        if message.destination_draft_instance_id != self.instance_id:
            raise ValueError("KV migration prepare belongs to another Draft")
        status_key = self._migration_status_key(
            message.migration_id, DraftKvMigrationPhase.PREPARED
        )
        if self._delivery_outbox.get("migration_status", status_key):
            self._delivery_outbox.retry_now("migration_status", status_key)
            self._flush_reliable_outbox()
            return
        if message.migration_id in self._migration_decisions:
            self._publish_migration_finalized(
                message.migration_id,
                self._migration_decisions[message.migration_id],
                self._migration_release_requests.get(message.migration_id, False),
                self._migration_cleanup_ids.get(message.migration_id, ""),
            )
            return
        if self._has_pending_feature_gpu_completions():
            # Reservation writes the request page table and then reads the
            # allocated GPU indices into its transport descriptor.  On the
            # shared model stream that CPU read would wait for an unrelated
            # materialization/proposal.  Queue only this GPU-metadata phase;
            # the scheduler continues receiving lifecycle controls and
            # polling independent feature/copy futures in the meantime.
            self._defer_migration_gpu_control(message)
            return

        started_ns = time.monotonic_ns()
        reservation = None
        req = None
        error = None
        try:
            transport = self._require_p2p_transport()
            if time.monotonic_ns() >= message.deadline_ns:
                raise TimeoutError("KV migration prepare deadline expired")
            if message.draft_profile_id != self.profile.profile_id:
                raise ValueError("KV migration prepare profile mismatch")
            if message.committed_kv_len <= 0:
                raise ValueError("KV migration prepare requires committed KV")
            if message.draft_kv_handle in self._persistent_requests:
                raise ValueError("KV migration destination handle already exists")
            self._require_migration_kv_layout()
            req = Req(
                rid=message.request_id,
                origin_input_text="",
                origin_input_ids=array("q"),
                sampling_params=SamplingParams(
                    temperature=0, max_new_tokens=self.profile.draft_k
                ),
            )
            req.full_untruncated_fill_ids = req.origin_input_ids
            req.logprob_start_len = -1
            req.set_extend_range(message.committed_kv_len, message.committed_kv_len)
            req_pool = self._draft_model_runner.req_to_token_pool
            allocator = self._draft_model_runner.token_to_kv_pool_allocator
            if req_pool.alloc([req]) is None:
                raise RuntimeError("destination request-row OOM")
            indices = allocator.alloc(message.committed_kv_len)
            if indices is None:
                raise RuntimeError("destination KV-page OOM")
            req.kv = ReqKvInfo(
                kv_allocated_len=message.committed_kv_len,
                swa_evicted_seqlen=0,
            )
            req.kv_committed_len = message.committed_kv_len
            req.prefix_indices = indices.to(torch.int64)
            req_pool.req_to_token[req.req_pool_idx, : message.committed_kv_len] = (
                indices.to(torch.int32)
            )
            reservation = DraftKvMigrationReservation(
                destination_session_id=self._client.instance_session_id,
                destination_pool_generation=transport.pool_generation,
                page_indices=tuple(int(index) for index in indices.tolist()),
                row_index=int(req.req_pool_idx),
            )
            self._prepared_migrations[message.migration_id] = PreparedDraftMigration(
                message=message,
                req=req,
                reservation=reservation,
                prepared_ns=time.monotonic_ns(),
            )
        except Exception as exc:
            error = exc
            if req is not None and req.req_pool_idx is not None:
                self._cleanup_partial_migration_request(req)
            logger.exception("failed to reserve destination Draft KV")

        self._migration_roles[message.migration_id] = "destination"
        status = DraftKvMigrationStatus(
            protocol_version=PROTOCOL_VERSION,
            migration_id=message.migration_id,
            phase=DraftKvMigrationPhase.PREPARED,
            draft_instance_id=self.instance_id,
            source_draft_instance_id=message.source_draft_instance_id,
            destination_draft_instance_id=message.destination_draft_instance_id,
            target_instance_id=message.target_instance_id,
            request_id=message.request_id,
            draft_kv_handle=message.draft_kv_handle,
            draft_profile_id=message.draft_profile_id,
            round_id=message.round_id,
            committed_kv_len=message.committed_kv_len,
            status=EntryStatus.OK if error is None else EntryStatus.RETRYABLE_ERROR,
            completed_ns=time.monotonic_ns(),
            reservation=reservation,
            error_code="" if error is None else type(error).__name__,
            error_message="" if error is None else str(error),
        )
        self._publish_migration_status(status)
        profile_emit(
            "migration_destination_prepare",
            migration_id=message.migration_id,
            draft_instance=self.instance_id,
            request_id=message.request_id,
            committed_kv_len=message.committed_kv_len,
            status="ok" if error is None else "error",
            reservation_ns=time.monotonic_ns() - started_ns,
        )

    def _handle_migration_export(self, message) -> None:
        from dt_protocol import (
            PROTOCOL_VERSION,
            DraftKvMigrationPhase,
            DraftKvMigrationStatus,
            DraftKvMigrationTransfer,
            EntryStatus,
        )

        if message.source_draft_instance_id != self.instance_id:
            raise ValueError("KV migration export belongs to another Draft")
        status_key = self._migration_status_key(
            message.migration_id, DraftKvMigrationPhase.EXPORTED
        )
        if self._delivery_outbox.get("migration_status", status_key):
            self._delivery_outbox.retry_now("migration_status", status_key)
            self._flush_reliable_outbox()
            return
        if message.migration_id in self._migration_decisions:
            self._publish_migration_finalized(
                message.migration_id,
                self._migration_decisions[message.migration_id],
                self._migration_release_requests.get(message.migration_id, False),
                self._migration_cleanup_ids.get(message.migration_id, ""),
            )
            return
        if self._has_pending_feature_gpu_completions():
            # Export snapshots GPU page-table indices into the CUDA-IPC
            # descriptor.  Guard against every outstanding DFlash batch, not
            # just work for the handle being migrated, so another request's
            # model-stream work cannot block Coordinator polling.
            self._defer_migration_gpu_control(message)
            return

        start_ns = time.monotonic_ns()
        transfer = None
        error = None
        try:
            transport = self._require_p2p_transport()
            if time.monotonic_ns() >= message.deadline_ns:
                raise TimeoutError("KV migration export deadline expired")
            if message.draft_profile_id != self.profile.profile_id:
                raise ValueError("KV migration export profile mismatch")
            if message.reservation is None:
                raise ValueError(
                    "KV migration export requires a destination reservation"
                )
            if message.pool is not None and message.pool != self._kv_pool:
                raise ValueError(
                    "KV migration export contains stale source pool metadata"
                )
            state = self._persistent_requests.get(message.draft_kv_handle)
            if state is None:
                raise ValueError("KV migration source handle does not exist")
            if self._dflash_handle_has_active_consumer(message.draft_kv_handle):
                raise RuntimeError(
                    "DFLASH KV migration cannot export during materialization"
                )
            self._validate_migration_state(message, state)
            req = state.req
            if req.kv is None or req.req_pool_idx is None:
                raise RuntimeError("KV migration source row is unavailable")
            self._require_migration_kv_layout()
            allocated_len = req.kv.kv_allocated_len
            if allocated_len < state.valid_committed_len:
                raise RuntimeError("source KV is shorter than committed boundary")
            self._free_persistent_kv_suffix(
                req, state.valid_committed_len, allocated_len
            )
            req.kv.kv_allocated_len = state.valid_committed_len
            req.kv_committed_len = state.valid_committed_len
            indices = self._draft_model_runner.req_to_token_pool.req_to_token[
                req.req_pool_idx, : state.valid_committed_len
            ].to(torch.int64)
            req.prefix_indices = indices
            source_page_indices = tuple(int(index) for index in indices.tolist())
            destination_page_indices = message.reservation.page_indices
            if len(source_page_indices) != len(destination_page_indices):
                raise ValueError("source and destination migration page counts differ")
            from sglang.srt.speculative.dt_cuda_ipc import coalesce_page_runs

            run_count = len(
                coalesce_page_runs(source_page_indices, destination_page_indices)
            )
            kv_bytes = len(source_page_indices) * sum(
                buffer.item_length for buffer in self._kv_pool.buffers
            )
            producer_event, producer_event_handle = transport.producer_event()
            self._migration_source_events[message.migration_id] = producer_event
            self._migration_source_handles[message.migration_id] = (
                message.draft_kv_handle
            )
            transfer = DraftKvMigrationTransfer(
                source_session_id=self._client.instance_session_id,
                source_pool_generation=transport.pool_generation,
                page_indices=source_page_indices,
                token_ids=tuple(state.committed_token_ids),
                producer_event_handle=producer_event_handle,
                kv_bytes=kv_bytes,
                page_count=state.valid_committed_len,
                segment_count=run_count * len(self._kv_pool.buffers),
            )
        except Exception as exc:  # keep the reconciled source owner usable
            error = exc
            self._migration_source_events.pop(message.migration_id, None)
            self._migration_source_handles.pop(message.migration_id, None)
            logger.exception("failed to publish committed Draft KV transfer metadata")
        self._migration_roles[message.migration_id] = "source"
        status = DraftKvMigrationStatus(
            protocol_version=PROTOCOL_VERSION,
            migration_id=message.migration_id,
            phase=DraftKvMigrationPhase.EXPORTED,
            draft_instance_id=self.instance_id,
            source_draft_instance_id=message.source_draft_instance_id,
            destination_draft_instance_id=message.destination_draft_instance_id,
            target_instance_id=message.target_instance_id,
            request_id=message.request_id,
            draft_kv_handle=message.draft_kv_handle,
            draft_profile_id=message.draft_profile_id,
            round_id=message.round_id,
            committed_kv_len=message.committed_kv_len,
            status=EntryStatus.OK if error is None else EntryStatus.RETRYABLE_ERROR,
            completed_ns=time.monotonic_ns(),
            pool=self._kv_pool if error is None else None,
            reservation=message.reservation,
            transfer=transfer,
            error_code="" if error is None else type(error).__name__,
            error_message="" if error is None else str(error),
        )
        self._publish_migration_status(status)
        profile_emit(
            "migration_source_export",
            migration_id=message.migration_id,
            draft_instance=self.instance_id,
            request_id=message.request_id,
            round_id=message.round_id,
            committed_kv_len=message.committed_kv_len,
            kv_bytes=0 if transfer is None else transfer.kv_bytes,
            page_count=0 if transfer is None else transfer.page_count,
            segment_count=0 if transfer is None else transfer.segment_count,
            status="ok" if error is None else "error",
            total_ns=time.monotonic_ns() - start_ns,
        )

    def _handle_migration_import(self, message) -> None:
        from dt_protocol import (
            DraftKvMigrationPhase,
        )

        if message.destination_draft_instance_id != self.instance_id:
            raise ValueError("KV migration import belongs to another Draft")
        status_key = self._migration_status_key(
            message.migration_id, DraftKvMigrationPhase.IMPORTED
        )
        if self._delivery_outbox.get("migration_status", status_key):
            self._delivery_outbox.retry_now("migration_status", status_key)
            self._flush_reliable_outbox()
            return
        if message.migration_id in self._migration_decisions:
            self._publish_migration_finalized(
                message.migration_id,
                self._migration_decisions[message.migration_id],
                self._migration_release_requests.get(message.migration_id, False),
                self._migration_cleanup_ids.get(message.migration_id, ""),
            )
            return

        prepared = self._prepared_migrations.get(message.migration_id)
        if prepared is not None and prepared.transfer_future is not None:
            if prepared.import_message != message:
                raise ValueError("migration import id was reused with another payload")
            return

        error = None
        try:
            transport = self._require_p2p_transport()
            if time.monotonic_ns() >= message.deadline_ns:
                raise TimeoutError("KV migration import deadline expired")
            if message.draft_profile_id != self.profile.profile_id:
                raise ValueError("KV migration import profile mismatch")
            if prepared is None:
                raise ValueError("KV migration destination was not prepared")
            self._validate_migration_control(message, prepared.message)
            if message.reservation != prepared.reservation:
                raise ValueError("KV migration reservation is stale or mismatched")
            if message.source_pool is None or message.transfer is None:
                raise ValueError("KV migration import requires CUDA IPC metadata")
            if message.source_pool.transport != "cuda-ipc":
                raise ValueError("KV migration source transport is not cuda-ipc")
            if (
                message.source_pool.instance_session_id
                != message.transfer.source_session_id
                or message.source_pool.pool_generation
                != message.transfer.source_pool_generation
            ):
                raise ValueError("KV migration source pool generation is stale")
            if (
                message.source_pool.profile_id != self.profile.profile_id
                or message.source_pool.layout != "NHD"
                or message.source_pool.page_size != 1
            ):
                raise ValueError("KV migration source pool layout/profile mismatch")
            if (
                message.transfer.page_count != message.committed_kv_len
                or len(message.transfer.page_indices) != message.committed_kv_len
                or len(message.transfer.token_ids) != message.committed_kv_len
                or message.transfer.kv_bytes <= 0
                or not message.transfer.producer_event_handle
            ):
                raise ValueError("KV migration transfer boundary is invalid")
            if any(
                other.transfer_future is not None and not other.transfer_future.done()
                for migration_id, other in self._prepared_migrations.items()
                if migration_id != message.migration_id
            ):
                raise RuntimeError("destination CUDA IPC transfer worker is busy")
            if message.inject_failure:
                raise RuntimeError("injected destination import failure")
            prepared.import_message = message
            prepared.scheduler_submit_ns = time.monotonic_ns()
            prepared.transfer_future = transport.submit(
                source_draft_instance_id=message.source_draft_instance_id,
                source_pool=message.source_pool,
                transfer=message.transfer,
                reservation=message.reservation,
            )
            profile_emit(
                "migration_transfer_submission",
                migration_id=message.migration_id,
                draft_instance=self.instance_id,
                source_draft=message.source_draft_instance_id,
                destination_draft=message.destination_draft_instance_id,
                kv_bytes=message.transfer.kv_bytes,
                page_count=message.transfer.page_count,
                segment_count=message.transfer.segment_count,
                scheduler_submit_ns=prepared.scheduler_submit_ns,
            )
        except Exception as exc:
            error = exc
            logger.exception("failed to submit committed Draft KV P2P import")
        if error is not None:
            self._complete_migration_import(message, error=error)

    @staticmethod
    def _validate_migration_control(message, prepared_message) -> None:
        fields = (
            "migration_id",
            "source_draft_instance_id",
            "destination_draft_instance_id",
            "target_instance_id",
            "request_id",
            "draft_kv_handle",
            "draft_profile_id",
            "round_id",
            "committed_kv_len",
            "deadline_ns",
        )
        if any(
            getattr(message, field) != getattr(prepared_message, field)
            for field in fields
        ):
            raise ValueError("KV migration control does not match its reservation")

    def _poll_migration_transfers(self) -> bool:
        if self._fatal_gpu_quarantine_active():
            return False
        changed = False
        for prepared in tuple(self._prepared_migrations.values()):
            future = prepared.transfer_future
            if future is None or not future.done():
                continue
            prepared.transfer_future = None
            error = None
            metrics = None
            try:
                metrics = future.result()
                if time.monotonic_ns() >= prepared.import_message.deadline_ns:
                    raise TimeoutError("KV migration completed after its deadline")
            except Exception as exc:  # noqa: BLE001 - surface worker failure
                error = exc
                logger.error(
                    "failed to complete committed Draft KV P2P import: %s",
                    exc,
                )
            self._complete_migration_import(
                prepared.import_message,
                metrics=metrics,
                error=error,
            )
            changed = True
        changed = self._poll_migration_quiescence_probes() or changed
        return changed

    def _start_migration_quiescence_probe(self, migration_id: str) -> None:
        transport = getattr(self, "_p2p_transport", None)
        if (
            transport is None
            or migration_id in self._migration_quiescence_probes
            or not hasattr(transport, "probe_quiescence")
        ):
            return
        self._migration_quiescence_probes[migration_id] = transport.probe_quiescence()

    def _poll_migration_quiescence_probes(self) -> bool:
        if self._fatal_gpu_quarantine_active():
            return False
        changed = False
        now_ns = time.monotonic_ns()
        for migration_id, future in tuple(self._migration_quiescence_probes.items()):
            if not future.done():
                continue
            self._migration_quiescence_probes.pop(migration_id, None)
            try:
                quiescent = bool(future.result())
            except Exception:  # noqa: BLE001 - a failed fence proves nothing
                quiescent = False
                logger.exception("failed to re-probe quarantined migration copy stream")
            if not quiescent:
                self._migration_quiescence_retry_ns[migration_id] = (
                    now_ns + CONTROL_RETRY_INTERVAL_NS
                )
                continue
            self._migration_quiescence_retry_ns.pop(migration_id, None)
            self._quarantined_migration_ids.discard(migration_id)
            self._migration_destination_copy_quiescent[migration_id] = True
            prepared = self._prepared_migrations.pop(migration_id, None)
            if (
                prepared is not None
                and prepared.message.draft_kv_handle not in self._persistent_requests
                and prepared.req.req_pool_idx is not None
            ):
                self._cleanup_partial_migration_request(prepared.req)
            if migration_id in self._migration_decisions:
                self._publish_migration_finalized(
                    migration_id,
                    self._migration_decisions[migration_id],
                    self._migration_release_requests.get(migration_id, False),
                    self._migration_cleanup_ids.get(migration_id, ""),
                    destination_copy_quiescent=True,
                )
            changed = True
        for migration_id in tuple(self._quarantined_migration_ids):
            if (
                migration_id not in self._migration_quiescence_probes
                and self._migration_quiescence_retry_ns.get(migration_id, 0) <= now_ns
            ):
                self._start_migration_quiescence_probe(migration_id)
        return changed

    def _complete_migration_import(self, message, *, metrics=None, error=None) -> None:
        from dt_protocol import (
            PROTOCOL_VERSION,
            DraftKvMigrationPhase,
            DraftKvMigrationStatus,
            EntryStatus,
        )

        prepared = self._prepared_migrations.get(message.migration_id)
        if error is None and prepared is None:
            error = ValueError("KV migration destination reservation disappeared")
        if error is None:
            token_ids = tuple(message.transfer.token_ids)
            req = prepared.req
            req.origin_input_ids = array("q", token_ids)
            req.full_untruncated_fill_ids = req.origin_input_ids
            state = PersistentDraftRequest(
                req=req,
                target_instance_id=message.target_instance_id,
                request_id=message.request_id,
                draft_kv_handle=message.draft_kv_handle,
                valid_committed_len=message.committed_kv_len,
                last_round_id=message.round_id,
                proposal_base_len=message.committed_kv_len,
                committed_token_ids=list(token_ids),
            )
            self._persistent_requests[message.draft_kv_handle] = state
            self._persistent_index_add(state)
        elif prepared is not None:
            if getattr(error, "resources_quiescent", True):
                self._cleanup_partial_migration_request(prepared.req)
                self._prepared_migrations.pop(message.migration_id, None)
            else:
                # A multi-segment copy may have submitted a prefix before an
                # error. If the worker cannot synchronize that stream, retain
                # the destination row/pages for the life of this process.
                self._quarantined_migration_ids.add(message.migration_id)
                logger.critical(
                    "quarantined destination KV reservation after unquiesced "
                    "GPU-copy failure: migration=%s",
                    message.migration_id,
                )
                self._migration_destination_copy_quiescent[message.migration_id] = False
                self._start_migration_quiescence_probe(message.migration_id)

        imported_bytes = 0 if metrics is None else metrics.kv_bytes
        imported_pages = 0 if metrics is None else metrics.page_count
        completed_ns = time.monotonic_ns()
        status = DraftKvMigrationStatus(
            protocol_version=PROTOCOL_VERSION,
            migration_id=message.migration_id,
            phase=DraftKvMigrationPhase.IMPORTED,
            draft_instance_id=self.instance_id,
            source_draft_instance_id=message.source_draft_instance_id,
            destination_draft_instance_id=message.destination_draft_instance_id,
            target_instance_id=message.target_instance_id,
            request_id=message.request_id,
            draft_kv_handle=message.draft_kv_handle,
            draft_profile_id=message.draft_profile_id,
            round_id=message.round_id,
            committed_kv_len=message.committed_kv_len,
            status=EntryStatus.OK if error is None else EntryStatus.RETRYABLE_ERROR,
            completed_ns=completed_ns,
            reservation=message.reservation,
            pool=message.source_pool,
            transfer=message.transfer,
            imported_kv_bytes=imported_bytes,
            imported_pages=imported_pages,
            replayed_tokens=0,
            error_code="" if error is None else type(error).__name__,
            error_message="" if error is None else str(error),
        )
        self._publish_migration_status(status)
        gpu_copy_ns = 0 if metrics is None else metrics.gpu_copy_ns
        profile_emit(
            "migration_destination_import",
            migration_id=message.migration_id,
            draft_instance=self.instance_id,
            request_id=message.request_id,
            round_id=message.round_id,
            committed_kv_len=message.committed_kv_len,
            kv_bytes=imported_bytes,
            page_count=imported_pages,
            segment_count=0 if metrics is None else metrics.segment_count,
            cold_mapping_count=(0 if metrics is None else metrics.cold_mapping_count),
            mapping_ns=0 if metrics is None else metrics.mapping_ns,
            driver_submit_ns=0 if metrics is None else metrics.driver_submit_ns,
            gpu_copy_ns=gpu_copy_ns,
            gpu_completed_ns=0 if metrics is None else metrics.gpu_completed_ns,
            transfer_submission_ns=(0 if metrics is None else metrics.submitted_ns),
            effective_bandwidth_gbps=(
                0.0 if gpu_copy_ns == 0 else imported_bytes / gpu_copy_ns
            ),
            worker_exposed_ns=(
                0
                if metrics is None
                else metrics.gpu_completed_ns - metrics.submitted_ns
            ),
            worker_queue_and_prepare_ns=(
                0
                if (
                    metrics is None
                    or prepared is None
                    or prepared.scheduler_submit_ns == 0
                )
                else metrics.submitted_ns - prepared.scheduler_submit_ns
            ),
            completion_poll_ns=(
                0 if metrics is None else completed_ns - metrics.gpu_completed_ns
            ),
            scheduler_exposed_ns=(
                0
                if prepared is None or prepared.scheduler_submit_ns == 0
                else completed_ns - prepared.scheduler_submit_ns
            ),
            status="ok" if error is None else "error",
            error_code="" if error is None else type(error).__name__,
            destination_ready_ns=completed_ns,
        )
        deferred = self._deferred_migration_finalizations.pop(
            message.migration_id, None
        )
        if deferred is not None:
            self._handle_migration_finalize(deferred)

    def _remove_pending_feature_task(self, task) -> None:
        """Drop an unsubmitted migration task before releasing its source."""

        retained = deque()
        removed = False
        for pending_task in self._pending:
            if pending_task.transfer_id == task.transfer_id:
                removed = True
                self._pending_ids.discard(task.transfer_id)
                getattr(self, "_ring_pending_ids", set()).discard(task.transfer_id)
                self._received_ns_by_transfer.pop(task.transfer_id, None)
                self._cancelled_ids.discard(task.transfer_id)
            else:
                retained.append(pending_task)
        self._pending = retained
        if removed:
            self._publish_capacity()

    def _resume_feature_finalizations(self) -> None:
        if self._fatal_gpu_quarantine_active():
            return
        # Pull completion no longer implies compute submission. Keep migration
        # finalization behind every INPUT_READY consumer and its submitted
        # compute batch, matching the original same-batch ownership boundary.
        if (
            getattr(self, "_input_ready_features", ())
            or getattr(self, "_feature_staging_compute_busy", ())
            or (
                self._slot_decouple
                and any(c.staging_slots for c in self._pending_dflash_completions)
            )
        ):
            return
        for migration_id, message in tuple(
            self._deferred_migration_finalizations.items()
        ):
            prepared = self._prepared_migrations.get(migration_id)
            if prepared is not None and prepared.transfer_future is not None:
                continue
            feature_key = (
                (migration_id, message.feature_transfer_id)
                if message.feature_transfer_id
                else None
            )
            if feature_key in self._feature_source_inflight:
                continue
            self._deferred_migration_finalizations.pop(migration_id, None)
            self._handle_migration_finalize(message)

    def _handle_migration_finalize(self, message) -> None:
        from dt_protocol import (
            FINALIZE_DRAFT_KV_MIGRATION,
            DraftKvMigrationFinalized,
            draft_kv_migration_finalized_id,
            pull_done_id,
        )

        if self.instance_id not in {
            message.source_draft_instance_id,
            message.destination_draft_instance_id,
        }:
            raise ValueError("KV migration finalization belongs to another Draft")
        prepared = self._prepared_migrations.get(message.migration_id)
        if prepared is not None and prepared.transfer_future is not None:
            previous = self._deferred_migration_finalizations.get(message.migration_id)
            if previous is not None and previous != message:
                raise ValueError(
                    "migration finalization changed before copy result processing"
                )
            self._deferred_migration_finalizations[message.migration_id] = message
            profile_emit(
                "migration_finalize_deferred",
                migration_id=message.migration_id,
                draft_instance=self.instance_id,
                committed=message.committed,
                reason="gpu_copy_result_pending",
            )
            return
        feature_key = (
            (message.migration_id, message.feature_transfer_id)
            if message.feature_transfer_id
            else None
        )
        if (
            message.release_feature_source
            and feature_key in self._feature_source_inflight
        ):
            previous = self._deferred_migration_finalizations.get(message.migration_id)
            if previous is not None and previous != message:
                raise ValueError(
                    "migration finalization changed before feature pull completed"
                )
            self._deferred_migration_finalizations[message.migration_id] = message
            profile_emit(
                "migration_finalize_deferred",
                migration_id=message.migration_id,
                draft_instance=self.instance_id,
                committed=message.committed,
                reason="feature_gpu_copy_pending",
            )
            return
        expected_cleanup_id = (
            ""
            if message.published_result_release is None
            else pull_done_id(message.published_result_release)
        )
        if message.cleanup_id != expected_cleanup_id:
            raise ValueError("KV migration cleanup identity does not match its release")
        if message.published_result_release is not None:
            if message.published_result_release.draft_instance_id != self.instance_id:
                raise ValueError("published result release belongs to another Draft")
            self._handle_pull_done(message.published_result_release, acknowledge=False)
        role = self._migration_roles.get(message.migration_id)
        is_source = self.instance_id == message.source_draft_instance_id
        is_destination = self.instance_id == message.destination_draft_instance_id
        previous = self._migration_decisions.get(message.migration_id)
        previous_release = self._migration_release_requests.get(
            message.migration_id, False
        )
        previous_cleanup_id = self._migration_cleanup_ids.get(message.migration_id, "")
        previous_lease_release = self._migration_release_source_leases.get(
            message.migration_id, False
        )
        previous_feature_transfer = self._migration_release_feature_transfers.get(
            message.migration_id, ""
        )
        previous_feature_release = self._migration_release_feature_sources.get(
            message.migration_id, False
        )
        reverting_destination = (
            previous is True and not message.committed and role == "destination"
        )
        terminating_source = (
            previous is True
            and not message.committed
            and message.release_request
            and role == "source"
        )
        if (
            previous is not None
            and previous != message.committed
            and not reverting_destination
            and not terminating_source
        ):
            logger.warning(
                "ignoring stale KV migration finalization decision: "
                "migration=%s current=%s received=%s",
                message.migration_id,
                previous,
                message.committed,
            )
            self._publish_migration_finalized(
                message.migration_id,
                previous,
                previous_release,
                previous_cleanup_id,
            )
            return
        if (
            previous == message.committed
            and previous_release
            and not message.release_request
        ):
            self._publish_migration_finalized(
                message.migration_id,
                previous,
                previous_release,
                previous_cleanup_id,
            )
            return
        if previous_lease_release and not message.release_source_lease:
            self._publish_migration_finalized(
                message.migration_id,
                previous,
                previous_release,
                previous_cleanup_id,
            )
            return
        if previous_feature_release and not message.release_feature_source:
            self._publish_migration_finalized(
                message.migration_id,
                previous,
                previous_release,
                previous_cleanup_id,
            )
            return
        if (
            previous == message.committed
            and previous_release == message.release_request
            and previous_cleanup_id
            and message.cleanup_id != previous_cleanup_id
        ):
            self._publish_migration_finalized(
                message.migration_id,
                previous,
                previous_release,
                previous_cleanup_id,
            )
            return
        if previous is not None and (
            previous != message.committed
            or previous_release != message.release_request
            or previous_cleanup_id != message.cleanup_id
            or previous_lease_release != message.release_source_lease
            or previous_feature_transfer != message.feature_transfer_id
            or previous_feature_release != message.release_feature_source
        ):
            previous_message = DraftKvMigrationFinalized(
                protocol_version=message.protocol_version,
                migration_id=message.migration_id,
                draft_instance_id=self.instance_id,
                committed=previous,
                completed_ns=0,
                release_request=previous_release,
                cleanup_id=previous_cleanup_id,
                destination_copy_quiescent=(
                    self._migration_destination_copy_quiescent.get(
                        message.migration_id, False
                    )
                ),
                source_lease_released=self._migration_source_leases_released.get(
                    message.migration_id, False
                ),
                feature_transfer_id=previous_feature_transfer,
                feature_source_released=self._migration_feature_sources_released.get(
                    message.migration_id, False
                ),
            )
            self._delivery_outbox.ack(
                FINALIZE_DRAFT_KV_MIGRATION,
                draft_kv_migration_finalized_id(previous_message),
            )
        destination_copy_quiescent = (
            not is_destination
            or message.migration_id not in self._quarantined_migration_ids
        )
        if is_destination:
            self._migration_destination_copy_quiescent[message.migration_id] = (
                destination_copy_quiescent
            )
        source_event_released = False
        if is_source and message.release_source_lease:
            source_handle = self._migration_source_handles.get(message.migration_id)
            if source_handle is not None and source_handle != message.draft_kv_handle:
                raise ValueError("migration source lease belongs to another KV handle")
            if message.migration_id in self._migration_source_events:
                self._migration_source_events.pop(message.migration_id)
                self._migration_source_handles.pop(message.migration_id, None)
                source_event_released = True
                self._migration_source_leases_released[message.migration_id] = True
            elif self._migration_source_leases_released.get(
                message.migration_id, False
            ):
                source_event_released = True
            else:
                # Finalize can overtake an Export that was still queued. Since
                # the Coordinator never sent an Import without receiving that
                # Export, the absent local event is a vacuously released lease.
                source_event_released = True
                self._migration_source_leases_released[message.migration_id] = True
        source_release_unproven = bool(
            is_source and message.release_source_lease and not source_event_released
        )
        if (
            message.release_request or is_destination and not message.committed
        ) and not source_release_unproven:
            self._release_persistent_request(
                message.draft_kv_handle, request_id=message.request_id
            )
        elif message.release_request and source_release_unproven:
            self._deferred_persistent_releases[message.draft_kv_handle] = (
                message.request_id
            )
        if is_source and message.release_source_lease and source_event_released:
            self._drain_deferred_persistent_release(message.draft_kv_handle)
        feature_source_released = False
        if message.release_feature_source:
            if not is_destination or not message.feature_transfer_id:
                raise ValueError(
                    "only a migration destination can release a deferred feature source"
                )
            deferred_key = (message.migration_id, message.feature_transfer_id)
            if deferred_key in self._feature_source_unquiescent:
                raise RuntimeError(
                    "cannot release an unquiesced migration feature source"
                )
            feature_task = self._deferred_feature_source_tasks.pop(deferred_key, None)
            if feature_task is None:
                if not self._migration_feature_sources_released.get(
                    message.migration_id, False
                ):
                    raise ValueError(
                        "migration feature source was not retained by destination"
                    )
            else:
                if (
                    feature_task.source_read_migration_id != message.migration_id
                    or feature_task.transfer_id != message.feature_transfer_id
                    or feature_task.request_id != message.request_id
                    or feature_task.draft_kv_handle != message.draft_kv_handle
                ):
                    raise ValueError(
                        "migration feature release does not match retained task"
                    )
                if deferred_key not in self._feature_source_quiescent:
                    self._remove_pending_feature_task(feature_task)
                self._publish_source_done(feature_task)
                self._flush_source_done()
            self._feature_source_quiescent.discard(deferred_key)
            self._feature_source_inflight.discard(deferred_key)
            self._feature_source_unquiescent.discard(deferred_key)
            feature_source_released = True
            self._migration_feature_sources_released[message.migration_id] = True
        elif is_destination and not message.committed and message.feature_transfer_id:
            # Rollback re-dispatches this still-leased descriptor to the
            # authoritative source Draft.  The abandoned destination copy
            # must not release the Target range.
            self._deferred_feature_source_tasks.pop(
                (message.migration_id, message.feature_transfer_id), None
            )
            self._feature_source_quiescent.discard(
                (message.migration_id, message.feature_transfer_id)
            )
            self._feature_source_inflight.discard(
                (message.migration_id, message.feature_transfer_id)
            )
            self._feature_source_unquiescent.discard(
                (message.migration_id, message.feature_transfer_id)
            )
        prepared = self._prepared_migrations.get(message.migration_id)
        if (
            prepared is not None
            and message.migration_id
            not in getattr(self, "_quarantined_migration_ids", ())
            and message.draft_kv_handle not in self._persistent_requests
            and prepared.req.req_pool_idx is not None
        ):
            self._cleanup_partial_migration_request(prepared.req)
        if message.migration_id not in getattr(self, "_quarantined_migration_ids", ()):
            self._prepared_migrations.pop(message.migration_id, None)
        self._migration_decisions[message.migration_id] = message.committed
        self._migration_release_requests[message.migration_id] = message.release_request
        self._migration_cleanup_ids[message.migration_id] = message.cleanup_id
        self._migration_release_source_leases[message.migration_id] = (
            message.release_source_lease
        )
        self._migration_release_feature_transfers[message.migration_id] = (
            message.feature_transfer_id
        )
        self._migration_release_feature_sources[message.migration_id] = (
            message.release_feature_source
        )
        from dt_protocol import DraftKvMigrationPhase

        for phase in DraftKvMigrationPhase:
            self._delivery_outbox.ack(
                "migration_status",
                self._migration_status_key(message.migration_id, phase),
            )
        self._publish_migration_finalized(
            message.migration_id,
            message.committed,
            message.release_request,
            message.cleanup_id,
            destination_copy_quiescent=destination_copy_quiescent,
            source_lease_released=self._migration_source_leases_released.get(
                message.migration_id, False
            ),
            feature_transfer_id=message.feature_transfer_id,
            feature_source_released=feature_source_released,
        )
        row_present = message.draft_kv_handle in self._persistent_requests
        profile_emit(
            "migration_finalize",
            migration_id=message.migration_id,
            draft_instance=self.instance_id,
            role=role or "none",
            committed=message.committed,
            row_present=row_present,
            destination_copy_quiescent=destination_copy_quiescent,
            source_lease_release_requested=message.release_source_lease,
            source_event_released=source_event_released,
            feature_transfer_id=message.feature_transfer_id,
            feature_source_released=feature_source_released,
        )

    def _publish_migration_finalized(
        self,
        migration_id: str,
        committed: bool,
        release_request: bool = False,
        cleanup_id: str = "",
        *,
        destination_copy_quiescent: bool | None = None,
        source_lease_released: bool | None = None,
        feature_transfer_id: str | None = None,
        feature_source_released: bool | None = None,
    ) -> None:
        from dt_protocol import (
            FINALIZE_DRAFT_KV_MIGRATION,
            PROTOCOL_VERSION,
            DraftKvMigrationFinalized,
            draft_kv_migration_finalized_id,
        )

        if destination_copy_quiescent is None:
            destination_copy_quiescent = self._migration_destination_copy_quiescent.get(
                migration_id, False
            )
        if source_lease_released is None:
            source_lease_released = self._migration_source_leases_released.get(
                migration_id, False
            )
        if feature_transfer_id is None:
            feature_transfer_id = self._migration_release_feature_transfers.get(
                migration_id, ""
            )
        if feature_source_released is None:
            feature_source_released = self._migration_feature_sources_released.get(
                migration_id, False
            )
        candidate = DraftKvMigrationFinalized(
            protocol_version=PROTOCOL_VERSION,
            migration_id=migration_id,
            draft_instance_id=self.instance_id,
            committed=committed,
            completed_ns=time.monotonic_ns(),
            release_request=release_request,
            cleanup_id=cleanup_id,
            destination_copy_quiescent=destination_copy_quiescent,
            source_lease_released=source_lease_released,
            feature_transfer_id=feature_transfer_id,
            feature_source_released=feature_source_released,
        )
        message_id = draft_kv_migration_finalized_id(candidate)
        for operation, pending_id, pending in self._delivery_outbox.pending():
            if (
                operation == FINALIZE_DRAFT_KV_MIGRATION
                and pending_id != message_id
                and isinstance(pending, DraftKvMigrationFinalized)
                and pending.migration_id == migration_id
                and pending.draft_instance_id == self.instance_id
            ):
                self._delivery_outbox.ack(operation, pending_id)
        message = self._delivery_outbox.get(FINALIZE_DRAFT_KV_MIGRATION, message_id)
        if message is None:
            message = candidate
        self._delivery_outbox.add(
            FINALIZE_DRAFT_KV_MIGRATION,
            draft_kv_migration_finalized_id(message),
            message,
        )
        self._flush_reliable_outbox()

    def _cleanup_partial_migration_request(self, req: Req) -> None:
        if req.kv is not None and req.req_pool_idx is not None:
            self._free_persistent_kv_suffix(req, 0, req.kv.kv_allocated_len)
            req.kv = None
        if req.req_pool_idx is not None:
            self._draft_model_runner.req_to_token_pool.free(req)

    def _require_migration_kv_layout(self) -> None:
        runner = self._draft_model_runner
        if runner.server_args.page_size != 1:
            raise RuntimeError("Draft KV migration baseline requires page_size == 1")
        allocator = runner.token_to_kv_pool_allocator
        if type(allocator).__name__ != "TokenToKVPoolAllocator":
            raise RuntimeError("Draft KV migration requires the plain token allocator")
        kv_cache = allocator._kvcache
        if getattr(kv_cache, "use_hnd", False):
            raise RuntimeError("Draft KV migration requires NHD KV layout")
        if getattr(kv_cache, "is_quantized_kv_cache", False):
            raise RuntimeError("Draft KV migration does not support quantized KV")

    def _validate_migration_state(self, message, state) -> None:
        if (
            state.target_instance_id != message.target_instance_id
            or state.request_id != message.request_id
            or state.draft_kv_handle != message.draft_kv_handle
            or state.last_round_id != message.round_id
            or state.valid_committed_len != message.committed_kv_len
            or len(state.committed_token_ids) != message.committed_kv_len
        ):
            raise ValueError("KV migration source state does not match admission")

    def _persistent_index_add(self, state: PersistentDraftRequest) -> None:
        """Index a persistent state by its rid-ledger cell key so the native
        FREEZE can evaluate READY rows without their f-string handle."""

        by_cell = getattr(self, "_persistent_by_cell", None)
        ledger = getattr(self, "_rid_ledger", None)
        if by_cell is None or ledger is None:
            return
        loc = ledger.handle(state.request_id)
        if loc is None:
            return
        state.cell_key = (loc[1] << 32) | loc[0]
        by_cell[state.cell_key] = (
            state,
            int(state.draft_kv_handle.rsplit(":", 1)[1]),
        )

    def _persistent_index_remove(self, state: PersistentDraftRequest) -> None:
        by_cell = getattr(self, "_persistent_by_cell", None)
        if by_cell is None:
            return
        entry = by_cell.get(getattr(state, "cell_key", 0))
        if entry is not None and entry[0] is state:
            del by_cell[state.cell_key]
        state.cell_key = 0

    def _defer_release_cell(self, state: PersistentDraftRequest) -> None:
        cells = getattr(self, "_deferred_release_cells", None)
        cell_key = getattr(state, "cell_key", 0)
        if cells is not None and cell_key:
            cells.add(cell_key)

    def _release_persistent_request(
        self, draft_kv_handle: str, *, request_id: str | None = None
    ) -> bool:
        state = self._persistent_requests.get(draft_kv_handle)
        if state is None:
            self._deferred_persistent_releases.pop(draft_kv_handle, None)
            return False
        if request_id is not None and state.request_id != request_id:
            raise ValueError("Draft release request identity does not match KV owner")
        if draft_kv_handle in self._quarantined_feature_draft_handles:
            self._deferred_persistent_releases[draft_kv_handle] = state.request_id
            self._defer_release_cell(state)
            profile_emit(
                "draft_request_release_quarantined",
                request_id=state.request_id,
                draft_kv_handle=draft_kv_handle,
                reason="unquiesced_feature_gpu_operation",
            )
            return False
        active_feature_consumer = self._dflash_handle_has_active_consumer(
            draft_kv_handle
        )
        if active_feature_consumer or any(
            handle == draft_kv_handle
            for handle in self._migration_source_handles.values()
        ):
            self._deferred_persistent_releases[draft_kv_handle] = state.request_id
            self._defer_release_cell(state)
            profile_emit(
                "draft_request_release_deferred",
                request_id=state.request_id,
                draft_kv_handle=draft_kv_handle,
                active_source_read_leases=sum(
                    handle == draft_kv_handle
                    for handle in self._migration_source_handles.values()
                ),
                active_feature_consumer=active_feature_consumer,
            )
            return False
        self._persistent_requests.pop(draft_kv_handle)
        deferred_cells = getattr(self, "_deferred_release_cells", None)
        if deferred_cells is not None:
            deferred_cells.discard(getattr(state, "cell_key", 0))
        self._persistent_index_remove(state)
        self._deferred_persistent_releases.pop(draft_kv_handle, None)
        ledger = getattr(self, "_rid_ledger", None)
        if ledger is not None:
            # Defer the ledger eviction until the take pointer of the ring
            # that owns this request has passed the last record its Target
            # could have published (the writer is at most one ring capacity
            # ahead of consumed).  The mapping must stay resolvable for
            # in-flight records, or the worker would stall the ring behind
            # them; unrelated idle rings must not hold the mapping hostage.
            endpoints = getattr(self, "_ring_endpoints", {})
            thresholds = {
                ring_id: endpoint.next_seq + endpoint.capacity
                for ring_id, endpoint in endpoints.items()
                if self._ring_targets.get(ring_id) == state.target_instance_id
            }
            if thresholds:
                horizon = getattr(self, "_rid_evict_horizon", None)
                if horizon is None:
                    horizon = self._rid_evict_horizon = {}
                horizon[state.request_id] = thresholds
                if len(horizon) > 65536:
                    oldest = next(iter(horizon))
                    horizon.pop(oldest, None)
                    ledger.evict(oldest)
            else:
                ledger.evict(state.request_id)
        req = state.req
        if req.kv is not None:
            if req.req_pool_idx is None:
                raise RuntimeError("Draft KV exists without a request-pool handle")
            self._free_persistent_kv_suffix(req, 0, req.kv.kv_allocated_len)
            req.kv = None
        if req.req_pool_idx is not None:
            self._draft_model_runner.req_to_token_pool.free(req)
        logger.debug(
            "DT Draft released persistent KV: request=%s handle=%s",
            state.request_id,
            state.draft_kv_handle,
        )
        self._ack_applied_request_releases(draft_kv_handle)
        return True

    def _dflash_handle_has_active_consumer(self, draft_kv_handle: str) -> bool:
        if any(task.draft_kv_handle == draft_kv_handle for task in self._pending):
            return True
        if any(
            item.task.draft_kv_handle == draft_kv_handle
            for feature_batch in self._pending_feature_batches
            for item in feature_batch.pending
        ):
            return True
        if any(
            entry.pending.task.draft_kv_handle == draft_kv_handle
            for entry in getattr(self, "_input_ready_features", ())
        ):
            return True
        if any(
            item[0].draft_kv_handle == draft_kv_handle
            for item in getattr(self, "_ring_admission", ())
        ):
            return True
        return any(
            item.task.draft_kv_handle == draft_kv_handle
            for completion in self._pending_dflash_completions
            for item in completion.pending
        ) or any(
            item.task.draft_kv_handle == draft_kv_handle
            for completion in getattr(self, "_pending_eagle3_completions", ())
            for item in completion.pending
        )

    def _drain_deferred_persistent_release(self, draft_kv_handle: str) -> bool:
        request_id = self._deferred_persistent_releases.get(draft_kv_handle)
        if request_id is None:
            return False
        released = self._release_persistent_request(
            draft_kv_handle, request_id=request_id
        )
        self._ack_applied_request_releases(draft_kv_handle)
        return released

    def _ack_applied_request_releases(self, draft_kv_handle: str) -> None:
        """Acknowledge only releases whose persistent state is actually gone."""

        if draft_kv_handle in self._persistent_requests:
            return
        from dt_protocol import (
            PROTOCOL_VERSION,
            RELEASE_DRAFT_REQUEST,
            ControlAck,
        )

        acknowledged = False
        for release_id, message in tuple(
            getattr(self, "_pending_request_release_acks", {}).items()
        ):
            if message.draft_kv_handle != draft_kv_handle:
                continue
            self._pending_request_release_acks.pop(release_id)
            self._delivery_outbox.add(
                RELEASE_DRAFT_REQUEST,
                release_id,
                ControlAck(
                    protocol_version=PROTOCOL_VERSION,
                    operation=RELEASE_DRAFT_REQUEST,
                    message_id=release_id,
                ),
            )
            acknowledged = True
        if acknowledged:
            self._flush_reliable_outbox()

    def _free_persistent_kv_suffix(self, req: Req, start: int, end: int) -> None:
        """Free an intentionally uncommitted proposal suffix from Draft KV."""
        if not 0 <= start <= end:
            raise ValueError("invalid persistent Draft KV suffix")
        if self._draft_model_runner.server_args.page_size != 1:
            raise RuntimeError("DT STANDALONE milestone 1 requires page_size == 1")
        if start == end:
            return
        indices = self._draft_model_runner.req_to_token_pool.req_to_token[
            req.req_pool_idx, start:end
        ]
        self._draft_model_runner.token_to_kv_pool_allocator.free_segment(
            indices, start_pos=start
        )

    def _submit_candidate_push(self, tasks, push_candidates, *, result_slot) -> None:
        """Submit one direct candidate push and track it until completion.

        The caller's proposal-admission credit moves to the recorded push.  A
        failed submit transfers nothing: the pooled slot (if any) returns here
        and the caller must release the credit.
        """

        try:
            push = self._candidate_push_transport.submit(
                tasks=tasks,
                candidates=push_candidates,
            )
        except Exception:
            if result_slot is not None:
                self._free_result_slots.append(result_slot)
            raise
        self._pending_candidate_pushes.append(
            PendingCandidatePush(
                push=push,
                tasks=list(tasks),
                result_slot=result_slot,
                candidates=push_candidates,
                admission_credit=result_slot is None,
            )
        )
        if self._profile_active:
            profile_emit(
                "draft_candidate_push_submit",
                draft_instance=self.instance_id,
                transfer_ids=tuple(task.transfer_id for task in tasks),
                entries=len(tasks),
                candidate_bytes=push.candidate_bytes,
                cold_mapping_count=push.cold_mapping_count,
                submitted_ns=push.submitted_ns,
                ticket=push.ticket,
            )

    def _profile_ring_push_held(self, tasks, reason: str) -> None:
        """Stamp why a §7.1 wave push is held back (#267 §8.3, issue #269).

        ``held_at_ns`` pairs with the ``submitted_ns`` of the
        ``draft_candidate_push_submit`` the release later emits for the same
        transfer ids, so the hold shows up as its own hop.
        """

        if not self._profile_active:
            return
        profile_emit(
            "draft_ring_push_held",
            draft_instance=self.instance_id,
            transfer_ids=tuple(task.transfer_id for task in tasks),
            reason=reason,
            held_at_ns=time.monotonic_ns(),
            held_waves=len(self._held_ring_pushes),
        )

    def _retire_tombstoned_ring_pushes(self) -> None:
        """Retire held ring-push rows whose ring destination is a tombstone.

        The Target's cleanup writes one (slot, generation, nonce) triple per
        record row into the ring's destination region about one scheduler
        iteration after the record itself.  A row that ended in its wave gets
        a zero triple, and no Coordinator confirmation copy will ever exist
        for it (the Coordinator only routes cancellations for tasks it
        dispatched).  Reading the tombstone through the same ring the record
        came through is the only way for the Draft to learn that cancellation;
        without it the wave's held push would wait for the confirmation
        forever and stall every other row of the wave.
        """

        changed = False
        for tasks, _push_candidates, _result_slot in self._held_ring_pushes:
            for task in tasks:
                if (
                    task.transfer_id in self._cancelled_ids
                    or task.transfer_id not in self._unconfirmed_direct
                    or task.candidate_destination is not None
                    or not task.direct_ring_seq
                ):
                    continue
                location = self._ring_record_positions.get(task.transfer_id)
                if location is None:
                    continue
                endpoint, position = location
                resolved = endpoint.destination(task.direct_ring_seq, position)
                if resolved is None:
                    # Not written yet (stale or a mid-rewrite lap).
                    continue
                slot_id, generation, nonce, _data_off, _meta_off = resolved
                if (slot_id, generation, nonce) != (0, 0, 0):
                    # A live row: its destination arrives here or with the
                    # Coordinator copy; nothing to retire.
                    continue
                self._cancelled_ids.add(task.transfer_id)
                changed = True
        if changed:
            self._release_held_ring_pushes()

    def _resolve_held_ring_destinations(self) -> None:
        """Re-read the destination region for the rows of held wave pushes.

        Confirmation-free, the region is a v2 row's only destination source
        (design §7.3): a row whose entry was not written yet when its batch
        retired is re-polled on every loop pass.  A live triple is adopted, a
        zero triple is the Target's tombstone and cancels the row, and a row
        still unresolved at its task deadline is failed: it is cancelled
        without a push, and the Target's own round deadline answers it.
        """

        if self._fatal_gpu_quarantine_active():
            return
        from msgspec.structs import replace

        now_ns = time.monotonic_ns()
        changed = False
        for tasks, _push_candidates, _result_slot in self._held_ring_pushes:
            for task in tasks:
                transfer_id = task.transfer_id
                current = self._direct_tasks_by_transfer.get(transfer_id, task)
                if (
                    transfer_id in self._cancelled_ids
                    or current.candidate_destination is not None
                ):
                    continue
                location = self._ring_record_positions.get(transfer_id)
                resolved = (
                    None
                    if location is None
                    else location[0].destination(task.direct_ring_seq, location[1])
                )
                destination = (
                    None
                    if resolved is None
                    else self._ring_destination_ref(task, resolved)
                )
                if destination is not None:
                    self._direct_tasks_by_transfer[transfer_id] = replace(
                        current, candidate_destination=destination
                    )
                elif resolved is not None and resolved[:3] == (0, 0, 0):
                    self._cancelled_ids.add(transfer_id)
                elif now_ns >= task.deadline_ns:
                    logger.warning(
                        "held ring push row has no destination at its deadline: %s",
                        transfer_id,
                    )
                    self._cancelled_ids.add(transfer_id)
                else:
                    continue
                changed = True
        if changed:
            self._release_held_ring_pushes()

    def _release_held_ring_pushes(self) -> None:
        """Submit or drop §7.1 wave pushes held for Coordinator confirmation."""

        held = getattr(self, "_held_ring_pushes", None)
        if not held:
            return
        transport = self._candidate_push_transport
        if transport is None:
            raise RuntimeError("direct candidate push transport is unavailable")
        remaining = []
        for tasks, push_candidates, result_slot in held:
            live_indices = [
                index
                for index, task in enumerate(tasks)
                if task.transfer_id not in self._cancelled_ids
            ]
            live_tasks = [tasks[index] for index in live_indices]
            if live_tasks and any(
                task.transfer_id in self._unconfirmed_direct
                if self._ring_confirmation
                else self._direct_tasks_by_transfer.get(
                    task.transfer_id, task
                ).candidate_destination
                is None
                for task in live_tasks
            ):
                # Still waiting on a confirmation (confirmation-free: on a
                # destination); nothing leaves the hold, so no cancelled
                # identity is retired yet either.
                remaining.append((tasks, push_candidates, result_slot))
                continue
            for task in tasks:
                if task.transfer_id in self._cancelled_ids:
                    self._retire_cancelled_transfer_identity(task)
            if not live_tasks:
                # Every task cancelled: retire the result slot or admission
                # credit like a failed submit.  A live task keeps its credit
                # and its push.  Direct pushes carry no pooled slot (None).
                if result_slot is not None:
                    self._free_result_slots.append(result_slot)
                else:
                    self._release_direct_proposal_admission()
                continue
            # Push the adopted Coordinator copies: they carry the candidate
            # destinations the provisional record tasks lacked.  Gather the
            # live rows on the push stream itself so the push copy is
            # stream-ordered after the gather; the rows are long past their
            # producing proposal since the hold released on confirmation.
            push_candidates.record_stream(transport.push_stream)
            with torch.cuda.stream(transport.push_stream):
                rows = _select_candidate_rows(
                    push_candidates[: len(tasks)], live_indices
                )
                adopted = [
                    self._direct_tasks_by_transfer.get(task.transfer_id, task)
                    for task in live_tasks
                ]
                try:
                    self._submit_candidate_push(
                        adopted, rows, result_slot=result_slot
                    )
                except Exception:
                    # The held entry is already dropped; its credit leaves
                    # with the push only on a successful submit.
                    if result_slot is None:
                        self._release_direct_proposal_admission()
                    raise
        self._held_ring_pushes = remaining

    def _ring_destination_ref(self, task, resolved):
        """Build a record task's arena destination from a resolved triple.

        ``resolved`` is the (slot_id, generation, nonce, data_offset,
        metadata_offset) the destination region returned for the task's ring
        seq and position.  The Target CPU-writes the triple when its cleanup
        lands the arena reservation — long before the Coordinator
        confirmation copy.
        """

        from dt_protocol import CandidateDestinationRef

        slot_id, generation, nonce, data_offset, metadata_offset = resolved
        if (slot_id, generation, nonce) == (0, 0, 0):
            return None
        registration = None
        for registered in self._candidate_push_transport._registered.values():
            if registered[0].target_instance_id == task.target_instance_id:
                registration = registered[0]
                break
        if registration is None:
            return None
        return CandidateDestinationRef(
            target_instance_id=registration.target_instance_id,
            target_session_id=registration.target_session_id,
            pool_generation=registration.pool_generation,
            transport_session=registration.transport_session,
            transfer_id=task.transfer_id,
            target_device_id=registration.target_device_id,
            device_uuid=registration.device_uuid,
            slot_id=slot_id,
            generation=generation,
            authorization_nonce=nonce,
            dtype="int32",
            row_width=registration.row_width,
            nbytes=registration.row_width * 4,
            data_allocation_length=registration.data_allocation_length,
            data_allocation_offset=(
                data_offset + slot_id * registration.row_width * 4
            ),
            data_ipc_handle=b"",
            metadata_allocation_length=registration.metadata_allocation_length,
            metadata_allocation_offset=(
                metadata_offset + slot_id * registration.metadata_words * 4
            ),
            metadata_ipc_handle=b"",
            notification_backend="transport-host-completion",
        )

    def _resolve_ring_destination(self, task):
        """Read the ring's destination region for one record task.

        Returns None when the region is not populated yet, when it holds a
        tombstone (the Target cancelled the row), or when the wave's ring
        generation is no longer observable (wrapped).
        """

        if not self._ring_dest_region:
            return None
        location = self._ring_record_positions.get(task.transfer_id)
        if location is None:
            return None
        endpoint, position = location
        resolved = endpoint.destination(task.direct_ring_seq, position)
        if resolved is None:
            return None
        return self._ring_destination_ref(task, resolved)

    def _publish_successes(
        self,
        tasks: list[object],
        candidates: bytes | torch.Tensor,
        *,
        result_slot: int | None = None,
        admission_credit: bool = False,
    ) -> None:
        from dt_protocol import (
            PROTOCOL_VERSION,
            DraftExecutionResultBatch,
            DraftResultEntry,
            ResultBufferRef,
        )

        profile = self._profile_active
        publish_started_ns = time.monotonic_ns() if profile else 0
        for task in tasks:
            self._cancelled_ids.discard(task.transfer_id)
        credit_held = False
        if result_slot is not None:
            slot_id = result_slot
        elif self.candidate_transport == "cuda-ipc":
            # The direct path publishes GPU→GPU into Target-owned candidate
            # destinations; there is no Draft-side result buffer to lease, so
            # the wave carries no pooled slot id.  It instead holds one
            # proposal-admission credit from here until its push retires
            # (unless the caller already holds the wave's credit).
            slot_id = None
            if not admission_credit:
                self._acquire_direct_proposal_admission()
            credit_held = True
        else:
            slot_id = self._free_result_slots.popleft()
        if self.candidate_transport == "cuda-ipc":
            # The credit this call holds leaves only with a queued push or a
            # held wave; a raise before either means it is still owned here.
            try:
                if self._candidate_push_transport is None:
                    raise RuntimeError("direct candidate push transport is unavailable")
                if not isinstance(candidates, torch.Tensor):
                    raise TypeError("direct candidate return requires GPU tensor rows")
                push_candidates = candidates.contiguous()
                # §7.1 record tasks adopt their Coordinator copy (destination,
                # scheduling metadata) at confirmation; hold the wave's whole
                # push while any of its tasks is still unconfirmed.  The
                # destination itself arrives earlier through the ring's
                # destination region.
                tasks = [
                    self._direct_tasks_by_transfer.get(task.transfer_id, task)
                    for task in tasks
                ]
                if any(
                    task.candidate_destination is None and task.direct_ring_seq
                    for task in tasks
                ):
                    from msgspec.structs import replace

                    resolved = []
                    missing = False
                    for task in tasks:
                        if task.candidate_destination is None and task.direct_ring_seq:
                            destination = self._resolve_ring_destination(task)
                            if destination is None:
                                missing = True
                                break
                            resolved.append(
                                replace(task, candidate_destination=destination)
                            )
                        else:
                            resolved.append(task)
                    if missing:
                        self._held_ring_pushes.append(
                            (tasks, push_candidates, slot_id)
                        )
                        self._profile_ring_push_held(tasks, "region_missing")
                        return
                    tasks = resolved
                if any(
                    task.candidate_destination is None
                    and task.direct_ring_seq
                    and task.transfer_id in self._unconfirmed_direct
                    for task in tasks
                ):
                    self._held_ring_pushes.append((tasks, push_candidates, slot_id))
                    self._profile_ring_push_held(tasks, "unconfirmed")
                    return
                self._submit_candidate_push(
                    tasks, push_candidates, result_slot=slot_id
                )
            except Exception:
                if credit_held:
                    self._release_direct_proposal_admission()
                raise
            return
        if not isinstance(candidates, bytes):
            raise TypeError("mmap candidate return requires serialized bytes")
        if self._result_store is None:
            raise RuntimeError("mmap result store is unavailable")
        candidate_bytes = candidates
        generation = self._result_store.write(slot_id, candidate_bytes)
        result_written_ns = time.monotonic_ns() if profile else 0
        self._result_sequence += 1
        batch_id = f"{self.instance_id}:result:{self._result_sequence}"
        entries = tuple(
            DraftResultEntry(
                transfer_id=task.transfer_id,
                request_id=task.request_id,
                round_id=task.round_id,
                attempt=task.attempt,
                target_instance_id=task.target_instance_id,
                status=self.EntryStatus.OK,
                row_index=row_index,
            )
            for row_index, task in enumerate(tasks)
        )
        message = DraftExecutionResultBatch(
            protocol_version=PROTOCOL_VERSION,
            batch_id=batch_id,
            draft_instance_id=self.instance_id,
            entries=entries,
            sent_ns=time.monotonic_ns(),
            buffer=ResultBufferRef(
                draft_instance_id=self.instance_id,
                source_device_id=self.gpu_id,
                buffer_id=slot_id,
                generation=generation,
                dtype="int32",
                row_width=self._result_row_width,
                transport="mmap",
                uri=self._result_store.uri,
                instance_session_id=self._client.instance_session_id,
                pool_generation=self._result_pool_generation,
            ),
        )
        range_transfer_ids = {}
        for target_id in dict.fromkeys(task.target_instance_id for task in tasks):
            rows = [
                index
                for index, task in enumerate(tasks)
                if task.target_instance_id == target_id
            ]
            key = (
                self._client.instance_session_id,
                self._result_pool_generation,
                slot_id,
                generation,
                rows[0],
                len(rows),
            )
            self._published_by_range[key] = batch_id
            range_transfer_ids[key] = frozenset(
                tasks[index].transfer_id for index in rows
            )
        published = PublishedResult(
            batch_id=batch_id,
            slot_id=slot_id,
            generation=generation,
            transfer_ids=tuple(task.transfer_id for task in tasks),
            transfer_rows={
                task.transfer_id: (task.target_instance_id, row_index)
                for row_index, task in enumerate(tasks)
            },
            range_transfer_ids=range_transfer_ids,
        )
        self._published_by_batch[batch_id] = published
        send_started_ns = time.monotonic_ns() if profile else 0
        self._send_result(message)
        if profile:
            send_done_ns = time.monotonic_ns()
            profile_emit(
                "draft_result_publish",
                draft_instance=self.instance_id,
                batch_id=batch_id,
                result_kind="proposal",
                entries=len(tasks),
                transfer_ids=tuple(task.transfer_id for task in tasks),
                candidate_rows=len(tasks),
                publish_started_ns=publish_started_ns,
                result_publish_completed_ns=send_done_ns,
                batch_completed_ns=send_done_ns,
                enqueue_flush_ns=send_done_ns - send_started_ns,
            )
            profile_emit(
                "draft_flashmtp_result_publish",
                draft_instance=self.instance_id,
                batch_id=batch_id,
                batch_size=len(tasks),
                transfer_ids=tuple(task.transfer_id for task in tasks),
                slot_id=slot_id,
                generation=generation,
                candidate_bytes=len(candidate_bytes),
                mmap_write_ns=result_written_ns - publish_started_ns,
                message_build_ns=send_started_ns - result_written_ns,
                enqueue_flush_ns=send_done_ns - send_started_ns,
                total_ns=send_done_ns - publish_started_ns,
            )

    def _poll_candidate_pushes(self) -> None:
        """Retire direct pushes only after the remote publication write completes."""

        if self._fatal_gpu_quarantine_active():
            return

        capacity_changed = False
        while self._pending_candidate_pushes:
            pending = self._pending_candidate_pushes[0]
            try:
                ready = bool(pending.push.event.query())
            except Exception as exc:
                exc.resources_quiescent = False
                self._enter_fatal_gpu_quarantine(
                    "unquiesced direct candidate push completion failure"
                )
                self._publish_failures([(task, exc) for task in pending.tasks])
                break
            if not ready:
                break
            self._pending_candidate_pushes.popleft()
            self._retire_candidate_push(
                pending, time.monotonic_ns() if self._profile_active else 0
            )
            capacity_changed = True
        if capacity_changed:
            self._publish_capacity()

    def _retire_candidate_push(
        self, pending, producer_completion_observed_ns: int, *, notice: bool = True
    ) -> None:
        """The bookkeeping of one completed direct push: notices, the
        Coordinator result, credit and identity retirement.  Shared by the
        legacy event poll and the D9 PUSH_DONE absorb, which passes the send
        loop's own observation of the completion.  ``notice`` False: rows the
        send loop pushed, which always wrote a completion record (and whose
        late destination only the loop knows)."""

        from dt_protocol import (
            PROTOCOL_VERSION,
            DraftExecutionResultBatch,
            DraftResultEntry,
        )

        # Target READY is driven by this direct, payload-free notice.  The
        # Coordinator result below is independent retirement/accounting.
        if notice:
            self._candidate_push_transport.publish_completion(
                pending.tasks, cancelled=pending.cancelled
            )
        if self._profile_active:
            notice_publish_returned_ns = time.monotonic_ns()
            profile_emit(
                "draft_candidate_producer_completion_observed",
                draft_instance=self.instance_id,
                transfer_ids=tuple(
                    task.transfer_id for task in pending.tasks
                ),
                entries=len(pending.tasks),
                submitted_ns=pending.push.submitted_ns,
                completion_observed_ns=producer_completion_observed_ns,
                notice_publish_returned_ns=notice_publish_returned_ns,
            )
        self._result_sequence += 1
        result_sent_ns = time.monotonic_ns()
        report = getattr(pending, "coordinator_result", True)
        message = DraftExecutionResultBatch(
            protocol_version=PROTOCOL_VERSION,
            batch_id=f"{self.instance_id}:direct-result:{self._result_sequence}",
            draft_instance_id=self.instance_id,
            entries=tuple(
                DraftResultEntry(
                    transfer_id=task.transfer_id,
                    request_id=task.request_id,
                    round_id=task.round_id,
                    attempt=task.attempt,
                    target_instance_id=task.target_instance_id,
                    status=(
                        self.EntryStatus.CANCELLED
                        if pending.cancelled
                        else self.EntryStatus.OK
                    ),
                    source_read_quiescent=pending.cancelled,
                )
                for task in pending.tasks
            ),
            sent_ns=result_sent_ns,
        )
        if report:
            self._send_result(message)
        if pending.result_slot is not None:
            self._free_result_slots.append(pending.result_slot)
        elif getattr(pending, "admission_credit", False):
            # The wave's admission credit returns only when its GPU push
            # completion retires here.
            self._release_direct_proposal_admission()
        early = getattr(self, "_early_pushed_ids", ())
        for task in pending.tasks:
            # A push queued at launch can retire before its completion
            # handler runs; that handler still needs the cancel mark.
            if task.transfer_id not in early and (
                not pending.cancelled
                or not self._has_local_execution_owner(task.transfer_id)
            ):
                self._cancelled_ids.discard(task.transfer_id)
            self._direct_tasks_by_transfer.pop(task.transfer_id, None)
        if self._profile_active:
            profile_emit(
                "draft_candidate_push_complete",
                draft_instance=self.instance_id,
                batch_id=message.batch_id,
                transfer_ids=tuple(task.transfer_id for task in pending.tasks),
                entries=len(pending.tasks),
                candidate_bytes=pending.push.candidate_bytes,
                submitted_ns=pending.push.submitted_ns,
                completed_ns=producer_completion_observed_ns,
                completion_ns=(
                    producer_completion_observed_ns
                    - pending.push.submitted_ns
                ),
                coordinator_result_sent_ns=result_sent_ns,
                cuda_copy_publish_ns=(
                    int(
                        pending.push.start_event.elapsed_time(
                            pending.push.event
                        )
                        * 1_000_000
                    )
                    if pending.push.start_event is not None
                    else 0
                ),
            )

    def _has_local_execution_owner(self, transfer_id: str) -> bool:
        """Return whether cancelled work still has a local queue/GPU owner."""

        if any(task.transfer_id == transfer_id for task in self._pending):
            return True
        if any(item[0].transfer_id == transfer_id for item in getattr(self, "_ring_admission", ())):
            return True
        for owner in (
            *self._pending_feature_batches,
            *(SimpleNamespace(pending=[entry.pending]) for entry in self._input_ready_features),
            *self._pending_flashmtp_completions,
            *self._pending_dflash_completions,
            *self._pending_eagle3_completions,
        ):
            if any(item.task.transfer_id == transfer_id for item in owner.pending):
                return True
        if not self._ring_confirmation and any(
            task.transfer_id == transfer_id
            for tasks, _candidates, _slot in self._held_ring_pushes
            for task in tasks
        ):
            # A held wave owns its rows until its release retires them; a
            # cancel push completing first must not revive the row, which
            # would hold the wave until the row's deadline.
            return True
        return False

    def _publish_materialization_successes(self, tasks: list[object]) -> None:
        from dt_protocol import (
            PROTOCOL_VERSION,
            DraftExecutionResultBatch,
            DraftResultEntry,
        )

        for task in tasks:
            self._cancelled_ids.discard(task.transfer_id)
        profile = self._profile_active
        publish_started_ns = time.monotonic_ns() if profile else 0
        self._result_sequence += 1
        batch_id = f"{self.instance_id}:materialized:{self._result_sequence}"
        message = DraftExecutionResultBatch(
            protocol_version=PROTOCOL_VERSION,
            batch_id=batch_id,
            draft_instance_id=self.instance_id,
            entries=tuple(
                DraftResultEntry(
                    transfer_id=task.transfer_id,
                    request_id=task.request_id,
                    round_id=task.round_id,
                    attempt=task.attempt,
                    target_instance_id=task.target_instance_id,
                    status=self.EntryStatus.OK,
                )
                for task in tasks
            ),
            sent_ns=time.monotonic_ns(),
        )
        self._send_result(message)
        if profile:
            publish_completed_ns = time.monotonic_ns()
            profile_emit(
                "draft_result_publish",
                draft_instance=self.instance_id,
                batch_id=batch_id,
                result_kind="materialization",
                entries=len(tasks),
                transfer_ids=tuple(task.transfer_id for task in tasks),
                candidate_rows=0,
                publish_started_ns=publish_started_ns,
                result_publish_completed_ns=publish_completed_ns,
                batch_completed_ns=publish_completed_ns,
                enqueue_flush_ns=publish_completed_ns - publish_started_ns,
            )

    def _publish_failures(self, failures: list[tuple[object, Exception]]) -> None:
        from dt_protocol import (
            PROTOCOL_VERSION,
            DraftExecutionResultBatch,
            DraftResultEntry,
        )

        for task, _exc in failures:
            self._cancelled_ids.discard(task.transfer_id)
        self._result_sequence += 1
        message = DraftExecutionResultBatch(
            protocol_version=PROTOCOL_VERSION,
            batch_id=f"{self.instance_id}:failure:{self._result_sequence}",
            draft_instance_id=self.instance_id,
            entries=tuple(
                DraftResultEntry(
                    transfer_id=task.transfer_id,
                    request_id=task.request_id,
                    round_id=task.round_id,
                    attempt=task.attempt,
                    target_instance_id=task.target_instance_id,
                    status=self.EntryStatus.FINAL_ERROR,
                    error_code=type(exc).__name__,
                    error_message=str(exc),
                    source_read_quiescent=bool(
                        getattr(exc, "resources_quiescent", True)
                    ),
                )
                for task, exc in failures
            ),
            sent_ns=time.monotonic_ns(),
        )
        self._send_result(message)

    def _publish_candidate_cancellations(
        self, transfer_ids: tuple[str, ...]
    ) -> None:
        """Fence cancelled direct destinations after all prior local GPU work.

        Coordinator cancellation is delivered at most once to the assigned
        Draft. A task with an already-submitted push is excluded: its normal
        generation publication is the stronger completion proof.
        """

        if self.candidate_transport != "cuda-ipc" or not transfer_ids:
            return
        if self._candidate_push_transport is None:
            raise RuntimeError("direct candidate push transport is unavailable")
        wanted = set(transfer_ids)
        already_pushing = {
            task.transfer_id
            for pending in self._pending_candidate_pushes
            for task in pending.tasks
        } | getattr(self, "_early_pushed_ids", set()) | {
            # D9: rows the send loop still owes a push (their PUSH_DONE).
            task.transfer_id
            for entry in getattr(self, "_send_loop_rows", {}).values()
            for task in entry[0]
            if task is not None
        } | {
            task.transfer_id
            for item in getattr(self, "_send_loop_remote", ())
            for task in item[3]
        }
        tasks_by_id = {
            transfer_id: task
            for transfer_id, task in self._direct_tasks_by_transfer.items()
            if transfer_id in wanted
        }
        for task in self._pending:
            if task.transfer_id in wanted:
                tasks_by_id[task.transfer_id] = task
        for owner in (
            *self._pending_feature_batches,
            *(SimpleNamespace(pending=[entry.pending]) for entry in getattr(self, "_input_ready_features", ())),
            *self._pending_flashmtp_completions,
            *self._pending_dflash_completions,
            *self._pending_eagle3_completions,
        ):
            for item in owner.pending:
                if item.task.transfer_id in wanted:
                    tasks_by_id[item.task.transfer_id] = item.task
        cancelled = [
            task
            for transfer_id, task in tasks_by_id.items()
            if transfer_id not in already_pushing
            and task.candidate_destination is not None
        ]
        if cancelled:
            push = self._candidate_push_transport.publish_cancelled(cancelled)
            self._pending_candidate_pushes.append(
                PendingCandidatePush(
                    push=push,
                    tasks=cancelled,
                    result_slot=None,
                    candidates=None,
                    cancelled=True,
                )
            )

    def _validate_task(self, task) -> None:
        from dt_protocol import DraftTaskKind

        source = task.source
        if task.draft_profile_id != self.profile.profile_id:
            raise ValueError("Draft task profile mismatch")
        if task.k != self.profile.draft_k:
            raise ValueError("Draft task k mismatch")
        destination = task.candidate_destination
        # A §7.1 ring-record task runs before its Coordinator confirmation
        # copy arrives; that copy carries the destination and the remaining
        # IPC fence fields.
        record_unconfirmed = bool(
            task.direct_ring_seq and destination is None and task.requires_proposal
        )
        if task.requires_proposal:
            if (
                self.candidate_transport == "cuda-ipc"
                and destination is None
                and not record_unconfirmed
            ):
                raise ValueError("direct candidate task lacks a Target destination")
            if self.candidate_transport == "mmap" and destination is not None:
                raise ValueError("mmap candidate task unexpectedly carries a GPU destination")
        elif destination is not None:
            raise ValueError("materialization-only task carries a candidate destination")
        if destination is not None and (
            destination.target_instance_id != task.target_instance_id
            or destination.transfer_id != task.transfer_id
            or destination.row_width != self.profile.block_size
            or destination.nbytes != self.profile.block_size * 4
        ):
            raise ValueError("candidate destination identity or shape mismatch")
        if self._is_feature_backed:
            if task.kind not in {
                DraftTaskKind.BOOTSTRAP,
                DraftTaskKind.RECONCILE,
            }:
                raise ValueError("feature-backed Draft requires a persistent task kind")
            if not task.draft_kv_handle:
                raise ValueError("feature-backed Draft task requires a KV handle")
            if source is None:
                raise ValueError("feature-backed Draft requires Target hidden states")
            if (
                source.target_instance_id != task.target_instance_id
                or source.transfer_id != task.transfer_id
            ):
                raise ValueError("feature source identity mismatch")
            if source.feature_schema_id != self.profile.feature_schema.schema_id:
                raise ValueError("feature schema mismatch")
            if source.transport != "cuda-ipc":
                raise ValueError("features require CUDA IPC transport")
            if source.dtype != "bfloat16":
                raise ValueError("feature source must be BF16")
            if tuple(source.shape) != (
                task.feature_token_count,
                self._pivot_width,
            ):
                raise ValueError("feature source shape mismatch")
            expected_nbytes = task.feature_token_count * self._pivot_width * 2
            if source.nbytes != expected_nbytes:
                raise ValueError("feature source byte length mismatch")
            if (
                task.feature_token_count <= 0
                or task.feature_token_count > self.feature_buffer_max_tokens
                or len(task.committed_token_ids) != task.feature_token_count
                or len(task.feature_token_ids) != task.feature_token_count
            ):
                raise ValueError("feature/token count is invalid")
            if (
                task.feature_token_offset < 0
                or task.committed_kv_len
                != task.feature_token_offset + task.feature_token_count
            ):
                raise ValueError("feature committed-length invariant failed")
            if task.kind == DraftTaskKind.BOOTSTRAP and (
                task.round_id != 0 or task.feature_token_offset != 0
            ):
                raise ValueError("feature bootstrap must begin at round/offset zero")
            if not (
                source.source_session_id
                and source.pool_generation
                and (record_unconfirmed or source.transport_session)
                and (record_unconfirmed or source.device_uuid)
                and source.ipc_handle
                and source.producer_event_handle
            ):
                raise ValueError("CUDA IPC feature descriptor fence is incomplete")
            if (
                source.allocation_length <= 0
                or source.allocation_offset < 0
                or source.byte_offset < 0
                or source.allocation_offset + source.byte_offset + source.nbytes
                > source.allocation_length
            ):
                raise ValueError("CUDA IPC feature source range is invalid")
            if any(
                token_id < 0 or token_id >= self.profile.vocab_size
                for token_id in (*task.committed_token_ids, task.next_token_id)
            ):
                raise ValueError("feature-backed token ID is outside vocabulary")
            return
        if self._persistent:
            if task.kind not in {
                DraftTaskKind.BOOTSTRAP,
                DraftTaskKind.RECONCILE,
            }:
                raise ValueError("STANDALONE Draft requires a persistent task kind")
            if source is not None:
                raise ValueError("STANDALONE Draft does not accept Target features")
            if not task.draft_kv_handle:
                raise ValueError("STANDALONE Draft task requires a KV handle")
            if not task.committed_token_ids:
                raise ValueError("STANDALONE Draft task requires committed token IDs")
            if task.committed_token_ids[-1] != task.next_token_id:
                raise ValueError(
                    "STANDALONE committed token delta must end in next_token_id"
                )
            if any(
                token_id < 0 or token_id >= self.profile.vocab_size
                for token_id in task.committed_token_ids
            ):
                raise ValueError("STANDALONE committed token ID is outside vocabulary")
            return
        if task.kind != DraftTaskKind.PROPOSE:
            raise ValueError("FlashMTP V1 accepts only pivot proposal tasks")
        if not task.requires_proposal or task.source_read_migration_id:
            raise ValueError("FlashMTP proposal must be immediate and non-migrating")
        if source is None:
            raise ValueError("FlashMTP V1 requires a Target feature source")
        if (
            source.target_instance_id != task.target_instance_id
            or source.transfer_id != task.transfer_id
        ):
            raise ValueError("FlashMTP source identity mismatch")
        if source.feature_schema_id != self.profile.feature_schema.schema_id:
            raise ValueError("Draft task feature schema mismatch")
        expected_nbytes = self._pivot_width * 2
        if source.nbytes != expected_nbytes:
            raise ValueError(
                f"Draft pivot nbytes mismatch: {source.nbytes} != {expected_nbytes}"
            )
        if (
            source.transport != "cuda-ipc"
            or source.uri
            or source.dtype != "bfloat16"
            or tuple(source.shape) != (1, self._pivot_width)
            or not source.source_session_id
            or not source.pool_generation
            or not source.transport_session
            or not source.device_uuid
            or not source.ipc_handle
            or not source.producer_event_handle
        ):
            raise ValueError(
                "FlashMTP requires a complete fenced CUDA IPC pivot descriptor"
            )
        if (
            source.source_device_id < 0
            or source.source_slot_id < 0
            or source.slot_generation < 0
            or source.byte_offset < 0
            or source.allocation_offset < 0
            or source.allocation_length <= 0
            or source.allocation_offset + source.byte_offset + source.nbytes
            > source.allocation_length
        ):
            raise ValueError("FlashMTP CUDA IPC pivot range is invalid")
        if (
            task.draft_kv_handle
            or task.committed_token_ids
            or task.committed_kv_len
            or task.feature_token_ids
            or task.feature_token_offset
            or task.feature_token_count
        ):
            raise ValueError("FlashMTP proposal must not carry persistent KV metadata")

    def _drop_cancelled_pending(self) -> None:
        if not self._cancelled_ids or not self._pending:
            return
        kept = deque()
        retired_handles: set[str] = set()
        for task in self._pending:
            if task.transfer_id in self._cancelled_ids:
                self._retire_cancelled_transfer_identity(task)
                if not self._persistent or self._is_feature_backed:
                    if not getattr(task, "source_read_migration_id", ""):
                        self._publish_source_done(task)
                if self._persistent:
                    retired_handles.add(task.draft_kv_handle)
            else:
                kept.append(task)
        self._pending = kept
        for draft_kv_handle in retired_handles:
            self._drain_deferred_persistent_release(draft_kv_handle)

    def _retire_cancelled_transfer_identity(self, task) -> None:
        """Retire a cancelled transfer identity and all of its direct holds.

        None of the direct-dispatch bookkeeping can be cleaned later for a
        cancelled identity: it has no Coordinator confirmation task, and the
        unconfirmed-direct timeout retires it without the held-result
        machinery.  Leaving any entry live leaks it per finished row, and
        unconfirmed entries are subtracted from the Coordinator
        queue-capacity bound, so the leak progressively weakens admission
        enforcement.
        """

        ring_pending = getattr(self, "_ring_pending_ids", set())
        if getattr(task, "candidate_destination", None) is None or (
            # Confirmation-free, no cancel result is relayed or ACKed for a
            # ring round, and a drop path need not have queued a cancel push.
            not self._ring_confirmation and task.transfer_id in ring_pending
        ):
            self._pending_ids.discard(task.transfer_id)
            ring_pending.discard(task.transfer_id)
        self._cancelled_ids.discard(task.transfer_id)
        self._received_ns_by_transfer.pop(task.transfer_id, None)
        getattr(self, "_ring_record_positions", {}).pop(task.transfer_id, None)
        for book in (
            "_unconfirmed_direct",
            "_direct_tasks_by_transfer",
            "_held_direct_source_done",
            "_held_direct_results",
        ):
            state = getattr(self, book, None)
            if state is not None:
                state.pop(task.transfer_id, None)

    def _release_result(self, published: PublishedResult) -> None:
        if self._published_by_batch.pop(published.batch_id, None) is None:
            return
        for key, batch_id in list(self._published_by_range.items()):
            if batch_id == published.batch_id:
                self._published_by_range.pop(key)
        self._free_result_slots.append(published.slot_id)

    def _handle_pull_done(self, message, *, acknowledge: bool = True) -> bool:
        from dt_protocol import PROTOCOL_VERSION, PULL_DONE, ControlAck, pull_done_id

        message_id = pull_done_id(message)
        if acknowledge and message_id in self._pull_done_applied_ids:
            self._flush_reliable_outbox()
            return False
        if (
            message.draft_session_id != self._client.instance_session_id
            or message.result_pool_generation != self._result_pool_generation
        ):
            return False
        released = False
        identity = (
            message.draft_session_id,
            message.result_pool_generation,
            message.buffer_id,
            message.generation,
        )
        batch_ids = {
            batch_id
            for candidate, batch_id in self._published_by_range.items()
            if candidate[:4] == identity
        }
        if len(batch_ids) > 1:
            raise RuntimeError("result buffer generation belongs to multiple batches")
        published = (
            self._published_by_batch[next(iter(batch_ids))] if batch_ids else None
        )
        if published is not None:
            # PullDone can release a coalesced Target range, a migration
            # singleton inside that range, or a covering range after another
            # cleanup already punched a hole in it. Validate against immutable
            # publication rows, then monotonically subtract every still-live
            # row in the acknowledged interval.
            row_start = message.row_offset
            row_end = row_start + message.row_count
            interval_transfer_ids = {
                transfer_id
                for transfer_id, (target_instance_id, row_index) in (
                    published.transfer_rows.items()
                )
                if target_instance_id == message.target_instance_id
                and row_start <= row_index < row_end
            }
            if len(interval_transfer_ids) != message.row_count:
                raise ValueError(
                    "PullDone range does not match published Target result rows"
                )
            live_transfer_ids = {
                transfer_id
                for transfer_ids in published.range_transfer_ids.values()
                for transfer_id in transfer_ids
            }
            self._rebuild_accepted_ranges(
                published, live_transfer_ids - interval_transfer_ids
            )
            if not published.range_transfer_ids:
                self._release_result(published)
                released = True
        if acknowledge:
            acknowledgement = ControlAck(
                protocol_version=PROTOCOL_VERSION,
                operation=PULL_DONE,
                message_id=message_id,
            )
            self._pull_done_applied_ids.add(message_id)
            self._delivery_outbox.add(PULL_DONE, message_id, acknowledgement)
            self._flush_reliable_outbox()
        return released

    def _rebuild_accepted_ranges(
        self, published: PublishedResult, accepted: set[str]
    ) -> None:
        # ResultBatchAck is delivered reliably and may be repeated after one
        # or more Targets have already returned PullDone.  Keep range state
        # monotonic: an ACK may discard rejected rows, but it must never
        # recreate a row whose reader has already released it.
        live_transfer_ids = {
            transfer_id
            for transfer_ids in published.range_transfer_ids.values()
            for transfer_id in transfer_ids
        }
        accepted.intersection_update(live_transfer_ids)
        for key in published.range_transfer_ids:
            self._published_by_range.pop(key, None)
        by_target: dict[str, list[tuple[int, str]]] = {}
        for transfer_id in accepted:
            target_row = published.transfer_rows.get(transfer_id)
            if target_row is None:
                continue
            target_id, row_index = target_row
            by_target.setdefault(target_id, []).append((row_index, transfer_id))

        rebuilt = {}
        for rows in by_target.values():
            rows.sort()
            segments: list[list[tuple[int, str]]] = []
            for row in rows:
                if not segments or row[0] != segments[-1][-1][0] + 1:
                    segments.append([])
                segments[-1].append(row)
            for segment in segments:
                key = (
                    self._client.instance_session_id,
                    self._result_pool_generation,
                    published.slot_id,
                    published.generation,
                    segment[0][0],
                    len(segment),
                )
                rebuilt[key] = frozenset(transfer_id for _, transfer_id in segment)
                self._published_by_range[key] = published.batch_id
        published.range_transfer_ids = rebuilt

    def _publish_capacity(self) -> None:
        from dt_protocol import PROTOCOL_VERSION, DraftCapacityUpdate

        req_pool = getattr(self._draft_model_runner, "req_to_token_pool", None)
        allocator = getattr(
            self._draft_model_runner, "token_to_kv_pool_allocator", None
        )
        active_transfers = sum(
            prepared.transfer_future is not None
            for prepared in self._prepared_migrations.values()
        )
        # Transfer identity, rather than transient queue membership, defines
        # the held request-round count.  Keep an accepted task held across
        # INPUT_READY, compute, and pending return; its publication path
        # retires this identity exactly once.
        ring_pending = getattr(self, "_ring_pending_ids", set())
        if self._ring_confirmation:
            unconfirmed = getattr(self, "_unconfirmed_direct", None) or {}
            # Coordinator copies waiting for their ring copy are not yet
            # received: the Coordinator already counts them as in flight.
            pending_tasks = len(self._pending_ids) - sum(
                transfer_id in self._pending_ids for transfer_id in unconfirmed
            )
            pending_work = pending_tasks + active_transfers
            ring_pending_tasks = len(ring_pending) - sum(
                transfer_id in ring_pending for transfer_id in unconfirmed
            )
        else:
            # Design §8.3: ring rounds never pass through the Coordinator, so
            # the cumulative pair (received_tasks, pending_tasks) counts
            # relayed rounds only, while pending_work, the Coordinator's
            # admission signal, counts every held row: READY, in flight and
            # awaiting its push, plus the relayed queue.
            pending_tasks = sum(
                transfer_id not in ring_pending for transfer_id in self._pending_ids
            )
            # Native ingress: READY-table rows get an identity only at freeze.
            ready_rows = len(self._ready_pending)
            pending_work = len(self._pending_ids) + active_transfers + ready_rows
            ring_pending_tasks = len(ring_pending) + ready_rows
        self._capacity_sequence += 1
        update = DraftCapacityUpdate(
            protocol_version=PROTOCOL_VERSION,
            draft_instance_id=self.instance_id,
            pending_tasks=pending_tasks,
            received_tasks=self._received_tasks,
            capacity_sequence=self._capacity_sequence,
            max_pending_tasks=self.max_pending_tasks,
            # Under the batch bound there is no result-row resource; report
            # the constant the Coordinator registers with (router.py, which
            # only stores this field).
            free_result_rows=(
                self.max_pending_tasks
                if self.candidate_transport == "cuda-ipc" and DRAFT_ADMISSION_BOUND
                else self._available_direct_proposal_admission() * self.max_batch_size
                if self.candidate_transport == "cuda-ipc"
                else (
                    self._available_dflash_result_slots() * self.max_batch_size
                    if self._uses_direct_gpu_source
                    else len(self._free_result_slots) * self.max_batch_size
                )
            ),
            healthy=not self._stop,
            observed_ns=time.monotonic_ns(),
            free_request_rows=(
                0 if req_pool is None else int(req_pool.available_size())
            ),
            free_kv_pages=(0 if allocator is None else int(allocator.available_size())),
            active_owner_count=len(self._persistent_requests),
            pending_work=pending_work,
            queue_delay_ns=self._last_queue_delay_ns,
            last_batch_size=self._last_batch_size,
            ring_pending_tasks=ring_pending_tasks,
        )
        if not self._client.update_capacity(update):
            logger.warning("Draft capacity update was locally backpressured")
        self._maybe_log_draft_stats(pending_work - active_transfers)

    def _maybe_log_draft_stats(self, pending_tasks: int) -> None:
        """Throttled draft-boundness sample for the capacity harness (#174).

        Only logged while the queue holds work, so an idle Draft stays silent
        and a measurement point's samples all reflect load.
        """
        if pending_tasks <= 0:
            return
        now_ns = time.monotonic_ns()
        if now_ns - self._last_stats_log_ns < DT_DRAFT_STATS_INTERVAL_NS:
            return
        self._last_stats_log_ns = now_ns
        from dt_protocol import DT_DRAFT_STATS_MARKER

        push_transport = self._candidate_push_transport
        logger.info(
            "%s%s",
            DT_DRAFT_STATS_MARKER,
            json.dumps(
                {
                    "instance": self.instance_id,
                    "observed_ns": now_ns,
                    "pending": pending_tasks,
                    "max_pending": self.max_pending_tasks,
                    "queue_delay_ms": round(self._last_queue_delay_ns / 1e6, 3),
                    "batch_size": self._last_batch_size,
                    "step_ms": round(self._last_step_ns / 1e6, 3),
                    # Ring entries the native worker deferred because every
                    # shared landing buffer was busy (cumulative).
                    "ring_landing_stalls": (
                        0 if self._ring_worker is None else self._ring_worker.landing_stalls
                    ),
                    # Loop-stage decomposition (issue #222): cumulative ns and
                    # call counts per stage since process start; a windowed
                    # diff of two samples gives ms per loop iteration.
                    "loop_iters": self._loop_iters,
                    "loop_wall_ns": self._loop_wall_ns,
                    "loop_stage_ns": dict(self._loop_stage_ns),
                    "loop_stage_calls": dict(self._loop_stage_calls),
                    "push_lane_ns": (
                        None
                        if push_transport is None
                        else dict(
                            getattr(push_transport, "submit_lane_ns", None) or {}
                        )
                    ),
                    # Send loop (D8): done_seq, descriptors outstanding, and
                    # the thread's cumulative wait on them; D9: its own CPU
                    # time (a windowed diff gives its core share) and the
                    # held list of late push rows.
                    "send_loop": (
                        None
                        if self._send_loop_worker is None
                        else dict(
                            zip(
                                self._send_loop_worker.STATS,
                                self._send_loop_worker.batch_stats().tolist(),
                            )
                        )
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    def _send_result(self, message) -> None:
        ring = self._ring_pending_ids if not self._ring_confirmation else ()
        if ring and any(entry.transfer_id in ring for entry in message.entries):
            # Confirmation-free, the Coordinator never dispatched a ring round
            # and would drop its result as stale (design §8.2): the candidate
            # push, or the Target's own deadline, is the round's outcome.
            # Retire the identity here, as the result's ACK would have.
            from msgspec.structs import replace

            relayed = tuple(
                entry for entry in message.entries if entry.transfer_id not in ring
            )
            for entry in message.entries:
                if entry.transfer_id in ring:
                    ring.discard(entry.transfer_id)
                    self._pending_ids.discard(entry.transfer_id)
                    # A failure result is the round's last trace here, and no
                    # Coordinator copy will retire its record position.
                    self._direct_tasks_by_transfer.pop(entry.transfer_id, None)
                    self._ring_record_positions.pop(entry.transfer_id, None)
            if not relayed:
                return
            message = replace(message, entries=relayed)
        unconfirmed = getattr(self, "_unconfirmed_direct", None)
        if unconfirmed and any(
            entry.transfer_id in unconfirmed for entry in message.entries
        ):
            from msgspec.structs import replace

            if getattr(message, "buffer", None) is not None:
                raise RuntimeError("direct Draft dispatch requires bufferless results")
            confirmed = tuple(
                entry for entry in message.entries if entry.transfer_id not in unconfirmed
            )
            for entry in message.entries:
                if entry.transfer_id in unconfirmed:
                    self._held_direct_results[entry.transfer_id] = replace(
                        message,
                        batch_id=f"{message.batch_id}:held:{entry.transfer_id}",
                        entries=(entry,),
                    )
            if not confirmed:
                return
            message = replace(message, entries=confirmed)
        self._delivery_outbox.add("result", message.batch_id, message)
        self._flush_reliable_outbox()

    def _flush_reliable_outbox(self) -> None:
        from dt_protocol import (
            FINALIZE_DRAFT_KV_MIGRATION,
            PULL_DONE,
            DraftKvMigrationFinalized,
            DraftKvMigrationStatus,
            SourceReadDoneBatch,
        )

        if getattr(self, "_stop", False):
            # Shutdown drains flush only the outbox; carry confirmed results.
            # While serving, the loop flushes them once per turn so that one
            # turn's confirmations share a batch.
            self._flush_confirmed_results()
        now_ns = time.monotonic_ns()
        for operation, message_id, message in self._delivery_outbox.due(now_ns=now_ns):
            if operation == "result":
                result_send_started_ns = (
                    time.monotonic_ns() if self._profile_active else 0
                )
                sent = self._client.publish_results(message)
                if self._profile_active and not self._persistent:
                    result_send_done_ns = time.monotonic_ns()
                    profile_emit(
                        "draft_flashmtp_result_send",
                        draft_instance=self.instance_id,
                        batch_id=message.batch_id,
                        batch_size=len(message.entries),
                        sent=bool(sent),
                        send_ns=result_send_done_ns - result_send_started_ns,
                    )
            elif operation == "source_read_done":
                sent = (
                    self._client.source_reads_done(message)
                    if isinstance(message, SourceReadDoneBatch)
                    else self._client.source_read_done(message)
                )
            elif operation == "migration_status":
                if not isinstance(message, DraftKvMigrationStatus):
                    raise TypeError("invalid migration status outbox payload")
                sent = self._client.migration_status(message)
            elif operation == FINALIZE_DRAFT_KV_MIGRATION:
                if not isinstance(message, DraftKvMigrationFinalized):
                    raise TypeError("invalid migration finalization outbox payload")
                sent = self._client.migration_finalized(message)
            elif operation == PULL_DONE:
                sent = self._client.send(message)
            elif operation == "release_draft_request":
                sent = self._client.send(message)
            else:
                raise RuntimeError(f"unknown Draft delivery operation {operation!r}")
            if not sent:
                break
            self._delivery_outbox.mark_sent(operation, message_id, now_ns=now_ns)

    def _publish_source_done(self, task) -> None:
        from dt_protocol import PROTOCOL_VERSION, SourceReadDone

        message = SourceReadDone(
            protocol_version=PROTOCOL_VERSION,
            transfer_id=task.transfer_id,
            target_instance_id=task.target_instance_id,
            draft_instance_id=self.instance_id,
            source_slot_id=task.source.source_slot_id,
            slot_generation=task.source.slot_generation,
            completed_ns=time.monotonic_ns(),
            source_session_id=task.source.source_session_id,
            pool_generation=task.source.pool_generation,
            instance_session_id=self._client.instance_session_id,
            coordinator_session_id=self._client.coordinator_session_id,
        )
        unconfirmed = getattr(self, "_unconfirmed_direct", None)
        if unconfirmed and task.transfer_id in unconfirmed:
            self._held_direct_source_done.setdefault(task.transfer_id, []).append(
                message
            )
            return
        self._queue_source_done(message)

    def _queue_source_done(self, message) -> None:
        if self.batch_source_read_done:
            self._source_done_pending.append(message)
        else:
            from dt_protocol import SOURCE_READ_DONE, source_read_done_id

            self._delivery_outbox.add(
                SOURCE_READ_DONE, source_read_done_id(message), message
            )
            self._flush_reliable_outbox()

    def _flush_source_done(self) -> None:
        if self.batch_source_read_done and self._source_done_pending:
            from dt_protocol import (
                PROTOCOL_VERSION,
                SOURCE_READ_DONE,
                SourceReadDoneBatch,
                source_read_done_id,
            )

            entries = tuple(self._source_done_pending)
            next_sequence = self._source_done_sequence + 1
            message = SourceReadDoneBatch(
                protocol_version=PROTOCOL_VERSION,
                batch_id=(
                    f"{self.instance_id}:{self._client.instance_session_id}:"
                    f"source-done:{next_sequence}"
                ),
                draft_instance_id=self.instance_id,
                entries=entries,
                sent_ns=time.monotonic_ns(),
            )
            self._source_done_pending.clear()
            self._source_done_sequence = next_sequence
            entry_ids = {source_read_done_id(entry) for entry in message.entries}
            self._source_done_batch_remaining[message.batch_id] = entry_ids
            for entry_id in entry_ids:
                self._source_done_batch_by_entry[entry_id] = message.batch_id
            self._delivery_outbox.add(
                SOURCE_READ_DONE, source_read_done_id(message), message
            )
        self._flush_reliable_outbox()

    def _ack_source_done(self, message_id: str) -> None:
        """Retire a direct completion or one entry of a batched completion."""

        batch_id = self._source_done_batch_by_entry.pop(message_id, None)
        if batch_id is None:
            self._delivery_outbox.ack("source_read_done", message_id)
            return
        remaining = self._source_done_batch_remaining.get(batch_id)
        if remaining is None:
            return
        remaining.discard(message_id)
        if remaining:
            return
        self._source_done_batch_remaining.pop(batch_id, None)
        self._delivery_outbox.ack("source_read_done", batch_id)

    def _handle_source_done_batch_ack(self, message) -> bool:
        """Apply only an ACK fenced to this live Draft and Coordinator epoch."""

        if message.draft_instance_id != self.instance_id:
            raise ValueError("SourceReadDoneBatchAck belongs to another Draft instance")
        if (
            message.instance_session_id != self._client.instance_session_id
            or message.coordinator_session_id != self._client.coordinator_session_id
        ):
            logger.warning(
                "ignoring stale SourceReadDoneBatchAck: batch=%s", message.batch_id
            )
            return False
        if not message.message_ids or len(set(message.message_ids)) != len(
            message.message_ids
        ):
            raise ValueError("SourceReadDoneBatchAck must contain unique message IDs")
        for message_id in message.message_ids:
            self._ack_source_done(message_id)
        return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SGLang external DT Draft engine")
    parser.add_argument("--coordinator-endpoint", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--nccl-port", type=int, default=29600)
    parser.add_argument("--max-pending-tasks", type=int, default=512)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--disable-continuous-batching", action="store_true")
    parser.add_argument("--result-slot-count", type=int, default=16)
    parser.add_argument(
        "--candidate-transport",
        choices=("mmap", "cuda-ipc"),
        default="mmap",
    )
    parser.add_argument("--data-dir", default="/dev/shm")
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--registration-timeout-s", type=float, default=10.0)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--enable-deterministic-inference", action="store_true")
    parser.add_argument("--batch-source-read-done", action="store_true")
    parser.add_argument(
        "--pinned-pivot-staging",
        action="store_true",
        help="deprecated compatibility flag; FlashMTP pivots always use CUDA IPC",
    )
    parser.add_argument("--feature-buffer-max-tokens", type=int, default=8192)
    parser.add_argument("--feature-buffer-slot-count", type=int, default=2)
    parser.add_argument("--dflash-submit-pipeline", action="store_true")
    parser.add_argument(
        "--migration-backend",
        choices=("disabled", "cuda-ipc"),
        default="disabled",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--dt-host-addr",
        default="",
        help="This Draft's own host address. A Target arena registration whose "
        "host_addr differs is cross-host: the direct ring is skipped and the "
        "hidden-pull/candidate-push legs use the RDMA backend.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    engine = FlashMTPDraftEngine(
        coordinator_endpoint=args.coordinator_endpoint,
        instance_id=args.instance_id,
        profile_path=args.profile,
        gpu_id=args.gpu_id,
        nccl_port=args.nccl_port,
        max_pending_tasks=args.max_pending_tasks,
        max_batch_size=args.max_batch_size,
        result_slot_count=args.result_slot_count,
        data_dir=args.data_dir,
        mem_fraction_static=args.mem_fraction_static,
        registration_timeout_s=args.registration_timeout_s,
        disable_cuda_graph=args.disable_cuda_graph,
        enable_deterministic_inference=args.enable_deterministic_inference,
        batch_source_read_done=args.batch_source_read_done,
        pinned_pivot_staging=args.pinned_pivot_staging,
        migration_backend=args.migration_backend,
        feature_buffer_max_tokens=args.feature_buffer_max_tokens,
        feature_buffer_slot_count=args.feature_buffer_slot_count,
        dflash_submit_pipeline=args.dflash_submit_pipeline,
        continuous_batching=not args.disable_continuous_batching,
        candidate_transport=args.candidate_transport,
        dt_host_addr=args.dt_host_addr,
    )

    def stop(_signum, _frame) -> None:
        engine.stop()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    if DRAFT_GC_MANAGED:
        import gc

        from sglang.srt.utils.common import freeze_gc

        # Later collections, full ones included, then scan only what serving
        # allocated.
        gc.collect()
        freeze_gc("DT Draft")
    try:
        engine.run()
    finally:
        engine.close()


if __name__ == "__main__":
    main()
