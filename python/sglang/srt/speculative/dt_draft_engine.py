from __future__ import annotations

import argparse
import bisect
import contextlib
import json
import logging
import os
import re
import signal
import time
import uuid
from array import array
from collections import OrderedDict, deque
from concurrent.futures import Future
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import msgspec
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
)
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.draft_worker_common import build_draft_tp_worker
from sglang.srt.speculative.flashmtp_utils import parse_flashmtp_draft_config
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)

IDLE_POLL_S = 0.0005
DIRECT_PUSH_SPIN_NS = 200_000
# Token-count buckets for the CUDA-graphed DFlash KV materializer.
DFLASH_MATERIALIZE_GRAPH_BUCKETS = (8, 16, 32, 64, 96, 128, 192, 256, 384, 512)


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
COALESCING_POLL_S = 0.0001
# Queue candidate pushes behind the proposal at launch and let the Target
# observe READY from GPU-written host records (dt_candidate_p2p).
GPU_READY_RECORD = os.environ.get("DT_GPU_READY_RECORD", "1") == "1"
PUSH_COMPLETION_RECORD = os.environ.get("DT_PUSH_COMPLETION_RECORD", "1") == "1"
DIRECT_RING = os.environ.get("DT_DIRECT_RING", "1") == "1"
RING_RETIRED_LIMIT = 65536
# cuda-ipc proposal admission bound.  The direct path carries no result-slot
# pool, so this explicit credit replaces the admission accounting the pool
# used to provide (issue #42); 64 is the validated dev operating point.
DEFAULT_DIRECT_PROPOSAL_ADMISSION_LIMIT = 64


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
_RING_RECORD_TIMEOUT_NS = (
    int(os.environ.get("DT_RING_RECORD_TIMEOUT_S", "30")) * 1_000_000_000
)
# Initial control delivery is immediate.  Retrying every scheduler tick-scale
# interval amplifies delayed ACKs under full-wave load; keep retries reliable
# but slow enough that the Coordinator can drain the original transaction.
CONTROL_RETRY_INTERVAL_NS = 1_000_000_000
DT_DRAFT_GRAPH_MEASUREMENT_MARKER = "DT Draft graph measurement evidence: "
DT_DIRECT_GPU_SOURCE_MARKER = "DT direct GPU source evidence: "
DT_LIFECYCLE_STATUS_MARKER = "DT lifecycle status: "
DT_LIFECYCLE_STATUS_INTERVAL_NS = 250_000_000
DT_DIRECT_GPU_REQUEST_EVIDENCE_IDLE_NS = 750_000_000


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
    batch_id: str
    pending: PendingDraftTask
    staging_slot: int
    row_range: tuple[int, int]
    ready_ns: int


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

    # Preserve the historical transport for partially constructed test and
    # diagnostic instances that intentionally bypass ``__init__``.
    candidate_transport = "mmap"

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
        batch_ready_threshold: int,
        batch_wait_ms: float,
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
        if not 0 < batch_ready_threshold <= max_batch_size:
            raise ValueError(
                "batch_ready_threshold must be positive and no larger than "
                "max_batch_size"
            )
        if result_slot_count <= 0:
            raise ValueError("result_slot_count must be positive")
        if batch_wait_ms < 0:
            raise ValueError("batch_wait_ms must be non-negative")
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
        self._persistent = self.profile.method in {
            "standalone",
            "dflash",
            "eagle3",
        }
        self.gpu_id = gpu_id
        self.max_pending_tasks = max_pending_tasks
        self.max_batch_size = max_batch_size
        self.batch_ready_threshold = batch_ready_threshold
        self.continuous_batching = bool(continuous_batching)
        self.batch_wait_s = batch_wait_ms / 1_000
        self.batch_source_read_done = batch_source_read_done
        self.pinned_pivot_staging = pinned_pivot_staging
        self.enable_deterministic_inference = bool(enable_deterministic_inference)
        self.migration_backend = migration_backend
        self.feature_buffer_max_tokens = int(feature_buffer_max_tokens)
        self.feature_buffer_slot_count = int(feature_buffer_slot_count)
        self.candidate_transport = candidate_transport
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
        self._proposal_req_pool_indices_host_buffers: list[torch.Tensor] = []
        self._proposal_req_pool_indices_buffers: list[torch.Tensor] = []
        self._proposal_prefix_lens_host_buffers: list[torch.Tensor] = []
        self._proposal_prefix_lens_buffers: list[torch.Tensor] = []
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
        # Attached ring source registrations: ring_id -> (ipc_handle, length).
        self._ring_source_registration: dict[int, tuple[bytes, int]] = {}
        self._ring_admission: deque[tuple[object, int, tuple[int, int]]] = deque()
        self._ring_waiting: dict[str, object] = {}
        self._ring_task_decoder = None
        self._landing_slot_base = -1
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
        # Held results released by confirmations, sent as one batch per turn.
        self._confirmed_held_results: list = []
        self._confirmed_held_sequence = 0
        # Bounded window of accepted transfer ids: a direct copy that arrives
        # after its Coordinator copy (even after retirement) must not re-run.
        self._seen_transfers: set[str] = set()
        self._seen_transfer_order: deque[str] = deque()
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
        self._proposal_req_pool_indices_host_buffers = [
            torch.empty(
                self.max_batch_size,
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(self.feature_buffer_slot_count)
        ]
        self._proposal_req_pool_indices_buffers = [
            torch.empty(
                self.max_batch_size,
                dtype=torch.int64,
                device=self.device,
            )
            for _ in range(self.feature_buffer_slot_count)
        ]
        self._proposal_prefix_lens_host_buffers = [
            torch.empty(
                self.max_batch_size,
                dtype=torch.int64,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(self.feature_buffer_slot_count)
        ]
        self._proposal_prefix_lens_buffers = [
            torch.empty(
                self.max_batch_size,
                dtype=torch.int64,
                device=self.device,
            )
            for _ in range(self.feature_buffer_slot_count)
        ]
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
        if self.dflash_submit_pipeline:
            self._candidate_result_host_buffers = [
                torch.empty(
                    (self.max_batch_size, self.profile.block_size),
                    dtype=torch.int32,
                    device="cpu",
                    pin_memory=True,
                )
                for _ in range(self.feature_buffer_slot_count)
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

    def run(self) -> None:
        logger.info("DT Draft engine serving")
        observer = None
        if os.environ.get("DT_OVERLAP_FIX_OBSERVER_DIR"):
            from overlap_observer import create

            observer = create(self.instance_id, "draft", 0, 1)
        self._overlap_fix_observer = observer
        while not self._stop:
            if observer is not None:
                observer.checkpoint()
            self._poll_messages()
            if self._candidate_push_transport is not None:
                self._candidate_push_transport.progress_completions()
                if self._profile_active:
                    for acknowledgement in (
                        self._candidate_push_transport.pop_acknowledgements()
                    ):
                        profile_emit(
                            "draft_candidate_notice_ack",
                            draft_instance=self.instance_id,
                            **acknowledgement,
                        )
            if self._stop:
                break
            if self._ring_worker is not None:
                self._poll_direct_rings()
            self._flush_confirmed_results()
            if self._uses_direct_gpu_source:
                self._poll_feature_batches()
                if self._stop:
                    break
                # Input preparation is independent of final compute
                # membership. Fill at most one free bounded arena per turn.
                preparation = self._take_batch()
                if preparation:
                    self._execute_batch(preparation)
                if self._ring_worker is not None:
                    self._poll_direct_rings()
                ready = self._take_input_ready_batch()
                if ready:
                    self._execute_input_ready_batch(ready)
            elif self._draft_compute_batches_inflight() < 2:
                batch = self._take_batch()
                if batch:
                    self._execute_batch(batch)

            # Publish/retire old results after attempting the next READY
            # submission. This bounded pass on every turn also prevents an
            # endless input stream from starving retirement.
            if getattr(self, "_pending_eagle3_completions", ()):
                self._poll_eagle3_completions()
            if self._uses_direct_gpu_source:
                self._poll_flashmtp_completions()
                if self._is_feature_backed:
                    self._poll_dflash_completions()
            if self._pending_candidate_pushes:
                self._poll_candidate_pushes()
            if self._ring_worker is not None:
                self._poll_direct_rings()
            if self._held_ring_pushes and self._ring_publish_at_launch:
                self._retire_tombstoned_ring_pushes()

            if (
                not self._pending
                and not self._input_ready_features
                and not self._pending_feature_batches
                and not self._ring_admission
            ):
                sleep_s = self._idle_sleep_s()
                if sleep_s > 0:
                    time.sleep(sleep_s)
            self._maybe_emit_lifecycle_status()

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
        if len(self._pending_candidate_pushes) > before:
            event = self._pending_candidate_pushes[-1].push.event
            deadline_ns = time.monotonic_ns() + DIRECT_PUSH_SPIN_NS
            while not event.query() and time.monotonic_ns() < deadline_ns:
                pass
            self._poll_candidate_pushes()

    def _queue_early_candidate_push(
        self, tasks, candidates: torch.Tensor | None, ready_event
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
                queued_at_launch=True,
            )

    def _open_direct_ring(self, registration) -> None:
        """Create this Draft's ring for one Target session (same host only)."""

        from sglang.srt.speculative.dt_direct_ring import (
            RING_CAPACITY,
            DraftRingEndpoint,
            DraftRingWorker,
            ring_path,
        )

        if (
            not self._persistent
            or self.candidate_transport != "cuda-ipc"
            or self._candidate_push_transport is None
            or registration.target_session_id in self._ring_endpoints
        ):
            return
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
                # Every per-staging-slot scratch list (proposal metadata, pinned
                # candidate rows, ...) gets entries for the landing slots too.
                for name, value in list(vars(self).items()):
                    if (
                        name == "_feature_staging_buffers"
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
            )
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

    def _poll_direct_rings(self) -> None:
        """Attach registered rings, take worker-completed entries, admit them."""

        from dt_protocol import DraftTask, DraftTaskKind
        from sglang.srt.speculative.dt_direct_ring import RingRecordV2, rid_hash

        worker = self._ring_worker
        worker.check()
        decoder = self._ring_task_decoder
        if decoder is None:
            decoder = self._ring_task_decoder = msgspec.msgpack.Decoder(DraftTask)
        for endpoint in self._ring_endpoints.values():
            if not endpoint.attached:
                registration = endpoint.registration()
                if registration is None:
                    continue
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
                )
                endpoint.mark_serving(
                    pool_generation=pool_generation,
                    source_session_id=source_session_id,
                )
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
        self._admit_ring_tasks()
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

    def _accept_ring_task(self, endpoint, entry, task, task_kinds) -> None:
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
            self._publish_source_done(task)
            self._ring_admission.append((task, slot, rows))
        else:
            self._pending.append(task)
        profile_emit(
            "draft_ring_take",
            draft_instance=self.instance_id,
            transfer_id=transfer_id,
            request_id=task.request_id,
            round_id=task.round_id,
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
            self._publish_source_done(waiting or task)
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
                    coordinator_result=report,
                )
            )
        self._cancelled_ids.discard(transfer_id)
        self._received_ns_by_transfer.pop(transfer_id, None)
        if waiting is None:
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
            if not rid or rid_hash(rid) != req.rid_hash:
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
            self._unconfirmed_direct[transfer_id] = task.deadline_ns
            self._seen_transfers.add(transfer_id)
            self._seen_transfer_order.append(transfer_id)
            if len(self._seen_transfer_order) > 65536:
                self._seen_transfers.discard(self._seen_transfer_order.popleft())
            self._pending_ids.add(transfer_id)
            self._ring_pending_ids.add(transfer_id)
            self._received_ns_by_transfer[transfer_id] = received_ns
            self._ring_record_positions[transfer_id] = (endpoint, position)
            first = record.landing_offset // row_bytes + req.token_offset
            rows = (first, first + req.token_count)
            refcounts = self._feature_staging_refcounts
            refcounts[slot] = refcounts.get(slot, 0) + 1
            # The worker's INPUT_READY write followed the copy on its stream.
            self._publish_source_done(task)
            self._ring_admission.append((task, slot, rows))
            profile_emit(
                "draft_ring_take",
                draft_instance=self.instance_id,
                transfer_id=transfer_id,
                request_id=task.request_id,
                round_id=task.round_id,
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
        self._publish_source_done(waiting or task)
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
                    coordinator_result=report,
                )
            )
        self._cancelled_ids.discard(transfer_id)
        self._received_ns_by_transfer.pop(transfer_id, None)
        if waiting is None:
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
        available_rows = self._draft_model_runner.req_to_token_pool.available_size()
        available_kv = self._draft_model_runner.token_to_kv_pool_allocator.available_size()
        for owner in self._pending_feature_batches:
            for item in owner.pending:
                rows, tokens = self._persistent_task_allocation(item.task)
                available_rows -= rows
                available_kv -= max(tokens, 0)
        for entry in self._input_ready_features:
            rows, tokens = self._persistent_task_allocation(entry.pending.task)
            available_rows -= rows
            available_kv -= max(tokens, 0)
        now_ns = time.monotonic_ns()
        from dt_protocol import DraftTaskKind

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
                break
            if task.requires_proposal and self._proposal_admission_blocked():
                break
            queue.popleft()
            available_rows -= row_cost
            available_kv -= max(kv_cost, 0)
            self._received_ns_by_transfer.pop(task.transfer_id, None)
            self._input_ready_features.append(
                InputReadyFeature(
                    batch_id=f"ring:{task.transfer_id}",
                    pending=PendingDraftTask(task=task, taken_ns=now_ns),
                    staging_slot=slot,
                    row_range=rows,
                    ready_ns=now_ns,
                )
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
        self._ring_worker = None
        rc = worker.destroy()
        if rc:
            raise RuntimeError(f"direct ring worker stream drain failed: {rc}")
        self._landing_free_pending.clear()
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
        if count > 2:
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
            if getattr(self, "_pending_candidate_pushes", ()):
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
        return (
            source.source_slot_id == other.source_slot_id
            and source.slot_generation == other.slot_generation
            and source.nbytes == other.nbytes
            and source.byte_offset == other.byte_offset
            and source.allocation_offset == other.allocation_offset
            and source.source_session_id == other.source_session_id
            and source.pool_generation == other.pool_generation
        )

    def _confirm_direct_task(self, task) -> None:
        """The Coordinator accepted a directly received task: release its holds."""
        direct = self._direct_tasks_by_transfer.get(task.transfer_id)
        if direct is not None and direct != task:
            if not (
                direct.direct_ring_seq
                and direct.candidate_destination is None
                and self._ring_record_task_matches(direct, task)
            ):
                raise ValueError("Coordinator task differs from its direct dispatch")
            # A §7.1 ring-record task was reconstructed without destination or
            # scheduling metadata; adopt the Coordinator copy for both.
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
        held_push_ids = {
            task.transfer_id
            for tasks, _candidates, _slot in getattr(self, "_held_ring_pushes", ())
            for task in tasks
        }
        expired_held_push = False
        now_ns = time.monotonic_ns()
        for transfer_id, deadline_ns in tuple(self._unconfirmed_direct.items()):
            if now_ns < deadline_ns + UNCONFIRMED_DIRECT_GRACE_NS:
                continue
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
                if DIRECT_RING:
                    self._open_direct_ring(registration)
        unconfirmed = getattr(self, "_unconfirmed_direct", None)
        if unconfirmed is None:
            unconfirmed = self._unconfirmed_direct = {}
        seen = getattr(self, "_seen_transfers", None)
        for task in message.entries:
            if task.transfer_id in self._pending_ids:
                if task.transfer_id in unconfirmed:
                    self._confirm_direct_task(task)
                    accepted = True
                continue
            if getattr(task, "direct_ring_seq", 0):
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
            active_feature_tasks = sum(
                len(batch.pending) for batch in self._pending_feature_batches
            ) + len(getattr(self, "_input_ready_features", ())) + sum(
                len(batch.pending)
                for batch in getattr(self, "_pending_flashmtp_completions", ())
            ) + sum(
                len(batch.pending) for batch in self._pending_dflash_completions
            ) + sum(
                len(batch.pending) for batch in self._pending_eagle3_completions
            )
            # Unconfirmed ring rounds are bounded by owned KV, not by the
            # Coordinator queue this check protects; Coordinator copies
            # waiting for their ring copy are Coordinator-charged.
            active_feature_tasks += len(getattr(self, "_ring_admission", ())) + len(
                getattr(self, "_ring_waiting", ())
            )
            if (
                len(self._pending) + active_feature_tasks - len(unconfirmed)
                >= self.max_pending_tasks
            ):
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
        if (
            not getattr(self, "continuous_batching", False)
            and len(self._pending) < self.batch_ready_threshold
            and self.batch_wait_s > 0
        ):
            oldest = self._pending[0]
            received_ns = self._received_ns_by_transfer.get(oldest.transfer_id)
            if (
                received_ns is not None
                and (time.monotonic_ns() - received_ns) / 1_000_000_000
                < self.batch_wait_s
            ):
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
                if (
                    next_task.requires_proposal
                    and not reserves_result_slot
                    and not result_slot_available
                ):
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
            reserves_result_slot = reserves_result_slot or bool(task.requires_proposal)
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
                self._uses_direct_gpu_source
                and next_task.requires_proposal
                and not reserves_result_slot
                and not result_slot_available
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

    def _idle_sleep_s(self) -> float:
        if (
            getattr(self, "continuous_batching", False)
            or not self._pending
            or (
                self._uses_direct_gpu_source
                and self._pending[0].requires_proposal
                and self._proposal_admission_blocked()
            )
            or (
                self._uses_direct_gpu_source
                and not self._free_feature_staging_slots
            )
            or (
                not self._is_feature_backed
                and self._proposal_wave_buffer_exhausted()
            )
            or len(self._pending) >= self.batch_ready_threshold
            or self.batch_wait_s == 0
        ):
            return IDLE_POLL_S
        received_ns = self._received_ns_by_transfer.get(self._pending[0].transfer_id)
        if received_ns is None:
            return 0
        age_s = (time.monotonic_ns() - received_ns) / 1_000_000_000
        remaining_s = max(0.0, self.batch_wait_s - age_s)
        return min(COALESCING_POLL_S, remaining_s)

    def _dflash_admission_reserved(self) -> int:
        """Submitted-but-unpublished proposal work holding an admission unit.

        Ring admission, INPUT_READY proposal entries, and pending
        feature/FlashMTP/DFlash/EAGLE3 completions each reserve one unit until
        their wave publishes (the same terms the result-pool accounting used).
        """

        return sum(
            any(item.task.requires_proposal for item in batch.pending)
            for batch in self._pending_feature_batches
        ) + sum(
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
        self._direct_proposal_admission_inflight += 1

    def _release_direct_proposal_admission(self) -> None:
        """Return one wave's credit; every holder must release exactly once."""
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
            return self._available_direct_proposal_admission() <= 0
        return self._available_dflash_result_slots() <= 0

    def _proposal_wave_buffer_exhausted(self) -> bool:
        """Raw per-wave publish buffer exhaustion (no reservation accounting)."""
        if self.candidate_transport == "cuda-ipc":
            return (
                self._direct_proposal_admission_inflight
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
        for source, destination in zip(sources, destinations, strict=True):
            try:
                future = self._feature_pull_transport.submit(
                    source=source,
                    destination=destination,
                )
            except Exception as exc:
                future = Future()
                future.set_exception(exc)
            futures.append(future)
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
        busy = getattr(self, "_feature_staging_compute_busy", None)
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
        retained = deque()
        capacity_changed = False
        while self._pending_feature_batches:
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
                        batch_id=feature_batch.batch_id,
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
        inflight_before = self._draft_compute_batches_inflight()
        waiting_before = tuple(
            entry.pending.task.transfer_id for entry in self._input_ready_features
        )
        if not self._input_ready_features or inflight_before >= 2:
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
            if (
                entry.staging_slot in self._feature_staging_compute_busy
                or len(selected) >= self.max_batch_size
            ):
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
        selection_finished_ns = time.monotonic_ns() if profile_active else 0
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
        inflight_before = self._draft_compute_batches_inflight()
        entry_slots = tuple(entry.staging_slot for entry in ready)
        staging_slots = tuple(dict.fromkeys(entry_slots))
        staging_slot_counts = tuple(
            entry_slots.count(slot) for slot in staging_slots
        )
        if any(
            slot in self._feature_staging_compute_busy for slot in staging_slots
        ):
            raise RuntimeError("feature staging slot already has submitted compute")
        self._feature_staging_compute_busy.update(staging_slots)
        landing_slot_base = getattr(self, "_landing_slot_base", -1)
        for slot in staging_slots:
            if 0 <= landing_slot_base <= slot:
                # New ring arrivals land in another buffer instead of waiting
                # behind this batch's compute ownership.
                self._ring_worker.seal(slot - landing_slot_base)
        pending = [entry.pending for entry in ready]
        row_ranges = tuple(entry.row_range for entry in ready)
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
                )
            if profile_active:
                submit_returned_ns = time.monotonic_ns()
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
                    staging_slots=staging_slots,
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
    ) -> None:
        if self._pending_dflash_completions and not self.dflash_submit_pipeline:
            raise RuntimeError(
                "DFLASH materialization submission must wait for the prior "
                "completion event"
            )
        # staging_slots is per entry, parallel to row_ranges.  Temporary H2D
        # staging (proposal metadata, candidate rows) is indexed by slot; any
        # slot this batch owns is valid, so use the first.
        slots_used = tuple(dict.fromkeys(staging_slots))
        staging_slot_counts = tuple(
            staging_slots.count(slot) for slot in slots_used
        )
        metadata_slot = slots_used[0]
        prior_unfinished_proposal_count = 0
        prior_candidate_copy_pending_count = 0
        for completion in self._pending_dflash_completions:
            if not completion.proposal_done_event.query():
                prior_unfinished_proposal_count += 1
            elif not completion.completion_event.query():
                prior_candidate_copy_pending_count += 1
        launch_started_ns = time.monotonic_ns()
        tasks = [item.task for item in pending]
        if getattr(self, "continuous_batching", False) and len({bool(task.requires_proposal) for task in tasks}) > 1:
            raise RuntimeError("DFLASH frozen batch mixed proposal boundaries")
        states = []
        cache_locs = []
        reconcile_prepare_phase_ns = [0] * 6 if profile_enabled() else None
        grouped_discarded = self._free_dflash_reconcile_suffixes(
            tasks, reconcile_prepare_phase_ns
        )
        grouped_rows = self._allocate_dflash_reconcile_rows(
            tasks, grouped_discarded, reconcile_prepare_phase_ns
        )
        grouped_cache_locs = None if grouped_rows is None else grouped_rows[0]
        batch_prefix_refresh = len(
            {task.draft_kv_handle for task in tasks}
        ) == len(tasks)
        prior_prefix_indices = []
        for task_index, task in enumerate(tasks):
            prior_state = self._persistent_requests.get(task.draft_kv_handle)
            prior_prefix_indices.append(
                None
                if not batch_prefix_refresh or prior_state is None
                else prior_state.req.prefix_indices
            )
            state, cache_loc = self._prepare_dflash_task(
                task,
                reconcile_prepare_phase_ns,
                suffix_already_freed=(
                    grouped_discarded is not None
                    and task.draft_kv_handle in grouped_discarded
                ),
                discarded_tokens_override=(
                    None
                    if grouped_discarded is None
                    else grouped_discarded.get(task.draft_kv_handle)
                ),
                refresh_prefix_indices=not batch_prefix_refresh,
                cache_loc_override=(
                    None
                    if grouped_cache_locs is None
                    else grouped_cache_locs[task_index]
                ),
                row_already_written=grouped_cache_locs is not None,
            )
            states.append(state)
            cache_locs.append(cache_loc)
        # Preserve Req's committed-prefix view with one GPU concatenation for
        # the whole batch.  Proposal suffixes never extend prefix_indices; the
        # prior tensor therefore ends exactly at feature_token_offset.
        if batch_prefix_refresh:
            prefix_refresh_started_ns = (
                time.monotonic_ns() if reconcile_prepare_phase_ns is not None else 0
            )
            prefix_parts = []
            prefix_lengths = []
            for task, prior_prefix, cache_loc in zip(
                tasks, prior_prefix_indices, cache_locs, strict=True
            ):
                if task.feature_token_offset:
                    if (
                        prior_prefix is None
                        or int(prior_prefix.numel()) != task.feature_token_offset
                    ):
                        raise RuntimeError(
                            "DFLASH committed prefix indices differ from feature offset"
                        )
                    prefix_parts.append(prior_prefix)
                prefix_parts.append(cache_loc)
                prefix_lengths.append(task.committed_kv_len)
            packed_prefix_indices = torch.cat(prefix_parts, dim=0)
            for state, prefix_indices in zip(
                states,
                torch.split(packed_prefix_indices, prefix_lengths),
                strict=True,
            ):
                state.req.prefix_indices = prefix_indices
            if reconcile_prepare_phase_ns is not None:
                reconcile_prepare_phase_ns[4] += (
                    time.monotonic_ns() - prefix_refresh_started_ns
                )
        if grouped_rows is None:
            packed_cache_locs = torch.cat(cache_locs, dim=0)
            async_metadata = getattr(self, "continuous_batching", False)
            positions = torch.tensor(
                [
                    position
                    for task in tasks
                    for position in range(
                        task.feature_token_offset, task.committed_kv_len
                    )
                ],
                dtype=torch.int64,
                device="cpu" if async_metadata else self.device,
                pin_memory=async_metadata,
            ).to(self.device, non_blocking=async_metadata)
        else:
            _grouped_cache_locs, packed_cache_locs, positions = grouped_rows
        tasks_prepared_ns = time.monotonic_ns()
        hidden_rows = [
            self._feature_staging_buffers[slot][start:end]
            for slot, (start, end) in zip(staging_slots, row_ranges, strict=True)
        ]
        if len(slots_used) == 1 and all(
            row_ranges[index][1] == row_ranges[index + 1][0]
            for index in range(len(row_ranges) - 1)
        ):
            # The contiguous-range view only exists within one slot's buffer.
            target_hidden = self._feature_staging_buffers[slots_used[0]][
                row_ranges[0][0] : row_ranges[-1][1]
            ]
        else:
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
        proposal_seq_lens_cpu = None
        proposal_bonus_tokens = None
        if proposal_indices:
            # Stage CPU-derived proposal metadata through per-slot pinned host
            # buffers.  The corresponding nonblocking H2D copies may be queued
            # behind a prior proposal without synchronizing the scheduler
            # thread, and the batch's slot ownership retains the buffers until
            # completion.  Committed lengths are already authoritative CPU
            # state, so seq_lens_cpu never needs a GPU round-trip.
            committed_lens = [state.valid_committed_len for state in proposal_states]
            proposal_count = len(proposal_states)
            req_pool_indices_host = self._proposal_req_pool_indices_host_buffers[
                metadata_slot
            ][:proposal_count]
            req_pool_indices_host.copy_(
                torch.tensor(
                    [state.req.req_pool_idx for state in proposal_states],
                    dtype=torch.int64,
                    device="cpu",
                )
            )
            proposal_req_pool_indices = self._proposal_req_pool_indices_buffers[
                metadata_slot
            ][:proposal_count]
            proposal_req_pool_indices.copy_(req_pool_indices_host, non_blocking=True)
            prefix_lens_host = self._proposal_prefix_lens_host_buffers[metadata_slot][
                :proposal_count
            ]
            prefix_lens_host.copy_(
                torch.tensor(committed_lens, dtype=torch.int64, device="cpu")
            )
            proposal_prefix_lens = self._proposal_prefix_lens_buffers[metadata_slot][
                :proposal_count
            ]
            proposal_prefix_lens.copy_(prefix_lens_host, non_blocking=True)
            proposal_seq_lens_cpu = torch.tensor(
                [length + int(self.profile.block_size) for length in committed_lens],
                dtype=torch.int32,
                device="cpu",
            )
            bonus_tokens_host = self._proposal_bonus_tokens_host_buffers[metadata_slot][
                :proposal_count
            ]
            bonus_tokens_host.copy_(
                torch.tensor(
                    [task.next_token_id for task in proposal_tasks],
                    dtype=torch.int64,
                    device="cpu",
                )
            )
            proposal_bonus_tokens = self._proposal_bonus_tokens_buffers[metadata_slot][
                :proposal_count
            ]
            proposal_bonus_tokens.copy_(bonus_tokens_host, non_blocking=True)
        proposal_metadata_ready_ns = time.monotonic_ns()

        materialize_start.record(torch.cuda.current_stream(self.device))
        self._dflash_materializer.append(
            target_hidden=target_hidden,
            cache_loc=packed_cache_locs,
            positions=positions,
        )
        materialize_done.record(torch.cuda.current_stream(self.device))
        materialize_submitted_ns = time.monotonic_ns()

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
                seq_lens_cpu=proposal_seq_lens_cpu,
                bonus_tokens=proposal_bonus_tokens,
            )
        if candidates is not None and self.candidate_transport == "cuda-ipc":
            # Preserve graph-owned proposals before recording the completion
            # event which makes this completion eligible for another replay.
            candidates = candidates.clone()
        proposal_done.record(torch.cuda.current_stream(self.device))
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
        )
        self._register_dflash_completion(
            completion=completion,
            copy_stream=copy_stream,
        )
        if copy_stream is None and proposal_tasks:
            self._queue_early_candidate_push(
                proposal_tasks, candidates, completion.completion_event
            )

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

    def _free_dflash_reconcile_suffixes(
        self, tasks, reconcile_prepare_phase_ns: list[int] | None = None
    ) -> dict[str, int] | None:
        """Batch-free unique reconcile suffixes before any page-table overwrite."""

        from dt_protocol import DraftTaskKind

        if not tasks or any(task.kind != DraftTaskKind.RECONCILE for task in tasks):
            return None
        handles = [task.draft_kv_handle for task in tasks]
        if len(set(handles)) != len(handles):
            return None

        validated = []
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
            validated.append((state, allocated_len))

        phase_started_ns = (
            time.monotonic_ns() if reconcile_prepare_phase_ns is not None else 0
        )
        allocator = self._draft_model_runner.token_to_kv_pool_allocator
        allocator.free_group_begin()
        try:
            for state, allocated_len in validated:
                self._free_persistent_kv_suffix(
                    state.req, state.valid_committed_len, allocated_len
                )
            allocator.free_group_end()
        except Exception as exc:
            # The allocator's ownership is indeterminate if a grouped update
            # fails.  The outer feature-batch guard must quarantine the rows
            # and stop instead of attempting a second free.
            setattr(exc, "resources_quiescent", False)
            raise
        discarded = {}
        for state, allocated_len in validated:
            discarded[state.draft_kv_handle] = (
                allocated_len - state.valid_committed_len
            )
            state.req.kv.kv_allocated_len = state.valid_committed_len
        if reconcile_prepare_phase_ns is not None:
            reconcile_prepare_phase_ns[1] += time.monotonic_ns() - phase_started_ns
        return discarded

    def _allocate_dflash_reconcile_rows(
        self,
        tasks,
        grouped_discarded: dict[str, int] | None,
        reconcile_prepare_phase_ns: list[int] | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor] | None:
        """Allocate and install one packed set of DFlash reconcile KV rows.

        The grouped suffix release has already shortened every request's
        ownership boundary.  One allocator call and one page-table write now
        establish the replacement ownership for the whole wave.  The returned
        packed cache and position tensors are also the materializer inputs, so
        the caller does not repack the same rows a second time.
        """

        if grouped_discarded is None:
            return None
        req_pool = self._draft_model_runner.req_to_token_pool
        allocator = self._draft_model_runner.token_to_kv_pool_allocator
        validated = []
        token_counts = []
        for task in tasks:
            state = self._persistent_requests.get(task.draft_kv_handle)
            if state is None or task.draft_kv_handle not in grouped_discarded:
                raise RuntimeError("DFLASH grouped allocation lost reconcile state")
            req = state.req
            if req.kv is None or req.req_pool_idx is None:
                raise RuntimeError("DFLASH grouped allocation lost KV ownership")
            token_count = int(task.feature_token_count)
            if (
                token_count <= 0
                or task.committed_kv_len - task.feature_token_offset != token_count
            ):
                raise ValueError("DFLASH grouped allocation token range mismatch")
            if req.kv.kv_allocated_len != task.feature_token_offset:
                raise RuntimeError(
                    "DFLASH grouped allocation did not observe the released suffix"
                )
            validated.append((task, req))
            token_counts.append(token_count)

        phase_started_ns = (
            time.monotonic_ns() if reconcile_prepare_phase_ns is not None else 0
        )
        packed_cache_locs = allocator.alloc(sum(token_counts))
        if packed_cache_locs is None:
            raise RuntimeError("DFLASH grouped reconcile KV-page OOM")
        if reconcile_prepare_phase_ns is not None:
            phase_completed_ns = time.monotonic_ns()
            reconcile_prepare_phase_ns[2] += phase_completed_ns - phase_started_ns
            phase_started_ns = phase_completed_ns
        try:
            cache_locs = list(torch.split(packed_cache_locs, token_counts))
            # Frozen metadata uploads precede the existing current-stream page-table
            # write; pinned allocation lifetime is retained by the async copy.
            async_metadata = getattr(self, "continuous_batching", False)
            row_indices = torch.tensor(
                [
                    req.req_pool_idx
                    for (_task, req), token_count in zip(
                        validated, token_counts, strict=True
                    )
                    for _ in range(token_count)
                ],
                dtype=torch.int64,
                device="cpu" if async_metadata else self.device,
                pin_memory=async_metadata,
            ).to(self.device, non_blocking=async_metadata)
            positions = torch.tensor(
                [
                    position
                    for task, _req in validated
                    for position in range(
                        task.feature_token_offset, task.committed_kv_len
                    )
                ],
                dtype=torch.int64,
                device="cpu" if async_metadata else self.device,
                pin_memory=async_metadata,
            ).to(self.device, non_blocking=async_metadata)
            req_pool.req_to_token[row_indices, positions] = packed_cache_locs.to(
                req_pool.req_to_token.dtype
            )
        except Exception as exc:
            try:
                allocator.free(packed_cache_locs)
            except Exception:
                setattr(exc, "resources_quiescent", False)
            raise
        # The page-table write and any later cleanup use the same CUDA stream.
        # Publish all CPU ownership boundaries only after the write is queued;
        # from here, outer failure cleanup owns every packed replacement page.
        for task, req in validated:
            req.kv.kv_allocated_len = task.committed_kv_len
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
                "DT DFLASH grouped reconcile allocation observed: batches=%d rows=%d",
                grouped_batches,
                grouped_rows,
            )
        if reconcile_prepare_phase_ns is not None:
            reconcile_prepare_phase_ns[3] += (
                time.monotonic_ns() - phase_started_ns
            )
        return cache_locs, packed_cache_locs, positions

    def _prepare_dflash_task(
        self,
        task,
        reconcile_prepare_phase_ns: list[int] | None = None,
        *,
        suffix_already_freed: bool = False,
        discarded_tokens_override: int | None = None,
        refresh_prefix_indices: bool = True,
        cache_loc_override: torch.Tensor | None = None,
        row_already_written: bool = False,
    ) -> tuple[PersistentDraftRequest, torch.Tensor]:
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
            if refresh_prefix_indices:
                req.prefix_indices = cache_loc.to(torch.int64)
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
            return state, cache_loc.to(torch.int64)

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
        discarded_tokens = (
            allocated_len - state.valid_committed_len
            if discarded_tokens_override is None
            else discarded_tokens_override
        )
        if reconcile_prepare_phase_ns is not None:
            phase_completed_ns = time.monotonic_ns()
            reconcile_prepare_phase_ns[0] += phase_completed_ns - phase_started_ns
            phase_started_ns = phase_completed_ns
        if not suffix_already_freed:
            self._free_persistent_kv_suffix(
                req, state.valid_committed_len, allocated_len
            )
        if reconcile_prepare_phase_ns is not None:
            phase_completed_ns = time.monotonic_ns()
            reconcile_prepare_phase_ns[1] += phase_completed_ns - phase_started_ns
            phase_started_ns = phase_completed_ns
        if cache_loc_override is None:
            req.kv.kv_allocated_len = state.valid_committed_len
            cache_loc = allocator.alloc(token_count)
            if cache_loc is None:
                raise RuntimeError("DFLASH reconcile KV-page OOM")
        else:
            if not suffix_already_freed or not row_already_written:
                raise RuntimeError(
                    "DFLASH grouped allocation requires released suffix and row ownership"
                )
            cache_loc = cache_loc_override
            if int(cache_loc.numel()) != token_count:
                raise RuntimeError("DFLASH grouped allocation length mismatch")
            if req.kv.kv_allocated_len != task.committed_kv_len:
                raise RuntimeError("DFLASH grouped allocation ownership mismatch")
        if reconcile_prepare_phase_ns is not None:
            phase_completed_ns = time.monotonic_ns()
            reconcile_prepare_phase_ns[2] += phase_completed_ns - phase_started_ns
            phase_started_ns = phase_completed_ns
        if not row_already_written:
            req_pool.req_to_token[
                req.req_pool_idx, task.feature_token_offset : task.committed_kv_len
            ] = cache_loc
        if reconcile_prepare_phase_ns is not None:
            phase_completed_ns = time.monotonic_ns()
            reconcile_prepare_phase_ns[3] += phase_completed_ns - phase_started_ns
            phase_started_ns = phase_completed_ns
        req.kv.kv_allocated_len = task.committed_kv_len
        req.kv_committed_len = task.committed_kv_len
        if refresh_prefix_indices:
            req.prefix_indices = req_pool.req_to_token[
                req.req_pool_idx, : task.committed_kv_len
            ].to(torch.int64)
        if reconcile_prepare_phase_ns is not None:
            phase_completed_ns = time.monotonic_ns()
            reconcile_prepare_phase_ns[4] += phase_completed_ns - phase_started_ns
            phase_started_ns = phase_completed_ns
        req.output_ids.extend(task.committed_token_ids)
        req._refresh_fill_ids()
        req.set_extend_range(task.feature_token_offset, task.committed_kv_len)
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
        return state, cache_loc.to(torch.int64)

    def _run_dflash_proposal(
        self,
        *,
        states: list[PersistentDraftRequest],
        req_pool_indices: torch.Tensor,
        prefix_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        bonus_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, str, str | None]:
        batch_size = len(states)
        block_size = int(self.profile.block_size)
        req_pool = self._draft_model_runner.req_to_token_pool
        allocator = self._draft_model_runner.token_to_kv_pool_allocator
        out_cache_loc = allocator.alloc(batch_size * block_size)
        if out_cache_loc is None:
            raise RuntimeError("DFLASH proposal KV-page OOM")
        try:
            out_cache_loc_2d = out_cache_loc.view(batch_size, block_size)
            positions = prefix_lens.unsqueeze(1) + self._block_pos_offsets
            req_rows = req_pool_indices.unsqueeze(1).expand_as(positions)
            req_pool.req_to_token[req_rows, positions] = out_cache_loc_2d.to(
                req_pool.req_to_token.dtype
            )
            for state in states:
                end = state.valid_committed_len + block_size
                state.req.kv.kv_allocated_len = end
                state.req.kv_committed_len = state.valid_committed_len
                state.proposal_base_len = state.valid_committed_len

            block_ids = torch.full(
                (batch_size, block_size),
                self._mask_token_id,
                dtype=torch.int64,
                device=self.device,
            )
            block_ids[:, 0].copy_(bonus_tokens)
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
            # No successful completion event will own this allocation.  Fence
            # exceptional GPU work before making the pages reusable; if that
            # fence itself fails, report the resource as unquiesced so the
            # caller quarantines every affected request instead.
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
            allocator.free(out_cache_loc)
            for state in states:
                if state.req.kv is not None:
                    state.req.kv.kv_allocated_len = state.valid_committed_len
                state.req.kv_committed_len = state.valid_committed_len
                state.proposal_base_len = state.valid_committed_len
            raise

    def _poll_dflash_completions(self) -> None:
        if self._fatal_gpu_quarantine_active():
            return
        capacity_changed = False
        while self._pending_dflash_completions:
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
            tasks = [item.task for item in completion.pending]
            completed_ns = time.monotonic_ns()
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
            for handle in {task.draft_kv_handle for task in tasks}:
                self._drain_deferred_persistent_release(handle)
            capacity_changed = True
        if capacity_changed:
            self._publish_capacity()

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
        if getattr(self, "_input_ready_features", ()) or getattr(
            self, "_feature_staging_compute_busy", ()
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
                task.transfer_id in self._unconfirmed_direct for task in live_tasks
            ):
                # Still waiting on a confirmation; nothing leaves the hold,
                # so no cancelled identity is retired yet either.
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

    def _publish_direct_wave(self, tasks, candidates, slot_id) -> None:
        """Queue one cuda-ipc wave's push (or hold it for confirmation).

        The caller owns the wave's proposal-admission credit; it leaves with
        the held entry or the queued push.  A raise means no transfer happened
        and the caller must release the credit.
        """

        if self._candidate_push_transport is None:
            raise RuntimeError("direct candidate push transport is unavailable")
        if not isinstance(candidates, torch.Tensor):
            raise TypeError("direct candidate return requires GPU tensor rows")
        push_candidates = candidates.contiguous()
        # §7.1 record tasks adopt their Coordinator copy (destination,
        # scheduling metadata) at confirmation; hold the wave's whole push
        # while any of its tasks is still unconfirmed.  The destination
        # itself arrives earlier through the ring's destination region.
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
                self._held_ring_pushes.append((tasks, push_candidates, slot_id))
                return
            tasks = resolved
        if any(
            task.candidate_destination is None
            and task.direct_ring_seq
            and task.transfer_id in self._unconfirmed_direct
            for task in tasks
        ):
            self._held_ring_pushes.append((tasks, push_candidates, slot_id))
            return
        self._submit_candidate_push(tasks, push_candidates, result_slot=slot_id)

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
            try:
                self._publish_direct_wave(tasks, candidates, slot_id)
            except Exception:
                # The credit only leaves with a queued push or a held wave; a
                # raise before either means this call still owns it.
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

        from dt_protocol import (
            PROTOCOL_VERSION,
            DraftExecutionResultBatch,
            DraftResultEntry,
        )

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
            producer_completion_observed_ns = (
                time.monotonic_ns() if self._profile_active else 0
            )
            self._pending_candidate_pushes.popleft()
            # Target READY is driven by this direct, payload-free notice.  The
            # Coordinator result below is independent retirement/accounting.
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
            capacity_changed = True
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
        if capacity_changed:
            self._publish_capacity()

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
        } | getattr(self, "_early_pushed_ids", set())
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

        if getattr(task, "candidate_destination", None) is None:
            self._pending_ids.discard(task.transfer_id)
            getattr(self, "_ring_pending_ids", set()).discard(task.transfer_id)
        self._cancelled_ids.discard(task.transfer_id)
        self._received_ns_by_transfer.pop(task.transfer_id, None)
        self._ring_record_positions.pop(task.transfer_id, None)
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
        unconfirmed = getattr(self, "_unconfirmed_direct", None) or {}
        # Coordinator copies waiting for their ring copy are not yet
        # received: the Coordinator already counts them as in flight.
        pending_tasks = len(self._pending_ids) - sum(
            transfer_id in self._pending_ids for transfer_id in unconfirmed
        )
        self._capacity_sequence += 1
        update = DraftCapacityUpdate(
            protocol_version=PROTOCOL_VERSION,
            draft_instance_id=self.instance_id,
            pending_tasks=pending_tasks,
            received_tasks=self._received_tasks,
            capacity_sequence=self._capacity_sequence,
            max_pending_tasks=self.max_pending_tasks,
            free_result_rows=(
                self._available_direct_proposal_admission() * self.max_batch_size
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
            pending_work=pending_tasks + active_transfers,
            queue_delay_ns=self._last_queue_delay_ns,
            last_batch_size=self._last_batch_size,
            ring_pending_tasks=(lambda rpi: len(rpi) - sum(t in rpi for t in unconfirmed))(getattr(self, "_ring_pending_ids", set())),
        )
        if not self._client.update_capacity(update):
            logger.warning("Draft capacity update was locally backpressured")

    def _send_result(self, message) -> None:
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
    parser.add_argument("--batch-ready-threshold", type=int, default=16)
    parser.add_argument("--batch-wait-ms", type=float, default=0.2)
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
        batch_ready_threshold=args.batch_ready_threshold,
        batch_wait_ms=args.batch_wait_ms,
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
    )

    def stop(_signum, _frame) -> None:
        engine.stop()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        engine.run()
    finally:
        engine.close()


if __name__ == "__main__":
    main()
