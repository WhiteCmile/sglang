from __future__ import annotations

import argparse
import bisect
import json
import logging
import os
import re
import signal
import time
import uuid
from array import array
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
from dt_client import RetryOutbox
from dt_profile import cuda_sync_enabled
from dt_profile import emit as profile_emit
from dt_profile import enabled as profile_enabled
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
COALESCING_POLL_S = 0.0001
# Initial control delivery is immediate.  Retrying every scheduler tick-scale
# interval amplifies delayed ACKs under full-wave load; keep retries reliable
# but slow enough that the Coordinator can drain the original transaction.
CONTROL_RETRY_INTERVAL_NS = 1_000_000_000
DT_DRAFT_GRAPH_MEASUREMENT_MARKER = "DT Draft graph measurement evidence: "


@dataclass(slots=True)
class PendingDraftTask:
    task: object
    pivot_bytes: bytes | None
    pivot_row: int = -1


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
    pending: list[PendingDraftTask]
    staging_slot: int
    row_ranges: tuple[tuple[int, int], ...]
    futures: tuple[Future, ...]
    submitted_ns: int


@dataclass(slots=True)
class PendingDFlashCompletion:
    pending: list[PendingDraftTask]
    staging_slot: int
    completion_event: torch.cuda.Event
    materialize_start_event: torch.cuda.Event
    materialize_done_event: torch.cuda.Event
    proposal_done_event: torch.cuda.Event
    proposal_tasks: tuple[object, ...]
    candidates: torch.Tensor | None
    launched_ns: int


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
    ) -> None:
        self.draft_model = draft_model
        self.token_embedding = token_embedding
        self.device = device
        self.pivot_width = int(pivot_width)
        self.draft_block_len = int(draft_block_len)
        self.block_size = int(block_size)
        self.mask_token_id = int(mask_token_id)
        self.local_positions = local_positions
        self.batch_sizes = draft_cuda_graph_batch_sizes(max_batch_size)
        self._captured: dict[int, CapturedDraftBatch] = {}

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
        proposals = self.draft_model.sample_draft_tokens(
            block_hidden=block_hidden,
            first_prev_token_ids=bonus_tokens,
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
        return self.draft_model.sample_draft_tokens(
            block_hidden=block_hidden,
            first_prev_token_ids=bonus_tokens,
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

        self.EntryStatus = EntryStatus
        self.instance_id = instance_id
        self.profile = load_draft_profile(profile_path)
        if self.profile.method not in {"flashmtp", "standalone", "dflash"}:
            raise ValueError(
                "DT Draft engine supports FlashMTP V1, STANDALONE V2, and DFlash V2"
            )
        self._is_dflash = self.profile.method == "dflash"
        self._persistent = self.profile.method in {"standalone", "dflash"}
        self.gpu_id = gpu_id
        self.max_pending_tasks = max_pending_tasks
        self.max_batch_size = max_batch_size
        self.batch_ready_threshold = batch_ready_threshold
        self.batch_wait_s = batch_wait_ms / 1_000
        self.batch_source_read_done = batch_source_read_done
        self.pinned_pivot_staging = pinned_pivot_staging
        self.enable_deterministic_inference = bool(enable_deterministic_inference)
        self.migration_backend = migration_backend
        self.feature_buffer_max_tokens = int(feature_buffer_max_tokens)
        self.feature_buffer_slot_count = int(feature_buffer_slot_count)
        self._result_row_width = self.profile.block_size
        self._pending: deque[object] = deque()
        self._pending_ids: set[str] = set()
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
        self._migration_quiescence_probes: dict[str, Future[bool]] = {}
        self._migration_quiescence_retry_ns: dict[str, int] = {}
        self._p2p_transport = None
        self._kv_pool = None
        self._last_batch_size = 0
        self._last_queue_delay_ns = 0
        self._data_dir = Path(data_dir).absolute()
        self._source_stores: dict[str, object] = {}
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
        self._result_sequence = 0
        self._result_pool_generation = uuid.uuid4().hex
        self._source_done_sequence = 0
        self._stop = False
        self._graph_measurement_counts: dict[tuple[str, str], list[int]] = {}
        self._quarantined_migration_ids: set[str] = set()
        self._feature_pull_transport = None
        self._feature_staging_buffers: list[torch.Tensor] = []
        self._free_feature_staging_slots: deque[int] = deque()
        self._pending_feature_batches: deque[PendingFeatureBatch] = deque()
        self._pending_dflash_completions: deque[PendingDFlashCompletion] = deque()

        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", instance_id)
        cleanup_stale_slot_files(Path(data_dir), instance_id=safe_id, kind="results")
        result_path = (
            Path(data_dir).absolute() / f"dt-{safe_id}-{os.getpid()}-results.slots"
        )
        self._result_store = MmapSlotStore.create(
            result_path,
            slot_count=result_slot_count,
            slot_bytes=max_batch_size * self._result_row_width * 4,
        )
        self._client = None
        try:
            self._client = DraftCoordinatorClient(
                endpoint=coordinator_endpoint,
                instance_id=instance_id,
                draft_profile_ids=(self.profile.profile_id,),
                max_pending_tasks=max_pending_tasks,
                data_endpoint=self._result_store.uri,
                result_pool_generation=self._result_pool_generation,
            )
            self._initialize_model(
                nccl_port=nccl_port,
                mem_fraction_static=mem_fraction_static,
                disable_cuda_graph=disable_cuda_graph,
            )
            self._initialize_migration_transport()
            self._initialize_pivot_staging()
            self._client.pool = self._kv_pool
            self._client.register_draft(timeout_s=registration_timeout_s)
            self._publish_capacity()
        except Exception:
            if self._feature_pull_transport is not None:
                self._feature_pull_transport.close()
                self._feature_pull_transport = None
            if self._p2p_transport is not None:
                self._p2p_transport.close()
                self._p2p_transport = None
            if self._client is not None:
                self._client.close()
                self._client = None
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
        self._pivot_host_staging = None
        self._pivot_host_views = ()
        if self._persistent or not self.pinned_pivot_staging:
            return
        self._pivot_host_staging = torch.empty(
            (self.max_batch_size, self._pivot_width),
            dtype=torch.bfloat16,
            device="cpu",
            pin_memory=True,
        )
        self._pivot_host_views = tuple(
            memoryview(row.view(torch.uint8).numpy())
            for row in self._pivot_host_staging
        )
        logger.info(
            "Enabled pinned DT pivot staging: shape=%s bytes=%d",
            tuple(self._pivot_host_staging.shape),
            self._pivot_host_staging.numel() * self._pivot_host_staging.element_size(),
        )

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
        if flash_config.markov_output_mode != "direct":
            raise ValueError("DT V1 requires a direct-output FlashMTP checkpoint")
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

        if self._draft_model.markov_head is None:
            raise ValueError("DT V1 requires the FlashMTP Markov direct-output head")
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
            make_draft_block_spec_info,
        )
        from sglang.srt.speculative.dt_feature_p2p import DraftFeaturePullTransport

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
            "page_size": 1,
            # The primary worker is not marked as a speculative child, but
            # DFlash proposal batches still use TARGET_VERIFY attention
            # geometry.  Seed its fixed block width for the attention backend.
            "speculative_num_draft_tokens": self.profile.block_size,
            "disable_overlap_schedule": True,
            # Remote proposal currently uses the eager DFlash forward so the
            # candidate head and persistent row ownership remain explicit.
            "disable_cuda_graph": True,
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
        self._draft_model.eval()
        self._draft_worker.alloc_memory_pool()
        self.max_pending_tasks = min(
            self.max_pending_tasks,
            int(self._draft_model_runner.max_running_requests),
        )
        self._client.max_pending_tasks = self.max_pending_tasks
        self._draft_worker.init_attention_backends()
        self._draft_worker.init_cuda_graphs(capture_decode_cuda_graph=False)

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
        self._feature_pull_transport = DraftFeaturePullTransport(device=self.device)
        self._feature_staging_buffers = [
            torch.empty(
                (self.feature_buffer_max_tokens, self._pivot_width),
                dtype=torch.bfloat16,
                device=self.device,
            )
            for _ in range(self.feature_buffer_slot_count)
        ]
        self._free_feature_staging_slots.extend(range(self.feature_buffer_slot_count))
        self._draft_cuda_graph = None
        logger.info(
            "DT DFLASH Draft ready: instance=%s block=%d feature_width=%d "
            "feature_slots=%d feature_tokens=%d cuda_graph=%s",
            self.instance_id,
            self.profile.block_size,
            self._pivot_width,
            self.feature_buffer_slot_count,
            self.feature_buffer_max_tokens,
            not disable_cuda_graph and False,
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
        while not self._stop:
            self._poll_messages()
            if self._is_dflash:
                self._poll_dflash_completions()
                self._poll_feature_batches()
            batch = self._take_batch()
            if batch:
                self._execute_batch(batch)
            else:
                sleep_s = self._idle_sleep_s()
                if sleep_s > 0:
                    time.sleep(sleep_s)

    def stop(self) -> None:
        self._stop = True

    def _quiesce_dflash_shutdown(self) -> None:
        # Fence new coordinator admission before retiring already-dispatched
        # source leases.
        self._publish_capacity()
        queued = tuple(self._pending)
        for task in queued:
            self._cancelled_ids.add(task.transfer_id)
            if not task.source_read_migration_id:
                self._publish_source_done(task)
            self._pending_ids.discard(task.transfer_id)
            self._received_ns_by_transfer.pop(task.transfer_id, None)
        self._pending.clear()
        for batch in self._pending_feature_batches:
            self._cancelled_ids.update(item.task.transfer_id for item in batch.pending)
        for completion in self._pending_dflash_completions:
            self._cancelled_ids.update(
                item.task.transfer_id for item in completion.pending
            )
        if self._feature_pull_transport is not None:
            self._feature_pull_transport.close()
        # The normal scheduler admits only one materialization/proposal batch
        # at a time.  Preserve that ordering during shutdown while draining
        # every already-copied batch and its source read-done lifecycle.
        while self._pending_feature_batches or self._pending_dflash_completions:
            self._poll_feature_batches()
            if self._pending_dflash_completions:
                self._pending_dflash_completions[
                    0
                ].completion_event.synchronize()
                self._poll_dflash_completions()
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
                self._received_ns_by_transfer.pop(task.transfer_id, None)
            self._pending.clear()
            self._flush_source_done()
            if time.monotonic_ns() >= deadline_ns:
                raise RuntimeError(
                    "DFLASH shutdown timed out waiting for source read-done ACKs"
                )
            time.sleep(IDLE_POLL_S)
        self._feature_pull_transport = None
        if self._feature_source_unquiescent:
            raise RuntimeError(
                "DFLASH shutdown requires process teardown for unquiesced feature reads"
            )

    def close(self) -> None:
        if self._is_dflash:
            self._quiesce_dflash_shutdown()
        if self._p2p_transport is not None:
            self._p2p_transport.close()
        for migration_id, prepared in tuple(self._prepared_migrations.items()):
            if (
                migration_id not in self._quarantined_migration_ids
                and prepared.transfer_future is None
                and prepared.message.draft_kv_handle not in self._persistent_requests
                and prepared.req.req_pool_idx is not None
            ):
                self._cleanup_partial_migration_request(prepared.req)
        self._prepared_migrations.clear()
        for draft_kv_handle in tuple(self._persistent_requests):
            self._release_persistent_request(draft_kv_handle)
        for store in self._source_stores.values():
            store.close()
        self._source_stores.clear()
        self._migration_source_events.clear()
        self._migration_source_handles.clear()
        self._client.close()
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
        )

        capacity_changed = False
        self._flush_source_done()
        capacity_changed = self._poll_migration_transfers() or capacity_changed
        for message in self._client.recv_nowait():
            if isinstance(message, DraftDispatchBatch):
                capacity_changed = self._accept_dispatch(message) or capacity_changed
            elif isinstance(message, CancelTransfers):
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
                    item.task.transfer_id
                    for batch in self._pending_dflash_completions
                    for item in batch.pending
                    if item.task.draft_kv_handle == message.draft_kv_handle
                    and item.task.request_id == message.request_id
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
                self._delivery_outbox.ack("result", message.batch_id)
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

    def _accept_dispatch(self, message) -> bool:
        received_ns = time.monotonic_ns()
        accepted = False
        for task in message.entries:
            if task.transfer_id in self._pending_ids:
                continue
            active_feature_tasks = sum(
                len(batch.pending) for batch in self._pending_feature_batches
            ) + sum(len(batch.pending) for batch in self._pending_dflash_completions)
            if len(self._pending) + active_feature_tasks >= self.max_pending_tasks:
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
            self._pending_ids.add(task.transfer_id)
            self._received_ns_by_transfer[task.transfer_id] = received_ns
            accepted = True
        return accepted

    def _take_batch(self) -> list[PendingDraftTask]:
        if not self._pending:
            return []
        if self._is_dflash:
            if not self._free_feature_staging_slots:
                return []
            if self._pending[0].requires_proposal and not self._free_result_slots:
                return []
        elif not self._free_result_slots:
            return []
        if len(self._pending) < self.batch_ready_threshold and self.batch_wait_s > 0:
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
        queue_delays = []
        feature_token_count = 0
        while self._pending and len(queued) < self.max_batch_size:
            next_task = self._pending[0]
            if self._is_dflash:
                if next_task.requires_proposal and not self._free_result_slots:
                    break
                next_feature_tokens = int(next_task.feature_token_count)
                if (
                    feature_token_count + next_feature_tokens
                    > self.feature_buffer_max_tokens
                ):
                    break
            task = self._pending.popleft()
            self._pending_ids.discard(task.transfer_id)
            received_ns = self._received_ns_by_transfer.pop(task.transfer_id, None)
            if received_ns is not None:
                queue_delays.append(take_ns - received_ns)
            if task.transfer_id in self._cancelled_ids:
                self._cancelled_ids.discard(task.transfer_id)
                if self._is_dflash and not task.source_read_migration_id:
                    self._publish_source_done(task)
                continue
            if self._is_dflash and take_ns >= task.deadline_ns:
                if not task.source_read_migration_id:
                    self._publish_source_done(task)
                self._release_persistent_request(task.draft_kv_handle)
                self._publish_failures(
                    [(task, TimeoutError("DFLASH feature task deadline expired"))]
                )
                continue
            feature_token_count += int(task.feature_token_count)
            queued.append(
                PendingDraftTask(
                    task=task,
                    pivot_bytes=None,
                )
            )
        if queue_delays:
            self._last_queue_delay_ns = max(queue_delays)

        # The result publisher needs rows grouped by Target. Sort before direct
        # mmap reads so pinned staging is already laid out in publish order.
        queued = target_major(queued)
        if self._is_dflash:
            return queued
        if self._persistent:
            return queued
        pinned_staging = self._pivot_host_staging
        tasks: list[PendingDraftTask] = []
        for item in queued:
            task = item.task
            try:
                pivot_row = -1
                pivot_bytes = None
                if pinned_staging is None:
                    pivot_bytes = self._read_pivot(task)
                else:
                    pivot_row = len(tasks)
                    self._read_pivot_into(task, self._pivot_host_views[pivot_row])
            except Exception as exc:
                logger.exception("failed to read DT pivot for %s", task.transfer_id)
                self._publish_source_done(task)
                self._publish_failures([(task, exc)])
                continue
            self._publish_source_done(task)
            tasks.append(
                PendingDraftTask(
                    task=task,
                    pivot_bytes=pivot_bytes,
                    pivot_row=pivot_row,
                )
            )
        if self.batch_source_read_done:
            self._flush_source_done()
        return tasks

    def _idle_sleep_s(self) -> float:
        if (
            not self._pending
            or not self._free_result_slots
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

    @torch.inference_mode()
    def _execute_batch(self, pending: list[PendingDraftTask]) -> None:
        self._last_batch_size = len(pending)
        if self._is_dflash:
            self._submit_dflash_feature_batch(pending)
            return
        if self._persistent:
            self._execute_standalone_batch(pending)
            return
        tasks = [item.task for item in pending]
        batch_size = len(tasks)
        if self._pivot_host_staging is None:
            pivot_blob = b"".join(item.pivot_bytes for item in pending)
            pivots = torch.frombuffer(
                bytearray(pivot_blob), dtype=torch.bfloat16
            ).reshape(batch_size, self._pivot_width)
        else:
            if any(item.pivot_row != index for index, item in enumerate(pending)):
                raise RuntimeError("pinned DT pivot rows are not contiguous")
            pivots = self._pivot_host_staging[:batch_size]
        pivots = pivots.to(
            self.device,
            non_blocking=self._pivot_host_staging is not None,
        )
        bonus_tokens = torch.tensor(
            [task.next_token_id for task in tasks],
            dtype=torch.int64,
            device=self.device,
        )
        if self._draft_cuda_graph is not None:
            candidates = self._draft_cuda_graph.replay(
                pivots=pivots,
                bonus_tokens=bonus_tokens,
            )
        else:
            block_ids = torch.full(
                (batch_size, self._draft_block_len),
                self._mask_token_id,
                dtype=torch.int64,
                device=self.device,
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
            proposals = self._draft_model.sample_draft_tokens(
                block_hidden=block_hidden,
                first_prev_token_ids=bonus_tokens,
            )
            if tuple(proposals.shape) != (batch_size, self._num_proposals):
                raise RuntimeError(
                    f"FlashMTP proposal shape mismatch: got {tuple(proposals.shape)}"
                )
            candidates = torch.empty(
                (batch_size, self.profile.block_size),
                dtype=torch.int32,
                device=self.device,
            )
            candidates[:, 0].copy_(bonus_tokens)
            candidates[:, 1:].copy_(proposals)
        candidate_bytes = candidates.cpu().contiguous().numpy().tobytes()
        self._publish_successes(tasks, candidate_bytes)
        self._publish_capacity()

    def _submit_dflash_feature_batch(self, pending: list[PendingDraftTask]) -> None:
        from sglang.srt.speculative.dt_feature_p2p import FeatureSourceDescriptor

        if self._feature_pull_transport is None:
            raise RuntimeError("DFLASH feature transport is not initialized")
        if not self._free_feature_staging_slots:
            raise RuntimeError("DFLASH feature staging admission was violated")
        staging_slot = self._free_feature_staging_slots.popleft()
        staging = self._feature_staging_buffers[staging_slot]
        futures = []
        row_ranges = []
        row_offset = 0
        submitted_ns = time.monotonic_ns()
        for item in pending:
            task = item.task
            source = task.source
            row_count = int(task.feature_token_count)
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
            try:
                future = self._feature_pull_transport.submit(
                    source=descriptor,
                    destination=staging[row_offset:row_end],
                )
            except Exception as exc:
                future = Future()
                future.set_exception(exc)
            futures.append(future)
            if task.source_read_migration_id:
                self._feature_source_inflight.add(
                    (task.source_read_migration_id, task.transfer_id)
                )
            row_ranges.append((row_offset, row_end))
            row_offset = row_end
            profile_emit(
                "draft_feature_pull_submit",
                draft_instance=self.instance_id,
                transfer_id=task.transfer_id,
                request_id=task.request_id,
                round_id=task.round_id,
                feature_bytes=source.nbytes,
                submitted_ns=submitted_ns,
            )
        self._pending_feature_batches.append(
            PendingFeatureBatch(
                pending=pending,
                staging_slot=staging_slot,
                row_ranges=tuple(row_ranges),
                futures=tuple(futures),
                submitted_ns=submitted_ns,
            )
        )
        self._publish_capacity()

    def _poll_feature_batches(self) -> None:
        capacity_changed = False
        while self._pending_feature_batches:
            # CUDA construction from pageable request metadata can wait for
            # earlier work on the materialization/proposal stream.  Keep at
            # most one such batch submitted: feature-copy workers may continue
            # filling the other staging slots, while the outer scheduler loop
            # remains free to poll control messages and the completion event.
            if self._pending_dflash_completions:
                break
            feature_batch = self._pending_feature_batches[0]
            if not all(future.done() for future in feature_batch.futures):
                break
            self._pending_feature_batches.popleft()
            failures = []
            resources_quiescent = True
            for item, future in zip(feature_batch.pending, feature_batch.futures):
                task = item.task
                deferred_key = (
                    (task.source_read_migration_id, task.transfer_id)
                    if task.source_read_migration_id
                    else None
                )
                source_safe = True
                copy_succeeded = False
                try:
                    metrics = future.result()
                except Exception as exc:
                    failures.append((task, exc))
                    source_safe = bool(getattr(exc, "resources_quiescent", True))
                    resources_quiescent = resources_quiescent and source_safe
                else:
                    copy_succeeded = True
                    profile_emit(
                        "draft_feature_pull_complete",
                        draft_instance=self.instance_id,
                        transfer_id=task.transfer_id,
                        request_id=task.request_id,
                        round_id=task.round_id,
                        submitted_ns=metrics.submitted_ns,
                        gpu_completed_ns=metrics.gpu_completed_ns,
                        mapping_ns=metrics.mapping_ns,
                        driver_submit_ns=metrics.driver_submit_ns,
                        gpu_copy_ns=metrics.gpu_copy_ns,
                        feature_bytes=metrics.feature_bytes,
                        cold_mapping_count=metrics.cold_mapping_count,
                    )
                if deferred_key is not None:
                    self._feature_source_inflight.discard(deferred_key)
                    if source_safe:
                        self._feature_source_quiescent.add(deferred_key)
                    else:
                        self._feature_source_unquiescent.add(deferred_key)
                if copy_succeeded and deferred_key is not None:
                    previous = self._deferred_feature_source_tasks.get(deferred_key)
                    if previous is not None and previous != task:
                        raise ValueError(
                            "deferred DFlash feature transfer changed payload"
                        )
                    self._deferred_feature_source_tasks[deferred_key] = task
                elif source_safe and not task.source_read_migration_id:
                    self._publish_source_done(task)

            if failures:
                first_error = failures[0][1]
                tasks = [item.task for item in feature_batch.pending]
                for handle in {task.draft_kv_handle for task in tasks}:
                    self._release_persistent_request(handle)
                self._publish_failures([(task, first_error) for task in tasks])
                if resources_quiescent:
                    self._free_feature_staging_slots.append(feature_batch.staging_slot)
                else:
                    self._stop = True
                capacity_changed = True
                continue

            completed_ns = time.monotonic_ns()
            expired = [
                item.task
                for item in feature_batch.pending
                if item.task.transfer_id not in self._cancelled_ids
                and completed_ns >= item.task.deadline_ns
            ]
            expired_ids = {task.transfer_id for task in expired}
            active = [
                (item, row_range)
                for item, row_range in zip(
                    feature_batch.pending, feature_batch.row_ranges
                )
                if item.task.transfer_id not in self._cancelled_ids
                and item.task.transfer_id not in expired_ids
            ]
            cancelled = [
                item.task
                for item in feature_batch.pending
                if item.task.transfer_id in self._cancelled_ids
            ]
            for task in cancelled:
                self._cancelled_ids.discard(task.transfer_id)
                self._release_persistent_request(task.draft_kv_handle)
            for task in expired:
                self._release_persistent_request(task.draft_kv_handle)
            if expired:
                self._publish_failures(
                    [
                        (
                            task,
                            TimeoutError(
                                "DFLASH feature pull completed after its deadline"
                            ),
                        )
                        for task in expired
                    ]
                )
            if not active:
                self._free_feature_staging_slots.append(feature_batch.staging_slot)
                capacity_changed = True
                continue
            active_pending = [item for item, _row_range in active]
            active_ranges = tuple(row_range for _item, row_range in active)
            try:
                self._launch_dflash_materialization(
                    pending=active_pending,
                    staging_slot=feature_batch.staging_slot,
                    row_ranges=active_ranges,
                )
            except Exception as exc:
                logger.exception("failed to launch DT DFLASH materialization")
                resources_quiescent = bool(getattr(exc, "resources_quiescent", True))
                if resources_quiescent:
                    try:
                        torch.cuda.current_stream(self.device).synchronize()
                    except Exception:
                        resources_quiescent = False
                if not resources_quiescent:
                    self._stop = True
                tasks = [item.task for item in active_pending]
                if resources_quiescent:
                    for handle in {task.draft_kv_handle for task in tasks}:
                        self._release_persistent_request(handle)
                else:
                    # A failed stream fence cannot prove that queued projection
                    # or proposal writes have stopped.  Keep both the staging
                    # range and every touched KV row quarantined until process
                    # exit; freeing either could turn a late write into silent
                    # corruption of unrelated work.
                    logger.critical(
                        "quarantined DFLASH staging and KV after an "
                        "unquiesced GPU failure: transfers=%s",
                        [task.transfer_id for task in tasks],
                    )
                self._publish_failures([(task, exc) for task in tasks])
                if resources_quiescent:
                    self._free_feature_staging_slots.append(feature_batch.staging_slot)
            capacity_changed = True
        if self.batch_source_read_done:
            self._flush_source_done()
        self._resume_feature_finalizations()
        if capacity_changed:
            self._publish_capacity()

    def _launch_dflash_materialization(
        self,
        *,
        pending: list[PendingDraftTask],
        staging_slot: int,
        row_ranges: tuple[tuple[int, int], ...],
    ) -> None:
        if self._pending_dflash_completions:
            raise RuntimeError(
                "DFLASH materialization submission must wait for the prior "
                "completion event"
            )
        tasks = [item.task for item in pending]
        states = []
        cache_locs = []
        positions = []
        for task in tasks:
            state, cache_loc = self._prepare_dflash_task(task)
            states.append(state)
            cache_locs.append(cache_loc)
            positions.append(
                torch.arange(
                    task.feature_token_offset,
                    task.committed_kv_len,
                    dtype=torch.int64,
                    device=self.device,
                )
            )
        staging = self._feature_staging_buffers[staging_slot]
        hidden_rows = [staging[start:end] for start, end in row_ranges]
        if all(
            row_ranges[index][1] == row_ranges[index + 1][0]
            for index in range(len(row_ranges) - 1)
        ):
            target_hidden = staging[row_ranges[0][0] : row_ranges[-1][1]]
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
            # Build every CPU-derived proposal tensor before materialization is
            # enqueued.  Creating a CUDA tensor from pageable CPU data after the
            # append can wait for preceding work on this stream and stall the
            # Draft scheduler loop.  The committed lengths are already
            # authoritative CPU state, so seq_lens_cpu never needs a GPU
            # round-trip.
            committed_lens = [state.valid_committed_len for state in proposal_states]
            proposal_req_pool_indices = torch.tensor(
                [state.req.req_pool_idx for state in proposal_states],
                dtype=torch.int64,
                device=self.device,
            )
            proposal_prefix_lens = torch.tensor(
                committed_lens,
                dtype=torch.int64,
                device=self.device,
            )
            proposal_seq_lens_cpu = torch.tensor(
                [length + int(self.profile.block_size) for length in committed_lens],
                dtype=torch.int32,
                device="cpu",
            )
            proposal_bonus_tokens = torch.tensor(
                [task.next_token_id for task in proposal_tasks],
                dtype=torch.int64,
                device=self.device,
            )

        materialize_start.record(torch.cuda.current_stream(self.device))
        self._dflash_materializer.append(
            target_hidden=target_hidden,
            cache_loc=torch.cat(cache_locs, dim=0),
            positions=torch.cat(positions, dim=0),
        )
        materialize_done.record(torch.cuda.current_stream(self.device))

        candidates = None
        if proposal_indices:
            candidates = self._run_dflash_proposal(
                states=proposal_states,
                req_pool_indices=proposal_req_pool_indices,
                prefix_lens=proposal_prefix_lens,
                seq_lens_cpu=proposal_seq_lens_cpu,
                bonus_tokens=proposal_bonus_tokens,
            )
        proposal_done.record(torch.cuda.current_stream(self.device))
        self._pending_dflash_completions.append(
            PendingDFlashCompletion(
                pending=pending,
                staging_slot=staging_slot,
                completion_event=proposal_done,
                materialize_start_event=materialize_start,
                materialize_done_event=materialize_done,
                proposal_done_event=proposal_done,
                proposal_tasks=proposal_tasks,
                candidates=candidates,
                launched_ns=time.monotonic_ns(),
            )
        )

    def _prepare_dflash_task(self, task) -> tuple[PersistentDraftRequest, torch.Tensor]:
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
            req.prefix_indices = cache_loc.to(torch.int64)
            req_pool.req_to_token[req.req_pool_idx, : task.committed_kv_len] = (
                cache_loc.to(torch.int32)
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
            self._persistent_requests[task.draft_kv_handle] = state
            return state, cache_loc.to(torch.int64)

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
        discarded_tokens = allocated_len - state.valid_committed_len
        self._free_persistent_kv_suffix(req, state.valid_committed_len, allocated_len)
        req.kv.kv_allocated_len = state.valid_committed_len
        cache_loc = allocator.alloc(token_count)
        if cache_loc is None:
            raise RuntimeError("DFLASH reconcile KV-page OOM")
        req_pool.req_to_token[
            req.req_pool_idx, task.feature_token_offset : task.committed_kv_len
        ] = cache_loc.to(torch.int32)
        req.kv.kv_allocated_len = task.committed_kv_len
        req.kv_committed_len = task.committed_kv_len
        req.prefix_indices = req_pool.req_to_token[
            req.req_pool_idx, : task.committed_kv_len
        ].to(torch.int64)
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
        return state, cache_loc.to(torch.int64)

    def _run_dflash_proposal(
        self,
        *,
        states: list[PersistentDraftRequest],
        req_pool_indices: torch.Tensor,
        prefix_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        bonus_tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = len(states)
        block_size = int(self.profile.block_size)
        req_pool = self._draft_model_runner.req_to_token_pool
        allocator = self._draft_model_runner.token_to_kv_pool_allocator
        out_cache_loc = allocator.alloc(batch_size * block_size)
        if out_cache_loc is None:
            raise RuntimeError("DFLASH proposal KV-page OOM")
        try:
            out_cache_loc_2d = out_cache_loc.view(batch_size, block_size)
            for row, state in enumerate(states):
                end = state.valid_committed_len + block_size
                req_pool.req_to_token[
                    state.req.req_pool_idx, state.valid_committed_len : end
                ] = out_cache_loc_2d[row].to(torch.int32)
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
            positions = prefix_lens.unsqueeze(1) + self._block_pos_offsets
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
            )
            draft_out = self._draft_model_runner.forward(forward_batch)
            hidden_states = draft_out.logits_output.hidden_states
            if hidden_states is None:
                raise RuntimeError("DFLASH Draft forward returned no hidden states")
            self._dflash_sampler(hidden_states)
            proposals = self._dflash_sampler.out[: batch_size * (block_size - 1)].view(
                batch_size, block_size - 1
            )
            candidates = torch.empty(
                (batch_size, block_size), dtype=torch.int32, device=self.device
            )
            candidates[:, 0].copy_(bonus_tokens)
            candidates[:, 1:].copy_(proposals)
            return candidates
        except Exception as exc:
            # No successful completion event will own this allocation.  Fence
            # exceptional GPU work before making the pages reusable; if that
            # fence itself fails, report the resource as unquiesced so the
            # caller quarantines every affected request instead.
            try:
                torch.cuda.current_stream(self.device).synchronize()
            except Exception:
                setattr(exc, "resources_quiescent", False)
                raise
            allocator.free(out_cache_loc)
            for state in states:
                if state.req.kv is not None:
                    state.req.kv.kv_allocated_len = state.valid_committed_len
                state.req.kv_committed_len = state.valid_committed_len
                state.proposal_base_len = state.valid_committed_len
            raise

    def _poll_dflash_completions(self) -> None:
        capacity_changed = False
        while self._pending_dflash_completions:
            completion = self._pending_dflash_completions[0]
            if not completion.completion_event.query():
                break
            self._pending_dflash_completions.popleft()
            tasks = [item.task for item in completion.pending]
            completed_ns = time.monotonic_ns()
            expired_tasks = [
                task
                for task in tasks
                if task.transfer_id not in self._cancelled_ids
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
            ]
            if materialize_only:
                self._publish_materialization_successes(materialize_only)
            if active_proposal_indices:
                candidate_rows = completion.candidates[active_proposal_indices]
                candidate_bytes = candidate_rows.cpu().contiguous().numpy().tobytes()
                self._publish_successes(
                    [
                        completion.proposal_tasks[index]
                        for index in active_proposal_indices
                    ],
                    candidate_bytes,
                )
            for task in active_tasks:
                proposal_gpu_ns = int(
                    completion.materialize_done_event.elapsed_time(
                        completion.proposal_done_event
                    )
                    * 1_000_000
                )
                profile_emit(
                    "draft_kv_materialization",
                    draft_instance=self.instance_id,
                    transfer_id=task.transfer_id,
                    request_id=task.request_id,
                    round_id=task.round_id,
                    committed_kv_len=task.committed_kv_len,
                    materialized_tokens=task.feature_token_count,
                    gpu_materialization_ns=int(
                        completion.materialize_start_event.elapsed_time(
                            completion.materialize_done_event
                        )
                        * 1_000_000
                    ),
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
                self._cancelled_ids.discard(task.transfer_id)
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
            self._free_feature_staging_slots.append(completion.staging_slot)
            for handle in {task.draft_kv_handle for task in tasks}:
                self._drain_deferred_persistent_release(handle)
            capacity_changed = True
        if capacity_changed:
            self._publish_capacity()

    def _execute_standalone_batch(self, pending: list[PendingDraftTask]) -> None:
        batch_start_ns = time.monotonic_ns()
        sync_profile = profile_enabled() and cuda_sync_enabled()
        tasks = [item.task for item in pending]
        states: list[PersistentDraftRequest] = []
        prepare_done_ns = batch_start_ns
        extend_done_ns = batch_start_ns
        decode_done_ns: list[int] = []
        extend_cuda_graph = False
        decode_graph_replays = 0
        try:
            states = [self._prepare_persistent_task(task) for task in tasks]
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
            prepare_done_ns = time.monotonic_ns()
            extend_result = self._draft_worker.forward_batch_generation(batch)
            next_tokens = extend_result.next_token_ids
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

            measurement_tasks = [
                task
                for task in tasks
                if task.request_id.startswith(DT_BENCHMARK_MEASUREMENT_RID_PREFIX)
            ]
            measurement_reconcile = sum(task.round_id > 0 for task in measurement_tasks)
            measurement_bootstrap = len(measurement_tasks) - measurement_reconcile
            if measurement_bootstrap:
                self._record_graph_measurement(
                    path="bootstrap_prefill",
                    execution="graph" if extend_cuda_graph else "eager",
                    measurement_rows=measurement_bootstrap,
                    measurement_batch_rows=len(measurement_tasks),
                    batch_size=len(tasks),
                    model_runner_event_id=extend_measurement_event_id,
                )
            if measurement_reconcile:
                self._record_graph_measurement(
                    path="reconcile_prefill",
                    execution="graph" if extend_cuda_graph else "eager",
                    measurement_rows=measurement_reconcile,
                    measurement_batch_rows=len(measurement_tasks),
                    batch_size=len(tasks),
                    model_runner_event_id=extend_measurement_event_id,
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

            roots = torch.tensor(
                [task.committed_token_ids[-1] for task in tasks],
                dtype=torch.int64,
                device=self.device,
            )
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
            candidate_bytes = candidates.cpu().contiguous().numpy().tobytes()
        except Exception as exc:
            logger.exception("failed DT STANDALONE batch")
            for task in tasks:
                self._release_persistent_request(task.draft_kv_handle)
            self._publish_failures([(task, exc) for task in tasks])
        else:
            self._publish_successes(tasks, candidate_bytes)
        finished_ns = time.monotonic_ns()
        decode_durations_ns = []
        previous_ns = extend_done_ns
        for done_ns in decode_done_ns:
            decode_durations_ns.append(done_ns - previous_ns)
            previous_ns = done_ns
        profile_emit(
            "draft_standalone_batch",
            batch_size=len(tasks),
            bootstrap_count=sum(task.round_id == 0 for task in tasks),
            reconcile_count=sum(task.round_id > 0 for task in tasks),
            extend_cuda_graph=extend_cuda_graph,
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
                None if execution == "graph" else "model_runner_reported"
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
        )

    def _poll_deferred_migration_gpu_controls(self) -> bool:
        controls = getattr(self, "_deferred_migration_gpu_controls", None)
        if not controls or getattr(self, "_pending_dflash_completions", ()):
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
        if self._is_dflash and getattr(self, "_pending_dflash_completions", ()):
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
        if self._is_dflash and getattr(self, "_pending_dflash_completions", ()):
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
                # cudaMemcpyBatchAsync may have submitted a prefix before an
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
                self._received_ns_by_transfer.pop(task.transfer_id, None)
                self._cancelled_ids.discard(task.transfer_id)
            else:
                retained.append(pending_task)
        self._pending = retained
        if removed:
            self._publish_capacity()

    def _resume_feature_finalizations(self) -> None:
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
        return any(
            item.task.draft_kv_handle == draft_kv_handle
            for completion in self._pending_dflash_completions
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

        for release_id, message in tuple(self._pending_request_release_acks.items()):
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

    def _publish_successes(self, tasks: list[object], candidate_bytes: bytes) -> None:
        from dt_protocol import (
            PROTOCOL_VERSION,
            DraftExecutionResultBatch,
            DraftResultEntry,
            ResultBufferRef,
        )

        for task in tasks:
            self._cancelled_ids.discard(task.transfer_id)
        slot_id = self._free_result_slots.popleft()
        generation = self._result_store.write(slot_id, candidate_bytes)
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
        self._send_result(message)

    def _publish_materialization_successes(self, tasks: list[object]) -> None:
        from dt_protocol import (
            PROTOCOL_VERSION,
            DraftExecutionResultBatch,
            DraftResultEntry,
        )

        for task in tasks:
            self._cancelled_ids.discard(task.transfer_id)
        self._result_sequence += 1
        message = DraftExecutionResultBatch(
            protocol_version=PROTOCOL_VERSION,
            batch_id=f"{self.instance_id}:materialized:{self._result_sequence}",
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

    def _read_pivot(self, task) -> bytes:
        source = task.source
        store = self._get_source_store(source.uri)
        payload = store.read(source.source_slot_id, source.slot_generation)
        if len(payload) != source.nbytes:
            raise ValueError(
                f"pivot byte size mismatch: descriptor={source.nbytes}, "
                f"actual={len(payload)}"
            )
        return payload

    def _read_pivot_into(self, task, destination: memoryview) -> None:
        source = task.source
        store = self._get_source_store(source.uri)
        length = store.read_into(
            source.source_slot_id,
            source.slot_generation,
            destination,
        )
        if length != source.nbytes:
            raise ValueError(
                f"pivot byte size mismatch: descriptor={source.nbytes}, actual={length}"
            )

    def _get_source_store(self, uri: str):
        from dt_transport import MmapSlotStore

        store = self._source_stores.get(uri)
        if store is None:
            store = MmapSlotStore.open(uri)
            self._source_stores[uri] = store
        return store

    def _validate_task(self, task) -> None:
        from dt_protocol import DraftTaskKind

        source = task.source
        if task.draft_profile_id != self.profile.profile_id:
            raise ValueError("Draft task profile mismatch")
        if task.k != self.profile.draft_k:
            raise ValueError("Draft task k mismatch")
        if self._is_dflash:
            if task.kind not in {
                DraftTaskKind.BOOTSTRAP,
                DraftTaskKind.RECONCILE,
            }:
                raise ValueError("DFLASH Draft requires a persistent task kind")
            if not task.draft_kv_handle:
                raise ValueError("DFLASH Draft task requires a KV handle")
            if source is None:
                raise ValueError("DFLASH Draft task requires Target hidden states")
            if (
                source.target_instance_id != task.target_instance_id
                or source.transfer_id != task.transfer_id
            ):
                raise ValueError("DFLASH feature source identity mismatch")
            if source.feature_schema_id != self.profile.feature_schema.schema_id:
                raise ValueError("DFLASH feature schema mismatch")
            if source.transport != "cuda-ipc":
                raise ValueError("DFLASH features require CUDA IPC transport")
            if source.dtype != "bfloat16":
                raise ValueError("DFLASH feature source must be BF16")
            if tuple(source.shape) != (
                task.feature_token_count,
                self._pivot_width,
            ):
                raise ValueError("DFLASH feature source shape mismatch")
            expected_nbytes = task.feature_token_count * self._pivot_width * 2
            if source.nbytes != expected_nbytes:
                raise ValueError("DFLASH feature source byte length mismatch")
            if (
                task.feature_token_count <= 0
                or task.feature_token_count > self.feature_buffer_max_tokens
                or len(task.committed_token_ids) != task.feature_token_count
            ):
                raise ValueError("DFLASH feature/token count is invalid")
            if (
                task.feature_token_offset < 0
                or task.committed_kv_len
                != task.feature_token_offset + task.feature_token_count
            ):
                raise ValueError("DFLASH committed-length invariant failed")
            if task.kind == DraftTaskKind.BOOTSTRAP and (
                task.round_id != 0 or task.feature_token_offset != 0
            ):
                raise ValueError("DFLASH bootstrap must begin at round/offset zero")
            if not (
                source.source_session_id
                and source.pool_generation
                and source.transport_session
                and source.device_uuid
                and source.ipc_handle
                and source.producer_event_handle
            ):
                raise ValueError("DFLASH CUDA IPC descriptor fence is incomplete")
            if (
                source.allocation_length <= 0
                or source.allocation_offset < 0
                or source.byte_offset < 0
                or source.allocation_offset + source.byte_offset + source.nbytes
                > source.allocation_length
            ):
                raise ValueError("DFLASH CUDA IPC source range is invalid")
            if any(
                token_id < 0 or token_id >= self.profile.vocab_size
                for token_id in (*task.committed_token_ids, task.next_token_id)
            ):
                raise ValueError("DFLASH token ID is outside vocabulary")
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
        if source is None:
            raise ValueError("FlashMTP V1 requires a Target feature source")
        if source.feature_schema_id != self.profile.feature_schema.schema_id:
            raise ValueError("Draft task feature schema mismatch")
        expected_nbytes = self._pivot_width * 2
        if source.nbytes != expected_nbytes:
            raise ValueError(
                f"Draft pivot nbytes mismatch: {source.nbytes} != {expected_nbytes}"
            )
        if source.transport != "mmap" or not source.uri:
            raise ValueError("DT V1 Draft supports only mmap source descriptors")

    def _drop_cancelled_pending(self) -> None:
        if not self._cancelled_ids or not self._pending:
            return
        kept = deque()
        for task in self._pending:
            if task.transfer_id in self._cancelled_ids:
                self._pending_ids.discard(task.transfer_id)
                self._cancelled_ids.discard(task.transfer_id)
                self._received_ns_by_transfer.pop(task.transfer_id, None)
                if not self._persistent or self._is_dflash:
                    if not getattr(task, "source_read_migration_id", ""):
                        self._publish_source_done(task)
            else:
                kept.append(task)
        self._pending = kept

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
        key = (
            message.draft_session_id,
            message.result_pool_generation,
            message.buffer_id,
            message.generation,
            message.row_offset,
            message.row_count,
        )
        batch_id = self._published_by_range.pop(key, None)
        released = False
        if batch_id is not None:
            published = self._published_by_batch[batch_id]
            published.range_transfer_ids.pop(key, None)
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
        active_feature_tasks = sum(
            len(batch.pending) for batch in self._pending_feature_batches
        ) + sum(len(batch.pending) for batch in self._pending_dflash_completions)
        pending_tasks = len(self._pending) + active_feature_tasks
        update = DraftCapacityUpdate(
            protocol_version=PROTOCOL_VERSION,
            draft_instance_id=self.instance_id,
            pending_tasks=pending_tasks,
            max_pending_tasks=self.max_pending_tasks,
            free_result_rows=len(self._free_result_slots) * self.max_batch_size,
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
        )
        if not self._client.update_capacity(update):
            logger.warning("Draft capacity update was locally backpressured")

    def _send_result(self, message) -> None:
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

        now_ns = time.monotonic_ns()
        for operation, message_id, message in self._delivery_outbox.due(now_ns=now_ns):
            if operation == "result":
                sent = self._client.publish_results(message)
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
    parser.add_argument("--result-slot-count", type=int, default=16)
    parser.add_argument("--data-dir", default="/dev/shm")
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--registration-timeout-s", type=float, default=10.0)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--enable-deterministic-inference", action="store_true")
    parser.add_argument("--batch-source-read-done", action="store_true")
    parser.add_argument("--pinned-pivot-staging", action="store_true")
    parser.add_argument("--feature-buffer-max-tokens", type=int, default=8192)
    parser.add_argument("--feature-buffer-slot-count", type=int, default=2)
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
