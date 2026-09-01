from __future__ import annotations

import argparse
import bisect
import json
import logging
import os
import re
import signal
import time
from collections import deque
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch
from dt_client import RetryOutbox
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import bootstrap
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.draft_worker_common import build_draft_tp_worker
from sglang.srt.speculative.flashmtp_utils import parse_flashmtp_draft_config

logger = logging.getLogger(__name__)

IDLE_POLL_S = 0.0005
COALESCING_POLL_S = 0.0001
CONTROL_RETRY_INTERVAL_NS = 50_000_000


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
    range_transfer_ids: dict[tuple[int, int, int, int], frozenset[str]]


@dataclass(slots=True)
class CapturedDraftBatch:
    graph: object
    pivots: torch.Tensor
    bonus_tokens: torch.Tensor
    block_ids: torch.Tensor
    candidates: torch.Tensor


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
    """One-GPU FlashMTP engine with an independent continuous batch queue."""

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

        self.EntryStatus = EntryStatus
        self.instance_id = instance_id
        self.profile = load_draft_profile(profile_path)
        self.gpu_id = gpu_id
        self.max_pending_tasks = max_pending_tasks
        self.max_batch_size = max_batch_size
        self.batch_ready_threshold = batch_ready_threshold
        self.batch_wait_s = batch_wait_ms / 1_000
        self.batch_source_read_done = batch_source_read_done
        self.pinned_pivot_staging = pinned_pivot_staging
        self.enable_deterministic_inference = bool(enable_deterministic_inference)
        self._result_row_width = self.profile.block_size
        self._pending: deque[object] = deque()
        self._pending_ids: set[str] = set()
        self._received_ns_by_transfer: dict[str, int] = {}
        self._cancelled_ids: set[str] = set()
        self._source_stores: dict[str, object] = {}
        self._published_by_batch: dict[str, PublishedResult] = {}
        self._published_by_range: dict[tuple[int, int, int, int], str] = {}
        self._delivery_outbox = RetryOutbox[object](
            retry_interval_ns=CONTROL_RETRY_INTERVAL_NS
        )
        self._source_done_pending: deque[object] = deque()
        self._free_result_slots = deque(range(result_slot_count))
        self._result_sequence = 0
        self._source_done_sequence = 0
        self._stop = False

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
            )
            self._initialize_model(
                nccl_port=nccl_port,
                mem_fraction_static=mem_fraction_static,
                disable_cuda_graph=disable_cuda_graph,
            )
            self._initialize_pivot_staging()
            self._client.register_draft(timeout_s=registration_timeout_s)
            self._publish_capacity()
        except Exception:
            if self._client is not None:
                self._client.close()
                self._client = None
            self._result_store.close()
            self._result_store.unlink()
            raise

    def _initialize_pivot_staging(self) -> None:
        self._pivot_host_staging = None
        self._pivot_host_views = ()
        if not self.pinned_pivot_staging:
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
        server_args_kwargs = {
            "model_path": self.profile.target_model_path,
            "tokenizer_path": self.profile.target_model_path,
            "skip_tokenizer_init": True,
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

    def run(self) -> None:
        logger.info("DT Draft engine serving")
        while not self._stop:
            self._poll_messages()
            batch = self._take_batch()
            if batch:
                self._execute_batch(batch)
            else:
                sleep_s = self._idle_sleep_s()
                if sleep_s > 0:
                    time.sleep(sleep_s)

    def stop(self) -> None:
        self._stop = True

    def close(self) -> None:
        for store in self._source_stores.values():
            store.close()
        self._source_stores.clear()
        self._client.close()
        self._result_store.close()
        self._result_store.unlink()

    def _poll_messages(self) -> None:
        from dt_protocol import (
            CancelTransfers,
            ControlAck,
            DraftDispatchBatch,
            ErrorMessage,
            PullDone,
            ResultBatchAck,
        )

        capacity_changed = False
        self._flush_source_done()
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
                self._delivery_outbox.ack(message.operation, message.message_id)
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
        if capacity_changed:
            self._drop_cancelled_pending()
            self._publish_capacity()

    def _accept_dispatch(self, message) -> bool:
        received_ns = time.monotonic_ns()
        accepted = False
        for task in message.entries:
            if task.transfer_id in self._pending_ids:
                continue
            if len(self._pending) >= self.max_pending_tasks:
                raise RuntimeError(
                    "Coordinator exceeded advertised Draft queue capacity"
                )
            self._validate_task(task)
            self._pending.append(task)
            self._pending_ids.add(task.transfer_id)
            self._received_ns_by_transfer[task.transfer_id] = received_ns
            accepted = True
        return accepted

    def _take_batch(self) -> list[PendingDraftTask]:
        if not self._pending or not self._free_result_slots:
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
        while self._pending and len(queued) < self.max_batch_size:
            task = self._pending.popleft()
            self._pending_ids.discard(task.transfer_id)
            self._received_ns_by_transfer.pop(task.transfer_id, None)
            if task.transfer_id in self._cancelled_ids:
                self._cancelled_ids.discard(task.transfer_id)
                continue
            queued.append(
                PendingDraftTask(
                    task=task,
                    pivot_bytes=None,
                )
            )

        # The result publisher needs rows grouped by Target. Sort before direct
        # mmap reads so pinned staging is already laid out in publish order.
        queued = target_major(queued)
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
            ),
        )
        range_transfer_ids = {}
        for target_id in dict.fromkeys(task.target_instance_id for task in tasks):
            rows = [
                index
                for index, task in enumerate(tasks)
                if task.target_instance_id == target_id
            ]
            key = (slot_id, generation, rows[0], len(rows))
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
        source = task.source
        if task.draft_profile_id != self.profile.profile_id:
            raise ValueError("Draft task profile mismatch")
        if task.k != self.profile.draft_k:
            raise ValueError("Draft task k mismatch")
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

    def _handle_pull_done(self, message) -> bool:
        key = (
            message.buffer_id,
            message.generation,
            message.row_offset,
            message.row_count,
        )
        batch_id = self._published_by_range.pop(key, None)
        if batch_id is None:
            return False
        published = self._published_by_batch[batch_id]
        published.range_transfer_ids.pop(key, None)
        if published.range_transfer_ids:
            return False
        self._release_result(published)
        return True

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

        update = DraftCapacityUpdate(
            protocol_version=PROTOCOL_VERSION,
            draft_instance_id=self.instance_id,
            pending_tasks=len(self._pending),
            max_pending_tasks=self.max_pending_tasks,
            free_result_rows=len(self._free_result_slots) * self.max_batch_size,
            healthy=not self._stop,
            observed_ns=time.monotonic_ns(),
        )
        if not self._client.update_capacity(update):
            logger.warning("Draft capacity update was locally backpressured")

    def _send_result(self, message) -> None:
        self._delivery_outbox.add("result", message.batch_id, message)
        self._flush_reliable_outbox()

    def _flush_reliable_outbox(self) -> None:
        from dt_protocol import SourceReadDoneBatch

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
            self._delivery_outbox.add(
                SOURCE_READ_DONE, source_read_done_id(message), message
            )
        self._flush_reliable_outbox()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SGLang DT FlashMTP Draft engine")
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
