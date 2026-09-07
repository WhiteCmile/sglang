from __future__ import annotations

import argparse
import bisect
import ctypes
import json
import logging
import os
import re
import signal
import time
from collections import Counter, deque
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
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
from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.draft_worker_common import build_draft_tp_worker
from sglang.srt.speculative.flashmtp_utils import parse_flashmtp_draft_config

logger = logging.getLogger(__name__)

IDLE_POLL_S = 0.0005
COALESCING_POLL_S = 0.0001
CONTROL_RETRY_INTERVAL_NS = 50_000_000
_SHARED_OPTIMIZATIONS_ENV = "SGLANG_FLASHMTP_SHARED_OPTIMIZATIONS"
_ADAPTIVE_VERIFY_ENV = "SGLANG_FLASHMTP_ADAPTIVE_VERIFY"
_BUDGETED_VERIFY_ENV = "SGLANG_FLASHMTP_BUDGETED_VERIFY"
_BUDGETED_VERIFY_THRESHOLDS_ENV = "SGLANG_FLASHMTP_BUDGETED_VERIFY_THRESHOLDS"


def flashmtp_component_enabled(env_name: str) -> bool:
    """Match Target-side component override and legacy-bundle semantics."""
    budgeted = os.environ.get(_BUDGETED_VERIFY_ENV, "0")
    if budgeted not in ("0", "1"):
        raise ValueError(f"{_BUDGETED_VERIFY_ENV} must be 0 or 1")
    if budgeted == "1" and env_name == _ADAPTIVE_VERIFY_ENV:
        return True
    raw = os.environ.get(env_name)
    if raw is None:
        return os.environ.get(_SHARED_OPTIMIZATIONS_ENV, "0") == "1"
    if raw not in ("0", "1"):
        raise ValueError(f"{env_name} must be 0 or 1")
    return raw == "1"


# Calibrated on the full-width confidence control trace at
# ``runs/dt-confidence-full8-profile-1t1d-c256-n512-r1-20260815``.  The model
# predicts whether the token immediately after an adaptive verify cutoff will
# be accepted.  It is intentionally gated to the block-8 DT experiment: the
# score is a scheduling hint, not part of the correctness contract.
_PADDING_PRIORITY_BLOCK_SIZE = 8
_PADDING_PRIORITY_MARGIN_MEAN = (
    5.312305310055057,
    4.299847417415242,
    3.684364586351782,
    3.357133801796581,
    2.90088017965807,
    2.809364586351782,
    2.6218057266009853,
)
_PADDING_PRIORITY_MARGIN_STD = (
    4.757945839400196,
    4.078800898015133,
    3.477991166682023,
    3.2262153793352852,
    2.9655024569448787,
    2.836258964922945,
    2.689668917526257,
)
_PADDING_PRIORITY_WEIGHTS = (
    (2.329, 2.349, -0.009, 0.032, -0.019, 0.059, -0.038, 0.080),
    (0.548, 0.776, 0.984, -0.153, 0.090, 0.105, -0.027, 0.168),
    (-0.540, 0.431, 0.309, 0.658, 0.167, 0.178, -0.043, 0.275),
    (-1.449, 0.268, 0.304, 0.455, 0.655, 0.150, -0.007, 0.301),
    (-2.127, 0.266, 0.242, 0.430, 0.393, 0.519, 0.067, 0.295),
    (-2.829, 0.168, 0.190, 0.267, 0.404, 0.413, 0.490, 0.323),
    (-3.205, 0.108, 0.178, 0.213, 0.318, 0.454, 0.404, 0.539),
)
_PADDING_PRIORITY_SCALE = 1 << 20


def select_adaptive_verify_lens(
    logit_margins: torch.Tensor,
    *,
    threshold: float | Sequence[float] | torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Choose the first low-confidence proposal as the verify cutoff.

    ``verify_len`` includes the anchor token, so a low-confidence proposal at
    position zero maps to a one-token Target verify.  If every proposal clears
    the threshold, verify the full block.
    """
    if logit_margins.ndim != 2:
        raise ValueError("logit_margins must have shape [batch, proposals]")
    if logit_margins.shape[1] != block_size - 1:
        raise ValueError(
            "logit margin width must equal block_size - 1: "
            f"got {logit_margins.shape[1]} for block_size={block_size}"
        )
    thresholds = torch.as_tensor(
        threshold, dtype=logit_margins.dtype, device=logit_margins.device
    )
    if thresholds.numel() not in (1, block_size - 1):
        raise ValueError(
            "threshold must be scalar or have one value per proposal position: "
            f"got {thresholds.numel()} values for block_size={block_size}"
        )
    below = logit_margins < thresholds
    first_below = torch.argmax(below.to(torch.int32), dim=1) + 1
    return torch.where(
        torch.any(below, dim=1),
        first_below,
        torch.full_like(first_below, block_size),
    ).to(torch.int32)


def _build_padding_priority_model(
    *, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(
            _PADDING_PRIORITY_MARGIN_MEAN,
            dtype=torch.float32,
            device=device,
        ),
        torch.tensor(
            _PADDING_PRIORITY_MARGIN_STD,
            dtype=torch.float32,
            device=device,
        ),
        torch.tensor(
            _PADDING_PRIORITY_WEIGHTS,
            dtype=torch.float32,
            device=device,
        ),
    )


def score_adaptive_padding_priorities(
    logit_margins: torch.Tensor,
    verify_lens: torch.Tensor,
    *,
    block_size: int,
    model: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Return a rank-only int32 hint for reclaiming one paid padding token.

    The sigmoid is monotonic and therefore omitted.  Quantization keeps the
    result row int32-only while preserving substantially more ordering
    precision than the scheduler needs.  Full-width rows use INT32_MIN because
    they cannot consume another token.
    """
    if block_size != _PADDING_PRIORITY_BLOCK_SIZE:
        raise ValueError(
            "adaptive padding priority is calibrated only for block_size=8"
        )
    if logit_margins.ndim != 2 or logit_margins.shape[1] != block_size - 1:
        raise ValueError("logit_margins must have shape [batch, block_size - 1]")
    if verify_lens.ndim != 1 or verify_lens.shape[0] != logit_margins.shape[0]:
        raise ValueError("verify_lens must have one value per confidence row")

    if model is None:
        model = _build_padding_priority_model(device=logit_margins.device)
    means, stds, weights = model
    normalized = (torch.clamp(logit_margins.float(), 0.0, 16.0) - means) / stds
    model_rows = weights.index_select(
        0,
        (verify_lens.to(torch.int64) - 1).clamp_(0, block_size - 2),
    )
    scores = model_rows[:, 0] + torch.sum(model_rows[:, 1:] * normalized, dim=1)
    quantized = torch.round(scores * _PADDING_PRIORITY_SCALE).clamp_(
        -2147483647, 2147483647
    )
    return torch.where(
        verify_lens < block_size,
        quantized,
        torch.full_like(quantized, -2147483648),
    ).to(torch.int32)


def parse_adaptive_verify_thresholds(
    raw: str | None, *, proposal_count: int
) -> tuple[float, ...] | None:
    if raw is None:
        return None
    try:
        thresholds = tuple(float(value.strip()) for value in raw.split(","))
    except ValueError as exc:
        raise ValueError(
            "SGLANG_DT_ADAPTIVE_VERIFY_THRESHOLDS must contain comma-separated numbers"
        ) from exc
    if len(thresholds) != proposal_count:
        raise ValueError(
            "SGLANG_DT_ADAPTIVE_VERIFY_THRESHOLDS must have one value per "
            f"proposal position: got {len(thresholds)}, expected {proposal_count}"
        )
    if any(threshold < 0 for threshold in thresholds):
        raise ValueError(
            "SGLANG_DT_ADAPTIVE_VERIFY_THRESHOLDS values must be non-negative"
        )
    return thresholds


@dataclass(slots=True)
class PendingDraftTask:
    task: object
    pivot_bytes: bytes | None
    received_ns: int = 0
    pivot_read_ns: int = 0
    pivot_row: int = -1
    pivot_buffer: int = 0
    dispatch_batch_id: str = ""


@dataclass(slots=True)
class PublishedResult:
    batch_id: str
    slot_id: int
    generation: int
    transfer_ids: tuple[str, ...]
    transfer_rows: dict[str, tuple[str, int]]
    range_transfer_ids: dict[tuple[int, int, int, int], frozenset[str]]


@dataclass(slots=True, frozen=True)
class DirectMmapH2DRun:
    destination_row: int
    row_count: int
    uri: str
    first_slot: int
    slot_step: int


@dataclass(slots=True)
class CapturedDraftBatch:
    graph: object
    pivots: torch.Tensor
    bonus_tokens: torch.Tensor
    block_ids: torch.Tensor
    candidates: torch.Tensor
    logit_margins: torch.Tensor | None
    verify_lens: torch.Tensor | None
    padding_priorities: torch.Tensor | None


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
        num_proposals: int,
        mask_token_id: int,
        local_positions: torch.Tensor,
        max_batch_size: int,
        profile_confidence: bool = False,
        confidence_mode: str = "margin",
        adaptive_verify_threshold: float | Sequence[float] | None = None,
        padding_priority_model: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None
        ) = None,
    ) -> None:
        self.draft_model = draft_model
        self.token_embedding = token_embedding
        self.device = device
        self.pivot_width = int(pivot_width)
        self.draft_block_len = int(draft_block_len)
        self.block_size = int(block_size)
        self.num_proposals = int(num_proposals)
        self.mask_token_id = int(mask_token_id)
        self.local_positions = local_positions
        self.profile_confidence = bool(profile_confidence)
        self.confidence_mode = confidence_mode
        self.adaptive_verify_threshold = (
            torch.as_tensor(
                adaptive_verify_threshold,
                dtype=torch.bfloat16,
                device=device,
            )
            if adaptive_verify_threshold is not None
            else None
        )
        self.padding_priority_model = padding_priority_model
        if padding_priority_model is not None and adaptive_verify_threshold is None:
            raise ValueError(
                "padding priority CUDA graph output requires adaptive verify"
            )
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
            logit_margins = (
                torch.empty(
                    (batch_size, self.num_proposals),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                if self.profile_confidence
                else None
            )
            verify_lens = (
                torch.empty(
                    (batch_size,),
                    dtype=torch.int32,
                    device=self.device,
                )
                if self.adaptive_verify_threshold is not None
                else None
            )
            padding_priorities = (
                torch.empty(
                    (batch_size,),
                    dtype=torch.int32,
                    device=self.device,
                )
                if self.padding_priority_model is not None
                else None
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
                logit_margins=logit_margins,
                verify_lens=verify_lens,
                padding_priorities=padding_priorities,
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
                logit_margins=logit_margins,
                verify_lens=verify_lens,
                padding_priorities=padding_priorities,
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
        logit_margins: torch.Tensor | None,
        verify_lens: torch.Tensor | None,
        padding_priorities: torch.Tensor | None,
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
        if logit_margins is None:
            proposals = self.draft_model.sample_draft_tokens(
                block_hidden=block_hidden,
                first_prev_token_ids=bonus_tokens,
            )
        else:
            confidence_sampler = {
                "margin": self.draft_model.sample_draft_tokens_with_margin,
                "top1": self.draft_model.sample_draft_tokens_with_top1,
                "top1_trace": self.draft_model.sample_draft_tokens_with_top1_trace,
            }[self.confidence_mode]
            proposals, margins = confidence_sampler(
                block_hidden=block_hidden, first_prev_token_ids=bonus_tokens
            )
            logit_margins.copy_(margins)
            if verify_lens is not None:
                selected_lens = select_adaptive_verify_lens(
                    margins,
                    threshold=self.adaptive_verify_threshold,
                    block_size=self.block_size,
                )
                verify_lens.copy_(selected_lens)
                if padding_priorities is not None:
                    assert self.padding_priority_model is not None
                    padding_priorities.copy_(
                        score_adaptive_padding_priorities(
                            margins,
                            selected_lens,
                            block_size=self.block_size,
                            model=self.padding_priority_model,
                        )
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
        if self.profile_confidence:
            confidence_sampler = {
                "margin": self.draft_model.sample_draft_tokens_with_margin,
                "top1": self.draft_model.sample_draft_tokens_with_top1,
                "top1_trace": self.draft_model.sample_draft_tokens_with_top1_trace,
            }[self.confidence_mode]
            return confidence_sampler(
                block_hidden=block_hidden, first_prev_token_ids=bonus_tokens
            )[0]
        return self.draft_model.sample_draft_tokens(
            block_hidden=block_hidden,
            first_prev_token_ids=bonus_tokens,
        )

    def replay(
        self, *, pivots: torch.Tensor, bonus_tokens: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        int,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
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
        margins = (
            captured.logit_margins[:batch_size]
            if captured.logit_margins is not None
            else None
        )
        verify_lens = (
            captured.verify_lens[:batch_size]
            if captured.verify_lens is not None
            else None
        )
        padding_priorities = (
            captured.padding_priorities[:batch_size]
            if captured.padding_priorities is not None
            else None
        )
        return (
            captured.candidates[:batch_size],
            padded_batch_size,
            margins,
            verify_lens,
            padding_priorities,
        )


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


def direct_mmap_h2d_runs(
    tasks: Sequence[PendingDraftTask],
) -> tuple[DirectMmapH2DRun, ...]:
    """Coalesce equal-stride source slots into pitched H2D transfers."""
    runs = []
    start = 0
    while start < len(tasks):
        source = tasks[start].task.source
        end = start + 1
        slot_step = 1
        if end < len(tasks) and tasks[end].task.source.uri == source.uri:
            candidate_step = (
                tasks[end].task.source.source_slot_id - source.source_slot_id
            )
            if candidate_step > 0:
                slot_step = candidate_step
                end += 1
                while (
                    end < len(tasks)
                    and tasks[end].task.source.uri == source.uri
                    and tasks[end].task.source.source_slot_id
                    - tasks[end - 1].task.source.source_slot_id
                    == slot_step
                ):
                    end += 1
        runs.append(
            DirectMmapH2DRun(
                destination_row=start,
                row_count=end - start,
                uri=source.uri,
                first_slot=source.source_slot_id,
                slot_step=slot_step,
            )
        )
        start = end
    return tuple(runs)


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
            from dt_client import DraftCoordinatorClient, DraftCoordinatorIoPump
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
        self._verify_block_size = self.profile.block_size
        self.gpu_id = gpu_id
        self.max_pending_tasks = max_pending_tasks
        self.max_batch_size = max_batch_size
        self.batch_ready_threshold = batch_ready_threshold
        self.batch_wait_s = batch_wait_ms / 1_000
        self.batch_source_read_done = batch_source_read_done
        self.pinned_pivot_staging = pinned_pivot_staging
        self.enable_deterministic_inference = bool(enable_deterministic_inference)
        self._direct_mmap_h2d = (
            os.environ.get("SGLANG_DT_DRAFT_DIRECT_MMAP_H2D", "0") == "1"
        )
        budgeted_verify = os.environ.get(_BUDGETED_VERIFY_ENV, "0") == "1"
        self._prepare_overlap = (
            os.environ.get("SGLANG_DT_DRAFT_PREPARE_OVERLAP", "0") == "1"
            or budgeted_verify
        )
        prepare_wait_ms = float(
            os.environ.get(
                "SGLANG_DT_DRAFT_PREPARE_WAIT_MS",
                "2" if budgeted_verify else "0",
            )
        )
        if prepare_wait_ms < 0:
            raise ValueError("SGLANG_DT_DRAFT_PREPARE_WAIT_MS must be non-negative")
        if prepare_wait_ms > 0 and not self._prepare_overlap:
            raise ValueError(
                "SGLANG_DT_DRAFT_PREPARE_WAIT_MS requires Draft prepare overlap"
            )
        self._prepare_wait_ns = round(prepare_wait_ms * 1_000_000)
        self._prepare_until_gpu_done = (
            os.environ.get("SGLANG_DT_DRAFT_PREPARE_UNTIL_GPU_DONE", "0") == "1"
        )
        self._coordinator_io_pump = (
            os.environ.get("SGLANG_DT_DRAFT_COORDINATOR_IO_PUMP", "0") == "1"
        )
        self._async_prepare = (
            os.environ.get("SGLANG_DT_DRAFT_ASYNC_PREPARE", "0") == "1"
        )
        async_prepare_wait_ms = float(
            os.environ.get(
                "SGLANG_DT_DRAFT_ASYNC_PREPARE_WAIT_MS",
                str(prepare_wait_ms),
            )
        )
        if async_prepare_wait_ms < 0:
            raise ValueError(
                "SGLANG_DT_DRAFT_ASYNC_PREPARE_WAIT_MS must be non-negative"
            )
        self._async_prepare_wait_ns = round(async_prepare_wait_ms * 1_000_000)
        self._prepare_executor = (
            ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"dt-draft-prepare-{instance_id}",
            )
            if self._async_prepare
            else None
        )
        if self._prepare_overlap and not self.pinned_pivot_staging:
            raise ValueError(
                "SGLANG_DT_DRAFT_PREPARE_OVERLAP requires pinned pivot staging"
            )
        if self._async_prepare and not self._prepare_overlap:
            raise ValueError(
                "SGLANG_DT_DRAFT_ASYNC_PREPARE requires Draft prepare overlap"
            )
        if self._async_prepare and not self._coordinator_io_pump:
            raise ValueError(
                "SGLANG_DT_DRAFT_ASYNC_PREPARE requires the Coordinator I/O pump"
            )
        self._profile_confidence = (
            os.environ.get("SGLANG_DT_DRAFT_CONFIDENCE_PROFILE", "0") == "1"
        )
        self._confidence_mode = (
            os.environ.get("SGLANG_DT_DRAFT_CONFIDENCE_MODE", "margin").strip().lower()
        )
        if self._confidence_mode not in ("margin", "top1", "top1_trace"):
            raise ValueError(
                "SGLANG_DT_DRAFT_CONFIDENCE_MODE must be 'margin', 'top1', "
                "or 'top1_trace'"
            )
        self._dispatch_batch_isolation_min_rows = int(
            os.environ.get(
                "SGLANG_DT_DRAFT_DISPATCH_BATCH_ISOLATION_MIN_ROWS",
                "0",
            )
        )
        if self._dispatch_batch_isolation_min_rows < 0:
            raise ValueError(
                "SGLANG_DT_DRAFT_DISPATCH_BATCH_ISOLATION_MIN_ROWS must be non-negative"
            )
        self._dispatch_batch_isolation_min_next_rows = int(
            os.environ.get(
                "SGLANG_DT_DRAFT_DISPATCH_BATCH_ISOLATION_MIN_NEXT_ROWS",
                "0",
            )
        )
        if self._dispatch_batch_isolation_min_next_rows < 0:
            raise ValueError(
                "SGLANG_DT_DRAFT_DISPATCH_BATCH_ISOLATION_MIN_NEXT_ROWS must "
                "be non-negative"
            )
        adaptive_component = flashmtp_component_enabled(_ADAPTIVE_VERIFY_ENV)
        adaptive_margin = os.environ.get("SGLANG_DT_ADAPTIVE_VERIFY_MARGIN")
        self._adaptive_verify_margin = (
            0.0
            if adaptive_component
            else (float(adaptive_margin) if adaptive_margin is not None else None)
        )
        if (
            self._adaptive_verify_margin is not None
            and self._adaptive_verify_margin < 0
        ):
            raise ValueError("SGLANG_DT_ADAPTIVE_VERIFY_MARGIN must be non-negative")
        shared_thresholds = os.environ.get(
            _BUDGETED_VERIFY_THRESHOLDS_ENV,
            os.environ.get(
                "SGLANG_FLASHMTP_ADAPTIVE_VERIFY_THRESHOLDS",
                "0.625,1.25,1.375,3.875,1.25,3.125,3.625",
            ),
        )
        self._adaptive_verify_thresholds = parse_adaptive_verify_thresholds(
            shared_thresholds
            if adaptive_component
            else os.environ.get("SGLANG_DT_ADAPTIVE_VERIFY_THRESHOLDS"),
            proposal_count=self.profile.block_size - 1,
        )
        if (
            self._adaptive_verify_thresholds is not None
            and self._adaptive_verify_margin is None
        ):
            raise ValueError(
                "SGLANG_DT_ADAPTIVE_VERIFY_THRESHOLDS requires "
                "SGLANG_DT_ADAPTIVE_VERIFY_MARGIN to enable the adaptive payload"
            )
        self._adaptive_padding_priority = (
            os.environ.get("SGLANG_DT_ADAPTIVE_PADDING_PRIORITY", "0") == "1"
        )
        if self._adaptive_padding_priority and self._adaptive_verify_margin is None:
            raise ValueError(
                "SGLANG_DT_ADAPTIVE_PADDING_PRIORITY requires "
                "SGLANG_DT_ADAPTIVE_VERIFY_MARGIN"
            )
        if self._adaptive_padding_priority and self._confidence_mode != "margin":
            raise ValueError(
                "SGLANG_DT_ADAPTIVE_PADDING_PRIORITY requires margin confidence"
            )
        if (
            self._adaptive_padding_priority
            and self.profile.block_size != _PADDING_PRIORITY_BLOCK_SIZE
        ):
            raise ValueError(
                "SGLANG_DT_ADAPTIVE_PADDING_PRIORITY is calibrated only for "
                "block_size=8"
            )
        self._collect_confidence = (
            self._profile_confidence or self._adaptive_verify_margin is not None
        )
        self._result_row_width = (
            self._verify_block_size
            + int(self._adaptive_verify_margin is not None)
            + int(self._adaptive_padding_priority)
        )
        self._padding_priority_model = None
        self._pending: deque[object] = deque()
        self._pending_ids: set[str] = set()
        self._received_ns_by_transfer: dict[str, int] = {}
        # This map is profile metadata in the default path and scheduler
        # metadata when dispatch-boundary isolation is enabled.
        self._profile_dispatch_batch_by_transfer: dict[str, str] | None = (
            {}
            if os.environ.get("DT_PROFILE_DIR")
            or self._dispatch_batch_isolation_min_rows > 0
            else None
        )
        self._cancelled_ids: set[str] = set()
        self._source_stores: dict[str, object] = {}
        self._direct_source_mapping_views: dict[str, memoryview] = {}
        self._direct_source_mapping_addresses: dict[str, int] = {}
        self._direct_cuda_rt = None
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
            if self._coordinator_io_pump:
                self._client = DraftCoordinatorIoPump(self._client)
            self._publish_capacity()
        except Exception:
            if self._prepare_executor is not None:
                self._prepare_executor.shutdown(wait=True, cancel_futures=True)
            if self._client is not None:
                self._client.close()
                self._client = None
            self._result_store.close()
            self._result_store.unlink()
            raise

    def _initialize_pivot_staging(self) -> None:
        self._pivot_host_staging = None
        self._pivot_host_views = ()
        self._pivot_host_staging_buffers = ()
        self._pivot_host_view_buffers = ()
        if not self.pinned_pivot_staging:
            return
        buffer_count = 2 if self._prepare_overlap else 1
        self._pivot_host_staging_buffers = tuple(
            torch.empty(
                (self.max_batch_size, self._pivot_width),
                dtype=torch.bfloat16,
                device="cpu",
                pin_memory=True,
            )
            for _ in range(buffer_count)
        )
        self._pivot_host_view_buffers = tuple(
            tuple(memoryview(row.view(torch.uint8).numpy()) for row in staging_buffer)
            for staging_buffer in self._pivot_host_staging_buffers
        )
        # Keep the original attributes as aliases for tests and for the serial
        # path.  The overlap path selects the other ring entry explicitly.
        self._pivot_host_staging = self._pivot_host_staging_buffers[0]
        self._pivot_host_views = self._pivot_host_view_buffers[0]
        logger.info(
            "Enabled pinned DT pivot staging: buffers=%d shape=%s bytes=%d",
            buffer_count,
            tuple(self._pivot_host_staging.shape),
            buffer_count
            * self._pivot_host_staging.numel()
            * self._pivot_host_staging.element_size(),
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
        if self._adaptive_padding_priority:
            self._padding_priority_model = _build_padding_priority_model(
                device=self.device
            )

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
                num_proposals=self._num_proposals,
                mask_token_id=self._mask_token_id,
                local_positions=self._local_positions,
                max_batch_size=self.max_batch_size,
                profile_confidence=self._collect_confidence,
                confidence_mode=self._confidence_mode,
                adaptive_verify_threshold=(
                    self._adaptive_verify_thresholds
                    if self._adaptive_verify_thresholds is not None
                    else self._adaptive_verify_margin
                ),
                padding_priority_model=self._padding_priority_model,
            )
            graph_runner.capture()
            self._draft_cuda_graph = graph_runner
            logger.info(
                "Captured standalone Draft CUDA graphs: batch_sizes=%s elapsed=%.2fs",
                graph_runner.batch_sizes,
                time.monotonic() - capture_start,
            )
        logger.info(
            "DT Draft model ready: instance=%s profile=%s model_block=%d "
            "verify_block=%d batch=%d embedding_shape=%s cuda_graph=%s",
            self.instance_id,
            self.profile.profile_id,
            self.profile.block_size,
            self._verify_block_size,
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
        if self._prepare_overlap and not cuda_sync_enabled():
            self._run_prepare_overlap()
            return
        while not self._stop:
            self._poll_messages()
            batch = self._take_batch()
            if batch:
                self._execute_batch(batch)
            else:
                sleep_s = self._idle_sleep_s()
                if sleep_s > 0:
                    time.sleep(sleep_s)

    def _run_prepare_overlap(self) -> None:
        """Prepare batch n+1 on CPU while batch n is executing on GPU.

        CUDA graph replay is asynchronous until the result copy reaches CPU.
        The inline mode keeps Coordinator/ZMQ access on this thread. The async
        mode uses a dedicated socket owner plus one persistent prepare worker,
        allowing mmap reads to overlap the current result D2H. A second pinned
        host buffer prevents those reads from racing the current H2D copy.
        """
        current: list[PendingDraftTask] = []
        current_buffer = 0
        while not self._stop:
            self._poll_messages()
            if not current:
                current = self._take_batch(pivot_buffer=current_buffer)
                if not current:
                    sleep_s = self._idle_sleep_s()
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    continue

            next_batch: list[PendingDraftTask] = []
            next_buffer = 1 - current_buffer

            if getattr(self, "_async_prepare", False):

                def prepare_next(
                    batch: list[PendingDraftTask] = next_batch,
                    buffer_index: int = next_buffer,
                    *,
                    gpu_done_event=None,
                ):
                    return self._start_async_prepare(
                        batch=batch,
                        pivot_buffer=buffer_index,
                        gpu_done_event=gpu_done_event,
                    )

            else:

                def prepare_next(
                    batch: list[PendingDraftTask] = next_batch,
                    buffer_index: int = next_buffer,
                    *,
                    gpu_done_event=None,
                ) -> int:
                    batch.extend(
                        self._take_overlap_batch(
                            pivot_buffer=buffer_index,
                            gpu_done_event=gpu_done_event,
                        )
                    )
                    return len(batch)

            self._execute_batch(current, overlap_callback=prepare_next)
            current = next_batch
            current_buffer = next_buffer

    def _start_async_prepare(
        self, *, batch: list[PendingDraftTask], pivot_buffer: int, gpu_done_event=None
    ):
        """Prepare the next pinned pivot buffer alongside current result D2H."""
        started_ns = time.monotonic_ns()
        finished_ns = []
        errors: list[Exception] = []

        def prepare() -> None:
            try:
                batch.extend(
                    self._take_overlap_batch(
                        pivot_buffer=pivot_buffer,
                        wait_ns=self._async_prepare_wait_ns,
                        dispatch_only=True,
                        gpu_done_event=gpu_done_event,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - re-raised on engine thread
                errors.append(exc)
            finally:
                finished_ns.append(time.monotonic_ns())

        executor = self._prepare_executor
        if executor is None:
            raise RuntimeError("async Draft prepare executor is not initialized")
        future = executor.submit(prepare)

        def finish() -> tuple[int, int]:
            future.result()
            if errors:
                raise errors[0]
            return len(batch), finished_ns[0] - started_ns

        return finish

    def _take_overlap_batch(
        self,
        *,
        pivot_buffer: int,
        wait_ns: int | None = None,
        dispatch_only: bool = False,
        gpu_done_event=None,
    ) -> list[PendingDraftTask]:
        """Poll for work during the current CUDA graph's otherwise idle CPU time."""
        if wait_ns is None:
            wait_ns = getattr(self, "_prepare_wait_ns", 0)
        deadline_ns = time.monotonic_ns() + wait_ns
        while True:
            if dispatch_only:
                self._poll_dispatch_messages()
            else:
                self._poll_messages()
            # The current result has not claimed its slot yet.  Keep one slot
            # for it and require another for the prepared batch.
            if len(self._free_result_slots) < 2:
                return []
            batch = self._take_batch(pivot_buffer=pivot_buffer)
            if batch:
                return batch
            if gpu_done_event is not None:
                # The event is recorded after every result-row packing kernel.
                # Stop immediately when D2H can make progress, so extending the
                # prepare window cannot add a fixed scheduler wait to short or
                # tail Draft graphs.
                if gpu_done_event.query() or self._stop:
                    return []
                time.sleep(COALESCING_POLL_S)
                continue
            now_ns = time.monotonic_ns()
            if now_ns >= deadline_ns or self._stop:
                return []
            time.sleep(min(COALESCING_POLL_S, (deadline_ns - now_ns) / 1_000_000_000))

    def _poll_dispatch_messages(self) -> None:
        recv_dispatch = getattr(self._client, "recv_dispatch_nowait", None)
        if recv_dispatch is None:
            raise RuntimeError(
                "background Draft prepare requires a Coordinator I/O pump"
            )
        capacity_changed = False
        for message in recv_dispatch():
            capacity_changed = self._accept_dispatch(message) or capacity_changed
        if capacity_changed:
            self._drop_cancelled_pending()
            self._publish_capacity()

    def stop(self) -> None:
        self._stop = True

    def close(self) -> None:
        if self._prepare_executor is not None:
            self._prepare_executor.shutdown(wait=True, cancel_futures=True)
            self._prepare_executor = None
        for uri, address in self._direct_source_mapping_addresses.items():
            result = torch.cuda.cudart().cudaHostUnregister(address)
            if result != torch.cuda.cudart().cudaError.success:
                logger.warning(
                    "cudaHostUnregister failed for DT Draft source mmap %s: %s",
                    uri,
                    result,
                )
        self._direct_source_mapping_addresses.clear()
        for mapping_view in self._direct_source_mapping_views.values():
            mapping_view.release()
        self._direct_source_mapping_views.clear()
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
                if message.code == "PermissionError" and hasattr(
                    self._client, "register_draft"
                ):
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
        pending_before = len(self._pending)
        profile_dispatch_batches = self._profile_dispatch_batch_by_transfer
        if profile_dispatch_batches is not None:
            dispatch_batch_id = message.batch_id.removesuffix(f":{self.instance_id}")
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
            if profile_dispatch_batches is not None:
                profile_dispatch_batches[task.transfer_id] = dispatch_batch_id
            accepted = True
        profile_emit(
            "draft_dispatch_received",
            instance_id=self.instance_id,
            batch_id=message.batch_id,
            entries=len(message.entries),
            coordinator_to_draft_ns=received_ns - message.sent_ns,
            pending_before=pending_before,
            pending_after=len(self._pending),
        )
        return accepted

    def _take_batch(self, *, pivot_buffer: int = 0) -> list[PendingDraftTask]:
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

        take_start_ns = time.monotonic_ns()
        profile_dispatch_batches = getattr(
            self, "_profile_dispatch_batch_by_transfer", None
        )
        isolation_min_rows = getattr(self, "_dispatch_batch_isolation_min_rows", 0)
        isolation_min_next_rows = getattr(
            self, "_dispatch_batch_isolation_min_next_rows", 0
        )
        isolated_dispatch_batch_id = None
        first_dispatch_rows = 0
        next_dispatch_rows = 0
        if isolation_min_rows > 0 and profile_dispatch_batches is not None:
            first_dispatch_batch_id = profile_dispatch_batches.get(
                self._pending[0].transfer_id, ""
            )
            if first_dispatch_batch_id:
                pending_iter = iter(self._pending)
                for task in pending_iter:
                    if (
                        profile_dispatch_batches.get(task.transfer_id)
                        != first_dispatch_batch_id
                    ):
                        next_dispatch_batch_id = profile_dispatch_batches.get(
                            task.transfer_id, ""
                        )
                        next_dispatch_rows = 1
                        break
                    first_dispatch_rows += 1
                else:
                    next_dispatch_batch_id = ""
                if next_dispatch_batch_id:
                    for task in pending_iter:
                        if (
                            profile_dispatch_batches.get(task.transfer_id)
                            != next_dispatch_batch_id
                        ):
                            break
                        next_dispatch_rows += 1
                if (
                    first_dispatch_rows >= isolation_min_rows
                    and next_dispatch_rows >= max(1, isolation_min_next_rows)
                ):
                    # A steady-state Target wave is striped across both Drafts.
                    # If only one Draft coalesces its ~64-row shard with the
                    # next cohort, that side promotes to a 128-row CUDA graph
                    # while its peer stays at 64.  The Target then waits for
                    # the slower shard even though the useful cohort itself
                    # fit the cheaper graph tier.  Preserve the dispatch batch
                    # boundary only when the following cohort is itself ready
                    # at a useful width.  Splitting off a 1-7 row partial next
                    # cohort makes prepare overlap launch a tiny graph early
                    # and destroys continuous batching efficiency.
                    isolated_dispatch_batch_id = first_dispatch_batch_id
        queued: list[PendingDraftTask] = []
        while self._pending and len(queued) < self.max_batch_size:
            if (
                isolated_dispatch_batch_id is not None
                and profile_dispatch_batches.get(self._pending[0].transfer_id)
                != isolated_dispatch_batch_id
            ):
                break
            task = self._pending.popleft()
            self._pending_ids.discard(task.transfer_id)
            received_ns = self._received_ns_by_transfer.pop(
                task.transfer_id, take_start_ns
            )
            dispatch_batch_id = (
                profile_dispatch_batches.pop(task.transfer_id, "")
                if profile_dispatch_batches is not None
                else ""
            )
            if task.transfer_id in self._cancelled_ids:
                self._cancelled_ids.discard(task.transfer_id)
                continue
            queued.append(
                PendingDraftTask(
                    task=task,
                    pivot_bytes=None,
                    received_ns=received_ns,
                    dispatch_batch_id=dispatch_batch_id,
                )
            )

        # The result publisher needs rows grouped by Target. Sort before direct
        # mmap reads so pinned staging is already laid out in publish order.
        queued = target_major(queued)
        staging_buffers = getattr(self, "_pivot_host_staging_buffers", ())
        view_buffers = getattr(self, "_pivot_host_view_buffers", ())
        if staging_buffers:
            if not 0 <= pivot_buffer < len(staging_buffers):
                raise ValueError(f"invalid DT pivot staging buffer {pivot_buffer}")
            pinned_staging = staging_buffers[pivot_buffer]
            pivot_views = view_buffers[pivot_buffer]
        else:
            pinned_staging = getattr(self, "_pivot_host_staging", None)
            pivot_views = getattr(self, "_pivot_host_views", ())
        direct_mmap_h2d = getattr(self, "_direct_mmap_h2d", False)
        tasks: list[PendingDraftTask] = []
        for item in queued:
            task = item.task
            try:
                read_start_ns = time.monotonic_ns()
                pivot_row = -1
                pivot_bytes = None
                if direct_mmap_h2d:
                    pivot_row = len(tasks)
                    self._validate_pivot_source(task)
                elif pinned_staging is None:
                    pivot_bytes = self._read_pivot(task)
                else:
                    pivot_row = len(tasks)
                    self._read_pivot_into(task, pivot_views[pivot_row])
                pivot_read_ns = time.monotonic_ns() - read_start_ns
            except Exception as exc:
                logger.exception("failed to read DT pivot for %s", task.transfer_id)
                self._publish_source_done(task)
                self._publish_failures([(task, exc)])
                continue
            if not direct_mmap_h2d:
                self._publish_source_done(task)
            tasks.append(
                PendingDraftTask(
                    task=task,
                    pivot_bytes=pivot_bytes,
                    received_ns=item.received_ns,
                    pivot_read_ns=pivot_read_ns,
                    pivot_row=pivot_row,
                    pivot_buffer=pivot_buffer,
                    dispatch_batch_id=item.dispatch_batch_id,
                )
            )
        if getattr(self, "batch_source_read_done", False) and not direct_mmap_h2d:
            self._flush_source_done()
        if tasks:
            queue_waits = [take_start_ns - item.received_ns for item in tasks]
            cohort_profile_fields = {}
            if profile_dispatch_batches is not None:
                dispatch_batch_counts = Counter(
                    item.dispatch_batch_id for item in tasks if item.dispatch_batch_id
                )
                target_counts = Counter(item.task.target_instance_id for item in tasks)
                round_ids = [getattr(item.task, "round_id", -1) for item in tasks]
                cohort_profile_fields = {
                    "dispatch_batch_counts": sorted(dispatch_batch_counts.items()),
                    "dispatch_boundary_isolated": (
                        isolated_dispatch_batch_id is not None
                    ),
                    "oldest_dispatch_rows": first_dispatch_rows,
                    "pending_next_dispatch_rows": next_dispatch_rows,
                    "target_counts": sorted(target_counts.items()),
                    "round_min": min(round_ids),
                    "round_max": max(round_ids),
                }
            profile_emit(
                "draft_take_batch",
                instance_id=self.instance_id,
                batch_size=len(tasks),
                pending_after=len(self._pending),
                free_result_slots=len(self._free_result_slots),
                queue_wait_mean_ns=sum(queue_waits) // len(queue_waits),
                queue_wait_max_ns=max(queue_waits),
                pivot_read_total_ns=sum(item.pivot_read_ns for item in tasks),
                pivot_read_max_ns=max(item.pivot_read_ns for item in tasks),
                take_ns=time.monotonic_ns() - take_start_ns,
                direct_mmap_h2d=direct_mmap_h2d,
                **cohort_profile_fields,
            )
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

    def _copy_pivots_direct_to_device(
        self,
        pending: Sequence[PendingDraftTask],
        pivots: torch.Tensor,
    ) -> tuple[int, int]:
        row_bytes = self._pivot_width * 2
        if self._direct_cuda_rt is None:
            cuda_rt = CudaRTLibrary()
            memcpy_2d_async = cuda_rt.lib.cudaMemcpy2DAsync
            memcpy_2d_async.restype = ctypes.c_int
            memcpy_2d_async.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            self._direct_cuda_rt = (cuda_rt, memcpy_2d_async)
        cuda_rt, memcpy_2d_async = self._direct_cuda_rt
        stream = torch.cuda.current_stream(self.device)
        runs = direct_mmap_h2d_runs(pending)
        for run in runs:
            store = self._source_stores[run.uri]
            source_address = self._direct_source_mapping_addresses[run.uri]
            result = memcpy_2d_async(
                pivots[run.destination_row].data_ptr(),
                row_bytes,
                source_address + store.payload_offset(run.first_slot),
                run.slot_step * store.slot_stride,
                row_bytes,
                run.row_count,
                1,
                stream.cuda_stream,
            )
            cuda_rt.CUDART_CHECK(result)
        return len(runs), max((run.row_count for run in runs), default=0)

    def _release_direct_pivot_sources(
        self,
        tasks: Sequence[object],
        *,
        validate: bool,
    ) -> None:
        validation_error = None
        for task in tasks:
            if validate:
                try:
                    self._validate_pivot_source(task)
                except Exception as exc:  # noqa: BLE001 - release every source slot
                    if validation_error is None:
                        validation_error = exc
            self._publish_source_done(task)
        if getattr(self, "batch_source_read_done", False):
            self._flush_source_done()
        if validation_error is not None:
            raise validation_error

    @torch.inference_mode()
    def _execute_batch(
        self, pending: list[PendingDraftTask], overlap_callback=None
    ) -> None:
        execute_start_ns = time.monotonic_ns()
        tasks = [item.task for item in pending]
        batch_size = len(tasks)
        cpu_prep_start_ns = time.monotonic_ns()
        pivot_nbytes = batch_size * self._pivot_width * 2
        pivot_buffers = {item.pivot_buffer for item in pending}
        if len(pivot_buffers) != 1:
            raise RuntimeError("one Draft batch cannot span pivot staging buffers")
        pivot_buffer = pivot_buffers.pop()
        staging_buffers = getattr(self, "_pivot_host_staging_buffers", ())
        pinned_staging = (
            staging_buffers[pivot_buffer]
            if staging_buffers
            else getattr(self, "_pivot_host_staging", None)
        )
        direct_mmap_h2d = getattr(self, "_direct_mmap_h2d", False)
        direct_h2d_runs = 0
        direct_h2d_max_run_rows = 0
        direct_source_release_ns = 0
        source_copy_done_event = None
        if direct_mmap_h2d:
            if any(item.pivot_row != index for index, item in enumerate(pending)):
                raise RuntimeError("direct mmap H2D pivot rows are not contiguous")
            pivots = torch.empty(
                (batch_size, self._pivot_width),
                dtype=torch.bfloat16,
                device=self.device,
            )
        elif pinned_staging is None:
            pivot_blob = b"".join(item.pivot_bytes for item in pending)
            pivots = torch.frombuffer(
                bytearray(pivot_blob), dtype=torch.bfloat16
            ).reshape(batch_size, self._pivot_width)
        else:
            if any(item.pivot_row != index for index, item in enumerate(pending)):
                raise RuntimeError("pinned DT pivot rows are not contiguous")
            pivots = pinned_staging[:batch_size]
        cpu_prep_end_ns = time.monotonic_ns()
        async_cuda_profile = profile_enabled() and not cuda_sync_enabled()
        h2d_start_event = (
            torch.cuda.Event(enable_timing=True) if async_cuda_profile else None
        )
        h2d_end_event = (
            torch.cuda.Event(enable_timing=True) if async_cuda_profile else None
        )
        gpu_start_event = (
            torch.cuda.Event(enable_timing=True) if async_cuda_profile else None
        )
        gpu_end_event = (
            torch.cuda.Event(enable_timing=True) if async_cuda_profile else None
        )
        if cuda_sync_enabled():
            torch.cuda.synchronize(self.device)
        h2d_start_ns = time.monotonic_ns()
        if h2d_start_event is not None:
            h2d_start_event.record()
        if direct_mmap_h2d:
            source_copy_done_event = torch.cuda.Event()
            try:
                direct_h2d_runs, direct_h2d_max_run_rows = (
                    self._copy_pivots_direct_to_device(pending, pivots)
                )
                source_copy_done_event.record()
            except Exception:
                self._release_direct_pivot_sources(tasks, validate=False)
                raise
        else:
            pivots = pivots.to(
                self.device,
                non_blocking=pinned_staging is not None,
            )
        if h2d_end_event is not None:
            h2d_end_event.record()
        if cuda_sync_enabled():
            torch.cuda.synchronize(self.device)
        h2d_end_ns = time.monotonic_ns()
        gpu_start_ns = time.monotonic_ns()
        if gpu_start_event is not None:
            gpu_start_event.record()
        bonus_tokens = torch.tensor(
            [task.next_token_id for task in tasks],
            dtype=torch.int64,
            device=self.device,
        )
        if self._draft_cuda_graph is not None:
            (
                candidates,
                compute_batch_size,
                logit_margins,
                verify_lens,
                padding_priorities,
            ) = self._draft_cuda_graph.replay(
                pivots=pivots,
                bonus_tokens=bonus_tokens,
            )
        else:
            compute_batch_size = batch_size
            logit_margins = None
            verify_lens = None
            padding_priorities = None
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
            if self._collect_confidence:
                confidence_sampler = {
                    "margin": self._draft_model.sample_draft_tokens_with_margin,
                    "top1": self._draft_model.sample_draft_tokens_with_top1,
                    "top1_trace": self._draft_model.sample_draft_tokens_with_top1_trace,
                }[self._confidence_mode]
                proposals, logit_margins = confidence_sampler(
                    block_hidden=block_hidden,
                    first_prev_token_ids=bonus_tokens,
                )
            else:
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
        if gpu_end_event is not None:
            gpu_end_event.record()
        if source_copy_done_event is not None:
            source_copy_done_event.synchronize()
            direct_source_release_start_ns = time.monotonic_ns()
            self._release_direct_pivot_sources(tasks, validate=True)
            direct_source_release_ns = (
                time.monotonic_ns() - direct_source_release_start_ns
            )
        if cuda_sync_enabled():
            torch.cuda.synchronize(self.device)
        gpu_end_ns = time.monotonic_ns()
        result_rows = candidates
        if self._adaptive_verify_margin is not None:
            if logit_margins is None:
                raise RuntimeError("adaptive verify requires Draft confidence margins")
            if verify_lens is None:
                verify_lens = select_adaptive_verify_lens(
                    logit_margins,
                    threshold=(
                        self._adaptive_verify_thresholds
                        if self._adaptive_verify_thresholds is not None
                        else self._adaptive_verify_margin
                    ),
                    block_size=self.profile.block_size,
                )
            result_rows = torch.empty(
                (batch_size, self._result_row_width),
                dtype=torch.int32,
                device=self.device,
            )
            result_rows[:, : self._verify_block_size].copy_(candidates)
            result_rows[:, self._verify_block_size].copy_(verify_lens)
            if self._adaptive_padding_priority:
                assert self._padding_priority_model is not None
                if padding_priorities is None:
                    padding_priorities = score_adaptive_padding_priorities(
                        logit_margins,
                        verify_lens,
                        block_size=self.profile.block_size,
                        model=self._padding_priority_model,
                    )
                result_rows[:, self._verify_block_size + 1].copy_(padding_priorities)
        prepare_done_event = None
        if overlap_callback is not None and getattr(
            self, "_prepare_until_gpu_done", False
        ):
            prepare_done_event = torch.cuda.Event()
            prepare_done_event.record()
        overlap_prep_start_ns = time.monotonic_ns()
        overlap_prepared_batch_size = 0
        overlap_prep_ns = 0
        overlap_finalize_wait_ns = 0
        overlap_finalize = None
        if overlap_callback is not None:
            overlap_result = overlap_callback(gpu_done_event=prepare_done_event)
            if callable(overlap_result):
                overlap_finalize = overlap_result
            else:
                overlap_prepared_batch_size = int(overlap_result or 0)
                overlap_prep_ns = time.monotonic_ns() - overlap_prep_start_ns
        d2h_start_ns = time.monotonic_ns()
        candidate_bytes = result_rows.cpu().contiguous().numpy().tobytes()
        if self._profile_confidence:
            assert logit_margins is not None
            profile_emit(
                "draft_confidence_batch",
                instance_id=self.instance_id,
                transfer_ids=[task.transfer_id for task in tasks],
                target_instance_ids=[task.target_instance_id for task in tasks],
                request_ids=[task.request_id for task in tasks],
                round_ids=[task.round_id for task in tasks],
                confidence_mode=self._confidence_mode,
                logit_margins=logit_margins.float().cpu().tolist(),
                verify_lens=(
                    verify_lens.cpu().tolist() if verify_lens is not None else None
                ),
            )
        d2h_end_ns = time.monotonic_ns()
        if overlap_finalize is not None:
            finalize_start_ns = time.monotonic_ns()
            overlap_prepared_batch_size, overlap_prep_ns = overlap_finalize()
            overlap_finalize_wait_ns = time.monotonic_ns() - finalize_start_ns
        if gpu_start_event is not None:
            gpu_compute_ns = round(
                gpu_start_event.elapsed_time(gpu_end_event) * 1_000_000
            )
            pivot_h2d_ns = round(
                h2d_start_event.elapsed_time(h2d_end_event) * 1_000_000
            )
        else:
            gpu_compute_ns = gpu_end_ns - gpu_start_ns
            pivot_h2d_ns = h2d_end_ns - h2d_start_ns
        self._publish_successes(tasks, candidate_bytes)
        self._publish_capacity()
        task_ages = [execute_start_ns - task.created_ns for task in tasks]
        profile_emit(
            "draft_execute_batch",
            instance_id=self.instance_id,
            batch_size=batch_size,
            compute_batch_size=compute_batch_size,
            cuda_graph=self._draft_cuda_graph is not None,
            pivot_bytes=pivot_nbytes,
            candidate_bytes=len(candidate_bytes),
            task_age_mean_ns=sum(task_ages) // len(task_ages),
            task_age_max_ns=max(task_ages),
            cpu_prep_ns=cpu_prep_end_ns - cpu_prep_start_ns,
            pivot_h2d_ns=pivot_h2d_ns,
            direct_mmap_h2d=direct_mmap_h2d,
            direct_h2d_runs=direct_h2d_runs,
            direct_h2d_max_run_rows=direct_h2d_max_run_rows,
            direct_source_release_ns=direct_source_release_ns,
            gpu_compute_ns=gpu_compute_ns,
            gpu_timing_source=(
                "cuda_event"
                if gpu_start_event is not None
                else "synchronized_wall"
                if cuda_sync_enabled()
                else "enqueue_wall"
            ),
            candidate_d2h_ns=d2h_end_ns - d2h_start_ns,
            prepare_overlap=overlap_callback is not None,
            prepare_overlap_mode=(
                "async" if overlap_finalize is not None else "inline"
            ),
            overlap_prep_ns=overlap_prep_ns,
            overlap_finalize_wait_ns=overlap_finalize_wait_ns,
            overlap_prepared_batch_size=overlap_prepared_batch_size,
            prepare_until_gpu_done=(prepare_done_event is not None),
            execute_and_publish_ns=time.monotonic_ns() - execute_start_ns,
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
        write_start_ns = time.monotonic_ns()
        generation = self._result_store.write(slot_id, candidate_bytes)
        write_end_ns = time.monotonic_ns()
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
        send_start_ns = time.monotonic_ns()
        self._send_result(message)
        profile_emit(
            "draft_result_publish",
            instance_id=self.instance_id,
            batch_id=batch_id,
            batch_size=len(tasks),
            nbytes=len(candidate_bytes),
            mmap_write_ns=write_end_ns - write_start_ns,
            control_submit_ns=time.monotonic_ns() - send_start_ns,
            outstanding_result_batches=len(self._published_by_batch),
            free_result_slots=len(self._free_result_slots),
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
            direct_mmap_h2d = getattr(self, "_direct_mmap_h2d", False)
            store = MmapSlotStore.open(uri, writable=direct_mmap_h2d)
            if direct_mmap_h2d:
                mapping_view = store.writable_mapping_view()
                try:
                    address = ctypes.addressof(ctypes.c_char.from_buffer(mapping_view))
                    cudart = torch.cuda.cudart()
                    result = cudart.cudaHostRegister(address, len(mapping_view), 0)
                    if result != cudart.cudaError.success:
                        raise RuntimeError(
                            "cudaHostRegister failed for the DT Draft source mmap "
                            f"{uri}: {result}"
                        )
                except Exception:
                    mapping_view.release()
                    store.close()
                    raise
                self._direct_source_mapping_views[uri] = mapping_view
                self._direct_source_mapping_addresses[uri] = address
            self._source_stores[uri] = store
        return store

    def _validate_pivot_source(self, task) -> None:
        source = task.source
        store = self._get_source_store(source.uri)
        store.validate_read(
            source.source_slot_id,
            source.slot_generation,
            expected_length=source.nbytes,
        )

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
                profile_dispatch_batches = getattr(
                    self, "_profile_dispatch_batch_by_transfer", None
                )
                if profile_dispatch_batches is not None:
                    profile_dispatch_batches.pop(task.transfer_id, None)
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
        if getattr(self, "batch_source_read_done", False):
            self._source_done_pending.append(message)
        else:
            from dt_protocol import SOURCE_READ_DONE, source_read_done_id

            self._delivery_outbox.add(
                SOURCE_READ_DONE, source_read_done_id(message), message
            )
            self._flush_reliable_outbox()

    def _flush_source_done(self) -> None:
        if getattr(self, "batch_source_read_done", False) and self._source_done_pending:
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
            profile_emit(
                "draft_source_done_batch",
                instance_id=self.instance_id,
                batch_id=message.batch_id,
                entries=len(entries),
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
