from __future__ import annotations

import logging
import os
import time

import torch
from dt_profile import event_enabled as profile_event_enabled

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_worker_v2 import split_dflash_committed_rows
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.eagle_utils import (
    _eagle_prefill_tail_tokens,
    default_tree_mask_mode,
)
from sglang.srt.speculative.eagle_worker_common import (
    build_eagle_verify_input,
    run_eagle_verify,
)
from sglang.srt.speculative.spec_utils import get_plan_stream

logger = logging.getLogger(__name__)

_TARGET_VERIFY_GPU_PROFILE_EVENT = "target_eagle3_verify_gpu_breakdown"


def _target_verify_gpu_profile_selected() -> bool:
    """Require the EAGLE3 CUDA corridor to be named explicitly.

    ``dt_profile.event_enabled`` deliberately treats an empty event filter as
    selecting every event.  CUDA events are more intrusive than host tracing,
    so this diagnostic has the stricter contract that its event name must be
    present in ``DT_PROFILE_EVENTS``.
    """

    selected = {
        event.strip()
        for event in os.environ.get("DT_PROFILE_EVENTS", "").split(",")
        if event.strip()
    }
    return (
        _TARGET_VERIFY_GPU_PROFILE_EVENT in selected
        and profile_event_enabled(_TARGET_VERIFY_GPU_PROFILE_EVENT)
    )


def _new_target_verify_gpu_profile_events(
    device: str, *, enabled: bool
) -> tuple[object, ...] | None:
    if not enabled:
        return None
    event_type = torch.get_device_module(device).Event
    return tuple(event_type(enable_timing=True) for _ in range(5))


def _freeze_eagle3_verify_features(
    result, *, packed_hidden, candidates, export_rank, reserve=None
):
    """Preserve one bounded verify wave before the next graph replay."""
    # Freeze every replay-owned output on the producing forward stream.
    # D2H is later submitted on copy_stream; retaining graph views alone
    # would not protect them against the other lane's replay.
    result.next_token_ids = result.next_token_ids.clone()
    result.accept_lens = result.accept_lens.clone()
    result.new_seq_lens = result.new_seq_lens.clone()
    bonus_tokens = result.next_draft_input.bonus_tokens.clone()
    result.dt_deferred_dflash_feature_stage = True
    result.dt_deferred_eagle3_feature_stage = True
    result.dt_dflash_packed_hidden = None
    result.dt_dflash_candidate_tokens = None
    result.dt_eagle3_committed_tokens = None
    if export_rank:
        try:
            # DT Target may lease a feature-source slot and write in place.
            reserved = reserve(packed_hidden) if reserve is not None else None
            result.dt_dflash_packed_hidden = (
                packed_hidden.clone() if reserved is None else reserved
            )
            # Predicted tokens condition features; the candidate prefix
            # is the distinct authoritative committed input sequence.
            result.dt_dflash_candidate_tokens = result.next_token_ids.view(
                candidates.shape[0], candidates.shape[1]
            )
            result.dt_eagle3_committed_tokens = candidates.clone()
        except torch.cuda.OutOfMemoryError as exc:
            # Reach the same owner-side TP error broadcast as peers.
            result.dt_deferred_feature_error = f"{type(exc).__name__}: {exc}"
    return bonus_tokens


class EAGLE3DisaggregatedTargetWorkerV2(BaseSpecWorker):
    """Target-only EAGLE3 verifier fed by remotely produced top-k=1 chains."""

    def __init__(
        self,
        server_args,
        gpu_id: int,
        ps: ParallelState,
        nccl_port: int,
        target_worker,
    ) -> None:
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.ps = ps
        self.nccl_port = nccl_port
        self._target_worker = target_worker
        self._draft_worker = None
        self.model_runner = target_worker.model_runner
        self.device = target_worker.device
        self.page_size = server_args.page_size
        self.topk = int(server_args.speculative_eagle_topk)
        self.speculative_num_steps = int(server_args.speculative_num_steps)
        self.speculative_num_draft_tokens = int(
            server_args.speculative_num_draft_tokens
        )
        self.block_size = self.speculative_num_draft_tokens
        self.tree_mask_mode = default_tree_mask_mode()
        self.plan_stream, self.plan_stream_ctx = get_plan_stream(self.device)
        # Cache once so every TP rank follows identical record/no-record control
        # flow throughout the worker lifetime.  Only TP0 later emits timings.
        self._profile_target_verify_gpu = _target_verify_gpu_profile_selected()

        draft_config = ModelConfig.from_server_args(
            server_args,
            model_path=server_args.speculative_draft_model_path,
            model_revision=server_args.speculative_draft_model_revision,
            is_draft_model=True,
        )
        target_config = target_worker.model_runner.model_config
        hf_config = draft_config.hf_config
        eagle_config = getattr(hf_config, "eagle_config", None) or {}
        get_eagle_config = (
            eagle_config.get
            if isinstance(eagle_config, dict)
            else lambda key, default=None: getattr(eagle_config, key, default)
        )
        target_hidden_size = getattr(hf_config, "target_hidden_size", None)
        if target_hidden_size is None:
            target_hidden_size = target_config.hidden_size
        target_hidden_size = int(target_hidden_size)
        num_aux = getattr(hf_config, "num_aux_hidden_states", None)
        if num_aux is None:
            layer_ids = get_eagle_config("eagle_aux_hidden_state_layer_ids", None)
            if layer_ids is None:
                layer_ids = getattr(hf_config, "eagle_aux_hidden_state_layer_ids", None)
            num_aux = len(layer_ids) if layer_ids else 3
        self._num_pivot_layers = int(num_aux)
        self._pivot_width = target_hidden_size * self._num_pivot_layers

        if target_hidden_size != int(target_config.hidden_size):
            raise ValueError(
                "EAGLE3 target hidden size does not match the draft checkpoint: "
                f"target={target_config.hidden_size}, draft_target={target_hidden_size}."
            )
        if int(target_config.vocab_size) != int(draft_config.vocab_size):
            raise ValueError(
                "EAGLE3 target/draft vocabulary size mismatch: "
                f"target={target_config.vocab_size}, draft={draft_config.vocab_size}."
            )

        if self.ps.tp_rank == 0:
            logger.info(
                "Initialized disaggregated EAGLE3 Target worker without Draft "
                "weights. steps=%d block_size=%d feature_layers=%d feature_width=%d",
                self.speculative_num_steps,
                self.block_size,
                self._num_pivot_layers,
                self._pivot_width,
            )

    @property
    def spec_v2_attn_backends(self) -> tuple:
        return (self.target_worker.model_runner.attn_backend,)

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ) -> None:
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

    def init_attention_backends(self) -> None:
        pass

    def init_cuda_graphs(self) -> None:
        pass

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        grammar_barrier=None,
    ) -> GenerationBatchResult:
        host_span_active = bool(
            getattr(batch, "dt_eagle3_host_span_active", False)
        )
        worker_entry_ns = time.monotonic_ns() if host_span_active else None
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            result = self._forward_prefill(batch, on_publish=on_publish)
            if host_span_active:
                result.dt_eagle3_worker_entry_ns = worker_entry_ns
                result.dt_eagle3_worker_branch = "target_prefill"
                result.dt_eagle3_model_forward_mode = batch.forward_mode.name
                result.dt_eagle3_worker_return_ns = time.monotonic_ns()
            return result

        if batch.spec_info is None:
            batch.spec_info = DFlashDraftInputV2.create_idle_input(device=self.device)
        remote_input = batch.spec_info
        if not isinstance(remote_input, DFlashDraftInputV2):
            raise TypeError(
                "disaggregated EAGLE3 Target expected DFlashDraftInputV2 handoff"
            )
        if batch.forward_mode.is_idle():
            result = self._forward_idle(on_publish=on_publish)
            if host_span_active:
                result.dt_eagle3_worker_entry_ns = worker_entry_ns
                result.dt_eagle3_worker_branch = "target_idle"
                result.dt_eagle3_model_forward_mode = "IDLE"
                result.dt_eagle3_worker_return_ns = time.monotonic_ns()
            return result
        if remote_input.draft_tokens is None:
            raise RuntimeError(
                "disaggregated EAGLE3 Target cannot verify before remote Draft "
                "candidates are attached"
            )

        batch_size = len(batch.seq_lens)
        candidates = remote_input.draft_tokens.to(device=self.device, dtype=torch.int64)
        if tuple(candidates.shape) != (batch_size, self.block_size):
            raise ValueError(
                "remote EAGLE3 candidate shape mismatch: "
                f"expected {(batch_size, self.block_size)}, "
                f"got {tuple(candidates.shape)}"
            )

        draft_input = EagleDraftInput(bonus_tokens=candidates[:, 0])
        parent_width = self.speculative_num_steps
        parent_list = torch.arange(
            -1, parent_width - 1, dtype=torch.long, device=self.device
        ).repeat(batch_size, 1)
        top_scores_index = torch.arange(
            self.speculative_num_steps, dtype=torch.long, device=self.device
        ).repeat(batch_size, 1)
        verify_input = build_eagle_verify_input(
            batch,
            draft_input,
            parent_list,
            top_scores_index,
            candidates[:, 1:],
            None,
            target_worker=self.target_worker,
            topk=self.topk,
            num_steps=self.speculative_num_steps,
            num_draft_tokens=self.speculative_num_draft_tokens,
            tree_mask_mode=self.tree_mask_mode,
            device=self.device,
        )
        batch.spec_info = verify_input
        post_input_build_ns = time.monotonic_ns() if host_span_active else None
        prefix_lens = (
            batch.seq_lens_cpu
            if batch.seq_lens_cpu is not None
            else batch.seq_lens.detach().to(device="cpu", dtype=torch.int64)
        )
        gpu_profile_events = _new_target_verify_gpu_profile_events(
            self.device,
            enabled=self._profile_target_verify_gpu,
        )
        pre_run_eagle_verify_ns = (
            time.monotonic_ns() if host_span_active else None
        )
        result = run_eagle_verify(
            batch,
            target_worker=self.target_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            plan_stream=self.plan_stream,
            plan_stream_ctx=self.plan_stream_ctx,
            topk=self.topk,
            num_steps=self.speculative_num_steps,
            num_draft_tokens=self.speculative_num_draft_tokens,
            device=self.device,
            metadata_ready_pre_pad=False,
            finalize_tree_path=True,
            grammar_barrier=grammar_barrier,
            gpu_profile_events=gpu_profile_events,
        )
        post_run_eagle_verify_ns = (
            time.monotonic_ns() if host_span_active else None
        )
        if on_publish is not None:
            on_publish(result.new_seq_lens)

        packed_hidden = result.logits_output.hidden_states
        if packed_hidden is None:
            raise RuntimeError(
                "disaggregated EAGLE3 verify requires target auxiliary hidden states"
            )
        deferred = (
            self.server_args.draft_target_overlap_schedule
            and not self.server_args.disable_overlap_schedule
            and not self.server_args.draft_target_disable_eagle3_deferred_handoff
        )
        feature_offsets = [int(value) for value in prefix_lens.tolist()]
        if deferred:
            bonus_tokens = _freeze_eagle3_verify_features(
                result, packed_hidden=packed_hidden, candidates=candidates,
                export_rank=self.ps.tp_rank == 0,
                reserve=getattr(self, "dt_deferred_hidden_destination", None),
            )
            feature_chunks = token_chunks = committed_token_chunks = None
        else:
            bonus_tokens = result.next_draft_input.bonus_tokens
            committed_tokens = result.next_token_ids.view(batch_size, self.block_size)
            feature_chunks, token_chunks = split_dflash_committed_rows(
                packed_hidden=packed_hidden,
                candidates=committed_tokens,
                commit_lens=result.accept_lens,
                block_size=self.block_size,
            )
            committed_token_chunks = [
                candidates[row, : int(hidden.shape[0])]
                for row, hidden in enumerate(feature_chunks)
            ]
        result.logits_output.hidden_states = None
        result.next_draft_input = self._make_handoff(
            bonus_tokens=bonus_tokens,
            new_seq_lens=result.new_seq_lens,
            feature_chunks=feature_chunks,
            token_chunks=token_chunks,
            committed_token_chunks=committed_token_chunks,
            feature_offsets=feature_offsets,
        )
        post_feature_handoff_ns = (
            time.monotonic_ns() if host_span_active else None
        )
        if gpu_profile_events is not None:
            gpu_profile_events[4].record()
            result.dt_eagle3_verify_gpu_events = gpu_profile_events
        result.dt_hidden_produced_ns = time.monotonic_ns()
        result.dt_cuda_graph_measurement_event_id = (
            self.target_worker.model_runner._last_cuda_graph_measurement_event_id
        )
        result.dt_full_wave_id = getattr(batch, "dt_full_wave_id", None)
        result.dt_full_wave_ready_ns = getattr(batch, "dt_full_wave_ready_ns", None)
        result.dt_pipeline_lane_id = getattr(batch, "dt_pipeline_lane_id", None)
        if host_span_active:
            result.dt_eagle3_worker_entry_ns = worker_entry_ns
            result.dt_eagle3_post_input_build_ns = post_input_build_ns
            result.dt_eagle3_pre_run_eagle_verify_ns = pre_run_eagle_verify_ns
            result.dt_eagle3_post_run_eagle_verify_ns = post_run_eagle_verify_ns
            result.dt_eagle3_post_feature_handoff_ns = post_feature_handoff_ns
            result.dt_eagle3_worker_branch = "remote_eagle3_verify"
            result.dt_eagle3_model_forward_mode = "TARGET_VERIFY"
            result.dt_eagle3_worker_return_ns = time.monotonic_ns()
        return result

    def _forward_prefill(
        self, batch: ScheduleBatch, *, on_publish=None
    ) -> GenerationBatchResult:
        input_ids = batch.input_ids
        result = self.target_worker.forward_batch_generation(
            batch, capture_hidden_mode=CaptureHiddenMode.FULL
        )
        packed_hidden = result.logits_output.hidden_states
        if packed_hidden is None:
            raise RuntimeError(
                "disaggregated EAGLE3 prefill requires target auxiliary hidden states"
            )
        if batch.extend_lens is None or batch.prefix_lens is None or input_ids is None:
            raise RuntimeError(
                "disaggregated EAGLE3 prefill requires input, extend, and prefix lengths"
            )
        total_tokens = int(sum(batch.extend_lens))
        if tuple(packed_hidden.shape) != (total_tokens, self._pivot_width):
            raise RuntimeError(
                "EAGLE3 prefill feature shape mismatch: "
                f"expected {(total_tokens, self._pivot_width)}, "
                f"got {tuple(packed_hidden.shape)}"
            )
        if int(input_ids.numel()) != total_tokens:
            raise RuntimeError("EAGLE3 prefill input ids do not match feature rows")

        tail_tokens = _eagle_prefill_tail_tokens(batch, result.next_token_ids)
        feature_chunks = list(torch.split(packed_hidden, batch.extend_lens, dim=0))
        input_chunks = torch.split(input_ids, batch.extend_lens, dim=0)
        # EAGLE conditions each Target feature row on the following token.  Send
        # that feature-aligned stream separately from the authoritative committed
        # ids reconstructed by the controller.  The remote worker supplies a
        # preimage to stock `_draft_extend_for_prefill`, whose one left shift then
        # reproduces these ids without changing the stock Draft-KV path.
        token_chunks = [
            torch.cat((chunk[1:], tail.reshape(1)))
            for chunk, tail in zip(input_chunks, tail_tokens, strict=True)
        ]

        result.logits_output.hidden_states = None
        result.new_seq_lens = batch.seq_lens
        if on_publish is not None:
            on_publish(result.new_seq_lens)
        result.next_draft_input = self._make_handoff(
            # Intermediate prompt chunks append the next prompt token; final
            # chunks append the sampled Target token.
            bonus_tokens=tail_tokens,
            new_seq_lens=batch.seq_lens,
            feature_chunks=feature_chunks,
            token_chunks=token_chunks,
            committed_token_chunks=list(input_chunks),
            feature_offsets=[int(value) for value in batch.prefix_lens],
        )
        result.dt_hidden_produced_ns = time.monotonic_ns()
        return result

    def _forward_idle(self, *, on_publish=None) -> GenerationBatchResult:
        empty_ids = torch.empty((0,), dtype=torch.int64, device=self.device)
        empty_lens = torch.empty((0,), dtype=torch.int32, device=self.device)
        next_input = self._make_handoff(
            bonus_tokens=empty_ids,
            new_seq_lens=empty_ids,
            feature_chunks=[],
            token_chunks=[],
            committed_token_chunks=[],
            feature_offsets=[],
        )
        if on_publish is not None:
            on_publish(empty_ids)
        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=empty_ids,
            accept_lens=empty_lens,
            next_draft_input=next_input,
            can_run_cuda_graph=False,
            speculative_num_draft_tokens=self.block_size,
            new_seq_lens=empty_ids,
        )

    def _make_handoff(
        self,
        *,
        bonus_tokens: torch.Tensor,
        new_seq_lens: torch.Tensor,
        feature_chunks,
        token_chunks,
        committed_token_chunks,
        feature_offsets,
    ) -> DFlashDraftInputV2:
        batch_size = int(bonus_tokens.numel())
        return DFlashDraftInputV2(
            topk_p=torch.empty(
                (batch_size, 0), dtype=torch.float32, device=self.device
            ),
            topk_index=torch.empty(
                (batch_size, 0), dtype=torch.int64, device=self.device
            ),
            bonus_tokens=bonus_tokens,
            new_seq_lens=new_seq_lens,
            hidden_states=torch.empty(
                (batch_size, 0), dtype=torch.bfloat16, device=self.device
            ),
            dt_feature_chunks=feature_chunks,
            dt_feature_token_chunks=token_chunks,
            dt_committed_token_chunks=committed_token_chunks,
            dt_feature_offsets=feature_offsets,
        )
