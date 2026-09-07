from __future__ import annotations

import dataclasses
import logging
import os
import time

import torch
from dt_profile import emit as profile_emit
from sglang.kernels.ops.speculative.dflash import (
    _prepare_dflash_draft_block_unchecked,
)
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    BuildRaggedVerifyWindow,
    ScatterCompactToStrided,
)
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import (
    compute_dflash_correct_drafts_and_bonus,
)
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.draft_worker_common import build_block_pos_offsets
from sglang.srt.speculative.dspark_components.dspark_verify import (
    DsparkVerifyEpilogue,
    VocabParallelGreedySelector,
    verify_logits_adjustments_are_noop,
)
from sglang.srt.speculative.flashmtp_shared import (
    adaptive_verify_enabled,
    graph_tier_padding_enabled,
    plan_ragged_execution_tier,
    ragged_epilogue_capacity,
    ragged_verify_enabled,
    reclaim_ragged_graph_padding,
    validate_component_dependencies,
    vocab_parallel_greedy_enabled,
)
from sglang.srt.speculative.flashmtp_utils import (
    gather_flashmtp_next_pivot,
    parse_flashmtp_draft_config,
)
from sglang.srt.speculative.flashmtp_worker_v2 import FlashMTPWorkerV2
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyLayout,
    ragged_verify_compact_enabled,
)
from sglang.srt.utils import is_cuda, is_hip
from sglang.srt.utils.common import is_pin_memory_available

logger = logging.getLogger(__name__)


class FlashMTPDisaggregatedTargetWorkerV2(FlashMTPWorkerV2):
    """Target-only FlashMTP worker fed by remotely produced candidates.

    The Draft checkpoint config is read to freeze the feature schema, but no
    Draft parameters, runner, attention backend, memory pool, or CUDA graph is
    created in this process.
    """

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
        self._need_mamba_verify_commit = False
        self._warned_sampling_fallback = False

        draft_model_config = ModelConfig.from_server_args(
            server_args,
            model_path=server_args.speculative_draft_model_path,
            model_revision=server_args.speculative_draft_model_revision,
            is_draft_model=True,
        )
        flash_config = parse_flashmtp_draft_config(
            draft_hf_config=draft_model_config.hf_config
        )
        self.remote_block_size = int(server_args.speculative_num_draft_tokens)
        self.block_size = self.remote_block_size
        self.speculative_num_draft_tokens = self.block_size
        self.num_proposed_tokens = self.block_size - 1
        self._mask_token_id = int(flash_config.mask_token_id)
        self._num_pivot_layers = len(flash_config.target_layer_ids)

        target_config = target_worker.model_runner.model_config
        if int(target_config.hidden_size) != int(draft_model_config.hidden_size):
            raise ValueError(
                "FLASHMTP target/draft hidden size mismatch: "
                f"target={target_config.hidden_size}, "
                f"draft={draft_model_config.hidden_size}."
            )
        if int(target_config.vocab_size) != int(draft_model_config.vocab_size):
            raise ValueError(
                "FLASHMTP target/draft vocab size mismatch: "
                f"target={target_config.vocab_size}, "
                f"draft={draft_model_config.vocab_size}."
            )
        self._pivot_width = self._num_pivot_layers * int(target_config.hidden_size)

        self._block_pos_offsets = build_block_pos_offsets(
            length=self.block_size, device=self.device
        )
        self._draft_block_ids_buf: torch.Tensor | None = None
        self._draft_block_positions_buf: torch.Tensor | None = None
        self._draft_block_tokens_buf: torch.Tensor | None = None
        self._draft_verify_out_cache_loc_buf: torch.Tensor | None = None
        self._draft_block_end_buf: torch.Tensor | None = None
        self._draft_seq_lens_cpu_buf: torch.Tensor | None = None

        supports_gpu_triton = is_cuda() or is_hip()
        self._use_triton_prepare_block = supports_gpu_triton
        self._use_triton_accept_bonus = supports_gpu_triton
        self._accept_bonus_buffer_cap = 0
        self._accept_bonus_buffer_slot = 0
        self._accept_len_buf = None
        self._commit_lens_bufs: list[torch.Tensor] = []
        self._bonus_id_bufs: list[torch.Tensor] = []
        self._out_tokens_bufs: list[torch.Tensor] = []
        self._new_seq_lens_bufs: list[torch.Tensor] = []
        validate_component_dependencies()
        adaptive_margin = os.environ.get("SGLANG_DT_ADAPTIVE_VERIFY_MARGIN")
        self._ragged_verify = ragged_verify_enabled() or adaptive_margin is not None
        self._adaptive_verify = adaptive_verify_enabled()
        self._adaptive_verify_margin = (
            0.0
            if self._adaptive_verify
            else (float(adaptive_margin) if adaptive_margin is not None else None)
        )
        self._adaptive_fill_graph_padding = (
            graph_tier_padding_enabled()
            or os.environ.get("SGLANG_DT_ADAPTIVE_FILL_GRAPH_PADDING", "0") == "1"
        )
        self._vocab_parallel_greedy = (
            vocab_parallel_greedy_enabled()
            or os.environ.get("SGLANG_DT_VOCAB_PARALLEL_GREEDY", "0") == "1"
        )
        self._single_load_verify_graph = (
            os.environ.get("SGLANG_DT_SINGLE_LOAD_VERIFY_GRAPH", "0") == "1"
        )
        self._adaptive_graph_tier_offset = int(
            os.environ.get("SGLANG_DT_ADAPTIVE_GRAPH_TIER_OFFSET", "0")
        )
        if self._adaptive_graph_tier_offset < 0:
            raise ValueError(
                "SGLANG_DT_ADAPTIVE_GRAPH_TIER_OFFSET must be non-negative"
            )
        if self._adaptive_graph_tier_offset and not self._adaptive_fill_graph_padding:
            raise ValueError(
                "SGLANG_DT_ADAPTIVE_GRAPH_TIER_OFFSET requires "
                "SGLANG_DT_ADAPTIVE_FILL_GRAPH_PADDING=1"
            )
        if (
            self._adaptive_verify_margin is not None
            and self._adaptive_verify_margin < 0
        ):
            raise ValueError("SGLANG_DT_ADAPTIVE_VERIFY_MARGIN must be non-negative")
        if self._ragged_verify and not ragged_verify_compact_enabled():
            raise ValueError(
                "ragged DT verify requires SGLANG_RAGGED_VERIFY_MODE=compact"
            )

        greedy_selector = None
        if self._vocab_parallel_greedy:
            if not self._ragged_verify:
                raise ValueError("DT vocab-parallel greedy requires ragged verify")
            if server_args.disable_cuda_graph or not is_cuda():
                raise ValueError(
                    "DT vocab-parallel greedy currently requires CUDA graph verify"
                )
            target_model = self.model_runner.model
            lm_head = getattr(target_model, "lm_head", None)
            if lm_head is None or not hasattr(lm_head, "weight"):
                raise ValueError("DT vocab-parallel greedy requires a target lm_head")
            if not torch.is_floating_point(lm_head.weight):
                raise ValueError(
                    "DT vocab-parallel greedy does not support a quantized lm_head"
                )
            shard = getattr(lm_head, "shard_indices", None)
            if shard is None:
                raise ValueError(
                    "DT vocab-parallel greedy requires contiguous vocab shard metadata"
                )
            if int(shard.num_added_elements) != 0:
                raise ValueError(
                    "DT vocab-parallel greedy does not support added vocab rows"
                )
            tp_group = self.model_runner.tp_group
            greedy_selector = VocabParallelGreedySelector(
                max_tokens=(ragged_epilogue_capacity(server_args) * self.block_size),
                num_org=int(shard.num_org_elements),
                org_vocab_start=int(shard.org_vocab_start_index),
                # LogitsProcessor copies CUDA-graph logits into its persistent
                # FP32 output buffer even when the lm_head weights are BF16.
                # Match that runtime contract instead of the weight dtype.
                value_dtype=torch.float32,
                device=self.device,
                tp_group=tp_group if int(tp_group.world_size) > 1 else None,
            )

        self._ragged_verify_epilogue = None
        if self._ragged_verify and not server_args.disable_cuda_graph and is_cuda():
            self._ragged_verify_epilogue = DsparkVerifyEpilogue(
                # CUDA graph capture stops at its configured batch size, but a
                # full remote wave can be larger and correctly falls back to
                # eager execution.  The tail hook and begin_step still use the
                # persistent epilogue in that eager path, so its capacity must
                # cover the scheduler's request limit rather than the graph
                # replay limit alone.
                max_bs=ragged_epilogue_capacity(server_args),
                verify_num_draft_tokens=self.block_size,
                device=self.device,
                greedy_selector=greedy_selector,
            )
            self.model_runner.capture_tail_hooks.append(
                self._ragged_verify_epilogue.capture_hook
            )

        if self.ps.tp_rank == 0:
            logger.info(
                "Initialized disaggregated FLASHMTP Target worker without Draft "
                "weights. block_size=%d remote_block_size=%d pivot_layers=%d "
                "pivot_width=%d vocab_parallel_greedy=%s",
                self.block_size,
                self.remote_block_size,
                self._num_pivot_layers,
                self._pivot_width,
                self._vocab_parallel_greedy,
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
        # Target attention backends are initialized by TpModelWorker.
        pass

    def init_cuda_graphs(self) -> None:
        # Target CUDA graphs are initialized by TpModelWorker. There is no
        # local Draft graph in this role.
        pass

    def _ensure_draft_block_buffers(self, bs: int) -> None:
        DFlashWorkerV2._ensure_draft_block_buffers(self, bs)

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        grammar_barrier=None,
    ) -> GenerationBatchResult:
        if getattr(batch, "return_logprob", False):
            raise ValueError(
                "FLASHMTP speculative decoding does not support return_logprob yet."
            )
        self._validate_phase1_sampling_support(batch)

        if batch.dt_mixed_composite:
            return self._forward_composite_phase(batch, on_publish=on_publish)
        if batch.dt_mixed_prefill_bs:
            return self._forward_mixed_phase(batch, on_publish=on_publish)

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return self._forward_prefill(batch, on_publish=on_publish)

        if batch.spec_info is None:
            batch.spec_info = DFlashDraftInputV2.create_idle_input(device=self.device)
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInputV2):
            raise TypeError(
                "disaggregated FLASHMTP Target expected DFlashDraftInputV2 state"
            )

        if batch.forward_mode.is_idle():
            return self._forward_idle(on_publish=on_publish)
        if draft_input.draft_tokens is None:
            raise RuntimeError(
                "disaggregated FLASHMTP Target cannot verify before remote Draft "
                "candidates are attached"
            )

        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )
        batch_size = len(batch.seq_lens)
        prefix_lens = batch.seq_lens
        remote_candidates = draft_input.draft_tokens
        if tuple(remote_candidates.shape) != (batch_size, self.block_size):
            raise ValueError(
                "remote FLASHMTP candidate shape mismatch: "
                f"expected {(batch_size, self.block_size)}, "
                f"got {tuple(remote_candidates.shape)}"
            )
        can_run_ragged_verify = self._can_run_ragged_verify(
            batch=batch, draft_input=draft_input
        )
        if self._vocab_parallel_greedy and not can_run_ragged_verify:
            raise RuntimeError(
                "DT vocab-parallel greedy requires adjustment-free greedy "
                "ragged verify for every request"
            )
        if can_run_ragged_verify:
            # READY batch construction already materializes remote candidates
            # as batch-owned int64 device storage. The ragged path does not
            # require the fixed-layout scratch buffer, so consuming it directly
            # avoids one scheduler-critical device copy per verify wave.
            candidates = remote_candidates.to(device=self.device, dtype=torch.int64)
            result = self._verify_candidates_ragged(
                batch=batch,
                draft_input=draft_input,
                candidates=candidates,
                prefix_lens=prefix_lens,
                on_publish=on_publish,
            )
            self._emit_accept_profile(batch=batch, result=result)
            return result

        self._ensure_draft_block_buffers(batch_size)
        assert self._draft_block_ids_buf is not None
        assert self._draft_block_positions_buf is not None
        assert self._draft_block_tokens_buf is not None
        assert self._draft_verify_out_cache_loc_buf is not None
        candidates = self._draft_block_tokens_buf[:batch_size]
        candidates.copy_(remote_candidates.to(device=self.device, dtype=torch.int64))
        # DT validates every candidate root against the round's bonus token in
        # ``DraftTargetController._consume_result_batch`` before publishing a
        # READY event, then broadcasts the validated bytes and metadata to all
        # TP ranks.  Repeating that check here with ``torch.equal`` forces the
        # scheduler thread to synchronize the forward stream.  Under overlap
        # scheduling that stream can still be executing the previous Target
        # wave, so the redundant validation prevents CPU result processing and
        # next-wave preparation from overlapping the remaining GPU work.

        block_ids = self._draft_block_ids_buf[:batch_size]
        target_positions_2d = self._draft_block_positions_buf[:batch_size]
        verify_out_cache_loc_2d = self._draft_verify_out_cache_loc_buf[:batch_size]
        self._prepare_target_layout(
            batch=batch,
            draft_input=draft_input,
            prefix_lens=prefix_lens,
            block_ids=block_ids,
            target_positions_2d=target_positions_2d,
            verify_out_cache_loc_2d=verify_out_cache_loc_2d,
        )

        result = self._verify_candidates(
            batch=batch,
            draft_input=draft_input,
            candidates=candidates,
            prefix_lens=prefix_lens,
            target_positions_2d=target_positions_2d,
            verify_out_cache_loc=verify_out_cache_loc_2d.reshape(-1),
            on_publish=on_publish,
            grammar_barrier=grammar_barrier,
        )
        self._emit_accept_profile(batch=batch, result=result)
        return result

    def _can_run_ragged_verify(
        self, *, batch: ScheduleBatch, draft_input: DFlashDraftInputV2
    ) -> bool:
        if not self._ragged_verify:
            return False
        if draft_input.verify_lens_cpu is None:
            raise RuntimeError("ragged DT verify result is missing verify lengths")
        sampling_info = batch.sampling_info
        return bool(
            not batch.has_grammar
            and sampling_info is not None
            and sampling_info.is_all_greedy
            and verify_logits_adjustments_are_noop(sampling_info)
        )

    def _verify_candidates_ragged(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        candidates: torch.Tensor,
        prefix_lens: torch.Tensor,
        on_publish=None,
    ) -> GenerationBatchResult:
        prepare_start_ns = time.monotonic_ns()
        batch_size = int(candidates.shape[0])
        verify_lens_cpu_tensor = draft_input.verify_lens_cpu
        assert verify_lens_cpu_tensor is not None
        verify_lens_cpu = [int(value) for value in verify_lens_cpu_tensor.tolist()]
        if len(verify_lens_cpu) != batch_size:
            raise ValueError(
                "adaptive verify length batch mismatch: "
                f"got {len(verify_lens_cpu)}, expected {batch_size}"
            )
        if not verify_lens_cpu or not (
            1 <= min(verify_lens_cpu) <= max(verify_lens_cpu) <= self.block_size
        ):
            raise ValueError(
                f"adaptive verify lengths must be in [1, {self.block_size}]"
            )
        validate_done_ns = time.monotonic_ns()

        graph_runner = self.model_runner.decode_cuda_graph_runner
        total_verify_tokens = sum(verify_lens_cpu)
        capture_grid = (
            graph_runner.capture_num_tokens
            if graph_runner is not None and graph_runner.capture_num_tokens
            else []
        )
        graph_num_tokens, execution_grid, graph_tier_available = (
            plan_ragged_execution_tier(
                total_verify_tokens=total_verify_tokens,
                max_verify_tokens=batch_size * self.block_size,
                capture_grid=list(capture_grid),
                tier_offset=self._adaptive_graph_tier_offset,
                batch_size=batch_size,
                max_graph_rows=(graph_runner.max_bs if graph_runner is not None else 0),
            )
        )
        original_total_verify_tokens = total_verify_tokens
        if (
            graph_tier_available
            and self._adaptive_fill_graph_padding
            and total_verify_tokens < graph_num_tokens
        ):
            padding_priorities = None
            if draft_input.padding_priorities_cpu is not None:
                padding_priorities = [
                    int(value) for value in draft_input.padding_priorities_cpu.tolist()
                ]
            verify_lens_cpu = reclaim_ragged_graph_padding(
                verify_lens_cpu,
                block_size=self.block_size,
                graph_num_tokens=graph_num_tokens,
                padding_priorities=padding_priorities,
            )
            total_verify_tokens = sum(verify_lens_cpu)
            verify_lens_cpu_tensor = torch.tensor(
                verify_lens_cpu,
                dtype=torch.int32,
                pin_memory=is_pin_memory_available(self.device),
            )
        tier_plan_done_ns = time.monotonic_ns()
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=verify_lens_cpu,
            device=self.device,
            grid=execution_grid,
            graph_num_tokens_floor=graph_num_tokens,
            verify_lens_host=verify_lens_cpu_tensor,
        )
        layout_done_ns = time.monotonic_ns()
        if self.ps.tp_rank == 0:
            profile_emit(
                "target_ragged_verify_batch",
                instance_id=self.server_args.draft_target_instance_id,
                batch_size=batch_size,
                total_verify_tokens=total_verify_tokens,
                graph_num_tokens=layout.graph_num_tokens,
                verify_lens=verify_lens_cpu,
                reclaimed_padding_tokens=(
                    total_verify_tokens - original_total_verify_tokens
                ),
                padding_priority_enabled=(
                    draft_input.padding_priorities_cpu is not None
                ),
                graph_tier_offset=self._adaptive_graph_tier_offset,
                graph_tier_available=graph_tier_available,
            )
        ragged_window = BuildRaggedVerifyWindow.execute(
            batch=batch,
            layout=layout,
            # The compact gather kernel uses the proposal width as its row
            # stride for this anchor buffer. Keep exactly gamma columns.
            draft_block_ids=candidates[:, : self.num_proposed_tokens],
            draft_tokens=candidates[:, 1:],
            bs=batch_size,
            device=self.device,
            verify_num_draft_tokens=self.block_size,
            model_runner=self.model_runner,
        )
        window_done_ns = time.monotonic_ns()

        verify_input = DFlashVerifyInput(
            draft_token=ragged_window.verify_ids,
            positions=ragged_window.positions,
            draft_token_num=self.block_size,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            ragged_verify_layout=layout,
        )
        batch.out_cache_loc = ragged_window.verify_cache_loc
        seq_lens_cpu_backup = batch.seq_lens_cpu
        seq_lens_sum_backup = batch.seq_lens_sum
        if seq_lens_cpu_backup is not None:
            batch.seq_lens_cpu = seq_lens_cpu_backup + verify_lens_cpu_tensor.to(
                dtype=seq_lens_cpu_backup.dtype
            )
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())

        epilogue = self._ragged_verify_epilogue
        if epilogue is not None:
            epilogue.begin_step(layout.verify_lens, armed=False)
            epilogue.draft_tokens_buf[: batch_size * self.num_proposed_tokens].copy_(
                candidates[:, 1:].reshape(-1)
            )
        epilogue_prepare_done_ns = time.monotonic_ns()

        verify_forward_batch, can_run_cuda_graph = verify_input.prepare_for_verify(
            batch,
            self.target_worker,
            defer_cuda_graph_load=self._single_load_verify_graph,
        )
        forward_prepare_done_ns = time.monotonic_ns()
        batch.seq_lens_cpu = seq_lens_cpu_backup
        batch.seq_lens_sum = seq_lens_sum_backup
        target_out = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            # With a deferred graph load, the ordinary graph runner must do
            # its full load in the forward.  Eager fallback remains preplanned.
            skip_attn_backend_init=(
                not self._single_load_verify_graph or not can_run_cuda_graph
            ),
        )
        forward_submit_done_ns = time.monotonic_ns()
        if self.ps.tp_rank == 0:
            profile_emit(
                "target_ragged_verify_forward",
                instance_id=self.server_args.draft_target_instance_id,
                batch_size=batch_size,
                total_verify_tokens=total_verify_tokens,
                graph_num_tokens=layout.graph_num_tokens,
                can_run_cuda_graph=target_out.can_run_cuda_graph,
                validate_ns=validate_done_ns - prepare_start_ns,
                tier_plan_ns=tier_plan_done_ns - validate_done_ns,
                layout_ns=layout_done_ns - tier_plan_done_ns,
                window_ns=window_done_ns - layout_done_ns,
                epilogue_prepare_ns=(epilogue_prepare_done_ns - window_done_ns),
                forward_prepare_ns=(forward_prepare_done_ns - epilogue_prepare_done_ns),
                forward_submit_ns=(forward_submit_done_ns - forward_prepare_done_ns),
                total_prepare_ns=forward_submit_done_ns - prepare_start_ns,
            )
        logits_output = target_out.logits_output

        epilogue_ran = target_out.can_run_cuda_graph and not (
            epilogue is not None and epilogue.uses_vocab_parallel_greedy
        )
        if (
            epilogue is not None
            and epilogue.uses_vocab_parallel_greedy
            and not epilogue_ran
        ):
            epilogue(
                compact_logits=logits_output.next_token_logits,
                compact_hidden=logits_output.hidden_states,
                input_ids=verify_forward_batch.input_ids,
                seq_lens=verify_forward_batch.seq_lens,
                req_pool_indices=verify_forward_batch.req_pool_indices,
                bs=batch_size,
            )
            epilogue_ran = True

        if epilogue_ran:
            if epilogue is None:
                raise RuntimeError(
                    "ragged CUDA graph replay requires a persistent verify epilogue"
                )
            accept_outs = epilogue.read_accept(batch_size)
            # The epilogue owns one graph-stable output buffer.  Overlap can
            # replay the next graph before the scheduler consumes this result,
            # so materialize every tensor that crosses the forward boundary.
            commit_lens = accept_outs.commit_lens.clone()
            accept_len = commit_lens.to(torch.int64) - 1
            bonus = accept_outs.bonus.clone()
            out_tokens = accept_outs.out_tokens.clone()
            new_seq_lens = accept_outs.new_seq_lens.clone()
            if epilogue.uses_vocab_parallel_greedy:
                # Every consumer in this path uses the already-materialized
                # token ids. Keeping rank-local logits in the public result
                # would misrepresent them as a full-vocab tensor.
                logits_output.next_token_logits = None
            else:
                strided_logits = epilogue.strided_logits
                assert strided_logits is not None
                logits_output.next_token_logits = strided_logits[
                    : batch_size * self.block_size
                ]
            packed_hidden = epilogue.strided_hidden
            assert packed_hidden is not None
            packed_hidden = packed_hidden[: batch_size * self.block_size]
        else:
            logits_output.next_token_logits = ScatterCompactToStrided.execute(
                compact=logits_output.next_token_logits,
                layout=layout,
                fill_value=0.0,
                verify_num_draft_tokens=self.block_size,
            )
            compact_hidden = logits_output.hidden_states
            if compact_hidden is None:
                raise RuntimeError(
                    "FLASHMTP ragged verify requires target auxiliary hidden states"
                )
            packed_hidden = ScatterCompactToStrided.execute(
                compact=compact_hidden,
                layout=layout,
                fill_value=0.0,
                verify_num_draft_tokens=self.block_size,
            )
            target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).view(
                batch_size, self.block_size
            )
            accept_len, _ = compute_dflash_correct_drafts_and_bonus(
                candidates=candidates,
                target_predict=target_predict,
            )
            accept_len = torch.minimum(
                accept_len, layout.verify_lens.to(torch.int64) - 1
            )
            bonus = target_predict.gather(1, accept_len[:, None]).view(-1)
            commit_lens = accept_len.to(torch.int32) + 1
            new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
            out_tokens = torch.empty(
                (batch_size, self.block_size),
                dtype=torch.int64,
                device=self.device,
            )
            out_tokens[:, : self.num_proposed_tokens].copy_(candidates[:, 1:])
            out_tokens[:, self.num_proposed_tokens].fill_(0)
            out_tokens.scatter_(1, accept_len[:, None], bonus[:, None])

        if on_publish is not None:
            on_publish(new_seq_lens)
        next_pivot = gather_flashmtp_next_pivot(
            packed_hidden=packed_hidden,
            accept_len=accept_len,
            block_size=self.block_size,
        )
        logits_output.hidden_states = None
        next_input = self._make_next_input(
            bonus_tokens=bonus,
            new_seq_lens=new_seq_lens,
            pivot_hidden=next_pivot,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=commit_lens,
            can_run_cuda_graph=target_out.can_run_cuda_graph,
            next_draft_input=next_input,
            speculative_num_draft_tokens=self.block_size,
            new_seq_lens=new_seq_lens,
        )

    def _emit_accept_profile(
        self, *, batch: ScheduleBatch, result: GenerationBatchResult
    ) -> None:
        if (
            self.ps.tp_rank == 0
            and os.environ.get("SGLANG_DT_DRAFT_CONFIDENCE_PROFILE", "0") == "1"
        ):
            profile_emit(
                "target_accept_immediate_batch",
                instance_id=self.server_args.draft_target_instance_id,
                request_ids=[req.rid for req in batch.reqs],
                accept_lens=result.accept_lens.tolist(),
            )

    def _forward_mixed_phase(
        self, batch: ScheduleBatch, *, on_publish=None
    ) -> GenerationBatchResult:
        """Execute replacement prefill and READY_VERIFY in one eager EXTEND."""
        prepare_start_ns = time.monotonic_ns()
        prefill_bs = batch.dt_mixed_prefill_bs
        prefill_num_tokens = batch.dt_mixed_prefill_num_tokens
        verify_batch = batch.dt_mixed_verify_batch
        verify_lens_cpu_tensor = batch.dt_mixed_verify_lens_cpu
        if verify_batch is None or verify_lens_cpu_tensor is None:
            raise RuntimeError("DT mixed-phase batch is missing verify metadata")
        draft_input = verify_batch.spec_info
        if not isinstance(draft_input, DFlashDraftInputV2):
            raise TypeError("DT mixed-phase verify state must be DFlashDraftInputV2")
        candidates = draft_input.draft_tokens
        if candidates is None:
            raise RuntimeError("DT mixed-phase verify state has no candidates")
        verify_bs = verify_batch.batch_size()
        if tuple(candidates.shape) != (verify_bs, self.block_size):
            raise ValueError("DT mixed-phase candidate shape mismatch")

        verify_lens_cpu = [int(value) for value in verify_lens_cpu_tensor.tolist()]
        if len(verify_lens_cpu) != verify_bs or not verify_lens_cpu:
            raise ValueError("DT mixed-phase verify length batch mismatch")
        if not (1 <= min(verify_lens_cpu) <= max(verify_lens_cpu) <= self.block_size):
            raise ValueError(
                f"DT mixed-phase verify lengths must be in [1, {self.block_size}]"
            )
        total_verify_tokens = sum(verify_lens_cpu)
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=verify_lens_cpu,
            device=self.device,
            grid=[total_verify_tokens],
            verify_lens_host=verify_lens_cpu_tensor,
        )
        ragged_window = BuildRaggedVerifyWindow.execute(
            batch=verify_batch,
            layout=layout,
            draft_block_ids=candidates[:, : self.num_proposed_tokens],
            draft_tokens=candidates[:, 1:],
            bs=verify_bs,
            device=self.device,
            verify_num_draft_tokens=self.block_size,
            model_runner=self.model_runner,
        )
        window_submit_done_ns = time.monotonic_ns()

        if batch.input_ids is None or batch.out_cache_loc is None:
            raise RuntimeError("DT mixed-phase prefill inputs were not materialized")
        batch.input_ids = torch.cat([batch.input_ids, ragged_window.verify_ids])
        batch.out_cache_loc = torch.cat(
            [batch.out_cache_loc, ragged_window.verify_cache_loc]
        )
        if batch.input_ids.numel() != batch.extend_num_tokens:
            raise RuntimeError("DT mixed-phase packed input length mismatch")
        if batch.out_cache_loc.numel() != batch.extend_num_tokens:
            raise RuntimeError("DT mixed-phase packed cache length mismatch")

        prefill_lens = torch.tensor(
            batch.extend_lens[:prefill_bs], dtype=torch.int64, device=self.device
        )
        verify_lens = layout.verify_lens.to(torch.int64)
        prefill_last = torch.cumsum(prefill_lens, dim=0) - 1
        verify_rows = torch.arange(
            prefill_num_tokens,
            prefill_num_tokens + total_verify_tokens,
            dtype=torch.int64,
            device=self.device,
        )
        batch.logits_select_index = torch.cat([prefill_last, verify_rows])
        mixed_forward_batch = ForwardBatch.init_new(
            batch,
            self.model_runner,
            # LAST returns every row selected above because mixed-phase
            # pruning happens before hidden-state capture.  It also matches
            # the existing FlashMTP prefill CUDA-graph capture contract.
            capture_hidden_mode=CaptureHiddenMode.LAST,
            return_hidden_states_before_norm=False,
        )
        mixed_forward_batch.device_timer_category = "mixed"
        pack_submit_done_ns = time.monotonic_ns()
        target_out = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=mixed_forward_batch,
            is_verify=True,
        )
        model_submit_done_ns = time.monotonic_ns()
        logits_output = target_out.logits_output
        all_logits = logits_output.next_token_logits
        all_hidden = logits_output.hidden_states
        if all_logits is None or all_hidden is None:
            raise RuntimeError("DT mixed-phase requires full logits and pivot states")
        selected_rows = prefill_bs + total_verify_tokens
        if all_logits.shape[0] != selected_rows:
            raise RuntimeError("DT mixed-phase logits row count mismatch")
        if all_hidden.shape[0] != selected_rows:
            raise RuntimeError("DT mixed-phase hidden-state row count mismatch")

        verify_last = prefill_bs + torch.cumsum(verify_lens, dim=0) - 1
        sample_indices = torch.cat(
            [torch.arange(prefill_bs, device=self.device), verify_last]
        )
        sample_output = LogitsProcessorOutput(
            next_token_logits=all_logits[sample_indices]
        )
        sampled_ids = self.model_runner.sample(sample_output, mixed_forward_batch)
        prefill_tokens = sampled_ids[:prefill_bs]
        prefill_pivot = all_hidden[:prefill_bs]
        sample_submit_done_ns = time.monotonic_ns()

        compact_verify_logits = all_logits[prefill_bs:]
        compact_verify_hidden = all_hidden[prefill_bs:]
        strided_logits = ScatterCompactToStrided.execute(
            compact=compact_verify_logits,
            layout=layout,
            fill_value=0.0,
            verify_num_draft_tokens=self.block_size,
        )
        strided_hidden = ScatterCompactToStrided.execute(
            compact=compact_verify_hidden,
            layout=layout,
            fill_value=0.0,
            verify_num_draft_tokens=self.block_size,
        )
        target_predict = torch.argmax(strided_logits, dim=-1).view(
            verify_bs, self.block_size
        )
        accept_len, _ = compute_dflash_correct_drafts_and_bonus(
            candidates=candidates,
            target_predict=target_predict,
        )
        accept_len = torch.minimum(accept_len, verify_lens - 1)
        verify_bonus = target_predict.gather(1, accept_len[:, None]).view(-1)
        verify_commit_lens = accept_len.to(torch.int32) + 1
        verify_new_seq_lens = verify_batch.seq_lens + verify_commit_lens.to(
            verify_batch.seq_lens.dtype
        )
        verify_out_tokens = torch.empty(
            (verify_bs, self.block_size), dtype=torch.int64, device=self.device
        )
        verify_out_tokens[:, : self.num_proposed_tokens].copy_(candidates[:, 1:])
        verify_out_tokens[:, self.num_proposed_tokens].fill_(0)
        verify_out_tokens.scatter_(1, accept_len[:, None], verify_bonus[:, None])
        verify_pivot = gather_flashmtp_next_pivot(
            packed_hidden=strided_hidden,
            accept_len=accept_len,
            block_size=self.block_size,
        )
        verify_epilogue_submit_done_ns = time.monotonic_ns()

        total_bs = prefill_bs + verify_bs
        out_tokens = torch.zeros(
            (total_bs, self.block_size), dtype=torch.int64, device=self.device
        )
        out_tokens[:prefill_bs, 0] = prefill_tokens
        out_tokens[prefill_bs:] = verify_out_tokens
        commit_lens = torch.cat(
            [
                torch.ones(prefill_bs, dtype=torch.int32, device=self.device),
                verify_commit_lens,
            ]
        )
        new_seq_lens = torch.cat([batch.seq_lens[:prefill_bs], verify_new_seq_lens])
        bonus_tokens = torch.cat([prefill_tokens, verify_bonus])
        pivot_hidden = torch.cat([prefill_pivot, verify_pivot])
        if pivot_hidden.ndim != 2 or pivot_hidden.shape[-1] != self._pivot_width:
            raise RuntimeError("DT mixed-phase pivot shape mismatch")
        result_pack_submit_done_ns = time.monotonic_ns()

        if on_publish is not None:
            on_publish(new_seq_lens)
        next_input = self._make_next_input(
            bonus_tokens=bonus_tokens,
            new_seq_lens=new_seq_lens,
            pivot_hidden=pivot_hidden,
        )
        next_input_submit_done_ns = time.monotonic_ns()
        logits_output.next_token_logits = None
        logits_output.hidden_states = None
        if self.ps.tp_rank == 0:
            profile_emit(
                "target_dt_mixed_phase",
                instance_id=self.server_args.draft_target_instance_id,
                prefill_batch_size=prefill_bs,
                verify_batch_size=verify_bs,
                prefill_tokens=prefill_num_tokens,
                verify_tokens=total_verify_tokens,
                selected_output_rows=selected_rows,
                can_run_cuda_graph=target_out.can_run_cuda_graph,
                window_submit_ns=window_submit_done_ns - prepare_start_ns,
                pack_submit_ns=pack_submit_done_ns - window_submit_done_ns,
                model_submit_ns=model_submit_done_ns - pack_submit_done_ns,
                prefill_graph_load_submit_ns=getattr(
                    mixed_forward_batch, "dt_prefill_graph_load_submit_ns", -1
                ),
                prefill_body_replay_submit_ns=getattr(
                    mixed_forward_batch, "dt_prefill_body_replay_submit_ns", -1
                ),
                prefill_eager_tail_submit_ns=getattr(
                    mixed_forward_batch, "dt_prefill_eager_tail_submit_ns", -1
                ),
                prefill_graph_finalize_submit_ns=getattr(
                    mixed_forward_batch, "dt_prefill_graph_finalize_submit_ns", -1
                ),
                bcg_segment_submit_ns=getattr(
                    mixed_forward_batch, "dt_bcg_segment_submit_ns", -1
                ),
                bcg_break_submit_ns=getattr(
                    mixed_forward_batch, "dt_bcg_break_submit_ns", -1
                ),
                bcg_segment_count=getattr(
                    mixed_forward_batch, "dt_bcg_segment_count", -1
                ),
                bcg_break_count=getattr(mixed_forward_batch, "dt_bcg_break_count", -1),
                sample_submit_ns=sample_submit_done_ns - model_submit_done_ns,
                verify_epilogue_submit_ns=(
                    verify_epilogue_submit_done_ns - sample_submit_done_ns
                ),
                result_pack_submit_ns=(
                    result_pack_submit_done_ns - verify_epilogue_submit_done_ns
                ),
                next_input_submit_ns=(
                    next_input_submit_done_ns - result_pack_submit_done_ns
                ),
                prepare_submit_ns=next_input_submit_done_ns - prepare_start_ns,
            )
        result = GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=commit_lens,
            can_run_cuda_graph=target_out.can_run_cuda_graph,
            next_draft_input=next_input,
            speculative_num_draft_tokens=self.block_size,
            new_seq_lens=new_seq_lens,
            extra_keep_alive_refs=[mixed_forward_batch, layout, ragged_window],
        )
        if (
            self.ps.tp_rank == 0
            and os.environ.get("SGLANG_DT_DRAFT_CONFIDENCE_PROFILE", "0") == "1"
        ):
            profile_emit(
                "target_accept_immediate_batch",
                instance_id=self.server_args.draft_target_instance_id,
                request_ids=[req.rid for req in verify_batch.reqs],
                accept_lens=verify_commit_lens.tolist(),
            )
        return result

    def _forward_composite_phase(
        self, batch: ScheduleBatch, *, on_publish=None
    ) -> GenerationBatchResult:
        """Submit prefill and verify back-to-back without changing their paths.

        The scheduler still exposes one logical batch to FutureMap and result
        processing, but the model sees the ordinary prefill forward followed
        by the optimized ragged verify forward.  This removes the scheduler
        round-trip without forcing verify through the slower mixed EXTEND path.
        """
        prepare_start_ns = time.monotonic_ns()
        prefill_batch = batch.dt_mixed_prefill_batch
        verify_batch = batch.dt_mixed_verify_batch
        prefill_bs = batch.dt_mixed_prefill_bs
        if prefill_batch is None or verify_batch is None or prefill_bs <= 0:
            raise RuntimeError("DT composite batch is missing logical sub-batches")
        verify_bs = verify_batch.batch_size()
        if verify_bs <= 0 or batch.batch_size() != prefill_bs + verify_bs:
            raise RuntimeError("DT composite logical batch size mismatch")
        if batch.input_ids is None:
            raise RuntimeError("DT composite prefill inputs were not materialized")
        if int(batch.input_ids.numel()) != batch.dt_mixed_prefill_num_tokens:
            raise RuntimeError("DT composite prefill input length mismatch")

        # Forward on shallow clones so phase-local ForwardBatch preparation can
        # mutate ScheduleBatch fields without corrupting the scheduler's two
        # logical result domains.  The prefill sampling snapshot was detached
        # before the top-level sampling merge; verify is detached here.
        prefill_forward_batch = dataclasses.replace(
            prefill_batch,
            input_ids=batch.input_ids,
            prefill_input_ids_cpu=None,
            forward_iter=batch.forward_iter,
            launch_ts=batch.launch_ts,
        )
        verify_forward_batch = dataclasses.replace(
            verify_batch,
            sampling_info=(
                verify_batch.sampling_info.copy_for_forward()
                if verify_batch.sampling_info is not None
                else None
            ),
            forward_iter=batch.forward_iter,
            launch_ts=batch.launch_ts,
        )
        clone_done_ns = time.monotonic_ns()

        prefill_result = self.forward_batch_generation(prefill_forward_batch)
        prefill_submit_done_ns = time.monotonic_ns()
        verify_result = self.forward_batch_generation(verify_forward_batch)
        verify_submit_done_ns = time.monotonic_ns()

        prefill_input = prefill_result.next_draft_input
        verify_input = verify_result.next_draft_input
        if prefill_input is None or verify_input is None:
            raise RuntimeError("DT composite phase did not produce Draft inputs")
        prefill_tokens = prefill_result.next_token_ids
        verify_tokens = verify_result.next_token_ids
        verify_commit_lens = verify_result.accept_lens
        if (
            not torch.is_tensor(prefill_tokens)
            or not torch.is_tensor(verify_tokens)
            or verify_commit_lens is None
        ):
            raise RuntimeError("DT composite phase result tensors are missing")
        if int(prefill_tokens.numel()) != prefill_bs:
            raise RuntimeError("DT composite prefill result size mismatch")
        if int(verify_tokens.numel()) != verify_bs * self.block_size:
            raise RuntimeError("DT composite verify result size mismatch")

        out_tokens = torch.zeros(
            (prefill_bs + verify_bs, self.block_size),
            dtype=torch.int64,
            device=self.device,
        )
        out_tokens[:prefill_bs, 0].copy_(prefill_tokens)
        out_tokens[prefill_bs:].copy_(verify_tokens.view(verify_bs, self.block_size))
        commit_lens = torch.cat(
            [
                torch.ones(prefill_bs, dtype=torch.int32, device=self.device),
                verify_commit_lens.to(dtype=torch.int32),
            ]
        )
        new_seq_lens = torch.cat(
            [prefill_input.new_seq_lens, verify_input.new_seq_lens]
        )
        bonus_tokens = torch.cat(
            [prefill_input.bonus_tokens, verify_input.bonus_tokens]
        )
        pivot_hidden = torch.cat(
            [prefill_input.hidden_states, verify_input.hidden_states]
        )
        if pivot_hidden.ndim != 2 or pivot_hidden.shape != (
            prefill_bs + verify_bs,
            self._pivot_width,
        ):
            raise RuntimeError("DT composite pivot shape mismatch")
        next_input = self._make_next_input(
            bonus_tokens=bonus_tokens,
            new_seq_lens=new_seq_lens,
            pivot_hidden=pivot_hidden,
        )
        pack_submit_done_ns = time.monotonic_ns()

        if on_publish is not None:
            on_publish(new_seq_lens)

        # Result processing is token-only for the supported greedy/no-logprob
        # envelope.  Drop both phase-local large logits tensors immediately.
        logits_output = prefill_result.logits_output
        if logits_output is None:
            logits_output = LogitsProcessorOutput(next_token_logits=None)
        else:
            logits_output.next_token_logits = None
            logits_output.hidden_states = None
        if verify_result.logits_output is not None:
            verify_result.logits_output.next_token_logits = None
            verify_result.logits_output.hidden_states = None

        if self.ps.tp_rank == 0:
            profile_emit(
                "target_dt_composite_phase",
                instance_id=self.server_args.draft_target_instance_id,
                prefill_batch_size=prefill_bs,
                verify_batch_size=verify_bs,
                prefill_tokens=batch.dt_mixed_prefill_num_tokens,
                verify_tokens=int(verify_tokens.numel()),
                prefill_can_run_cuda_graph=prefill_result.can_run_cuda_graph,
                verify_can_run_cuda_graph=verify_result.can_run_cuda_graph,
                clone_submit_ns=clone_done_ns - prepare_start_ns,
                prefill_submit_ns=prefill_submit_done_ns - clone_done_ns,
                verify_submit_ns=verify_submit_done_ns - prefill_submit_done_ns,
                pack_submit_ns=pack_submit_done_ns - verify_submit_done_ns,
                total_submit_ns=pack_submit_done_ns - prepare_start_ns,
            )

        keep_alive = [
            prefill_forward_batch,
            verify_forward_batch,
            prefill_result,
            verify_result,
        ]
        if prefill_result.extra_keep_alive_refs:
            keep_alive.extend(prefill_result.extra_keep_alive_refs)
        if verify_result.extra_keep_alive_refs:
            keep_alive.extend(verify_result.extra_keep_alive_refs)
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=commit_lens,
            can_run_cuda_graph=(
                prefill_result.can_run_cuda_graph and verify_result.can_run_cuda_graph
            ),
            next_draft_input=next_input,
            speculative_num_draft_tokens=self.block_size,
            new_seq_lens=new_seq_lens,
            extra_keep_alive_refs=keep_alive,
        )

    def _forward_prefill(
        self, batch: ScheduleBatch, *, on_publish=None
    ) -> GenerationBatchResult:
        batch_output = self.target_worker.forward_batch_generation(
            batch, capture_hidden_mode=CaptureHiddenMode.LAST
        )
        logits_output = batch_output.logits_output
        next_token_ids = batch_output.next_token_ids
        pivot_hidden = logits_output.hidden_states
        if pivot_hidden is None:
            raise RuntimeError(
                "disaggregated FLASHMTP prefill requires target pivot hidden states"
            )
        if pivot_hidden.ndim != 2 or pivot_hidden.shape[-1] != self._pivot_width:
            raise RuntimeError(
                "FLASHMTP prefill pivot shape mismatch: "
                f"expected [bs, {self._pivot_width}], got {tuple(pivot_hidden.shape)}"
            )
        logits_output.hidden_states = None
        batch_output.new_seq_lens = batch.seq_lens
        if on_publish is not None:
            on_publish(batch_output.new_seq_lens)
        batch_output.next_draft_input = self._make_next_input(
            bonus_tokens=next_token_ids,
            new_seq_lens=batch.seq_lens,
            pivot_hidden=pivot_hidden,
        )
        return batch_output

    def _forward_idle(self, *, on_publish=None) -> GenerationBatchResult:
        empty_ids = torch.empty((0,), dtype=torch.int64, device=self.device)
        empty_lens = torch.empty((0,), dtype=torch.int32, device=self.device)
        next_input = self._make_next_input(
            bonus_tokens=empty_ids,
            new_seq_lens=empty_ids,
            pivot_hidden=torch.empty(
                (0, self._pivot_width), dtype=torch.bfloat16, device=self.device
            ),
        )
        if on_publish is not None:
            on_publish(next_input.new_seq_lens)
        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=empty_ids,
            accept_lens=empty_lens,
            next_draft_input=next_input,
            can_run_cuda_graph=False,
            speculative_num_draft_tokens=self.block_size,
            new_seq_lens=next_input.new_seq_lens,
        )

    def _prepare_target_layout(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        prefix_lens: torch.Tensor,
        block_ids: torch.Tensor,
        target_positions_2d: torch.Tensor,
        verify_out_cache_loc_2d: torch.Tensor,
    ) -> None:
        if self._use_triton_prepare_block:
            try:
                _prepare_dflash_draft_block_unchecked(
                    bonus_tokens=draft_input.bonus_tokens.view(-1),
                    prefix_lens=prefix_lens.view(-1),
                    req_pool_indices=batch.req_pool_indices.view(-1),
                    req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                    block_ids_out=block_ids,
                    positions_out=target_positions_2d,
                    cache_loc_out=verify_out_cache_loc_2d,
                    mask_token_id=self._mask_token_id,
                )
                return
            except Exception as exc:  # noqa: BLE001
                self._use_triton_prepare_block = False
                logger.warning(
                    "FLASHMTP Target Triton prepare_block failed; falling back "
                    "to eager path: %s",
                    exc,
                )
        self._prepare_draft_block_eager(
            draft_input=draft_input,
            prefix_lens=prefix_lens,
            req_pool_indices=batch.req_pool_indices,
            block_ids=block_ids,
            target_positions_2d=target_positions_2d,
            verify_out_cache_loc_2d=verify_out_cache_loc_2d,
        )
