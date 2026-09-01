from __future__ import annotations

import logging

import torch
from sglang.kernels.ops.speculative.dflash import (
    _prepare_dflash_draft_block_unchecked,
)
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.draft_worker_common import build_block_pos_offsets
from sglang.srt.speculative.flashmtp_utils import parse_flashmtp_draft_config
from sglang.srt.speculative.flashmtp_worker_v2 import FlashMTPWorkerV2
from sglang.srt.utils import is_cuda, is_hip

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
        self.block_size = int(server_args.speculative_num_draft_tokens)
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

        if self.ps.tp_rank == 0:
            logger.info(
                "Initialized disaggregated FLASHMTP Target worker without Draft "
                "weights. block_size=%d pivot_layers=%d pivot_width=%d",
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
        return result

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
