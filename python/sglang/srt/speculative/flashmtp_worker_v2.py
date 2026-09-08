from __future__ import annotations

import logging
import os
import time
from dataclasses import replace

import torch
from sglang.kernels.ops.speculative.cache_locs import assign_extend_cache_locs_func
from sglang.kernels.ops.speculative.dflash import (
    _compute_dflash_accept_bonus_triton_unchecked,
    _prepare_dflash_draft_block_unchecked,
)
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_verify_logits_adjustments,
    compute_dflash_correct_drafts_and_bonus,
    compute_dflash_sampling_correct_drafts_and_bonus,
    is_dflash_sampling_verify_available,
)
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.draft_worker_common import (
    build_block_pos_offsets,
    build_draft_tp_worker,
    make_draft_block_spec_info,
    make_draft_input_v2,
    make_draft_sampler_capture_hook,
)
from sglang.srt.speculative.flashmtp_utils import (
    gather_flashmtp_next_pivot,
    parse_flashmtp_draft_config,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    GrammarTree,
    build_grammar_vocab_mask,
)
from sglang.srt.utils import get_available_gpu_memory, is_cuda, is_hip

logger = logging.getLogger(__name__)


class _FlashMTPMarkovDraftSampler:
    """Capture-safe wrapper for FlashMTP v2's serial Markov sampler."""

    def __init__(
        self,
        *,
        draft_model,
        draft_block_len: int,
        num_proposals: int,
        max_bs: int,
    ):
        self.draft_model = draft_model
        self.draft_block_len = int(draft_block_len)
        self.num_proposals = int(num_proposals)
        device = next(draft_model.parameters()).device
        self.out = torch.empty(
            (int(max_bs) * self.num_proposals,), dtype=torch.int64, device=device
        )

    def __call__(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> None:
        batch_size = hidden_states.shape[0] // self.draft_block_len
        block_hidden = hidden_states.view(batch_size, self.draft_block_len, -1)
        anchors = input_ids.view(batch_size, self.draft_block_len)[:, 0]
        sampled = self.draft_model.sample_draft_tokens(
            block_hidden=block_hidden,
            first_prev_token_ids=anchors,
        )
        self.out[: batch_size * self.num_proposals].copy_(sampled.reshape(-1))


class FlashMTPWorkerV2(DFlashWorkerV2):
    """FlashMTP V2 worker.

    It reuses DFlash's linear-chain verify utilities and overlap transport, but
    replaces the draft-KV path with a per-request multi-layer pivot hidden state.
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
        self.model_runner = target_worker.model_runner
        self.device = target_worker.device
        self.page_size = server_args.page_size
        self._need_mamba_verify_commit = False
        self.draft_window_size = None
        self.use_compact_draft_cache = False
        self._warned_sampling_fallback = False
        self._logged_first_verify = False

        bundle = build_draft_tp_worker(
            server_args=server_args,
            gpu_id=gpu_id,
            ps=replace(ps, pp_rank=0),
            nccl_port=nccl_port,
            target_model_config=target_worker.model_runner.model_config,
            algo_label="FLASHMTP",
        )
        self._draft_worker = bundle.draft_worker
        self.draft_model_runner = bundle.draft_model_runner
        self.draft_model = bundle.draft_model
        self._draft_sampler = None

        flash_config = parse_flashmtp_draft_config(
            draft_hf_config=self.draft_model_runner.model_config.hf_config
        )
        self.block_size = int(server_args.speculative_num_draft_tokens)
        self.speculative_num_draft_tokens = self.block_size
        self.num_proposed_tokens = self.block_size - 1
        self._draft_block_len = (
            self.block_size - 1 if flash_config.left_shift else self.block_size
        )
        self._mask_token_id = int(flash_config.mask_token_id)
        self._num_pivot_layers = len(flash_config.target_layer_ids)

        target_config = target_worker.model_runner.model_config
        if int(target_config.hidden_size) != int(
            self.draft_model_runner.model_config.hidden_size
        ):
            raise ValueError(
                "FLASHMTP target/draft hidden size mismatch: "
                f"target={target_config.hidden_size}, "
                f"draft={self.draft_model_runner.model_config.hidden_size}."
            )
        if int(target_config.vocab_size) != int(
            self.draft_model_runner.model_config.vocab_size
        ):
            raise ValueError(
                "FLASHMTP target/draft vocab size mismatch: "
                f"target={target_config.vocab_size}, "
                f"draft={self.draft_model_runner.model_config.vocab_size}."
            )
        if self._mask_token_id >= int(target_config.vocab_size):
            raise ValueError(
                "FLASHMTP mask token is outside the target vocabulary: "
                f"mask_token_id={self._mask_token_id}, "
                f"vocab_size={target_config.vocab_size}."
            )

        self._block_pos_offsets = build_block_pos_offsets(
            length=self.block_size, device=self.device
        )
        self._draft_local_positions = self._block_pos_offsets + 1
        self._draft_local_positions_buf: torch.Tensor | None = None
        self._draft_block_ids_buf: torch.Tensor | None = None
        self._draft_block_positions_buf: torch.Tensor | None = None
        self._draft_block_tokens_buf: torch.Tensor | None = None
        self._draft_verify_out_cache_loc_buf: torch.Tensor | None = None
        self._draft_block_end_buf: torch.Tensor | None = None
        self._draft_seq_lens_cpu_buf: torch.Tensor | None = None
        self._draft_block_spec_info = make_draft_block_spec_info(
            draft_token_num=self._draft_block_len, device=self.device
        )

        # Buffers consumed by inherited vocab-parallel sampling helpers.
        self._draft_greedy_gathered_max_buf = None
        self._draft_greedy_gathered_ids_buf = None
        self._draft_greedy_gather_cap = 0
        self._draft_greedy_local_max_buf = None
        self._draft_greedy_local_arg_buf = None
        self._draft_greedy_local_cap = 0
        self._draft_greedy_best_rank_buf = None
        self._draft_greedy_rank_index_buf = None
        self._draft_greedy_selected_ids_buf = None
        self._draft_greedy_index_cap = 0

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
                "Initialized FLASHMTP draft runner. model=%s, block_size=%d, "
                "num_proposed_tokens=%d, pivot_layers=%d, mask_token_id=%d",
                self.draft_model.__class__.__name__,
                self.block_size,
                self.num_proposed_tokens,
                self._num_pivot_layers,
                self._mask_token_id,
            )

    def _maybe_build_draft_sampler(self):
        if getattr(self.draft_model, "markov_head", None) is None:
            if bool(getattr(self.draft_model, "left_shift", False)):
                if self.ps.tp_rank == 0:
                    logger.info(
                        "FLASHMTP left-shift target LM head kept eager; the "
                        "legacy graph sampler assumes an anchor-inclusive draft block."
                    )
                return None
            return super()._maybe_build_draft_sampler()
        if self.ps.tp_rank == 0:
            logger.info("FLASHMTP v2 Markov head folded into the draft cuda graph.")
        return _FlashMTPMarkovDraftSampler(
            draft_model=self.draft_model,
            draft_block_len=self._draft_block_len,
            num_proposals=self.num_proposed_tokens,
            max_bs=max(self.server_args.cuda_graph_config.decode.bs),
        )

    def init_cuda_graphs(self):
        capture_decode_cuda_graph = (
            self.server_args.cuda_graph_config.decode.backend != Backend.DISABLED
        )
        if is_cuda() and capture_decode_cuda_graph:
            available_mem = get_available_gpu_memory(self.device, self.gpu_id)
            if available_mem < 1.0:
                capture_decode_cuda_graph = False
                logger.warning(
                    "Disable FLASHMTP draft cuda graph because only %.2f GB GPU "
                    "memory is available after target backend initialization.",
                    available_mem,
                )
        if capture_decode_cuda_graph:
            # Fold the borrowed target LM head and greedy argmax into the same
            # graph as the FlashMTP draft transformer.
            self._draft_sampler = self._maybe_build_draft_sampler()
            if self._draft_sampler is not None:
                self.draft_model_runner.capture_tail_hooks.append(
                    make_draft_sampler_capture_hook(self._draft_sampler)
                )
        self._draft_worker.init_cuda_graphs(
            capture_decode_cuda_graph=capture_decode_cuda_graph
        )

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        # FlashMTP uses plain SDPA and never reads or writes a draft KV cache.
        # ModelRunner still expects its pool wiring to be initialized, so give
        # the draft runner a tiny inert pool while sharing the target request
        # mapping and allocator. In particular, do not allocate a full-size
        # five-layer draft KV pool.
        if memory_pool_config is None:
            raise ValueError(
                "FLASHMTP requires the resolved target memory pool config."
            )
        draft_pool_config = MemoryPoolConfig(
            max_total_num_tokens=64,
            max_running_requests=memory_pool_config.max_running_requests,
        )
        self._draft_worker.alloc_memory_pool(
            memory_pool_config=draft_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )

    def _make_next_input(
        self,
        *,
        bonus_tokens: torch.Tensor,
        new_seq_lens: torch.Tensor,
        pivot_hidden: torch.Tensor,
    ) -> DFlashDraftInputV2:
        return make_draft_input_v2(
            bonus_tokens=bonus_tokens,
            new_seq_lens=new_seq_lens,
            hidden_states=pivot_hidden,
        )

    def _ensure_draft_block_buffers(self, bs: int) -> None:
        """Grow the inherited block buffers and cache local draft positions."""
        super()._ensure_draft_block_buffers(bs)
        assert self._draft_block_ids_buf is not None
        cap = int(self._draft_block_ids_buf.shape[0])
        if (
            self._draft_local_positions_buf is None
            or int(self._draft_local_positions_buf.shape[0]) < cap
        ):
            self._draft_local_positions_buf = self._draft_local_positions.expand(
                cap, -1
            ).contiguous()

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
            batch_output = self.target_worker.forward_batch_generation(
                batch, capture_hidden_mode=CaptureHiddenMode.LAST
            )
            logits_output = batch_output.logits_output
            next_token_ids = batch_output.next_token_ids
            if logits_output.hidden_states is None:
                raise RuntimeError(
                    "FLASHMTP prefill requires target auxiliary hidden states."
                )

            pivot_hidden = logits_output.hidden_states
            expected_width = self._num_pivot_layers * int(
                self.draft_model_runner.model_config.hidden_size
            )
            if pivot_hidden.ndim != 2 or pivot_hidden.shape[-1] != expected_width:
                raise RuntimeError(
                    "FLASHMTP prefill pivot shape mismatch: "
                    f"expected [bs, {expected_width}], "
                    f"got {tuple(pivot_hidden.shape)}."
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

        if batch.spec_info is None:
            batch.spec_info = DFlashDraftInputV2.create_idle_input(device=self.device)
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInputV2):
            raise TypeError(
                "FLASHMTP expected block-draft V2 state on the running batch."
            )

        if batch.forward_mode.is_idle():
            empty_ids = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_lens = torch.empty((0,), dtype=torch.int32, device=self.device)
            next_input = self._make_next_input(
                bonus_tokens=empty_ids,
                new_seq_lens=empty_ids,
                pivot_hidden=torch.empty(
                    (0, 0), dtype=torch.float16, device=self.device
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

        batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )
        batch_size = len(batch.seq_lens)
        prefix_lens = batch.seq_lens
        self._ensure_draft_block_buffers(batch_size)
        assert self._draft_block_ids_buf is not None
        assert self._draft_block_positions_buf is not None
        assert self._draft_block_tokens_buf is not None
        assert self._draft_verify_out_cache_loc_buf is not None
        assert self._draft_seq_lens_cpu_buf is not None
        assert self._draft_local_positions_buf is not None

        block_ids = self._draft_block_ids_buf[:batch_size]
        target_positions_2d = self._draft_block_positions_buf[:batch_size]
        verify_out_cache_loc_2d = self._draft_verify_out_cache_loc_buf[:batch_size]
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
            except Exception as exc:  # noqa: BLE001
                self._use_triton_prepare_block = False
                logger.warning(
                    "FLASHMTP Triton prepare_block failed; falling back to eager "
                    "path: %s",
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
        else:
            self._prepare_draft_block_eager(
                draft_input=draft_input,
                prefix_lens=prefix_lens,
                req_pool_indices=batch.req_pool_indices,
                block_ids=block_ids,
                target_positions_2d=target_positions_2d,
                verify_out_cache_loc_2d=verify_out_cache_loc_2d,
            )
        verify_out_cache_loc = verify_out_cache_loc_2d.reshape(-1)

        target_model = self.target_worker.model_runner.model
        embed_module = target_model.get_input_embeddings()
        lm_head = getattr(target_model, "lm_head", None)
        if getattr(self.draft_model, "markov_head", None) is None and (
            lm_head is None or not hasattr(lm_head, "weight")
        ):
            raise RuntimeError(
                "FLASHMTP requires the target model's vocabulary-parallel lm_head."
            )
        draft_block_ids = block_ids[:, : self._draft_block_len]
        input_embeds = embed_module(draft_block_ids).reshape(
            batch_size * self._draft_block_len, -1
        )
        local_positions = self._draft_local_positions_buf[
            :batch_size, : self._draft_block_len
        ].reshape(-1)

        # DFlashVerifyInput is the existing generic fixed-width linear-chain
        # schema used by attention/runner plumbing. Its hidden_states attribute
        # is copied into the FlashMTP graph's stable pivot buffer on replay.
        draft_spec_info = self._draft_block_spec_info
        draft_spec_info.hidden_states = draft_input.hidden_states
        draft_seq_lens_cpu = self._draft_seq_lens_cpu_buf[:batch_size]
        if batch.seq_lens_cpu is not None:
            draft_seq_lens_cpu.copy_(batch.seq_lens_cpu)
            draft_seq_lens_sum = int(draft_seq_lens_cpu.sum())
        else:
            draft_seq_lens_cpu.copy_(prefix_lens.to("cpu", dtype=torch.int32))
            draft_seq_lens_sum = int(prefix_lens.sum().item())

        draft_forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=batch_size,
            input_ids=draft_block_ids.reshape(-1),
            req_pool_indices=batch.req_pool_indices,
            seq_lens=prefix_lens,
            out_cache_loc=verify_out_cache_loc_2d[:, : self._draft_block_len].reshape(
                -1
            ),
            seq_lens_sum=draft_seq_lens_sum,
            seq_lens_cpu=draft_seq_lens_cpu,
            positions=local_positions,
            input_embeds=input_embeds,
            spec_algorithm=SpeculativeAlgorithm.FLASHMTP,
            device_timer_category="flashmtp_draft",
            spec_info=draft_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        with torch.inference_mode():
            draft_out = self.draft_model_runner.forward(draft_forward_batch)
        if self._draft_sampler is not None and draft_out.can_run_graph:
            draft_next = self._draft_sampler.out[
                : batch_size * self.num_proposed_tokens
            ].view(batch_size, self.num_proposed_tokens)
        else:
            draft_hidden = draft_out.logits_output.hidden_states
            if draft_hidden is None:
                raise RuntimeError("FLASHMTP draft model returned no hidden states.")
            draft_hidden = draft_hidden.view(batch_size, self._draft_block_len, -1)
            if getattr(self.draft_model, "markov_head", None) is not None:
                draft_next = self.draft_model.sample_draft_tokens(
                    block_hidden=draft_hidden,
                    first_prev_token_ids=draft_block_ids[:, 0],
                )
            else:
                prediction_hidden = self.draft_model.prediction_hidden(draft_hidden)
                draft_next = self._greedy_sample_from_vocab_parallel_head(
                    hidden_states=prediction_hidden.reshape(
                        batch_size * self.num_proposed_tokens, -1
                    ),
                    lm_head=lm_head,
                ).view(batch_size, self.num_proposed_tokens)

        candidates = self._draft_block_tokens_buf[:batch_size]
        candidates[:, 0].copy_(block_ids[:, 0])
        candidates[:, 1:].copy_(draft_next)
        return self._verify_candidates(
            batch=batch,
            draft_input=draft_input,
            candidates=candidates,
            prefix_lens=prefix_lens,
            target_positions_2d=target_positions_2d,
            verify_out_cache_loc=verify_out_cache_loc,
            on_publish=on_publish,
            grammar_barrier=grammar_barrier,
        )

    def _verify_candidates(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        candidates: torch.Tensor,
        prefix_lens: torch.Tensor,
        target_positions_2d: torch.Tensor,
        verify_out_cache_loc: torch.Tensor,
        on_publish=None,
        grammar_barrier=None,
    ) -> GenerationBatchResult:
        """Verify a caller-provided FlashMTP candidate block on the target.

        Keeping this boundary independent of the local draft forward lets the
        disaggregated Target worker reuse the exact same acceptance, KV commit,
        grammar, and next-pivot path.
        """
        profile = bool(os.environ.get("DT_PROFILE_DIR")) and getattr(
            self, "_is_token_only", False
        )
        started_ns = time.monotonic_ns()
        batch_size = int(candidates.shape[0])
        if tuple(candidates.shape) != (batch_size, self.block_size):
            raise ValueError(
                "FLASHMTP candidate shape mismatch: "
                f"expected [bs, {self.block_size}], got {tuple(candidates.shape)}"
            )
        grammar_tree = (
            GrammarTree.from_linear_chain(candidates) if batch.has_grammar else None
        )

        target_positions = target_positions_2d.reshape(-1)
        verify_input = DFlashVerifyInput(
            draft_token=candidates.reshape(-1),
            positions=target_positions,
            draft_token_num=self.block_size,
            custom_mask=None,
            capture_hidden_mode=(
                CaptureHiddenMode.FULL
                if self._num_pivot_layers
                else CaptureHiddenMode.NULL
            ),
        )
        batch.out_cache_loc = verify_out_cache_loc
        sampling_info = batch.sampling_info
        seq_lens_cpu_backup = batch.seq_lens_cpu
        seq_lens_sum_backup = batch.seq_lens_sum
        if seq_lens_cpu_backup is not None:
            verify_host_seq_lens = seq_lens_cpu_backup + self.block_size
            batch.seq_lens_cpu = verify_host_seq_lens
            batch.seq_lens_sum = int(verify_host_seq_lens.sum())
        elif draft_input.reserved_seq_lens_cpu is not None:
            batch.seq_lens_cpu = draft_input.reserved_seq_lens_cpu
            batch.seq_lens_sum = int(draft_input.reserved_seq_lens_sum)

        verify_forward_batch, _ = verify_input.prepare_for_verify(
            batch, self.target_worker
        )
        prepared_ns = time.monotonic_ns()
        batch.seq_lens_cpu = seq_lens_cpu_backup
        batch.seq_lens_sum = seq_lens_sum_backup
        target_out = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        if profile:
            torch.get_device_module(self.device).synchronize()
        target_done_ns = time.monotonic_ns()
        logits_output = target_out.logits_output

        grammar_mask = None
        if batch.has_grammar:
            grammar_mask = build_grammar_vocab_mask(
                reqs=batch.reqs,
                tree=grammar_tree,
                sampling_info=sampling_info,
                device=logits_output.next_token_logits.device,
                barrier=grammar_barrier,
            )
        if sampling_info is not None:
            apply_dflash_verify_logits_adjustments(
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                draft_token_num=self.block_size,
            )
        if grammar_mask is not None:
            grammar_mask.apply(logits_output.next_token_logits)

        new_seq_lens = None
        if (
            sampling_info is not None
            and not sampling_info.is_all_greedy
            and is_dflash_sampling_verify_available()
        ):
            accept_len, bonus = compute_dflash_sampling_correct_drafts_and_bonus(
                candidates=candidates,
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                max_top_k=draft_input.max_top_k,
                uniform_top_k_value=draft_input.uniform_top_k_value,
            )
            commit_lens = accept_len.to(torch.int32) + 1
            out_tokens = torch.empty(
                (batch_size, self.block_size),
                dtype=torch.int64,
                device=self.device,
            )
            out_tokens[:, : self.num_proposed_tokens].copy_(candidates[:, 1:])
            out_tokens[:, self.num_proposed_tokens].fill_(0)
            out_tokens.scatter_(1, accept_len.to(torch.int64)[:, None], bonus[:, None])
        else:
            target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).view(
                batch_size, self.block_size
            )
            if self._use_triton_accept_bonus:
                try:
                    (
                        accept_len,
                        commit_lens,
                        bonus,
                        out_tokens,
                        new_seq_lens,
                    ) = self._next_accept_bonus_buffers(batch_size)
                    _compute_dflash_accept_bonus_triton_unchecked(
                        candidates=candidates,
                        target_top1=target_predict,
                        accept_lens_out=accept_len,
                        commit_lens_out=commit_lens,
                        bonus_ids_out=bonus,
                        out_tokens_out=out_tokens,
                        prefix_lens=prefix_lens,
                        new_seq_lens_out=new_seq_lens,
                    )
                except Exception as exc:  # noqa: BLE001
                    self._use_triton_accept_bonus = False
                    logger.warning(
                        "FLASHMTP Triton accept/bonus failed; falling back to "
                        "eager path: %s",
                        exc,
                    )
                    accept_len, bonus = compute_dflash_correct_drafts_and_bonus(
                        candidates=candidates,
                        target_predict=target_predict,
                    )
                    commit_lens = accept_len.to(torch.int32) + 1
                    out_tokens = torch.empty(
                        (batch_size, self.block_size),
                        dtype=torch.int64,
                        device=self.device,
                    )
                    out_tokens[:, : self.num_proposed_tokens].copy_(candidates[:, 1:])
                    out_tokens[:, self.num_proposed_tokens].fill_(0)
                    out_tokens.scatter_(
                        1, accept_len.to(torch.int64)[:, None], bonus[:, None]
                    )
            else:
                accept_len, bonus = compute_dflash_correct_drafts_and_bonus(
                    candidates=candidates,
                    target_predict=target_predict,
                )
                commit_lens = accept_len.to(torch.int32) + 1
                out_tokens = torch.empty(
                    (batch_size, self.block_size),
                    dtype=torch.int64,
                    device=self.device,
                )
                out_tokens[:, : self.num_proposed_tokens].copy_(candidates[:, 1:])
                out_tokens[:, self.num_proposed_tokens].fill_(0)
                out_tokens.scatter_(
                    1, accept_len.to(torch.int64)[:, None], bonus[:, None]
                )

        if profile:
            torch.get_device_module(self.device).synchronize()
        accepted_ns = time.monotonic_ns()

        if new_seq_lens is None:
            new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
        if on_publish is not None:
            on_publish(new_seq_lens)

        if self._num_pivot_layers:
            packed_hidden = logits_output.hidden_states
            if packed_hidden is None:
                raise RuntimeError(
                    "FLASHMTP verify requires captured target auxiliary hidden states."
                )
            next_pivot = gather_flashmtp_next_pivot(
                packed_hidden=packed_hidden,
                accept_len=accept_len,
                block_size=self.block_size,
            )
            logits_output.hidden_states = None
        else:
            # Token-only disaggregation has no Target feature payload.  Keep the
            # linear-chain verifier and its authoritative accepted-token result,
            # but do not manufacture hidden state for the next Draft round.
            next_pivot = torch.empty(
                (batch_size, 0), dtype=torch.bfloat16, device=self.device
            )

        next_input = self._make_next_input(
            bonus_tokens=bonus,
            new_seq_lens=new_seq_lens,
            pivot_hidden=next_pivot,
        )
        if profile:
            from dt_profile import emit as profile_emit

            profile_emit(
                "target_verify_breakdown",
                batch_size=batch_size,
                prepare_ns=prepared_ns - started_ns,
                target_forward_ns=target_done_ns - prepared_ns,
                accept_ns=accepted_ns - target_done_ns,
                finish_ns=time.monotonic_ns() - accepted_ns,
            )
        batch_result = GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=commit_lens,
            can_run_cuda_graph=target_out.can_run_cuda_graph,
            next_draft_input=next_input,
            speculative_num_draft_tokens=self.block_size,
            new_seq_lens=new_seq_lens,
        )
        batch_result.dt_cuda_graph_measurement_event_id = (
            self.target_worker.model_runner._last_cuda_graph_measurement_event_id
        )
        return batch_result

    def _prepare_draft_block_eager(
        self,
        *,
        draft_input: DFlashDraftInputV2,
        prefix_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        block_ids: torch.Tensor,
        target_positions_2d: torch.Tensor,
        verify_out_cache_loc_2d: torch.Tensor,
    ) -> None:
        block_ids.fill_(self._mask_token_id)
        block_ids[:, 0].copy_(draft_input.bonus_tokens.view(-1))
        torch.add(
            prefix_lens.unsqueeze(1),
            self._block_pos_offsets,
            out=target_positions_2d,
        )
        batch_size = int(prefix_lens.numel())
        verify_out_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=req_pool_indices,
            req_to_token=self.model_runner.req_to_token_pool.req_to_token,
            start_offset=prefix_lens,
            end_offset=prefix_lens + self.block_size,
            batch_size=batch_size,
            draft_token_num=self.block_size,
            device=self.device,
        )
        verify_out_cache_loc_2d.copy_(
            verify_out_cache_loc.view(batch_size, self.block_size)
        )
