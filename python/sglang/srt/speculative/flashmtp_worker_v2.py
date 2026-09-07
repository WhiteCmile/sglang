from __future__ import annotations

import logging
import time
from dataclasses import replace

import torch

try:
    from dt_profile import emit as profile_emit
except ImportError:  # Standalone SGLang use does not require dt-runtime.

    def profile_emit(*_args, **_kwargs) -> None:
        return None


from sglang.kernels.ops.speculative.cache_locs import assign_extend_cache_locs_func
from sglang.kernels.ops.speculative.dflash import (
    _compute_dflash_accept_bonus_triton_unchecked,
    _prepare_dflash_draft_block_unchecked,
)
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
    BuildRaggedVerifyWindow,
    ScatterCompactToStrided,
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
from sglang.srt.speculative.dspark_components.dspark_verify import (
    DsparkVerifyEpilogue,
    VocabParallelGreedySelector,
    verify_logits_adjustments_are_noop,
)
from sglang.srt.speculative.flashmtp_shared import (
    adaptive_verify_enabled,
    build_chard_continuation,
    graph_tier_padding_enabled,
    pinned_verify_metadata_enabled,
    plan_ragged_execution_tier,
    ragged_epilogue_capacity,
    ragged_verify_enabled,
    read_adaptive_thresholds,
    read_adaptive_min_batch,
    read_chard_chunk_size,
    reclaim_ragged_graph_padding,
    select_adaptive_verify_lens,
    shared_optimizations_enabled,
    validate_component_dependencies,
    vocab_parallel_greedy_enabled,
)
from sglang.srt.speculative.flashmtp_utils import (
    gather_flashmtp_next_pivot,
    parse_flashmtp_draft_config,
)
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyLayout,
    ragged_verify_compact_enabled,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    GrammarTree,
    build_grammar_vocab_mask,
)
from sglang.srt.utils import get_available_gpu_memory, is_cuda, is_hip
from sglang.srt.utils.common import is_pin_memory_available

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
        adaptive_thresholds: tuple[float, ...] | None = None,
    ):
        self.draft_model = draft_model
        self.draft_block_len = int(draft_block_len)
        self.num_proposals = int(num_proposals)
        device = next(draft_model.parameters()).device
        self.out = torch.empty(
            (int(max_bs) * self.num_proposals,), dtype=torch.int64, device=device
        )
        self.adaptive_thresholds = (
            torch.tensor(adaptive_thresholds, dtype=torch.float32, device=device)
            if adaptive_thresholds is not None
            else None
        )
        self.verify_lens_out = (
            torch.empty((int(max_bs),), dtype=torch.int32, device=device)
            if adaptive_thresholds is not None
            else None
        )

    def __call__(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> None:
        batch_size = hidden_states.shape[0] // self.draft_block_len
        block_hidden = hidden_states.view(batch_size, self.draft_block_len, -1)
        anchors = input_ids.view(batch_size, self.draft_block_len)[:, 0]
        if self.adaptive_thresholds is None:
            sampled = self.draft_model.sample_draft_tokens(
                block_hidden=block_hidden,
                first_prev_token_ids=anchors,
            )
        else:
            sampled, margins = self.draft_model.sample_draft_tokens_with_margin(
                block_hidden=block_hidden,
                first_prev_token_ids=anchors,
            )
            assert self.verify_lens_out is not None
            self.verify_lens_out[:batch_size].copy_(
                select_adaptive_verify_lens(
                    margins,
                    threshold=self.adaptive_thresholds,
                    block_size=self.num_proposals + 1,
                )
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
        self.model_block_size = int(server_args.speculative_num_draft_tokens)
        self.block_size = self.model_block_size
        self.speculative_num_draft_tokens = self.block_size
        self.num_proposed_tokens = self.block_size - 1
        self.model_num_proposed_tokens = int(flash_config.num_proposed_tokens)
        self._draft_block_len = int(flash_config.draft_block_len)
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

        # Keep the historical all-on bundle, but expose its legal components
        # for matched ablation.  DT transport/lifecycle flags remain separate.
        validate_component_dependencies()
        self._shared_optimizations = shared_optimizations_enabled()
        self._chard_chunk_size = read_chard_chunk_size(block_size=self.block_size)
        self._ragged_verify = (
            ragged_verify_enabled() or self._chard_chunk_size is not None
        )
        self._adaptive_verify = adaptive_verify_enabled()
        self._pinned_verify_metadata = pinned_verify_metadata_enabled()
        self._vocab_parallel_greedy = vocab_parallel_greedy_enabled()
        self._shared_adaptive_thresholds = read_adaptive_thresholds(
            proposal_count=self.model_num_proposed_tokens
        )
        self._shared_adaptive_min_batch = read_adaptive_min_batch()
        self._adaptive_fill_graph_padding = graph_tier_padding_enabled()
        self._adaptive_graph_tier_offset = 0
        self._single_load_verify_graph = False
        self._verify_lens_cpu_buf: torch.Tensor | None = None
        self._verify_lens_cpu_cap = 0
        if self._ragged_verify:
            if not ragged_verify_compact_enabled():
                raise ValueError(
                    "FlashMTP ragged verify requires SGLANG_RAGGED_VERIFY_MODE=compact"
                )
        greedy_selector = None
        if self._vocab_parallel_greedy:
            if server_args.disable_cuda_graph or not is_cuda():
                raise ValueError(
                    "FlashMTP vocab-parallel greedy currently requires "
                    "CUDA graph verify"
                )
            target_model = self.model_runner.model
            lm_head = getattr(target_model, "lm_head", None)
            if lm_head is None or not hasattr(lm_head, "weight"):
                raise ValueError(
                    "FlashMTP vocab-parallel greedy requires target lm_head"
                )
            if not torch.is_floating_point(lm_head.weight):
                raise ValueError(
                    "FlashMTP vocab-parallel greedy does not support a "
                    "quantized lm_head"
                )
            shard = getattr(lm_head, "shard_indices", None)
            if shard is None or int(shard.num_added_elements) != 0:
                raise ValueError(
                    "FlashMTP vocab-parallel greedy requires contiguous "
                    "base-vocabulary shard metadata"
                )
            tp_group = self.model_runner.tp_group
            greedy_selector = VocabParallelGreedySelector(
                max_tokens=ragged_epilogue_capacity(server_args) * self.block_size,
                num_org=int(shard.num_org_elements),
                org_vocab_start=int(shard.org_vocab_start_index),
                value_dtype=torch.float32,
                device=self.device,
                tp_group=tp_group if int(tp_group.world_size) > 1 else None,
            )
        if self._ragged_verify and not server_args.disable_cuda_graph and is_cuda():
            self._ragged_verify_epilogue = DsparkVerifyEpilogue(
                max_bs=ragged_epilogue_capacity(server_args),
                verify_num_draft_tokens=self.block_size,
                device=self.device,
                greedy_selector=greedy_selector,
            )
            self.model_runner.capture_tail_hooks.append(
                self._ragged_verify_epilogue.capture_hook
            )
        else:
            self._ragged_verify_epilogue = None

        if self.ps.tp_rank == 0:
            logger.info(
                "Initialized FLASHMTP draft runner. model=%s, block_size=%d, "
                "model_block_size=%d, num_proposed_tokens=%d, pivot_layers=%d, "
                "mask_token_id=%d, "
                "shared_optimizations=%s, ragged=%s, adaptive=%s, pinned=%s, "
                "vocab_greedy=%s, graph_padding=%s",
                self.draft_model.__class__.__name__,
                self.block_size,
                self.model_block_size,
                self.num_proposed_tokens,
                self._num_pivot_layers,
                self._mask_token_id,
                self._shared_optimizations,
                self._ragged_verify,
                self._adaptive_verify,
                self._pinned_verify_metadata,
                self._vocab_parallel_greedy,
                self._adaptive_fill_graph_padding,
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
            num_proposals=self.model_num_proposed_tokens,
            max_bs=max(self.server_args.cuda_graph_config.decode.bs),
            adaptive_thresholds=self._shared_adaptive_thresholds,
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

    def _build_chard_continuation(
        self,
        *,
        draft_input: DFlashDraftInputV2,
        candidates: torch.Tensor,
        verify_lens: torch.Tensor,
        commit_lens: torch.Tensor,
        bonus_tokens: torch.Tensor,
        next_input: DFlashDraftInputV2,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self._chard_chunk_size is None:
            return None, None
        queue_lens = draft_input.chard_queue_lens
        if queue_lens is None:
            queue_lens = torch.full(
                (candidates.shape[0],),
                self.block_size,
                dtype=torch.int32,
                device=candidates.device,
            )
        next_tokens, next_lens = build_chard_continuation(
            candidates=candidates,
            queue_lens=queue_lens,
            verify_lens=verify_lens,
            commit_lens=commit_lens,
            bonus_tokens=bonus_tokens,
        )
        next_input.chard_queue_tokens = next_tokens
        next_input.chard_queue_lens = next_lens
        return next_tokens, next_lens

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

    def _stage_verify_lens_cpu(
        self,
        verify_lens: torch.Tensor | None,
        *,
        batch_size: int,
        fixed_width: int | None = None,
    ) -> torch.Tensor:
        """Materialize ragged widths, optionally through pinned host storage.

        CUDA graph tier selection is a host decision, so the small D2H result
        must become visible before verify preparation.  ``verify_lens=None``
        represents the legal fixed-width ragged-preparation ablation.
        """
        if verify_lens is None and not self._pinned_verify_metadata:
            width = self.model_block_size if fixed_width is None else int(fixed_width)
            return torch.full(
                (batch_size,), width, dtype=torch.int32, device="cpu"
            )
        if verify_lens is not None and (
            self._adaptive_verify and batch_size < self._shared_adaptive_min_batch
        ):
            verify_lens = torch.full_like(
                verify_lens[:batch_size],
                self.model_block_size,
                dtype=torch.int32,
            )
        elif verify_lens is not None:
            verify_lens = verify_lens[:batch_size].to(torch.int32)
        if verify_lens is not None and not self._pinned_verify_metadata:
            return verify_lens.to(device="cpu", dtype=torch.int32)
        if self._verify_lens_cpu_cap < batch_size or self._verify_lens_cpu_buf is None:
            capacity = max(batch_size, 32, self._verify_lens_cpu_cap * 2)
            self._verify_lens_cpu_buf = torch.empty(
                (capacity,),
                dtype=torch.int32,
                device="cpu",
                pin_memory=is_pin_memory_available(self.device),
            )
            self._verify_lens_cpu_cap = capacity
        staged = self._verify_lens_cpu_buf[:batch_size]
        if verify_lens is None:
            staged.fill_(
                self.model_block_size if fixed_width is None else int(fixed_width)
            )
        else:
            staged.copy_(verify_lens, non_blocking=True)
            torch.get_device_module(self.device).current_stream().synchronize()
        return staged

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
        verify_lens_device = None
        if self._draft_sampler is not None and draft_out.can_run_graph:
            draft_next = self._draft_sampler.out[
                : batch_size * self.model_num_proposed_tokens
            ].view(batch_size, self.model_num_proposed_tokens)
            if self._adaptive_verify:
                verify_lens_device = self._draft_sampler.verify_lens_out
                if verify_lens_device is None:
                    raise RuntimeError(
                        "adaptive FlashMTP graph sampler did not publish verify lengths"
                    )
        else:
            draft_hidden = draft_out.logits_output.hidden_states
            if draft_hidden is None:
                raise RuntimeError("FLASHMTP draft model returned no hidden states.")
            draft_hidden = draft_hidden.view(batch_size, self._draft_block_len, -1)
            if getattr(self.draft_model, "markov_head", None) is not None:
                if self._adaptive_verify:
                    draft_next, margins = (
                        self.draft_model.sample_draft_tokens_with_margin(
                            block_hidden=draft_hidden,
                            first_prev_token_ids=draft_block_ids[:, 0],
                        )
                    )
                    assert self._shared_adaptive_thresholds is not None
                    verify_lens_device = select_adaptive_verify_lens(
                        margins,
                        threshold=self._shared_adaptive_thresholds,
                        block_size=self.model_block_size,
                    )
                else:
                    draft_next = self.draft_model.sample_draft_tokens(
                        block_hidden=draft_hidden,
                        first_prev_token_ids=draft_block_ids[:, 0],
                    )
            else:
                prediction_hidden = self.draft_model.prediction_hidden(draft_hidden)
                draft_next = self._greedy_sample_from_vocab_parallel_head(
                    hidden_states=prediction_hidden.reshape(
                        batch_size * self.model_num_proposed_tokens, -1
                    ),
                    lm_head=lm_head,
                ).view(batch_size, self.model_num_proposed_tokens)

        candidates = self._draft_block_tokens_buf[:batch_size]
        candidates.zero_()
        candidates[:, 0].copy_(block_ids[:, 0])
        candidates[:, 1 : 1 + self.model_num_proposed_tokens].copy_(draft_next)
        if self._chard_chunk_size is not None:
            previous_tokens = draft_input.chard_queue_tokens
            previous_lens = draft_input.chard_queue_lens
            if (previous_tokens is None) != (previous_lens is None):
                raise RuntimeError("CHARD queue tokens and lengths must travel together")
            if previous_tokens is None:
                current_lens = torch.full(
                    (batch_size,),
                    self.block_size,
                    dtype=torch.int32,
                    device=self.device,
                )
            else:
                if tuple(previous_tokens.shape) != (batch_size, self.block_size):
                    raise ValueError("CHARD queue shape does not match the running batch")
                active = previous_lens.view(-1) > 0
                candidates.copy_(
                    torch.where(active[:, None], previous_tokens, candidates)
                )
                current_lens = torch.where(
                    active,
                    previous_lens.to(dtype=torch.int32),
                    torch.full_like(previous_lens, self.block_size, dtype=torch.int32),
                )
            draft_input.chard_queue_lens = current_lens
        if self._ragged_verify:
            sampling_info = batch.sampling_info
            if not (
                not batch.has_grammar
                and sampling_info is not None
                and sampling_info.is_all_greedy
                and verify_logits_adjustments_are_noop(sampling_info)
            ):
                raise RuntimeError(
                    "FlashMTP ragged verify requires adjustment-free "
                    "greedy decoding without grammar constraints"
                )
            if self._adaptive_verify and verify_lens_device is None:
                raise RuntimeError(
                    "adaptive FlashMTP verify requires Markov confidence"
                )
            if self._chard_chunk_size is not None:
                if verify_lens_device is not None:
                    verify_lens_device = torch.minimum(
                        verify_lens_device,
                        torch.full_like(verify_lens_device, self._chard_chunk_size),
                    )
                draft_input.verify_lens_cpu = self._stage_verify_lens_cpu(
                    verify_lens_device,
                    batch_size=batch_size,
                    fixed_width=self._chard_chunk_size,
                )
            else:
                draft_input.verify_lens_cpu = self._stage_verify_lens_cpu(
                    verify_lens_device, batch_size=batch_size
                )
            return self._verify_candidates_ragged(
                batch=batch,
                draft_input=draft_input,
                candidates=candidates,
                prefix_lens=prefix_lens,
                on_publish=on_publish,
            )
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

    def _verify_candidates_ragged(
        self,
        *,
        batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        candidates: torch.Tensor,
        prefix_lens: torch.Tensor,
        on_publish=None,
    ) -> GenerationBatchResult:
        """Run the topology-neutral compact/adaptive Target verify path."""
        prepare_start_ns = time.monotonic_ns()
        batch_size = int(candidates.shape[0])
        verify_lens_cpu_tensor = draft_input.verify_lens_cpu
        if verify_lens_cpu_tensor is None:
            raise RuntimeError("shared adaptive verify is missing verify lengths")
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
            and self._chard_chunk_size is None
            and total_verify_tokens < graph_num_tokens
        ):
            verify_lens_cpu = reclaim_ragged_graph_padding(
                verify_lens_cpu,
                block_size=self.block_size,
                graph_num_tokens=graph_num_tokens,
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
                instance_id=f"baseline-{getattr(self.server_args, 'port', 0)}",
                topology="baseline",
                batch_size=batch_size,
                total_verify_tokens=total_verify_tokens,
                graph_num_tokens=layout.graph_num_tokens,
                verify_lens=verify_lens_cpu,
                reclaimed_padding_tokens=(
                    total_verify_tokens - original_total_verify_tokens
                ),
                graph_tier_offset=self._adaptive_graph_tier_offset,
                graph_tier_available=graph_tier_available,
            )

        ragged_window = BuildRaggedVerifyWindow.execute(
            batch=batch,
            layout=layout,
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
            skip_attn_backend_init=(
                not self._single_load_verify_graph or not can_run_cuda_graph
            ),
        )
        forward_submit_done_ns = time.monotonic_ns()
        if self.ps.tp_rank == 0:
            profile_emit(
                "target_ragged_verify_forward",
                instance_id=f"baseline-{getattr(self.server_args, 'port', 0)}",
                topology="baseline",
                batch_size=batch_size,
                total_verify_tokens=total_verify_tokens,
                graph_num_tokens=layout.graph_num_tokens,
                can_run_cuda_graph=target_out.can_run_cuda_graph,
                validate_ns=validate_done_ns - prepare_start_ns,
                tier_plan_ns=tier_plan_done_ns - validate_done_ns,
                layout_ns=layout_done_ns - tier_plan_done_ns,
                window_ns=window_done_ns - layout_done_ns,
                epilogue_prepare_ns=epilogue_prepare_done_ns - window_done_ns,
                forward_prepare_ns=forward_prepare_done_ns - epilogue_prepare_done_ns,
                forward_submit_ns=forward_submit_done_ns - forward_prepare_done_ns,
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
            commit_lens = accept_outs.commit_lens.clone()
            accept_len = commit_lens.to(torch.int64) - 1
            bonus = accept_outs.bonus.clone()
            out_tokens = accept_outs.out_tokens.clone()
            new_seq_lens = accept_outs.new_seq_lens.clone()
            if epilogue.uses_vocab_parallel_greedy:
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
        chard_tokens, chard_lens = self._build_chard_continuation(
            draft_input=draft_input,
            candidates=candidates,
            verify_lens=layout.verify_lens,
            commit_lens=commit_lens,
            bonus_tokens=bonus,
            next_input=next_input,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=commit_lens,
            can_run_cuda_graph=target_out.can_run_cuda_graph,
            next_draft_input=next_input,
            speculative_num_draft_tokens=self.block_size,
            new_seq_lens=new_seq_lens,
            chard_continuation_tokens=chard_tokens,
            chard_continuation_lens=chard_lens,
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
            capture_hidden_mode=CaptureHiddenMode.FULL,
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
        batch.seq_lens_cpu = seq_lens_cpu_backup
        batch.seq_lens_sum = seq_lens_sum_backup
        target_out = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
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

        if new_seq_lens is None:
            new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
        if on_publish is not None:
            on_publish(new_seq_lens)

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
