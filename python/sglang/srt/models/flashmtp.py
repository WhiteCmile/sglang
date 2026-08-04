# SPDX-License-Identifier: Apache-2.0

"""Inference-only Qwen3 FlashMTP draft model."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dflash_utils import can_dflash_slice_qkv_weight
from sglang.srt.speculative.flashmtp_utils import parse_flashmtp_draft_config
from sglang.srt.utils.hf_transformers_utils import get_rope_config
from torch import nn


class FlashMTPAttention(nn.Module):
    """Non-causal attention from draft Q to ``[pivot context, draft]`` K/V."""

    def __init__(
        self,
        config,
        layer_id: int,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        flash_config = parse_flashmtp_draft_config(draft_hf_config=config)
        hidden_size = int(config.hidden_size)
        total_num_heads = int(config.num_attention_heads)
        total_num_kv_heads = int(config.num_key_value_heads)
        head_dim = int(getattr(config, "head_dim", hidden_size // total_num_heads))
        tp_size = int(get_parallel().tp_size)

        if total_num_heads % tp_size != 0:
            raise ValueError(
                "FlashMTP num_attention_heads must be divisible by TP size: "
                f"{total_num_heads=} {tp_size=}."
            )
        if total_num_kv_heads >= tp_size:
            if total_num_kv_heads % tp_size != 0:
                raise ValueError(
                    "FlashMTP num_key_value_heads must be divisible by TP size "
                    "when num_key_value_heads >= TP size."
                )
        elif tp_size % total_num_kv_heads != 0:
            raise ValueError(
                "FlashMTP TP size must be divisible by num_key_value_heads "
                "when TP size is larger."
            )

        self.num_heads = total_num_heads // tp_size
        self.num_kv_heads = max(1, total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * head_dim
        self.kv_size = self.num_kv_heads * head_dim
        self.scaling = head_dim**-0.5
        self.pivot_fuse_mode = flash_config.pivot_fuse_mode

        attention_bias = bool(getattr(config, "attention_bias", False))
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=head_dim,
            total_num_heads=total_num_heads,
            total_num_kv_heads=total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            total_num_heads * head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        eps = float(config.rms_norm_eps)
        self.q_norm = RMSNorm(head_dim, eps=eps)
        self.k_norm = RMSNorm(head_dim, eps=eps)

        rope_theta, rope_scaling = get_rope_config(config)
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=int(config.max_position_embeddings),
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=bool(getattr(config, "rope_is_neox_style", True)),
        )

        layer_types = getattr(config, "layer_types", None)
        self.sliding_window = (
            getattr(config, "sliding_window", None)
            if layer_types and layer_types[layer_id] == "sliding_attention"
            else None
        )

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        qkv_draft, _ = self.qkv_proj(hidden_states)
        q, k_draft, v_draft = qkv_draft.split(
            [self.q_size, self.kv_size, self.kv_size], dim=-1
        )

        can_slice_weight, _ = can_dflash_slice_qkv_weight(self.qkv_proj)
        if can_slice_weight:
            # Context queries are never consumed. For ordinary BF16/FP16
            # weights, reuse the K/V rows of the fused QKV parameter rather
            # than launching a full QKV GEMM or keeping a duplicate KV module.
            kv_slice = slice(self.q_size, self.q_size + 2 * self.kv_size)
            weight = self.qkv_proj.weight[kv_slice]
            bias = (
                self.qkv_proj.bias[kv_slice] if self.qkv_proj.bias is not None else None
            )
            kv_context = F.linear(target_hidden, weight, bias)
            k_context, v_context = kv_context.split(
                [self.kv_size, self.kv_size], dim=-1
            )
        else:
            # Packed/quantized weights cannot be sliced safely. Preserve
            # compatibility by using the configured linear method and
            # discarding the context Q output.
            qkv_context, _ = self.qkv_proj(target_hidden)
            _, k_context, v_context = qkv_context.split(
                [self.q_size, self.kv_size, self.kv_size], dim=-1
            )

        return q, k_draft, v_draft, k_context, v_context

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, query_len, _ = hidden_states.shape
        context_len = int(target_hidden.shape[1])
        kv_len = context_len + query_len

        q, k_draft, v_draft, k_context, v_context = self._project_qkv(
            hidden_states, target_hidden
        )

        q = self.q_norm(q.reshape(-1, self.head_dim)).view(batch_size, query_len, -1)
        flat_positions = positions.reshape(-1)
        q, k_draft = self.rotary_emb(
            flat_positions,
            q.reshape(batch_size * query_len, -1),
            k_draft.reshape(batch_size * query_len, -1),
        )
        q = q.view(batch_size, query_len, -1)
        k_draft = k_draft.view(batch_size, query_len, -1)

        # prefix_condition: context keys intentionally receive no RoPE, and
        # K norm is applied after concatenating context and draft keys.
        k = torch.cat([k_context, k_draft], dim=1)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view(batch_size, kv_len, -1)
        v = torch.cat([v_context, v_draft], dim=1)

        q = q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        attention_mask = None
        if self.sliding_window is not None:
            query_positions = positions
            context_positions = torch.zeros(
                (batch_size, context_len),
                dtype=positions.dtype,
                device=positions.device,
            )
            key_positions = torch.cat([context_positions, positions], dim=1)
            distance = (query_positions.unsqueeze(2) - key_positions.unsqueeze(1)).abs()
            attention_mask = (distance <= int(self.sliding_window)).unsqueeze(1)

        attention_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=False,
            scale=self.scaling,
            enable_gqa=self.num_heads != self.num_kv_heads,
        )
        attention_output = attention_output.transpose(1, 2).reshape(
            batch_size, query_len, -1
        )
        attention_output, _ = self.o_proj(attention_output)
        return attention_output


class FlashMTPMLP(nn.Module):
    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        intermediate_size = int(config.intermediate_size)
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class FlashMTPDecoderLayer(nn.Module):
    def __init__(
        self, config, layer_id: int, quant_config=None, prefix: str = ""
    ) -> None:
        super().__init__()
        eps = float(config.rms_norm_eps)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.self_attn = FlashMTPAttention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.mlp = FlashMTPMLP(
            config=config, quant_config=quant_config, prefix=f"{prefix}.mlp"
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        original_shape = hidden_states.shape
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states.reshape(-1, original_shape[-1]),
                residual.reshape(-1, original_shape[-1]),
            )
            hidden_states = hidden_states.view(original_shape)
            residual = residual.view(original_shape)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            target_hidden=target_hidden,
        )
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states.reshape(-1, original_shape[-1]),
            residual.reshape(-1, original_shape[-1]),
        )
        hidden_states = hidden_states.view(original_shape)
        residual = residual.view(original_shape)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class FlashMTPDraftModel(nn.Module):
    """FlashMTP draft model without embedding, LM head, or KV cache."""

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.flash_config = parse_flashmtp_draft_config(draft_hf_config=config)
        hidden_size = int(config.hidden_size)
        eps = float(config.rms_norm_eps)

        self.layer_depth_embedding = nn.Embedding(
            self.flash_config.num_target_layers, hidden_size
        )
        self.hidden_norm = RMSNorm(hidden_size, eps=eps)
        self.layers = nn.ModuleList(
            [
                FlashMTPDecoderLayer(
                    config=config,
                    layer_id=layer_id,
                    quant_config=quant_config,
                    prefix=f"layers.{layer_id}",
                )
                for layer_id in range(self.flash_config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=eps)
        self.block_size = self.flash_config.block_size
        self.target_layer_ids = self.flash_config.target_layer_ids
        # Keep the depth ids at a stable device address. Constructing this
        # tensor inside forward allocates during warmup/capture and makes the
        # FlashMTP draft forward unsuitable for CUDA Graph replay.
        self.register_buffer(
            "target_layer_ids_tensor",
            torch.tensor(self.target_layer_ids, dtype=torch.long),
            persistent=False,
        )

    def _fuse_target_hidden(self, target_hidden: torch.Tensor) -> torch.Tensor:
        batch_size = target_hidden.shape[0]
        hidden_size = int(self.config.hidden_size)
        target_hidden = target_hidden.view(
            batch_size, len(self.target_layer_ids), hidden_size
        )
        depth_embedding = self.layer_depth_embedding(self.target_layer_ids_tensor).view(
            1, len(self.target_layer_ids), hidden_size
        )
        return self.hidden_norm(target_hidden + depth_embedding)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
    ) -> LogitsProcessorOutput:
        del input_ids, get_embedding, pp_proxy_tensors
        if input_embeds is None:
            raise ValueError(
                "FlashMTPDraftModel requires input_embeds from the target model."
            )
        if forward_batch.spec_info is None or not hasattr(
            forward_batch.spec_info, "hidden_states"
        ):
            raise ValueError(
                "FlashMTP draft forward requires pivot hidden states in spec_info."
            )

        batch_size = int(forward_batch.batch_size)
        query_len = int(input_embeds.shape[0]) // batch_size
        hidden_states = input_embeds.view(batch_size, query_len, -1)
        positions = positions.view(batch_size, query_len)
        target_hidden = self._fuse_target_hidden(forward_batch.spec_info.hidden_states)

        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions, hidden_states, target_hidden, residual
            )
        assert residual is not None
        original_shape = hidden_states.shape
        hidden_states, _ = self.norm(
            hidden_states.reshape(-1, original_shape[-1]),
            residual.reshape(-1, original_shape[-1]),
        )
        hidden_states = hidden_states.view(original_shape)
        return LogitsProcessorOutput(
            next_token_logits=None,
            hidden_states=hidden_states.reshape(batch_size * query_len, -1),
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        def resolve_name(name: str) -> str | None:
            if name in params_dict:
                return name
            if name.startswith("model."):
                stripped = name[len("model.") :]
                if stripped in params_dict:
                    return stripped
            return None

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if f".{weight_name}." not in name:
                    continue
                resolved_name = resolve_name(name.replace(weight_name, param_name))
                if resolved_name is None:
                    break
                param = params_dict[resolved_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(resolved_name)
                break
            else:
                resolved_name = resolve_name(name)
                if resolved_name is None:
                    continue
                param = params_dict[resolved_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(resolved_name)

        return loaded_params


EntryClass = FlashMTPDraftModel
