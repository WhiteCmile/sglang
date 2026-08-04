from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.models.flashmtp import FlashMTPAttention, FlashMTPDecoderLayer
from sglang.srt.models.qwen2 import Qwen2Model
from sglang.srt.speculative.flashmtp_utils import (
    gather_flashmtp_next_pivot,
    parse_flashmtp_draft_config,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from torch import nn


def _config(**flash_overrides):
    flash_config = {
        "local_position": True,
        "mask_token_id": 99,
        "pivot_fuse_mode": "prefix_condition",
        "target_layer_ids": [0, 2, 3],
    }
    flash_config.update(flash_overrides)
    return SimpleNamespace(
        block_size=16,
        flashmtp_config=flash_config,
        num_hidden_layers=5,
        num_target_layers=4,
    )


def test_flashmtp_block_size_includes_bonus_token():
    config = parse_flashmtp_draft_config(draft_hf_config=_config())
    assert config.block_size == 16
    assert config.num_proposed_tokens == 15


def test_flashmtp_rejects_unsupported_modes():
    with pytest.raises(NotImplementedError, match="prefix_condition"):
        parse_flashmtp_draft_config(
            draft_hf_config=_config(pivot_fuse_mode="linear_fuse")
        )
    with pytest.raises(NotImplementedError, match="local_position=true"):
        parse_flashmtp_draft_config(draft_hf_config=_config(local_position=False))


def test_flashmtp_rejects_ambiguous_target_layer_order():
    with pytest.raises(ValueError, match="strictly increasing"):
        parse_flashmtp_draft_config(draft_hf_config=_config(target_layer_ids=[0, 3, 2]))
    with pytest.raises(ValueError, match="strictly increasing"):
        parse_flashmtp_draft_config(draft_hf_config=_config(target_layer_ids=[0, 2, 2]))


def test_flashmtp_gathers_pivot_at_accept_len():
    block_size = 4
    hidden = torch.arange(2 * block_size * 3).view(2 * block_size, 3)
    accept_len = torch.tensor([0, 2], dtype=torch.int32)
    gathered = gather_flashmtp_next_pivot(
        packed_hidden=hidden,
        accept_len=accept_len,
        block_size=block_size,
    )
    expected = torch.stack([hidden[0], hidden[block_size + 2]])
    torch.testing.assert_close(gathered, expected)


def test_flashmtp_fused_residual_path_matches_reference():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    class _Attention(nn.Module):
        def forward(self, positions, hidden_states, target_hidden):
            del positions, target_hidden
            return hidden_states * 0.25

    class _MLP(nn.Module):
        def forward(self, hidden_states):
            return hidden_states * 0.5

    layer = FlashMTPDecoderLayer.__new__(FlashMTPDecoderLayer)
    nn.Module.__init__(layer)
    layer.input_layernorm = RMSNorm(4, eps=1e-6)
    layer.self_attn = _Attention()
    layer.post_attention_layernorm = RMSNorm(4, eps=1e-6)
    layer.mlp = _MLP()
    layer.to(device=device, dtype=dtype)

    def reference(hidden_states):
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        hidden_states = layer.self_attn(None, hidden_states, None)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        return residual + layer.mlp(hidden_states)

    inputs = torch.tensor(
        [
            [[0.5, -1.0, 2.0, 0.25], [1.5, 0.75, -0.5, 2.5]],
            [[-0.25, 1.25, 0.5, -2.0], [2.0, -1.5, 0.75, 1.0]],
        ],
        dtype=dtype,
        device=device,
    )
    expected = reference(reference(inputs.clone()))

    hidden_states, residual = layer(None, inputs.clone(), None, None)
    hidden_states, residual = layer(None, hidden_states, None, residual)
    actual = hidden_states + residual

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("slice_weight", [True, False])
def test_flashmtp_fused_qkv_projection_matches_reference(slice_weight):
    class _QKVProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(
                torch.arange(8 * 5, dtype=torch.float32).view(8, 5) / 17
            )
            self.bias = nn.Parameter(torch.arange(8, dtype=torch.float32) / 13)
            self.quant_method = UnquantizedLinearMethod() if slice_weight else object()

        def forward(self, hidden_states):
            return torch.nn.functional.linear(
                hidden_states, self.weight, self.bias
            ), None

    attention = FlashMTPAttention.__new__(FlashMTPAttention)
    nn.Module.__init__(attention)
    attention.q_size = 4
    attention.kv_size = 2
    attention.qkv_proj = _QKVProjection()

    hidden_states = torch.arange(2 * 3 * 5, dtype=torch.float32).view(2, 3, 5) / 7
    target_hidden = torch.arange(2 * 4 * 5, dtype=torch.float32).view(2, 4, 5) / 11
    actual = attention._project_qkv(hidden_states, target_hidden)

    qkv_draft = torch.nn.functional.linear(
        hidden_states, attention.qkv_proj.weight, attention.qkv_proj.bias
    )
    q, k_draft, v_draft = qkv_draft.split([4, 2, 2], dim=-1)
    kv_context = torch.nn.functional.linear(
        target_hidden,
        attention.qkv_proj.weight[4:],
        attention.qkv_proj.bias[4:],
    )
    k_context, v_context = kv_context.split([2, 2], dim=-1)

    for output, expected in zip(
        actual, (q, k_draft, v_draft, k_context, v_context), strict=True
    ):
        torch.testing.assert_close(output, expected)


def test_flashmtp_algorithm_capabilities():
    algo = SpeculativeAlgorithm.from_string("flashmtp")
    assert algo.is_flashmtp()
    assert algo.uses_linear_chain_verify()
    assert algo.uses_block_verify_reservation()
    assert algo.supports_target_verify_for_draft()
    assert not algo.has_draft_kv()


def test_flashmtp_cuda_graph_pivot_buffer_load_and_padding():
    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    runner.flashmtp_pivot_hidden = torch.full((4, 6), -1.0)
    source = torch.arange(12, dtype=torch.float32).view(2, 6)
    forward_batch = SimpleNamespace(spec_info=SimpleNamespace(hidden_states=source))

    runner._load_flashmtp_pivot_hidden(forward_batch, raw_bs=2, padded_bs=4)

    torch.testing.assert_close(runner.flashmtp_pivot_hidden[:2], source)
    torch.testing.assert_close(runner.flashmtp_pivot_hidden[2:], torch.zeros((2, 6)))


def test_flashmtp_cuda_graph_rejects_wrong_pivot_shape():
    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    runner.flashmtp_pivot_hidden = torch.zeros((4, 6))
    forward_batch = SimpleNamespace(
        spec_info=SimpleNamespace(hidden_states=torch.zeros((2, 5)))
    )

    with pytest.raises(ValueError, match="pivot shape mismatch"):
        runner._load_flashmtp_pivot_hidden(forward_batch, raw_bs=2, padded_bs=4)


def test_qwen_captures_post_final_layer_boundary():
    class _Layer(nn.Module):
        def forward(self, positions, hidden_states, forward_batch, residual):
            return hidden_states + 1, residual

    class _Norm(nn.Module):
        def forward(self, hidden_states, residual=None):
            if residual is None:
                return hidden_states
            return hidden_states + residual, None

    model = Qwen2Model.__new__(Qwen2Model)
    nn.Module.__init__(model)
    model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    model.layers = nn.ModuleList([_Layer()])
    model.start_layer = 0
    model.end_layer = 1
    model.layers_to_capture = [1]
    model.norm = _Norm()

    input_embeds = torch.tensor([[1.0, 2.0]])
    hidden_states, aux_hidden_states = model(
        input_ids=None,
        positions=torch.tensor([0]),
        forward_batch=None,
        input_embeds=input_embeds,
    )

    expected = input_embeds + 1
    torch.testing.assert_close(hidden_states, expected)
    assert len(aux_hidden_states) == 1
    torch.testing.assert_close(aux_hidden_states[0], expected)
