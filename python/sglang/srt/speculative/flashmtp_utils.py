from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import torch


def _cfg_get(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _as_dict(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, dict):
        return config
    if hasattr(config, "items"):
        return dict(config.items())
    return {}


def _parse_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    try:
        value = int(value)
    except Exception as exc:
        raise ValueError(f"Invalid {field_name}={value!r}.") from exc
    if value <= 0:
        raise ValueError(f"{field_name} must be positive, got {value}.")
    return value


@dataclass(frozen=True)
class FlashMTPDraftConfig:
    num_hidden_layers: int
    num_target_layers: int
    block_size: int
    target_layer_ids: list[int]
    mask_token_id: int
    pivot_fuse_mode: str
    local_position: bool
    left_shift: bool
    markov_head_type: str
    markov_output_mode: str
    markov_rank: int

    @property
    def num_proposed_tokens(self) -> int:
        return self.block_size - 1

    @property
    def draft_block_len(self) -> int:
        """Number of slots evaluated by the parallel draft backbone."""
        return self.block_size - 1 if self.left_shift else self.block_size


def gather_flashmtp_next_pivot(
    *,
    packed_hidden: torch.Tensor,
    accept_len: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Gather the target pivot at the last committed verify-input position."""
    batch_size = int(accept_len.numel())
    expected_rows = batch_size * int(block_size)
    if packed_hidden.ndim != 2 or packed_hidden.shape[0] != expected_rows:
        raise ValueError(
            "FlashMTP verify hidden shape mismatch: "
            f"expected first dimension {expected_rows}, "
            f"got {tuple(packed_hidden.shape)}."
        )
    hidden = packed_hidden.view(batch_size, block_size, -1)
    gather_index = (
        accept_len.to(torch.long).view(batch_size, 1, 1).expand(-1, 1, hidden.shape[-1])
    )
    return torch.gather(hidden, dim=1, index=gather_index).squeeze(1).contiguous()


def parse_flashmtp_draft_config(*, draft_hf_config: Any) -> FlashMTPDraftConfig:
    """Parse the FlashMTP checkpoint contract used by the SGLang runtime."""
    flashmtp_config = _as_dict(_cfg_get(draft_hf_config, "flashmtp_config", None))

    num_hidden_layers = _parse_positive_int(
        _cfg_get(draft_hf_config, "num_hidden_layers", None),
        "FlashMTP num_hidden_layers",
    )
    num_target_layers = _parse_positive_int(
        _cfg_get(draft_hf_config, "num_target_layers", None),
        "FlashMTP num_target_layers",
    )
    block_size = _parse_positive_int(
        _cfg_get(draft_hf_config, "block_size", None),
        "FlashMTP block_size",
    )
    if num_hidden_layers is None or num_target_layers is None or block_size is None:
        raise ValueError(
            "FlashMTP requires num_hidden_layers, num_target_layers, and block_size "
            "in the draft config."
        )
    if block_size < 2:
        raise ValueError(
            "FlashMTP block_size includes one bonus token and must be >= 2, "
            f"got {block_size}."
        )

    raw_layer_ids = flashmtp_config.get("target_layer_ids")
    if not isinstance(raw_layer_ids, (list, tuple)) or not raw_layer_ids:
        raise ValueError(
            "FlashMTP requires a non-empty flashmtp_config.target_layer_ids list."
        )
    target_layer_ids = [int(layer_id) for layer_id in raw_layer_ids]
    if target_layer_ids != sorted(set(target_layer_ids)):
        raise ValueError(
            "FlashMTP flashmtp_config.target_layer_ids must be unique and "
            f"strictly increasing, got {target_layer_ids}."
        )
    for index, layer_id in enumerate(target_layer_ids):
        if layer_id < 0 or layer_id >= num_target_layers:
            raise ValueError(
                "FlashMTP target layer id is out of range: "
                f"target_layer_ids[{index}]={layer_id}, "
                f"num_target_layers={num_target_layers}."
            )

    mask_token_id = flashmtp_config.get("mask_token_id")
    if (
        not isinstance(mask_token_id, Integral)
        or isinstance(mask_token_id, bool)
        or int(mask_token_id) < 0
    ):
        raise ValueError(
            "FlashMTP requires a non-negative integer flashmtp_config.mask_token_id."
        )

    pivot_fuse_mode = str(flashmtp_config.get("pivot_fuse_mode", "prefix_condition"))
    if pivot_fuse_mode != "prefix_condition":
        raise NotImplementedError(
            "SGLang FlashMTP currently supports only "
            f"pivot_fuse_mode='prefix_condition', got {pivot_fuse_mode!r}."
        )

    local_position = bool(flashmtp_config.get("local_position", False))
    if not local_position:
        raise NotImplementedError(
            "SGLang FlashMTP currently supports only local_position=true."
        )

    left_shift = bool(flashmtp_config.get("left_shift", False))
    markov_head_type = str(flashmtp_config.get("markov_head_type", "none")).lower()
    markov_output_mode = str(
        flashmtp_config.get("markov_output_mode", "additive")
    ).lower()
    try:
        configured_markov_rank = int(flashmtp_config.get("markov_rank", 0))
    except Exception as exc:
        raise ValueError(
            "FlashMTP flashmtp_config.markov_rank must be an integer."
        ) from exc
    supported_markov_heads = {"none", "vanilla", "gated", "rnn", "rnn_easy", "mlp"}
    if markov_head_type not in supported_markov_heads:
        raise ValueError(
            "Unknown FlashMTP markov_head_type="
            f"{markov_head_type!r}; expected one of {sorted(supported_markov_heads)}."
        )
    if markov_output_mode not in {"additive", "direct"}:
        raise ValueError(
            "Unknown FlashMTP markov_output_mode="
            f"{markov_output_mode!r}; expected 'additive' or 'direct'."
        )
    markov_rank = configured_markov_rank if markov_head_type != "none" else 0
    if markov_head_type != "none" and markov_rank <= 0:
        raise ValueError(
            "FlashMTP markov_rank must be positive when a Markov head is enabled, "
            f"got {markov_rank}."
        )
    if markov_head_type == "none" and markov_output_mode == "direct":
        raise ValueError("FlashMTP markov_output_mode='direct' requires a Markov head.")

    model_type = _cfg_get(draft_hf_config, "model_type", None)
    if model_type is not None and str(model_type) != "qwen3":
        raise NotImplementedError(
            "SGLang FlashMTP currently supports only Qwen3 draft checkpoints, "
            f"got model_type={model_type!r}."
        )
    hidden_act = _cfg_get(draft_hf_config, "hidden_act", None)
    if hidden_act is not None and str(hidden_act) != "silu":
        raise NotImplementedError(
            "SGLang FlashMTP currently supports only hidden_act='silu', "
            f"got {hidden_act!r}."
        )

    return FlashMTPDraftConfig(
        num_hidden_layers=num_hidden_layers,
        num_target_layers=num_target_layers,
        block_size=block_size,
        target_layer_ids=target_layer_ids,
        mask_token_id=int(mask_token_id),
        pivot_fuse_mode=pivot_fuse_mode,
        local_position=local_position,
        left_shift=left_shift,
        markov_head_type=markov_head_type,
        markov_output_mode=markov_output_mode,
        markov_rank=markov_rank,
    )
