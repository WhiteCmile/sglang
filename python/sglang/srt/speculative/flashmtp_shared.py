from __future__ import annotations

import bisect
import os
from collections.abc import Sequence

import torch

SHARED_OPTIMIZATIONS_ENV = "SGLANG_FLASHMTP_SHARED_OPTIMIZATIONS"
RAGGED_VERIFY_ENV = "SGLANG_FLASHMTP_RAGGED_VERIFY"
ADAPTIVE_VERIFY_ENV = "SGLANG_FLASHMTP_ADAPTIVE_VERIFY"
PINNED_VERIFY_METADATA_ENV = "SGLANG_FLASHMTP_PINNED_VERIFY_METADATA"
VOCAB_PARALLEL_GREEDY_ENV = "SGLANG_FLASHMTP_VOCAB_PARALLEL_GREEDY"
GRAPH_TIER_PADDING_ENV = "SGLANG_FLASHMTP_GRAPH_TIER_PADDING"
ADAPTIVE_THRESHOLDS_ENV = "SGLANG_FLASHMTP_ADAPTIVE_VERIFY_THRESHOLDS"
BUDGETED_VERIFY_ENV = "SGLANG_FLASHMTP_BUDGETED_VERIFY"
BUDGETED_VERIFY_THRESHOLDS_ENV = "SGLANG_FLASHMTP_BUDGETED_VERIFY_THRESHOLDS"
BUDGETED_VERIFY_MIN_BATCH_ENV = "SGLANG_FLASHMTP_BUDGETED_VERIFY_MIN_BATCH"
CHARD_CHUNK_SIZE_ENV = "SGLANG_FLASHMTP_CHARD_CHUNK_SIZE"
DEFAULT_ADAPTIVE_THRESHOLDS = (
    0.625,
    1.25,
    1.375,
    3.875,
    1.25,
    3.125,
    3.625,
)


def shared_optimizations_enabled() -> bool:
    return os.environ.get(SHARED_OPTIMIZATIONS_ENV, "0") == "1"


def budgeted_verify_enabled() -> bool:
    """Enable the production-safe R+A+P verify bundle.

    Unlike the historical all-on bundle, this deliberately leaves vocabulary
    epilogue and graph-tier padding unchanged.  Baseline and DT therefore use
    the same configured CUDA Graph domain while sharing only the algorithmic
    verify-budget policy.
    """
    raw = os.environ.get(BUDGETED_VERIFY_ENV, "0")
    if raw not in ("0", "1"):
        raise ValueError(f"{BUDGETED_VERIFY_ENV} must be 0 or 1")
    return raw == "1"


def read_chard_chunk_size(*, block_size: int) -> int | None:
    """Return the anchor-inclusive CHARD verify width, or ``None`` when off."""
    raw = os.environ.get(CHARD_CHUNK_SIZE_ENV, "0")
    try:
        chunk_size = int(raw)
    except ValueError as exc:
        raise ValueError(f"{CHARD_CHUNK_SIZE_ENV} must be an integer") from exc
    if chunk_size == 0:
        return None
    if not 2 <= chunk_size < block_size:
        raise ValueError(
            f"{CHARD_CHUNK_SIZE_ENV} must be in [2, {block_size - 1}]"
        )
    if block_size % chunk_size != 0:
        raise ValueError(
            f"{CHARD_CHUNK_SIZE_ENV}={chunk_size} must divide block_size={block_size}"
        )
    return chunk_size


def build_chard_continuation(
    *,
    candidates: torch.Tensor,
    queue_lens: torch.Tensor,
    verify_lens: torch.Tensor,
    commit_lens: torch.Tensor,
    bonus_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift fully accepted, anchor-safe candidate suffixes to the next round.

    Candidate rows are anchor-inclusive.  A verified prefix of width ``C`` may
    continue only when all ``C`` output tokens were committed and the Target's
    last (bonus) token equals candidate ``C``, which becomes the next anchor.
    """
    if candidates.ndim != 2:
        raise ValueError("CHARD candidates must have shape [batch, block_size]")
    batch_size, block_size = candidates.shape
    for name, value in (
        ("queue_lens", queue_lens),
        ("verify_lens", verify_lens),
        ("commit_lens", commit_lens),
        ("bonus_tokens", bonus_tokens),
    ):
        if value.ndim != 1 or value.numel() != batch_size:
            raise ValueError(f"CHARD {name} must have shape [batch]")

    verify_lens = verify_lens.to(device=candidates.device, dtype=torch.int64)
    queue_lens = queue_lens.to(device=candidates.device, dtype=torch.int64)
    commit_lens = commit_lens.to(device=candidates.device, dtype=torch.int64)
    bonus_tokens = bonus_tokens.to(device=candidates.device, dtype=candidates.dtype)
    boundary_indices = verify_lens.clamp(max=block_size - 1)
    boundary_tokens = candidates.gather(1, boundary_indices[:, None]).view(-1)
    remaining = queue_lens - verify_lens
    can_continue = (
        (verify_lens < queue_lens)
        & (commit_lens == verify_lens)
        & (bonus_tokens == boundary_tokens)
    )
    next_lens = torch.where(can_continue, remaining, torch.zeros_like(remaining))

    columns = torch.arange(block_size, device=candidates.device)[None, :]
    source_columns = columns + verify_lens[:, None]
    shifted = candidates.gather(1, source_columns.clamp(max=block_size - 1))
    shifted.masked_fill_(source_columns >= block_size, 0)
    shifted.masked_fill_(~can_continue[:, None], 0)
    return shifted, next_lens.to(dtype=torch.int32)


def component_enabled(env_name: str) -> bool:
    """Read an ablatable component while preserving the legacy bundle.

    Explicit component switches win over the compatibility bundle so a
    controlled leave-one-out cell can be represented without changing the
    historical ``SGLANG_FLASHMTP_SHARED_OPTIMIZATIONS=1`` meaning.  When no
    component switch is present, the bundle keeps enabling every A-class
    optimization exactly as before.
    """
    raw = os.environ.get(env_name)
    if raw is None:
        return shared_optimizations_enabled()
    if raw not in ("0", "1"):
        raise ValueError(f"{env_name} must be 0 or 1")
    return raw == "1"


def ragged_verify_enabled() -> bool:
    return budgeted_verify_enabled() or component_enabled(RAGGED_VERIFY_ENV)


def adaptive_verify_enabled() -> bool:
    return budgeted_verify_enabled() or component_enabled(ADAPTIVE_VERIFY_ENV)


def pinned_verify_metadata_enabled() -> bool:
    return budgeted_verify_enabled() or component_enabled(PINNED_VERIFY_METADATA_ENV)


def vocab_parallel_greedy_enabled() -> bool:
    return component_enabled(VOCAB_PARALLEL_GREEDY_ENV)


def graph_tier_padding_enabled() -> bool:
    return component_enabled(GRAPH_TIER_PADDING_ENV)


def validate_component_dependencies() -> None:
    """Reject component combinations that have no executable interpretation."""
    ragged = ragged_verify_enabled()
    dependents = {
        ADAPTIVE_VERIFY_ENV: adaptive_verify_enabled(),
        PINNED_VERIFY_METADATA_ENV: pinned_verify_metadata_enabled(),
        VOCAB_PARALLEL_GREEDY_ENV: vocab_parallel_greedy_enabled(),
    }
    invalid = [name for name, enabled in dependents.items() if enabled and not ragged]
    if invalid:
        raise ValueError(", ".join(invalid) + f" require {RAGGED_VERIFY_ENV}=1")


def read_adaptive_thresholds(*, proposal_count: int) -> tuple[float, ...] | None:
    if not adaptive_verify_enabled():
        return None
    raw = os.environ.get(
        BUDGETED_VERIFY_THRESHOLDS_ENV,
        os.environ.get(ADAPTIVE_THRESHOLDS_ENV),
    )
    if raw is None:
        thresholds = DEFAULT_ADAPTIVE_THRESHOLDS
    else:
        try:
            thresholds = tuple(float(value.strip()) for value in raw.split(","))
        except ValueError as exc:
            raise ValueError(
                f"{ADAPTIVE_THRESHOLDS_ENV} must contain comma-separated numbers"
            ) from exc
    if len(thresholds) != proposal_count:
        raise ValueError(
            f"{ADAPTIVE_THRESHOLDS_ENV} must have {proposal_count} values; "
            f"got {len(thresholds)}"
        )
    if any(threshold < 0 for threshold in thresholds):
        raise ValueError(f"{ADAPTIVE_THRESHOLDS_ENV} values must be non-negative")
    return thresholds


def read_adaptive_min_batch() -> int:
    raw = os.environ.get(
        BUDGETED_VERIFY_MIN_BATCH_ENV,
        os.environ.get("SGLANG_FLASHMTP_ADAPTIVE_MIN_BATCH", "24"),
    )
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{BUDGETED_VERIFY_MIN_BATCH_ENV} must be positive")
    return value


def select_adaptive_verify_lens(
    logit_margins: torch.Tensor,
    *,
    threshold: float | Sequence[float] | torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Choose the first low-confidence proposal as the verify cutoff."""
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


def ragged_epilogue_capacity(server_args) -> int:
    graph_max_bs = max(server_args.cuda_graph_config.decode.bs, default=0)
    return max(1, graph_max_bs, int(server_args.max_running_requests or 0))


def advance_ragged_graph_tier(
    *,
    total_verify_tokens: int,
    max_verify_tokens: int,
    grid: list[int],
    tier_offset: int,
) -> int:
    if tier_offset < 0:
        raise ValueError("tier_offset must be non-negative")
    base_index = bisect.bisect_left(grid, total_verify_tokens)
    if base_index == len(grid):
        raise ValueError("no graph tier can cover the requested verify tokens")
    max_packable_index = bisect.bisect_right(grid, max_verify_tokens) - 1
    if max_packable_index < base_index:
        return grid[base_index]
    return grid[min(base_index + tier_offset, max_packable_index)]


def plan_ragged_execution_tier(
    *,
    total_verify_tokens: int,
    max_verify_tokens: int,
    capture_grid: list[int],
    tier_offset: int,
    batch_size: int,
    max_graph_rows: int,
) -> tuple[int, list[int], bool]:
    if not capture_grid or total_verify_tokens > capture_grid[-1]:
        return total_verify_tokens, [total_verify_tokens], False
    graph_num_tokens = advance_ragged_graph_tier(
        total_verify_tokens=total_verify_tokens,
        max_verify_tokens=max_verify_tokens,
        grid=capture_grid,
        tier_offset=tier_offset,
    )
    graph_row_capacity = min(graph_num_tokens, max_graph_rows)
    if batch_size > graph_row_capacity:
        return total_verify_tokens, [total_verify_tokens], False
    return graph_num_tokens, capture_grid, True


def reclaim_ragged_graph_padding(
    verify_lens: list[int],
    *,
    block_size: int,
    graph_num_tokens: int,
    padding_priorities: list[int] | None = None,
) -> list[int]:
    if not verify_lens:
        raise ValueError("verify_lens must not be empty")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if min(verify_lens) < 1 or max(verify_lens) > block_size:
        raise ValueError(f"verify_lens must be in [1, {block_size}]")
    total = sum(verify_lens)
    if graph_num_tokens < total:
        raise ValueError("graph_num_tokens must cover every requested verify token")
    if padding_priorities is not None and len(padding_priorities) != len(verify_lens):
        raise ValueError("padding_priorities must have one value per verify row")

    reclaimed = list(verify_lens)
    remaining = min(
        graph_num_tokens - total,
        len(reclaimed) * block_size - total,
    )
    if padding_priorities is not None and remaining:
        ranked = sorted(
            (index for index, value in enumerate(reclaimed) if value < block_size),
            key=lambda index: (-padding_priorities[index], index),
        )
        for index in ranked:
            reclaimed[index] += 1
            remaining -= 1
            if remaining == 0:
                break
    while remaining:
        eligible = sorted(
            (index for index, value in enumerate(reclaimed) if value < block_size),
            key=lambda index: (reclaimed[index], index),
        )
        if not eligible:
            break
        for index in eligible:
            reclaimed[index] += 1
            remaining -= 1
            if remaining == 0:
                break
    return reclaimed
