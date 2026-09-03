from __future__ import annotations

import bisect
import os
from collections import Counter, OrderedDict
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
NGRAM_CORRECTION_ENV = "SGLANG_FLASHMTP_NGRAM_CORRECTION"
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


def ngram_correction_enabled() -> bool:
    """Enable online target-feedback proposal correction.

    An explicit component switch wins so the correction can be ablated from
    the budgeted bundle without changing any other experimental factor.
    """
    raw = os.environ.get(NGRAM_CORRECTION_ENV)
    if raw is None:
        return budgeted_verify_enabled()
    if raw not in ("0", "1"):
        raise ValueError(f"{NGRAM_CORRECTION_ENV} must be 0 or 1")
    return raw == "1"


class OnlineNgramCorrectionCache:
    """Learn target-verified continuations and patch later draft proposals.

    It observes only tokens already committed by a Target, so a stale or
    ambiguous prediction remains safe: ordinary speculative verification
    rejects it and emits the exact Target token.  Co-located workers feed it
    request updates directly; a shared DT Draft can feed the same transition
    learner short committed-context suffixes carried by the task protocol.
    """

    def __init__(
        self,
        *,
        order: int,
        max_proposals: int,
        max_rows: int,
        max_entries: int,
        min_count: int,
        device: torch.device | str,
        pin_memory: bool,
        min_sources: int = 1,
        source_gated_from: int | None = None,
    ) -> None:
        if (
            order <= 0
            or max_proposals <= 0
            or max_rows <= 0
            or max_entries <= 0
            or min_count <= 0
            or min_sources <= 0
        ):
            raise ValueError("online ngram cache dimensions must be positive")
        if source_gated_from is not None and not (
            0 <= source_gated_from <= max_proposals
        ):
            raise ValueError("online ngram source gate is outside proposal width")
        self.order = int(order)
        self.max_proposals = int(max_proposals)
        self.max_entries = int(max_entries)
        self.min_count = int(min_count)
        self.min_sources = int(min_sources)
        self.source_gated_from = (
            self.max_proposals
            if source_gated_from is None
            else int(source_gated_from)
        )
        self._transitions: OrderedDict[tuple[int, ...], Counter[int]] = OrderedDict()
        self._transition_sources: dict[
            tuple[int, ...], dict[int, set[str]]
        ] = {}
        self._request_state: OrderedDict[str, tuple[int, tuple[int, ...]]] = (
            OrderedDict()
        )
        self._max_request_states = max(1024, int(max_rows) * 32)
        self._host_tokens = torch.empty(
            (max_rows, max_proposals), dtype=torch.int64, pin_memory=pin_memory
        )
        self._host_mask = torch.empty(
            (max_rows, max_proposals), dtype=torch.bool, pin_memory=pin_memory
        )
        self._device_tokens = torch.empty(
            (max_rows, max_proposals), dtype=torch.int64, device=device
        )
        self._device_mask = torch.empty(
            (max_rows, max_proposals), dtype=torch.bool, device=device
        )

    @staticmethod
    def _request_tokens(req) -> tuple[Sequence[int], Sequence[int]]:
        return req.origin_input_ids, req.output_ids

    def _record(
        self, context: tuple[int, ...], token: int, source_id: str
    ) -> tuple[int, ...]:
        counts = self._transitions.get(context)
        if counts is None:
            if len(self._transitions) >= self.max_entries:
                evicted_context, _ = self._transitions.popitem(last=False)
                self._transition_sources.pop(evicted_context, None)
            counts = Counter()
            self._transitions[context] = counts
        counts[int(token)] += 1
        if self.min_sources > 1:
            sources = self._transition_sources.setdefault(context, {})
            sources.setdefault(int(token), set()).add(source_id)
        return (*context, int(token))[-self.order :]

    @staticmethod
    def _modal_token(counts: Counter[int]) -> int:
        # Resolve count ties by token id, independently of insertion order.
        return max(counts, key=lambda token: (counts[token], -token))

    def _update_request(self, req) -> tuple[int, ...]:
        prompt_ids, output_ids = self._request_tokens(req)
        rid = str(req.rid)
        state = self._request_state.pop(rid, None)
        if state is None or state[0] > len(output_ids):
            context = tuple(int(token) for token in prompt_ids[-self.order :])
            seen = 0
        else:
            seen, context = state
        for token in output_ids[seen:]:
            context = self._record(context, int(token), "local")
        self._request_state[rid] = (len(output_ids), context)
        if len(self._request_state) > self._max_request_states:
            self._request_state.popitem(last=False)
        return context

    def _predict(self, context: tuple[int, ...]) -> tuple[int, ...]:
        predicted: list[int] = []
        for proposal_index in range(self.max_proposals):
            counts = self._transitions.get(context)
            if not counts:
                break
            token = self._modal_token(counts)
            if counts[token] < self.min_count:
                break
            if proposal_index >= self.source_gated_from and self.min_sources > 1:
                sources = self._transition_sources.get(context, {}).get(token, ())
                if len(sources) < self.min_sources:
                    break
            predicted.append(token)
            context = (*context, token)[-self.order :]
        return tuple(predicted)

    def _patch(
        self,
        *,
        predictions: Sequence[tuple[int, ...]],
        candidates: torch.Tensor,
        verify_lens: torch.Tensor | None,
    ) -> tuple[int, int]:
        batch_size = len(predictions)
        if tuple(candidates.shape) != (batch_size, self.max_proposals + 1):
            raise ValueError("online ngram candidate shape mismatch")
        if batch_size > self._host_tokens.shape[0]:
            raise ValueError("online ngram batch exceeds configured capacity")

        host_mask = self._host_mask[:batch_size]
        host_mask.zero_()
        host_tokens = self._host_tokens[:batch_size]
        hit_rows = 0
        predicted_tokens = 0
        for row, prediction in enumerate(predictions):
            width = len(prediction)
            if width == 0:
                continue
            hit_rows += 1
            predicted_tokens += width
            host_mask[row, :width] = True
            host_tokens[row, :width] = torch.as_tensor(
                prediction, dtype=torch.int64
            )

        if predicted_tokens == 0:
            return 0, 0
        device_tokens = self._device_tokens[:batch_size]
        device_mask = self._device_mask[:batch_size]
        device_tokens.copy_(host_tokens, non_blocking=True)
        device_mask.copy_(host_mask, non_blocking=True)
        proposal_view = candidates[:, 1:]
        proposal_view.copy_(torch.where(device_mask, device_tokens, proposal_view))

        if verify_lens is not None:
            if verify_lens.ndim != 1 or verify_lens.numel() != batch_size:
                raise ValueError("online ngram verify lengths must be one row")
            cached_lens = host_mask.sum(dim=1, dtype=torch.int32) + 1
            if verify_lens.device.type == "cpu":
                torch.maximum(verify_lens, cached_lens, out=verify_lens)
            else:
                cached_lens_device = cached_lens.to(
                    verify_lens.device, non_blocking=True
                )
                torch.maximum(verify_lens, cached_lens_device, out=verify_lens)
        return hit_rows, predicted_tokens

    def predict_contexts(
        self,
        context_rows: Sequence[Sequence[int]],
        *,
        source_ids: Sequence[str] | None = None,
    ) -> list[tuple[int, ...]]:
        """Learn explicit committed suffixes and return cached continuations."""
        normalized = [tuple(int(token) for token in row) for row in context_rows]
        if source_ids is None:
            normalized_sources = ["local"] * len(normalized)
        else:
            if len(source_ids) != len(normalized):
                raise ValueError("online ngram source count does not match rows")
            normalized_sources = [str(source_id) for source_id in source_ids]
        for row, source_id in zip(normalized, normalized_sources, strict=True):
            for index in range(self.order, len(row)):
                self._record(
                    row[index - self.order : index], row[index], source_id
                )
        return [
            self._predict(row[-self.order :]) if len(row) >= self.order else ()
            for row in normalized
        ]

    def predict_requests(self, reqs: Sequence) -> list[tuple[int, ...]]:
        """Update co-located request state and return cached continuations."""
        contexts = [self._update_request(req) for req in reqs]
        return [self._predict(context) for context in contexts]

    def patch_predictions(
        self,
        *,
        predictions: Sequence[tuple[int, ...]],
        candidates: torch.Tensor,
        verify_lens: torch.Tensor | None,
    ) -> tuple[int, int]:
        return self._patch(
            predictions=predictions,
            candidates=candidates,
            verify_lens=verify_lens,
        )

    def apply_contexts(
        self,
        *,
        context_rows: Sequence[Sequence[int]],
        candidates: torch.Tensor,
        verify_lens: torch.Tensor | None,
    ) -> tuple[int, int]:
        """Learn from explicit committed suffixes and patch a shared Draft batch."""
        predictions = self.predict_contexts(context_rows)
        return self._patch(
            predictions=predictions,
            candidates=candidates,
            verify_lens=verify_lens,
        )

    def apply(
        self,
        *,
        reqs: Sequence,
        candidates: torch.Tensor,
        verify_lens_cpu: torch.Tensor | None,
    ) -> tuple[int, int]:
        """Patch cached proposal prefixes and return ``(hit_rows, tokens)``."""
        predictions = self.predict_requests(reqs)
        return self._patch(
            predictions=predictions,
            candidates=candidates,
            verify_lens=verify_lens_cpu,
        )


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
        raise ValueError(
            ", ".join(invalid)
            + f" require {RAGGED_VERIFY_ENV}=1"
        )


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
