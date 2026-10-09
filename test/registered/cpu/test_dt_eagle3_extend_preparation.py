"""Differential CPU contract for EAGLE3's dead-input preparation optimization.

Real ScheduleBatch preparation is the oracle. CUDA allocation is replaced at
its boundary, not by another implementation of batch preparation. This guards
the specialized helper against drift in the generic request/KV bookkeeping;
GPU ownership, token parity and performance require separate integration runs.
"""

from array import array
import dataclasses
from enum import Enum
from types import MethodType, SimpleNamespace

import pytest
import torch

import sglang.srt.managers.schedule_batch as schedule_module
import sglang.srt.mem_cache.allocation as allocation_module
import sglang.srt.runtime_context as runtime_context
import sglang.srt.utils.common as common_module
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo, ScheduleBatch
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.dt_draft_engine import _prepare_eagle3_extend_batch
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils.common import Range


class CacheStub:
    def __init__(self):
        self.swa = False
        self.mamba = False

    def supports_swa(self):
        return self.swa

    def supports_mamba(self):
        return self.mamba


def snapshot(value):
    """Compare every instance field, including fields added by generic prep."""
    if isinstance(value, torch.Tensor):
        return (
            "tensor",
            str(value.dtype),
            str(value.device),
            tuple(value.shape),
            value.tolist(),
        )
    if isinstance(value, array):
        return ("array", value.typecode, list(value))
    if isinstance(value, Enum):
        return (type(value).__name__, value.name)
    if dataclasses.is_dataclass(value):
        # vars also observes dynamic fields on ScheduleBatch.
        members = (
            vars(value)
            if hasattr(value, "__dict__")
            else {
                field.name: getattr(value, field.name)
                for field in dataclasses.fields(value)
            }
        )
        return {
            key: snapshot(item) for key, item in members.items() if not callable(item)
        }
    if isinstance(value, (SimpleNamespace, CacheStub)):
        return {
            key: snapshot(item)
            for key, item in vars(value).items()
            if not callable(item)
        }
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [snapshot(item) for item in value]
    return value


def make_batch(rows):
    reqs = []
    for index, row in enumerate(rows):
        prefix, delta = row["prefix"], row["delta"]
        end = prefix + delta
        origin_len = row.get("origin_len", min(prefix, 7))
        # An origin longer than the current range covers max(end, origin_len).
        full_len = max(end, origin_len)
        history = array("q", (token % 97 for token in range(full_len)))
        req = SimpleNamespace(
            rid=f"row-{index}",
            req_pool_idx=index + 1,
            origin_input_ids=history[:origin_len],
            output_ids=history[origin_len:],
            full_untruncated_fill_ids=history,
            prefix_indices=torch.arange(
                1000 * (index + 1), 1000 * (index + 1) + prefix, dtype=torch.int64
            ),
            extend_range=Range(prefix, end),
            kv=ReqKvInfo(kv_allocated_len=prefix, swa_evicted_seqlen=0),
            kv_committed_len=prefix,
            extend_batch_idx=row.get("extend_batch_idx", 3),
            decode_batch_idx=4,
            logprob_start_len=row.get("logprob_start_len", -1),
            input_embeds=None,
            positional_embed_overrides=None,
            multimodal_inputs=None,
            mamba_pool_idx=None,
            retracted_stain=row.get("retracted_stain", False),
            is_retracted=True,
            cached_tokens=11,
            already_computed=row.get("already_computed", max(0, prefix - 2)),
            _cache_breakdown_computed=row.get("breakdown_computed", False),
            host_hit_length=row.get("host_hit_length", 3),
            storage_hit_length=row.get("storage_hit_length", 1),
            cached_tokens_device=71,
            cached_tokens_host=72,
            cached_tokens_storage=73,
            sampling_params=SimpleNamespace(
                temperature=0.0, top_k=1, top_p=1.0, min_p=0.0
            ),
            return_logprob=False,
            grammar=None,
            return_hidden_states=False,
            is_prefill_only=False,
        )
        req.get_fill_ids = MethodType(Req.get_fill_ids, req)
        reqs.append(req)
    max_end = max((req.extend_range.end for req in reqs), default=1)
    pool = SimpleNamespace(
        device="cpu",
        req_to_token=torch.full((len(reqs) + 1, max_end + 4), -1, dtype=torch.int32),
    )
    for req in reqs:
        pool.req_to_token[req.req_pool_idx, : len(req.prefix_indices)] = (
            req.prefix_indices.to(torch.int32)
        )
    return ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=SimpleNamespace(page_size=1),
        tree_cache=CacheStub(),
        model_config=SimpleNamespace(is_encoder_decoder=False, vocab_size=128),
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.EAGLE3,
    )


@pytest.fixture
def cpu_contract(monkeypatch):
    state = SimpleNamespace(alloc_calls=[], sampling_calls=[], fail=None, mamba=False)
    args = SimpleNamespace(enable_mamba_extra_buffer=lambda: state.mamba)
    monkeypatch.setattr(runtime_context, "get_server_args", lambda: args)
    monkeypatch.setattr(schedule_module, "get_server_args", lambda: args)
    monkeypatch.setattr(common_module, "is_pin_memory_available", lambda _device: False)
    monkeypatch.setattr(
        schedule_module, "is_pin_memory_available", lambda _device: False
    )

    def allocate(batch):
        # Observe the real preparation/allocator contract, including CPU mirrors
        # and the old owned boundaries. Simulate only allocator-owned mutations.
        state.alloc_calls.append(
            snapshot(
                {
                    "forward_mode": batch.forward_mode,
                    "prefix_lens": batch.prefix_lens,
                    "extend_lens": batch.extend_lens,
                    "extend_num_tokens": batch.extend_num_tokens,
                    "seq_lens": batch.seq_lens,
                    "seq_lens_cpu": batch.seq_lens_cpu,
                    "prefix_indices": [req.prefix_indices for req in batch.reqs],
                    "old_owned": [req.kv.kv_allocated_len for req in batch.reqs],
                    "old_committed": [req.kv_committed_len for req in batch.reqs],
                }
            )
        )
        if state.fail == "before_allocation":
            raise RuntimeError("injected allocation failure")
        out = torch.arange(20000, 20000 + batch.extend_num_tokens, dtype=torch.int64)
        offset = 0
        for req, prefix, end, length in zip(
            batch.reqs,
            batch.prefix_lens,
            batch.seq_lens_cpu.tolist(),
            batch.extend_lens,
        ):
            assert end - prefix == length
            batch.req_to_token_pool.req_to_token[req.req_pool_idx, prefix:end] = out[
                offset : offset + length
            ].to(torch.int32)
            req.kv.kv_allocated_len = end
            offset += length
        if state.fail == "after_owned_update":
            raise RuntimeError("injected allocation failure")
        indices_cpu = torch.tensor(
            [req.req_pool_idx for req in batch.reqs], dtype=torch.int64
        )
        return out, indices_cpu.clone(), indices_cpu

    def sampling_factory(_cls, batch, vocab_size):
        # Same metadata contract on both sides; no fake CUDA sampling kernels.
        result = snapshot(
            {
                "vocab_size": vocab_size,
                "rids": [req.rid for req in batch.reqs],
                "sampling_params": [req.sampling_params for req in batch.reqs],
                "output_ids": [req.output_ids for req in batch.reqs],
                "forward_mode": batch.forward_mode,
                "seq_lens": batch.seq_lens,
                "extend_lens": batch.extend_lens,
                "req_pool_indices": batch.req_pool_indices,
            }
        )
        state.sampling_calls.append(result)
        return result

    monkeypatch.setattr(schedule_module, "alloc_for_extend", allocate)
    monkeypatch.setattr(allocation_module, "alloc_for_extend", allocate)
    monkeypatch.setattr(
        SamplingBatchInfo, "from_schedule_batch", classmethod(sampling_factory)
    )
    return state


ROWS = [
    [{"prefix": 1, "delta": 1}],
    [
        {"prefix": 2, "delta": 3, "host_hit_length": 5, "storage_hit_length": 9},
        {"prefix": 19, "delta": 1, "breakdown_computed": True, "already_computed": 21},
        {"prefix": 4096, "delta": 7, "logprob_start_len": 4098},
    ],
    [
        {"prefix": 8, "delta": 2, "retracted_stain": True},
        {"prefix": 12, "delta": 4, "retracted_stain": True, "breakdown_computed": True},
        {"prefix": 3, "delta": 1, "origin_len": 9, "logprob_start_len": 1},
    ],
]


@pytest.mark.parametrize("rows", ROWS)
def test_preparation_matches_generic_except_unused_input_staging(
    cpu_contract, monkeypatch, rows
):
    generic, optimized = make_batch(rows), make_batch(rows)
    before_history = [snapshot(req.full_untruncated_fill_ids) for req in optimized.reqs]
    generic.prepare_for_extend()
    expected_ids = [
        token
        for req in generic.reqs
        for token in req.get_fill_ids()[len(req.prefix_indices) :]
    ]
    assert generic.prefill_input_ids_cpu.tolist() == expected_ids
    # Forbid full-history reads in the candidate: metadata/feature IDs suffice.
    for req in optimized.reqs:
        monkeypatch.setattr(
            req,
            "get_fill_ids",
            lambda: pytest.fail("candidate rebuilt committed inputs"),
        )
    assert _prepare_eagle3_extend_batch(optimized) is True
    assert optimized.prefill_input_ids_cpu is None
    assert optimized.input_ids is None
    generic.prefill_input_ids_cpu = None  # The one intentional semantic difference.
    assert snapshot(optimized) == snapshot(generic)
    assert cpu_contract.alloc_calls[0] == cpu_contract.alloc_calls[1]
    assert cpu_contract.sampling_calls[0] == cpu_contract.sampling_calls[1]
    assert [
        snapshot(req.full_untruncated_fill_ids) for req in optimized.reqs
    ] == before_history


@pytest.mark.parametrize(
    "guard",
    [
        "empty",
        "dllm",
        "overlap",
        "return_logprob",
        "encoder",
        "mamba_buffer",
        "mamba_row",
        "swa",
        "mamba_cache",
        "multimodal",
        "embeds",
        "positions",
        "other_algorithm",
    ],
)
def test_unsupported_guard_returns_before_any_mutation(cpu_contract, guard):
    batch = make_batch([{"prefix": 8, "delta": 2}])
    req = batch.reqs[0]
    if guard == "empty":
        batch.reqs = []
    elif guard == "dllm":
        batch.dllm_config = SimpleNamespace()
    elif guard == "overlap":
        batch.enable_overlap = True
    elif guard == "return_logprob":
        batch.return_logprob = True
    elif guard == "encoder":
        batch.model_config.is_encoder_decoder = True
    elif guard == "mamba_buffer":
        cpu_contract.mamba = True
    elif guard == "mamba_row":
        req.mamba_pool_idx = 7
    elif guard == "swa":
        batch.tree_cache.swa = True
    elif guard == "mamba_cache":
        batch.tree_cache.mamba = True
    elif guard == "multimodal":
        req.multimodal_inputs = SimpleNamespace()
    elif guard == "embeds":
        req.input_embeds = [[1.0]]
    elif guard == "positions":
        req.positional_embed_overrides = SimpleNamespace()
    elif guard == "other_algorithm":
        batch.spec_algorithm = SpeculativeAlgorithm.STANDALONE
    before = snapshot(batch)
    assert _prepare_eagle3_extend_batch(batch) is False
    assert snapshot(batch) == before
    assert cpu_contract.alloc_calls == []
    assert cpu_contract.sampling_calls == []


@pytest.mark.parametrize("guard", ["overlap", "other_algorithm"])
def test_rejected_candidate_can_fall_back_to_generic_unchanged(cpu_contract, guard):
    fallback, generic = make_batch(ROWS[1]), make_batch(ROWS[1])
    for batch in (fallback, generic):
        if guard == "overlap":
            batch.enable_overlap = True
        else:
            batch.spec_algorithm = SpeculativeAlgorithm.STANDALONE
    assert _prepare_eagle3_extend_batch(fallback) is False
    fallback.prepare_for_extend()
    generic.prepare_for_extend()
    assert snapshot(fallback) == snapshot(generic)
    assert cpu_contract.alloc_calls[0] == cpu_contract.alloc_calls[1]


@pytest.mark.parametrize(
    "malformation", ["truncated_history", "wrong_prefix", "backward_range"]
)
def test_malformed_history_fails_before_allocation(cpu_contract, malformation):
    batch = make_batch([{"prefix": 8, "delta": 2}])
    req = batch.reqs[0]
    if malformation == "truncated_history":
        req.full_untruncated_fill_ids = req.full_untruncated_fill_ids[:-1]
    elif malformation == "wrong_prefix":
        req.extend_range = Range(7, 10)
    else:
        req.extend_range = Range(8, 7)
    before = snapshot(batch)
    with pytest.raises(ValueError, match="committed history"):
        _prepare_eagle3_extend_batch(batch)
    assert snapshot(batch) == before
    assert cpu_contract.alloc_calls == []
    assert cpu_contract.sampling_calls == []


@pytest.mark.parametrize("failure", ["before_allocation", "after_owned_update"])
def test_allocation_failure_preserves_generic_cleanup_state(cpu_contract, failure):
    # A failure after allocator-owned KV updates models the outer cleanup's
    # observable state. This does not claim to test real CUDA failure fencing.
    generic, optimized = make_batch(ROWS[1]), make_batch(ROWS[1])
    cpu_contract.fail = failure
    for batch, prepare in (
        (generic, generic.prepare_for_extend),
        (optimized, lambda: _prepare_eagle3_extend_batch(optimized)),
    ):
        with pytest.raises(RuntimeError, match="injected allocation failure"):
            prepare()
        assert [req.extend_batch_idx for req in batch.reqs] == [3, 3, 3]
    assert snapshot(optimized) == snapshot(generic)
    assert cpu_contract.alloc_calls[0] == cpu_contract.alloc_calls[1]
    assert cpu_contract.sampling_calls == []
