from types import SimpleNamespace

import torch

from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
    MultimemAllGatherer,
)
from sglang.srt.environ import envs


def test_multimem_all_gather_can_be_disabled_before_distributed_init() -> None:
    with envs.SGLANG_DISABLE_MULTIMEM_ALL_GATHER.override(True):
        gatherer = MultimemAllGatherer(max_tokens=128, enabled=True)

    assert gatherer._state is None


def test_multimem_all_gather_remains_enabled_by_default(monkeypatch) -> None:
    monkeypatch.setattr(
        "sglang.srt.distributed.get_tp_group",
        lambda: SimpleNamespace(world_size=1),
    )

    with envs.SGLANG_DISABLE_MULTIMEM_ALL_GATHER.override(False):
        gatherer = MultimemAllGatherer(max_tokens=128, enabled=True)

    assert gatherer._state is gatherer._UNINIT


def test_disabled_multimem_dispatches_to_standard_tp_gather(monkeypatch) -> None:
    calls = []
    expected = torch.empty((2, 4), dtype=torch.bfloat16)

    def fake_all_gather(tensor, *, dim):
        calls.append((tensor, dim))
        return expected

    monkeypatch.setattr(
        "sglang.srt.distributed.tensor_model_parallel_all_gather",
        fake_all_gather,
    )
    with envs.SGLANG_DISABLE_MULTIMEM_ALL_GATHER.override(True):
        gatherer = MultimemAllGatherer(max_tokens=128, enabled=True)

    source = torch.empty((2, 2), dtype=torch.bfloat16)
    assert gatherer(source) is expected
    assert calls == [(source, -1)]
