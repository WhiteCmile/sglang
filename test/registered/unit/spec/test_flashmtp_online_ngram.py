from types import SimpleNamespace

import pytest
import torch

from sglang.srt.speculative.flashmtp_shared import OnlineNgramCorrectionCache


def _cache(*, min_sources: int = 2, source_gated_from: int = 0):
    return OnlineNgramCorrectionCache(
        order=2,
        max_proposals=3,
        max_rows=2,
        max_entries=32,
        min_count=1,
        min_sources=min_sources,
        source_gated_from=source_gated_from,
        device="cpu",
        pin_memory=False,
    )


def test_online_ngram_requires_distinct_producers_for_gated_tokens():
    cache = _cache()
    evidence = [1, 2, 3, 4, 5, 6]

    cache.predict_contexts([evidence], source_ids=["target-0"])
    cache.predict_contexts([evidence], source_ids=["target-0"])
    assert cache.predict_contexts([[1, 2]], source_ids=["target-0"]) == [()]

    cache.predict_contexts([evidence], source_ids=["target-1"])
    assert cache.predict_contexts([[1, 2]], source_ids=["target-1"]) == [
        (3, 4, 5)
    ]


def test_online_ngram_source_gate_only_applies_to_tail():
    cache = _cache(source_gated_from=2)
    cache.predict_contexts(
        [[1, 2, 3, 4, 5, 6]], source_ids=["target-0"]
    )

    assert cache.predict_contexts([[1, 2]], source_ids=["target-0"]) == [(3, 4)]


def test_online_ngram_rejects_misaligned_source_ids():
    cache = _cache()
    with pytest.raises(ValueError, match="source count"):
        cache.predict_contexts([[1, 2], [3, 4]], source_ids=["target-0"])


def test_online_ngram_single_source_request_path_stays_conservative():
    cache = _cache()
    req = SimpleNamespace(
        rid="request-0",
        origin_input_ids=[1, 2],
        output_ids=[3, 4, 5, 6],
    )
    candidates = torch.zeros((1, 4), dtype=torch.int64)
    verify_lens = torch.ones(1, dtype=torch.int32)

    assert cache.apply(
        reqs=[req], candidates=candidates, verify_lens_cpu=verify_lens
    ) == (0, 0)
    torch.testing.assert_close(candidates, torch.zeros_like(candidates))
    torch.testing.assert_close(verify_lens, torch.ones_like(verify_lens))
