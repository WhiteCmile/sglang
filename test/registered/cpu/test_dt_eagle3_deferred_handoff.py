"""CPU ownership contracts. CUDA/event doubles do not establish GPU parity.

Run only after the task's external release/gate; imports can load heavy modules.
"""
from types import SimpleNamespace

import pytest
import torch

import sglang.srt.managers.utils as result_utils
import sglang.srt.speculative.dt_target as target
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.eagle3_target_worker_v2 import (
    _freeze_eagle3_verify_features,
)


class DeviceRows(torch.Tensor):
    """CPU storage with only the controller's device boundary replaced."""
    @property
    def device(self):
        return torch.device('cuda:0')

    @property
    def is_cuda(self):
        return True

    def record_stream(self, stream):
        pass


def rows(value):
    return value.as_subclass(DeviceRows)


def cpu(value):
    return value.as_subclass(torch.Tensor).clone()


def wave():
    hidden = rows(torch.arange(24).reshape(8, 3).to(torch.bfloat16))
    candidates = rows(torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]]))
    output = rows(torch.tensor([[11, 99, 0, 0], [21, 22, 23, 88]]).flatten())
    result = SimpleNamespace(
        next_token_ids=output, accept_lens=rows(torch.tensor([2, 4])),
        new_seq_lens=rows(torch.tensor([12, 24])),
        next_draft_input=SimpleNamespace(bonus_tokens=rows(torch.tensor([99, 88]))),
    )
    return hidden, candidates, result


def freeze(hidden, candidates, result):
    return _freeze_eagle3_verify_features(
        result, packed_hidden=hidden, candidates=candidates, export_rank=True
    )


def test_forward_freeze_survives_replay_without_reading_accept_lengths(monkeypatch):
    hidden, candidates, result = wave()
    original = [hidden, candidates, result.next_token_ids, result.accept_lens,
                result.new_seq_lens, result.next_draft_input.bonus_tokens]
    expected = [cpu(value) for value in original]
    # No host length or token materialization is allowed in the freeze boundary.
    monkeypatch.setattr(DeviceRows, 'tolist', lambda _self: pytest.fail('host read'))
    bonus = freeze(hidden, candidates, result)
    for value in original:
        value.fill_(-1)  # simulate graph replay/control-buffer reuse
    preserved = [result.dt_dflash_packed_hidden, result.dt_eagle3_committed_tokens,
                 result.next_token_ids, result.accept_lens, result.new_seq_lens, bonus]
    for actual, wanted in zip(preserved, expected, strict=True):
        torch.testing.assert_close(cpu(actual), wanted)


class Done:
    def __init__(self):
        self.recorded = False
        self.waited = False

    def record(self):
        self.recorded = True

    def synchronize(self):
        assert self.recorded
        self.waited = True


def setup_controller(monkeypatch):
    monkeypatch.setattr(result_utils, '_async_d2h', cpu)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda *_args: object())
    controller = object.__new__(target.DTTargetController)
    controller.scheduler = SimpleNamespace(
        device_module=SimpleNamespace(Event=Done), ps=SimpleNamespace(tp_size=2),
        tp_group=SimpleNamespace(rank=0, ranks=[0, 1]), tp_cpu_group=object(),
        forward_stream=SimpleNamespace(wait_event=lambda _event: None),
    )
    controller.is_export_rank = True
    controller.is_feature_backed = True
    controller.is_eagle3 = True
    controller.is_token_only = False
    controller.defer_dflash_feature_stage = False  # EAGLE must still debit credit
    controller._deferred_dflash_feature_stages = 0
    controller._feature_source_free_slots = 1
    controller._feature_source_stall_started_ns = 0
    controller.instance_id = 'target'
    controller.block_size = 4
    controller.pivot_width = 3
    controller.feature_schema_id = 'eagle3'
    controller._next_round = {'a': 1, 'b': 1}
    controller._draft_committed_len = {'a': 10, 'b': 20}
    controller._draft_kv_handle_by_request = {'a': 'kv-a', 'b': 'kv-b'}
    controller._verify_transfer_by_request = {'a': 'old-a', 'b': 'old-b'}
    controller._lifecycle_attempt_by_request = {'a': 0, 'b': 1}
    controller._next_lifecycle_attempt = 2
    controller._source_slot_by_transfer = {}
    controller._pending_send = []
    controller._parked = {}
    controller._profile_emit = lambda *_args, **_kwargs: None
    monkeypatch.setattr(target, 'profile_event_enabled', lambda _event: False)
    packets = []
    monkeypatch.setattr(target, 'broadcast_pyobj',
                        lambda packet, *_args, **_kwargs: packets.append(packet) or packet)
    published, quarantined = [], []

    def publish(**kwargs):
        published.append(kwargs)
        descriptors = tuple(SimpleNamespace(
            transfer_id=item.transfer_id, source_slot_id=0, slot_generation=1,
            source_session_id='session', pool_generation='pool',
        ) for item in kwargs['transfers'])
        return SimpleNamespace(
            producer_event=object(), descriptors=descriptors,
            pack_strategy='cat-out', pack_chunk_count=len(descriptors),
            descriptor_published_ns=1,
        )

    controller._feature_source_pool = SimpleNamespace(
        publish=publish, quarantine=lambda **kw: quarantined.append(kw)
    )
    hidden, candidates, result = wave()
    bonus = freeze(hidden, candidates, result)
    handoff = DFlashDraftInputV2.create_idle_input(torch.device('cpu'))
    handoff.bonus_tokens = bonus
    handoff.new_seq_lens = result.new_seq_lens
    handoff.dt_feature_offsets = (10, 20)
    result.next_draft_input = handoff
    batch = SimpleNamespace(reqs=[SimpleNamespace(
        rid=rid, finished=lambda: False, is_retracted=False
    ) for rid in ('a', 'b')])
    return controller, batch, result, packets, published, quarantined


@pytest.mark.parametrize('retired', [None, 'finished', 'cancelled'])
def test_deferred_two_stream_parity_and_finished_credit(monkeypatch, retired):
    c, batch, result, packets, published, _ = setup_controller(monkeypatch)
    c.stage_overlap_pivot_copy(batch, result)
    done = result.dt_deferred_feature_copy_done
    assert not c.has_feature_source_capacity()
    assert not published and not done.waited
    if retired == 'finished':
        batch.reqs[0].finished = lambda: True
    elif retired == 'cancelled':
        batch.reqs[0].is_retracted = True
    c._stage_dflash_feature_copy(batch, result, finalize_deferred=True)
    tasks = result.dt_staged_dflash_features.tasks
    assert done.waited
    assert len(tasks) == (2 if retired is None else 1)
    if retired is None:
        assert tasks[0].feature_token_ids == (11, 99)
        assert tasks[0].committed_token_ids == (10, 11)
        assert tasks[0].previous_transfer_id == 'old-a'
    assert tasks[-1].feature_token_ids == (21, 22, 23, 88)
    assert tasks[-1].committed_token_ids == (20, 21, 22, 23)
    assert c._deferred_dflash_feature_stages == 0
    assert result.dt_dflash_packed_hidden is None
    assert result.next_draft_input.dt_feature_chunks is None
    # Finalize is idempotent after ownership moves into the staged source lease.
    c._stage_dflash_feature_copy(batch, result, finalize_deferred=True)
    assert len(published) == 1 and len(packets) == 1


@pytest.mark.parametrize('failure', ['length', 'schema', 'fencing', 'd2h_oom', 'publish_oom', 'post_publish'])
def test_errors_broadcast_before_credit_release(monkeypatch, failure):
    c, batch, result, packets, published, quarantined = setup_controller(monkeypatch)
    if failure == 'd2h_oom':
        def oom(_value):
            raise torch.cuda.OutOfMemoryError('injected pinned allocation failure')
        monkeypatch.setattr(result_utils, '_async_d2h', oom)
    c.stage_overlap_pivot_copy(batch, result)
    if failure == 'length':
        result.dt_dflash_accept_lens_cpu[0] = 5
    elif failure == 'schema':
        result.dt_eagle3_committed_tokens_cpu = torch.zeros((2, 3), dtype=torch.int64)
    elif failure == 'fencing':
        c._lifecycle_attempt_by_request['a'] += 1
    elif failure == 'publish_oom':
        def oom(**_kwargs):
            raise torch.cuda.OutOfMemoryError('injected pool publish failure')
        c._feature_source_pool.publish = oom
    elif failure == 'post_publish':
        def bad_wait(_event):
            raise RuntimeError('injected post publication failure')
        c.scheduler.forward_stream.wait_event = bad_wait
    with pytest.raises(RuntimeError, match='feature publication failed'):
        c._stage_dflash_feature_copy(batch, result, finalize_deferred=True)
    assert len(packets) == 1 and packets[0][0] is None
    assert c._deferred_dflash_feature_stages == 0
    assert result.dt_dflash_packed_hidden is None
    assert len(quarantined) == (1 if failure == 'post_publish' else 0)
    assert not c._source_slot_by_transfer
    # TP follower receives the same failure, then releases its own credit.
    c.is_export_rank = False
    c._deferred_dflash_feature_stages = 1
    result.dt_dflash_feature_stage_credit = True
    monkeypatch.setattr(target, 'broadcast_pyobj', lambda *_args, **_kw: packets[0])
    with pytest.raises(RuntimeError, match='feature publication failed'):
        c._stage_dflash_feature_copy(batch, result, finalize_deferred=True)
    assert c._deferred_dflash_feature_stages == 0


def test_export_clone_oom_becomes_ordered_result_failure():
    hidden, candidates, result = wave()
    class FailClone:
        def clone(self):
            raise torch.cuda.OutOfMemoryError('injected hidden clone failure')
    freeze(FailClone(), candidates, result)
    assert 'OutOfMemoryError' in result.dt_deferred_feature_error
    assert result.dt_dflash_packed_hidden is None
    assert result.dt_deferred_eagle3_feature_stage


def test_empty_result_retires_credit_without_publication(monkeypatch):
    c, batch, result, packets, published, _ = setup_controller(monkeypatch)
    c.stage_overlap_pivot_copy(batch, result)
    batch.reqs = []
    c._park_completed_batch(batch, batch_result=result)
    assert c._deferred_dflash_feature_stages == 0
    assert result.dt_dflash_packed_hidden is None
    assert not published and not packets
