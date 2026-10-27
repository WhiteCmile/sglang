"""CPU contracts for DT_RING_PUBLISH_AT_LAUNCH (Q15b).

CUDA/publisher doubles do not establish GPU parity; they pin the ownership
and fallback bookkeeping around the launch-time ring publish: the in-place
lease is consumed exactly once, the launch never touches the candidate arena
(the cleanup reserves destinations as on the park path), pre-kernel failures
restore the park path verbatim, post-kernel failures fail closed, and rows
ending in the wave stay in the record for the tombstone/cancel path.

Run only after the task's external release/gate; imports can load heavy modules.
"""

import struct
from types import SimpleNamespace

import msgspec
import numpy as np
import torch

import sglang.srt.speculative.dt_direct_ring as direct_ring
import sglang.srt.speculative.dt_target as target
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dt_direct_ring import (
    ENTRY_BYTES,
    V2_HEADER_BYTES,
    V2_REQ_ENTRY_BYTES,
    V2_TOKEN_OFFSET,
)


class DeviceRows(torch.Tensor):
    """CPU storage with only the controller's device boundary replaced."""

    @property
    def device(self):
        return torch.device("cuda:0")

    @property
    def is_cuda(self):
        return True


def rows(value):
    return value.as_subclass(DeviceRows)


class FakePublisher:
    def __init__(self):
        self.writer = SimpleNamespace(
            next_seq=41,
            capacity=64,
            max_entry_bytes=1 << 20,
            write_destinations=lambda *args, **kwargs: self.calls.append(
                ("destinations", args, kwargs)
            ),
        )
        self.calls = []

    def free_entries(self):
        return 3

    def rid_hash_for(self, rid):
        return 0xA5 ^ len(rid)

    def next_launch_slot(self):
        return 1

    def stage_requests(self, entries, *, staging_slot=0):
        self.calls.append(("stage", list(entries), staging_slot))

    def publish(self, **kwargs):
        self.calls.append(("publish", kwargs))
        seq = self.writer.next_seq
        self.writer.next_seq = seq + 1
        return seq


class FakeArena:
    def __init__(self, fail_at=None, free_slots=256):
        self.reserved = []
        self.cancelled = []
        self.fail_at = fail_at
        self.fail_exc = RuntimeError("arena full")
        self.free_slots = free_slots
        self._data_buffer = SimpleNamespace(allocation_offset=4096)
        self._metadata_buffer = SimpleNamespace(allocation_offset=8192)

    @property
    def free_slot_count(self):
        return self.free_slots

    def reserve(self, transfer_id):
        if self.fail_at is not None and len(self.reserved) >= self.fail_at:
            raise self.fail_exc
        self.reserved.append(transfer_id)
        return SimpleNamespace(
            slot_id=len(self.reserved), generation=1, authorization_nonce=7
        )

    def cancel(self, transfer_id, *, remote_quiescent=False):
        self.cancelled.append((transfer_id, remote_quiescent))


def make_controller():
    c = object.__new__(target.DTTargetController)
    c.scheduler = SimpleNamespace(
        forward_stream=SimpleNamespace(cuda_stream=0xBEEF, wait_event=lambda _e: None)
    )
    c.is_export_rank = True
    c.is_eagle3 = False
    c.is_persistent = True
    c.candidate_transport = "cuda-ipc"
    c._direct_ring = True
    c._ring_publish = True
    c._ring_publish_at_launch = True
    c._ledger = object()
    c._feature_source_pool = SimpleNamespace(
        _buffer=SimpleNamespace(allocation_offset=1 << 20),
        _slot_nbytes=6,
        quarantine_reserved=lambda *args: setattr(c, "_quarantined", args),
    )
    c._quarantined = None
    c.instance_id = "target"
    c.block_size = 4
    c.pivot_width = 3
    c.feature_schema_id = "dflash"
    c._next_round = {"a": 1, "b": 1}
    c._draft_committed_len = {"a": 10, "b": 20}
    c._draft_kv_handle_by_request = {"a": "kv-a", "b": "kv-b"}
    c._verify_transfer_by_request = {"a": "old-a", "b": "old-b"}
    c._lifecycle_attempt_by_request = {"a": 5, "b": 7}
    c._next_lifecycle_attempt = 8
    c._draft_binding = {"a": ("di", "session", 9), "b": ("di", "session", 9)}
    c._target_inflight_counts = {}
    c._inplace_hidden = {}
    c._slot_event_handles = {}
    c._ring_publish_pending = target.deque()
    c._ring_publish_orphaned_events = target.deque(maxlen=1024)
    c._cancelled_transfers = set()
    c._profile_active = False
    c._profile_emit = lambda *_args, **_kwargs: None
    return c


def make_wave(c):
    publisher = FakePublisher()
    c._ring_publish_publisher = lambda _session: publisher
    arena = FakeArena()
    c._candidate_arena = arena
    ready_event = SimpleNamespace(ipc_handle=lambda: b"h" * 64)
    packed = rows(torch.zeros((8, 3), dtype=torch.bfloat16))
    candidates = rows(torch.zeros((2, 4), dtype=torch.int64))
    c._inplace_hidden[int(packed.data_ptr())] = (3, 11, ready_event)
    draft_input = DFlashDraftInputV2.create_idle_input(torch.device("cpu"))
    draft_input.bonus_tokens = rows(torch.tensor([99, 88]))
    draft_input.new_seq_lens = rows(torch.tensor([12, 24]))
    draft_input.dt_feature_offsets = (10, 20)
    result = SimpleNamespace(
        next_draft_input=draft_input,
        accept_lens=rows(torch.tensor([2, 4], dtype=torch.int32)),
        dt_dflash_candidate_tokens=candidates,
        dt_dflash_packed_hidden=packed,
        dt_deferred_feature_error=None,
        dt_deferred_prefill_feature_stage=False,
    )
    batch = SimpleNamespace(
        reqs=[
            SimpleNamespace(
                rid=rid, finished=lambda: False, is_retracted=False,
                inflight_middle_chunks=0,
            )
            for rid in ("a", "b")
        ],
        chunked_req=None,
        dt_full_wave_id=5,
        dt_pipeline_lane_id=1,
    )
    return c, batch, result, publisher, arena, packed


def test_launch_publish_stages_record_and_finalize():
    c, batch, result, publisher, arena, packed = make_wave(make_controller())
    c._stage_dflash_ring_publish_at_launch(batch, result)
    launch = result.dt_ring_publish_launch
    assert result.dt_ring_publish_launch_error is None
    assert launch is not None and launch.wave_seq == 41
    assert [item[0] for item in launch.staged_meta] == [0, 1]
    # The launch never touches the candidate arena; the cleanup reserves
    # and writes destinations exactly as on the park path.
    assert not arena.reserved and not arena.cancelled
    # The in-place lease is consumed by the launch staging, not by park.
    assert not c._inplace_hidden
    order = [call[0] for call in publisher.calls]
    assert order == ["stage", "publish"]
    publish = publisher.calls[1][1]
    assert publish["stream"] == 0xBEEF  # the verify stream
    assert publish["offset_stride"] == c.block_size  # in-place layout
    assert publish["row_count"] == 8
    assert publish["token_total"] == 2 * c.block_size  # conservative bound
    assert publish["accept_dev"] == int(result.accept_lens.data_ptr())
    assert publish["staging_slot"] == 1
    # A row ending in this wave stays in the record; park's cancel path
    # retires it.  The finalize specs must keep the launch-time table.
    batch.reqs[0].finished = lambda: True
    specs = c._dflash_commit_specs(
        list(launch.staged_meta),
        result.accept_lens.numpy(),
        result.next_draft_input.bonus_tokens.numpy(),
        result.next_draft_input.new_seq_lens.numpy(),
    )
    assert [spec[1] for spec in specs] == ["target:a:1:5", "target:b:1:7"]
    assert specs[0][6] == 2 and specs[1][6] == 4
    assert specs[0][9] == "old-a"
    assert c._finalize_dflash_ring_publish_at_launch(batch, result, launch, specs)
    assert len(c._ring_publish_pending) == 1
    wave = c._ring_publish_pending[0]
    assert wave.wave_seq == 41 and wave.slot_id == 3 and wave.generation == 11
    assert result.dt_ring_publish_wave is wave


def test_launch_never_reserves_arena():
    c, batch, result, publisher, arena, packed = make_wave(make_controller())
    arena.fail_at = 0  # any reservation would fail
    c._stage_dflash_ring_publish_at_launch(batch, result)
    # The publish does not consult the arena at launch: no fallback, no
    # reservation attempt, the kernel goes out.
    assert result.dt_ring_publish_launch is not None
    assert result.dt_ring_publish_launch_error is None
    assert not arena.reserved and not arena.cancelled
    assert [call[0] for call in publisher.calls] == ["stage", "publish"]


def test_launch_kernel_failure_restores_lease():
    c, batch, result, publisher, arena, packed = make_wave(make_controller())

    def bad_stage(entries, *, staging_slot=0):
        raise RuntimeError("injected staging failure")

    publisher.stage_requests = bad_stage
    c._stage_dflash_ring_publish_at_launch(batch, result)
    assert result.dt_ring_publish_launch is None
    assert result.dt_ring_publish_launch_error is None
    assert not arena.reserved and not arena.cancelled
    assert c._inplace_hidden[int(packed.data_ptr())][0] == 3


def test_launch_post_kernel_failure_fails_closed():
    c, batch, result, publisher, arena, packed = make_wave(make_controller())

    def bad_publish(**_kwargs):
        raise RuntimeError("injected launch failure")

    publisher.publish = bad_publish
    c._stage_dflash_ring_publish_at_launch(batch, result)
    assert result.dt_ring_publish_launch is None
    assert result.dt_ring_publish_launch_error.startswith("RingPublishCommittedError")
    # The record may be out: the slot is quarantined, the event is retained,
    # and the lease record is gone so park cannot free a quarantined slot.
    assert c._quarantined == (3, 11)
    assert not c._inplace_hidden
    assert len(c._ring_publish_orphaned_events) == 1


def test_launch_ineligible_without_inplace_lease_or_binding():
    c, batch, result, publisher, arena, packed = make_wave(make_controller())
    c._inplace_hidden.clear()
    c._stage_dflash_ring_publish_at_launch(batch, result)
    assert result.dt_ring_publish_launch is None
    assert not publisher.calls

    c, batch, result, publisher, arena, packed = make_wave(make_controller())
    del c._draft_binding["b"]
    c._stage_dflash_ring_publish_at_launch(batch, result)
    assert result.dt_ring_publish_launch is None
    assert not publisher.calls
    assert c._inplace_hidden  # lease untouched for the park path


def _pack_record(buf, base, *, seq, row_count, reqs, tokens):
    struct.pack_into("<Q", buf, base, seq)
    struct.pack_into("<I", buf, base + 48, row_count)
    struct.pack_into("<I", buf, base + 56, len(reqs))
    struct.pack_into("<Q", buf, base + 64, 123456)
    for i, rec in enumerate(reqs):
        struct.pack_into("<Q6I", buf, base + V2_HEADER_BYTES + i * V2_REQ_ENTRY_BYTES, *rec)
    struct.pack_into(f"<{len(tokens)}I", buf, base + V2_TOKEN_OFFSET, *tokens)


def _setup_cleanup_doubles(c):
    """Ring record for wave 41 plus the pool/task doubles cleanup consumes."""

    raw = np.zeros(ENTRY_BYTES * 64, dtype=np.uint8)
    base = 41 % 64 * ENTRY_BYTES
    reqs = [
        (0xA5 ^ 1, 1, 5, 0, 2, 99, 12),
        (0xA5 ^ 1, 1, 7, 4, 4, 88, 24),
    ]
    _pack_record(raw, base, seq=41, row_count=8, reqs=reqs, tokens=(10, 11, 20, 21, 22, 23))
    c._ring_writers = {
        "session": SimpleNamespace(
            _ring=SimpleNamespace(entries_offset=0, raw=raw),
            capacity=64,
            write_destinations=lambda *args, **kwargs: None,
        )
    }
    published = []

    def publish_reserved(**kwargs):
        published.append(kwargs)
        return SimpleNamespace(
            descriptors=[
                SimpleNamespace(transfer_id=item.transfer_id)
                for item in kwargs["transfers"]
            ]
        )

    c._feature_source_pool.publish_reserved = publish_reserved
    c._feature_source_pool.cancel = lambda **_kwargs: None
    c._feature_source_pool.source_session_id = "session"
    c._feature_source_pool.pool_generation = "pool"
    c._source_slot_by_transfer = {}
    c._ring_publish_transfers = set()
    c._ring_unproven = {}
    c._lease_marks = []
    c._direct_sent = set()
    c._ring_outstanding = {"session": target.deque()}

    class Task(msgspec.Struct):
        transfer_id: str
        direct_ring_seq: int = 0
        candidate_destination: object = None

    made = []
    c._make_task = lambda **kwargs: Task(transfer_id=kwargs["transfer_id"])
    c._queue_task_batch = lambda tasks: made.extend(tasks)
    c._flush_pending_send = lambda: None
    return published, made


def _finalize_launch_wave(c, batch, result):
    launch = result.dt_ring_publish_launch
    specs = c._dflash_commit_specs(
        list(launch.staged_meta),
        result.accept_lens.numpy(),
        result.next_draft_input.bonus_tokens.numpy(),
        result.next_draft_input.new_seq_lens.numpy(),
    )
    c._finalize_dflash_ring_publish_at_launch(batch, result, launch, specs)
    return launch, specs


def test_cleanup_reserves_destinations_and_cancels():
    c, batch, result, publisher, arena, packed = make_wave(make_controller())
    c._stage_dflash_ring_publish_at_launch(batch, result)
    _finalize_launch_wave(c, batch, result)
    # The first row ended in the wave: park's finished branch cancelled it.
    c._cancelled_transfers.add("target:a:1:5")
    published, made = _setup_cleanup_doubles(c)
    c._ring_publish_cleanup()
    # The cleanup reserves only the live row (the launch reserved nothing);
    # the cancelled row holds no destination, so its source-share cancel is
    # the whole retirement, and the marker itself is removed (quiescence
    # must not leak it).
    assert arena.reserved == ["target:b:1:7"]
    assert not arena.cancelled
    assert "target:a:1:5" not in c._cancelled_transfers
    assert len(made) == 1
    assert made[0].transfer_id == "target:b:1:7"
    assert made[0].direct_ring_seq == 41
    assert made[0].candidate_destination.slot_id == 1
    assert len(published) == 1
    assert not c._ring_publish_pending


def test_cleanup_defers_when_arena_is_full():
    from sglang.srt.speculative.dt_candidate_p2p import CandidateTransportError

    c, batch, result, publisher, arena, packed = make_wave(make_controller())
    c._stage_dflash_ring_publish_at_launch(batch, result)
    launch = result.dt_ring_publish_launch
    specs = c._dflash_commit_specs(
        list(launch.staged_meta),
        result.accept_lens.numpy(),
        result.next_draft_input.bonus_tokens.numpy(),
        result.next_draft_input.new_seq_lens.numpy(),
    )
    c._finalize_dflash_ring_publish_at_launch(batch, result, launch, specs)
    _setup_cleanup_doubles(c)
    arena.fail_at = 1  # first reservation lands, the second sees a full arena
    arena.fail_exc = CandidateTransportError("all candidate arena slots are active")
    # A full arena is backpressure: the wave stays pending, the partial
    # reservation is cancelled, and nothing raises.
    c._ring_publish_cleanup()
    assert len(c._ring_publish_pending) == 1
    assert arena.cancelled == [("target:a:1:5", True)]
    arena.fail_at = None
    c._ring_publish_cleanup()
    assert not c._ring_publish_pending


def test_cleanup_quarantines_wave_when_writer_gone():
    c, batch, result, publisher, arena, packed = make_wave(make_controller())
    c._stage_dflash_ring_publish_at_launch(batch, result)
    _finalize_launch_wave(c, batch, result)
    quarantined = []
    c._feature_source_pool.quarantine_reserved = lambda *args: quarantined.append(args)
    c._ring_writers = {}  # owner lease revoked before the cleanup
    c._ring_publish_cleanup()
    assert quarantined == [(3, 11)]
    # Nothing was ever reserved for this wave, so there is nothing to retire.
    assert not arena.cancelled
    assert not c._ring_publish_pending


def test_publisher_launch_slot_rotation(monkeypatch):
    monkeypatch.setattr(direct_ring, "RING_PUBLISH_AT_LAUNCH", True)
    max_batch_slots = 8
    publisher = object.__new__(direct_ring.RingGpuPublisher)
    publisher.block_size = 4
    publisher.row_bytes = 6
    publisher.max_batch_slots = max_batch_slots
    publisher._active_offset = max_batch_slots * direct_ring.V2_STAGING_ENTRY_BYTES
    publisher._counts_offset = publisher._active_offset + 4 * direct_ring.V2_MAX_REQ
    nbytes = publisher._counts_offset + 4 * max_batch_slots
    publisher._launch_slot_stride = (
        direct_ring.V2_MAX_REQ * direct_ring.V2_STAGING_ENTRY_BYTES
        + 4 * direct_ring.V2_MAX_REQ
    )
    publisher._launch_entries_base = nbytes
    publisher._launch_active_base = publisher._launch_entries_base + (
        direct_ring.LAUNCH_STAGING_SLOTS * publisher._launch_slot_stride
    )
    nbytes = publisher._launch_active_base + (
        direct_ring.LAUNCH_STAGING_SLOTS * 4 * direct_ring.V2_MAX_REQ
    )
    publisher._launch_slot_cycle = -1
    publisher._entries_dev = 0
    publisher._staging_dev = 0
    buf = np.zeros(nbytes, dtype=np.uint8)
    publisher._stg_u8 = buf
    publisher._stg_u64 = buf.view(np.uint64)
    publisher._stg_u32 = buf.view(np.uint32)
    publisher._hash_by_rid = {}
    launched = []
    publisher._ext = SimpleNamespace(
        publish=lambda **kwargs: launched.append(kwargs)
    )
    publisher._pool_tag = 0
    header = np.zeros(4096 // 4 + 8, dtype=np.uint32)
    publisher._writer = SimpleNamespace(
        state=direct_ring.STATE_SERVING,
        capacity=64,
        next_seq=41,
        max_entry_bytes=1 << 20,
        free_entries=lambda: 3,
        _ring=SimpleNamespace(
            header=header, write_entry_bytes=lambda *_args: None
        ),
    )
    # The rotation cycles through disjoint staging regions, so a queued
    # kernel never reads entries a later wave's host staging overwrote.
    offsets = set()
    for iteration, expected_slot in enumerate((1, 2, 3, 4, 5, 6, 7, 8, 1)):
        slot = publisher.next_launch_slot()
        assert slot == expected_slot
        entries_offset, active_offset = publisher._slot_offsets(slot)
        if iteration < direct_ring.LAUNCH_STAGING_SLOTS:
            assert entries_offset not in offsets
            offsets.add(entries_offset)
        publisher.stage_requests([(7, 1, 5, 9, 0xDEAD)], staging_slot=slot)
        seq = publisher.publish(
            stream=0xBEEF,
            active_slots=[2],
            slot_id=3,
            generation=11,
            source_byte_offset=4096,
            nbytes=48,
            row_count=8,
            event_handle=b"h" * 64,
            accept_dev=1,
            bonus_dev=2,
            seq_lens_dev=3,
            offset_stride=4,
            token_total=4,
            staging_slot=slot,
        )
        assert seq == 41 + iteration
        call = launched[-1]
        assert call["staging_entries_offset"] == entries_offset
        assert call["staging_active_offset"] == active_offset
        assert buf.view(np.uint32)[active_offset // 4] == 2
    assert int(header[(direct_ring.GENERATION_TABLE_OFFSET >> 2) + 3]) == 11
    # Slot 0 keeps the park path's legacy layout.
    assert publisher._slot_offsets(0) == (0, publisher._active_offset)



def test_tombstoned_ring_push_row_is_retired_draft_side():
    """A row ending in a launch-published wave gets a zero destination triple
    from the Target's cleanup, and no Coordinator confirmation copy can ever
    route for it (the Coordinator only routes cancellations for tasks it
    dispatched).  The Draft must retire the row from the ring tombstone, or
    the wave's held candidate push waits forever (the issue-39 warmup wedge).
    """
    import sglang.srt.speculative.dt_draft_engine as draft_mod

    engine = object.__new__(draft_mod.FlashMTPDraftEngine)
    engine._cancelled_ids = set()
    engine._unconfirmed_direct = {
        "target:t:live:1:0": 1,
        "target:t:zombie:1:0": 1,
        "target:t:pending:1:0": 1,
    }
    live_endpoint = SimpleNamespace(
        destination=lambda seq, position: (4, 9, 123, 4096, 2048)
    )
    dead_endpoint = SimpleNamespace(destination=lambda seq, position: (0, 0, 0, 0, 0))
    stale_endpoint = SimpleNamespace(destination=lambda seq, position: None)
    engine._ring_record_positions = {
        "target:t:live:1:0": (live_endpoint, 0),
        "target:t:zombie:1:0": (dead_endpoint, 1),
        "target:t:pending:1:0": (stale_endpoint, 2),
    }

    def _task(transfer_id):
        return SimpleNamespace(
            transfer_id=transfer_id, candidate_destination=None, direct_ring_seq=7
        )

    tasks = [_task("target:t:live:1:0"), _task("target:t:zombie:1:0"), _task("target:t:pending:1:0")]
    engine._held_ring_pushes = [(tasks, b"candidates", 3)]
    releases = []
    engine._release_held_ring_pushes = lambda: releases.append(True)

    engine._retire_tombstoned_ring_pushes()

    assert "target:t:zombie:1:0" in engine._cancelled_ids
    assert "target:t:live:1:0" not in engine._cancelled_ids
    assert "target:t:pending:1:0" not in engine._cancelled_ids
    assert releases == [True]

    # Already-retired rows are not retired twice.
    releases.clear()
    engine._retire_tombstoned_ring_pushes()
    assert releases == []


def test_cleanup_writes_zero_destinations_for_all_cancelled_wave():
    """An all-cancelled wave still gets its destination region written (zero
    triples).  Without it the region stays stale and the Draft cannot retire
    its held push — the all-cancelled-wave form of the issue-39 wedge."""
    c, batch, result, publisher, arena, packed = make_wave(make_controller())
    c._stage_dflash_ring_publish_at_launch(batch, result)
    _finalize_launch_wave(c, batch, result)
    # Both rows ended in the wave: park's finished branch cancelled them.
    c._cancelled_transfers.add("target:a:1:5")
    c._cancelled_transfers.add("target:b:1:7")
    writes = []
    published, made = _setup_cleanup_doubles(c)
    c._ring_writers["session"].write_destinations = (
        lambda seq, entries, **kwargs: writes.append((seq, entries))
    )
    c._ring_publish_cleanup()
    assert not arena.reserved
    assert not made
    assert len(writes) == 1
    seq, entries = writes[0]
    assert seq == 41
    assert len(entries) == 2
    assert all(entry == (0, 0, 0) for entry in entries)
