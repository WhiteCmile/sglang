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
        self.authorized = []
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

    def authorize(self, assignment):
        self.authorized.append(assignment)

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
    c._target_dest_at_launch = False
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
    c._ring_publish_backpressure = set()
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
    assert launch is not None
    # One owner session, one geometry chunk: one record naming both rows.
    assert launch.records == (("session", 41, (0, 1), 8),)
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
    assert len(wave.records) == 1
    assert wave.records[0].wave_seq == 41
    assert len(wave.records[0].specs) == 2
    assert wave.slot_id == 3 and wave.generation == 11
    assert result.dt_ring_publish_wave is wave


def test_launch_multi_owner_wave_splits_one_record_per_owner():
    """P1: a wave mixing two owner Draft sessions publishes one record per
    owner at launch (the park path's split shape) instead of falling back;
    each record stages into its own rotating launch slot and the finalize
    splits the commit specs by the records' staged positions."""
    c, batch, result, publisher, arena, packed = make_wave(make_controller())
    other = FakePublisher()
    publishers = {"session": publisher, "session-b": other}
    c._ring_publish_publisher = lambda session: publishers.get(session)
    c._draft_binding = {"a": ("di", "session", 9), "b": ("di", "session-b", 9)}
    c._stage_dflash_ring_publish_at_launch(batch, result)
    launch = result.dt_ring_publish_launch
    assert result.dt_ring_publish_launch_error is None
    assert launch is not None
    # Row a names rows [0, 4) of the slot, row b rows [0, 8): each record's
    # window reaches from the slot base through its own last strided row.
    assert launch.records == (
        ("session", 41, (0,), 4),
        ("session-b", 41, (1,), 8),
    )
    pa = publisher.calls[0]
    pb = other.calls[0]
    assert pa[0] == "stage" and pb[0] == "stage"
    # Each record stages into its own publisher's launch-slot rotation.
    assert pa[2] == 1 and pb[2] == 1
    assert publisher.calls[1][1]["row_count"] == 4
    assert other.calls[1][1]["row_count"] == 8
    assert not c._inplace_hidden  # the lease is consumed exactly once
    specs = c._dflash_commit_specs(
        list(launch.staged_meta),
        result.accept_lens.numpy(),
        result.next_draft_input.bonus_tokens.numpy(),
        result.next_draft_input.new_seq_lens.numpy(),
    )
    assert c._finalize_dflash_ring_publish_at_launch(batch, result, launch, specs)
    wave = c._ring_publish_pending[0]
    assert [r.session_id for r in wave.records] == ["session", "session-b"]
    assert [r.specs[0][1] for r in wave.records] == ["target:a:1:5", "target:b:1:7"]
    assert all(r.row_start == 0 for r in wave.records)


def test_launch_geometry_chunk_shares_staging_rotation():
    """P1: a wave past one record's geometry publishes multiple chunks on the
    same owner ring; every chunk gets its own rotating staging slot and the
    ring credit claim covers all chunks up front."""
    import sglang.srt.speculative.dt_direct_ring as direct_ring

    c, batch, result, publisher, arena, packed = make_wave(make_controller())
    slots = iter((1, 2))
    publisher.next_launch_slot = lambda: next(slots)
    original = direct_ring.V2_MAX_REQ
    direct_ring.V2_MAX_REQ = 1  # force one request per chunk
    try:
        c._stage_dflash_ring_publish_at_launch(batch, result)
    finally:
        direct_ring.V2_MAX_REQ = original
    launch = result.dt_ring_publish_launch
    assert launch is not None
    assert launch.records == (("session", 41, (0,), 4), ("session", 42, (1,), 8))
    stages = [call for call in publisher.calls if call[0] == "stage"]
    publishes = [call for call in publisher.calls if call[0] == "publish"]
    assert [call[2] for call in stages] == [1, 2]
    assert len(publishes) == 2


def test_launch_second_chunk_direct_error_fails_closed():
    """A DirectRingError after an earlier chunk's kernel went out is a
    partial wave on the ring: fail closed (quarantine + committed error),
    never restore the lease for a park re-publish."""
    from sglang.srt.speculative.dt_direct_ring import DirectRingError

    c, batch, result, publisher, arena, packed = make_wave(make_controller())
    other = FakePublisher()

    def not_serving(**_kwargs):
        raise DirectRingError("ring is not serving")

    other.publish = not_serving
    publishers = {"session": publisher, "session-b": other}
    c._ring_publish_publisher = lambda session: publishers.get(session)
    c._draft_binding = {"a": ("di", "session", 9), "b": ("di", "session-b", 9)}
    c._stage_dflash_ring_publish_at_launch(batch, result)
    assert result.dt_ring_publish_launch is None
    assert result.dt_ring_publish_launch_error.startswith("RingPublishCommittedError")
    assert c._quarantined == (3, 11)
    assert not c._inplace_hidden
    assert len(c._ring_publish_orphaned_events) == 1


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


def test_launch_pre_kernel_failure_falls_back_to_park():
    from sglang.srt.speculative.dt_direct_ring import DirectRingError

    c, batch, result, publisher, arena, packed = make_wave(make_controller())

    def not_serving(**_kwargs):
        raise DirectRingError("ring is not serving")

    publisher.publish = not_serving
    c._stage_dflash_ring_publish_at_launch(batch, result)
    # publish() raised from its pre-launch validation: the kernel never went
    # out and no record can appear.  The wave must fall back to the park
    # path with the in-place lease restored — not quarantine a healthy slot
    # and fail the wave closed over an owner-revocation race.
    assert result.dt_ring_publish_launch is None
    assert result.dt_ring_publish_launch_error is None
    assert c._quarantined is None
    assert not c._ring_publish_orphaned_events
    assert c._inplace_hidden[int(packed.data_ptr())][0] == 3
    assert not arena.reserved and not arena.cancelled


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
    published, made = _setup_cleanup_doubles(c)
    writes = []
    c._ring_writers["session"].write_destinations = (
        lambda seq, entries, **kwargs: writes.append((seq, entries))
    )
    arena.fail_at = 1  # first reservation lands, the second sees a full arena
    arena.fail_exc = CandidateTransportError("all candidate arena slots are active")
    # A full arena is backpressure, not kill the scheduler: the wave stays
    # pending, the partial reservation is cancelled, nothing raises, and the
    # session's ring publication is backpressured so no new record can reuse
    # this wave's slot while it is retained.
    c._ring_publish_cleanup()
    assert len(c._ring_publish_pending) == 1
    assert c._ring_publish_backpressure == {"session"}
    assert arena.cancelled == [("target:a:1:5", True)]
    assert not writes
    # The Draft consumes records independently and the producer wraps after
    # `capacity` further publications: overwrite the slot with a newer
    # sequence.  The deferred wave must still be processed from its record
    # snapshot; waiting for the old seq word to reappear would strand it
    # (and its source reservation and parked requests) forever.
    raw = c._ring_writers["session"]._ring.raw
    _pack_record(
        raw,
        41 % 64 * ENTRY_BYTES,
        seq=41 + 64,
        row_count=8,
        reqs=[(0xA5 ^ 1, 1, 5, 0, 2, 99, 12), (0xA5 ^ 1, 1, 7, 4, 4, 88, 24)],
        tokens=(10, 11, 20, 21, 22, 23),
    )
    arena.fail_at = None
    c._ring_publish_cleanup()
    assert not c._ring_publish_pending
    assert c._ring_publish_backpressure == set()
    # The retry reserves both rows from the snapshot (the first call's
    # partial reservation stayed recorded on the double even though it was
    # cancelled) and builds the Coordinator confirmation copies as usual.
    assert arena.reserved[-2:] == ["target:a:1:5", "target:b:1:7"]
    assert len(made) == 2
    # Generation ownership: the destination slot now shows the newer lap, so
    # the retried write must NOT publish the old seq into it — that would
    # clobber the newer wave's destinations.  The Coordinator copies built
    # above already carry these rows' destinations.
    assert not writes


def test_cleanup_drops_wave_when_record_slot_wrapped_before_snapshot():
    """If the ring wrapped before the wave's first cleanup sighting, the slot
    shows a newer lap and the record is unrecoverable from the ring.  The
    wave must be dropped (slot quarantined), not retained to poll a seq word
    that can never reappear; its parked rounds fail at their own deadlines.
    With the publication backpressure active this requires a full ring of
    cleanup lag."""
    c, batch, result, publisher, arena, packed = make_wave(make_controller())
    c._stage_dflash_ring_publish_at_launch(batch, result)
    _finalize_launch_wave(c, batch, result)
    _setup_cleanup_doubles(c)
    raw = c._ring_writers["session"]._ring.raw
    _pack_record(
        raw,
        41 % 64 * ENTRY_BYTES,
        seq=41 + 64,
        row_count=8,
        reqs=[(0xA5 ^ 1, 1, 5, 0, 2, 99, 12), (0xA5 ^ 1, 1, 7, 4, 4, 88, 24)],
        tokens=(10, 11, 20, 21, 22, 23),
    )
    c._ring_publish_cleanup()
    assert not c._ring_publish_pending
    assert c._quarantined == (3, 11)
    assert not c._ring_publish_backpressure


def test_launch_falls_back_when_ring_backpressured():
    c, batch, result, publisher, arena, packed = make_wave(make_controller())
    # Another wave of this Draft session is arena-retained: no new record may
    # publish on its ring until the retained cleanup lands.
    c._ring_publish_backpressure = {"session"}
    c._stage_dflash_ring_publish_at_launch(batch, result)
    assert result.dt_ring_publish_launch is None
    assert result.dt_ring_publish_launch_error is None
    assert not publisher.calls
    assert c._inplace_hidden  # lease untouched for the park path


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


def test_tombstoned_ring_push_row_is_retired_draft_side():
    """A row ending in a launch-published wave gets a zero destination triple
    from the Target's cleanup, and no Coordinator confirmation copy can ever
    route for it (the Coordinator only routes cancellations for tasks it
    dispatched).  The Draft must retire the row from the ring tombstone, or
    the wave's held candidate push waits forever (the issue-39 warmup wedge).
    The real release path runs (no mocks) so the test proves the terminal
    bookkeeping, not the scanner's own bookkeeping.
    """
    import sglang.srt.speculative.dt_draft_engine as draft_mod

    tids = ("target:t:live:1:0", "target:t:zombie:1:0", "target:t:pending:1:0")
    engine = object.__new__(draft_mod.FlashMTPDraftEngine)
    engine._cancelled_ids = set()
    engine._unconfirmed_direct = {tid: 1 for tid in tids}
    engine._direct_tasks_by_transfer = {
        tid: SimpleNamespace(
            transfer_id=tid, candidate_destination=None, direct_ring_seq=7
        )
        for tid in tids
    }
    engine._held_direct_source_done = {tid: ["held-source-done"] for tid in tids}
    engine._held_direct_results = {tid: "held-result" for tid in tids}
    engine._pending_ids = set(tids)
    engine._ring_pending_ids = set(tids)
    engine._received_ns_by_transfer = {tid: 1 for tid in tids}
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
    engine._held_ring_pushes = [
        (list(engine._direct_tasks_by_transfer.values()), b"candidates", 3)
    ]
    engine._candidate_push_transport = SimpleNamespace()
    engine._free_result_slots = []

    engine._retire_tombstoned_ring_pushes()

    # The zombie row is marked cancelled; its identity retires when the hold
    # releases (the wave's live rows are still awaiting confirmation), so the
    # wave stays held and the live rows' bookkeeping is untouched.
    zombie = "target:t:zombie:1:0"
    assert zombie in engine._cancelled_ids
    for tid in ("target:t:live:1:0", "target:t:pending:1:0"):
        assert tid in engine._unconfirmed_direct
        assert tid in engine._direct_tasks_by_transfer
    assert len(engine._held_ring_pushes) == 1
    assert engine._free_result_slots == []

    # A second scan is idempotent: nothing is retired twice, the wave stays
    # held while any row is still unconfirmed.
    engine._retire_tombstoned_ring_pushes()
    assert len(engine._held_ring_pushes) == 1
    assert "target:t:live:1:0" in engine._unconfirmed_direct

    # Tombstone the remaining rows too: the hold releases and every row must
    # reach its terminal state through the real release path — cancelled
    # identity, no unconfirmed/direct/source-done/held-result bookkeeping, no
    # record position — and the result slot is freed.  A leak here would
    # accumulate per finished row and weaken the Coordinator queue-capacity
    # bound.
    engine._ring_record_positions["target:t:live:1:0"] = (dead_endpoint, 0)
    engine._ring_record_positions["target:t:pending:1:0"] = (dead_endpoint, 2)
    engine._retire_tombstoned_ring_pushes()
    assert engine._held_ring_pushes == []
    assert engine._free_result_slots == [3]
    assert not engine._unconfirmed_direct
    assert not engine._direct_tasks_by_transfer
    assert not engine._held_direct_source_done
    assert not engine._held_direct_results
    assert not engine._cancelled_ids
    assert not engine._ring_record_positions


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


def _make_record_engine(destination, *, at_launch=True):
    """Engine doubles for _accept_ring_record with a one-row wave at seq 7.

    ``destination`` is what the endpoint's destination region returns for the
    record's seq and position: a live triple, a (0,0,0) tombstone, or None.
    """

    import sglang.srt.speculative.dt_draft_engine as draft_mod
    from sglang.srt.speculative.dt_direct_ring import (
        RingRecordV2,
        RingRecordV2Request,
        rid_hash,
    )

    engine = object.__new__(draft_mod.FlashMTPDraftEngine)
    engine._ring_publish_at_launch = at_launch
    engine._rid_ledger = object()
    engine._ring_source_registration = {0: (b"h" * 64, 1 << 20)}
    engine._ring_targets = {0: "target"}
    engine._landing_slot_base = 0
    engine._landing_taken = [0]
    engine._feature_staging_refcounts = {}
    engine._ring_admission = []
    engine._pending_ids = set()
    engine._ring_abandoned = set()
    engine._cancelled_ids = set()
    engine._seen_transfers = set()
    engine._seen_transfer_order = []
    engine._rid_hash_cache = {}
    engine._received_ns_by_transfer = {}
    engine._direct_tasks_by_transfer = {}
    engine._unconfirmed_direct = {}
    engine._ring_pending_ids = set()
    engine._ring_record_positions = {}
    engine._ring_waiting = {}
    engine._ring_retired = {}
    engine._persistent_requests = {}
    engine._deferred_persistent_releases = {}
    engine.profile = SimpleNamespace(
        profile_id="p",
        draft_k=3,
        block_size=4,
        vocab_size=1000,
        feature_schema=SimpleNamespace(schema_id="schema"),
    )
    engine.candidate_transport = "cuda-ipc"
    engine._is_feature_backed = True
    engine._pivot_width = 4
    engine.feature_buffer_max_tokens = 512
    engine.batch_source_read_done = True
    engine._source_done_pending = []
    engine._held_direct_source_done = {}
    engine._held_direct_results = {}
    engine.instance_id = "draft"
    engine._client = SimpleNamespace(
        instance_session_id="draft-session", coordinator_session_id="coordinator"
    )
    engine._candidate_push_transport = SimpleNamespace(
        _registered={
            "di": (
                SimpleNamespace(
                    target_instance_id="target",
                    target_session_id="session",
                    pool_generation="pool",
                    transport_session="transport",
                    target_device_id=0,
                    device_uuid="uuid",
                    row_width=4,
                    data_allocation_length=1 << 20,
                    metadata_allocation_length=1 << 16,
                    metadata_words=8,
                ),
            )
        }
    )
    engine._ring_dest_region = True
    endpoint = SimpleNamespace(
        ring_id=0,
        target_session_id="session",
        source_session_id="source-session",
        pool_generation="pool",
        destination=lambda seq, position: destination,
    )
    record = RingRecordV2(
        seq=7,
        owner_epoch=0,
        source_slot_id=1,
        slot_generation=1,
        source_byte_offset=0,
        nbytes=8 * 8,
        row_count=8,
        row_bytes=8,
        block_size=4,
        hidden_produced_ns=0,
        requests=(
            RingRecordV2Request(
                rid_hash=rid_hash("req-a"),
                round_id=1,
                attempt=5,
                token_offset=0,
                token_count=2,
                bonus_token=99,
                seq_len=12,
                rid="req-a",
            ),
        ),
        token_ids=(10, 11),
        producer_event_handle=b"h" * 64,
        landing_index=0,
        landing_offset=0,
        dropped_mask=0,
    )
    return engine, endpoint, record, "target:req-a:1:5"


def test_record_take_scans_destination_while_generation_is_observable():
    """The record take is the one moment the ring generation is provably
    current.  A tombstoned row (finished in its wave, no Coordinator task) is
    retired right there, before candidate compute can create the wave's held
    push — by then the ring may have wrapped and the tombstone would be
    unrecoverable.  A live triple is latched onto the task as its arena
    destination; a not-yet-written region (None) leaves the row pending."""

    # Tombstone at take: the row is dropped, never admitted.
    engine, endpoint, record, tid = _make_record_engine((0, 0, 0, 0, 0))
    engine._accept_ring_record(endpoint, record)
    assert engine._pending_ids == set()
    assert engine._ring_admission == []
    assert engine._unconfirmed_direct == {}
    assert engine._direct_tasks_by_transfer == {}
    assert engine._ring_record_positions == {}
    assert engine._landing_taken == [1]  # the wave landing is claimed once

    # Live destination at take: latched onto the admitted task.
    engine, endpoint, record, tid = _make_record_engine((3, 9, 123, 4096, 2048))
    engine._accept_ring_record(endpoint, record)
    assert tid in engine._pending_ids
    destination = engine._direct_tasks_by_transfer[tid].candidate_destination
    assert destination is not None
    assert (destination.slot_id, destination.generation) == (3, 9)
    assert destination.authorization_nonce == 123
    assert destination.data_allocation_offset == 4096 + 3 * 4 * 4

    # Not yet written: the row is admitted pending, exactly as before.
    engine, endpoint, record, tid = _make_record_engine(None)
    engine._accept_ring_record(endpoint, record)
    assert tid in engine._pending_ids
    assert engine._direct_tasks_by_transfer[tid].candidate_destination is None

    # The probe is launch-path only: flag off keeps the park-path behavior.
    engine, endpoint, record, tid = _make_record_engine(
        (0, 0, 0, 0, 0), at_launch=False
    )
    engine._accept_ring_record(endpoint, record)
    assert tid in engine._pending_ids


def test_held_ring_push_expires_after_record_deadline():
    """A held ring push row whose ring destination or tombstone never became
    observable — the ring may have wrapped long ago — must not hold its wave
    forever.  Past the record deadline it retires as cancelled, like a
    tombstone row; the wave holds only while genuinely unconfirmed rows
    remain."""

    import sglang.srt.speculative.dt_draft_engine as draft_mod

    future = __import__("time").monotonic_ns() + 60 * 10**9
    tids = ("target:t:live:1:0", "target:t:zombie:1:0")
    engine = object.__new__(draft_mod.FlashMTPDraftEngine)
    engine._cancelled_ids = set()
    engine._unconfirmed_direct = {tids[0]: future, tids[1]: 0}
    engine._direct_tasks_by_transfer = {
        tid: SimpleNamespace(
            transfer_id=tid, candidate_destination=None, direct_ring_seq=7
        )
        for tid in tids
    }
    engine._held_direct_source_done = {tid: ["held-source-done"] for tid in tids}
    engine._held_direct_results = {}
    engine._pending_ids = set(tids)
    engine._ring_pending_ids = set(tids)
    engine._received_ns_by_transfer = {tid: 1 for tid in tids}
    unresolved_endpoint = SimpleNamespace(destination=lambda seq, position: None)
    engine._ring_record_positions = {
        tid: (unresolved_endpoint, index) for index, tid in enumerate(tids)
    }
    engine._held_ring_pushes = [
        (list(engine._direct_tasks_by_transfer.values()), b"candidates", 3)
    ]
    engine._candidate_push_transport = SimpleNamespace()
    engine._free_result_slots = []

    # Only the zombie is past its deadline: it retires as cancelled, the live
    # row keeps the wave held with its bookkeeping untouched.
    engine._expire_unconfirmed_direct()
    assert tids[1] in engine._cancelled_ids
    assert tids[1] not in engine._unconfirmed_direct
    assert tids[1] not in engine._held_direct_source_done
    assert tids[0] in engine._unconfirmed_direct
    assert engine._held_direct_source_done[tids[0]] == ["held-source-done"]
    assert len(engine._held_ring_pushes) == 1
    assert engine._free_result_slots == []

    # Past the live row's deadline too: the hold releases through the real
    # path — cancelled identities retired, result slot freed, no ledgers.
    engine._unconfirmed_direct[tids[0]] = 0
    engine._expire_unconfirmed_direct()
    assert engine._held_ring_pushes == []
    assert engine._free_result_slots == [3]
    assert not engine._unconfirmed_direct
    assert not engine._direct_tasks_by_transfer
    assert not engine._held_direct_source_done
    assert not engine._cancelled_ids
    assert not engine._ring_record_positions


# ---------------------------------------------------------------------------
# P2 (DT_TARGET_DEST_AT_LAUNCH, issue #321): launch-time reserve + authorize
# + destination write, park-time local ledger begin, shrunk cleanup.
# ---------------------------------------------------------------------------


def make_controller_p2():
    c = make_controller()
    c._target_dest_at_launch = True
    return c


def make_dest_ref(transfer_id, slot_id=1):
    from dt_protocol import CandidateDestinationRef

    return CandidateDestinationRef(
        target_instance_id="target",
        target_session_id="session",
        pool_generation="pool",
        transport_session="transport",
        transfer_id=transfer_id,
        target_device_id=0,
        device_uuid="uuid",
        slot_id=slot_id,
        generation=1,
        authorization_nonce=7,
        dtype="int32",
        row_width=3,
        nbytes=12,
        data_allocation_length=1 << 20,
        data_allocation_offset=4096,
        data_ipc_handle=b"",
        metadata_allocation_length=1 << 16,
        metadata_allocation_offset=8192,
        metadata_ipc_handle=b"",
    )


def test_launch_p2_reserves_authorizes_and_writes_destinations():
    """With the P2 flag on the launch reserves and authorizes every staged
    row's destination, writes the destination region for the seq the publish
    claims BEFORE the kernel goes out, and carries the reservations to the
    wave for the cleanup."""
    c, batch, result, publisher, arena, packed = make_wave(make_controller_p2())
    c._stage_dflash_ring_publish_at_launch(batch, result)
    launch = result.dt_ring_publish_launch
    assert result.dt_ring_publish_launch_error is None
    assert launch is not None
    # Every staged row is reserved and authorized to its owner binding.
    assert arena.reserved == ["target:a:1:5", "target:b:1:7"]
    assert [a.draft_instance_id for a in arena.authorized] == ["di", "di"]
    assert [a.draft_session_id for a in arena.authorized] == ["session"] * 2
    assert [a.authorization_nonce for a in arena.authorized] == [7, 7]
    assert [a.owner_epoch for a in arena.authorized] == [9, 9]
    # The region write names the seq the publish claims and lands first.
    order = [call[0] for call in publisher.calls]
    assert order == ["stage", "destinations", "publish"]
    dest_call = publisher.calls[1]
    assert dest_call[1][0] == 41  # the seq publish() then claims
    assert dest_call[1][1] == [(1, 1, 7), (2, 1, 7)]
    assert dest_call[2]["data_buffer_offset"] == 4096
    assert dest_call[2]["metadata_buffer_offset"] == 8192
    assert set(launch.destinations) == {"target:a:1:5", "target:b:1:7"}
    # The finalize carries the reservations to the wave's cleanup.
    specs = c._dflash_commit_specs(
        list(launch.staged_meta),
        result.accept_lens.numpy(),
        result.next_draft_input.bonus_tokens.numpy(),
        result.next_draft_input.new_seq_lens.numpy(),
    )
    assert c._finalize_dflash_ring_publish_at_launch(batch, result, launch, specs)
    wave = c._ring_publish_pending[0]
    assert wave.launch_destinations is launch.destinations


def test_launch_p2_arena_full_falls_back_without_a_record():
    """An arena that cannot place every row publishes nothing at launch:
    the wave falls back to park whole (design §5.3 item 3)."""
    c, batch, result, publisher, arena, packed = make_wave(make_controller_p2())
    arena.free_slots = 0
    c._stage_dflash_ring_publish_at_launch(batch, result)
    assert result.dt_ring_publish_launch is None
    assert result.dt_ring_publish_launch_error is None
    assert not publisher.calls
    assert not arena.reserved and not arena.cancelled
    assert c._inplace_hidden  # the lease waits for the park path


def test_launch_p2_mid_reserve_arena_full_cancels_partials():
    from sglang.srt.speculative.dt_candidate_p2p import CandidateTransportError

    c, batch, result, publisher, arena, packed = make_wave(make_controller_p2())
    arena.fail_at = 1
    arena.fail_exc = CandidateTransportError("all candidate arena slots are active")
    c._stage_dflash_ring_publish_at_launch(batch, result)
    assert result.dt_ring_publish_launch is None
    assert result.dt_ring_publish_launch_error is None
    # The partial reservation is rolled back; nothing was published.
    assert arena.reserved == ["target:a:1:5"]
    assert arena.cancelled == [("target:a:1:5", True)]
    assert not publisher.calls
    assert c._inplace_hidden


def test_launch_p2_pre_kernel_failure_cancels_destinations():
    """A pre-enqueue DirectRingError restores the lease AND rolls the
    launch reservations back, so the park path can reserve them again."""
    from sglang.srt.speculative.dt_direct_ring import DirectRingError

    c, batch, result, publisher, arena, packed = make_wave(make_controller_p2())

    def not_serving(**_kwargs):
        raise DirectRingError("ring is not serving")

    publisher.publish = not_serving
    c._stage_dflash_ring_publish_at_launch(batch, result)
    assert result.dt_ring_publish_launch is None
    assert result.dt_ring_publish_launch_error is None
    assert sorted(arena.cancelled) == [
        ("target:a:1:5", True),
        ("target:b:1:7", True),
    ]
    assert c._inplace_hidden[int(packed.data_ptr())][0] == 3


def test_cleanup_p2_stamps_launch_destinations_without_reserving():
    """The P2 cleanup reserves nothing and writes no region: it stamps the
    launch reservations into the confirmation copies and retires ended rows
    from the same reservations."""
    c, batch, result, publisher, arena, packed = make_wave(make_controller_p2())
    c._stage_dflash_ring_publish_at_launch(batch, result)
    launch, _specs = _finalize_launch_wave(c, batch, result)
    # The first row ended in the wave: park's finished branch cancelled it.
    c._cancelled_transfers.add("target:a:1:5")
    published, made = _setup_cleanup_doubles(c)
    reserved_before = list(arena.reserved)
    writes = []
    c._ring_writers["session"].write_destinations = (
        lambda *args, **kwargs: writes.append(args)
    )
    c._ring_publish_cleanup()
    assert arena.reserved == reserved_before  # no cleanup reservation
    assert not writes  # no cleanup region write
    # The live row's confirmation copy carries the launch reservation.
    assert len(made) == 1
    assert made[0].transfer_id == "target:b:1:7"
    assert made[0].direct_ring_seq == 41
    assert made[0].candidate_destination is launch.destinations["target:b:1:7"]
    # The ended row retires its launch reservation through the cancel path.
    assert ("target:a:1:5", True) in arena.cancelled
    assert "target:a:1:5" not in c._cancelled_transfers
    assert not c._ring_publish_pending


def test_park_p2_begins_launch_rounds_in_the_local_ledger():
    """P2: a launch-published wave's rounds enter the ledger on the same
    park pass that parks them — READY must not wait for the confirmation
    copy's submit — and the copy's submit adopts the full task."""
    import msgspec
    from dt_client import TargetRoundLedger, TargetRoundPhase
    from dt_protocol import DraftTask, DraftTaskKind, TargetTaskBatch, PROTOCOL_VERSION

    c = make_controller()
    c._target_dest_at_launch = True
    c.round_timeout_ns = 10**12
    c.profile_id = "p"
    ledger = c._ledger = TargetRoundLedger(target_instance_id="target")
    c._verify_transfer_by_request = {"a": "old-a"}
    c._parked = {}
    c._feature_control_rounds = {}
    import collections

    c._lane_target_sizes = collections.defaultdict(int)
    c._lane_for_request = lambda _req: 0
    c._target_inflight_count = lambda _rid: 0
    c._retire_dflash_rounds_for_request = lambda *_args, **_kwargs: None
    c._forget_request_lane = lambda *_args: None
    c._release_draft_request = lambda *_args: None
    c._source_slot_by_transfer = {}
    # The round this wave verified is mid-lifecycle in the ledger.
    ledger.begin_local(
        DraftTask(
            transfer_id="old-a",
            request_id="a",
            round_id=0,
            attempt=5,
            target_instance_id="target",
            draft_profile_id="p",
            k=3,
            next_token_id=1,
            source=None,
            created_ns=0,
            deadline_ns=10**13,
            kind=DraftTaskKind.BOOTSTRAP,
            candidate_destination=make_dest_ref("old-a", slot_id=9),
        )
    )
    ledger.mark_lease_dispatched("old-a", draft_instance_id="di")
    ledger.mark_direct_ready("old-a", draft_instance_id="di")

    destination = make_dest_ref("target:a:1:5")
    req = SimpleNamespace(
        rid="a", finished=lambda: False, is_retracted=False,
        inflight_middle_chunks=0,
    )
    staged = target.StagedDFlashFeatureBatch(
        tasks=(
            target.StagedDFlashTask(
                request_id="a",
                transfer_id="target:a:1:5",
                round_id=1,
                attempt=5,
                draft_kv_handle="kv-a",
                feature_token_offset=10,
                feature_token_count=3,
                feature_token_ids=(),
                committed_token_ids=(),
                previous_transfer_id="old-a",
                bonus_token=77,
                seq_len=13,
            ),
        ),
        descriptors_by_transfer={},
        ring_publish=True,
    )
    wave = target.RingPublishWave(
        slot_id=3,
        generation=11,
        row_count=4,
        ready_event=None,
        reqs={"a": req},
        proposals={},
        records=(),
        launch_destinations={"target:a:1:5": destination},
    )
    batch = SimpleNamespace(reqs=[req], chunked_req=None)
    result = SimpleNamespace(dt_staged_dflash_features=staged, dt_ring_publish_wave=wave)

    c._park_dflash_completed_batch(batch, None, batch_result=result)

    # The verified round is finished; the new round is parked AND in the
    # ledger from local state, lease-dispatched, before any submit exists.
    assert ledger.get("old-a") is None
    assert "target:a:1:5" in c._parked
    state = ledger.get("target:a:1:5")
    assert state is not None and state.local_only
    assert state.phase == TargetRoundPhase.WAIT_DRAFT
    assert state.dispatch_accepted and state.lease_dispatched
    assert state.draft_instance_id == "di"
    task = state.task
    assert task.next_token_id == 77
    assert task.draft_kv_handle == "kv-a"
    assert task.feature_token_offset == 10
    assert task.feature_token_count == 3
    assert task.committed_kv_len == 13
    assert task.candidate_destination is destination
    assert wave.proposals["target:a:1:5"] == (True, "old-a")
    # READY can land now, before the confirmation copy's submit.
    ledger.mark_direct_ready("target:a:1:5", draft_instance_id="di")
    assert state.phase == TargetRoundPhase.VERIFY_READY
    # The confirmation copy's submit adopts the full task in place.
    full = msgspec.structs.replace(
        task, committed_token_ids=(1, 2, 3), feature_token_ids=(1, 2, 3),
        direct_ring_seq=41,
    )
    ledger.begin(
        TargetTaskBatch(
            protocol_version=PROTOCOL_VERSION,
            batch_id="target:tasks:9",
            target_instance_id="target",
            entries=(full,),
            sent_ns=0,
        )
    )
    assert state.task is full
    assert not state.local_only
    assert state.phase == TargetRoundPhase.VERIFY_READY
