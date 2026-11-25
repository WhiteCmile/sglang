"""CPU contracts of the Target's confirmation-free mode (DT_RING_CONFIRMATION=0).

The per-PR GPU gate runs the confirmation-free default, so these pin the
contracts nothing else checks before the Target stops sending its per-round
Coordinator copies (design §7 T3, issue #329):

- the ring cleanup builds no Coordinator copy; the round begins in the local
  ledger with the owner lease's assignment acceptance, which is what the
  direct completion path requires;
- the flush routes only the unstamped remainder to the Coordinator: ring
  rounds never join a submitted batch, and the local v1 ring write no longer
  waits for a Coordinator submit;
- a remote ring round's authenticated push/tombstone proof retires its
  source lease directly, with no accepted/unaccepted split and no
  Coordinator relay.

Run only after the task's external release/gate; imports can load heavy modules.
"""

from collections import deque
from types import SimpleNamespace

import msgspec
import torch

import sglang.srt.speculative.dt_target as target
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dt_direct_ring import (
    ENTRY_BYTES,
    V2_HEADER_BYTES,
    V2_REQ_ENTRY_BYTES,
    V2_TOKEN_OFFSET,
)

from dt_client import TargetRoundLedger
from dt_protocol import (
    PROTOCOL_VERSION,
    DraftTask,
    DraftTaskKind,
    TargetTaskBatch,
)


def _task(transfer_id, request_id, *, ring_seq=0, destination=None):
    return DraftTask(
        transfer_id=transfer_id,
        request_id=request_id,
        round_id=1,
        attempt=0,
        target_instance_id="target",
        draft_profile_id="p",
        k=3,
        next_token_id=1,
        source=None,
        created_ns=0,
        deadline_ns=60 * 10**9,
        kind=DraftTaskKind.RECONCILE,
        draft_kv_handle="kv",
        committed_kv_len=8,
        committed_token_ids=(1, 2, 3),
        requires_proposal=True,
        candidate_destination=destination,
        direct_ring_seq=ring_seq,
    )


class _Destination(msgspec.Struct):
    slot_id: int = 1
    generation: int = 1
    authorization_nonce: int = 7


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


class _Publisher:
    def __init__(self):
        self.writer = SimpleNamespace(
            next_seq=41,
            capacity=64,
            max_entry_bytes=1 << 20,
            write_destinations=lambda *args, **kwargs: None,
        )

    def free_entries(self):
        return 3

    def rid_hash_for(self, rid):
        return 0xA5 ^ len(rid)

    def next_launch_slot(self):
        return 1

    def stage_requests(self, entries, *, staging_slot=0):
        pass

    def publish(self, **kwargs):
        seq = self.writer.next_seq
        self.writer.next_seq = seq + 1
        return seq


class _Arena:
    def __init__(self):
        self.reserved = []
        self.authorized = []
        self._data_buffer = SimpleNamespace(allocation_offset=4096)
        self._metadata_buffer = SimpleNamespace(allocation_offset=8192)

    @property
    def free_slot_count(self):
        return 256

    def reserve(self, transfer_id):
        self.reserved.append(transfer_id)
        return _Destination()

    def authorize(self, assignment):
        self.authorized.append(assignment)

    def cancel(self, transfer_id, *, remote_quiescent=False):
        pass


def _pack_record(buf, base, *, seq, row_count, reqs, tokens):
    import struct

    struct.pack_into("<Q", buf, base, seq)
    struct.pack_into("<I", buf, base + 48, row_count)
    struct.pack_into("<I", buf, base + 56, len(reqs))
    struct.pack_into("<Q", buf, base + 64, 123456)
    for i, rec in enumerate(reqs):
        struct.pack_into(
            "<Q6I", buf, base + V2_HEADER_BYTES + i * V2_REQ_ENTRY_BYTES, *rec
        )
    struct.pack_into(f"<{len(tokens)}I", buf, base + V2_TOKEN_OFFSET, *tokens)


def _cleanup_controller():
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
    c._ring_confirmation = False  # the confirmation-free Target under test
    c._ledger = TargetRoundLedger(target_instance_id="target")
    c._feature_source_pool = SimpleNamespace(
        _buffer=SimpleNamespace(allocation_offset=1 << 20),
        _slot_nbytes=6,
        quarantine_reserved=lambda *args: None,
        source_session_id="session",
        pool_generation="pool",
    )
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
    c._ring_publish_publisher = lambda _session: _Publisher()
    c._candidate_arena = _Arena()
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
                rid=rid,
                finished=lambda: False,
                is_retracted=False,
                inflight_middle_chunks=0,
            )
            for rid in ("a", "b")
        ],
        chunked_req=None,
        dt_full_wave_id=5,
        dt_pipeline_lane_id=1,
    )
    return c, batch, result


def _finalize(c, batch, result):
    launch = result.dt_ring_publish_launch
    specs = c._dflash_commit_specs(
        list(launch.staged_meta),
        result.accept_lens.numpy(),
        result.next_draft_input.bonus_tokens.numpy(),
        result.next_draft_input.new_seq_lens.numpy(),
    )
    c._finalize_dflash_ring_publish_at_launch(batch, result, launch, specs)


def test_cleanup_confirmation_free_begins_local_ledger():
    c, batch, result = _cleanup_controller()
    c._stage_dflash_ring_publish_at_launch(batch, result)
    _finalize(c, batch, result)
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
    raw = bytearray(ENTRY_BYTES * 64)
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
    c._source_slot_by_transfer = {}
    c._ring_publish_transfers = set()
    c._ring_unproven = {}
    c._lease_marks = []
    c._direct_sent = set()
    c._ring_outstanding = {"session": target.deque()}
    queued = []
    c._queue_task_batch = lambda tasks: queued.extend(tasks)
    c._flush_pending_send = lambda: None
    c._make_task = lambda **kwargs: _task(kwargs["transfer_id"], kwargs["req"].rid)

    c._ring_publish_cleanup()

    # No confirmation copy is built or queued for a confirmation-free wave.
    assert not queued
    assert not c._ring_publish_transfers
    assert not c._lease_marks
    assert len(published) == 1
    assert not c._ring_publish_pending
    # Both rounds began in the local ledger from the ring readback, with the
    # owner lease's assignment acceptance — the direct completion path's
    # precondition — and the ring bookkeeping for the read proof.
    for transfer_id in ("target:a:1:5", "target:b:1:7"):
        state = c._ledger.get(transfer_id)
        assert state is not None
        assert state.task.direct_ring_seq == 41
        assert state.task.candidate_destination is not None
        assert state.dispatch_accepted and state.lease_dispatched
        assert state.draft_instance_id == "di"
        assert transfer_id in c._direct_sent
        assert transfer_id in c._ring_unproven
        # The completion path accepts the round without any Coordinator ACK.
        c._ledger.mark_direct_ready(transfer_id, draft_instance_id="di")
    assert [item[1] for item in c._ring_outstanding["session"]] == [
        "target:a:1:5",
        "target:b:1:7",
    ]
    # The owner lease authorized every reserved destination: without this the
    # arena rejects the Draft's candidate push and the round never completes
    # (gate-measured pre-warmup quiescence wedge, issue #329).
    assert [item.transfer_id for item in c._candidate_arena.authorized] == [
        "target:a:1:5",
        "target:b:1:7",
    ]
    assert all(
        item.draft_instance_id == "di" and item.draft_session_id == "session"
        for item in c._candidate_arena.authorized
    )


def _flush_controller(ring_confirmation: bool):
    c = object.__new__(target.DTTargetController)
    c.is_export_rank = True
    c.is_persistent = True
    c.candidate_transport = "cuda-ipc"
    c._direct_ring = True
    c._ring_confirmation = ring_confirmation
    c.instance_id = "target"
    c._batch_sequence = 0
    c._pending_send = deque()
    c._last_pending_send_block = None
    c.whole_batch_task_fifo_ablation = False
    c._flush_reliable_outbox = lambda: None
    c._profile_active = False
    c._profile_emit = lambda *_args, **_kwargs: None
    c._ledger = TargetRoundLedger(target_instance_id="target")
    c._candidate_arena = _Arena()
    submitted = []
    c._client = SimpleNamespace(submit=lambda value: submitted.append(value) or True)
    outbox_added = []
    c._delivery_outbox = SimpleNamespace(
        add=lambda *args: outbox_added.append(args),
        mark_sent=lambda *_args, **_kwargs: None,
    )
    c._lease_marks = []
    c._remote_ring_records = {}
    c._remote_ring_destinations = {}
    return c, submitted, outbox_added


def test_flush_confirmation_free_routes_only_unstamped_rounds():
    c, submitted, _ = _flush_controller(ring_confirmation=False)
    ring_task = _task(
        "t-ring", "req-ring", ring_seq=7, destination=_Destination()
    )
    routed_task = _task("t-routed", "req-routed", destination=_Destination())
    c._pending_send.append(
        TargetTaskBatch(
            protocol_version=PROTOCOL_VERSION,
            batch_id="batch-1",
            target_instance_id="target",
            entries=(ring_task, routed_task),
            sent_ns=1,
        )
    )
    written = []
    plan = (("session", object(), ring_task, b"payload"),)
    c._write_direct_ring = lambda plan_arg: written.append(plan_arg)
    c._plan_direct_ring = lambda tasks: (tasks, plan)

    c._flush_pending_send()

    # Only the unstamped remainder is submitted; the local ring write ran
    # without waiting for any submit.
    assert len(submitted) == 1
    assert [task.transfer_id for task in submitted[0].entries] == ["t-routed"]
    assert written == [plan]
    # Both rounds are in the ledger: the routed one via the submitted batch,
    # the ring one from local state, ready for the direct completion path.
    ring_state = c._ledger.get("t-ring")
    assert ring_state is not None and ring_state.task.direct_ring_seq == 7
    assert c._ledger.get("t-routed") is not None
    assert not c._pending_send

    # An all-ring flush submits nothing at all: zero Coordinator messages.
    c2, submitted2, _ = _flush_controller(ring_confirmation=False)
    c2._pending_send.append(
        TargetTaskBatch(
            protocol_version=PROTOCOL_VERSION,
            batch_id="batch-2",
            target_instance_id="target",
            entries=(
                _task("t-ring-2", "req-ring-2", ring_seq=8, destination=_Destination()),
            ),
            sent_ns=2,
        )
    )
    written2 = []
    c2._write_direct_ring = lambda plan_arg: written2.append(plan_arg)
    c2._plan_direct_ring = lambda tasks: (tasks, ())
    c2._flush_pending_send()
    assert not submitted2
    assert c2._ledger.get("t-ring-2") is not None
    assert not c2._pending_send


def test_flush_ring_confirmation_keeps_ring_copies_in_the_batch():
    c, submitted, _ = _flush_controller(ring_confirmation=True)
    ring_task = _task(
        "t-ring", "req-ring", ring_seq=7, destination=_Destination()
    )
    c._pending_send.append(
        TargetTaskBatch(
            protocol_version=PROTOCOL_VERSION,
            batch_id="batch-1",
            target_instance_id="target",
            entries=(ring_task,),
            sent_ns=1,
        )
    )
    written = []
    plan = (("session", object(), ring_task, b"payload"),)
    c._write_direct_ring = lambda plan_arg: written.append(plan_arg)
    c._plan_direct_ring = lambda tasks: (tasks, plan)
    c._flush_pending_send()
    # Ring confirmation (DT_RING_CONFIRMATION=1): the stamped copy still
    # rides the Coordinator batch and the local write still follows the
    # submit.
    assert [task.transfer_id for task in submitted[0].entries] == ["t-ring"]
    assert written == [plan]
    assert c._ledger.get("t-ring") is not None


def test_remote_source_proof_retires_every_ring_round_confirmation_free():
    import threading

    c = object.__new__(target.DTTargetController)
    c._ring_confirmation = False
    c._remote_source_lock = threading.Lock()
    c._remote_source_retirement = {}
    c._direct_sent = {"t-1"}
    c._remote_ring_sessions = {"t-1": "sess"}
    c._remote_ring_records = {}
    c._remote_ring_destinations = {}
    c._remote_source_proof_landed = set()
    read_done = []
    c.is_feature_backed = True
    c.is_token_only = False
    c._feature_source_pool = SimpleNamespace(
        read_done=lambda **kwargs: read_done.append(kwargs),
        source_session_id="session",
        pool_generation="pool",
    )
    c._source_slot_by_transfer = {"t-1": (3, 11)}
    c._source_deadline_by_transfer = {"t-1": 5}
    c._staged_flashmtp_source_leases = set()

    c._consume_remote_source_proof("t-1")

    # The authenticated proof retired the source lease directly and cleaned
    # the unconfirmed-round bookkeeping — no waiter, no relayed SRD, no
    # acceptance event involved.
    assert read_done == [
        {
            "transfer_id": "t-1",
            "source_slot_id": 3,
            "slot_generation": 11,
            "source_session_id": "session",
            "pool_generation": "pool",
        }
    ]
    assert not c._direct_sent
    assert not c._remote_ring_sessions
    assert not c._source_slot_by_transfer
    assert not c._source_deadline_by_transfer

    # Ring confirmation (DT_RING_CONFIRMATION=1): the same early proof has no
    # waiter and no acceptance, so it is only cached for the retire replay.
    c2 = object.__new__(target.DTTargetController)
    c2._ring_confirmation = True
    c2._remote_source_lock = threading.Lock()
    c2._remote_source_retirement = {}
    c2._direct_sent = {"t-1"}
    c2._remote_ring_sessions = {"t-1": "sess"}
    c2._remote_ring_records = {}
    c2._remote_ring_destinations = {}
    c2._remote_source_proof_landed = set()
    c2._source_slot_by_transfer = {"t-1": (3, 11)}
    c2._consume_remote_source_proof("t-1")
    assert c2._remote_source_proof_landed == {"t-1"}
    assert c2._direct_sent == {"t-1"}
    assert c2._source_slot_by_transfer == {"t-1": (3, 11)}
