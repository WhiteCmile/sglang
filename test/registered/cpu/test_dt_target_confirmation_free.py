"""CPU contracts of the Target's confirmation-free mode (DT_RING_CONFIRMATION=0).

The per-PR GPU gate runs the confirmation-free default, so these pin the
contracts nothing else checks before the Target stops sending its per-round
Coordinator copies (design §7 T3, issue #329, owner decision 06:23 UTC):

- a same-host ring round's Coordinator copy is held — one per request, the
  latest — and catch-up flushed ahead of the next Coordinator-routed message
  about the request (a fallback RECONCILE, a cancel, a release), so the
  Coordinator absorbs it before its length/offset checks and cancel routing
  see the round;
- a cross-host ring round keeps sending its copy per round (its source read
  is acknowledged through the Coordinator);
- the flush routes only the unstamped remainder to the Coordinator: ring
  rounds never join a submitted batch as dispatches, and the local v1 ring
  write no longer waits for a Coordinator submit;
- ring rounds begin in the local ledger with the owner lease's assignment
  acceptance, which is what the direct completion path requires;
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

from dt_client import RetryOutbox, TargetRoundLedger
from dt_protocol import (
    CANCEL_TRANSFERS,
    PROTOCOL_VERSION,
    CancelTransfers,
    CandidateArenaRegistration,
    CandidateDestinationRef,
    DraftCapacityUpdate,
    DraftTask,
    DraftTaskKind,
    TargetTaskBatch,
    cancel_transfers_id,
)

PROFILE = "flashmtp-qwen3-14b-v2"
TARGET_SESSION = "target-session"


def _task(
    transfer_id,
    request_id,
    *,
    ring_seq=0,
    destination=None,
    round_id=1,
    committed_kv_len=8,
    committed_token_ids=(1, 2, 3),
    next_token_id=None,
):
    return DraftTask(
        transfer_id=transfer_id,
        request_id=request_id,
        round_id=round_id,
        attempt=0,
        target_instance_id="target",
        draft_profile_id=PROFILE,
        k=3,
        next_token_id=committed_token_ids[-1] if next_token_id is None else next_token_id,
        source=None,
        created_ns=0,
        deadline_ns=60 * 10**9,
        kind=DraftTaskKind.BOOTSTRAP if round_id == 0 else DraftTaskKind.RECONCILE,
        draft_kv_handle=f"kv-{request_id}",
        committed_kv_len=committed_kv_len,
        committed_token_ids=committed_token_ids,
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
    c._held_ring_copies = {}
    c._held_ring_copy_requests = {}
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

    # No per-round copy is queued; each round's copy is held as its
    # request's catch-up copy instead.
    assert not queued
    assert not c._ring_publish_transfers
    assert not c._lease_marks
    assert len(published) == 1
    assert not c._ring_publish_pending
    assert set(c._held_ring_copies) == {"a", "b"}
    assert c._held_ring_copies["a"].transfer_id == "target:a:1:5"
    assert c._held_ring_copies["b"].transfer_id == "target:b:1:7"
    assert all(
        copy.direct_ring_seq == 41 for copy in c._held_ring_copies.values()
    )
    assert c._ring_copy_suppressed == 2
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
    c._held_ring_copies = {}
    c._held_ring_copy_requests = {}
    c._remote_ring_sessions = {}
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


def _queued_batch(*tasks):
    return TargetTaskBatch(
        protocol_version=PROTOCOL_VERSION,
        batch_id="batch-" + tasks[0].transfer_id,
        target_instance_id="target",
        entries=tuple(tasks),
        sent_ns=1,
        instance_session_id=TARGET_SESSION,
    )


def test_flush_confirmation_free_holds_ring_copies_and_routes_the_remainder():
    c, submitted, _ = _flush_controller(ring_confirmation=False)
    ring_task = _task(
        "t-ring", "req-ring", ring_seq=7, destination=_Destination()
    )
    routed_task = _task("t-routed", "req-routed", destination=_Destination())
    c._pending_send.append(_queued_batch(ring_task, routed_task))
    written = []
    plan = (("session", object(), ring_task, b"payload"),)
    c._write_direct_ring = lambda plan_arg: written.append(plan_arg)
    c._plan_direct_ring = lambda tasks: (tasks, plan)

    c._flush_pending_send()

    # Only the unstamped remainder is submitted; the local ring write ran
    # without waiting for any submit; the ring round's copy is held.
    assert len(submitted) == 1
    assert [task.transfer_id for task in submitted[0].entries] == ["t-routed"]
    assert written == [plan]
    assert c._held_ring_copies["req-ring"].transfer_id == "t-ring"
    assert c._held_ring_copies["req-ring"].direct_ring_seq == 7
    assert c._ring_copy_suppressed == 1
    # Both rounds are in the ledger: the routed one via the submitted batch,
    # the ring one from local state, ready for the direct completion path.
    ring_state = c._ledger.get("t-ring")
    assert ring_state is not None and ring_state.task.direct_ring_seq == 7
    assert c._ledger.get("t-routed") is not None
    assert not c._pending_send

    # An all-ring flush submits nothing at all: zero Coordinator messages,
    # the copy held.
    c2, submitted2, _ = _flush_controller(ring_confirmation=False)
    c2._pending_send.append(
        _queued_batch(
            _task("t-ring-2", "req-ring-2", ring_seq=8, destination=_Destination()),
        )
    )
    written2 = []
    c2._write_direct_ring = lambda plan_arg: written2.append(plan_arg)
    c2._plan_direct_ring = lambda tasks: (tasks, ())
    c2._flush_pending_send()
    assert not submitted2
    assert c2._held_ring_copies["req-ring-2"].transfer_id == "t-ring-2"
    assert c2._ledger.get("t-ring-2") is not None
    assert not c2._pending_send


def test_flush_catchup_flushes_held_copy_before_the_fallback_reconcile():
    c, submitted, _ = _flush_controller(ring_confirmation=False)
    c._write_direct_ring = lambda plan: None
    c._plan_direct_ring = lambda tasks: (tasks, ())

    ring_task = _task(
        "t-ring",
        "req-a",
        ring_seq=7,
        destination=_Destination(),
        round_id=1,
        committed_kv_len=4,
        committed_token_ids=(11, 12),
    )
    c._pending_send.append(_queued_batch(ring_task))
    c._flush_pending_send()
    assert not submitted
    assert c._held_ring_copies["req-a"].transfer_id == "t-ring"

    # The ring round retires locally (result applied, entry finished) before
    # the next round stages; the held copy outlives it — that is exactly why
    # the catch-up batch bypasses the ledger.
    c._ledger.finish("t-ring")
    fallback = _task(
        "t-fallback",
        "req-a",
        round_id=2,
        committed_kv_len=6,
        committed_token_ids=(13, 14),
    )
    c._pending_send.append(_queued_batch(fallback))
    c._flush_pending_send()

    # The copy goes out in its own batch, ahead of the fallback RECONCILE,
    # on the same ordered socket.
    assert len(submitted) == 2
    assert [task.transfer_id for task in submitted[0].entries] == ["t-ring"]
    assert submitted[0].entries[0].direct_ring_seq == 7
    assert [task.transfer_id for task in submitted[1].entries] == ["t-fallback"]
    assert not c._held_ring_copies
    assert c._ring_copy_flushed == 1
    assert c._fallback_reconcile_sent == 1


def test_flush_cross_host_ring_copy_stays_in_the_batch():
    c, submitted, _ = _flush_controller(ring_confirmation=False)
    ring_task = _task(
        "t-remote", "req-remote", ring_seq=9, destination=_Destination()
    )
    c._remote_ring_sessions["t-remote"] = "remote-session"
    c._pending_send.append(_queued_batch(ring_task))
    c._plan_direct_ring = lambda tasks: (tasks, ())

    c._flush_pending_send()

    # Cross-host round: the Coordinator acknowledges the source read, so the
    # copy is sent per round as today — no holding.
    assert len(submitted) == 1
    assert [task.transfer_id for task in submitted[0].entries] == ["t-remote"]
    assert c._ring_copy_sent == 1
    assert not c._held_ring_copies
    assert c._ring_copy_suppressed == 0


def test_flush_ring_confirmation_keeps_ring_copies_in_the_batch():
    c, submitted, _ = _flush_controller(ring_confirmation=True)
    ring_task = _task(
        "t-ring", "req-ring", ring_seq=7, destination=_Destination()
    )
    c._pending_send.append(_queued_batch(ring_task))
    written = []
    plan = (("session", object(), ring_task, b"payload"),)
    c._write_direct_ring = lambda plan_arg: written.append(plan_arg)
    c._plan_direct_ring = lambda tasks: (tasks, plan)
    c._flush_pending_send()
    # Ring confirmation (DT_RING_CONFIRMATION=1): the stamped copy still
    # rides the Coordinator batch and the local write still follows the
    # submit; nothing is held.
    assert [task.transfer_id for task in submitted[0].entries] == ["t-ring"]
    assert written == [plan]
    assert c._ledger.get("t-ring") is not None
    assert not c._held_ring_copies
    assert c._ring_copy_suppressed == 0


def _outbox_controller():
    """Confirmation-free controller with the real outbox flush and a
    capturing client, for the catch-up-before-cancel/release ordering."""

    c = object.__new__(target.DTTargetController)
    c.is_export_rank = True
    c.is_persistent = True
    c.candidate_transport = "cuda-ipc"
    c._direct_ring = True
    c._ring_confirmation = False
    c.instance_id = "target"
    c._batch_sequence = 0
    c._pending_send = deque()
    c._last_pending_send_block = None
    c.whole_batch_task_fifo_ablation = False
    c._profile_active = False
    c._profile_emit = lambda *_args, **_kwargs: None
    c._ledger = TargetRoundLedger(target_instance_id="target")
    c._candidate_arena = _Arena()
    c._held_ring_copies = {}
    c._held_ring_copy_requests = {}
    c._remote_ring_sessions = {}
    c._remote_ring_records = {}
    c._remote_ring_destinations = {}
    c._lease_marks = []
    c._write_direct_ring = lambda plan: None
    c._plan_direct_ring = lambda tasks: (tasks, ())
    sent_ops = []

    def _submit(value):
        # The real client stamps instance_session_id at encode time
        # (TargetCoordinatorClient.encode); mirror it here.
        sent_ops.append(
            (
                "submit",
                msgspec.structs.replace(value, instance_session_id=TARGET_SESSION),
            )
        )
        return True

    def _cancel(value):
        sent_ops.append(("cancel", value))
        return True

    def _release(value):
        sent_ops.append(("release", value))
        return True

    c._client = SimpleNamespace(
        submit=_submit, cancel=_cancel, release_draft_request=_release
    )
    c._delivery_outbox = RetryOutbox[object](retry_interval_ns=60 * 10**9)
    return c, sent_ops


def test_cancel_flushes_the_held_copy_before_the_cancel_message():
    c, sent_ops = _outbox_controller()
    ring_task = _task(
        "t-ring", "req-a", ring_seq=7, destination=_Destination()
    )
    c._pending_send.append(_queued_batch(ring_task))
    c._flush_pending_send()
    assert c._held_ring_copies["req-a"].transfer_id == "t-ring"
    assert not sent_ops

    cancel = CancelTransfers(
        protocol_version=PROTOCOL_VERSION,
        target_instance_id="target",
        transfer_ids=("t-ring",),
        reason="client_abort",
    )
    c._delivery_outbox.add(CANCEL_TRANSFERS, cancel_transfers_id(cancel), cancel)
    c._flush_reliable_outbox()

    # Copy first, cancel second; the copy is the round's absorb at the
    # Coordinator, without it the cancel would be dropped there.
    assert [op for op, _ in sent_ops] == ["submit", "cancel"]
    copy_batch = sent_ops[0][1]
    assert [task.transfer_id for task in copy_batch.entries] == ["t-ring"]
    assert sent_ops[1][1] is cancel
    assert not c._held_ring_copies
    assert c._ring_copy_flushed == 1


def _candidate_registration():
    return CandidateArenaRegistration(
        target_instance_id="target",
        target_session_id=TARGET_SESSION,
        pool_generation="candidate-pool-generation",
        transport_session="candidate-transport-session",
        target_device_id=0,
        device_uuid="GPU-target-uuid",
        row_width=4,  # task.k (3) + 1
        data_allocation_length=1 << 20,
        data_ipc_handle=b"candidate-data-ipc-handle",
        metadata_allocation_length=1 << 20,
        metadata_ipc_handle=b"candidate-metadata-ipc-handle",
        completion_endpoint="tcp://127.0.0.1:32000",
    )


def _candidate_destination(transfer_id, *, generation=1, nonce=12345):
    return CandidateDestinationRef(
        target_instance_id="target",
        target_session_id=TARGET_SESSION,
        pool_generation="candidate-pool-generation",
        transport_session="candidate-transport-session",
        transfer_id=transfer_id,
        target_device_id=0,
        device_uuid="GPU-target-uuid",
        slot_id=0,
        generation=generation,
        authorization_nonce=nonce,
        dtype="int32",
        row_width=4,  # task.k (3) + 1
        nbytes=4 * 4,
        data_allocation_length=1 << 20,
        data_allocation_offset=0,
        data_ipc_handle=b"",
        metadata_allocation_length=1 << 20,
        metadata_allocation_offset=0,
        metadata_ipc_handle=b"",
    )


def _router():
    from dt_coordinator import CoordinatorRouter

    router = CoordinatorRouter(terminal_ttl_ns=10_000_000_000, ring_confirmation=False)
    router.register_target("target", candidate_arena=_candidate_registration())
    router.register_draft(
        "draft-0",
        (PROFILE,),
        max_pending_tasks=8,
        host_id="host-0",
        instance_session_id="draft-0-session",
        result_pool_generation="draft-0-results",
    )
    draft = router.drafts["draft-0"]
    router.update_capacity(
        DraftCapacityUpdate(
            protocol_version=PROTOCOL_VERSION,
            draft_instance_id="draft-0",
            instance_session_id=draft.instance_session_id,
            received_tasks=0,
            capacity_sequence=draft.capacity_sequence + 1,
            pending_tasks=0,
            max_pending_tasks=draft.max_pending_tasks,
            free_result_rows=draft.max_pending_tasks,
            pending_work=0,
        )
    )
    return router


def _dispatch(router, batch, *, now_ns):
    return router.dispatch(batch, now_ns=now_ns)


def test_ring_rounds_fallback_reconcile_passes_and_cancel_reaches_the_draft():
    # End to end across the two runtimes: the confirmation-free Target holds
    # a ring round's copy, catches it up ahead of the request's next
    # Coordinator-routed messages, and the Coordinator — with every check
    # strict — absorbs the copy, passes the fallback RECONCILE's length
    # check, and routes the round's cancel to the owner Draft (not the
    # timeout path).
    router = _router()
    bootstrap = _task(
        "t-boot",
        "request-a",
        round_id=0,
        destination=_candidate_destination("t-boot"),
        committed_kv_len=1,
        committed_token_ids=(100,),
    )
    plan = _dispatch(
        router,
        _router_batch("boot-batch", bootstrap),
        now_ns=1,
    )
    assert plan.ack.accepted_transfer_ids == ("t-boot",)
    assert [batch.draft_instance_id for batch in plan.batches] == ["draft-0"]
    draft = router.drafts["draft-0"]
    router.update_capacity(
        DraftCapacityUpdate(
            protocol_version=PROTOCOL_VERSION,
            draft_instance_id="draft-0",
            instance_session_id=draft.instance_session_id,
            received_tasks=1,
            capacity_sequence=draft.capacity_sequence + 1,
            pending_tasks=1,
            max_pending_tasks=draft.max_pending_tasks,
            free_result_rows=draft.max_pending_tasks,
            pending_work=1,
        )
    )

    # Target side: two ring rounds (rounds 1-2) are held, latest only; the
    # fallback RECONCILE (round 3) catches the round-2 copy up ahead of
    # itself.  Round 1's copy never leaves the Target — the production
    # shape that failed before the Coordinator stopped demanding one-round
    # continuity from a catch-up copy (issue #329 legs b1-on/b1t-on).
    c, sent_ops = _outbox_controller()
    ring = _task(
        "t-ring",
        "request-a",
        ring_seq=7,
        destination=_candidate_destination("t-ring", generation=2, nonce=23456),
        round_id=1,
        committed_kv_len=2,
        committed_token_ids=(101,),
    )
    c._pending_send.append(_queued_batch(ring))
    c._flush_pending_send()
    assert not sent_ops
    assert c._held_ring_copies["request-a"].transfer_id == "t-ring"

    c._ledger.finish("t-ring")
    ring2 = _task(
        "t-ring2",
        "request-a",
        ring_seq=8,
        destination=_candidate_destination("t-ring2", generation=3, nonce=34567),
        round_id=2,
        committed_kv_len=3,
        committed_token_ids=(102,),
    )
    c._pending_send.append(_queued_batch(ring2))
    c._flush_pending_send()
    assert not sent_ops
    assert c._held_ring_copies["request-a"].transfer_id == "t-ring2"

    c._ledger.finish("t-ring2")
    fallback = _task(
        "t-fallback",
        "request-a",
        round_id=3,
        destination=_candidate_destination("t-fallback", generation=4, nonce=45678),
        committed_kv_len=5,
        committed_token_ids=(103, 104),
    )
    c._pending_send.append(_queued_batch(fallback))
    c._flush_pending_send()
    assert [op for op, _ in sent_ops] == ["submit", "submit"]
    copy_batch, fallback_batch = sent_ops[0][1], sent_ops[1][1]
    assert [task.transfer_id for task in copy_batch.entries] == ["t-ring2"]
    assert [task.transfer_id for task in fallback_batch.entries] == ["t-fallback"]

    # Coordinator side, in socket order: the catch-up copy skips round 1
    # and is absorbed past the gap (committed_kv_len 1 -> 3), then the
    # fallback RECONCILE passes its length check against the advanced
    # owner record.
    copy_plan = _dispatch(router, copy_batch, now_ns=2)
    assert copy_plan.ack.accepted_transfer_ids == ("t-ring2",)
    assert copy_plan.batches == ()
    assert router.ring_copies_absorbed == 1
    assert router.request_owners[("target", "request-a")].committed_kv_len == 3
    fallback_plan = _dispatch(router, fallback_batch, now_ns=3)
    assert fallback_plan.ack.accepted_transfer_ids == ("t-fallback",)
    assert [batch.draft_instance_id for batch in fallback_plan.batches] == ["draft-0"]

    # The ring round's cancel travels with nothing left to catch up (the copy
    # went out with the fallback): the Coordinator routes it to the owner
    # Draft because the copy was absorbed, not onto the timeout path.
    cancel = CancelTransfers(
        protocol_version=PROTOCOL_VERSION,
        target_instance_id="target",
        transfer_ids=("t-ring2",),
        reason="client_abort",
    )
    c._delivery_outbox.add(CANCEL_TRANSFERS, cancel_transfers_id(cancel), cancel)
    c._flush_reliable_outbox()
    assert [op for op, _ in sent_ops[2:]] == ["cancel"]
    cancel_message = sent_ops[2][1]
    assert cancel_message is cancel
    routed = router.cancel(cancel_message, now_ns=5)
    assert [(item.draft_instance_id, item.message.transfer_ids) for item in routed] == [
        ("draft-0", ("t-ring2",))
    ]
    # ... and a cancel that arrives before any copy would have been dropped:
    # no absorbed owner, no inflight entry, nothing to route.
    vanished = router.cancel(
        CancelTransfers(
            protocol_version=PROTOCOL_VERSION,
            target_instance_id="target",
            transfer_ids=("t-never-absorbed",),
            reason="client_abort",
        ),
        now_ns=6,
    )
    assert vanished == ()


def _router_batch(batch_id, *tasks):
    return TargetTaskBatch(
        protocol_version=PROTOCOL_VERSION,
        batch_id=batch_id,
        target_instance_id="target",
        entries=tuple(tasks),
        sent_ns=100,
        instance_session_id=TARGET_SESSION,
    )


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
