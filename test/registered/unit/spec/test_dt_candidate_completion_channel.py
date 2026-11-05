"""Focused CPU tests for the direct candidate completion channel.

Covers the transport behaviors the standalone cuda-ipc acceptance failure
exposed and the review pinned down:

- Draft retry throttling is per Target socket: backpressure on one peer
  must not suppress retries owed to a healthy peer in the same call.
- Retry attempts measure Target silence, not local pipe pressure: EAGAIN
  sends bump send_failures, not attempts; either counter exhausts.
- Deferred target ACKs are bounded, deduplicated per transfer, flushed
  within the poll deadline, and purged when the peer is gone.
- ROUTER_MANDATORY makes the two ROUTER failure modes explicit: EAGAIN for
  a backlogged peer, EHOSTUNREACH for a departed one.
- poll_completion_notices honors its wall-clock cap even under a flood.
- The native record observer records each first-matching completion or
  legacy READY record exactly once and the drain path consumes it exactly
  once, including across disarm/rearm and destroy.

The arena/transport constructors need CUDA, so objects are built with
object.__new__ plus only the attributes the methods under test touch.
"""

import os
import shutil
import time
import uuid
from collections import OrderedDict, deque
from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest
import zmq

pytest.importorskip("dt_protocol")

from sglang.srt.speculative.dt_candidate_p2p import (
    COMPLETION_HASH_WORD,
    COMPLETION_META_WORD,
    COMPLETION_OFFSET_WORD,
    COMPLETION_PENDING_ACK_LIMIT,
    COMPLETION_PRODUCED_WORD,
    COMPLETION_RECORD_BYTES,
    COMPLETION_RECORD_WORDS,
    COMPLETION_RETRY_EAGAIN_LIMIT,
    COMPLETION_RETRY_NS,
    COMPLETION_SEQ_WORD,
    COMPLETION_TOKEN_WORD,
    READY_RECORD_WORDS,
    CandidateSlotState,
    CandidateTransportError,
    DraftCandidatePushTransport,
    TargetCandidateArena,
    _CandidateSlot,
    _PendingCompletionNotice,
    _ReadyRecordFile,
    _RecordObserver,
    _RECORD_KIND_COMPLETION,
    _RECORD_KIND_READY,
    producer_tag,
    transfer_id_hash,
)


class FakeSocket:
    """Minimal stand-in for a zmq.Socket on the tested paths."""

    def __init__(self, *, send_error=None, recv_parts=()):
        self.send_error = send_error
        self.recv_queue = deque(recv_parts)
        self.sent = []
        self.recv_calls = 0

    def send(self, data, *, flags=0):
        if self.send_error is not None:
            raise self.send_error
        self.sent.append(data)

    def send_multipart(self, parts, *, flags=0):
        if self.send_error is not None:
            raise self.send_error
        self.sent.append(tuple(parts))

    def recv_multipart(self, *, flags=0):
        self.recv_calls += 1
        if not self.recv_queue:
            raise zmq.Again()
        return self.recv_queue.popleft()


def make_draft(sockets, pendings):
    transport = object.__new__(DraftCandidatePushTransport)
    transport._closed = False
    transport._ring_debug = None
    transport._completion_sockets = sockets
    transport._pending_notices = pendings
    transport.draft_instance_id = "draft-0"
    transport.draft_session_id = "session-0"
    return transport


def make_pending(socket, *, attempts=0, send_failures=0, next_retry_ns=0):
    return _PendingCompletionNotice(
        notice=SimpleNamespace(sent_ns=0),
        socket=socket,
        encoded=b"notice",
        attempts=attempts,
        next_retry_ns=next_retry_ns,
        send_failures=send_failures,
    )


def make_arena(socket=None):
    arena = object.__new__(TargetCandidateArena)
    arena._ring_debug = None
    arena._pending_acks = deque()
    arena._pending_ack_keys = set()
    arena._completion_socket = socket if socket is not None else FakeSocket()
    return arena


class TestDraftRetryThrottle:
    def test_full_pipe_on_one_target_does_not_starve_another(self):
        busy, healthy = FakeSocket(send_error=zmq.Again()), FakeSocket()
        pendings = {
            f"t{i}": make_pending(busy if i < 17 else healthy)
            for i in range(18)
        }
        transport = make_draft(
            {"busy": busy, "healthy": healthy}, pendings
        )
        transport.progress_completions(now_ns=1)
        # COMPLETION_RETRY_EAGAIN_LIMIT busy sends happened; the rest of the
        # busy peer's notices were deferred without burning send_failures.
        busy_failures = sum(p.send_failures for p in pendings.values())
        assert busy_failures == COMPLETION_RETRY_EAGAIN_LIMIT
        # The healthy peer's retry still went out in the same call.
        assert healthy.sent == [b"notice"]
        assert pendings["t17"].attempts == 1

    def test_eagain_counts_send_failures_not_attempts(self):
        socket = FakeSocket(send_error=zmq.Again())
        pendings = {"t0": make_pending(socket)}
        transport = make_draft({"s": socket}, pendings)
        transport.progress_completions(now_ns=1)
        assert pendings["t0"].attempts == 0
        assert pendings["t0"].send_failures == 1
        assert pendings["t0"].next_retry_ns == 1 + COMPLETION_RETRY_NS

    def test_successful_send_counts_attempts(self):
        socket = FakeSocket()
        pendings = {"t0": make_pending(socket)}
        transport = make_draft({"s": socket}, pendings)
        transport.progress_completions(now_ns=1)
        assert pendings["t0"].attempts == 1
        assert pendings["t0"].send_failures == 0

    @pytest.mark.parametrize("exhausted_field", ["attempts", "send_failures"])
    def test_exhaustion_raises_on_either_counter(self, exhausted_field):
        socket = FakeSocket()
        pendings = {"t0": make_pending(socket, **{exhausted_field: 100})}
        transport = make_draft({"s": socket}, pendings)
        with pytest.raises(CandidateTransportError, match="retry budget exhausted"):
            transport.progress_completions(now_ns=1)

    def test_eagain_streak_recovers_after_success(self):
        flaky = FakeSocket(send_error=zmq.Again())
        pendings = {f"t{i}": make_pending(flaky) for i in range(20)}
        transport = make_draft({"s": flaky}, pendings)
        transport.progress_completions(now_ns=1)
        assert sum(p.send_failures for p in pendings.values()) == 16
        # Pipe drains: the next call resumes sending where it stopped.
        flaky.send_error = None
        transport.progress_completions(now_ns=1 + COMPLETION_RETRY_NS)
        assert sum(p.attempts for p in pendings.values()) == 20


class TestDeferredAcks:
    def test_defer_deduplicates_per_transfer(self):
        arena = make_arena()
        arena._defer_ack(b"peer", "t1", b"ack-a")
        arena._defer_ack(b"peer", "t1", b"ack-b")
        arena._defer_ack(b"peer", "t2", b"ack-c")
        assert len(arena._pending_acks) == 2
        assert arena._pending_ack_keys == {(b"peer", "t1"), (b"peer", "t2")}

    def test_defer_queue_is_bounded(self):
        arena = make_arena()
        for i in range(COMPLETION_PENDING_ACK_LIMIT + 10):
            arena._defer_ack(b"peer", f"t{i}", b"ack")
        assert len(arena._pending_acks) == COMPLETION_PENDING_ACK_LIMIT

    def test_flush_respects_deadline(self):
        socket = FakeSocket()
        arena = make_arena(socket)
        for i in range(1000):
            arena._defer_ack(b"peer", f"t{i}", b"ack")
        arena._flush_pending_acks(deadline_ns=0)  # long past
        assert len(socket.sent) == 1  # one send, then the deadline check
        assert len(arena._pending_acks) == 999
        assert len(arena._pending_ack_keys) == 999

    def test_flush_keeps_entries_when_peer_stays_full(self):
        socket = FakeSocket(send_error=zmq.Again())
        arena = make_arena(socket)
        arena._defer_ack(b"peer", "t1", b"ack")
        arena._flush_pending_acks(deadline_ns=time.monotonic_ns() + 10**9)
        assert len(arena._pending_acks) == 1

    def test_flush_purges_departed_peer_and_serves_others(self):
        gone = zmq.ZMQError(zmq.EHOSTUNREACH)
        socket = FakeSocket()
        arena = make_arena(socket)
        arena._defer_ack(b"gone", "t1", b"ack")
        arena._defer_ack(b"live", "t2", b"ack")

        def selective_send(parts, *, flags=0):
            if parts[0] == b"gone":
                raise gone
            socket.sent.append(tuple(parts))

        socket.send_multipart = selective_send
        arena._flush_pending_acks(deadline_ns=time.monotonic_ns() + 10**9)
        assert socket.sent == [(b"live", b"ack")]
        assert len(arena._pending_acks) == 0
        assert len(arena._pending_ack_keys) == 0

    def test_revoke_producer_drops_deferred_acks(self):
        arena = make_arena()
        arena._defer_ack(b"session-a", "t1", b"ack")
        arena._defer_ack(b"session-b", "t2", b"ack")
        arena._drop_peer_pending_acks(b"session-a")
        assert [entry[0] for entry in arena._pending_acks] == [b"session-b"]
        assert arena._pending_ack_keys == {(b"session-b", "t2")}


class TestRouterMandatorySemantics:
    """Pin the libzmq behavior the ACK channel relies on."""

    def _pair(self, context):
        router, dealer = context.socket(zmq.ROUTER), context.socket(zmq.DEALER)
        for sock in (router, dealer):
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.SNDHWM, 1)
            sock.setsockopt(zmq.RCVHWM, 1)
        dealer.setsockopt(zmq.IDENTITY, b"draft")
        router.setsockopt(zmq.ROUTER_MANDATORY, 1)
        router.bind("inproc://test-router-mandatory")
        dealer.connect("inproc://test-router-mandatory")
        dealer.send(b"ready")
        assert router.poll(1000)
        router.recv_multipart()
        return router, dealer

    def test_full_peer_queue_raises_eagain(self):
        context = zmq.Context()
        router, dealer = self._pair(context)
        try:
            with pytest.raises(zmq.Again):
                for _ in range(10):
                    router.send_multipart([b"draft", b"x"], flags=zmq.NOBLOCK)
        finally:
            router.close(0)
            dealer.close(0)
            context.term()

    def test_departed_peer_raises_ehostunreach(self):
        context = zmq.Context()
        router, dealer = self._pair(context)
        dealer.close(0)
        try:
            for _ in range(50):
                router.poll(10)  # let the router process the pipe teardown
                try:
                    router.send_multipart([b"draft", b"x"], flags=zmq.NOBLOCK)
                except zmq.ZMQError as exc:
                    assert exc.errno == zmq.EHOSTUNREACH
                    break
            else:
                pytest.fail("departed peer never became unroutable")
        finally:
            router.close(0)
            context.term()


class TestPollBoundedDrain:
    def test_notice_drain_honors_time_cap(self):
        garbage = [(b"peer", b"not-a-notice")] * 1000
        socket = FakeSocket(recv_parts=garbage)
        arena = make_arena(socket)
        drained = arena.poll_completion_notices(max_messages=1000, max_ns=1)
        assert drained == 0  # all malformed; nothing accepted
        assert socket.recv_calls == 64  # stopped at the first deadline check


ROW_WIDTH = 8


def make_completion_arena(slot_count=1):
    """Arena carrying only the state the completion record paths touch."""

    arena = object.__new__(TargetCandidateArena)
    arena.slot_count = slot_count
    arena.row_width = ROW_WIDTH
    arena._lock = Lock()
    arena._slots = [_CandidateSlot() for _ in range(slot_count)]
    arena._active = {}
    arena._retired_transfers = OrderedDict()
    arena._completion_records = SimpleNamespace()  # sentinel: feature enabled
    arena._completion_words = np.zeros((slot_count, COMPLETION_RECORD_WORDS), dtype=np.uint64)
    arena._record_generation = np.zeros(slot_count, dtype=np.uint32)
    arena._record_seq_expected = np.zeros(slot_count, dtype=np.uint64)
    arena._record_hash_expected = np.zeros(slot_count, dtype=np.uint64)
    arena._record_tag = np.zeros(slot_count, dtype=np.uint32)
    arena._scanned_slot_epoch = 0
    arena._release_scan = (-1, True)
    arena._cancel_scan = (-1, True)
    arena._recent_notice_keys = deque()
    arena._recent_notice_set = set()
    arena._recent_notice_limit = 100
    return arena


def arm_completion_slot(
    arena,
    slot_id,
    *,
    transfer_id,
    generation,
    nonce,
    session="draft",
    state=CandidateSlotState.ARMED,
    cancel_pending=False,
):
    """Arm one slot the way reserve()+authorize() would."""

    slot = arena._slots[slot_id]
    slot.generation = generation
    slot.state = state
    slot.transfer_id = transfer_id
    slot.authorization_nonce = nonce
    slot.draft_instance_id = "d"
    slot.draft_session_id = session
    slot.cancel_pending = cancel_pending
    arena._active[transfer_id] = slot_id
    arena._record_generation[slot_id] = generation
    arena._record_seq_expected[slot_id] = nonce
    arena._record_hash_expected[slot_id] = transfer_id_hash(transfer_id)
    arena._record_tag[slot_id] = producer_tag(session)


def write_completion_record(
    arena,
    slot_id,
    *,
    nonce,
    generation,
    transfer_id,
    root=77,
    session="draft",
    produced=0,
    offset=0,
    release=True,
):
    """Emulate the producer's record writes: payload first, nonce last."""

    words = arena._completion_words[slot_id]
    words[COMPLETION_META_WORD] = generation | (ROW_WIDTH << 32)
    words[COMPLETION_HASH_WORD] = transfer_id_hash(transfer_id)
    words[COMPLETION_TOKEN_WORD] = (root & 0xFFFFFFFF) | (
        producer_tag(session) << 32
    )
    words[COMPLETION_PRODUCED_WORD] = produced
    words[COMPLETION_OFFSET_WORD] = offset
    if release:
        words[COMPLETION_SEQ_WORD] = nonce


class FakeRuntime:
    """CudaIpcRuntime stand-in capturing the stream-memop write sequence."""

    def __init__(self):
        self.registered = []
        self.unregistered = []
        self.writes = []

    def host_register(self, address, length):
        self.registered.append((address, length))
        return address

    def host_unregister(self, address):
        self.unregistered.append(address)

    def _stream_write_value32(self, stream, address, value, flags):
        self.writes.append((int(address.value), int(value.value)))
        return 0

    def _check_driver(self, rc, what):
        if rc:
            raise CandidateTransportError(f"{what} failed with {rc}")


def make_push_transport():
    transport = object.__new__(DraftCandidatePushTransport)
    transport.device_index = 0
    transport._producer_tag = producer_tag("draft")
    transport._runtime = FakeRuntime()
    return transport


class TestPublishMemopSequence:
    """The memop word sequence is the publish edge the Target acquires on."""

    def test_completion_record_words_and_release_order(self):
        transport = make_push_transport()
        nonce = (3 << 32) | 222
        destination = SimpleNamespace(
            authorization_nonce=nonce, generation=2, row_width=ROW_WIDTH
        )
        task = SimpleNamespace(transfer_id="tid-1", next_token_id=77)
        transport._publish_batch_memops(
            [(task, destination, 8192, 4096, _RECORD_KIND_COMPLETION)],
            SimpleNamespace(cuda_stream=0),
        )
        record_hash = transfer_id_hash("tid-1")
        assert transport._runtime.writes == [
            # Reservation metadata words, the generation word published last.
            (8192 + 4, 222),
            (8192 + 8, 3),
            (8192 + 12, ROW_WIDTH),
            (8192 + 16, 77),
            (8192 + 0, 2),
            # Completion record payload words; produced/offset zeroed.
            (4096 + 8, 2),
            (4096 + 12, ROW_WIDTH),
            (4096 + 16, record_hash & 0xFFFFFFFF),
            (4096 + 20, (record_hash >> 32) & 0xFFFFFFFF),
            (4096 + 24, 77),
            (4096 + 28, producer_tag("draft")),
            (4096 + 32, 0),
            (4096 + 36, 0),
            (4096 + 40, 0),
            (4096 + 44, 0),
            # The seq word released last, high half before low.
            (4096 + 4, 3),
            (4096 + 0, 222),
        ]

    def test_ready_record_words_and_release_order(self):
        transport = make_push_transport()
        nonce = (3 << 32) | 222
        destination = SimpleNamespace(
            authorization_nonce=nonce, generation=2, row_width=ROW_WIDTH
        )
        task = SimpleNamespace(transfer_id="tid-1", next_token_id=77)
        transport._publish_batch_memops(
            [(task, destination, 8192, 4096, _RECORD_KIND_READY)],
            SimpleNamespace(cuda_stream=0),
        )
        assert transport._runtime.writes[5:] == [
            # READY record: the metadata words minus generation, then the
            # producer tag, with the generation word published last.
            (4096 + 4, 222),
            (4096 + 8, 3),
            (4096 + 12, ROW_WIDTH),
            (4096 + 16, 77),
            (4096 + 20, producer_tag("draft")),
            (4096 + 0, 2),
        ]

    def test_row_without_record_writes_metadata_only(self):
        transport = make_push_transport()
        destination = SimpleNamespace(
            authorization_nonce=222, generation=2, row_width=ROW_WIDTH
        )
        task = SimpleNamespace(transfer_id="tid-1", next_token_id=77)
        transport._publish_batch_memops(
            [(task, destination, 8192, 0, 0)],
            SimpleNamespace(cuda_stream=0),
        )
        assert transport._runtime.writes == [
            (8192 + 4, 222),
            (8192 + 8, 0),
            (8192 + 12, ROW_WIDTH),
            (8192 + 16, 77),
            (8192 + 0, 2),
        ]


class TestCompletionRecords:
    def test_reused_slot_ignores_stale_record_then_accepts_fresh_release(self):
        arena = make_completion_arena()
        # A previous reservation's fully released record still occupies slot 0.
        write_completion_record(
            arena, 0, nonce=111, generation=1, transfer_id="old"
        )
        arm_completion_slot(arena, 0, transfer_id="new", generation=2, nonce=222)
        assert arena.poll_completion_records() == 0
        assert arena._slots[0].state is CandidateSlotState.ARMED
        # The new reservation's own release is accepted exactly once.
        write_completion_record(
            arena, 0, nonce=222, generation=2, transfer_id="new"
        )
        assert arena.poll_completion_records() == 1
        slot = arena._slots[0]
        assert slot.state is CandidateSlotState.READY
        assert slot.completion.ready_source == "completion_record"
        assert slot.completion.root_token == 77
        assert arena._record_seq_expected[0] == 0
        assert arena.poll_completion_records() == 0

    def test_stale_nonce_with_new_payload_is_inert(self):
        # The new kernel's first payload write (generation) has landed, but
        # the seq word still holds the previous reservation's nonce.
        arena = make_completion_arena()
        arm_completion_slot(arena, 0, transfer_id="new", generation=2, nonce=222)
        write_completion_record(
            arena, 0, nonce=111, generation=2, transfer_id="new"
        )
        assert arena.poll_completion_records() == 0
        assert arena._slots[0].state is CandidateSlotState.ARMED

    def test_payload_visible_before_nonce_release_is_inert(self):
        # Every payload word is visible, but the kernel has not released the
        # reservation nonce yet (seq word still zeroed).
        arena = make_completion_arena()
        arm_completion_slot(arena, 0, transfer_id="new", generation=2, nonce=222)
        write_completion_record(
            arena, 0, nonce=222, generation=2, transfer_id="new", release=False
        )
        assert arena.poll_completion_records() == 0
        assert arena._slots[0].state is CandidateSlotState.ARMED
        arena._completion_words[0, COMPLETION_SEQ_WORD] = 222
        assert arena.poll_completion_records() == 1
        assert arena._slots[0].state is CandidateSlotState.READY

    def test_nonce_high_word_collision_requires_full_release(self):
        # Nonces (1<<32)|7 then (2<<32)|7 share the low word.  The seq word
        # is one 64-bit release store, so the previous reservation's record
        # must be completely inert for the new one: neither its presence nor
        # a mid-publication payload update may mark READY.
        arena = make_completion_arena()
        write_completion_record(
            arena, 0, nonce=(1 << 32) | 7, generation=1, transfer_id="old"
        )
        arm_completion_slot(
            arena, 0, transfer_id="new", generation=2, nonce=(2 << 32) | 7
        )
        assert arena.poll_completion_records() == 0
        assert arena._slots[0].state is CandidateSlotState.ARMED
        # Mid-publication: the kernel's payload words are visible while the
        # seq word still holds the previous reservation's nonce.
        words = arena._completion_words[0]
        words[COMPLETION_META_WORD] = 2 | (ROW_WIDTH << 32)
        words[COMPLETION_HASH_WORD] = transfer_id_hash("new")
        words[COMPLETION_TOKEN_WORD] = 77 | (producer_tag("draft") << 32)
        assert arena.poll_completion_records() == 0
        assert arena._slots[0].state is CandidateSlotState.ARMED
        # Only the new reservation's own full 64-bit release completes it.
        words[COMPLETION_SEQ_WORD] = (2 << 32) | 7
        assert arena.poll_completion_records() == 1
        slot = arena._slots[0]
        assert slot.state is CandidateSlotState.READY
        assert slot.completion.transfer_id == "new"
        assert slot.completion.ready_source == "completion_record"

    def test_nonce_high_word_collision_cancel_retirement(self):
        # Same low-word collision on a cancel-pending slot: the stale record
        # must not retire it early; the in-flight push's own release must.
        arena = make_completion_arena()
        write_completion_record(
            arena, 0, nonce=(1 << 32) | 7, generation=1, transfer_id="old"
        )
        arm_completion_slot(
            arena,
            0,
            transfer_id="new",
            generation=2,
            nonce=(2 << 32) | 7,
            state=CandidateSlotState.QUARANTINED,
            cancel_pending=True,
        )
        assert arena.poll_completion_records() == 0
        slot = arena._slots[0]
        assert slot.state is CandidateSlotState.QUARANTINED
        assert slot.cancel_pending
        assert "new" in arena._active
        # The in-flight push's kernel publishes payload words, then the nonce.
        words = arena._completion_words[0]
        words[COMPLETION_META_WORD] = 2 | (ROW_WIDTH << 32)
        words[COMPLETION_HASH_WORD] = transfer_id_hash("new")
        words[COMPLETION_TOKEN_WORD] = 77 | (producer_tag("draft") << 32)
        words[COMPLETION_SEQ_WORD] = (2 << 32) | 7
        assert arena.poll_completion_records() == 1
        slot = arena._slots[0]
        assert slot.state is CandidateSlotState.FREE
        assert "new" not in arena._active
        assert "new" in arena._retired_transfers

    def test_wrong_transfer_hash_quarantines_once(self):
        arena = make_completion_arena()
        arm_completion_slot(arena, 0, transfer_id="new", generation=2, nonce=222)
        write_completion_record(
            arena, 0, nonce=222, generation=2, transfer_id="other"
        )
        with pytest.raises(CandidateTransportError, match="completion record"):
            arena.poll_completion_records()
        assert arena._slots[0].state is CandidateSlotState.QUARANTINED
        # The expectation was consumed: the failed record cannot re-raise.
        assert arena._record_seq_expected[0] == 0
        assert arena.poll_completion_records() == 0

    def test_cancel_pending_slot_retired_by_record(self):
        arena = make_completion_arena()
        arm_completion_slot(
            arena,
            0,
            transfer_id="new",
            generation=2,
            nonce=222,
            state=CandidateSlotState.QUARANTINED,
            cancel_pending=True,
        )
        # The cancelled push's kernel still publishes the original nonce.
        write_completion_record(
            arena, 0, nonce=222, generation=2, transfer_id="new"
        )
        assert arena.poll_completion_records() == 1
        slot = arena._slots[0]
        assert slot.state is CandidateSlotState.FREE
        assert "new" not in arena._active
        assert "new" in arena._retired_transfers
        assert arena._record_seq_expected[0] == 0
        assert arena._record_generation[0] == 0

    def test_completion_record_is_authoritative_over_legacy_ready(self):
        # Both record paths are available; only the Q6 record was published
        # (a Q6-negotiated producer never writes legacy READY records).
        arena = make_completion_arena()
        arena._ready_records = SimpleNamespace(
            words=np.zeros((1, 8), dtype=np.uint32)
        )
        arm_completion_slot(arena, 0, transfer_id="new", generation=2, nonce=222)
        write_completion_record(
            arena, 0, nonce=222, generation=2, transfer_id="new"
        )
        assert arena.poll_ready_records() == 0
        assert arena.poll_completion_records() == 1
        assert arena._slots[0].completion.ready_source == "completion_record"

    def test_delivery_latency_from_produced_globaltimer(self, monkeypatch):
        discovery_ns = 1_000_000_000
        monkeypatch.setattr(time, "time_ns", lambda: discovery_ns)
        arena = make_completion_arena()
        arm_completion_slot(arena, 0, transfer_id="new", generation=2, nonce=222)
        write_completion_record(
            arena,
            0,
            nonce=222,
            generation=2,
            transfer_id="new",
            produced=500,
            offset=1000,
        )
        assert arena.poll_completion_records() == 1
        completion = arena._slots[0].completion
        assert completion.producer_globaltimer_ns == 500
        assert completion.delivery_ns == discovery_ns - (500 + 1000)
        # The monotonic observed chain is untouched by the realtime stamp.
        assert completion.observed_ns == completion.metadata_submitted_ns

    def test_delivery_ns_zero_without_producer_offset(self):
        # An uncalibrated producer stamps offset 0; the stamp itself is kept.
        arena = make_completion_arena()
        arm_completion_slot(arena, 0, transfer_id="new", generation=2, nonce=222)
        write_completion_record(
            arena, 0, nonce=222, generation=2, transfer_id="new", produced=500
        )
        assert arena.poll_completion_records() == 1
        completion = arena._slots[0].completion
        assert completion.producer_globaltimer_ns == 500
        assert completion.delivery_ns == 0

    def test_legacy_paths_default_delivery_fields(self):
        arena = make_completion_arena()
        arena._ready_records = SimpleNamespace(
            words=np.array(
                [[2, 222, 0, ROW_WIDTH, 77, producer_tag("draft"), 0, 0]],
                dtype=np.uint32,
            )
        )
        arm_completion_slot(arena, 0, transfer_id="new", generation=2, nonce=222)
        assert arena.poll_ready_records() == 1
        completion = arena._slots[0].completion
        assert completion.ready_source == "gpu_record"
        assert completion.producer_globaltimer_ns == 0
        assert completion.delivery_ns == 0

        arena = make_completion_arena()
        arm_completion_slot(arena, 0, transfer_id="new", generation=2, nonce=222)
        slot = arena._slots[0]
        # The notice fast path still requires the readback resources to exist.
        slot.wait_stream = SimpleNamespace()
        slot.metadata_host = SimpleNamespace()
        arena.notify_host_completion(
            "new",
            draft_instance_id="d",
            draft_session_id="draft",
            valid_length=ROW_WIDTH,
            root_token=77,
        )
        completion = arena._slots[0].completion
        assert completion.ready_source == "notice"
        assert completion.producer_globaltimer_ns == 0
        assert completion.delivery_ns == 0

    def test_close_releases_completion_record_mapping(self):
        path = f"/dev/shm/dt-q6-test-{uuid.uuid4().hex}"
        records = _ReadyRecordFile(
            path, record_bytes=COMPLETION_RECORD_BYTES, slot_count=1, create=True
        )
        arena = make_completion_arena()
        arena._completion_records = records
        arena.completion_record_path = path
        arena._completion_words = records.words.view(np.uint64).reshape(
            records.slot_capacity, COMPLETION_RECORD_WORDS
        )
        arena._closed = False
        arena._ready_records = None
        unregistered = []
        arena._runtime = SimpleNamespace(
            host_unregister=unregistered.append
        )
        arena._completion_socket = SimpleNamespace(close=lambda **kwargs: None)
        try:
            arena.close()
            assert arena._completion_records is None
            assert arena._completion_words is None
            assert arena.completion_record_path == ""
            assert records._mm.closed
            assert unregistered == [records.host_address]
            assert not os.path.exists(path)
        finally:
            if not records._mm.closed:
                records._mm.close()
            if os.path.exists(path):
                os.unlink(path)


class ReadyEvent:
    """Completion event whose stream work is already done."""

    def query(self):
        return True


def make_notice_arena(slot_count=1):
    """Arena carrying only the state poll()'s readback path touches."""

    arena = object.__new__(TargetCandidateArena)
    arena.slot_count = slot_count
    arena.row_width = ROW_WIDTH
    arena._lock = Lock()
    arena._slots = [_CandidateSlot() for _ in range(slot_count)]
    arena._active = {}
    arena._retired_transfers = OrderedDict()
    arena._event_pool = []
    arena._poll_cursor = 0
    arena._ready_pickup_pending = False
    return arena


def arm_notice_slot(arena, slot_id, *, transfer_id, generation, nonce, valid_length):
    """Arm one slot the way a cancelled-notice readback would leave it."""

    slot = arena._slots[slot_id]
    slot.generation = generation
    slot.state = CandidateSlotState.ARMED
    slot.transfer_id = transfer_id
    slot.authorization_nonce = nonce
    slot.draft_instance_id = "d"
    slot.draft_session_id = "draft"
    slot.completion_draft_instance_id = "d"
    slot.completion_draft_session_id = "draft"
    slot.completion_event = ReadyEvent()
    slot.metadata_host = [
        generation,
        nonce & 0xFFFFFFFF,
        (nonce >> 32) & 0xFFFFFFFF,
        valid_length,
        77,
    ]
    arena._active[transfer_id] = slot_id


class TestProducerTombstoneOnArmedSlot:
    """Issue #178: an authenticated producer tombstone (valid length 0) that
    arrives while this side still holds the slot ARMED retires the slot. It
    must not fail the arena with a generation/authentication error — the
    overloaded point that races cancellations this way used to crash it."""

    def test_notice_tombstone_on_armed_slot_retires_it(self):
        arena = make_notice_arena()
        arm_notice_slot(arena, 0, transfer_id="t1", generation=2, nonce=222,
                        valid_length=0)
        assert arena.poll() == ()
        slot = arena._slots[0]
        assert slot.state is CandidateSlotState.FREE
        assert "t1" not in arena._active
        assert "t1" in arena._retired_transfers

    def test_notice_candidate_still_completes(self):
        arena = make_notice_arena()
        arm_notice_slot(arena, 0, transfer_id="t1", generation=2, nonce=222,
                        valid_length=ROW_WIDTH)
        (completion,) = arena.poll()
        assert completion.transfer_id == "t1"
        assert arena._slots[0].state is CandidateSlotState.READY

    def test_notice_bad_nonce_still_quarantines(self):
        arena = make_notice_arena()
        arm_notice_slot(arena, 0, transfer_id="t1", generation=2, nonce=222,
                        valid_length=ROW_WIDTH)
        arena._slots[0].authorization_nonce = 999
        with pytest.raises(CandidateTransportError, match="generation/authentication"):
            arena.poll()
        assert arena._slots[0].state is CandidateSlotState.QUARANTINED

    def test_record_tombstone_on_armed_slot_retires_it(self):
        arena = make_completion_arena()
        arm_completion_slot(arena, 0, transfer_id="new", generation=2, nonce=222)
        words = arena._completion_words[0]
        # The producer's tombstone: authenticated words, valid length 0.
        words[COMPLETION_META_WORD] = 2
        words[COMPLETION_HASH_WORD] = transfer_id_hash("new")
        words[COMPLETION_TOKEN_WORD] = 77 | (producer_tag("draft") << 32)
        words[COMPLETION_SEQ_WORD] = 222
        assert arena.poll_completion_records() == 1
        slot = arena._slots[0]
        assert slot.state is CandidateSlotState.FREE
        assert "new" not in arena._active
        assert "new" in arena._retired_transfers

    def test_record_tombstone_on_cancel_pending_slot_retires_it(self):
        arena = make_completion_arena()
        arm_completion_slot(
            arena,
            0,
            transfer_id="new",
            generation=2,
            nonce=222,
            state=CandidateSlotState.QUARANTINED,
            cancel_pending=True,
        )
        words = arena._completion_words[0]
        words[COMPLETION_META_WORD] = 2
        words[COMPLETION_HASH_WORD] = transfer_id_hash("new")
        words[COMPLETION_TOKEN_WORD] = 77 | (producer_tag("draft") << 32)
        words[COMPLETION_SEQ_WORD] = 222
        assert arena.poll_completion_records() == 1
        slot = arena._slots[0]
        assert slot.state is CandidateSlotState.FREE
        assert "new" not in arena._active
        assert "new" in arena._retired_transfers


OBSERVER_CXX_AVAILABLE = shutil.which(os.environ.get("CXX", "g++")) is not None


def make_observed_arena(slot_count=2):
    """Bare completion arena wired to a real native observer over numpy."""

    arena = make_completion_arena(slot_count)
    completion_words = np.zeros((slot_count, COMPLETION_RECORD_WORDS), dtype=np.uint64)
    ready_words = np.zeros((slot_count, READY_RECORD_WORDS), dtype=np.uint32)
    arena._completion_words = completion_words
    arena._ready_records = SimpleNamespace(words=ready_words)
    observer = _RecordObserver(
        slot_count=slot_count,
        completion_words=completion_words,
        ready_words=ready_words,
        record_seq_expected=arena._record_seq_expected,
        record_generation=arena._record_generation,
        record_hash_expected=arena._record_hash_expected,
        record_tag=arena._record_tag,
        row_width=ROW_WIDTH,
    )
    arena._observer = observer
    arena._observer_dropped_seen = 0
    return arena, observer


def poll_observed(arena, observer, method, deadline_s=10.0):
    """Drains until the observer-recorded arrival lands (the native thread
    needs a scheduling quantum to see the record)."""

    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        if method() == 1:
            return True
        assert observer.dropped == arena._observer_dropped_seen
        time.sleep(0.001)
    return False


@pytest.mark.skipif(
    not OBSERVER_CXX_AVAILABLE, reason="needs a C++ compiler for the observer"
)
class TestRecordObserver:
    def test_observer_records_completion_once(self):
        arena, observer = make_observed_arena()
        try:
            arm_completion_slot(arena, 0, transfer_id="t", generation=2, nonce=111)
            write_completion_record(arena, 0, nonce=111, generation=2, transfer_id="t")
            assert poll_observed(arena, observer, arena.poll_completion_records)
            # The record keeps matching, but is consumed exactly once.
            assert arena.poll_completion_records() == 0
            assert arena._record_seq_expected[0] == 0
            slot = arena._slots[0]
            assert slot.state is CandidateSlotState.READY
            assert slot.completion.ready_source == "completion_record"
            assert slot.completion.observed_ns > 0
        finally:
            observer.destroy()

    def test_observer_gen_mismatch_then_rearm(self):
        arena, observer = make_observed_arena()
        try:
            arm_completion_slot(arena, 0, transfer_id="t", generation=2, nonce=111)
            # Wrong generation in the record: no arrival, expectation kept.
            write_completion_record(arena, 0, nonce=111, generation=3, transfer_id="t")
            time.sleep(0.02)
            assert arena.poll_completion_records() == 0
            assert arena._record_seq_expected[0] == 111
            # Disarm and rearm with a fresh reservation: the new record is
            # observed even though the slot reported the stale one already.
            arena._record_seq_expected[0] = 0
            arena._record_generation[0] = 0
            arm_completion_slot(arena, 0, transfer_id="u", generation=4, nonce=222)
            write_completion_record(arena, 0, nonce=222, generation=4, transfer_id="u")
            assert poll_observed(arena, observer, arena.poll_completion_records)
            assert arena._slots[0].completion.transfer_id == "u"
        finally:
            observer.destroy()

    def test_observer_wrong_producer_tag_quarantines_once(self):
        # A record with the reservation's nonce and transfer but another
        # producer's tag fails native authentication: quarantine, consume
        # once, and the authenticated record of a rearm is still accepted.
        arena, observer = make_observed_arena()
        try:
            arm_completion_slot(arena, 0, transfer_id="t", generation=2, nonce=111)
            write_completion_record(
                arena, 0, nonce=111, generation=2, transfer_id="t", session="other"
            )
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                try:
                    arena.poll_completion_records()
                except CandidateTransportError as exc:
                    assert "completion record" in str(exc)
                    break
                time.sleep(0.001)
            else:
                pytest.fail("forged producer tag was not quarantined")
            assert arena._slots[0].state is CandidateSlotState.QUARANTINED
            # The failed record cannot re-raise on every poll.
            assert arena._record_seq_expected[0] == 0
            assert arena.poll_completion_records() == 0
        finally:
            observer.destroy()

    def test_observer_record_beats_ack_then_authorized(self):
        # The record lands while the expectations are still unarmed (the
        # assignment ACK has not been processed): it must stay inert, and
        # arming the expectations (authorize) plus the kick must make the
        # observer re-check the slot without waiting for a backoff cycle.
        arena, observer = make_observed_arena()
        try:
            slot = arena._slots[0]
            slot.generation = 2
            slot.state = CandidateSlotState.ARMED
            slot.transfer_id = "t"
            slot.authorization_nonce = 111
            slot.draft_instance_id = "d"
            slot.draft_session_id = "draft"
            arena._active["t"] = 0
            arena._record_hash_expected[0] = transfer_id_hash("t")
            # No _record_seq_expected/_record_generation/_record_tag yet:
            # the reservation is armed but the assignment is not.
            write_completion_record(arena, 0, nonce=111, generation=2, transfer_id="t")
            time.sleep(0.02)
            assert arena.poll_completion_records() == 0
            assert arena._slots[0].state is CandidateSlotState.ARMED
            # authorize() arms the expectations and kicks the observer.
            arena._record_tag[0] = producer_tag("draft")
            arena._record_generation[0] = 2
            arena._record_seq_expected[0] = 111
            observer.kick()
            assert poll_observed(arena, observer, arena.poll_completion_records)
            slot = arena._slots[0]
            assert slot.state is CandidateSlotState.READY
            assert slot.completion.transfer_id == "t"
        finally:
            observer.destroy()

    def test_observer_records_legacy_ready_record(self):
        arena, observer = make_observed_arena()
        try:
            arm_completion_slot(arena, 0, transfer_id="t", generation=2, nonce=111)
            values = arena._ready_records.words[0]
            values[1] = 111 & 0xFFFFFFFF
            values[2] = 111 >> 32
            values[3] = ROW_WIDTH
            values[4] = 77
            values[5] = producer_tag("draft")
            values[0] = 2  # generation published last
            assert poll_observed(arena, observer, arena.poll_ready_records)
            assert arena.poll_ready_records() == 0
            completion = arena._slots[0].completion
            assert completion.ready_source == "gpu_record"
            assert completion.root_token == 77
        finally:
            observer.destroy()

    def test_observer_destroy_stops_draining(self):
        arena, observer = make_observed_arena()
        # close() destroys the observer and drops it; polls then fall back
        # to the legacy scan and stay consistent.
        observer.destroy()
        arena._observer = None
        assert observer.drain(_RecordObserver.KIND_COMPLETION) == []
        arm_completion_slot(arena, 0, transfer_id="t", generation=2, nonce=111)
        write_completion_record(arena, 0, nonce=111, generation=2, transfer_id="t")
        assert arena.poll_completion_records() == 1
        observer.destroy()  # idempotent
