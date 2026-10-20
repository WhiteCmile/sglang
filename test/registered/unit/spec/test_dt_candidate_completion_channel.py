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

The arena/transport constructors need CUDA, so objects are built with
object.__new__ plus only the attributes the methods under test touch.
"""

import time
from collections import deque
from types import SimpleNamespace

import pytest
import zmq

pytest.importorskip("dt_protocol")

from sglang.srt.speculative.dt_candidate_p2p import (
    COMPLETION_PENDING_ACK_LIMIT,
    COMPLETION_RETRY_EAGAIN_LIMIT,
    COMPLETION_RETRY_NS,
    CandidateTransportError,
    DraftCandidatePushTransport,
    TargetCandidateArena,
    _PendingCompletionNotice,
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
