"""P3 (issue #328, DT_TARGET_POLL_SPLIT=1): the per-loop poll splits into a
loop-head ready_sync (completion snapshot + arena poll + mark-ready, no
Coordinator I/O or cleanup before the READY snapshot) and a post-launch,
TP0-only control phase (Coordinator recv, flush, ring cleanup, releases,
read proofs, expiry).  The control phase's TP-relevant events ride the next
ready_sync packet ahead of its fresh events, preserving the unsplit poll's
Coordinator-before-arena apply order on every rank.
"""

from collections import deque
from types import SimpleNamespace

import sglang.srt.speculative.dt_target as dt_target_module

from sglang.srt.speculative.dt_target import DTTargetController


def _bare_controller(*, split: bool) -> DTTargetController:
    controller = object.__new__(DTTargetController)
    controller._target_poll_split = split
    controller.is_export_rank = True
    controller._profile_active = False
    controller._profile_emit = lambda *args, **kwargs: None
    controller._profile_flush = lambda *args, **kwargs: None
    controller.scheduler = SimpleNamespace(
        ps=SimpleNamespace(tp_size=1),
        tp_group=SimpleNamespace(rank=0, ranks=[0]),
        tp_cpu_group=None,
    )
    controller._last_poll_ns = 0
    controller._overlap_launch_ns = 0
    controller._feature_source_pool = None
    controller._ring_publish_fatal = None
    controller._ring_publish_pending = deque()
    controller._late_freeze = False
    controller._candidate_arena = None
    controller._parked = {}
    controller._feature_control_rounds = {}
    controller._lifecycle_attempt_by_request = {}
    controller._draft_admitted_attempts = {}
    return controller


def _parked_state(rid: str, *, bonus_token: int = 11, seq_len: int = 5):
    return SimpleNamespace(
        req=SimpleNamespace(rid=rid),
        attempt=1,
        bonus_token=bonus_token,
        seq_len=seq_len,
        candidate_bytes=None,
        candidate_completion=None,
        draft_admitted=False,
        failure=None,
    )


def _control_stubs(controller, calls, events):
    """Record the control-phase call order; feed the recv/expire events."""
    controller._ring_publish_cleanup = lambda: calls.append("ring_cleanup")
    controller._flush_pending_send = lambda: calls.append("flush_send")
    controller._poll_ring_read_proofs = lambda: calls.append("read_proofs")

    def recv(out):
        calls.append("zmq_recv")
        out.extend(events)

    controller._recv_coordinator_events = recv
    controller._expire_local_rounds = lambda: calls.append("expire_rounds") or []
    controller._expire_source_leases = (
        lambda *, now_ns: calls.append("expire_leases")
    )
    controller._maybe_emit_lifecycle_status = lambda: calls.append("lifecycle")
    arena = SimpleNamespace(
        poll_releases=lambda: calls.append("releases"),
        poll_cancelled=lambda: calls.append("cancelled"),
    )
    controller._candidate_arena = arena
    return arena


def test_poll_split_off_keeps_monolith_and_control_phase_noop() -> None:
    controller = _bare_controller(split=False)
    marker = [("ready", "t0", (b"x", 11, 5))]
    calls = []
    controller._poll_export_rank = lambda: (calls.append("poll") or marker, {})
    controller._poll_ring_completions = lambda: calls.append("completions")
    controller._recv_coordinator_events = lambda out: calls.append("zmq_recv")

    controller.ready_sync()
    assert calls == ["poll"]

    controller.control_phase()
    assert calls == ["poll"], "control phase must be a no-op with the flag off"

    # poll() stays the unsplit entry point (event_loop_normal).
    controller.poll()
    assert calls == ["poll", "poll"]


def test_poll_split_ready_sync_defers_coordinator_io_to_control_phase() -> None:
    controller = _bare_controller(split=True)
    calls = []
    controller._ring_publish_pending = deque([object()])
    _control_stubs(controller, calls, [])
    controller._poll_ring_completions = lambda: calls.append("completions")
    controller._poll_arena_ready = lambda out: calls.append("arena_ready")

    controller.ready_sync()
    assert calls == ["completions", "arena_ready"], (
        "ready_sync must not run Coordinator I/O, cleanup, releases or expiry"
    )

    controller.control_phase()
    assert calls == [
        "completions",
        "arena_ready",
        "ring_cleanup",
        "flush_send",
        "read_proofs",
        "zmq_recv",
        "releases",
        "cancelled",
        "expire_rounds",
        "expire_leases",
        "lifecycle",
    ]

    # Non-export ranks never run the control phase.
    controller.is_export_rank = False
    calls.clear()
    controller.control_phase()
    assert calls == []


def test_poll_split_control_events_ride_next_ready_sync_in_order(
    monkeypatch,
) -> None:
    controller = _bare_controller(split=True)
    packets = []

    def fake_broadcast(packet, rank, group, src):
        packets.append(packet)
        return packet

    monkeypatch.setattr(dt_target_module, "broadcast_pyobj", fake_broadcast)
    controller.scheduler.ps.tp_size = 2

    calls = []
    deferred = [
        ("failed", "t1", "draft failed"),
        ("accepted", "t2", None),
    ]
    _control_stubs(controller, calls, deferred)
    completion = SimpleNamespace(observed_ns=123)
    controller._poll_ring_completions = lambda: calls.append("completions")

    def arena_ready(out):
        calls.append("arena_ready")
        out.append(("ready_direct", "t3", (completion, 11, 5)))

    controller._poll_arena_ready = arena_ready
    controller._parked = {
        "t1": _parked_state("r1"),
        "t2": _parked_state("r2"),
        "t3": _parked_state("r3"),
    }

    # Loop head: the stash starts empty, so only the fresh event rides the
    # packet.
    controller.ready_sync()
    assert [(kind, tid) for kind, tid, _ in packets[0][1]] == [("ready_direct", "t3")]

    # Post-launch control phase: events are stashed, not applied.
    controller.control_phase()
    assert controller._parked["t1"].failure is None
    assert not controller._parked["t2"].draft_admitted

    # Next loop head: the stashed control events lead the packet, ahead of
    # the fresh arena READY — the unsplit poll's Coordinator-before-arena
    # order.
    controller.ready_sync()
    kinds = [kind for kind, _tid, _payload in packets[1][1]]
    tids = [tid for _kind, tid, _payload in packets[1][1]]
    assert list(zip(kinds, tids)) == [
        ("failed", "t1"),
        ("accepted", "t2"),
        ("ready_direct", "t3"),
    ]
    assert controller._parked["t1"].failure == "draft failed"
    assert controller._parked["t2"].draft_admitted
    assert controller._parked["t3"].candidate_completion is completion

    # The stash drained: a quiet control phase leaves no residue, only the
    # loop's own fresh events ride the packet.
    packets.clear()
    deferred.clear()
    controller.control_phase()
    controller.ready_sync()
    kinds = [kind for kind, _tid, _payload in packets[0][1]]
    assert kinds == ["ready_direct"]
