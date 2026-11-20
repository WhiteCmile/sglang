"""CPU contract of the native Draft control thread (csrc/dt_draft_control.cpp,
issue #288; design §8.1, §9 R4).

Against a ROUTER standing in for the Coordinator:
- a Coordinator frame reaches the scheduler, with a CONTROL wake, while the
  scheduler thread holds the GIL (a Python control thread would be starved:
  the switch interval is raised so it could not run);
- the heartbeat is the native timer's, re-stamped after a re-registration;
- a send queued right before close() leaves, session-stamped, ahead of the
  shutdown notice.
"""

import sys
import threading
import time

import zmq
from dt_client import DraftCoordinatorClient
from dt_protocol import (
    PROTOCOL_VERSION,
    ControlAck,
    DraftCapacityUpdate,
    InstanceHeartbeat,
    InstanceShutdown,
    RegisterAck,
    RegisterInstance,
    decode_message,
    encode_message,
)

from sglang.srt.speculative.dt_control import DraftControlLoop
from sglang.srt.speculative.dt_wake import WAKE_CONTROL, DraftWake


def _next(router, kind, deadline_s=2.0, where=lambda m: True):
    """The next frame of ``kind`` the Draft sent, skipping others."""

    deadline = time.monotonic() + deadline_s
    while True:
        left_ms = int((deadline - time.monotonic()) * 1000)
        assert left_ms > 0, f"no {kind.__name__} arrived"
        if router.poll(left_ms):
            identity, frame = router.recv_multipart()
            message = decode_message(frame)
            if isinstance(message, kind) and where(message):
                return identity, message


def _answer_registration(router, coordinator_session_id):
    def answer():
        identity, message = _next(router, RegisterInstance)
        ack = RegisterAck(
            protocol_version=PROTOCOL_VERSION,
            instance_id=message.instance_id,
            accepted=True,
            coordinator_session_id=coordinator_session_id,
        )
        router.send_multipart([identity, encode_message(ack)])

    thread = threading.Thread(target=answer)
    thread.start()
    return thread


def test_native_control_thread_owns_the_draft_socket(tmp_path):
    context = zmq.Context()
    router = context.socket(zmq.ROUTER)
    router.setsockopt(zmq.LINGER, 0)
    endpoint = f"ipc://{tmp_path / 'draft.sock'}"
    router.bind(endpoint)
    # An ipc endpoint counts as remote, so this client sends heartbeats.
    client = DraftCoordinatorClient(
        endpoint=endpoint,
        instance_id="draft-0",
        draft_profile_ids=("flashmtp-qwen3-14b-v2",),
        max_pending_tasks=8,
    )
    wake = DraftWake()
    loop = None
    try:
        answer = _answer_registration(router, "session-1")
        client.register_draft(timeout_s=2)
        answer.join()
        loop = DraftControlLoop(client, wake)

        identity, _ = _next(
            router,
            InstanceHeartbeat,
            where=lambda m: m.coordinator_session_id == "session-1",
        )
        epoch = wake.epoch()
        ack = ControlAck(
            protocol_version=PROTOCOL_VERSION, operation="probe", message_id="m-1"
        )
        router.send_multipart([identity, encode_message(ack)])
        interval = sys.getswitchinterval()
        sys.setswitchinterval(30.0)
        try:
            started = time.monotonic()
            messages = []
            while not messages:
                assert time.monotonic() - started < 5.0, "frame never delivered"
                messages = loop.recv_nowait()
            delivered_s = time.monotonic() - started
        finally:
            sys.setswitchinterval(interval)
        assert messages == [ack]
        assert delivered_s < 1.0
        assert wake.wait(epoch, 0.001) & WAKE_CONTROL

        answer = _answer_registration(router, "session-2")
        loop.register_draft(timeout_s=2)
        answer.join()
        _next(
            router,
            InstanceHeartbeat,
            where=lambda m: m.coordinator_session_id == "session-2",
        )

        update = DraftCapacityUpdate(
            protocol_version=PROTOCOL_VERSION,
            draft_instance_id="draft-0",
            pending_tasks=0,
            max_pending_tasks=8,
            free_result_rows=8,
            received_tasks=0,
            capacity_sequence=7,
            free_request_rows=8,
            free_kv_pages=1024,
            pending_work=0,
        )
        assert loop.update_capacity(update)
        loop.close()
        loop = None
        seen = []
        while not seen or not isinstance(seen[-1], (InstanceShutdown)):
            _, message = _next(router, object)
            if not isinstance(message, InstanceHeartbeat):
                seen.append(message)
        assert len(seen) == 2 and isinstance(seen[0], DraftCapacityUpdate)
        assert seen[0].capacity_sequence == 7
        assert seen[0].instance_session_id == client.instance_session_id
        assert seen[0].coordinator_session_id == "session-2"
    finally:
        if loop is not None:
            loop.close()
        wake.close()
        router.close()
        context.term()
