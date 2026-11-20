"""Native owner of the Draft's Coordinator socket (csrc/dt_draft_control.cpp).

Draft redesign D5b (issue #288; design §8.1, §9 R4).  After construction a
native thread owns the registered Draft DEALER socket and never takes the
GIL; it moves raw frames between the socket and two SPSC rings and sets
``WAKE_CONTROL`` on the scheduler's wake word per received batch.  Decoding,
encoding and session stamping stay on the scheduler thread, as with the plain
client.  A failure of the native thread is raised once, at the scheduler's
next call.
"""

from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path

import zmq
from dt_client import CoordinatorClientError, DraftCoordinatorClient, RegistrationError
from dt_protocol import (
    PROTOCOL_VERSION,
    REMOTE_HEARTBEAT_INTERVAL_S,
    InstanceHeartbeat,
    RegisterAck,
    WireMessage,
    decode_message,
)

from sglang.srt.speculative.dt_wake import WAKE_CONTROL, DraftWake

logger = logging.getLogger(__name__)


def _build_control_library() -> Path:
    from sglang.srt.speculative.dt_direct_ring import _build_native_library

    source = Path(__file__).with_name("csrc") / "dt_draft_control.cpp"
    return _build_native_library(
        "dt_draft_control",
        source,
        [
            os.environ.get("CXX", "g++"),
            "-O2",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-ldl",
            "-lpthread",
        ],
        ["--version"],
        "DT_WAKE_BUILD_DIR",
    )


def _pyzmq_extension() -> str:
    """pyzmq's backend extension: symbols looked up through it resolve to the
    libzmq that created the sockets (another libzmq may be loaded too)."""

    import zmq.backend.cython._zmq as backend

    return backend.__file__


class DraftControlLoop:
    def __init__(self, client: DraftCoordinatorClient, wake: DraftWake) -> None:
        if client.registered_instance_id != client.instance_id:
            raise RegistrationError("Draft client must be registered first")
        self.instance_id = client.instance_id
        self._client = client
        # First, so no heartbeat thread touches the socket from here on.
        self._heartbeats = client.adopt_heartbeat()
        socket = client._socket
        # PyDLL: every call below is short and non-blocking (dt_ctl_stop
        # joins a thread that needs no GIL), so it keeps the GIL.
        lib = ctypes.PyDLL(str(_build_control_library()))
        vp, u32, u64, i64 = (
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_int64,
        )
        lib.dt_ctl_create.argtypes = [ctypes.c_char_p, vp, vp, vp, u32, u64, u64]
        lib.dt_ctl_create.restype = vp
        lib.dt_ctl_start.argtypes = [vp, ctypes.c_char_p, u64, i64]
        lib.dt_ctl_start.restype = None
        lib.dt_ctl_stop.argtypes = [vp]
        lib.dt_ctl_stop.restype = ctypes.c_int
        lib.dt_ctl_destroy.argtypes = [vp]
        lib.dt_ctl_destroy.restype = None
        lib.dt_ctl_send.argtypes = [vp, ctypes.c_char_p, u64]
        lib.dt_ctl_send.restype = ctypes.c_int
        lib.dt_ctl_front.argtypes = [vp, ctypes.POINTER(vp)]
        lib.dt_ctl_front.restype = i64
        lib.dt_ctl_pop.argtypes = [vp]
        lib.dt_ctl_pop.restype = None
        lib.dt_ctl_heartbeats_refused.argtypes = [vp]
        lib.dt_ctl_heartbeats_refused.restype = u64
        self._lib = lib
        wake_set, wake_handle = wake.native_setter()
        self._handle = lib.dt_ctl_create(
            _pyzmq_extension().encode(),
            socket.underlying,
            wake_set,
            wake_handle,
            WAKE_CONTROL,
            socket.getsockopt(zmq.SNDHWM),
            socket.getsockopt(zmq.RCVHWM),
        )
        if not self._handle:
            raise CoordinatorClientError("cannot create the Draft control thread")
        self._frame = ctypes.c_void_p()
        self._failed = False
        self._closed = False
        self._start()

    @property
    def registered_instance_id(self) -> str:
        return self.instance_id

    @property
    def instance_session_id(self) -> str:
        return self._client.instance_session_id

    @property
    def coordinator_session_id(self) -> str:
        return self._client.coordinator_session_id

    def _start(self) -> None:
        heartbeat = b""
        if self._heartbeats:
            # Stamped with the current sessions, so re-encoded per start.
            heartbeat = self._client.encode(
                InstanceHeartbeat(
                    protocol_version=PROTOCOL_VERSION, instance_id=self.instance_id
                )
            )
        self._lib.dt_ctl_start(
            self._handle,
            heartbeat,
            len(heartbeat),
            int(REMOTE_HEARTBEAT_INTERVAL_S * 1_000_000_000) if heartbeat else 0,
        )

    def _stop(self) -> None:
        error = self._lib.dt_ctl_stop(self._handle)
        if error and not self._failed:
            self._raise_failed(error)

    def _raise_failed(self, error: int) -> None:
        # Raised once.  The teardown after it then sees refused sends, like a
        # legacy client whose Coordinator is gone, instead of the same error.
        self._failed = True
        raise CoordinatorClientError(
            f"Draft control thread failed: {os.strerror(error)}"
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            # Every frame queued so far leaves before the shutdown notice.
            self._stop()
        finally:
            refused = self._lib.dt_ctl_heartbeats_refused(self._handle)
            self._lib.dt_ctl_destroy(self._handle)
            self._handle = None
            if refused:
                logger.warning(
                    "DT Draft %s: %d heartbeat sends refused at the socket's "
                    "send high-water mark",
                    self.instance_id,
                    refused,
                )
            self._client.close()

    def recv_nowait(self, *, max_messages: int = 256) -> list[WireMessage]:
        messages: list[WireMessage] = []
        lib, handle, frame = self._lib, self._handle, self._frame
        while len(messages) < max_messages:
            size = lib.dt_ctl_front(handle, ctypes.byref(frame))
            if size < 0:
                if size == -2 and not self._failed:
                    self._raise_failed(lib.dt_ctl_stop(handle))
                break
            messages.append(decode_message(ctypes.string_at(frame.value, size)))
            lib.dt_ctl_pop(handle)
        return messages

    def register_draft(self, *, timeout_s: float = 10.0) -> RegisterAck:
        """Re-register on the caller's thread, after the frames queued so far.

        The native thread stops (after sending them), the plain client
        registers on the socket, and the thread restarts with the new
        session's heartbeat; every later send is stamped with that session.
        A frame refused at the send high-water mark when the thread stopped
        leaves after the registration, with the old session.
        """

        self._stop()
        try:
            return self._client.register_draft(timeout_s=timeout_s)
        finally:
            self._start()

    def send(self, message: WireMessage) -> bool:
        if self._closed:
            raise CoordinatorClientError("Draft control loop is closed")
        payload = self._client.encode(message)
        queued = self._lib.dt_ctl_send(self._handle, payload, len(payload))
        if queued < 0:
            if not self._failed:
                self._raise_failed(-queued)
            return False
        return queued == 1

    # The Draft engine's typed senders.  The plain client's per-type checks
    # that a message names this Draft guard programming errors only; the
    # Coordinator checks every message's sender route as well.
    update_capacity = publish_results = source_read_done = send
    source_reads_done = migration_status = migration_finalized = send
