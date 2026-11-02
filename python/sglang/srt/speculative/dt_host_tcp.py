"""Host-staged TCP fallback for cross-host DT data planes (issue #164).

Same-host Draft-Target pairs keep the CUDA-IPC paths unchanged.  When a
Target's arena registration carries a ``host_addr`` that differs from the
Draft's own ``--dt-host-addr``, the Draft falls back to this framed TCP
protocol for the two cross-host data legs:

- ``hidden_pull_req`` / ``hidden_pull_resp``: the Draft reads one published
  Target feature range (raw little-endian BF16 rows exactly as laid out in
  GPU memory).
- ``cand_push`` / ``cand_push_resp``: the Draft sends one candidate row plus
  its metadata words; the Target copies them into its own arena and then
  writes the 64-byte completion record into its own /dev/shm record file, so
  the existing poll/authenticate path consumes cross-host pushes unchanged.

One server runs per Target instance on its host (``--dt-tcp-port``); both
legs share it and are dispatched by the request ``type``.  Wire format
(single source of truth for the whole fallback):

    u32le total frame length N
    N bytes:
        u32le header length M
        M bytes msgpack header dict (always carries a "type" string)
        raw payload bytes

The msgpack codec is ``msgspec.msgpack``, the same one every other DT
module uses.  Long-lived workers (one hidden-pull worker and one
candidate-push sender per Draft-Target pair) keep one persistent framed
connection each: per-request connections churned a fresh TCP TIME_WAIT
slot per request and, at the measured full-concurrency round rate,
exhausted the ephemeral-port space of a (source, destination, port)
five-tuple within a minute.  The wire protocol is unchanged and the
server keeps a connection open for sequential request/response rounds.
"""

from __future__ import annotations

import logging
import socket
import struct
import threading

import msgspec

logger = logging.getLogger(__name__)

# Request types carried in the header "type" field.  The response type is
# always the request base plus "_resp" (a trailing "_req" is stripped first):
# "hidden_pull_req" -> "hidden_pull_resp", "cand_push" -> "cand_push_resp".
MSG_HIDDEN_PULL_REQ = "hidden_pull_req"
MSG_CAND_PUSH = "cand_push"
MSG_HIDDEN_PULL_RESP = "hidden_pull_resp"
MSG_CAND_PUSH_RESP = "cand_push_resp"

_HEADER_LEN = struct.Struct("<I")
_FRAME_LEN = struct.Struct("<I")


class HostTcpError(RuntimeError):
    """The host-staged TCP fallback failed at transport or application level."""


def _read_exact(conn: socket.socket, nbytes: int) -> bytes:
    chunks = []
    remaining = nbytes
    while remaining:
        chunk = conn.recv(remaining)
        if not chunk:
            raise HostTcpError("connection closed mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def encode_frame(msg_type: str, header: dict, payload: bytes = b"") -> bytes:
    body_header = dict(header)
    body_header["type"] = msg_type
    header_bytes = msgspec.msgpack.encode(body_header)
    body = _HEADER_LEN.pack(len(header_bytes)) + header_bytes + payload
    return _FRAME_LEN.pack(len(body)) + body


def decode_frame(frame: bytes) -> tuple[dict, bytes]:
    header_len = _HEADER_LEN.unpack_from(frame, 0)[0]
    header_end = _HEADER_LEN.size + header_len
    if header_end > len(frame):
        raise HostTcpError("frame header length exceeds the frame")
    header = msgspec.msgpack.decode(frame[_HEADER_LEN.size : header_end])
    if not isinstance(header, dict):
        raise HostTcpError("frame header is not a msgpack map")
    return header, frame[header_end:]


def _response_type(req_type: str) -> str:
    base = req_type[: -len("_req")] if req_type.endswith("_req") else req_type
    return f"{base}_resp"


class HostTcpServer:
    """Framed request/response server; one daemon thread per connection."""

    def __init__(self, host: str, port: int) -> None:
        self._handlers: dict[str, object] = {}
        self._handlers_lock = threading.Lock()
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((host, port))
        self._listener.listen()
        self.host, self.port = self._listener.getsockname()[:2]
        self._stopping = threading.Event()
        self._connection_threads: list[threading.Thread] = []
        self._thread = threading.Thread(
            target=self._serve, name="dt-host-tcp", daemon=True
        )
        self._thread.start()

    def register_handler(self, msg_type: str, fn) -> None:
        """Bind ``fn(header: dict, payload: bytes) -> (resp_header, payload)``."""

        with self._handlers_lock:
            self._handlers[msg_type] = fn

    def _serve(self) -> None:
        while not self._stopping.is_set():
            try:
                conn, _addr = self._listener.accept()
            except OSError:
                break
            # Reap finished threads before retaining the new one: fallback
            # requests are one connection each, so the list must track live
            # concurrency, not the whole connection history.
            self._connection_threads = [
                t for t in self._connection_threads if t.is_alive()
            ]
            thread = threading.Thread(
                target=self._handle_connection,
                args=(conn,),
                name="dt-host-tcp-conn",
                daemon=True,
            )
            self._connection_threads.append(thread)
            thread.start()

    def _handle_connection(self, conn: socket.socket) -> None:
        with conn:
            while not self._stopping.is_set():
                try:
                    raw_len = _read_exact(conn, _FRAME_LEN.size)
                except (OSError, HostTcpError):
                    return
                frame_len = _FRAME_LEN.unpack(raw_len)[0]
                if frame_len <= 0:
                    return
                try:
                    frame = _read_exact(conn, frame_len)
                except (OSError, HostTcpError):
                    return
                req_type = ""
                try:
                    header, payload = decode_frame(frame)
                    req_type = str(header.get("type", ""))
                    with self._handlers_lock:
                        handler = self._handlers.get(req_type)
                    if handler is None:
                        raise HostTcpError(f"unknown request type {req_type!r}")
                    resp_header, resp_payload = handler(header, payload)
                    if resp_header is None:
                        resp_header = {}
                    if not isinstance(resp_header, dict):
                        raise HostTcpError("handler returned a non-dict header")
                    resp_header = dict(resp_header)
                    resp_header.setdefault("ok", True)
                    out = encode_frame(_response_type(req_type), resp_header, resp_payload or b"")
                except Exception as exc:  # noqa: BLE001 - report, then keep serving
                    logger.warning("dt host tcp handler failed: %s", exc)
                    out = encode_frame(
                        _response_type(req_type) if req_type else "error_resp",
                        {"ok": False, "error": str(exc)},
                    )
                try:
                    conn.sendall(out)
                except OSError:
                    return

    def stop(self) -> None:
        self._stopping.set()
        try:
            self._listener.close()
        except OSError:
            pass
        self._thread.join(timeout=5.0)
        for thread in self._connection_threads:
            thread.join(timeout=5.0)


class HostTcpConnection:
    """One persistent framed-TCP connection owned by a single thread.

    The server keeps a connection open for sequential request/response
    rounds, so a long-lived worker (one per Draft-Target pair) reuses one
    connection instead of churning a fresh TCP connection — and its TIME_WAIT
    slot — per request.  At the measured full-concurrency round rate the
    per-request connections exhausted the ephemeral-port space of a
    (source, destination, port) five-tuple within a minute; a persistent
    connection makes the rate irrelevant.  Not thread-safe: each owner is one
    dedicated worker thread.
    """

    def __init__(self, host: str, port: int, timeout_s: float) -> None:
        self._addr = (host, port)
        self._timeout_s = timeout_s
        self._conn: socket.socket | None = None

    def _ensure(self) -> socket.socket:
        if self._conn is None:
            self._conn = socket.create_connection(self._addr, timeout=self._timeout_s)
        return self._conn

    def request(
        self,
        msg_type: str,
        header: dict,
        payload: bytes = b"",
        *,
        attempts: int = 2,
    ) -> tuple[dict, bytes]:
        """Send one framed request and read its response.

        Retries once on a transport error with a fresh connection: a retry
        can only race a lost response, and a replayed frame is idempotent on
        the server (same transfer nonce/generation, same payload words).
        Application-level rejections (``ok: false``) are never retried.
        """

        last_error: HostTcpError | None = None
        for attempt in range(max(1, attempts)):
            frame = encode_frame(msg_type, header, payload)
            try:
                conn = self._ensure()
                conn.settimeout(self._timeout_s)
                conn.sendall(frame)
                raw_len = _read_exact(conn, _FRAME_LEN.size)
                frame_len = _FRAME_LEN.unpack(raw_len)[0]
                if frame_len <= 0:
                    raise HostTcpError("empty response frame")
                resp_header, resp_payload = decode_frame(
                    _read_exact(conn, frame_len)
                )
            except HostTcpError as exc:
                last_error = exc
            except OSError as exc:
                last_error = HostTcpError(
                    f"dt host tcp request to {self._addr[0]}:{self._addr[1]} "
                    f"failed: {exc}"
                )
            else:
                if not resp_header.get("ok"):
                    # The server processed the request and rejected it; a
                    # retry would hit the same rejection.
                    raise HostTcpError(
                        f"dt host tcp request {msg_type!r} rejected: "
                        f"{resp_header.get('error', 'unknown error')}"
                    )
                return resp_header, resp_payload
            self.close()
        assert last_error is not None
        raise last_error

    def close(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass

    def __enter__(self) -> "HostTcpConnection":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()
