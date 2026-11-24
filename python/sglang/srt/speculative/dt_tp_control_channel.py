"""Same-host TP control-packet channel over /dev/shm (Target poll fast path).

``DTTargetController.poll`` sends one small pickled packet from TP0 to every
other TP rank per scheduler iteration.  ``broadcast_pyobj`` does that with two
gloo broadcasts over TCP (size, then bytes), ~0.2 ms on TP0 plus ~0.1 ms of
receive/unpickle on the peer, all on the candidate -> verify critical path.
TP ranks of one Target always share a host, so this channel hands the bytes
over through a shared mapping instead.

Protocol (single producer TP0, N consumers, strictly lock-step like the
broadcast it replaces): two data slots; packet ``seq`` (1, 2, ...) goes into
slot ``seq % 2``.  The producer writes the payload, then its length, then the
slot sequence word; a consumer spins on the slot sequence word, copies the
payload, then publishes ``ack = seq``.  Before reusing a slot (``seq - 2``) the
producer waits until every consumer acked it, so it never overwrites unread
data and runs at most one packet ahead.  x86 keeps stores and loads in program
order, which is what the length/sequence publication relies on.

A packet larger than the slot is sent with the ``OVERSIZE`` length marker and
both sides fall back to ``broadcast_pyobj`` for that packet, so the decision is
identical on every rank.  Every wait is bounded: a peer that stops making
progress raises ``TimeoutError`` instead of hanging the scheduler.
"""

from __future__ import annotations

import mmap
import os
import pickle
import socket
import struct
import time

_HEADER = struct.Struct("<QQ")  # slot sequence, payload length
_WORD = struct.Struct("<Q")
SLOT_BYTES = 1 << 20  # control packets are a few KB; oversize falls back
OVERSIZE = (1 << 64) - 1
ACK_STRIDE = 64  # one cache line per consumer ack word
# Same bound as the default torch.distributed (gloo) collective timeout the
# broadcast it replaces had; an idle scheduler legitimately waits long here.
WAIT_TIMEOUT_S = float(os.environ.get("DT_TARGET_SHM_CONTROL_TIMEOUT_S", "1800"))
SPIN_BEFORE_YIELD = 2000  # tight spins before the first sleep releases the GIL
YIELD_UNTIL_S = 0.002  # then sleep(0) (GIL release only) up to 2 ms of waiting
NAP_UNTIL_S = 0.1  # then 100 us naps up to 100 ms, then 1 ms naps (idle)


class TPControlChannel:
    def __init__(self, path: str, tp_rank: int, tp_size: int, create: bool) -> None:
        self.path = path
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self._ack_base = 0
        self._slot_base = ACK_STRIDE * tp_size
        size = self._slot_base + 2 * (_HEADER.size + SLOT_BYTES)
        flags = os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0)
        fd = os.open(path, flags, 0o600)
        try:
            if create:
                os.ftruncate(fd, size)
            self._map = mmap.mmap(fd, size)
        finally:
            os.close(fd)
        self._view = memoryview(self._map)
        self._seq = 0

    def _slot_offset(self, seq: int) -> int:
        return self._slot_base + (seq % 2) * (_HEADER.size + SLOT_BYTES)

    def _ack_offset(self, rank: int) -> int:
        return self._ack_base + ACK_STRIDE * rank

    def _wait(self, ready, what: str) -> None:
        spins = 0
        started = None
        while not ready():
            spins += 1
            if spins < SPIN_BEFORE_YIELD:
                continue
            now = time.monotonic()
            if started is None:
                started = now
            waited = now - started
            if waited > WAIT_TIMEOUT_S:
                raise TimeoutError(
                    f"DT TP control channel: timed out waiting for {what} "
                    f"(seq {self._seq}, rank {self.tp_rank}, {self.path})"
                )
            time.sleep(0 if waited < YIELD_UNTIL_S else 1e-4 if waited < NAP_UNTIL_S else 1e-3)

    def send(self, packet) -> bool:
        """TP0: publish ``packet``; False means 'oversize, use the fallback'."""
        self._seq += 1
        seq = self._seq
        if seq > 2:
            target = seq - 2
            for rank in range(1, self.tp_size):
                offset = self._ack_offset(rank)
                self._wait(
                    lambda o=offset: _WORD.unpack_from(self._map, o)[0] >= target,
                    f"rank {rank} ack {target}",
                )
        data = pickle.dumps(packet, protocol=pickle.HIGHEST_PROTOCOL)
        offset = self._slot_offset(seq)
        fits = len(data) <= SLOT_BYTES
        if fits:
            self._view[offset + _HEADER.size : offset + _HEADER.size + len(data)] = data
        _WORD.pack_into(self._map, offset + 8, len(data) if fits else OVERSIZE)
        _WORD.pack_into(self._map, offset, seq)
        return fits

    def recv(self):
        """Peer rank: return ``(True, packet)`` or ``(False, None)`` for fallback."""
        self._seq += 1
        seq = self._seq
        offset = self._slot_offset(seq)
        self._wait(
            lambda: _WORD.unpack_from(self._map, offset)[0] == seq, f"packet {seq}"
        )
        length = _WORD.unpack_from(self._map, offset + 8)[0]
        if length == OVERSIZE:
            packet, ok = None, False
        else:
            start = offset + _HEADER.size
            packet, ok = pickle.loads(self._view[start : start + length]), True
        _WORD.pack_into(self._map, self._ack_offset(self.tp_rank), seq)
        return ok, packet

    def close(self) -> None:
        try:
            self._view.release()
            self._map.close()
        except (BufferError, ValueError):
            pass


def open_tp_control_channel(*, instance_id: str, tp_rank: int, tp_size: int,
                            broadcast, all_reduce_min, data_dir: str = "/dev/shm"):
    """Collectively open the channel on every TP rank, or return None on all.

    ``broadcast(obj)`` broadcasts a picklable object from TP0; ``all_reduce_min(int)``
    returns the minimum over the TP group.  Both run on the existing gloo group,
    once, at controller start.
    """
    channel = None
    info = None
    if tp_rank == 0:
        path = os.path.join(
            data_dir, f"dt-tp-control-{instance_id}-{os.getpid()}-{time.monotonic_ns()}"
        )
        try:
            channel = TPControlChannel(path, tp_rank, tp_size, create=True)
            info = (socket.gethostname(), path)
        except OSError:
            channel = None
    info = broadcast(info)
    ok = 0
    if info is not None:
        host, path = info
        if tp_rank == 0:
            ok = int(channel is not None)
        elif host == socket.gethostname():
            try:
                channel = TPControlChannel(path, tp_rank, tp_size, create=False)
                ok = 1
            except OSError:
                channel = None
    agreed = all_reduce_min(ok)
    if tp_rank == 0 and info is not None:
        # Every rank has mapped (or failed); the name is no longer needed.
        try:
            os.unlink(info[1])
        except OSError:
            pass
    if not agreed:
        if channel is not None:
            channel.close()
        return None
    return channel
