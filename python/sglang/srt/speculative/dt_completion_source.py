"""Transport-neutral completion observation for the Target data plane.

The producer publishes one small per-slot record — payload words first, the
sequence word last with a system-scope release store — so an observed
matching sequence proves every payload word is visible.  ShmSeqRecordSource
is the only CompletionSource today: it wraps the same-host host-mapped
record file and reports generic CompletionEvents.  Everything bound to the
shm backend (the file path, the host address, the fixed word layouts) lives
in this module; the arena's slot bookkeeping consumes only events.
"""

from __future__ import annotations

import mmap
import os
from dataclasses import dataclass
from typing import Protocol

import numpy as np

# One 32-byte host READY record per slot: the metadata words plus the
# producer tag.  The generation word is written last.
READY_RECORD_WORDS = 8
READY_RECORD_BYTES = READY_RECORD_WORDS * 4
GENERATION_WORD = 0
NONCE_LOW_WORD = 1
NONCE_HIGH_WORD = 2
VALID_LENGTH_WORD = 3
ROOT_TOKEN_WORD = 4
PRODUCER_TAG_WORD = 5

# One 64-byte host completion record per slot: eight u64 words, the seq word
# published last with a system-scope release store.
COMPLETION_RECORD_WORDS = 8
COMPLETION_RECORD_BYTES = COMPLETION_RECORD_WORDS * 8
COMPLETION_SEQ_WORD = 0
COMPLETION_META_WORD = 1  # generation (low u32), valid length (high u32)
COMPLETION_HASH_WORD = 2  # transfer id hash
COMPLETION_TOKEN_WORD = 3  # root token (low u32), producer tag (high u32)
COMPLETION_PRODUCED_WORD = 4  # %globaltimer publication timestamp
COMPLETION_OFFSET_WORD = 5  # producer's globaltimer->realtime offset (0 = none)


class CandidateTransportError(RuntimeError):
    pass


def ready_records_path(transport_session: str) -> str:
    """The same-host READY record file one arena incarnation publishes."""

    return f"/dev/shm/dt-ready-{transport_session}"


def completion_records_path(transport_session: str) -> str:
    """The same-host completion record file one arena incarnation publishes."""

    return f"/dev/shm/dt-completion-{transport_session}"


class _ReadyRecordFile:
    """A /dev/shm mapping of per-slot records shared by Target and Draft."""

    def __init__(
        self,
        path: str,
        *,
        record_bytes: int = READY_RECORD_BYTES,
        slot_count: int = 0,
        create: bool,
    ) -> None:
        if create and slot_count <= 0:
            raise ValueError("record file needs a positive slot count")
        page = mmap.PAGESIZE
        flags = os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0)
        fd = os.open(path, flags, 0o600)
        try:
            if create:
                nbytes = -(-slot_count * record_bytes // page) * page
                os.ftruncate(fd, nbytes)
            else:
                nbytes = os.fstat(fd).st_size
            if nbytes <= 0 or nbytes % record_bytes:
                raise CandidateTransportError("record file has an invalid size")
            self._mm = mmap.mmap(fd, nbytes)
        finally:
            os.close(fd)
        self.path = path
        self.nbytes = nbytes
        self.slot_capacity = nbytes // record_bytes
        self.words = np.frombuffer(self._mm, dtype=np.uint32).reshape(
            self.slot_capacity, record_bytes // 4
        )
        self.host_address = int(self.words.ctypes.data)

    def close(self, *, unlink: bool) -> None:
        self.words = None
        try:
            self._mm.close()
        except BufferError:
            # A caller still holds a view; the mapping dies with the process.
            pass
        if unlink:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass


@dataclass(frozen=True, slots=True)
class CompletionEvent:
    """One completion as observed by the Target event path, backend-neutral.

    Identity fields the record cannot prove (``transfer_id``,
    ``endpoint_session``, ``region_generation``) stay empty here; the
    consumer binds them from the armed slot after authentication.
    ``slot_id``, ``slot_generation``, and ``seq`` locate the event inside
    the consumer's region; ``status`` carries the backend's terminal state
    and the remaining fields are small record metadata.
    """

    slot_id: int
    status: str
    transfer_id: str = ""
    endpoint_session: str = ""
    region_generation: str = ""
    slot_generation: int = 0
    seq: int = 0
    valid_length: int = 0
    root_token: int = 0
    producer_tag: int = 0
    transfer_digest: int = 0
    producer_globaltimer_ns: int = 0
    globaltimer_offset_ns: int = 0


class CompletionSource(Protocol):
    """Transport-neutral source of completion events (see #22 step 1)."""

    def poll(self) -> tuple[CompletionEvent, ...]:
        """Return the newly observed completions, if any."""
        ...

    def make_gpu_visible(self) -> None:
        """Establish GPU visibility for everything this source observed.

        Observing a completion is not, in general, proof that a running GPU
        can consistently read the payload; the backend that owns the source
        establishes that order here.  The call site exists on the consumer
        even while the only backend needs no fence.
        """
        ...


class ShmSeqRecordSource:
    """The only CompletionSource: one host-mapped seq-last record file.

    Two record shapes exist today, both written payload-first with the
    sequence word released last:

    - READY records (u32): the generation word is the sequence; the consumer
      arms one expected generation per slot.
    - completion records (u64): the reservation nonce is the sequence; the
      consumer arms one expected nonce per slot and the record's metadata
      word must additionally carry the expected generation.

    The source owns the mapping and both word layouts and reports decoded,
    still-unauthenticated events.  Arming, consume-once clearing,
    authentication, and the slot state machine stay with the consumer.
    """

    READY_RECORDS = "ready"
    COMPLETION_RECORDS = "completion"

    def __init__(
        self,
        records: _ReadyRecordFile,
        *,
        slot_count: int,
        kind: str,
        expected: np.ndarray,
        expected_seq: np.ndarray | None = None,
    ) -> None:
        if slot_count <= 0:
            raise ValueError("record source needs a positive slot count")
        if kind not in {self.READY_RECORDS, self.COMPLETION_RECORDS}:
            raise ValueError(f"unknown record source kind {kind!r}")
        if kind == self.COMPLETION_RECORDS and expected_seq is None:
            raise ValueError("completion records need a per-slot seq expectation")
        self._records = records
        self._slot_count = slot_count
        self._kind = kind
        self._expected = expected
        self._expected_seq = expected_seq
        if kind == self.COMPLETION_RECORDS:
            self.words: np.ndarray | None = records.words.view(np.uint64).reshape(
                records.slot_capacity, COMPLETION_RECORD_WORDS
            )
        else:
            self.words = records.words

    @classmethod
    def ready_records(
        cls, *, records: _ReadyRecordFile, slot_count: int, expected: np.ndarray
    ) -> ShmSeqRecordSource:
        return cls(
            records, slot_count=slot_count, kind=cls.READY_RECORDS, expected=expected
        )

    @classmethod
    def completion_records(
        cls,
        *,
        records: _ReadyRecordFile,
        slot_count: int,
        expected: np.ndarray,
        expected_seq: np.ndarray,
    ) -> ShmSeqRecordSource:
        return cls(
            records,
            slot_count=slot_count,
            kind=cls.COMPLETION_RECORDS,
            expected=expected,
            expected_seq=expected_seq,
        )

    def poll(self) -> tuple[CompletionEvent, ...]:
        words = self.words
        if words is None:
            return ()
        if self._kind == self.READY_RECORDS:
            expected = self._expected
            hits = np.flatnonzero(
                (expected != 0)
                & (words[: self._slot_count, GENERATION_WORD] == expected)
            )
            if hits.size == 0:
                return ()
            events = []
            for slot_id in hits.tolist():
                row = words[slot_id]
                events.append(
                    CompletionEvent(
                        slot_id=slot_id,
                        status="ok",
                        slot_generation=int(row[GENERATION_WORD]),
                        seq=int(row[NONCE_LOW_WORD])
                        | (int(row[NONCE_HIGH_WORD]) << 32),
                        valid_length=int(row[VALID_LENGTH_WORD]),
                        root_token=int(row[ROOT_TOKEN_WORD]),
                        producer_tag=int(row[PRODUCER_TAG_WORD]),
                    )
                )
            return tuple(events)
        expected = self._expected
        expected_seq = self._expected_seq
        rows = words[: self._slot_count]
        hits = np.flatnonzero(
            (expected_seq != 0)
            & (rows[:, COMPLETION_SEQ_WORD] == expected_seq)
            & ((rows[:, COMPLETION_META_WORD] & 0xFFFFFFFF) == expected)
        )
        if hits.size == 0:
            return ()
        events = []
        for slot_id in hits.tolist():
            row = words[slot_id]
            meta = int(row[COMPLETION_META_WORD])
            token = int(row[COMPLETION_TOKEN_WORD])
            events.append(
                CompletionEvent(
                    slot_id=slot_id,
                    status="ok",
                    slot_generation=meta & 0xFFFFFFFF,
                    seq=int(row[COMPLETION_SEQ_WORD]),
                    valid_length=meta >> 32,
                    root_token=token & 0xFFFFFFFF,
                    producer_tag=token >> 32,
                    transfer_digest=int(row[COMPLETION_HASH_WORD]),
                    producer_globaltimer_ns=int(row[COMPLETION_PRODUCED_WORD]),
                    globaltimer_offset_ns=int(row[COMPLETION_OFFSET_WORD]),
                )
            )
        return tuple(events)

    def make_gpu_visible(self) -> None:
        # Observing a seq-last record already proves payload visibility for
        # the shm backend: the producer released the record with system scope
        # after its copy.  A future backend (e.g. RDMA) performs its real
        # GPU-visibility fence here instead.
        return None

    def close(self) -> None:
        # Drop the word view before the arena closes the underlying mapping.
        self.words = None
