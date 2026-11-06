"""CPU contract for the publish worker's completion gate (issue #244).

With DT_PUBLISH_WORKER=1 the scheduler thread records a push's event on the
submitting stream and the native worker re-records it after issuing the
push.  Until that re-record is issued the event still answers for the first
record, and a torch event that was never recorded (or whose record already
ran) reports query() == True -- the "recorded gate" bug #222 hit when it
moved publication to a thread.  These tests pin that the push event stays
incomplete until the worker reports the ticket issued, and that a failed
job raises so the push-completion quarantine takes over.  No CUDA context is
opened: the worker is a stand-in for its status() call.
"""

import pytest
import torch

from sglang.srt.speculative.dt_candidate_p2p import (
    CandidateTransportError,
    _WorkerPushEvent,
)


class FakeWorker:
    def __init__(self) -> None:
        self.state = {}

    def status(self, ticket: int) -> int:
        return self.state.get(ticket, 0)


def make_event(worker, ticket):
    event = _WorkerPushEvent()
    event.watch([])
    event.gate(worker, ticket)
    return event


def test_event_stays_incomplete_until_the_worker_issued_its_record():
    worker = FakeWorker()
    event = make_event(worker, 7)
    # The trap: the underlying torch event alone already reads complete.
    assert torch.cuda.Event.query(event)
    assert not event.query()
    worker.state[7] = 1
    assert event.query()


def test_failed_job_raises_from_query_and_synchronize():
    worker = FakeWorker()
    event = make_event(worker, 3)
    worker.state[3] = -700
    with pytest.raises(CandidateTransportError, match="700"):
        event.query()
    with pytest.raises(CandidateTransportError, match="700"):
        event.synchronize()
