from __future__ import annotations

from collections import deque
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass

import torch


def device_timer_ctx(timer: DeviceTimer | None, category: str):
    """Timing context for one forward segment; no-op when the timer is absent.

    A segment that skips this stays out of the fwd_occupancy numerator while
    still counting in its wall-clock denominator, i.e. reads as GPU idle.
    """
    if timer is None:
        return nullcontext()
    return timer.wrap(metadata={"category": category})


class DeviceTimer:
    def __init__(self, reporter: Callable):
        self._intervals: deque[_TimingInterval] = deque()
        self._reporters: list[Callable] = [reporter]
        self._gap_reporters: list[Callable] = []
        self._previous_interval: _TimingInterval | None = None
        self._in_wrap = False

    def add_reporter(self, reporter: Callable):
        self._reporters.append(reporter)

    def add_gap_reporter(self, reporter: Callable):
        """Report the device-stream gap before each completed interval.

        Unlike subtracting asynchronously reported wall-clock timestamps, the
        elapsed time between the previous end event and the current start event
        is measured entirely on the CUDA stream.  Nothing is recorded unless a
        consumer explicitly registers a gap reporter.
        """
        self._gap_reporters.append(reporter)

    @contextmanager
    def wrap(self, metadata: dict):
        # Not re-entrant: a nested wrap would end the wrong interval and leave
        # an un-ended one at the head of the queue for _report() to trip over.
        assert not self._in_wrap, "DeviceTimer.wrap is not re-entrant"
        interval = _TimingInterval.create()
        self._intervals.append(interval)
        self._in_wrap = True
        try:
            yield
        finally:
            self._in_wrap = False
            interval.end(metadata=metadata)
            self._report()

    def _report(self):
        while len(self._intervals) > 0:
            interval = self._intervals[0]
            if not interval.end_event.query():
                break

            self._intervals.popleft()
            previous = self._previous_interval
            if previous is not None and self._gap_reporters:
                gap = max(
                    0.0,
                    previous.end_event.elapsed_time(interval.start_event) / 1000.0,
                )
                gap_metadata = {
                    "previous_category": previous.metadata.get("category", ""),
                    "category": interval.metadata.get("category", ""),
                }
                for reporter in self._gap_reporters:
                    reporter(t=gap, **gap_metadata)
            elapsed = interval.elapsed_time() / 1000.0
            for reporter in self._reporters:
                reporter(t=elapsed, **interval.metadata)
            self._previous_interval = interval


class GapTimer(DeviceTimer):
    """Measures GPU idle gaps between consecutive uses of a stream.

    Where DeviceTimer.wrap() measures the duration *inside* a block,
    GapTimer.wrap() measures the time *between* consecutive blocks
    (gap = next_block_start - last_block_end).
    """

    def __init__(self, reporter: Callable):
        super().__init__(reporter)
        self._pending: _TimingInterval | None = None

    @contextmanager
    def wrap(self, metadata: dict):
        if self._pending is not None:
            self._pending.end(metadata=metadata)
            self._intervals.append(self._pending)
            self._pending = None
            self._report()
        try:
            yield
        finally:
            self._pending = _TimingInterval.create()

    def cancel(self):
        """Discard a pending gap (e.g. server went idle)."""
        self._pending = None


@dataclass
class _TimingInterval:
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event | None = None
    metadata: dict | None = None

    @staticmethod
    def create():
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        return _TimingInterval(start_event=start_event)

    def end(self, metadata: dict):
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()

        assert self.end_event is None
        self.end_event = end_event
        self.metadata = metadata

    def elapsed_time(self) -> float:
        return self.start_event.elapsed_time(self.end_event)
