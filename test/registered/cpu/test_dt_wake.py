"""CPU contract of the Draft scheduler wake word (csrc/dt_wake.cpp, issue #274).

The scheduler reads the epoch, drains its queues, then waits on that epoch
(design §4.4).  A reason set after the read must end the wait at once, or a
message that lands while the scheduler drains sits out the whole timeout;
D6/D8 add native setters to the same word.  The wait must also release the
GIL, or no Python setter (the control thread) could run while it sleeps.
"""

import threading
import time

from sglang.srt.speculative.dt_wake import WAKE_CONTROL, DraftWake


def test_a_set_after_the_epoch_read_is_not_lost():
    wake = DraftWake()
    try:
        epoch = wake.epoch()
        wake.set(WAKE_CONTROL)
        started = time.monotonic()
        assert wake.wait(epoch, 5.0) == WAKE_CONTROL
        assert time.monotonic() - started < 1.0
        # The reasons were taken; a fresh epoch times out with none.
        assert wake.wait(wake.epoch(), 0.001) == 0
    finally:
        wake.close()


def test_the_wait_releases_the_gil_for_a_python_setter():
    wake = DraftWake()
    try:
        epoch = wake.epoch()
        setter = threading.Timer(0.05, wake.set, args=(WAKE_CONTROL,))
        setter.start()
        started = time.monotonic()
        assert wake.wait(epoch, 5.0) == WAKE_CONTROL
        assert time.monotonic() - started < 1.0
        setter.join()
    finally:
        wake.close()
