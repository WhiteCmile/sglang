"""Profile-only driver-call duration histograms (issue #303, Q8).

TP0 has several executors issuing CUDA driver calls on the same context
(the scheduler thread, the native record observer's GPUDirect flush, the
RDMA threads, PyTorch itself).  Contention shows up as long tails on the
scheduler thread's own calls, so the traced runs time the scheduler-thread
call sites and accumulate them here; the controller dumps and resets the
table every few seconds as ``target_driver_call_hist`` trace events.

Call sites follow the usual zero-cost guard: they read ``time.monotonic_ns``
and call :meth:`DriverCallStats.record` only when profiling is active, so
with ``DT_PROFILE_DIR`` unset nothing here is touched.
"""

from __future__ import annotations


class DriverCallStats:
    BUCKET_EDGES_NS = (
        10_000,
        50_000,
        100_000,
        500_000,
        1_000_000,
        5_000_000,
        10_000_000,
        50_000_000,
    )

    def __init__(self) -> None:
        self._rows: dict[str, list] = {}

    def record(self, name: str, duration_ns: int) -> None:
        row = self._rows.get(name)
        if row is None:
            row = self._rows[name] = [
                0,
                0,
                0,
                [0] * (len(self.BUCKET_EDGES_NS) + 1),
            ]
        row[0] += 1
        row[1] += duration_ns
        if duration_ns > row[2]:
            row[2] = duration_ns
        buckets = row[3]
        for index, edge in enumerate(self.BUCKET_EDGES_NS):
            if duration_ns < edge:
                buckets[index] += 1
                break
        else:
            buckets[len(self.BUCKET_EDGES_NS)] += 1

    def snapshot_and_reset(self) -> dict[str, list]:
        rows = self._rows
        self._rows = {}
        return rows


DRIVER_CALL_STATS = DriverCallStats()
