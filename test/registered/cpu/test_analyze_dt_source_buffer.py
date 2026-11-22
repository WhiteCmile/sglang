from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[3] / "scripts" / "analyze_dt_source_buffer.py"
SPEC = importlib.util.spec_from_file_location("analyze_dt_source_buffer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_summary_keeps_capacity_pinning_and_gate_metrics_separate() -> None:
    events = [
        {
            "event": MODULE.EVENT,
            "stage": "publication",
            "ts_ns": 100,
            "component": "target-0",
            "source_slot_id": 1,
            "slot_generation": 2,
            "requested_rows": 48,
            "allocated_rows": 64,
            "reserve_to_publish_ns": 10,
            "source_allocation_count": 1,
            "source_failure_count": 0,
            "source_requested_rows_total": 48,
            "source_allocated_rows_total": 64,
            "source_internal_fragmentation_rows_total": 16,
            "source_identity_high_water": 1,
            "source_leased_pages_high_water": 1,
            "source_pages_total": 512,
        },
        {
            "event": MODULE.EVENT,
            "stage": "effective_free_low",
            "phase": "start",
            "ts_ns": 120,
            "observed_ns": 120,
            "component": "target-0",
        },
        {
            "event": MODULE.EVENT,
            "stage": "source_gate",
            "phase": "start",
            "gate": "prefill",
            "ts_ns": 130,
            "observed_ns": 130,
            "component": "target-0",
        },
        {
            "event": MODULE.EVENT,
            "stage": "blocked_launch_opportunity",
            "ts_ns": 140,
            "work_kind": "verify",
            "component": "target-0",
        },
        {
            "event": MODULE.EVENT,
            "stage": "source_gate",
            "phase": "end",
            "gate": "prefill",
            "ts_ns": 180,
            "observed_ns": 180,
            "duration_ns": 50,
            "blocked_decisions": 3,
            "component": "target-0",
        },
        {
            "event": MODULE.EVENT,
            "stage": "effective_free_low",
            "phase": "end",
            "ts_ns": 200,
            "observed_ns": 200,
            "duration_ns": 80,
            "component": "target-0",
        },
        {
            "event": MODULE.EVENT,
            "stage": "publication_complete",
            "ts_ns": 240,
            "reserve_to_first_read_ns": 90,
            "reserve_to_release_ns": 150,
            "first_reader_delay_ns": 80,
            "slowest_reader_pin_ns": 60,
            "slowest_owner": "draft-b",
            "slowest_channel": "relayed_source_read_done",
            "component": "target-0",
        },
    ]

    summary, completed = MODULE.summarize(events)

    assert len(completed) == 1
    assert summary["m1_effective_free_le_one"]["duration"]["total_ns"] == 80
    assert (
        summary["m2_source_lease_lifetime"]["slowest_reader_pin"]["median_ns"]
        == 60
    )
    assert summary["m3_allocation_efficiency"]["requested_over_allocated"] == 0.75
    assert summary["m4_source_gate_delay"]["blocked_launches_by_work_kind"] == {
        "verify": 1
    }
    assert summary["m4_source_gate_delay"]["gate_episodes"]["prefill"][
        "blocked_decisions"
    ] == 3
