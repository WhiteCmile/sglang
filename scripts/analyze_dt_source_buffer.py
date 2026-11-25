#!/usr/bin/env python3
"""Summarize the four Target feature-source buffer decision metrics.

Input is one or more dt_profile JSONL files or directories.  Instrumentation
is emitted only when ``target_source_buffer`` is explicitly included in
``DT_PROFILE_EVENTS``.  The output keeps allocation efficiency, source-lease
lifetime, slow-reader pinning, and scheduler-gate delay separate so a change
in slot count cannot be mistaken for a fix to reuse efficiency.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


EVENT = "target_source_buffer"


def _paths(inputs: list[Path]) -> list[Path]:
    output: list[Path] = []
    for item in inputs:
        output.extend(sorted(item.rglob("*.jsonl")) if item.is_dir() else (item,))
    return output


def load_events(inputs: list[Path]) -> list[dict]:
    events: list[dict] = []
    for path in _paths(inputs):
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
                if event.get("event") != EVENT:
                    continue
                event["_path"] = str(path)
                events.append(event)
    return sorted(events, key=lambda item: int(item.get("ts_ns", 0)))


def _percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return int(ordered[round((len(ordered) - 1) * fraction)])


def _stats(values: list[int]) -> dict:
    return {
        "count": len(values),
        "median_ns": int(statistics.median(values)) if values else None,
        "p90_ns": _percentile(values, 0.90),
        "p99_ns": _percentile(values, 0.99),
        "mean_ns": int(statistics.fmean(values)) if values else None,
        "total_ns": sum(values),
    }


def _stream(event: dict) -> tuple[str, str]:
    return str(event.get("_path", "")), str(event.get("component", ""))


def _closed_intervals(
    events: list[dict], *, stage: str, key_field: str | None = None
) -> tuple[list[dict], list[dict]]:
    starts: dict[tuple, dict] = {}
    closed: list[dict] = []
    for event in events:
        if event.get("stage") != stage:
            continue
        identity = (*_stream(event), event.get(key_field) if key_field else None)
        if event.get("phase") == "start":
            starts[identity] = event
        elif event.get("phase") == "end":
            start = starts.pop(identity, None)
            row = dict(event)
            if start is not None:
                row.setdefault("started_ns", start.get("observed_ns", start.get("ts_ns")))
                row.setdefault(
                    "duration_ns",
                    int(row.get("observed_ns", row.get("ts_ns", 0)))
                    - int(row.get("started_ns", 0)),
                )
            closed.append(row)
    return closed, list(starts.values())


def summarize(events: list[dict]) -> tuple[dict, list[dict]]:
    publications = [event for event in events if event.get("stage") == "publication"]
    completed = [
        event for event in events if event.get("stage") == "publication_complete"
    ]
    low_intervals, low_open = _closed_intervals(events, stage="effective_free_low")
    gate_intervals, gate_open = _closed_intervals(
        events, stage="source_gate", key_field="gate"
    )
    blocked_launches = [
        event
        for event in events
        if event.get("stage") == "blocked_launch_opportunity"
    ]

    low_durations = [int(item.get("duration_ns", 0)) for item in low_intervals]
    gate_by_kind: dict[str, list[dict]] = defaultdict(list)
    for item in gate_intervals:
        gate_by_kind[str(item.get("gate", "unknown"))].append(item)

    requested = sum(int(item.get("requested_rows", 0)) for item in publications)
    allocated = sum(int(item.get("allocated_rows", 0)) for item in publications)
    waste = allocated - requested
    counter_fields = (
        "source_allocation_count",
        "source_failure_count",
        "source_requested_rows_total",
        "source_allocated_rows_total",
        "source_internal_fragmentation_rows_total",
        "source_identity_high_water",
        "source_leased_pages_high_water",
        "source_pages_total",
    )
    latest_counters = {
        field: max(
            (int(item[field]) for item in events if item.get(field) is not None),
            default=None,
        )
        for field in counter_fields
    }
    slowest_owner = Counter(str(item.get("slowest_owner", "")) for item in completed)
    slowest_channel = Counter(
        str(item.get("slowest_channel", "")) for item in completed
    )
    launch_kinds = Counter(str(item.get("work_kind", "unknown")) for item in blocked_launches)

    summary = {
        "schema": "dt-target-source-buffer-v1",
        "events": len(events),
        "measurement_coverage": {
            "publications": len(publications),
            "completed_publications": len(completed),
            "open_publications": max(0, len(publications) - len(completed)),
        },
        "m1_effective_free_le_one": {
            "closed_intervals": len(low_intervals),
            "open_intervals": len(low_open),
            "duration": _stats(low_durations),
        },
        "m2_source_lease_lifetime": {
            "reserve_to_publish": _stats(
                [
                    int(item["reserve_to_publish_ns"])
                    for item in publications
                    if item.get("reserve_to_publish_ns") is not None
                ]
            ),
            "reserve_to_first_read": _stats(
                [
                    int(item["reserve_to_first_read_ns"])
                    for item in completed
                    if item.get("reserve_to_first_read_ns") is not None
                ]
            ),
            "reserve_to_release": _stats(
                [
                    int(item["reserve_to_release_ns"])
                    for item in completed
                    if item.get("reserve_to_release_ns") is not None
                ]
            ),
            "publish_to_first_read": _stats(
                [int(item.get("first_reader_delay_ns", 0)) for item in completed]
            ),
            "slowest_reader_pin": _stats(
                [int(item.get("slowest_reader_pin_ns", 0)) for item in completed]
            ),
            "slowest_owner": dict(slowest_owner.most_common()),
            "slowest_channel": dict(slowest_channel.most_common()),
        },
        "m3_allocation_efficiency": {
            "requested_rows": requested,
            "allocated_rows": allocated,
            "internal_fragmentation_rows": waste,
            "requested_over_allocated": (
                requested / allocated if allocated else None
            ),
            "latest_allocator_counters": latest_counters,
        },
        "m4_source_gate_delay": {
            "blocked_launch_opportunities": len(blocked_launches),
            "blocked_launches_by_work_kind": dict(launch_kinds.most_common()),
            "gate_episodes": {
                gate: {
                    "closed_intervals": len(items),
                    "duration": _stats(
                        [int(item.get("duration_ns", 0)) for item in items]
                    ),
                    "blocked_decisions": sum(
                        int(item.get("blocked_decisions", 0)) for item in items
                    ),
                }
                for gate, items in sorted(gate_by_kind.items())
            },
            "open_gate_intervals": len(gate_open),
        },
    }
    return summary, completed


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row if key != "_path"})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    events = load_events(args.inputs)
    summary, completed = summarize(events)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "target-source-publications.csv", completed)
    (args.output_dir / "target-source-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if events else 2


if __name__ == "__main__":
    raise SystemExit(main())
