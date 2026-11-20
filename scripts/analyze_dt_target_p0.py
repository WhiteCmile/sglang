#!/usr/bin/env python3
"""Join issue #281 Target P0 hop events and rank the next experiment.

Input is one or more dt_profile JSONL files (or directories containing them).
The instrumentation is emitted only when ``target_p0_hop`` is explicitly in
``DT_PROFILE_EVENTS``.  One CSV row is produced per transfer; the JSON summary
compares the Target send-side corridor (P1+P2) with the post-candidate
observation/freeze corridor (P3).  Draft execution remains visible as a
separate round-trip segment and is never attributed to either Target option.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path


P0_EVENT = "target_p0_hop"
HOP_ORDER = {
    "feature_published": 0,
    "record_visible": 1,
    "destination_written": 2,
    "ring_written": 3,
    "dispatch_registered": 4,
    "candidate_observed": 5,
    "ready_snapshot": 6,
    "tp_ready_applied": 7,
    "frozen_for_verify": 8,
    "verify_submitted": 9,
    "verify_gpu_end": 10,
}


def _paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if item.is_dir():
            paths.extend(sorted(item.rglob("*.jsonl")))
        else:
            paths.append(item)
    return paths


def load_events(inputs: list[Path]) -> tuple[list[dict], Counter]:
    events: list[dict] = []
    fallback = Counter()
    for path in _paths(inputs):
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
                if event.get("event") != P0_EVENT:
                    continue
                hop = event.get("hop")
                if hop == "at_launch_fallback":
                    fallback[str(event.get("reason", "unknown"))] += 1
                    continue
                if hop not in HOP_ORDER or not event.get("transfer_id"):
                    continue
                event["_path"] = str(path)
                events.append(event)
    return events, fallback


def _put(row: dict, event: dict) -> None:
    hop = str(event["hop"])
    previous = row.get(f"_{hop}_ts")
    timestamp = int(event.get("ts_ns", 0))
    if previous is not None and timestamp >= previous:
        return
    row[f"_{hop}_ts"] = timestamp
    row["component"] = event.get("component", row.get("component", ""))
    row["source_path"] = event.get("_path", row.get("source_path", ""))
    row["request_id"] = event.get("request_id", row.get("request_id", ""))
    row["round_id"] = event.get("round_id", row.get("round_id", ""))
    row["wave_id"] = event.get("wave_id", row.get("wave_id", ""))
    for key, value in event.items():
        if key not in {"event", "hop", "component", "ts_ns", "_path"}:
            row[key] = value


def _delta(row: dict, end: str, start: str) -> int | None:
    end_value = int(row.get(end) or 0)
    start_value = int(row.get(start) or 0)
    if not end_value or not start_value or end_value < start_value:
        return None
    return end_value - start_value


def build_rows(events: list[dict]) -> list[dict]:
    rows: dict[str, dict] = {}
    for event in sorted(
        events,
        key=lambda item: (
            str(item["transfer_id"]),
            HOP_ORDER[str(item["hop"])],
            int(item.get("ts_ns", 0)),
        ),
    ):
        transfer_id = str(event["transfer_id"])
        row = rows.setdefault(transfer_id, {"transfer_id": transfer_id})
        _put(row, event)

    output = []
    for row in rows.values():
        row["hidden_to_descriptor_ns"] = _delta(
            row, "descriptor_published_ns", "hidden_produced_ns"
        )
        row["hidden_to_record_visible_ns"] = _delta(
            row, "record_visible_ns", "hidden_produced_ns"
        )
        row["record_to_destination_ns"] = _delta(
            row, "destination_written_ns", "record_visible_ns"
        )
        row["descriptor_to_dispatch_ns"] = _delta(
            row, "ledger_begin_finished_ns", "descriptor_published_ns"
        )
        row["dispatch_to_candidate_ns"] = _delta(
            row, "completion_observed_ns", "ledger_begin_finished_ns"
        )
        row["candidate_to_tp_apply_ns"] = _delta(
            row, "tp_apply_ns", "completion_observed_ns"
        )
        row["candidate_to_ready_snapshot_ns"] = _delta(
            row, "ready_snapshot_ns", "completion_observed_ns"
        )
        row["ready_snapshot_to_tp_end_ns"] = _delta(
            row, "tp_broadcast_end_ns", "ready_snapshot_ns"
        )
        row["candidate_to_freeze_ns"] = _delta(
            row, "freeze_ns", "completion_observed_ns"
        )
        row["freeze_to_submit_ns"] = _delta(
            row, "submission_returned_ns", "freeze_ns"
        )
        row["submit_to_verify_observed_ns"] = _delta(
            row, "gpu_end_observed_ns", "submission_returned_ns"
        )
        row["p1_p2_corridor_ns"] = _delta(
            row, "ledger_begin_finished_ns", "hidden_produced_ns"
        )
        row["p3_corridor_ns"] = row["candidate_to_freeze_ns"]
        for key in tuple(row):
            if key.startswith("_"):
                del row[key]
        output.append(row)

    freeze_by_source: dict[tuple[str, str], set[int]] = {}
    for row in output:
        freeze_ns = int(row.get("freeze_ns") or 0)
        if freeze_ns:
            source_key = (
                str(row.get("source_path", "")),
                str(row.get("component", "")),
            )
            freeze_by_source.setdefault(source_key, set()).add(freeze_ns)
    for row in output:
        observed_ns = int(row.get("completion_observed_ns") or 0)
        own_freeze_ns = int(row.get("freeze_ns") or 0)
        if not observed_ns or not own_freeze_ns:
            row["missed_freeze_windows"] = None
            continue
        source_key = (
            str(row.get("source_path", "")),
            str(row.get("component", "")),
        )
        row["missed_freeze_windows"] = sum(
            observed_ns < freeze_ns < own_freeze_ns
            for freeze_ns in freeze_by_source.get(source_key, set())
        )
    return sorted(output, key=lambda row: row["transfer_id"])


def _percentile(values: list[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return int(ordered[index])


def _stats(rows: list[dict], field: str) -> dict:
    values = [int(row[field]) for row in rows if row.get(field) is not None]
    return {
        "count": len(values),
        "median_ns": int(statistics.median(values)) if values else None,
        "p90_ns": _percentile(values, 0.90),
        "mean_ns": int(statistics.fmean(values)) if values else None,
    }


def summarize(rows: list[dict], fallback: Counter) -> dict:
    fields = (
        "hidden_to_descriptor_ns",
        "hidden_to_record_visible_ns",
        "record_to_destination_ns",
        "descriptor_to_dispatch_ns",
        "dispatch_to_candidate_ns",
        "candidate_to_tp_apply_ns",
        "candidate_to_ready_snapshot_ns",
        "ready_snapshot_to_tp_end_ns",
        "candidate_to_freeze_ns",
        "freeze_to_submit_ns",
        "submit_to_verify_observed_ns",
        "p1_p2_corridor_ns",
        "p3_corridor_ns",
        "control_phase_ns",
        "poll_phase_ns",
    )
    stats = {field: _stats(rows, field) for field in fields}
    p12 = stats["p1_p2_corridor_ns"]
    p3 = stats["p3_corridor_ns"]
    comparable = min(p12["count"], p3["count"])
    if not comparable:
        recommendation = "insufficient_p0_coverage"
        reason = "need transfers containing both send-side and READY-to-freeze stamps"
    elif int(p12["median_ns"]) >= int(p3["median_ns"]):
        recommendation = "P1+P2"
        reason = "median Target send-side corridor is not smaller than P3"
    else:
        recommendation = "P3"
        reason = "median candidate-observed-to-freeze corridor dominates P1+P2"
    candidate_active = [
        int(row["candidate_slots_total"]) - int(row["candidate_slots_free"])
        for row in rows
        if row.get("candidate_slots_total") is not None
        and row.get("candidate_slots_free") is not None
    ]
    source_pages_active = [
        int(row["source_pages_total"]) - int(row["source_pages_free"])
        for row in rows
        if row.get("source_pages_total")
    ]
    missed_freeze_windows = [
        int(row["missed_freeze_windows"])
        for row in rows
        if row.get("missed_freeze_windows") is not None
    ]
    return {
        "schema": "dt-target-p0-v1",
        "transfers": len(rows),
        "complete_comparison_transfers": comparable,
        "segments": stats,
        "counters": {
            "at_launch_fallback_by_reason": dict(sorted(fallback.items())),
            "at_launch_credit_fallbacks": int(fallback.get("credit", 0)),
            "candidate_slots_high_water": max(candidate_active, default=None),
            "source_pages_high_water": max(source_pages_active, default=None),
            "rows_missing_at_least_one_freeze": sum(
                count > 0 for count in missed_freeze_windows
            ),
            "missed_freeze_windows": sum(missed_freeze_windows),
            "minimum_ring_credit_free": min(
                (
                    int(row["ring_credit_free"])
                    for row in rows
                    if row.get("ring_credit_free") is not None
                ),
                default=None,
            ),
        },
        "recommendation": recommendation,
        "reason": reason,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    events, fallback = load_events(args.inputs)
    rows = build_rows(events)
    summary = summarize(rows, fallback)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "target-p0-hops.csv", rows)
    (args.output_dir / "target-p0-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
