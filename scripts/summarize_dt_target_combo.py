#!/usr/bin/env python3
"""Summarize the focused Target-side combination experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ARMS = ("control", "early_slots8", "early_page64", "early_page64_slots8")


def load_arm(root: Path, arm: str) -> dict[str, object]:
    path = root / arm / "benchmark.json"
    data = json.loads(path.read_text())
    return {
        "arm": arm,
        "success_count": data.get("success_count"),
        "failed_requests": data.get("failed_requests"),
        "fixed_work_valid": data.get("fixed_work_valid"),
        "output_tokens_per_s": data.get("output_tokens_per_s"),
        "requests_per_s": data.get("completed_requests_per_s"),
        "mean_latency_s": data.get("mean_latency_s"),
        "p95_latency_s": data.get("p95_latency_s"),
        "wall_time_s": data.get("wall_time_s"),
        "spec_accept_length": data.get("spec_accept_length"),
        "workload_fingerprint": data.get("workload_fingerprint"),
    }


def fmt(value: object, digits: int = 3) -> str:
    return "" if value is None else f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [load_arm(args.root, arm) for arm in ARMS]
    baseline = float(rows[0]["output_tokens_per_s"])
    for row in rows:
        throughput = float(row["output_tokens_per_s"])
        row["vs_control_pct"] = 100.0 * (throughput / baseline - 1.0)

    fingerprints = {row["workload_fingerprint"] for row in rows}
    if len(fingerprints) != 1:
        raise SystemExit(f"workload fingerprint mismatch: {sorted(fingerprints)}")
    if any(row["fixed_work_valid"] is not True for row in rows):
        raise SystemExit("at least one arm did not complete the fixed workload")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("| arm | output tok/s | vs control | req/s | mean latency | p95 latency | accept |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['arm']} | {fmt(row['output_tokens_per_s'])} | "
            f"{float(row['vs_control_pct']):+.2f}% | {fmt(row['requests_per_s'])} | "
            f"{fmt(row['mean_latency_s'])} s | {fmt(row['p95_latency_s'])} s | "
            f"{fmt(row['spec_accept_length'], 4)} |"
        )


if __name__ == "__main__":
    main()
