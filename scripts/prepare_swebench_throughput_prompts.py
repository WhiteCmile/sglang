#!/usr/bin/env python3
"""Build a unique, disjoint-warmup SWE-bench throughput prompt corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def format_prompt(row) -> str:
    body = (
        f"Repository: {row.repo}\n"
        f"Instance: {row.instance_id}\n"
        f"Problem Statement:\n{row.problem_statement}\n\n"
        "Analyze the issue and describe a concrete patch."
    )
    return (
        f"<|im_start|>user\n{body}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def percentile(sorted_values: list[int], fraction: float) -> int:
    index = min(len(sorted_values) - 1, int(fraction * (len(sorted_values) - 1)))
    return sorted_values[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dev-parquet", type=Path, required=True)
    parser.add_argument("--test-parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("/share/dai-sys/models/qwen3-14b")
    )
    parser.add_argument("--warmup-requests", type=int, default=32)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup_requests <= 0:
        raise ValueError("warmup_requests must be positive")
    if args.output.exists() or args.output.with_suffix(".metadata.json").exists():
        raise FileExistsError("output or metadata already exists")

    dev_rows = list(pd.read_parquet(args.dev_parquet).itertuples(index=False))
    test_rows = list(pd.read_parquet(args.test_parquet).itertuples(index=False))
    if args.warmup_requests > len(dev_rows):
        raise ValueError("dev split cannot cover the requested warmup")

    dev_rng = random.Random(args.shuffle_seed)
    test_rng = random.Random(args.shuffle_seed)
    dev_rng.shuffle(dev_rows)
    test_rng.shuffle(test_rows)
    warmup_rows = dev_rows[: args.warmup_requests]

    warmup_ids = {str(row.instance_id) for row in warmup_rows}
    test_ids = {str(row.instance_id) for row in test_rows}
    if len(test_ids) != len(test_rows):
        raise ValueError("test instance ids are not unique")
    if warmup_ids & test_ids:
        raise ValueError("warmup and measurement instance ids overlap")

    warmup_prompts = [format_prompt(row) for row in warmup_rows]
    measurement_prompts = [format_prompt(row) for row in test_rows]
    prompts = warmup_prompts + measurement_prompts
    if len(set(prompts)) != len(prompts):
        raise ValueError("formatted prompts are not unique")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True
    )
    token_lengths = sorted(
        len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts
    )
    metadata = {
        "dataset": "princeton-nlp/SWE-bench",
        "source_files": {
            "dev": {
                "path": str(args.dev_parquet.resolve()),
                "sha256": sha256(args.dev_parquet),
            },
            "test": {
                "path": str(args.test_parquet.resolve()),
                "sha256": sha256(args.test_parquet),
            },
        },
        "tokenizer": str(args.tokenizer),
        "shuffle_seed": args.shuffle_seed,
        "warmup_requests": len(warmup_prompts),
        "measurement_requests": len(measurement_prompts),
        "total_unique_prompts": len(prompts),
        "measurement_unique_prompts": len(set(measurement_prompts)),
        "warmup_measurement_overlap": len(
            set(warmup_prompts) & set(measurement_prompts)
        ),
        "token_lengths": {
            "min": token_lengths[0],
            "p50": percentile(token_lengths, 0.50),
            "p90": percentile(token_lengths, 0.90),
            "p95": percentile(token_lengths, 0.95),
            "p99": percentile(token_lengths, 0.99),
            "max": token_lengths[-1],
            "mean": sum(token_lengths) / len(token_lengths),
        },
        "template": (
            "Qwen chat text with repository, instance id, problem statement, "
            "a patch-analysis instruction, and an empty think block"
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(prompts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata["prompts_sha256"] = sha256(args.output)
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
