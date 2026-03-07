#!/usr/bin/env python3
"""Read speculative verify expert-topk dumps and print a compact summary."""

import argparse
from pathlib import Path
from typing import Iterable

import torch


def iter_dump_files(input_dir: Path) -> Iterable[Path]:
    return sorted(input_dir.glob("verify_expert_topk_*.pt"))


def summarize_file(path: Path) -> str:
    obj = torch.load(path, map_location="cpu")
    records = obj.get("records", [])
    verify_tree_meta = obj.get("verify_tree_meta", {})

    draft_token = verify_tree_meta.get("draft_token")
    positions = verify_tree_meta.get("positions")

    topk_shapes = []
    for rec in records:
        topk = rec.get("topk_ids_of_layer")
        if isinstance(topk, torch.Tensor):
            topk_shapes.append(tuple(topk.shape))

    return (
        f"{path.name} "
        f"bid={obj.get('bid')} "
        f"forward_pass_id={obj.get('forward_pass_id')} "
        f"records={len(records)} "
        f"tree_tokens={0 if draft_token is None else draft_token.numel()} "
        f"positions={0 if positions is None else positions.numel()} "
        f"topk_shapes={topk_shapes[:2]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing verify_expert_topk_*.pt files.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=20,
        help="Maximum number of files to print.",
    )
    args = parser.parse_args()

    files = list(iter_dump_files(args.input_dir))
    if not files:
        print(f"No files found in {args.input_dir}")
        return

    print(f"Found {len(files)} files in {args.input_dir}")
    for path in files[: args.max_files]:
        print(summarize_file(path))


if __name__ == "__main__":
    main()
