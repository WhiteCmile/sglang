#!/usr/bin/env python3
"""Analyze verify expert-topk dumps by draft-tree depth and MoE layer."""

import argparse
import concurrent.futures
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


def iter_dump_files(input_dir: Path) -> Iterable[Path]:
    return sorted(input_dir.glob("verify_expert_topk_*.pt"))


def _to_1d_long_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().reshape(-1).to(torch.long)
    return torch.tensor(x, dtype=torch.long).reshape(-1)


def compute_depth_of_tokens(positions: torch.Tensor, draft_token_num: int) -> torch.Tensor:
    """Compute per-token tree depth from flattened positions.

    The flattened token order is expected to be per-request contiguous with stride
    `draft_token_num`.
    """
    n = positions.numel()
    if draft_token_num <= 0:
        raise ValueError(f"Invalid draft_token_num={draft_token_num}")
    if n % draft_token_num != 0:
        raise ValueError(
            f"positions length ({n}) is not divisible by draft_token_num ({draft_token_num})"
        )

    bs = n // draft_token_num
    pos_2d = positions.view(bs, draft_token_num)
    base = pos_2d.min(dim=1, keepdim=True).values
    return (pos_2d - base).reshape(-1).to(torch.long)


def _token_expert_set(topk_ids_for_token: torch.Tensor) -> set:
    experts = topk_ids_for_token[topk_ids_for_token >= 0].tolist()
    return set(int(x) for x in experts)


def _build_parent_index_from_retrieve(
    retrive_next_token: torch.Tensor, retrive_next_sibling: torch.Tensor
) -> torch.Tensor:
    assert retrive_next_token.dim() == 2
    assert retrive_next_sibling.shape == retrive_next_token.shape

    bs, draft_token_num = retrive_next_token.shape
    parent_index = torch.full_like(retrive_next_token, -1)
    for b in range(bs):
        for p in range(draft_token_num):
            child = int(retrive_next_token[b, p].item())
            steps = 0
            while 0 <= child < draft_token_num:
                if parent_index[b, child] == -1:
                    parent_index[b, child] = p
                child = int(retrive_next_sibling[b, child].item())
                steps += 1
                if steps > draft_token_num:
                    break
    return parent_index


def analyze_file(path: Path, min_tokens_per_group: int) -> List[Dict]:
    obj = torch.load(path, map_location="cpu")
    records = obj.get("records", [])
    verify_tree_meta = obj.get("verify_tree_meta", {})

    if not records:
        return []

    positions = _to_1d_long_tensor(verify_tree_meta.get("positions", []))
    draft_token = _to_1d_long_tensor(verify_tree_meta.get("draft_token", []))
    draft_token_num = int(verify_tree_meta.get("draft_token_num", 0))
    rows = []
    for rec in records:
        topk_ids_of_layer = rec.get("topk_ids_of_layer")
        if (
            not isinstance(topk_ids_of_layer, torch.Tensor)
            or topk_ids_of_layer.dim() != 3
        ):
            continue

        num_layers, num_tokens, _ = topk_ids_of_layer.shape
        if positions.numel() != num_tokens:
            continue

        curr_draft_token = draft_token
        if curr_draft_token.numel() != num_tokens:
            curr_draft_token = torch.full((num_tokens,), 0, dtype=torch.long)

        depths = compute_depth_of_tokens(positions, draft_token_num)
        valid_token_mask = curr_draft_token >= 0
        num_tokens_total = positions.numel()
        if draft_token_num <= 0:
            continue
        bs = num_tokens_total // draft_token_num
        parent_index = verify_tree_meta.get("parent_index")
        if parent_index is not None:
            parent_index = _to_1d_long_tensor(parent_index).view(bs, draft_token_num)
        else:
            retrive_next_token = verify_tree_meta.get("retrive_next_token")
            retrive_next_sibling = verify_tree_meta.get("retrive_next_sibling")
            if retrive_next_token is None or retrive_next_sibling is None:
                continue
            parent_index = _build_parent_index_from_retrieve(
                _to_1d_long_tensor(retrive_next_token).view(bs, draft_token_num),
                _to_1d_long_tensor(retrive_next_sibling).view(bs, draft_token_num),
            )
        parent_index_flat = parent_index.reshape(-1)
        forward_pass_id = rec.get("forward_pass_id", obj.get("forward_pass_id"))

        for layer_idx in range(num_layers):
            topk_of_layer = topk_ids_of_layer[layer_idx]  # [num_tokens, topk]
            for request_id in range(bs):
                st = request_id * draft_token_num
                ed = st + draft_token_num
                token_mask_base = torch.zeros(num_tokens_total, dtype=torch.bool)
                token_mask_base[st:ed] = True
                for depth in torch.unique(depths).tolist():
                    depth_mask = (
                        (depths == depth) & valid_token_mask & token_mask_base
                    )  # request + depth
                    depth_indices = torch.nonzero(depth_mask, as_tuple=False).reshape(-1)
                    if depth_indices.numel() == 0:
                        continue

                    for parent_slot in torch.unique(
                        parent_index_flat[depth_indices]
                    ).tolist():
                        parent_mask = depth_mask & (parent_index_flat == parent_slot)
                        token_indices = torch.nonzero(
                            parent_mask, as_tuple=False
                        ).reshape(-1)
                        if token_indices.numel() < min_tokens_per_group:
                            continue

                        expert_sets: List[set] = []
                        for token_idx in token_indices.tolist():
                            s = _token_expert_set(topk_of_layer[token_idx])
                            if s:
                                expert_sets.append(s)

                        if len(expert_sets) < min_tokens_per_group:
                            continue

                        inter = set.intersection(*expert_sets)
                        union = set.union(*expert_sets)
                        experts_per_token = max(len(s) for s in expert_sets)
                        inter_over_token_expert_count = (
                            (len(inter) / experts_per_token)
                            if experts_per_token > 0
                            else 0.0
                        )
                        inter_over_union = (len(inter) / len(union)) if union else 0.0

                        rows.append(
                            {
                                "file": path.name,
                                "bid": obj.get("bid"),
                                "forward_pass_id": forward_pass_id,
                                "request_id": int(request_id),
                                "layer": int(layer_idx),
                                "depth": int(depth),
                                "parent_slot": int(parent_slot),
                                "num_tokens": int(len(expert_sets)),
                                "intersection_size": int(len(inter)),
                                "experts_per_token": int(experts_per_token),
                                "union_size": int(len(union)),
                                "inter_over_token_expert_count": float(
                                    inter_over_token_expert_count
                                ),
                                "inter_over_union": float(inter_over_union),
                            }
                        )
    return rows


def summarize_by_file_layer(rows: Sequence[Dict]) -> List[Dict]:
    by_key: Dict[Tuple[str, int, int, int], List[Dict]] = defaultdict(list)
    for row in rows:
        by_key[(row["file"], row["bid"], row["forward_pass_id"], row["layer"])].append(
            row
        )

    summary_rows = []
    for (file_name, bid, forward_pass_id, layer), group in sorted(by_key.items()):
        sum_intersection = sum(int(x["intersection_size"]) for x in group)
        sum_experts_per_token = sum(int(x["experts_per_token"]) for x in group)
        sum_union = sum(int(x["union_size"]) for x in group)
        sum_num_tokens = sum(int(x["num_tokens"]) for x in group)
        summary_rows.append(
            {
                "file": file_name,
                "bid": bid,
                "forward_pass_id": forward_pass_id,
                "layer": int(layer),
                "num_groups": int(len(group)),
                "sum_num_tokens": int(sum_num_tokens),
                "sum_intersection_size": int(sum_intersection),
                "sum_experts_per_token": int(sum_experts_per_token),
                "sum_union_size": int(sum_union),
                "weighted_inter_over_token_expert_count": (
                    float(sum_intersection / sum_experts_per_token)
                    if sum_experts_per_token > 0
                    else 0.0
                ),
                "weighted_inter_over_union": (
                    float(sum_intersection / sum_union) if sum_union > 0 else 0.0
                ),
            }
        )

    return summary_rows


def summarize_by_file_request(rows: Sequence[Dict]) -> List[Dict]:
    by_key: Dict[Tuple[str, int, int, int], List[Dict]] = defaultdict(list)
    for row in rows:
        by_key[
            (row["file"], row["bid"], row["forward_pass_id"], row["request_id"])
        ].append(row)

    summary_rows = []
    for (file_name, bid, forward_pass_id, request_id), group in sorted(by_key.items()):
        mean_inter = sum(float(x["inter_over_token_expert_count"]) for x in group) / len(
            group
        )
        mean_union = sum(float(x["inter_over_union"]) for x in group) / len(group)
        summary_rows.append(
            {
                "file": file_name,
                "bid": bid,
                "forward_pass_id": forward_pass_id,
                "request_id": int(request_id),
                "num_groups": int(len(group)),
                "avg_inter_over_token_expert_count": float(mean_inter),
                "avg_inter_over_union": float(mean_union),
            }
        )
    return summary_rows


def maybe_write_csv(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_write_single_row_csv(path: Path, row: Dict) -> None:
    fieldnames = list(row.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def summarize_global_request_stats(request_rows: Sequence[Dict]) -> Dict:
    count = len(request_rows)
    if count == 0:
        return {
            "num_requests": 0,
            "mean_avg_inter_over_token_expert_count": 0.0,
            "mean_avg_inter_over_union": 0.0,
            "weighted_avg_inter_over_token_expert_count": 0.0,
            "weighted_avg_inter_over_union": 0.0,
        }

    mean_inter = sum(float(x["avg_inter_over_token_expert_count"]) for x in request_rows) / count
    mean_union = sum(float(x["avg_inter_over_union"]) for x in request_rows) / count
    total_groups = sum(int(x["num_groups"]) for x in request_rows)
    weighted_inter = (
        sum(float(x["avg_inter_over_token_expert_count"]) * int(x["num_groups"]) for x in request_rows)
        / total_groups
        if total_groups > 0
        else 0.0
    )
    weighted_union = (
        sum(float(x["avg_inter_over_union"]) * int(x["num_groups"]) for x in request_rows)
        / total_groups
        if total_groups > 0
        else 0.0
    )
    return {
        "num_requests": count,
        "mean_avg_inter_over_token_expert_count": float(mean_inter),
        "mean_avg_inter_over_union": float(mean_union),
        "weighted_avg_inter_over_token_expert_count": float(weighted_inter),
        "weighted_avg_inter_over_union": float(weighted_union),
    }


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}h{m:02d}m{s:02d}s"
    return f"{m:02d}m{s:02d}s"


def _resolve_num_workers(num_workers: int) -> int:
    if num_workers > 0:
        return num_workers
    cpu = os.cpu_count() or 4
    return max(1, min(16, cpu))


def analyze_files(
    files: Sequence[Path],
    min_tokens_per_group: int,
    num_workers: int,
    show_progress: bool,
) -> List[Dict]:
    all_rows: List[Dict] = []
    total = len(files)
    if total == 0:
        return all_rows

    start = time.perf_counter()
    completed = 0

    def print_progress(force_newline: bool = False) -> None:
        if not show_progress:
            return
        elapsed = time.perf_counter() - start
        speed = completed / elapsed if elapsed > 0 else 0.0
        remain = total - completed
        eta = (remain / speed) if speed > 0 else 0.0
        msg = (
            f"\rProgress: {completed}/{total} "
            f"({(completed / total) * 100:5.1f}%) "
            f"rows={len(all_rows)} "
            f"speed={speed:5.2f} files/s "
            f"eta={_format_duration(eta)}"
        )
        sys.stderr.write(msg)
        if force_newline:
            sys.stderr.write("\n")
        sys.stderr.flush()

    workers = _resolve_num_workers(num_workers)
    if workers == 1:
        for path in files:
            all_rows.extend(analyze_file(path, min_tokens_per_group))
            completed += 1
            print_progress()
        print_progress(force_newline=True)
        return all_rows

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        fut_to_path = {
            executor.submit(analyze_file, p, min_tokens_per_group): p for p in files
        }
        for fut in concurrent.futures.as_completed(fut_to_path):
            all_rows.extend(fut.result())
            completed += 1
            print_progress()
    print_progress(force_newline=True)
    return all_rows


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
        default=0,
        help="Maximum files to analyze. 0 means all.",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Analyze all files in input-dir. Overrides --max-files.",
    )
    parser.add_argument(
        "--min-tokens-per-group",
        type=int,
        default=2,
        help="Minimum token count for a (layer, depth) group.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of worker threads for file analysis. 0 means auto.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable live progress display.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to write csv results.",
    )
    parser.add_argument(
        "--output-global-csv",
        type=Path,
        default=None,
        help=(
            "Optional path to write global summary csv. If omitted and --output-csv "
            "is set, defaults to <output_csv_stem>_global.csv."
        ),
    )
    parser.add_argument(
        "--csv-final-only",
        action="store_true",
        default=True,
        help=(
            "Write only final request-level average similarity rows to csv "
            "(default: on)."
        ),
    )
    parser.add_argument(
        "--csv-include-groups",
        dest="csv_final_only",
        action="store_false",
        help="Write per-group rows (request, depth, parent) to csv instead.",
    )
    args = parser.parse_args()

    files = list(iter_dump_files(args.input_dir))
    if not args.all_files and args.max_files > 0:
        files = files[: args.max_files]

    if not files:
        print(f"No files found in {args.input_dir}")
        return

    all_rows = analyze_files(
        files=files,
        min_tokens_per_group=args.min_tokens_per_group,
        num_workers=args.num_workers,
        show_progress=not args.no_progress,
    )

    print(f"Analyzed files: {len(files)}")
    print(f"Valid (file, layer, request, depth, parent) groups: {len(all_rows)}")
    if not all_rows:
        return

    summary_rows = summarize_by_file_layer(all_rows)
    request_summary_rows = summarize_by_file_request(all_rows)
    global_stats = summarize_global_request_stats(request_summary_rows)
    print(f"Layer summary rows: {len(summary_rows)}")
    print(f"Request summary rows: {len(request_summary_rows)}")
    print(
        "Global request stats: "
        f"num_requests={global_stats['num_requests']} "
        f"mean(inter/token_expert_count)={global_stats['mean_avg_inter_over_token_expert_count']:.6f} "
        f"weighted(inter/token_expert_count)={global_stats['weighted_avg_inter_over_token_expert_count']:.6f}"
    )
    print(
        "Global request stats (union): "
        f"mean(inter/union)={global_stats['mean_avg_inter_over_union']:.6f} "
        f"weighted(inter/union)={global_stats['weighted_avg_inter_over_union']:.6f}"
    )

    if args.output_csv is not None:
        rows_to_write = request_summary_rows if args.csv_final_only else all_rows
        maybe_write_csv(args.output_csv, rows_to_write)
        mode = "request-level averages" if args.csv_final_only else "group rows"
        print(f"Wrote {mode} to {args.output_csv}")
        global_csv_path = (
            args.output_global_csv
            if args.output_global_csv is not None
            else args.output_csv.with_name(f"{args.output_csv.stem}_global.csv")
        )
        maybe_write_single_row_csv(global_csv_path, global_stats)
        print(f"Wrote global summary to {global_csv_path}")


if __name__ == "__main__":
    main()
