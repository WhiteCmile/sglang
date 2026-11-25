#!/usr/bin/env python3
"""Create a per-run DT topology with a collision-resistant port block."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-port", type=int, required=True)
    return parser.parse_args()


def rewrite_topology(text: str, base: int) -> str:
    if not 20000 <= base <= 64000 - 306:
        raise ValueError(f"base port is outside the safe range: {base}")

    current_instance: str | None = None
    output: list[str] = []
    for line in text.splitlines(keepends=True):
        if line.startswith("coordinator ="):
            line = re.sub(r"target_port\s*=\s*\d+", f"target_port = {base}", line)
            line = re.sub(r"draft_port\s*=\s*\d+", f"draft_port = {base + 1}", line)
        elif line.strip() == "[[instances]]":
            current_instance = None
        elif match := re.match(r'\s*name\s*=\s*"([^"]+)"', line):
            current_instance = match.group(1)
        elif current_instance and (match := re.match(r"(\s*)(\w+_port)\s*=\s*\d+(.*)", line)):
            indent, key, suffix = match.groups()
            if current_instance.startswith("target-"):
                index = int(current_instance.removeprefix("target-"))
                offsets = {"http_port": 100, "nccl_port": 200, "tcp_port": 300}
                if key in offsets:
                    line = f"{indent}{key} = {base + offsets[key] + index}{suffix}\n"
            elif current_instance.startswith("draft-") and key == "nccl_port":
                index = int(current_instance.removeprefix("draft-"))
                line = f"{indent}{key} = {base + 220 + index}{suffix}\n"
        output.append(line)
    return "".join(output)


def main() -> None:
    args = parse_args()
    rewritten = rewrite_topology(args.template.read_text(), args.base_port)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rewritten)


if __name__ == "__main__":
    main()
