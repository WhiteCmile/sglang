#!/usr/bin/env python3
"""Exit successfully only when every requested local TCP port is bindable."""

from __future__ import annotations

import argparse
import socket


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--ports", nargs="+", type=int, required=True)
    args = parser.parse_args()

    held: list[socket.socket] = []
    try:
        for port in args.ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind((args.host, port))
            except OSError as exc:
                print(f"{args.host}:{port} is not bindable: {exc}")
                sock.close()
                return 1
            held.append(sock)
    finally:
        for sock in held:
            sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
