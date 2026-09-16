#!/usr/bin/env python3
"""Summarize model latency emitted by serve_policy.py."""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path


LATENCY_RE = re.compile(
    r"MODEL_LATENCY_S=(?P<seconds>[0-9]+(?:\.[0-9]+)?)\s+MODE=(?P<mode>action|joint)"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Fast-WAM model-only latency")
    parser.add_argument("log", type=Path, help="serve_policy.py log")
    parser.add_argument("--mode", choices=("action", "joint"), required=True)
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="number of matching calls to discard (default: 5)",
    )
    args = parser.parse_args()

    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    samples_ms = []
    for line in args.log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LATENCY_RE.search(line)
        if match and match.group("mode") == args.mode:
            samples_ms.append(float(match.group("seconds")) * 1000.0)

    if len(samples_ms) <= args.warmup:
        parser.error(
            f"found {len(samples_ms)} {args.mode!r} samples, "
            f"which is not enough for warmup={args.warmup}"
        )

    measured = samples_ms[args.warmup :]
    print(
        f"mode={args.mode} warmup={args.warmup} samples={len(measured)} "
        f"mean_ms_per_chunk={statistics.fmean(measured):.1f}"
    )


if __name__ == "__main__":
    main()
