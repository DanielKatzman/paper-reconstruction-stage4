from __future__ import annotations

import argparse
import json
from pathlib import Path

from .training_data import (
    BoundaryTrainingIndex,
    BoundaryWindowConfig,
    inspect_samples,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the streaming Stage 2 boundary-training loader."
    )
    parser.add_argument("--boundaries", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--sequence-length", type=int, default=129)
    parser.add_argument("--strip-width", type=int, default=33)
    parser.add_argument("--outside-extent", type=float, default=8.0)
    parser.add_argument("--inside-extent", type=float, default=24.0)
    parser.add_argument("--minimum-label-confidence", type=float, default=0.25)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.samples < 1:
        raise SystemExit("--samples must be positive")
    index = BoundaryTrainingIndex(args.boundaries, args.observations)
    config = BoundaryWindowConfig(
        sequence_length=args.sequence_length,
        strip_width=args.strip_width,
        outside_extent_px=args.outside_extent,
        inside_extent_px=args.inside_extent,
        minimum_label_confidence=args.minimum_label_confidence,
    )
    report = inspect_samples(
        index,
        config,
        split=args.split,
        sample_count=args.samples,
        seed=args.seed,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["valid"] else 1
