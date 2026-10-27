from __future__ import annotations

import argparse
from pathlib import Path

from .statistics import StatisticsConfig, TrainingStatisticsCompiler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute resumable Stage 2 boundary-training statistics."
    )
    parser.add_argument("--boundaries", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = TrainingStatisticsCompiler(
        StatisticsConfig(
            boundary_dataset_dir=args.boundaries,
            observation_dataset_dir=args.observations,
            output_path=args.output,
            split=args.split,
            resume=args.resume,
        )
    ).compile()
    print(
        f"Training statistics complete: {report['accumulators']['point_count']} "
        f"points across {len(report['completed_boundary_paths'])} puzzle bundles"
    )
    return 0
