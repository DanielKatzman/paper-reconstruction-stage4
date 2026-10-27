from __future__ import annotations

import argparse
from pathlib import Path

from .model_candidates import candidate_names
from .profiling import profile_training_pipeline


def _integer_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _string_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile end-to-end Stage 2 boundary-role training throughput."
    )
    parser.add_argument("--boundaries", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--statistics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--batch-sizes", type=_integer_list, default=[8, 16, 32, 64, 96, 128]
    )
    parser.add_argument("--workers", type=_integer_list, default=[0, 2, 4])
    parser.add_argument("--precisions", type=_string_list, default=["fp32", "amp"])
    parser.add_argument("--models", type=_string_list, default=list(candidate_names()))
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--memory-headroom", type=float, default=0.82)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    unknown = set(args.precisions) - {"fp32", "amp"}
    if unknown:
        raise SystemExit(f"unknown precisions: {sorted(unknown)}")
    unknown_models = set(args.models) - set(candidate_names())
    if unknown_models:
        raise SystemExit(f"unknown models: {sorted(unknown_models)}")
    profile_training_pipeline(
        boundary_dataset_dir=args.boundaries,
        observation_dataset_dir=args.observations,
        statistics_path=args.statistics,
        output_path=args.output,
        batch_sizes=args.batch_sizes,
        worker_counts=args.workers,
        precisions=args.precisions,
        model_names=args.models,
        batches=args.batches,
        warmup_batches=args.warmup_batches,
        device=args.device,
        memory_headroom=args.memory_headroom,
    )
    return 0
