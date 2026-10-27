from __future__ import annotations

import argparse
import json
from pathlib import Path

from .role_pipeline import (
    ROLE_PIPELINE_STAGES,
    BoundaryRolePipeline,
    BoundaryRolePipelineConfig,
)


def _optional_positive(value: str) -> int | None:
    parsed = int(value)
    return None if parsed == 0 else parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete boundary-role model pipeline: statistics, exact-resume "
            "training, calibration, and natural full-contour validation."
        )
    )
    parser.add_argument("--boundaries", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples-per-epoch", type=int, default=50_000)
    parser.add_argument("--validation-samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--precision", choices=("fp32", "amp"), default="fp32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--calibration-observations", type=_optional_positive, default=512)
    parser.add_argument("--evaluation-observations", type=_optional_positive, default=512)
    parser.add_argument("--inference-stride", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", choices=ROLE_PIPELINE_STAGES)
    parser.add_argument("--maximum-steps", type=int, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pipeline = BoundaryRolePipeline(
        BoundaryRolePipelineConfig(
            boundary_dataset_dir=args.boundaries,
            observation_dataset_dir=args.observations,
            output_dir=args.output,
            seed=args.seed,
            epochs=args.epochs,
            samples_per_epoch=args.samples_per_epoch,
            validation_samples=args.validation_samples,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            checkpoint_every_steps=args.checkpoint_every,
            early_stopping_patience=args.early_stopping_patience,
            precision=args.precision,
            device=args.device,
            num_workers=args.workers,
            prefetch_factor=args.prefetch_factor,
            calibration_maximum_observations=args.calibration_observations,
            evaluation_maximum_observations=args.evaluation_observations,
            inference_stride=args.inference_stride,
            resume=args.resume,
            stop_after=args.stop_after,
            maximum_steps=args.maximum_steps,
        )
    )
    state = pipeline.run()
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0
