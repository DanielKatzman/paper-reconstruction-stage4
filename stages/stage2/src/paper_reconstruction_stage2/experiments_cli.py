from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiments import BoundaryExperimentConfig, BoundaryRoleExperimentRunner
from .model_candidates import candidate_names


def _string_tuple(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run resumable, validation-only successive halving for Stage 2 "
            "boundary-role contour models."
        )
    )
    parser.add_argument("--boundaries", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--statistics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--models", type=_string_tuple, default=candidate_names())
    parser.add_argument("--samples-per-epoch", type=int, default=50_000)
    parser.add_argument("--validation-samples", type=int, default=10_000)
    parser.add_argument("--screen-epochs", type=int, default=3)
    parser.add_argument("--promotion-epochs", type=int, default=10)
    parser.add_argument("--final-epochs", type=int, default=20)
    parser.add_argument("--promote-count", type=int, default=2)
    parser.add_argument("--screen-observations", type=int, default=128)
    parser.add_argument("--final-observations", type=int, default=512)
    parser.add_argument("--calibration-observations", type=int, default=512)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--precision", choices=("fp32", "amp"))
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--inference-stride", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    unknown = set(args.models) - set(candidate_names())
    if unknown:
        raise SystemExit(f"unknown models: {sorted(unknown)}")
    runner = BoundaryRoleExperimentRunner(
        BoundaryExperimentConfig(
            boundary_dataset_dir=args.boundaries,
            observation_dataset_dir=args.observations,
            statistics_path=args.statistics,
            output_dir=args.output,
            profile_path=args.profile,
            candidates=args.models,
            seed=args.seed,
            samples_per_epoch=args.samples_per_epoch,
            validation_samples=args.validation_samples,
            screen_epochs=args.screen_epochs,
            promotion_epochs=args.promotion_epochs,
            final_epochs=args.final_epochs,
            promote_count=args.promote_count,
            screen_observations=args.screen_observations,
            final_observations=args.final_observations,
            calibration_observations=args.calibration_observations,
            batch_size=args.batch_size,
            num_workers=args.workers,
            precision=args.precision,
            prefetch_factor=args.prefetch_factor,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            checkpoint_every_steps=args.checkpoint_every,
            inference_stride=args.inference_stride,
            device=args.device,
            resume=args.resume,
        )
    )
    state = runner.run()
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0
