from __future__ import annotations

import argparse
import json
from pathlib import Path

from .calibration import CalibrationConfig, calibrate_checkpoint
from .fragments import FragmentProposalConfig
from .inference import (
    ContourInferenceConfig,
    FullContourPredictor,
    load_boundary_role_model,
)
from .training_data import (
    BoundaryTrainingIndex,
    BoundaryWindowConfig,
    BoundaryWindowSampler,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate a trained boundary-role checkpoint on the reserved "
            "calibration split without touching the holdout split."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--boundaries", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--output-checkpoint", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--maximum-observations", type=int, default=512)
    parser.add_argument("--maximum-temperature-points", type=int, default=2_000_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20261006)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("calibration requires PyTorch") from exc
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, checkpoint = load_boundary_role_model(args.checkpoint, device)
    index = BoundaryTrainingIndex(args.boundaries, args.observations)
    if (
        checkpoint.get("boundary_dataset_fingerprint")
        != index.boundary_dataset.get("boundary_dataset_fingerprint")
    ):
        raise SystemExit("checkpoint belongs to a different boundary dataset")
    window = BoundaryWindowConfig(**checkpoint["window_config"])
    sampler = BoundaryWindowSampler(index, window)
    predictor = FullContourPredictor(
        model,
        sampler,
        ContourInferenceConfig(
            sequence_length=window.sequence_length,
            strip_width=window.strip_width,
            stride=args.stride,
            batch_size=args.batch_size,
        ),
        FragmentProposalConfig(),
        device=device,
    )
    report = calibrate_checkpoint(
        config=CalibrationConfig(
            checkpoint_path=args.checkpoint,
            output_checkpoint_path=args.output_checkpoint,
            report_path=args.report,
            maximum_observations=args.maximum_observations,
            maximum_temperature_points=args.maximum_temperature_points,
            seed=args.seed,
        ),
        index=index,
        sampler=sampler,
        predictor=predictor,
        checkpoint=checkpoint,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0
