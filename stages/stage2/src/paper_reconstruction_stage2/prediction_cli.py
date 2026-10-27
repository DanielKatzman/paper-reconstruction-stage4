from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fragments import FragmentProposalConfig
from .inference import (
    ContourInferenceConfig,
    FullContourPredictor,
    load_boundary_role_model,
)
from .prediction import BoundaryPredictionExporter, PredictionExportConfig
from .training_data import (
    BoundaryTrainingIndex,
    BoundaryWindowConfig,
    BoundaryWindowSampler,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export calibrated full-contour boundary roles and ranked candidate "
            "edge fragments with safe per-observation resume."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--boundaries", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--split", choices=("all", "train", "validation", "calibration", "holdout"), default="all"
    )
    parser.add_argument("--maximum-observations", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-overlays", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("prediction requires PyTorch") from exc
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
    calibration = checkpoint.get("calibration", {})
    fragment_config = FragmentProposalConfig(
        **checkpoint.get("fragment_config", {})
    )
    predictor = FullContourPredictor(
        model,
        sampler,
        ContourInferenceConfig(
            sequence_length=window.sequence_length,
            strip_width=window.strip_width,
            stride=args.stride,
            batch_size=args.batch_size,
            role_temperature=float(calibration.get("role_temperature", 1.0)),
            transition_temperature=float(
                calibration.get("transition_temperature", 1.0)
            ),
        ),
        fragment_config,
        device=device,
    )
    exporter = BoundaryPredictionExporter(
        config=PredictionExportConfig(
            checkpoint_path=args.checkpoint,
            output_dir=args.output,
            split=None if args.split == "all" else args.split,
            maximum_observations=args.maximum_observations,
            write_overlays=not args.no_overlays,
            resume=args.resume,
        ),
        index=index,
        sampler=sampler,
        predictor=predictor,
    )
    manifest = exporter.export()
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0
