from __future__ import annotations

import argparse
import json
from pathlib import Path

from .losses import BoundaryRoleLossConfig
from .model import BoundaryRoleModelConfig
from .model_candidates import candidate_model_config, candidate_names
from .training import BoundaryRoleTrainer, BoundaryTrainingConfig
from .training_data import BoundaryWindowConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train BoundaryRoleNetV1 with exact safe-resume checkpoints."
    )
    parser.add_argument("--boundaries", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--statistics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--samples-per-epoch", type=int, default=50_000)
    parser.add_argument("--validation-samples", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--precision", choices=("fp32", "amp"), default="fp32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--maximum-steps", type=int)
    parser.add_argument("--maximum-completed-epochs", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--sequence-length", type=int, default=129)
    parser.add_argument("--strip-width", type=int, default=33)
    parser.add_argument("--outside-extent", type=float, default=8.0)
    parser.add_argument("--inside-extent", type=float, default=24.0)
    parser.add_argument("--minimum-label-confidence", type=float, default=0.25)
    parser.add_argument("--branch-channels", type=int, default=64)
    parser.add_argument("--fusion-channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--model-candidate", choices=candidate_names())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_config = (
        candidate_model_config(args.model_candidate)
        if args.model_candidate
        else BoundaryRoleModelConfig(
            branch_channels=args.branch_channels,
            fusion_channels=args.fusion_channels,
            dropout=args.dropout,
            use_raster=not args.geometry_only,
        )
    )
    trainer = BoundaryRoleTrainer(
        BoundaryTrainingConfig(
            boundary_dataset_dir=args.boundaries,
            observation_dataset_dir=args.observations,
            statistics_path=args.statistics,
            output_dir=args.output,
            seed=args.seed,
            epochs=args.epochs,
            samples_per_epoch=args.samples_per_epoch,
            validation_samples=args.validation_samples,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_fraction=args.warmup_fraction,
            gradient_clip_norm=args.gradient_clip,
            checkpoint_every_steps=args.checkpoint_every,
            early_stopping_patience=args.early_stopping_patience,
            precision=args.precision,
            device=args.device,
            num_workers=args.workers,
            prefetch_factor=args.prefetch_factor,
            resume=args.resume,
            maximum_steps=args.maximum_steps,
            maximum_completed_epochs=args.maximum_completed_epochs,
        ),
        window_config=BoundaryWindowConfig(
            sequence_length=args.sequence_length,
            strip_width=args.strip_width,
            outside_extent_px=args.outside_extent,
            inside_extent_px=args.inside_extent,
            minimum_label_confidence=args.minimum_label_confidence,
        ),
        model_config=model_config,
        loss_config=BoundaryRoleLossConfig(),
    )
    state = trainer.train()
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0
