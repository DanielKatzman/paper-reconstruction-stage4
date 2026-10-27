from __future__ import annotations

import argparse
import json
from pathlib import Path

from .boundary import BoundaryConfig
from .config import GeneratorConfig
from .observation import ObservationConfig
from .pipeline import PipelinePaths, Stage2Pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete Stage 2 data pipeline with atomic per-puzzle "
            "checkpoints and safe resume."
        )
    )
    parser.add_argument("--output", required=True, type=Path, help="Pipeline root")
    parser.add_argument("--sources", type=Path, help="Directory containing images/PDFs")
    parser.add_argument("--canonical-dir", type=Path, help="Optional canonical root")
    parser.add_argument("--observations-dir", type=Path, help="Optional observation root")
    parser.add_argument("--boundaries-dir", type=Path, help="Optional boundary root")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--min-pieces", type=int, default=2)
    parser.add_argument("--max-pieces", type=int, default=30)
    parser.add_argument("--width", type=int, default=1240)
    parser.add_argument("--height", type=int, default=1754)
    parser.add_argument("--min-piece-area-fraction", type=float, default=0.0025)
    parser.add_argument("--min-piece-dimension", type=int, default=10)
    parser.add_argument("--max-split-attempts", type=int, default=300)
    parser.add_argument("--canonical-preview-max-side", type=int, default=1400)

    parser.add_argument("--observation-seed", type=int, default=20261005)
    parser.add_argument("--views-per-piece", type=int, default=2)
    parser.add_argument("--scale-min", type=float, default=0.82)
    parser.add_argument("--scale-max", type=float, default=1.18)
    parser.add_argument("--perspective-jitter", type=float, default=0.025)
    parser.add_argument("--canvas-margin", type=int, default=16)
    parser.add_argument("--max-mask-morphology", type=int, default=2)
    parser.add_argument("--local-damage-probability", type=float, default=0.35)
    parser.add_argument("--max-blur-radius", type=float, default=1.2)
    parser.add_argument("--jpeg-quality-min", type=int, default=72)
    parser.add_argument("--jpeg-quality-max", type=int, default=100)
    parser.add_argument("--observation-preview-max-side", type=int, default=1400)

    parser.add_argument("--boundary-spacing", type=float, default=2.0)
    parser.add_argument("--minimum-boundary-points", type=int, default=64)
    parser.add_argument("--maximum-boundary-points", type=int, default=4096)
    parser.add_argument("--tangent-smoothing-sigma", type=float, default=1.5)
    parser.add_argument("--truth-sigma", type=float, default=3.0)
    parser.add_argument("--role-temperature", type=float, default=1.5)
    parser.add_argument("--junction-sigma", type=float, default=2.0)
    parser.add_argument("--max-snap-distance", type=float, default=12.0)
    parser.add_argument("--correspondence-spacing", type=float, default=3.0)
    parser.add_argument("--minimum-matchable-seam-length", type=float, default=24.0)
    parser.add_argument("--minimum-track-valid-fraction", type=float, default=0.60)
    parser.add_argument("--minimum-track-valid-samples", type=int, default=8)
    parser.add_argument("--max-inward-thickness", type=int, default=96)
    parser.add_argument("--minimum-aligned-fraction", type=float, default=0.80)
    parser.add_argument("--boundary-preview-max-side", type=int, default=1600)

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse all completed puzzle checkpoints",
    )
    parser.add_argument(
        "--stop-after",
        choices=("canonical", "observations", "boundaries"),
        default="boundaries",
    )
    parser.add_argument("--status", action="store_true", help="Show checkpoint status")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify a completed pipeline without changing it",
    )
    return parser


def _pipeline_from_args(args: argparse.Namespace) -> Stage2Pipeline:
    paths = PipelinePaths.from_root(
        args.output,
        canonical=args.canonical_dir,
        observations=args.observations_dir,
        boundaries=args.boundaries_dir,
    )
    canonical = GeneratorConfig(
        output_dir=paths.canonical,
        source_dir=args.sources,
        puzzle_count=args.count,
        seed=args.seed,
        min_pieces=args.min_pieces,
        max_pieces=args.max_pieces,
        page_width=args.width,
        page_height=args.height,
        min_piece_area_fraction=args.min_piece_area_fraction,
        min_piece_dimension_px=args.min_piece_dimension,
        max_split_attempts=args.max_split_attempts,
        preview_max_side=args.canonical_preview_max_side,
        resume=args.resume,
    )
    observations = ObservationConfig(
        canonical_dataset_dir=paths.canonical,
        output_dir=paths.observations,
        seed=args.observation_seed,
        views_per_piece=args.views_per_piece,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        perspective_jitter=args.perspective_jitter,
        canvas_margin_px=args.canvas_margin,
        max_mask_morphology_px=args.max_mask_morphology,
        local_boundary_damage_probability=args.local_damage_probability,
        max_blur_radius=args.max_blur_radius,
        jpeg_quality_min=args.jpeg_quality_min,
        jpeg_quality_max=args.jpeg_quality_max,
        preview_max_side=args.observation_preview_max_side,
        resume=args.resume,
    )
    boundaries = BoundaryConfig(
        canonical_dataset_dir=paths.canonical,
        observation_dataset_dir=paths.observations,
        output_dir=paths.boundaries,
        boundary_spacing_px=args.boundary_spacing,
        minimum_boundary_points=args.minimum_boundary_points,
        maximum_boundary_points=args.maximum_boundary_points,
        tangent_smoothing_sigma_samples=args.tangent_smoothing_sigma,
        truth_sigma_px=args.truth_sigma,
        role_temperature_px=args.role_temperature,
        junction_sigma_px=args.junction_sigma,
        max_snap_distance_px=args.max_snap_distance,
        correspondence_spacing_px=args.correspondence_spacing,
        minimum_matchable_seam_length_px=args.minimum_matchable_seam_length,
        minimum_track_valid_fraction=args.minimum_track_valid_fraction,
        minimum_track_valid_samples=args.minimum_track_valid_samples,
        max_inward_thickness_px=args.max_inward_thickness,
        minimum_aligned_fraction=args.minimum_aligned_fraction,
        preview_max_side=args.boundary_preview_max_side,
        resume=args.resume,
    )
    return Stage2Pipeline(
        paths=paths,
        canonical_config=canonical,
        observation_config=observations,
        boundary_config=boundaries,
        resume=args.resume,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pipeline = _pipeline_from_args(args)
    if args.status:
        print(json.dumps(pipeline.status(), indent=2, sort_keys=True))
        return 0
    if args.verify_only:
        print(json.dumps(pipeline.verify(), indent=2, sort_keys=True))
        return 0
    try:
        state = pipeline.run(stop_after=args.stop_after)
    except KeyboardInterrupt:
        print(
            "\nStopped safely. Re-run the same command with --resume; only the "
            "interrupted puzzle will be repeated."
        )
        return 130
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0

