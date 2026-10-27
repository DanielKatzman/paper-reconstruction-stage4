from __future__ import annotations

import argparse
from pathlib import Path

from .boundary import BoundaryConfig, BoundaryDatasetCompiler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile ordered observed-piece boundaries, soft boundary-role truth, "
            "and canonical seam correspondence tracks."
        )
    )
    parser.add_argument(
        "--canonical",
        required=True,
        type=Path,
        help="Canonical dataset root containing dataset_manifest.json",
    )
    parser.add_argument(
        "--observations",
        required=True,
        type=Path,
        help="Observation dataset root containing observation_dataset_manifest.json",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="New or empty output directory"
    )
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
    parser.add_argument("--preview-max-side", type=int, default=1600)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed puzzle outputs and rebuild only missing puzzles",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BoundaryConfig(
        canonical_dataset_dir=args.canonical,
        observation_dataset_dir=args.observations,
        output_dir=args.output,
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
        preview_max_side=args.preview_max_side,
        resume=args.resume,
    )
    manifest = BoundaryDatasetCompiler(config).compile()
    print(
        f"Boundary dataset complete: {manifest['observation_count']} observations, "
        f"{manifest['boundary_point_count']} boundary points, fingerprint "
        f"{manifest['boundary_dataset_fingerprint']}"
    )
    return 0
