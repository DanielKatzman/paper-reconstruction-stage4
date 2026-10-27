from __future__ import annotations

import argparse
from pathlib import Path

from .observation import ObservationCompiler, ObservationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile deterministic Stage 1-like piece observations from a "
            "canonical Stage 2 dataset."
        )
    )
    parser.add_argument(
        "--canonical",
        required=True,
        type=Path,
        help="Canonical dataset root containing dataset_manifest.json",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="New or empty output directory"
    )
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--views-per-piece", type=int, default=1)
    parser.add_argument("--scale-min", type=float, default=0.82)
    parser.add_argument("--scale-max", type=float, default=1.18)
    parser.add_argument("--perspective-jitter", type=float, default=0.025)
    parser.add_argument("--canvas-margin", type=int, default=16)
    parser.add_argument("--max-mask-morphology", type=int, default=2)
    parser.add_argument("--local-damage-probability", type=float, default=0.35)
    parser.add_argument("--max-blur-radius", type=float, default=1.2)
    parser.add_argument("--jpeg-quality-min", type=int, default=72)
    parser.add_argument("--jpeg-quality-max", type=int, default=100)
    parser.add_argument("--preview-max-side", type=int, default=1400)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed puzzle outputs and rebuild only the interrupted puzzle",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ObservationConfig(
        canonical_dataset_dir=args.canonical,
        output_dir=args.output,
        seed=args.seed,
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
        preview_max_side=args.preview_max_side,
        resume=args.resume,
    )
    manifest = ObservationCompiler(config).compile()
    print(
        f"Observation dataset complete: {manifest['observation_count']} observations, "
        f"fingerprint {manifest['observation_dataset_fingerprint']}"
    )
    return 0
