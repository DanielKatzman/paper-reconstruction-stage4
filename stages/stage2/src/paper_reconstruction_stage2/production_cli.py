from __future__ import annotations

import argparse
import json
from pathlib import Path

from .onnx_runtime import OnnxModelRelease, predictor_from_release
from .production import (
    BoundaryGeometryConfig,
    ProductionBoundarySampler,
    ProductionInferenceConfig,
    Stage2ProductionRunner,
    load_stage1_output,
)


_SOURCE_RELEASE_PATH = Path(__file__).resolve().parents[2] / "models" / "release.json"
DEFAULT_RELEASE_PATH = (
    _SOURCE_RELEASE_PATH
    if _SOURCE_RELEASE_PATH.is_file()
    else Path("models/release.json")
)


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Stage 2 boundary-understanding release on CPU using "
            "the canonical Stage 1 segmentation_manifest.jsonl handoff."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Stage 1 output directory or segmentation_manifest.jsonl path.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--release",
        type=Path,
        default=DEFAULT_RELEASE_PATH,
        help="Frozen release.json; its sibling ONNX model is loaded automatically.",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive,
        help="Override the tested release default (normally 8).",
    )
    parser.add_argument(
        "--stride",
        type=_positive,
        help="Override the frozen contour-window stride (normally 32).",
    )
    parser.add_argument(
        "--cpu-threads",
        type=_nonnegative,
        default=0,
        help="ONNX Runtime intra-op threads; 0 lets the runtime choose.",
    )
    parser.add_argument("--maximum-pieces", type=_positive)
    parser.add_argument("--include-unaccepted", action="store_true")
    parser.add_argument("--no-overlays", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    release = OnnxModelRelease.load(args.release)
    inputs = load_stage1_output(
        args.input,
        include_unaccepted=args.include_unaccepted,
        maximum_pieces=args.maximum_pieces,
    )
    geometry_values = release.metadata.get("geometry_config", {})
    if not isinstance(geometry_values, dict):
        raise SystemExit("release geometry_config must be an object")
    geometry_config = BoundaryGeometryConfig(**geometry_values)
    geometry_config.validate()
    sampler = ProductionBoundarySampler(release.window_config, geometry_config)
    predictor = predictor_from_release(
        release,
        sampler,
        batch_size=args.batch_size,
        stride=args.stride,
        cpu_threads=args.cpu_threads,
    )
    runner = Stage2ProductionRunner(
        config=ProductionInferenceConfig(
            input_path=args.input,
            model_path=release.model_path,
            output_dir=args.output,
            release_version=release.release_version,
            release_manifest_sha256=release.release_manifest_sha256,
            source_checkpoint_sha256=release.source_checkpoint_sha256,
            write_overlays=not args.no_overlays,
            resume=args.resume,
            include_unaccepted=args.include_unaccepted,
            maximum_pieces=args.maximum_pieces,
            expected_model_sha256=release.model_sha256,
        ),
        inputs=inputs,
        sampler=sampler,
        predictor=predictor,
    )
    print(json.dumps(runner.run(), indent=2, sort_keys=True))
    return 0

