from __future__ import annotations

import argparse
import json
from pathlib import Path

from .onnx_export import OnnxExportConfig, export_onnx_release
from .release import STAGE2_RELEASE_CHECKPOINT_SHA256


def _positive_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("batch sizes must be positive")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export the frozen Stage 2 PyTorch checkpoint to the portable "
            "CPU-only ONNX production release and verify numerical parity."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/stage2_boundary_v1.0.0.onnx"),
    )
    parser.add_argument(
        "--release-output", type=Path, default=Path("models/release.json")
    )
    parser.add_argument(
        "--base-release",
        type=Path,
        default=Path("models/release.json"),
        help="Existing release metadata from which evaluation metrics are retained.",
    )
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--parity-batches", type=_positive_ints, default=(1, 2, 8))
    parser.add_argument("--maximum-absolute-error", type=float, default=1e-4)
    parser.add_argument("--default-batch-size", type=int, default=8)
    parser.add_argument("--default-stride", type=int, default=32)
    parser.add_argument(
        "--allow-unverified-checkpoint",
        action="store_true",
        help="Allow intentional research exports from a checkpoint other than v1.0.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    release = export_onnx_release(
        OnnxExportConfig(
            checkpoint_path=args.checkpoint,
            model_output_path=args.model_output,
            release_output_path=args.release_output,
            base_release_path=args.base_release,
            expected_checkpoint_sha256=(
                None
                if args.allow_unverified_checkpoint
                else STAGE2_RELEASE_CHECKPOINT_SHA256
            ),
            opset=args.opset,
            parity_batches=args.parity_batches,
            maximum_absolute_error=args.maximum_absolute_error,
            default_batch_size=args.default_batch_size,
            default_stride=args.default_stride,
        )
    )
    print(json.dumps(release, indent=2, sort_keys=True))
    return 0
