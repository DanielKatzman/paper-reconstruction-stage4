from __future__ import annotations

import argparse
import json
from pathlib import Path

from .onnx_parity import EndToEndParityConfig, validate_end_to_end_parity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare PyTorch and ONNX full-contour outputs on canonical Stage 1 "
            "pieces, including final candidate structure."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--release", type=Path, default=Path("models/release.json"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--maximum-pieces", type=int)
    parser.add_argument("--maximum-absolute-error", type=float, default=1e-4)
    parser.add_argument("--cpu-threads", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_end_to_end_parity(
        EndToEndParityConfig(
            checkpoint_path=args.checkpoint,
            release_path=args.release,
            stage1_input_path=args.input,
            output_path=args.output,
            maximum_pieces=args.maximum_pieces,
            maximum_absolute_error=args.maximum_absolute_error,
            cpu_threads=args.cpu_threads,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0
