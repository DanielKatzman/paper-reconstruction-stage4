from __future__ import annotations

import argparse
import json
from pathlib import Path

from .verification import verify_stage2_output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a completed Stage 2 edge-hypothesis export."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--release",
        type=Path,
        help="Optional release.json and sibling ONNX model to verify.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Optional ONNX model to verify directly against the manifest.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(
        json.dumps(
            verify_stage2_output(
                args.input, release_path=args.release, model_path=args.model
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0
