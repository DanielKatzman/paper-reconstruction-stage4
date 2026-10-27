from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import GeneratorConfig
from .generator import DatasetGenerator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate canonical variable-size Stage 2 paper puzzles."
    )
    parser.add_argument("--config", type=Path, help="Optional YAML configuration")
    parser.add_argument("--output", type=Path, help="New or empty output directory")
    parser.add_argument("--sources", type=Path, help="Directory containing images/PDFs")
    parser.add_argument("--count", type=int, help="Number of puzzles")
    parser.add_argument("--seed", type=int, help="Root deterministic seed")
    parser.add_argument("--min-pieces", type=int, help="Minimum total pieces (>=2)")
    parser.add_argument("--max-pieces", type=int, help="Maximum total pieces")
    parser.add_argument("--width", type=int, help="Canonical page width")
    parser.add_argument("--height", type=int, help="Canonical page height")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed puzzle outputs and rebuild only the interrupted puzzle",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> GeneratorConfig:
    if args.config:
        config = GeneratorConfig.from_yaml(
            args.config,
            output_dir=args.output,
            source_dir=args.sources,
        )
    else:
        if args.output is None:
            raise SystemExit("--output is required when --config is not supplied")
        config = GeneratorConfig(output_dir=args.output, source_dir=args.sources)

    replacements = {}
    mapping = {
        "count": "puzzle_count",
        "seed": "seed",
        "min_pieces": "min_pieces",
        "max_pieces": "max_pieces",
        "width": "page_width",
        "height": "page_height",
        "resume": "resume",
    }
    for argument, field_name in mapping.items():
        value = getattr(args, argument)
        if value is not None:
            replacements[field_name] = value
    if args.output is not None:
        replacements["output_dir"] = args.output
    if args.sources is not None:
        replacements["source_dir"] = args.sources
    config = replace(config, **replacements)
    config.validate()
    return config


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _config_from_args(args)
    manifest = DatasetGenerator(config).generate()
    print(
        f"Dataset complete: {len(manifest['puzzles'])} puzzles, "
        f"fingerprint {manifest['dataset_fingerprint']}"
    )
    return 0
