#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _piece_mask(width: int, height: int, variant: int) -> np.ndarray:
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    if variant == 0:
        polygon = [(4, 3), (width - 6, 8), (width - 3, height - 5), (8, height - 2)]
    else:
        polygon = [
            (5, 6),
            (width - 4, 3),
            (width - 7, height // 2 - 5),
            (width - 2, height // 2 + 4),
            (width - 8, height - 3),
            (3, height - 7),
        ]
    draw.polygon(polygon, fill=255)
    return np.asarray(image, dtype=np.uint8)


def build_example(output: Path, *, force: bool = False) -> Path:
    if output.exists() and any(output.iterdir()):
        if not force:
            raise FileExistsError(
                f"example output is not empty: {output}; pass --force to recreate it"
            )
        shutil.rmtree(output)
    for name in ("captures", "masks", "pieces", "piece_masks", "overlays"):
        (output / name).mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    original_height, original_width = 220, 300
    specifications = [
        ("capture_piece_b", 48, 42, 96, 128, 1),
        ("capture_piece_a", 156, 58, 84, 116, 0),
    ]
    for capture_id, x, y, width, height, variant in specifications:
        mask = _piece_mask(width, height, variant)
        yy, xx = np.mgrid[0:height, 0:width]
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[..., 0] = np.clip(225 + xx // 10, 0, 255)
        rgb[..., 1] = np.clip(220 + yy // 14, 0, 255)
        rgb[..., 2] = 205
        rgba = np.dstack((rgb, mask))

        piece_path = output / "pieces" / f"{capture_id}.png"
        piece_mask_path = output / "piece_masks" / f"{capture_id}.png"
        full_mask_path = output / "masks" / f"{capture_id}.png"
        source_path = output / "captures" / f"{capture_id}.png"
        overlay_path = output / "overlays" / f"{capture_id}.png"

        Image.fromarray(rgba, mode="RGBA").save(piece_path)
        Image.fromarray(mask, mode="L").save(piece_mask_path)
        full_mask = np.zeros((original_height, original_width), dtype=np.uint8)
        full_mask[y : y + height, x : x + width] = mask
        Image.fromarray(full_mask, mode="L").save(full_mask_path)
        source = Image.new("RGB", (original_width, original_height), (42, 47, 54))
        source.paste(Image.fromarray(rgb, mode="RGB"), (x, y), Image.fromarray(mask))
        source.save(source_path)
        overlay = source.copy()
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle((x, y, x + width, y + height), outline=(255, 80, 70), width=2)
        overlay.save(overlay_path)

        rows.append(
            {
                "schema_version": "stage1.segmentation-manifest.v1",
                "capture_id": capture_id,
                "source_path": source_path.relative_to(output).as_posix(),
                "source_sha256": _sha256(source_path),
                "capture_type_hint": "example",
                "original_size_hw": [original_height, original_width],
                "bbox_xywh": [x, y, width, height],
                "mask_path": full_mask_path.relative_to(output).as_posix(),
                "piece_path": piece_path.relative_to(output).as_posix(),
                "piece_mask_path": piece_mask_path.relative_to(output).as_posix(),
                "overlay_path": overlay_path.relative_to(output).as_posix(),
                "model_release": "stage1-example",
                "model_version": "1.0.0",
                "model_sha256": "0" * 64,
                "dataset_fingerprint": "0" * 64,
                "threshold": 0.5,
                "postprocess_policy": "example",
                "decision_source": "manual_example",
                "review_status": "reviewed",
                "accepted": True,
                "reason_codes": [],
                "manual_changed_fraction": 0.0,
                "cache_reused": False,
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }
        )

    manifest = output / "segmentation_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output / "segmentation_run.json").write_text(
        json.dumps(
            {
                "schema_version": "stage1.segmentation-run.example.v1",
                "status": "complete",
                "capture_count": len(rows),
                "manifest_path": "segmentation_manifest.jsonl",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a tiny Stage 1 handoff example.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "examples" / "stage1_output",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    print(build_example(args.output, force=args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

