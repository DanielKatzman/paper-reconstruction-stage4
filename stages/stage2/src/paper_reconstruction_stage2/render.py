from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _stable_color(label: int) -> tuple[int, int, int]:
    digest = hashlib.sha256(str(label).encode("ascii")).digest()
    return tuple(75 + (value % 150) for value in digest[:3])


def save_piece_artifacts(
    source: Image.Image,
    labels: np.ndarray,
    pieces: list[dict],
    puzzle_dir: Path,
) -> None:
    directory = puzzle_dir / "canonical_pieces"
    directory.mkdir(parents=True, exist_ok=True)
    source_rgba = source.convert("RGBA")
    for piece in pieces:
        x, y, width, height = piece["bbox_xywh"]
        crop_box = (x, y, x + width, y + height)
        crop = source_rgba.crop(crop_box)
        mask_array = ((labels[y : y + height, x : x + width] == piece["label"]) * 255).astype(np.uint8)
        mask = Image.fromarray(mask_array, mode="L")
        crop.putalpha(mask)
        image_name = f"{piece['piece_id']}.png"
        mask_name = f"{piece['piece_id']}_mask.png"
        crop.save(directory / image_name)
        mask.save(directory / mask_name)
        piece["canonical_piece_path"] = f"canonical_pieces/{image_name}"
        piece["canonical_mask_path"] = f"canonical_pieces/{mask_name}"


def save_partition_preview(source: Image.Image, labels: np.ndarray, output: Path) -> None:
    overlay = np.zeros((*labels.shape, 4), dtype=np.uint8)
    for label in np.unique(labels):
        color = _stable_color(int(label))
        mask = labels == label
        overlay[mask, :3] = color
        overlay[mask, 3] = 92
    preview = Image.alpha_composite(source.convert("RGBA"), Image.fromarray(overlay, mode="RGBA"))

    horizontal = np.zeros(labels.shape, dtype=bool)
    vertical = np.zeros(labels.shape, dtype=bool)
    horizontal[:, 1:] = labels[:, 1:] != labels[:, :-1]
    vertical[1:, :] = labels[1:, :] != labels[:-1, :]
    boundary = horizontal | vertical
    array = np.asarray(preview).copy()
    array[boundary, :3] = (220, 25, 35)
    array[boundary, 3] = 255
    Image.fromarray(array, mode="RGBA").convert("RGB").save(output)


def save_seam_preview(source: Image.Image, seams: list[dict], output: Path) -> None:
    preview = source.convert("RGB").copy()
    draw = ImageDraw.Draw(preview)
    for index, seam in enumerate(seams, start=1):
        color = _stable_color(10000 + index)
        points = [(round(x), round(y)) for x, y in seam["canonical_polyline_xy"]]
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
    preview.save(output)


def save_scattered_preview(
    pieces: list[dict],
    puzzle_dir: Path,
    output: Path,
    rng: np.random.Generator,
    max_side: int,
) -> None:
    count = len(pieces)
    columns = max(2, int(np.ceil(np.sqrt(count))))
    rows = int(np.ceil(count / columns))
    cell = max(120, min(300, max_side // columns))
    canvas = Image.new("RGBA", (columns * cell, rows * cell), (45, 47, 51, 255))
    for index, piece in enumerate(pieces):
        with Image.open(puzzle_dir / piece["canonical_piece_path"]) as opened:
            image = opened.convert("RGBA")
        angle = float(rng.uniform(-180.0, 180.0))
        image = image.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
        image.thumbnail((cell - 18, cell - 18), Image.Resampling.LANCZOS)
        row, column = divmod(index, columns)
        base_x = column * cell + (cell - image.width) // 2
        base_y = row * cell + (cell - image.height) // 2
        jitter_x = int(rng.integers(-5, 6))
        jitter_y = int(rng.integers(-5, 6))
        canvas.alpha_composite(image, (base_x + jitter_x, base_y + jitter_y))
    if max(canvas.size) > max_side:
        scale = max_side / max(canvas.size)
        canvas = canvas.resize(
            (round(canvas.width * scale), round(canvas.height * scale)),
            Image.Resampling.LANCZOS,
        )
    canvas.convert("RGB").save(output, quality=92)

