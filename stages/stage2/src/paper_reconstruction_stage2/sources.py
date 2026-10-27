from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass(frozen=True, slots=True)
class SourcePage:
    image: Image.Image
    source_id: str
    origin: str
    page_index: int | None


def discover_sources(source_dir: Path | None) -> list[Path]:
    if source_dir is None:
        return []
    if not source_dir.exists():
        raise FileNotFoundError(f"source directory does not exist: {source_dir}")
    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and (path.suffix.lower() in IMAGE_SUFFIXES or path.suffix.lower() == ".pdf")
    )


def _fit_on_page(image: Image.Image, width: int, height: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (width, height), "white")
    offset = ((width - resized.width) // 2, (height - resized.height) // 2)
    canvas.paste(resized, offset)
    return canvas


def _load_pdf(path: Path, rng: np.random.Generator) -> tuple[Image.Image, int]:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PDF input requires PyMuPDF. In the active virtual environment run "
            "'python -m pip install -r requirements.txt' or "
            "'python -m pip install PyMuPDF'."
        ) from exc
    document = fitz.open(path)
    if document.page_count < 1:
        raise ValueError(f"PDF contains no pages: {path}")
    page_index = int(rng.integers(0, document.page_count))
    page = document.load_page(page_index)
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    document.close()
    return image, page_index


def _synthetic_document(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    image = Image.new("RGB", (width, height), (250, 249, 246))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    margin_x = round(width * 0.08)
    margin_y = round(height * 0.06)
    line_height = max(12, round(height * 0.018))
    colors = [(42, 47, 55), (55, 62, 72), (25, 75, 125)]
    y = margin_y
    title_width = int(rng.uniform(0.38, 0.78) * (width - 2 * margin_x))
    draw.rounded_rectangle(
        (margin_x, y, margin_x + title_width, y + line_height),
        radius=max(2, line_height // 5),
        fill=(45, 55, 70),
    )
    y += line_height * 3
    while y < height - margin_y - line_height:
        if rng.random() < 0.10 and y + line_height * 6 < height - margin_y:
            box_width = int(rng.uniform(0.20, 0.45) * (width - 2 * margin_x))
            box_height = int(rng.uniform(3.0, 6.0) * line_height)
            x0 = int(rng.uniform(margin_x, width - margin_x - box_width))
            color = tuple(int(value) for value in rng.integers(105, 220, size=3))
            draw.rounded_rectangle(
                (x0, y, x0 + box_width, y + box_height),
                radius=max(3, line_height // 3),
                fill=color,
                outline=(80, 80, 80),
                width=1,
            )
            y += box_height + line_height
            continue
        length = int(rng.uniform(0.32, 0.95) * (width - 2 * margin_x))
        thickness = max(1, round(line_height * rng.uniform(0.12, 0.22)))
        color = colors[int(rng.integers(0, len(colors)))]
        draw.line((margin_x, y, margin_x + length, y), fill=color, width=thickness)
        if rng.random() < 0.16:
            draw.text((margin_x, y + 2), f"section {int(rng.integers(1, 100))}", fill=color, font=font)
        y += line_height
    return image


def load_source_page(
    sources: list[Path],
    width: int,
    height: int,
    rng: np.random.Generator,
    puzzle_index: int,
) -> SourcePage:
    if sources:
        path = sources[int(rng.integers(0, len(sources)))]
        page_index: int | None = None
        if path.suffix.lower() == ".pdf":
            image, page_index = _load_pdf(path, rng)
        else:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
        image = _fit_on_page(image, width, height)
        origin = str(path.resolve())
    else:
        image = _synthetic_document(width, height, rng)
        page_index = None
        origin = f"generated://synthetic-document/{puzzle_index:06d}"

    digest = hashlib.sha256()
    digest.update(image.tobytes())
    digest.update(str(page_index).encode("utf-8"))
    source_id = f"source_{digest.hexdigest()[:20]}"
    return SourcePage(image=image, source_id=source_id, origin=origin, page_index=page_index)
