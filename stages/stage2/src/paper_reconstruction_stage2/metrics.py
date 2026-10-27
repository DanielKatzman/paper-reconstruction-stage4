from __future__ import annotations

from math import pi

import numpy as np
from scipy.ndimage import gaussian_filter1d


def _piece_perimeter(mask: np.ndarray) -> int:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    perimeter = 0
    perimeter += int(np.sum(center & ~padded[:-2, 1:-1]))
    perimeter += int(np.sum(center & ~padded[2:, 1:-1]))
    perimeter += int(np.sum(center & ~padded[1:-1, :-2]))
    perimeter += int(np.sum(center & ~padded[1:-1, 2:]))
    return perimeter


def add_piece_metrics(labels: np.ndarray, pieces: list[dict]) -> None:
    for piece in pieces:
        mask = labels == piece["label"]
        area = float(piece["area_px"])
        _, _, width, height = piece["bbox_xywh"]
        perimeter = float(_piece_perimeter(mask))
        fill_ratio = area / max(float(width * height), 1.0)
        compactness = 4.0 * pi * area / max(perimeter * perimeter, 1.0)
        characteristic_thickness = 2.0 * area / max(perimeter, 1.0)
        piece["geometry_metrics"] = {
            "perimeter_px": round(perimeter, 4),
            "bbox_fill_ratio": round(fill_ratio, 6),
            "compactness": round(compactness, 6),
            "characteristic_thickness_px": round(characteristic_thickness, 4),
            "aspect_ratio": round(max(width, height) / max(min(width, height), 1), 6),
            "extreme_shape_flag": bool(
                fill_ratio < 0.12
                or compactness < 0.025
                or characteristic_thickness < 3.0
            ),
        }


def add_seam_metrics(seams: list[dict]) -> None:
    for seam in seams:
        points = np.asarray(seam["canonical_polyline_xy"], dtype=np.float64)
        chord = float(np.linalg.norm(points[-1] - points[0])) if len(points) > 1 else 0.0
        smoothed = points.copy()
        if len(points) >= 7:
            smoothed[:, 0] = gaussian_filter1d(points[:, 0], 2.0, mode="nearest")
            smoothed[:, 1] = gaussian_filter1d(points[:, 1], 2.0, mode="nearest")
        vectors = np.diff(smoothed, axis=0)
        nonzero = np.linalg.norm(vectors, axis=1) > 1e-8
        vectors = vectors[nonzero]
        if len(vectors) >= 2:
            angles = np.unwrap(np.arctan2(vectors[:, 1], vectors[:, 0]))
            turns = np.abs(np.diff(angles))
            mean_turn = float(np.mean(turns))
            p95_turn = float(np.percentile(turns, 95))
        else:
            mean_turn = 0.0
            p95_turn = 0.0
        seam["geometry_metrics"] = {
            "chord_px": round(chord, 4),
            "tortuosity": round(float(seam["length_px"]) / max(chord, 1e-6), 6),
            "mean_abs_turn_rad": round(mean_turn, 6),
            "p95_abs_turn_rad": round(p95_turn, 6),
        }


def puzzle_geometry_metrics(pieces: list[dict], seams: list[dict]) -> dict:
    areas = np.asarray([piece["area_px"] for piece in pieces], dtype=np.float64)
    extreme_fraction = float(
        np.mean(
            [piece["geometry_metrics"]["extreme_shape_flag"] for piece in pieces]
        )
    )
    tortuosities = np.asarray(
        [seam["geometry_metrics"]["tortuosity"] for seam in seams],
        dtype=np.float64,
    )
    return {
        "piece_area_coefficient_of_variation": round(
            float(areas.std() / max(areas.mean(), 1e-8)), 6
        ),
        "extreme_piece_fraction": round(extreme_fraction, 6),
        "mean_seam_tortuosity": round(
            float(tortuosities.mean()) if len(tortuosities) else 0.0, 6
        ),
        "p95_seam_tortuosity": round(
            float(np.percentile(tortuosities, 95)) if len(tortuosities) else 0.0,
            6,
        ),
    }

