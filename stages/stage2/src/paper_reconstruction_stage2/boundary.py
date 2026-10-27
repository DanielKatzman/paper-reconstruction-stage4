from __future__ import annotations

import colorsys
import hashlib
import json
import shutil
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.spatial import cKDTree

from . import __version__


BOUNDARY_DATASET_SCHEMA_VERSION = "stage2.boundary-dataset.v1"
BOUNDARY_PUZZLE_SCHEMA_VERSION = "stage2.boundary-puzzle.v1"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _array_bundle_sha256(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(_canonical_json(list(array.shape)))
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(slots=True)
class BoundaryConfig:
    canonical_dataset_dir: Path
    observation_dataset_dir: Path
    output_dir: Path
    boundary_spacing_px: float = 2.0
    minimum_boundary_points: int = 64
    maximum_boundary_points: int = 4096
    tangent_smoothing_sigma_samples: float = 1.5
    truth_sigma_px: float = 3.0
    role_temperature_px: float = 1.5
    junction_sigma_px: float = 2.0
    max_snap_distance_px: float = 12.0
    correspondence_spacing_px: float = 3.0
    minimum_matchable_seam_length_px: float = 24.0
    minimum_track_valid_fraction: float = 0.60
    minimum_track_valid_samples: int = 8
    max_inward_thickness_px: int = 96
    minimum_aligned_fraction: float = 0.80
    preview_max_side: int = 1600
    resume: bool = False

    def validate(self) -> None:
        if not 0.25 <= self.boundary_spacing_px <= 20.0:
            raise ValueError("boundary_spacing_px must be in [0.25, 20]")
        if self.minimum_boundary_points < 16:
            raise ValueError("minimum_boundary_points must be at least 16")
        if self.maximum_boundary_points < self.minimum_boundary_points:
            raise ValueError(
                "maximum_boundary_points must be >= minimum_boundary_points"
            )
        if not 0.0 <= self.tangent_smoothing_sigma_samples <= 20.0:
            raise ValueError("tangent_smoothing_sigma_samples must be in [0, 20]")
        if self.truth_sigma_px <= 0.0:
            raise ValueError("truth_sigma_px must be positive")
        if self.role_temperature_px <= 0.0:
            raise ValueError("role_temperature_px must be positive")
        if self.junction_sigma_px <= 0.0:
            raise ValueError("junction_sigma_px must be positive")
        if self.max_snap_distance_px < self.truth_sigma_px:
            raise ValueError("max_snap_distance_px must be >= truth_sigma_px")
        if self.correspondence_spacing_px <= 0.0:
            raise ValueError("correspondence_spacing_px must be positive")
        if self.minimum_matchable_seam_length_px < 0.0:
            raise ValueError("minimum_matchable_seam_length_px must be non-negative")
        if not 0.0 <= self.minimum_track_valid_fraction <= 1.0:
            raise ValueError("minimum_track_valid_fraction must be in [0, 1]")
        if self.minimum_track_valid_samples < 1:
            raise ValueError("minimum_track_valid_samples must be positive")
        if self.max_inward_thickness_px < 1:
            raise ValueError("max_inward_thickness_px must be positive")
        if not 0.0 <= self.minimum_aligned_fraction <= 1.0:
            raise ValueError("minimum_aligned_fraction must be in [0, 1]")
        if self.preview_max_side < 128:
            raise ValueError("preview_max_side must be at least 128")

    def public_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("canonical_dataset_dir", None)
        result.pop("observation_dataset_dir", None)
        result.pop("output_dir", None)
        result.pop("resume", None)
        return result


def _transform_points(points_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    transformed = (matrix @ homogeneous.T).T
    denominators = transformed[:, 2:3]
    if np.any(np.abs(denominators) < 1e-12):
        raise RuntimeError("homography maps truth points to infinity")
    return transformed[:, :2] / denominators


def _oriented_grid_edges(mask: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Return clockwise grid edges with foreground pixels on the right."""
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []

    top = mask & ~np.pad(mask[:-1, :], ((1, 0), (0, 0)), constant_values=False)
    ys, xs = np.nonzero(top)
    edges.extend(
        ((int(x), int(y)), (int(x + 1), int(y)))
        for y, x in zip(ys.tolist(), xs.tolist(), strict=True)
    )

    right = mask & ~np.pad(mask[:, 1:], ((0, 0), (0, 1)), constant_values=False)
    ys, xs = np.nonzero(right)
    edges.extend(
        ((int(x + 1), int(y)), (int(x + 1), int(y + 1)))
        for y, x in zip(ys.tolist(), xs.tolist(), strict=True)
    )

    bottom = mask & ~np.pad(mask[1:, :], ((0, 1), (0, 0)), constant_values=False)
    ys, xs = np.nonzero(bottom)
    edges.extend(
        ((int(x + 1), int(y + 1)), (int(x), int(y + 1)))
        for y, x in zip(ys.tolist(), xs.tolist(), strict=True)
    )

    left = mask & ~np.pad(mask[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    ys, xs = np.nonzero(left)
    edges.extend(
        ((int(x), int(y + 1)), (int(x), int(y)))
        for y, x in zip(ys.tolist(), xs.tolist(), strict=True)
    )
    if not edges:
        raise ValueError(f"mask has no boundary edges ({height}x{width})")
    return edges


def extract_closed_contour(mask: np.ndarray) -> np.ndarray:
    """Trace the largest Eulerian boundary loop of a connected binary mask."""
    edges = _oriented_grid_edges(mask)
    outgoing: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for start, end in edges:
        outgoing[start].append(end)
    for values in outgoing.values():
        values.sort(reverse=True)

    circuits: list[list[tuple[int, int]]] = []
    while True:
        starts = [vertex for vertex, values in outgoing.items() if values]
        if not starts:
            break
        stack = [min(starts)]
        circuit: list[tuple[int, int]] = []
        while stack:
            current = stack[-1]
            if outgoing.get(current):
                stack.append(outgoing[current].pop())
            else:
                circuit.append(stack.pop())
        circuit.reverse()
        if len(circuit) >= 4:
            circuits.append(circuit)
    if not circuits:
        raise ValueError("could not trace a closed mask contour")
    circuit = max(circuits, key=len)
    points = np.asarray(circuit, dtype=np.float64)
    if not np.array_equal(points[0], points[-1]):
        points = np.vstack((points, points[0]))
    if len(points) < 4:
        raise ValueError("traced contour is too short")
    return points


def _resample_closed_polyline(
    points_xy: np.ndarray,
    spacing_px: float,
    minimum_points: int,
    maximum_points: int,
) -> tuple[np.ndarray, float, float]:
    points = np.asarray(points_xy, dtype=np.float64)
    if not np.allclose(points[0], points[-1]):
        points = np.vstack((points, points[0]))
    segment_vectors = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    keep = segment_lengths > 1e-9
    segment_vectors = segment_vectors[keep]
    segment_lengths = segment_lengths[keep]
    starts = points[:-1][keep]
    perimeter = float(segment_lengths.sum())
    if perimeter <= 0.0:
        raise ValueError("contour perimeter is zero")
    sample_count = int(round(perimeter / spacing_px))
    sample_count = max(minimum_points, min(maximum_points, sample_count))
    actual_spacing = perimeter / sample_count
    distances = np.arange(sample_count, dtype=np.float64) * actual_spacing
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    indices = np.searchsorted(cumulative, distances, side="right") - 1
    indices = np.clip(indices, 0, len(segment_lengths) - 1)
    fractions = (distances - cumulative[indices]) / segment_lengths[indices]
    samples = starts[indices] + segment_vectors[indices] * fractions[:, None]
    return samples, perimeter, actual_spacing


def _sample_mask(mask: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    x = np.rint(points_xy[:, 0]).astype(np.int64)
    y = np.rint(points_xy[:, 1]).astype(np.int64)
    valid = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
    result = np.zeros(len(points_xy), dtype=bool)
    result[valid] = mask[y[valid], x[valid]]
    return result


def boundary_features(
    mask: np.ndarray,
    points_xy: np.ndarray,
    actual_spacing_px: float,
    *,
    smoothing_sigma_samples: float,
    max_inward_thickness_px: int,
) -> dict[str, np.ndarray]:
    points = np.asarray(points_xy, dtype=np.float64)
    smooth = points.copy()
    if smoothing_sigma_samples > 0.0:
        smooth[:, 0] = ndimage.gaussian_filter1d(
            points[:, 0], smoothing_sigma_samples, mode="wrap"
        )
        smooth[:, 1] = ndimage.gaussian_filter1d(
            points[:, 1], smoothing_sigma_samples, mode="wrap"
        )
    tangent = np.roll(smooth, -1, axis=0) - np.roll(smooth, 1, axis=0)
    norms = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent /= np.maximum(norms, 1e-8)
    inward_normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))

    plus_inside = _sample_mask(mask, points + inward_normal * 1.5).mean()
    minus_inside = _sample_mask(mask, points - inward_normal * 1.5).mean()
    if plus_inside < minus_inside:
        inward_normal *= -1.0

    previous = np.roll(tangent, 1, axis=0)
    cross = previous[:, 0] * tangent[:, 1] - previous[:, 1] * tangent[:, 0]
    dot = np.sum(previous * tangent, axis=1)
    curvature = np.arctan2(cross, np.clip(dot, -1.0, 1.0)) / max(
        actual_spacing_px, 1e-8
    )

    distances = np.arange(
        1, max_inward_thickness_px + 1, dtype=np.float64
    )[None, :, None]
    ray_points = points[:, None, :] + inward_normal[:, None, :] * distances
    inside = _sample_mask(mask, ray_points.reshape(-1, 2)).reshape(
        len(points), max_inward_thickness_px
    )
    continuous_inside = np.logical_and.accumulate(inside, axis=1)
    thickness = continuous_inside.sum(axis=1).astype(np.float32)

    return {
        "tangent_xy": tangent.astype(np.float32),
        "inward_normal_xy": inward_normal.astype(np.float32),
        "curvature_per_px": curvature.astype(np.float32),
        "corner_strength": np.abs(curvature).astype(np.float32),
        "inward_thickness_px": thickness,
    }


def _resample_open_polyline(
    points_xy: np.ndarray, spacing_px: float
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_xy, dtype=np.float64)
    if len(points) < 2:
        return points.copy(), np.zeros(len(points), dtype=np.float64)
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    keep = lengths > 1e-9
    vectors = vectors[keep]
    lengths = lengths[keep]
    starts = points[:-1][keep]
    total = float(lengths.sum())
    if total <= 0.0:
        return points[:1].copy(), np.zeros(1, dtype=np.float64)
    count = max(2, int(np.ceil(total / spacing_px)) + 1)
    distances = np.linspace(0.0, total, count)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    indices = np.searchsorted(cumulative, distances, side="right") - 1
    indices = np.clip(indices, 0, len(lengths) - 1)
    fractions = (distances - cumulative[indices]) / lengths[indices]
    samples = starts[indices] + vectors[indices] * fractions[:, None]
    return samples, distances / total


def _load_truth_polylines(
    canonical_puzzle: dict[str, Any], canonical_puzzle_dir: Path
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    archive_path = canonical_puzzle_dir / canonical_puzzle["truth_path"]
    seams: dict[str, np.ndarray] = {}
    borders: dict[str, np.ndarray] = {}
    with np.load(archive_path, allow_pickle=False) as archive:
        for record in canonical_puzzle["seams"]:
            reference = record["canonical_polyline_ref"]
            offsets = archive[reference["offsets_array"]]
            index = int(reference["index"])
            seams[record["seam_id"]] = np.asarray(
                archive[reference["points_array"]][offsets[index] : offsets[index + 1]],
                dtype=np.float64,
            )
        for record in canonical_puzzle["page_border_segments"]:
            reference = record["canonical_polyline_ref"]
            offsets = archive[reference["offsets_array"]]
            index = int(reference["index"])
            borders[record["border_segment_id"]] = np.asarray(
                archive[reference["points_array"]][offsets[index] : offsets[index + 1]],
                dtype=np.float64,
            )
    return seams, borders


def _nearest_role(
    boundary_points: np.ndarray,
    transformed_polylines: list[tuple[int, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    if not transformed_polylines:
        return (
            np.full(len(boundary_points), np.inf, dtype=np.float64),
            np.full(len(boundary_points), -1, dtype=np.int32),
        )
    points = np.concatenate([item[1] for item in transformed_polylines], axis=0)
    identifiers = np.concatenate(
        [
            np.full(len(polyline), identifier, dtype=np.int32)
            for identifier, polyline in transformed_polylines
        ]
    )
    distances, indices = cKDTree(points).query(boundary_points, k=1)
    return np.asarray(distances), identifiers[np.asarray(indices)]


def soft_role_targets(
    seam_distance: np.ndarray,
    border_distance: np.ndarray,
    *,
    truth_sigma_px: float,
    role_temperature_px: float,
    junction_sigma_px: float,
    max_snap_distance_px: float,
) -> dict[str, np.ndarray]:
    seam_distance = np.asarray(seam_distance, dtype=np.float64)
    border_distance = np.asarray(border_distance, dtype=np.float64)
    nearest = np.minimum(seam_distance, border_distance)
    finite = np.isfinite(nearest)
    alignment = np.zeros(len(nearest), dtype=np.float64)
    alignment[finite] = np.exp(
        -0.5 * (nearest[finite] / truth_sigma_px) ** 2
    )
    alignment[nearest > max_snap_distance_px] = 0.0

    both = np.isfinite(seam_distance) & np.isfinite(border_distance)
    seam_choice = np.zeros(len(nearest), dtype=np.float64)
    seam_only = np.isfinite(seam_distance) & ~np.isfinite(border_distance)
    seam_choice[seam_only] = 1.0
    difference = np.clip(
        (border_distance[both] - seam_distance[both]) / role_temperature_px,
        -60.0,
        60.0,
    )
    seam_choice[both] = 1.0 / (1.0 + np.exp(-difference))

    ambiguity = np.zeros(len(nearest), dtype=np.float64)
    ambiguity[both] = np.exp(
        -np.abs(seam_distance[both] - border_distance[both]) / junction_sigma_px
    )
    uncertainty = np.clip(1.0 - alignment + 0.5 * ambiguity * alignment, 0.0, 1.0)
    assigned = 1.0 - uncertainty
    seam_target = assigned * seam_choice
    border_target = assigned * (1.0 - seam_choice)
    confidence = alignment * (1.0 - 0.5 * ambiguity)
    return {
        "seam_target": seam_target.astype(np.float32),
        "page_border_target": border_target.astype(np.float32),
        "uncertainty_target": uncertainty.astype(np.float32),
        "label_confidence": confidence.astype(np.float32),
        "nearest_truth_distance_px": nearest.astype(np.float32),
        "aligned": (nearest <= max_snap_distance_px).astype(np.uint8),
    }


def _stable_color(identifier: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], "big") / 65535.0
    saturation = 0.62 + (digest[2] / 255.0) * 0.22
    value = 0.88 + (digest[3] / 255.0) * 0.10
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return round(red * 255), round(green * 255), round(blue * 255)


def _draw_boundary_overlay(
    image: Image.Image,
    points_xy: np.ndarray,
    seam_target: np.ndarray,
    border_target: np.ndarray,
    uncertainty_target: np.ndarray,
    nearest_seam_index: np.ndarray,
    seam_ids: list[str],
    *,
    mode: str,
) -> Image.Image:
    result = image.convert("RGBA").copy()
    draw = ImageDraw.Draw(result)
    width = max(1, round(min(result.size) / 220))
    count = len(points_xy)
    for index in range(count):
        a = tuple(float(value) for value in points_xy[index])
        b = tuple(float(value) for value in points_xy[(index + 1) % count])
        if mode == "roles":
            classes = (
                float(seam_target[index]),
                float(border_target[index]),
                float(uncertainty_target[index]),
            )
            role = int(np.argmax(classes))
            color = ((238, 70, 70), (55, 205, 235), (255, 205, 65))[role]
        else:
            seam_index = int(nearest_seam_index[index])
            if seam_target[index] >= max(border_target[index], uncertainty_target[index]) and seam_index >= 0:
                color = _stable_color(seam_ids[seam_index])
            elif uncertainty_target[index] >= max(seam_target[index], border_target[index]):
                color = (255, 205, 65)
            else:
                color = (150, 155, 165)
        draw.line((a, b), fill=(*color, 255), width=width)
    return result


def _save_preview_montage(
    images: list[Image.Image],
    labels: list[str],
    output: Path,
    *,
    max_side: int,
    legend: list[tuple[str, tuple[int, int, int]]],
) -> None:
    if not images:
        return
    count = len(images)
    columns = max(2, int(np.ceil(np.sqrt(count))))
    rows = int(np.ceil(count / columns))
    cell = max(120, min(280, max_side // columns))
    header = 34
    canvas = Image.new(
        "RGBA", (columns * cell, header + rows * cell), (42, 44, 49, 255)
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    legend_x = 10
    for text, color in legend:
        draw.rectangle((legend_x, 10, legend_x + 12, 22), fill=(*color, 255))
        draw.text((legend_x + 17, 9), text, fill=(240, 240, 240, 255), font=font)
        legend_x += 20 + len(text) * 7
    for index, (opened, label) in enumerate(zip(images, labels, strict=True)):
        image = opened.copy()
        image.thumbnail((cell - 12, cell - 28), Image.Resampling.LANCZOS)
        row, column = divmod(index, columns)
        x = column * cell + (cell - image.width) // 2
        y = header + row * cell + 16 + (cell - 28 - image.height) // 2
        canvas.alpha_composite(image, (x, y))
        draw.text(
            (column * cell + 5, header + row * cell + 3),
            label,
            fill=(225, 225, 225, 255),
            font=font,
        )
    if max(canvas.size) > max_side:
        scale = max_side / max(canvas.size)
        canvas = canvas.resize(
            (round(canvas.width * scale), round(canvas.height * scale)),
            Image.Resampling.LANCZOS,
        )
    canvas.convert("RGB").save(output, quality=92)


class BoundaryDatasetCompiler:
    def __init__(self, config: BoundaryConfig):
        config.validate()
        self.config = config
        canonical_manifest_path = config.canonical_dataset_dir / "dataset_manifest.json"
        observation_manifest_path = (
            config.observation_dataset_dir / "observation_dataset_manifest.json"
        )
        if not canonical_manifest_path.is_file():
            raise FileNotFoundError(
                f"canonical dataset manifest not found: {canonical_manifest_path}"
            )
        if not observation_manifest_path.is_file():
            raise FileNotFoundError(
                f"observation dataset manifest not found: {observation_manifest_path}"
            )
        self.canonical_dataset = _read_json(canonical_manifest_path)
        self.observation_dataset = _read_json(observation_manifest_path)
        expected = self.canonical_dataset.get("dataset_fingerprint")
        observed = self.observation_dataset.get("canonical_dataset_fingerprint")
        if expected != observed:
            raise ValueError(
                "observation dataset was produced from a different canonical dataset: "
                f"expected {expected}, got {observed}"
            )

    def _prepare_output(self) -> None:
        output = self.config.output_dir
        if output.exists() and any(output.iterdir()) and not self.config.resume:
            raise FileExistsError(
                f"output directory is not empty: {output}. "
                "Choose a new directory or pass --resume; completed puzzles are "
                "never overwritten."
            )
        output.mkdir(parents=True, exist_ok=True)
        (output / "puzzles").mkdir(exist_ok=True)

    def _summary_from_existing_manifest(
        self,
        manifest: dict[str, Any],
        observation_summary: dict[str, Any],
    ) -> dict[str, Any]:
        if manifest.get("compiler_version") != __version__:
            raise RuntimeError(
                f"cannot resume {manifest.get('puzzle_id')}: compiler version "
                f"{manifest.get('compiler_version')} does not match {__version__}"
            )
        if manifest.get("boundary_config") != self.config.public_dict():
            raise RuntimeError(
                f"cannot resume {manifest.get('puzzle_id')}: boundary configuration "
                "does not match the existing output"
            )
        if (
            manifest.get("observation_puzzle_fingerprint")
            != observation_summary.get("observation_puzzle_fingerprint")
            or manifest.get("canonical_puzzle_fingerprint")
            != observation_summary.get("canonical_puzzle_fingerprint")
        ):
            raise RuntimeError(
                f"cannot resume {manifest.get('puzzle_id')}: input puzzle "
                "fingerprints do not match the existing output"
            )
        return {
            "puzzle_id": manifest["puzzle_id"],
            "boundary_manifest_path": (
                f"puzzles/{manifest['puzzle_id']}/boundary_manifest.json"
            ),
            "boundary_puzzle_fingerprint": manifest[
                "boundary_puzzle_fingerprint"
            ],
            "canonical_puzzle_fingerprint": manifest[
                "canonical_puzzle_fingerprint"
            ],
            "observation_puzzle_fingerprint": manifest[
                "observation_puzzle_fingerprint"
            ],
            "split": manifest.get("split"),
            "difficulty_tier": manifest.get("difficulty_tier"),
            "observation_count": manifest["observation_count"],
            "boundary_point_count": manifest["boundary_point_count"],
            "correspondence_track_count": manifest[
                "correspondence_track_count"
            ],
            "usable_correspondence_track_count": manifest[
                "usable_correspondence_track_count"
            ],
            "matchable_seam_count": manifest["matchable_seam_count"],
            "missing_matchable_owner_track_count": manifest["validation"][
                "missing_matchable_owner_track_count"
            ],
            "minimum_aligned_fraction": manifest["validation"][
                "minimum_aligned_fraction"
            ],
        }

    def _compile_observation(
        self,
        observation: dict[str, Any],
        observation_index: int,
        observation_puzzle_dir: Path,
        piece: dict[str, Any],
        seam_polylines: dict[str, np.ndarray],
        border_polylines: dict[str, np.ndarray],
        seam_index_by_id: dict[str, int],
        border_index_by_id: dict[str, int],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]]]:
        mask_path = observation_puzzle_dir / observation["observed_mask_path"]
        with Image.open(mask_path) as opened:
            mask = np.asarray(opened.convert("L")) >= 128
        contour = extract_closed_contour(mask)
        points, perimeter, actual_spacing = _resample_closed_polyline(
            contour,
            self.config.boundary_spacing_px,
            self.config.minimum_boundary_points,
            self.config.maximum_boundary_points,
        )
        features = boundary_features(
            mask,
            points,
            actual_spacing,
            smoothing_sigma_samples=self.config.tangent_smoothing_sigma_samples,
            max_inward_thickness_px=self.config.max_inward_thickness_px,
        )
        matrix = np.asarray(
            observation["source_page_to_observed_homography"], dtype=np.float64
        )

        transformed_seams: list[tuple[int, np.ndarray]] = []
        transformed_borders: list[tuple[int, np.ndarray]] = []
        for seam_id in piece["seam_ids"]:
            dense, _ = _resample_open_polyline(seam_polylines[seam_id], 1.0)
            transformed = _transform_points(dense, matrix)
            transformed_seams.append((seam_index_by_id[seam_id], transformed))
        for border_id in piece["page_border_segment_ids"]:
            dense, _ = _resample_open_polyline(border_polylines[border_id], 1.0)
            transformed_borders.append(
                (
                    border_index_by_id[border_id],
                    _transform_points(dense, matrix),
                )
            )
        seam_distance, nearest_seam_index = _nearest_role(points, transformed_seams)
        border_distance, nearest_border_index = _nearest_role(
            points, transformed_borders
        )
        targets = soft_role_targets(
            seam_distance,
            border_distance,
            truth_sigma_px=self.config.truth_sigma_px,
            role_temperature_px=self.config.role_temperature_px,
            junction_sigma_px=self.config.junction_sigma_px,
            max_snap_distance_px=self.config.max_snap_distance_px,
        )

        target_sum = (
            targets["seam_target"]
            + targets["page_border_target"]
            + targets["uncertainty_target"]
        )
        if not np.allclose(target_sum, 1.0, atol=1e-5):
            raise RuntimeError(
                f"role targets do not sum to one for {observation['observation_id']}"
            )
        aligned_fraction = float(targets["aligned"].mean())
        if aligned_fraction < self.config.minimum_aligned_fraction:
            raise RuntimeError(
                f"only {aligned_fraction:.3f} of boundary truth aligned for "
                f"{observation['observation_id']}; check coordinate mappings or "
                "increase --max-snap-distance"
            )

        arrays = {
            "points_xy": points.astype(np.float32),
            "arc_length_px": (
                np.arange(len(points), dtype=np.float32) * actual_spacing
            ),
            **features,
            **targets,
            "nearest_seam_index": nearest_seam_index.astype(np.int32),
            "nearest_border_index": nearest_border_index.astype(np.int32),
        }

        tracks: list[dict[str, Any]] = []
        boundary_tree = cKDTree(points)
        for seam_id in piece["seam_ids"]:
            canonical_samples, canonical_u = _resample_open_polyline(
                seam_polylines[seam_id], self.config.correspondence_spacing_px
            )
            observed_truth = _transform_points(canonical_samples, matrix)
            distances, boundary_indices = boundary_tree.query(observed_truth, k=1)
            valid = distances <= self.config.max_snap_distance_px
            tracks.append(
                {
                    "observation_index": observation_index,
                    "observation_id": observation["observation_id"],
                    "piece_id": piece["piece_id"],
                    "view_index": int(observation["view_index"]),
                    "seam_id": seam_id,
                    "seam_index": seam_index_by_id[seam_id],
                    "canonical_u": canonical_u.astype(np.float32),
                    "observed_truth_xy": observed_truth.astype(np.float32),
                    "boundary_index_local": np.asarray(
                        boundary_indices, dtype=np.int32
                    ),
                    "projection_distance_px": np.asarray(
                        distances, dtype=np.float32
                    ),
                    "valid": valid.astype(np.uint8),
                }
            )

        hard_role = np.argmax(
            np.column_stack(
                (
                    targets["seam_target"],
                    targets["page_border_target"],
                    targets["uncertainty_target"],
                )
            ),
            axis=1,
        )
        summary = {
            "observation_index": observation_index,
            "observation_id": observation["observation_id"],
            "observation_fingerprint": observation["observation_fingerprint"],
            "piece_id": piece["piece_id"],
            "view_index": int(observation["view_index"]),
            "observed_piece_dataset_path": (
                f"puzzles/{observation_puzzle_dir.name}/"
                f"{observation['observed_piece_path']}"
            ),
            "observed_mask_dataset_path": (
                f"puzzles/{observation_puzzle_dir.name}/"
                f"{observation['observed_mask_path']}"
            ),
            "point_count": len(points),
            "perimeter_px": round(perimeter, 5),
            "actual_spacing_px": round(actual_spacing, 8),
            "aligned_fraction": round(aligned_fraction, 6),
            "mean_label_confidence": round(
                float(targets["label_confidence"].mean()), 6
            ),
            "hard_role_counts": {
                "seam": int(np.sum(hard_role == 0)),
                "page_border": int(np.sum(hard_role == 1)),
                "uncertain": int(np.sum(hard_role == 2)),
            },
        }
        return arrays, summary, tracks

    def _compile_puzzle(
        self,
        puzzle_index: int,
        observation_summary: dict[str, Any],
    ) -> dict[str, Any]:
        observation_manifest_path = (
            self.config.observation_dataset_dir
            / observation_summary["observation_manifest_path"]
        )
        observation_puzzle_dir = observation_manifest_path.parent
        observation_manifest = _read_json(observation_manifest_path)
        canonical_puzzle_path = (
            self.config.canonical_dataset_dir
            / observation_manifest["canonical_puzzle_path"]
        )
        canonical_puzzle_dir = canonical_puzzle_path.parent
        canonical_puzzle = _read_json(canonical_puzzle_path)
        puzzle_id = canonical_puzzle["puzzle_id"]
        if puzzle_id != observation_manifest["puzzle_id"]:
            raise ValueError("canonical and observation puzzle IDs do not match")

        final_output_dir = self.config.output_dir / "puzzles" / puzzle_id
        output_dir = self.config.output_dir / "puzzles" / f"{puzzle_id}.partial"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        (output_dir / "previews").mkdir(parents=True, exist_ok=False)
        seam_polylines, border_polylines = _load_truth_polylines(
            canonical_puzzle, canonical_puzzle_dir
        )
        seam_ids = [record["seam_id"] for record in canonical_puzzle["seams"]]
        border_ids = [
            record["border_segment_id"]
            for record in canonical_puzzle["page_border_segments"]
        ]
        seam_index_by_id = {identifier: index for index, identifier in enumerate(seam_ids)}
        border_index_by_id = {
            identifier: index for index, identifier in enumerate(border_ids)
        }
        piece_by_id = {
            piece["piece_id"]: piece for piece in canonical_puzzle["pieces"]
        }
        seam_record_by_id = {
            record["seam_id"]: record for record in canonical_puzzle["seams"]
        }

        observations = observation_manifest["observations"]
        array_lists: dict[str, list[np.ndarray]] = defaultdict(list)
        observation_offsets = [0]
        observation_records: list[dict[str, Any]] = []
        all_tracks: list[dict[str, Any]] = []
        role_previews: list[Image.Image] = []
        seam_previews: list[Image.Image] = []
        preview_labels: list[str] = []
        preview_arrays: list[tuple[dict[str, np.ndarray], dict[str, Any]]] = []

        for observation_index, observation in enumerate(observations):
            piece = piece_by_id[observation["piece_id"]]
            arrays, summary, tracks = self._compile_observation(
                observation,
                observation_index,
                observation_puzzle_dir,
                piece,
                seam_polylines,
                border_polylines,
                seam_index_by_id,
                border_index_by_id,
            )
            for name, array in arrays.items():
                array_lists[name].append(array)
            observation_offsets.append(observation_offsets[-1] + len(arrays["points_xy"]))
            summary["boundary_offset_start"] = observation_offsets[-2]
            summary["boundary_offset_end"] = observation_offsets[-1]
            observation_records.append(summary)
            all_tracks.extend(tracks)
            preview_arrays.append((arrays, observation))

        boundary_arrays = {
            name: np.concatenate(values, axis=0)
            for name, values in array_lists.items()
        }
        boundary_arrays["observation_offsets"] = np.asarray(
            observation_offsets, dtype=np.int64
        )
        np.savez_compressed(output_dir / "boundaries.npz", **boundary_arrays)

        track_offsets = [0]
        track_arrays: dict[str, list[np.ndarray]] = defaultdict(list)
        track_records: list[dict[str, Any]] = []
        for track_index, track in enumerate(all_tracks):
            point_count = len(track["canonical_u"])
            for name in (
                "canonical_u",
                "observed_truth_xy",
                "boundary_index_local",
                "projection_distance_px",
                "valid",
            ):
                track_arrays[name].append(track.pop(name))
            track_offsets.append(track_offsets[-1] + point_count)
            seam_record = seam_record_by_id[track["seam_id"]]
            owner_role = (
                "owner_a"
                if track["piece_id"] == seam_record["owner_a"]
                else "owner_b"
            )
            direction = seam_record[f"{owner_role}_direction"]
            valid_values = track_arrays["valid"][-1]
            boundary_indices = track_arrays["boundary_index_local"][-1]
            valid_sample_count = int(valid_values.sum())
            required_valid_samples = min(
                self.config.minimum_track_valid_samples, point_count
            )
            seam_length_px = float(seam_record["length_px"])
            usable_for_matching = bool(
                seam_length_px >= self.config.minimum_matchable_seam_length_px
                and float(valid_values.mean())
                >= self.config.minimum_track_valid_fraction
                and valid_sample_count >= required_valid_samples
            )
            track_records.append(
                {
                    "track_index": track_index,
                    **track,
                    "owner_role": owner_role,
                    "owner_direction": direction,
                    "track_offset_start": track_offsets[-2],
                    "track_offset_end": track_offsets[-1],
                    "sample_count": point_count,
                    "canonical_seam_length_px": seam_length_px,
                    "valid_sample_count": valid_sample_count,
                    "valid_fraction": round(float(valid_values.mean()), 6),
                    "usable_for_matching": usable_for_matching,
                    "unique_boundary_point_count": int(
                        len(np.unique(boundary_indices[valid_values.astype(bool)]))
                    ),
                }
            )
        correspondence_arrays = {
            name: (
                np.concatenate(values, axis=0)
                if values
                else np.empty(
                    (0, 2) if name == "observed_truth_xy" else (0,),
                    dtype=np.float32,
                )
            )
            for name, values in track_arrays.items()
        }
        if "boundary_index_local" in correspondence_arrays:
            correspondence_arrays["boundary_index_local"] = correspondence_arrays[
                "boundary_index_local"
            ].astype(np.int32)
        if "valid" in correspondence_arrays:
            correspondence_arrays["valid"] = correspondence_arrays["valid"].astype(
                np.uint8
            )
        correspondence_arrays["track_offsets"] = np.asarray(
            track_offsets, dtype=np.int64
        )
        np.savez_compressed(
            output_dir / "correspondences.npz", **correspondence_arrays
        )

        for arrays, observation in preview_arrays:
            image_path = observation_puzzle_dir / observation["observed_piece_path"]
            with Image.open(image_path) as opened:
                image = opened.convert("RGBA")
            role_preview = _draw_boundary_overlay(
                    image,
                    arrays["points_xy"],
                    arrays["seam_target"],
                    arrays["page_border_target"],
                    arrays["uncertainty_target"],
                    arrays["nearest_seam_index"],
                    seam_ids,
                    mode="roles",
                )
            seam_preview = _draw_boundary_overlay(
                    image,
                    arrays["points_xy"],
                    arrays["seam_target"],
                    arrays["page_border_target"],
                    arrays["uncertainty_target"],
                    arrays["nearest_seam_index"],
                    seam_ids,
                    mode="seams",
                )
            role_preview.thumbnail((360, 360), Image.Resampling.LANCZOS)
            seam_preview.thumbnail((360, 360), Image.Resampling.LANCZOS)
            role_previews.append(role_preview)
            seam_previews.append(seam_preview)
            preview_labels.append(
                f"{observation['piece_id'].replace('piece_', 'p')}/"
                f"v{int(observation['view_index'])}"
            )
        _save_preview_montage(
            role_previews,
            preview_labels,
            output_dir / "previews" / "boundary_roles.png",
            max_side=self.config.preview_max_side,
            legend=[
                ("seam", (238, 70, 70)),
                ("page border", (55, 205, 235)),
                ("uncertain", (255, 205, 65)),
            ],
        )
        _save_preview_montage(
            seam_previews,
            preview_labels,
            output_dir / "previews" / "seam_ids.png",
            max_side=self.config.preview_max_side,
            legend=[
                ("matched seams share colors", (232, 84, 84)),
                ("page border", (150, 155, 165)),
                ("uncertain", (255, 205, 65)),
            ],
        )

        boundary_sha = _array_bundle_sha256(boundary_arrays)
        correspondence_sha = _array_bundle_sha256(correspondence_arrays)
        minimum_alignment = min(
            record["aligned_fraction"] for record in observation_records
        )
        mean_alignment = float(
            np.mean([record["aligned_fraction"] for record in observation_records])
        )
        matchable_seam_ids = {
            record["seam_id"]
            for record in canonical_puzzle["seams"]
            if float(record["length_px"])
            >= self.config.minimum_matchable_seam_length_px
        }
        usable_owner_pairs = {
            (record["seam_id"], record["owner_role"])
            for record in track_records
            if record["usable_for_matching"]
        }
        missing_owner_tracks = [
            {"seam_id": seam_id, "owner_role": owner_role}
            for seam_id in sorted(matchable_seam_ids)
            for owner_role in ("owner_a", "owner_b")
            if (seam_id, owner_role) not in usable_owner_pairs
        ]
        usable_track_count = sum(
            record["usable_for_matching"] for record in track_records
        )
        manifest = {
            "schema_version": BOUNDARY_PUZZLE_SCHEMA_VERSION,
            "compiler_version": __version__,
            "puzzle_id": puzzle_id,
            "canonical_puzzle_fingerprint": canonical_puzzle[
                "puzzle_fingerprint"
            ],
            "observation_puzzle_fingerprint": observation_manifest[
                "observation_puzzle_fingerprint"
            ],
            "split": canonical_puzzle.get("split"),
            "difficulty_tier": canonical_puzzle.get("difficulty_tier"),
            "boundary_config": self.config.public_dict(),
            "coordinate_system": {
                "frame": "individual observed-piece pixels",
                "x_direction": "right",
                "y_direction": "down",
                "contour_order": "clockwise with the piece interior on the right",
                "homography_truth_semantics": (
                    "canonical truth is projected, then softly snapped to the "
                    "possibly perturbed observed mask boundary"
                ),
            },
            "seam_ids": seam_ids,
            "page_border_segment_ids": border_ids,
            "boundaries_path": "boundaries.npz",
            "boundary_arrays_sha256": boundary_sha,
            "correspondences_path": "correspondences.npz",
            "correspondence_arrays_sha256": correspondence_sha,
            "observation_count": len(observation_records),
            "boundary_point_count": int(len(boundary_arrays["points_xy"])),
            "correspondence_track_count": len(track_records),
            "usable_correspondence_track_count": usable_track_count,
            "matchable_seam_count": len(matchable_seam_ids),
            "observations": observation_records,
            "correspondence_tracks": track_records,
            "validation": {
                "valid": True,
                "minimum_aligned_fraction": round(minimum_alignment, 6),
                "mean_aligned_fraction": round(mean_alignment, 6),
                "all_role_targets_sum_to_one": True,
                "finite_boundary_arrays": bool(
                    all(np.all(np.isfinite(array)) for array in boundary_arrays.values())
                ),
                "matchable_seams_with_both_owners_covered": (
                    len(missing_owner_tracks) == 0
                ),
                "missing_matchable_owner_track_count": len(missing_owner_tracks),
                "missing_matchable_owner_tracks": missing_owner_tracks,
            },
        }
        fingerprint_payload = deepcopy(manifest)
        manifest["boundary_puzzle_fingerprint"] = _sha256_bytes(
            _canonical_json(fingerprint_payload)
        )
        _write_json(output_dir / "boundary_manifest.json", manifest)
        output_dir.replace(final_output_dir)
        return {
            "puzzle_id": puzzle_id,
            "boundary_manifest_path": f"puzzles/{puzzle_id}/boundary_manifest.json",
            "boundary_puzzle_fingerprint": manifest[
                "boundary_puzzle_fingerprint"
            ],
            "canonical_puzzle_fingerprint": canonical_puzzle[
                "puzzle_fingerprint"
            ],
            "observation_puzzle_fingerprint": observation_manifest[
                "observation_puzzle_fingerprint"
            ],
            "split": canonical_puzzle.get("split"),
            "difficulty_tier": canonical_puzzle.get("difficulty_tier"),
            "observation_count": len(observation_records),
            "boundary_point_count": int(len(boundary_arrays["points_xy"])),
            "correspondence_track_count": len(track_records),
            "usable_correspondence_track_count": usable_track_count,
            "matchable_seam_count": len(matchable_seam_ids),
            "missing_matchable_owner_track_count": len(missing_owner_tracks),
            "minimum_aligned_fraction": round(minimum_alignment, 6),
        }

    def compile(self) -> dict[str, Any]:
        self._prepare_output()
        started = time.perf_counter()
        puzzle_summaries: list[dict[str, Any]] = []
        observation_puzzles = self.observation_dataset.get("puzzles", [])
        for puzzle_index, observation_summary in enumerate(observation_puzzles):
            puzzle_id = observation_summary["puzzle_id"]
            existing_manifest_path = (
                self.config.output_dir
                / "puzzles"
                / puzzle_id
                / "boundary_manifest.json"
            )
            reused = False
            if self.config.resume and existing_manifest_path.is_file():
                summary = self._summary_from_existing_manifest(
                    _read_json(existing_manifest_path), observation_summary
                )
                reused = True
            else:
                final_puzzle_dir = (
                    self.config.output_dir / "puzzles" / puzzle_id
                )
                if final_puzzle_dir.exists():
                    raise RuntimeError(
                        f"incomplete or incompatible puzzle output exists: "
                        f"{final_puzzle_dir}"
                    )
                summary = self._compile_puzzle(puzzle_index, observation_summary)
            puzzle_summaries.append(summary)
            elapsed = time.perf_counter() - started
            rate = (puzzle_index + 1) / max(elapsed, 1e-9)
            remaining = (len(observation_puzzles) - puzzle_index - 1) / max(
                rate, 1e-9
            )
            print(
                f"[{puzzle_index + 1}/{len(observation_puzzles)}] "
                f"{summary['puzzle_id']}: {summary['boundary_point_count']} boundary "
                f"points, {summary['correspondence_track_count']} seam tracks"
                f"{' (reused)' if reused else ''}, "
                f"ETA {remaining:.1f}s",
                flush=True,
            )

        split_counts: dict[str, int] = defaultdict(int)
        for summary in puzzle_summaries:
            split_counts[str(summary["split"])] += 1
        payload = {
            "schema_version": BOUNDARY_DATASET_SCHEMA_VERSION,
            "compiler_version": __version__,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "canonical_dataset_fingerprint": self.canonical_dataset[
                "dataset_fingerprint"
            ],
            "observation_dataset_fingerprint": self.observation_dataset[
                "observation_dataset_fingerprint"
            ],
            "boundary_config": self.config.public_dict(),
            "puzzle_count": len(puzzle_summaries),
            "observation_count": sum(
                item["observation_count"] for item in puzzle_summaries
            ),
            "boundary_point_count": sum(
                item["boundary_point_count"] for item in puzzle_summaries
            ),
            "correspondence_track_count": sum(
                item["correspondence_track_count"] for item in puzzle_summaries
            ),
            "usable_correspondence_track_count": sum(
                item["usable_correspondence_track_count"]
                for item in puzzle_summaries
            ),
            "matchable_seam_count": sum(
                item["matchable_seam_count"] for item in puzzle_summaries
            ),
            "missing_matchable_owner_track_count": sum(
                item["missing_matchable_owner_track_count"]
                for item in puzzle_summaries
            ),
            "split_counts": dict(sorted(split_counts.items())),
            "puzzles": puzzle_summaries,
        }
        fingerprint_payload = deepcopy(payload)
        fingerprint_payload.pop("created_utc", None)
        payload["boundary_dataset_fingerprint"] = _sha256_bytes(
            _canonical_json(fingerprint_payload)
        )
        payload["elapsed_seconds"] = round(time.perf_counter() - started, 4)
        _write_json(
            self.config.output_dir / "boundary_dataset_manifest.json", payload
        )
        return payload
