from __future__ import annotations

import hashlib
import io
import json
import shutil
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage

from . import __version__


OBSERVATION_DATASET_SCHEMA_VERSION = "stage2.observation-dataset.v1"
OBSERVATION_PUZZLE_SCHEMA_VERSION = "stage2.observation-puzzle.v1"


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


def _matrix_list(matrix: np.ndarray) -> list[list[float]]:
    return [[round(float(value), 12) for value in row] for row in matrix]


@dataclass(slots=True)
class ObservationConfig:
    canonical_dataset_dir: Path
    output_dir: Path
    seed: int = 20261005
    views_per_piece: int = 1
    scale_min: float = 0.82
    scale_max: float = 1.18
    perspective_jitter: float = 0.025
    canvas_margin_px: int = 16
    max_mask_morphology_px: int = 2
    local_boundary_damage_probability: float = 0.35
    max_blur_radius: float = 1.2
    jpeg_quality_min: int = 72
    jpeg_quality_max: int = 100
    preview_max_side: int = 1400
    resume: bool = False

    def validate(self) -> None:
        if self.views_per_piece < 1:
            raise ValueError("views_per_piece must be at least 1")
        if not 0.1 <= self.scale_min <= self.scale_max <= 4.0:
            raise ValueError("scale range must satisfy 0.1 <= min <= max <= 4.0")
        if not 0.0 <= self.perspective_jitter <= 0.15:
            raise ValueError("perspective_jitter must be in [0, 0.15]")
        if self.canvas_margin_px < 0:
            raise ValueError("canvas_margin_px must be non-negative")
        if not 0 <= self.max_mask_morphology_px <= 10:
            raise ValueError("max_mask_morphology_px must be in [0, 10]")
        if not 0.0 <= self.local_boundary_damage_probability <= 1.0:
            raise ValueError("local_boundary_damage_probability must be in [0, 1]")
        if not 0.0 <= self.max_blur_radius <= 5.0:
            raise ValueError("max_blur_radius must be in [0, 5]")
        if not 20 <= self.jpeg_quality_min <= self.jpeg_quality_max <= 100:
            raise ValueError("JPEG quality range must be within [20, 100]")
        if self.preview_max_side < 128:
            raise ValueError("preview_max_side must be at least 128")

    def public_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("canonical_dataset_dir", None)
        result.pop("output_dir", None)
        result.pop("resume", None)
        return result


def _transform_points(points_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack(
        [np.asarray(points_xy, dtype=np.float64), np.ones(len(points_xy))]
    )
    transformed = (matrix @ homogeneous.T).T
    return transformed[:, :2] / transformed[:, 2:3]


def _sample_homography(
    width: int,
    height: int,
    rng: np.random.Generator,
    config: ObservationConfig,
) -> tuple[np.ndarray, int, int, dict[str, float]]:
    angle_degrees = float(rng.uniform(-180.0, 180.0))
    angle = np.deg2rad(angle_degrees)
    scale = float(rng.uniform(config.scale_min, config.scale_max))
    cos_value = float(np.cos(angle)) * scale
    sin_value = float(np.sin(angle)) * scale
    center_x = width / 2.0
    center_y = height / 2.0
    translate_to_origin = np.array(
        [[1.0, 0.0, -center_x], [0.0, 1.0, -center_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    similarity = np.array(
        [[cos_value, -sin_value, 0.0], [sin_value, cos_value, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    perspective_x = float(
        rng.uniform(-config.perspective_jitter, config.perspective_jitter)
        / max(width, 1)
    )
    perspective_y = float(
        rng.uniform(-config.perspective_jitter, config.perspective_jitter)
        / max(height, 1)
    )
    perspective = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [perspective_x, perspective_y, 1.0]],
        dtype=np.float64,
    )
    unshifted = perspective @ similarity @ translate_to_origin
    corners = np.asarray(
        [[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]],
        dtype=np.float64,
    )
    transformed = _transform_points(corners, unshifted)
    minimum = transformed.min(axis=0)
    maximum = transformed.max(axis=0)
    margin = float(config.canvas_margin_px)
    shift = np.array(
        [[1.0, 0.0, margin - minimum[0]], [0.0, 1.0, margin - minimum[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    matrix = shift @ unshifted
    output_width = max(1, int(np.ceil(maximum[0] - minimum[0] + 2.0 * margin)))
    output_height = max(1, int(np.ceil(maximum[1] - minimum[1] + 2.0 * margin)))
    return matrix, output_width, output_height, {
        "rotation_degrees": round(angle_degrees, 8),
        "scale": round(scale, 8),
        "perspective_x": round(perspective_x, 12),
        "perspective_y": round(perspective_y, 12),
    }


def _warp_image(
    image: Image.Image,
    matrix: np.ndarray,
    output_size: tuple[int, int],
    *,
    resample: Image.Resampling,
) -> Image.Image:
    inverse = np.linalg.inv(matrix)
    inverse /= inverse[2, 2]
    coefficients = (
        inverse[0, 0],
        inverse[0, 1],
        inverse[0, 2],
        inverse[1, 0],
        inverse[1, 1],
        inverse[1, 2],
        inverse[2, 0],
        inverse[2, 1],
    )
    return image.transform(
        output_size,
        Image.Transform.PERSPECTIVE,
        tuple(float(value) for value in coefficients),
        resample=resample,
        fillcolor=0,
    )


def _perturb_mask(
    mask: np.ndarray,
    rng: np.random.Generator,
    config: ObservationConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    original = mask.astype(bool)
    candidate = original.copy()
    rollback = False
    rejected_disconnected_perturbation = False
    rejected_excessive_area_change = False
    morphology_px = int(
        rng.integers(
            -config.max_mask_morphology_px,
            config.max_mask_morphology_px + 1,
        )
    )
    if morphology_px < 0:
        candidate = ndimage.binary_erosion(candidate, iterations=-morphology_px)
    elif morphology_px > 0:
        candidate = ndimage.binary_dilation(candidate, iterations=morphology_px)

    smoothing_sigma = float(rng.uniform(0.0, 1.0))
    if smoothing_sigma >= 0.15:
        candidate = ndimage.gaussian_filter(
            candidate.astype(np.float32), smoothing_sigma
        ) >= 0.5

    candidate = ndimage.binary_fill_holes(candidate)
    _, morphology_components = ndimage.label(candidate)
    if morphology_components != 1:
        candidate = original.copy()
        rollback = True
        rejected_disconnected_perturbation = True
    elif candidate.sum() < max(16, int(original.sum() * 0.85)):
        candidate = original.copy()
        rollback = True
        rejected_excessive_area_change = True

    local_damage_applied = False
    local_damage_radius_px = 0
    if (
        candidate.any()
        and rng.random() < config.local_boundary_damage_probability
    ):
        boundary = candidate & ~ndimage.binary_erosion(candidate)
        boundary_y, boundary_x = np.nonzero(boundary)
        if len(boundary_x):
            point_index = int(rng.integers(0, len(boundary_x)))
            center_x = int(boundary_x[point_index])
            center_y = int(boundary_y[point_index])
            maximum_radius = max(1, min(6, round(min(mask.shape) * 0.025)))
            local_damage_radius_px = int(rng.integers(1, maximum_radius + 1))
            yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
            disk = (
                (xx - center_x) ** 2 + (yy - center_y) ** 2
                <= local_damage_radius_px**2
            )
            damaged = candidate & ~disk
            _, damaged_components = ndimage.label(damaged)
            if (
                damaged_components == 1
                and damaged.sum() >= max(16, int(original.sum() * 0.85))
            ):
                candidate = damaged
                local_damage_applied = True
            elif damaged_components != 1:
                rejected_disconnected_perturbation = True
            else:
                rejected_excessive_area_change = True

    candidate = ndimage.binary_fill_holes(candidate)
    _, components_before_cleanup = ndimage.label(candidate)
    if (
        components_before_cleanup != 1
        or candidate.sum() < max(16, int(original.sum() * 0.85))
    ):
        candidate = original.copy()
        rollback = True
    _, components_after_cleanup = ndimage.label(candidate)
    return candidate.astype(bool), {
        "morphology_px": morphology_px,
        "smoothing_sigma": round(smoothing_sigma, 8),
        "local_boundary_damage_applied": local_damage_applied,
        "local_boundary_damage_radius_px": local_damage_radius_px,
        "components_before_cleanup": components_before_cleanup,
        "components_after_cleanup": components_after_cleanup,
        "rollback_to_unperturbed_mask": rollback,
        "rejected_disconnected_perturbation": rejected_disconnected_perturbation,
        "rejected_excessive_area_change": rejected_excessive_area_change,
    }


def _apply_photometric_changes(
    image: Image.Image,
    rng: np.random.Generator,
    config: ObservationConfig,
) -> tuple[Image.Image, dict[str, Any]]:
    brightness = float(rng.uniform(0.86, 1.14))
    contrast = float(rng.uniform(0.86, 1.14))
    color = float(rng.uniform(0.92, 1.08))
    shadow_strength = float(rng.uniform(0.0, 0.22))
    shadow_angle = float(rng.uniform(0.0, 2.0 * np.pi))
    blur_radius = float(rng.uniform(0.0, config.max_blur_radius))
    jpeg_quality = int(
        rng.integers(config.jpeg_quality_min, config.jpeg_quality_max + 1)
    )

    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    rgb = rgba.convert("RGB")
    rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
    rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
    rgb = ImageEnhance.Color(rgb).enhance(color)

    array = np.asarray(rgb, dtype=np.float32)
    height, width = array.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    projection = xx * np.cos(shadow_angle) + yy * np.sin(shadow_angle)
    projection -= projection.min()
    projection /= max(float(projection.max()), 1.0)
    multiplier = 1.0 - shadow_strength * projection[..., None]
    array = np.clip(array * multiplier, 0, 255).astype(np.uint8)
    rgb = Image.fromarray(array, mode="RGB")
    if blur_radius >= 0.05:
        rgb = rgb.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=jpeg_quality, optimize=False)
    buffer.seek(0)
    rgb = Image.open(buffer).convert("RGB")
    result = rgb.convert("RGBA")
    result.putalpha(alpha)
    return result, {
        "brightness": round(brightness, 8),
        "contrast": round(contrast, 8),
        "color": round(color, 8),
        "shadow_strength": round(shadow_strength, 8),
        "shadow_angle_radians": round(shadow_angle, 8),
        "blur_radius": round(blur_radius, 8),
        "jpeg_quality": jpeg_quality,
    }


def _apply_observed_alpha(
    image: Image.Image,
    geometric_mask: np.ndarray,
    observed_mask: np.ndarray,
) -> Image.Image:
    array = np.asarray(image.convert("RGBA")).copy()
    added = observed_mask & ~geometric_mask
    if added.any():
        interior = array[geometric_mask, :3]
        fill = (
            np.median(interior, axis=0).round().astype(np.uint8)
            if len(interior)
            else np.asarray([245, 245, 245], dtype=np.uint8)
        )
        array[added, :3] = fill
    array[..., 3] = observed_mask.astype(np.uint8) * 255
    return Image.fromarray(array, mode="RGBA")


def _save_preview(
    observation_records: list[dict[str, Any]],
    puzzle_output_dir: Path,
    output: Path,
    max_side: int,
) -> None:
    if not observation_records:
        return
    count = len(observation_records)
    columns = max(2, int(np.ceil(np.sqrt(count))))
    rows = int(np.ceil(count / columns))
    cell = max(120, min(300, max_side // columns))
    canvas = Image.new("RGBA", (columns * cell, rows * cell), (45, 47, 51, 255))
    for index, record in enumerate(observation_records):
        with Image.open(puzzle_output_dir / record["observed_piece_path"]) as opened:
            piece = opened.convert("RGBA")
        piece.thumbnail((cell - 18, cell - 18), Image.Resampling.LANCZOS)
        row, column = divmod(index, columns)
        x = column * cell + (cell - piece.width) // 2
        y = row * cell + (cell - piece.height) // 2
        canvas.alpha_composite(piece, (x, y))
    if max(canvas.size) > max_side:
        scale = max_side / max(canvas.size)
        canvas = canvas.resize(
            (round(canvas.width * scale), round(canvas.height * scale)),
            Image.Resampling.LANCZOS,
        )
    canvas.convert("RGB").save(output, quality=92)


class ObservationCompiler:
    def __init__(self, config: ObservationConfig):
        config.validate()
        self.config = config
        manifest_path = config.canonical_dataset_dir / "dataset_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"canonical dataset manifest not found: {manifest_path}")
        self.canonical_dataset = _read_json(manifest_path)

    def _prepare_output(self) -> None:
        output = self.config.output_dir
        if output.exists() and any(output.iterdir()) and not self.config.resume:
            raise FileExistsError(
                f"output directory is not empty: {output}. "
                "Choose a new directory or pass --resume; completed puzzles are "
                "never overwritten."
            )
        if output.exists() and self.config.resume:
            allowed = {
                "observation_dataset_manifest.json",
                "observation_dataset_manifest.json.tmp",
                "puzzles",
            }
            unexpected = sorted(path.name for path in output.iterdir() if path.name not in allowed)
            if unexpected:
                raise FileExistsError(
                    f"cannot resume observation dataset with unrelated entries in "
                    f"{output}: {unexpected}"
                )
        output.mkdir(parents=True, exist_ok=True)
        (output / "puzzles").mkdir(exist_ok=True)

    def _summary_from_existing_manifest(
        self,
        manifest: dict[str, Any],
        canonical_summary: dict[str, Any],
    ) -> dict[str, Any]:
        if manifest.get("compiler_version") != __version__:
            raise RuntimeError(
                f"cannot resume {manifest.get('puzzle_id')}: compiler version "
                f"{manifest.get('compiler_version')} does not match {__version__}"
            )
        if manifest.get("observation_config") != self.config.public_dict():
            raise RuntimeError(
                f"cannot resume {manifest.get('puzzle_id')}: observation "
                "configuration changed"
            )
        if (
            manifest.get("canonical_puzzle_fingerprint")
            != canonical_summary.get("puzzle_fingerprint")
        ):
            raise RuntimeError(
                f"cannot resume {manifest.get('puzzle_id')}: canonical puzzle "
                "fingerprint changed"
            )
        puzzle_dir = (
            self.config.output_dir / "puzzles" / str(manifest["puzzle_id"])
        )
        required: list[Path] = []
        for observation in manifest.get("observations", []):
            required.extend(
                [
                    puzzle_dir / observation["observed_piece_path"],
                    puzzle_dir / observation["observed_mask_path"],
                ]
            )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(
                f"cannot resume incomplete committed observation puzzle {puzzle_dir}: "
                f"missing {missing[:3]}"
            )
        return {
            "puzzle_id": manifest["puzzle_id"],
            "observation_manifest_path": (
                f"puzzles/{manifest['puzzle_id']}/observation_manifest.json"
            ),
            "observation_puzzle_fingerprint": manifest[
                "observation_puzzle_fingerprint"
            ],
            "canonical_puzzle_fingerprint": manifest[
                "canonical_puzzle_fingerprint"
            ],
            "split": manifest.get("split"),
            "difficulty_tier": manifest.get("difficulty_tier"),
            "piece_count": manifest["piece_count"],
            "observation_count": manifest["observation_count"],
        }

    def _compile_piece_view(
        self,
        canonical_piece: dict[str, Any],
        canonical_puzzle_dir: Path,
        puzzle_output_dir: Path,
        puzzle_id: str,
        puzzle_index: int,
        piece_index: int,
        view_index: int,
    ) -> dict[str, Any]:
        sequence = np.random.SeedSequence(
            [self.config.seed, puzzle_index, piece_index, view_index]
        )
        view_seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
        rng = np.random.default_rng(sequence)
        piece_id = canonical_piece["piece_id"]
        image_path = canonical_puzzle_dir / canonical_piece["canonical_piece_path"]
        mask_path = canonical_puzzle_dir / canonical_piece["canonical_mask_path"]
        with Image.open(image_path) as opened:
            canonical_image = opened.convert("RGBA")
        with Image.open(mask_path) as opened:
            canonical_mask_image = opened.convert("L")

        matrix, output_width, output_height, geometric_parameters = _sample_homography(
            canonical_image.width,
            canonical_image.height,
            rng,
            self.config,
        )
        warped_image = _warp_image(
            canonical_image,
            matrix,
            (output_width, output_height),
            resample=Image.Resampling.BICUBIC,
        )
        warped_mask_image = _warp_image(
            canonical_mask_image,
            matrix,
            (output_width, output_height),
            resample=Image.Resampling.BILINEAR,
        )
        # A low antialiased threshold preserves one-pixel canonical necks that can
        # otherwise disconnect under rotation with nearest-neighbor sampling.
        geometric_mask = np.asarray(warped_mask_image) >= 32
        _, geometric_components = ndimage.label(geometric_mask)
        if geometric_components != 1:
            for iterations in range(1, 4):
                repaired = ndimage.binary_closing(
                    geometric_mask, iterations=iterations
                )
                _, repaired_components = ndimage.label(repaired)
                if repaired_components == 1:
                    geometric_mask = repaired
                    break
        observed_mask, mask_parameters = _perturb_mask(
            geometric_mask, rng, self.config
        )
        photographed, photometric_parameters = _apply_photometric_changes(
            warped_image, rng, self.config
        )
        observed_image = _apply_observed_alpha(
            photographed, geometric_mask, observed_mask
        )

        inverse = np.linalg.inv(matrix)
        inverse /= inverse[2, 2]
        x, y, _, _ = canonical_piece["bbox_xywh"]
        source_to_crop = np.asarray(
            [[1.0, 0.0, -float(x)], [0.0, 1.0, -float(y)], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        source_to_observed = matrix @ source_to_crop
        observed_to_source = np.linalg.inv(source_to_observed)
        observed_to_source /= observed_to_source[2, 2]

        view_id = f"{piece_id}__view_{view_index:02d}"
        image_relative = f"pieces/{view_id}.png"
        mask_relative = f"pieces/{view_id}_mask.png"
        observed_image.save(puzzle_output_dir / image_relative)
        Image.fromarray(observed_mask.astype(np.uint8) * 255, mode="L").save(
            puzzle_output_dir / mask_relative
        )

        _, connected_components = ndimage.label(observed_mask)
        roundtrip = observed_to_source @ source_to_observed
        roundtrip /= roundtrip[2, 2]
        validation = {
            "valid": bool(
                connected_components == 1
                and observed_mask.any()
                and np.allclose(roundtrip, np.eye(3), atol=1e-8)
            ),
            "connected_component_count": int(connected_components),
            "observed_area_px": int(observed_mask.sum()),
            "geometric_area_px": int(geometric_mask.sum()),
            "homography_roundtrip_max_abs_error": float(
                np.max(np.abs(roundtrip - np.eye(3)))
            ),
        }
        if not validation["valid"]:
            raise RuntimeError(
                f"invalid observation {puzzle_id}/{view_id}: {validation}"
            )

        fingerprint_payload = {
            "puzzle_id": puzzle_id,
            "piece_id": piece_id,
            "view_index": view_index,
            "view_seed": view_seed,
            "canonical_crop_to_observed_homography": _matrix_list(matrix),
            "mask_parameters": mask_parameters,
            "photometric_parameters": photometric_parameters,
        }
        return {
            "observation_id": view_id,
            "observation_fingerprint": _sha256_bytes(
                _canonical_json(fingerprint_payload)
            ),
            "piece_id": piece_id,
            "view_index": view_index,
            "view_seed": view_seed,
            "observed_piece_path": image_relative,
            "observed_mask_path": mask_relative,
            "observed_size_hw": [output_height, output_width],
            "canonical_bbox_xywh": canonical_piece["bbox_xywh"],
            "mapping_semantics": (
                "homographies map the ideal canonical geometry before observed-mask "
                "perturbation"
            ),
            "canonical_crop_to_observed_homography": _matrix_list(matrix),
            "observed_to_canonical_crop_homography": _matrix_list(inverse),
            "source_page_to_observed_homography": _matrix_list(source_to_observed),
            "observed_to_source_page_homography": _matrix_list(observed_to_source),
            "geometric_parameters": geometric_parameters,
            "mask_perturbation": mask_parameters,
            "photometric_parameters": photometric_parameters,
            "validation": validation,
        }

    def compile(self) -> dict[str, Any]:
        self._prepare_output()
        started = time.perf_counter()
        puzzle_summaries: list[dict[str, Any]] = []
        total_observations = 0
        canonical_puzzles = self.canonical_dataset.get("puzzles", [])
        for puzzle_index, puzzle_summary in enumerate(canonical_puzzles):
            existing_manifest_path = (
                self.config.output_dir
                / "puzzles"
                / puzzle_summary["puzzle_id"]
                / "observation_manifest.json"
            )
            reused = False
            if self.config.resume and existing_manifest_path.is_file():
                summary = self._summary_from_existing_manifest(
                    _read_json(existing_manifest_path), puzzle_summary
                )
                puzzle_summaries.append(summary)
                total_observations += summary["observation_count"]
                reused = True
                print(
                    f"[{puzzle_index + 1}/{len(canonical_puzzles)}] "
                    f"{summary['puzzle_id']}: {summary['observation_count']} "
                    "observations (reused)",
                    flush=True,
                )
                continue
            canonical_puzzle_path = (
                self.config.canonical_dataset_dir / puzzle_summary["puzzle_path"]
            )
            canonical_puzzle_dir = canonical_puzzle_path.parent
            canonical_puzzle = _read_json(canonical_puzzle_path)
            puzzle_id = canonical_puzzle["puzzle_id"]
            final_puzzle_dir = self.config.output_dir / "puzzles" / puzzle_id
            puzzle_output_dir = (
                self.config.output_dir / "puzzles" / f"{puzzle_id}.partial"
            )
            if puzzle_output_dir.exists():
                shutil.rmtree(puzzle_output_dir)
            if final_puzzle_dir.exists():
                raise RuntimeError(
                    f"incomplete or incompatible observation puzzle exists: "
                    f"{final_puzzle_dir}"
                )
            (puzzle_output_dir / "pieces").mkdir(parents=True, exist_ok=False)
            (puzzle_output_dir / "previews").mkdir(exist_ok=False)
            observations: list[dict[str, Any]] = []
            for piece_index, piece in enumerate(canonical_puzzle["pieces"]):
                for view_index in range(self.config.views_per_piece):
                    observations.append(
                        self._compile_piece_view(
                            piece,
                            canonical_puzzle_dir,
                            puzzle_output_dir,
                            puzzle_id,
                            puzzle_index,
                            piece_index,
                            view_index,
                        )
                    )
            _save_preview(
                observations,
                puzzle_output_dir,
                puzzle_output_dir / "previews" / "observations.png",
                self.config.preview_max_side,
            )
            observation_puzzle_payload = {
                "schema_version": OBSERVATION_PUZZLE_SCHEMA_VERSION,
                "compiler_version": __version__,
                "puzzle_id": puzzle_id,
                "canonical_puzzle_fingerprint": canonical_puzzle[
                    "puzzle_fingerprint"
                ],
                "canonical_puzzle_path": str(
                    canonical_puzzle_path.relative_to(
                        self.config.canonical_dataset_dir
                    ).as_posix()
                ),
                "split": canonical_puzzle.get("split"),
                "difficulty_tier": canonical_puzzle.get("difficulty_tier"),
                "observation_config": self.config.public_dict(),
                "piece_count": len(canonical_puzzle["pieces"]),
                "views_per_piece": self.config.views_per_piece,
                "observation_count": len(observations),
                "observations": observations,
                "validation": {
                    "valid": all(
                        item["validation"]["valid"] for item in observations
                    ),
                    "connected_observation_count": sum(
                        item["validation"]["connected_component_count"] == 1
                        for item in observations
                    ),
                },
            }
            puzzle_fingerprint_payload = deepcopy(observation_puzzle_payload)
            observation_puzzle_payload["observation_puzzle_fingerprint"] = (
                _sha256_bytes(_canonical_json(puzzle_fingerprint_payload))
            )
            _write_json(
                puzzle_output_dir / "observation_manifest.json",
                observation_puzzle_payload,
            )
            puzzle_output_dir.replace(final_puzzle_dir)
            total_observations += len(observations)
            puzzle_summaries.append(
                self._summary_from_existing_manifest(
                    observation_puzzle_payload, puzzle_summary
                )
            )
            print(
                f"[{puzzle_index + 1}/{len(canonical_puzzles)}] {puzzle_id}: "
                f"{len(observations)} observations",
                flush=True,
            )

        payload = {
            "schema_version": OBSERVATION_DATASET_SCHEMA_VERSION,
            "compiler_version": __version__,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "canonical_dataset_fingerprint": self.canonical_dataset[
                "dataset_fingerprint"
            ],
            "observation_config": self.config.public_dict(),
            "puzzle_count": len(puzzle_summaries),
            "observation_count": total_observations,
            "puzzles": puzzle_summaries,
        }
        fingerprint_payload = deepcopy(payload)
        fingerprint_payload.pop("created_utc", None)
        payload["observation_dataset_fingerprint"] = _sha256_bytes(
            _canonical_json(fingerprint_payload)
        )
        payload["elapsed_seconds"] = round(time.perf_counter() - started, 4)
        _write_json(
            self.config.output_dir / "observation_dataset_manifest.json", payload
        )
        return payload
