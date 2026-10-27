from __future__ import annotations

import hashlib
import json
import re
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from . import __version__
from .atomic_io import atomic_output_path, atomic_write_json
from .boundary import (
    _resample_closed_polyline,
    boundary_features,
    extract_closed_contour,
)
from .inference import ContourPredictorBase
from .onnx_runtime import file_sha256
from .pipeline import PipelineLock
from .prediction import render_role_overlay
from .training_data import (
    ROLE_NAMES,
    BoundaryWindowConfig,
    BoundaryWindowSampler,
    ObservationBoundaryEntry,
)


STAGE1_COMPATIBILITY_SCHEMA_VERSION = "stage1.segmentation-manifest-row.compat.v1"
EDGE_HYPOTHESES_SCHEMA_VERSION = "stage2.edge-hypotheses.v1"
EDGE_HYPOTHESES_MANIFEST_SCHEMA_VERSION = "stage2.edge-hypotheses-manifest.v1"
PRODUCTION_STATE_SCHEMA_VERSION = "stage2.production-inference-state.v2"

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_STAGE1_PROVENANCE_FIELDS = (
    "schema_version",
    "capture_id",
    "source_path",
    "source_sha256",
    "capture_type_hint",
    "original_size_hw",
    "bbox_xywh",
    "mask_path",
    "piece_path",
    "piece_mask_path",
    "overlay_path",
    "model_release",
    "model_version",
    "model_sha256",
    "dataset_fingerprint",
    "threshold",
    "postprocess_policy",
    "decision_source",
    "review_status",
    "accepted",
    "reason_codes",
    "manual_changed_fraction",
    "cache_reused",
    "created_utc",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    atomic_write_json(path, value)


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with atomic_output_path(path) as temporary:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)


def _resolve_input_path(root: Path, relative_value: Any, field: str) -> Path:
    if not isinstance(relative_value, str) or not relative_value.strip():
        raise ValueError(f"{field} must be a non-empty relative path")
    relative = Path(relative_value)
    if relative.is_absolute():
        raise ValueError(f"{field} must be relative to the Stage 1 output directory")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{field} escapes the Stage 1 output directory")
    if not resolved.is_file():
        raise FileNotFoundError(f"{field} does not exist: {resolved}")
    return resolved


def _positive_int_list(value: Any, length: int, field: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{field} must contain {length} integers")
    return [int(item) for item in value]


@dataclass(frozen=True, slots=True)
class Stage1Capture:
    capture_id: str
    row_index: int
    row: dict[str, Any]
    piece_path: Path
    piece_mask_path: Path
    piece_sha256: str
    piece_mask_sha256: str
    record_fingerprint: str
    original_size_hw: tuple[int, int]
    bbox_xywh: tuple[int, int, int, int]

    @property
    def piece_id(self) -> str:
        return self.capture_id


@dataclass(frozen=True, slots=True)
class Stage1InputSet:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    input_fingerprint: str
    captures: tuple[Stage1Capture, ...]
    total_row_count: int
    skipped_unaccepted_count: int


def _validate_capture_artifacts(
    *,
    capture_id: str,
    piece_path: Path,
    mask_path: Path,
    bbox_xywh: tuple[int, int, int, int],
) -> None:
    with Image.open(piece_path) as opened:
        piece_size = opened.size
    with Image.open(mask_path) as opened:
        mask_image = opened.convert("L")
        mask_size = mask_image.size
        mask_values = np.asarray(mask_image)
    if piece_size != mask_size:
        raise ValueError(
            f"{capture_id}: piece and crop mask dimensions differ: "
            f"{piece_size} versus {mask_size}"
        )
    if piece_size != (bbox_xywh[2], bbox_xywh[3]):
        raise ValueError(
            f"{capture_id}: bbox width/height do not match the tight crop: "
            f"{bbox_xywh[2:]} versus {piece_size}"
        )
    unique = np.unique(mask_values)
    if not set(int(value) for value in unique).issubset({0, 255}):
        raise ValueError(f"{capture_id}: piece_mask_path must be a binary 0/255 mask")
    foreground = mask_values >= 128
    if not foreground.any():
        raise ValueError(f"{capture_id}: crop mask is empty")
    _, component_count = ndimage.label(foreground)
    if component_count != 1:
        raise ValueError(
            f"{capture_id}: crop mask must contain one connected component; "
            f"found {component_count}"
        )


def load_stage1_output(
    input_path: Path,
    *,
    include_unaccepted: bool = False,
    maximum_pieces: int | None = None,
) -> Stage1InputSet:
    """Load and semantically validate the frozen Stage 1 JSONL handoff."""

    supplied = Path(input_path)
    if supplied.is_dir():
        root = supplied.resolve()
        manifest_path = root / "segmentation_manifest.jsonl"
    else:
        manifest_path = supplied.resolve()
        root = manifest_path.parent
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Stage 1 segmentation manifest not found: {manifest_path}"
        )
    if maximum_pieces is not None and maximum_pieces < 1:
        raise ValueError("maximum_pieces must be positive")

    captures: list[Stage1Capture] = []
    capture_ids: set[str] = set()
    total_rows = 0
    skipped = 0
    with manifest_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            total_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in {manifest_path.name} line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"manifest line {line_number} is not a JSON object")
            capture_id = row.get("capture_id")
            if not isinstance(capture_id, str) or not _SAFE_IDENTIFIER.fullmatch(
                capture_id
            ):
                raise ValueError(
                    f"manifest line {line_number} has unsafe capture_id {capture_id!r}"
                )
            if capture_id in capture_ids:
                raise ValueError(f"duplicate capture_id in Stage 1 manifest: {capture_id}")
            capture_ids.add(capture_id)
            if not isinstance(row.get("schema_version"), str):
                raise ValueError(f"{capture_id}: schema_version must be a string")
            if not isinstance(row.get("accepted"), bool):
                raise ValueError(f"{capture_id}: accepted must be boolean")
            if not row["accepted"] and not include_unaccepted:
                skipped += 1
                continue

            original_hw = _positive_int_list(
                row.get("original_size_hw"), 2, f"{capture_id}.original_size_hw"
            )
            bbox = _positive_int_list(
                row.get("bbox_xywh"), 4, f"{capture_id}.bbox_xywh"
            )
            original_height, original_width = original_hw
            x, y, width, height = bbox
            if min(original_height, original_width, width, height) <= 0 or min(x, y) < 0:
                raise ValueError(f"{capture_id}: original size and bbox are invalid")
            if x + width > original_width or y + height > original_height:
                raise ValueError(f"{capture_id}: bbox lies outside original_size_hw")

            piece_path = _resolve_input_path(root, row.get("piece_path"), "piece_path")
            piece_mask_path = _resolve_input_path(
                root, row.get("piece_mask_path"), "piece_mask_path"
            )
            _validate_capture_artifacts(
                capture_id=capture_id,
                piece_path=piece_path,
                mask_path=piece_mask_path,
                bbox_xywh=(x, y, width, height),
            )
            piece_sha = _sha256_file(piece_path)
            mask_sha = _sha256_file(piece_mask_path)
            record_fingerprint = _sha256_bytes(
                _canonical_json(
                    {
                        "row": row,
                        "piece_sha256": piece_sha,
                        "piece_mask_sha256": mask_sha,
                    }
                )
            )
            captures.append(
                Stage1Capture(
                    capture_id=capture_id,
                    row_index=line_number,
                    row=row,
                    piece_path=piece_path,
                    piece_mask_path=piece_mask_path,
                    piece_sha256=piece_sha,
                    piece_mask_sha256=mask_sha,
                    record_fingerprint=record_fingerprint,
                    original_size_hw=(original_height, original_width),
                    bbox_xywh=(x, y, width, height),
                )
            )
            if maximum_pieces is not None and len(captures) >= maximum_pieces:
                break

    if not captures:
        raise ValueError("Stage 1 manifest contains no selected pieces")
    manifest_sha = _sha256_file(manifest_path)
    input_fingerprint = _sha256_bytes(
        _canonical_json(
            {
                "manifest_sha256": manifest_sha,
                "include_unaccepted": include_unaccepted,
                "maximum_pieces": maximum_pieces,
                "capture_fingerprints": [
                    capture.record_fingerprint for capture in captures
                ],
            }
        )
    )
    return Stage1InputSet(
        root=root,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        input_fingerprint=input_fingerprint,
        captures=tuple(captures),
        total_row_count=total_rows,
        skipped_unaccepted_count=skipped,
    )


@dataclass(frozen=True, slots=True)
class BoundaryGeometryConfig:
    boundary_spacing_px: float = 2.0
    minimum_boundary_points: int = 64
    maximum_boundary_points: int = 4096
    tangent_smoothing_sigma_samples: float = 1.5
    max_inward_thickness_px: int = 96

    def validate(self) -> None:
        if not 0.25 <= self.boundary_spacing_px <= 20.0:
            raise ValueError("boundary_spacing_px must be in [0.25, 20]")
        if self.minimum_boundary_points < 16:
            raise ValueError("minimum_boundary_points must be at least 16")
        if self.maximum_boundary_points < self.minimum_boundary_points:
            raise ValueError("maximum_boundary_points must be >= minimum_boundary_points")
        if not 0.0 <= self.tangent_smoothing_sigma_samples <= 20.0:
            raise ValueError("tangent_smoothing_sigma_samples must be in [0, 20]")
        if self.max_inward_thickness_px < 1:
            raise ValueError("max_inward_thickness_px must be positive")


def compile_observed_boundary(
    mask: np.ndarray,
    config: BoundaryGeometryConfig | None = None,
) -> dict[str, np.ndarray]:
    """Compile truth-free contour geometry using the frozen training recipe."""

    geometry_config = config or BoundaryGeometryConfig()
    geometry_config.validate()
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not binary.any():
        raise ValueError("production mask must be a non-empty 2D array")
    contour = extract_closed_contour(binary)
    points, perimeter, spacing = _resample_closed_polyline(
        contour,
        geometry_config.boundary_spacing_px,
        geometry_config.minimum_boundary_points,
        geometry_config.maximum_boundary_points,
    )
    features = boundary_features(
        binary,
        points,
        spacing,
        smoothing_sigma_samples=geometry_config.tangent_smoothing_sigma_samples,
        max_inward_thickness_px=geometry_config.max_inward_thickness_px,
    )
    count = len(points)
    # The inference sampler shares the training window implementation. Targets are
    # placeholders only and are never provided to the model or exported.
    return {
        "points_xy": points.astype(np.float32),
        "arc_length_px": np.arange(count, dtype=np.float32) * spacing,
        "perimeter_px": np.asarray([perimeter], dtype=np.float32),
        "actual_spacing_px": np.asarray([spacing], dtype=np.float32),
        **features,
        "seam_target": np.zeros(count, dtype=np.float32),
        "page_border_target": np.zeros(count, dtype=np.float32),
        "uncertainty_target": np.ones(count, dtype=np.float32),
        "label_confidence": np.zeros(count, dtype=np.float32),
        "nearest_truth_distance_px": np.full(count, np.inf, dtype=np.float32),
    }


class ProductionBoundarySampler(BoundaryWindowSampler):
    """Truth-free adapter that reuses the exact trained window representation."""

    def __init__(
        self,
        config: BoundaryWindowConfig,
        geometry_config: BoundaryGeometryConfig | None = None,
    ):
        config.validate()
        self.config = config
        self.geometry_config = geometry_config or BoundaryGeometryConfig()
        self.geometry_config.validate()
        self.index = SimpleNamespace(observation_dataset_dir=Path("/"))
        self._production_arrays: dict[str, dict[str, np.ndarray]] = {}
        self._puzzle_cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self._image_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._transition_cache: OrderedDict[tuple[str, int], np.ndarray] = OrderedDict()
        self._global_context_cache: OrderedDict[
            tuple[str, int, int, int], tuple[np.ndarray, np.ndarray]
        ] = OrderedDict()

    def prepare_capture(
        self, capture: Stage1Capture
    ) -> tuple[ObservationBoundaryEntry, dict[str, np.ndarray]]:
        with Image.open(capture.piece_mask_path) as opened:
            mask = np.asarray(opened.convert("L")) >= 128
        arrays = compile_observed_boundary(mask, self.geometry_config)
        key = f"production:{capture.record_fingerprint}"
        self._production_arrays[key] = arrays
        entry = ObservationBoundaryEntry(
            puzzle_id="stage1_handoff",
            split="production",
            difficulty_tier="unknown",
            boundary_manifest_path="",
            boundaries_path=key,
            observation_index=0,
            observation_id=capture.capture_id,
            piece_id=capture.piece_id,
            view_index=0,
            image_path=str(capture.piece_path),
            mask_path=str(capture.piece_mask_path),
            point_count=len(arrays["points_xy"]),
            boundary_offset_start=0,
            boundary_offset_end=len(arrays["points_xy"]),
            hard_role_counts={name: 0 for name in ROLE_NAMES},
            puzzle_piece_count=0,
        )
        return entry, arrays

    def release_capture(self, entry: ObservationBoundaryEntry) -> None:
        self._production_arrays.pop(entry.boundaries_path, None)
        self._global_context_cache.clear()

    def _load_puzzle(self, entry: ObservationBoundaryEntry) -> dict[str, np.ndarray]:
        try:
            return self._production_arrays[entry.boundaries_path]
        except KeyError as exc:
            raise RuntimeError(
                f"production arrays are not prepared for {entry.observation_id}"
            ) from exc

    def _load_image(self, relative_path: str, *, mask: bool) -> np.ndarray:
        key = f"{'mask' if mask else 'image'}:{relative_path}"
        if key in self._image_cache:
            image = self._image_cache.pop(key)
            self._image_cache[key] = image
            return image
        path = Path(relative_path)
        with Image.open(path) as opened:
            if mask:
                image = np.asarray(opened.convert("L"), dtype=np.float32) / 255.0
            else:
                rgba = np.asarray(opened.convert("RGBA"), dtype=np.float32) / 255.0
                image = rgba[..., :3] * rgba[..., 3:4]
        self._image_cache[key] = image
        while len(self._image_cache) > self.config.image_cache_size:
            self._image_cache.popitem(last=False)
        return image


@dataclass(slots=True)
class ProductionInferenceConfig:
    input_path: Path
    model_path: Path
    output_dir: Path
    release_version: str
    release_manifest_sha256: str
    source_checkpoint_sha256: str
    write_overlays: bool = True
    resume: bool = False
    include_unaccepted: bool = False
    maximum_pieces: int | None = None
    expected_model_sha256: str | None = None
    model_format: str = "onnx"
    runtime: str = "onnxruntime-cpu"

    def validate(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(f"production model not found: {self.model_path}")
        if not self.release_version:
            raise ValueError("release_version must not be empty")
        if self.model_format != "onnx" or self.runtime != "onnxruntime-cpu":
            raise ValueError("production inference must use ONNX Runtime on CPU")
        for value, field in (
            (self.source_checkpoint_sha256, "source_checkpoint_sha256"),
            (self.release_manifest_sha256, "release_manifest_sha256"),
            (self.expected_model_sha256, "expected_model_sha256"),
        ):
            if value is not None and (
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{field} must be a lowercase SHA-256 digest")
        if self.maximum_pieces is not None and self.maximum_pieces < 1:
            raise ValueError("maximum_pieces must be positive")
        input_root = (
            self.input_path.resolve()
            if self.input_path.is_dir()
            else self.input_path.resolve().parent
        )
        output = self.output_dir.resolve()
        if output == input_root or output.is_relative_to(input_root):
            raise ValueError(
                "Stage 2 output must be outside the immutable Stage 1 directory"
            )


class Stage2ProductionRunner:
    def __init__(
        self,
        *,
        config: ProductionInferenceConfig,
        inputs: Stage1InputSet,
        sampler: ProductionBoundarySampler,
        predictor: ContourPredictorBase,
    ):
        config.validate()
        self.config = config
        self.inputs = inputs
        self.sampler = sampler
        self.predictor = predictor
        self.model_sha = file_sha256(config.model_path)
        if (
            config.expected_model_sha256 is not None
            and self.model_sha != config.expected_model_sha256
        ):
            raise RuntimeError(
                "ONNX model SHA-256 does not match the frozen Stage 2 release"
            )
        self.configuration = {
            "software_version": __version__,
            "release_version": config.release_version,
            "release_manifest_sha256": config.release_manifest_sha256,
            "model_format": config.model_format,
            "runtime": config.runtime,
            "model_sha256": self.model_sha,
            "source_checkpoint_sha256": config.source_checkpoint_sha256,
            "stage1_input_fingerprint": inputs.input_fingerprint,
            "geometry_config": asdict(sampler.geometry_config),
            "window_config": asdict(sampler.config),
            "inference_config": asdict(predictor.config),
            "fragment_config": predictor.fragment_config.public_dict(),
            "write_overlays": config.write_overlays,
            "include_unaccepted": config.include_unaccepted,
            "maximum_pieces": config.maximum_pieces,
        }
        self.config_hash = _sha256_bytes(_canonical_json(self.configuration))
        self.state_path = config.output_dir / "inference_state.json"
        self.manifest_path = config.output_dir / "edge_hypotheses_manifest.json"

    def _paths(self, capture: Stage1Capture) -> tuple[Path, Path, Path]:
        return (
            self.config.output_dir / "pieces" / f"{capture.capture_id}.npz",
            self.config.output_dir / "pieces" / f"{capture.capture_id}.json",
            self.config.output_dir / "overlays" / f"{capture.capture_id}.png",
        )

    def _prepare_output(self) -> dict[str, Any]:
        output = self.config.output_dir
        if output.exists() and any(output.iterdir()) and not self.config.resume:
            raise FileExistsError(
                f"Stage 2 output is not empty: {output}; pass --resume or choose "
                "a new directory"
            )
        output.mkdir(parents=True, exist_ok=True)
        if self.state_path.is_file():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not self.config.resume:
                raise FileExistsError("inference state exists; pass --resume")
            if state.get("schema_version") != PRODUCTION_STATE_SCHEMA_VERSION:
                raise RuntimeError("production inference state schema is incompatible")
            if state.get("config_hash") != self.config_hash:
                raise RuntimeError("production configuration changed; unsafe resume refused")
            return state
        if self.config.resume:
            unexpected = [
                path.name
                for path in output.iterdir()
                if path.name != "inference.lock"
            ]
            if unexpected:
                raise RuntimeError(
                    "resume requested without inference_state.json in a non-empty "
                    f"directory: {unexpected}"
                )
        state = {
            "schema_version": PRODUCTION_STATE_SCHEMA_VERSION,
            "created_utc": _utc_now(),
            "updated_utc": _utc_now(),
            "status": "running",
            "config_hash": self.config_hash,
            "completed_piece_count": 0,
        }
        _write_json_atomic(self.state_path, state)
        return state

    def _is_committed(self, capture: Stage1Capture) -> bool:
        npz_path, json_path, overlay_path = self._paths(capture)
        if not npz_path.is_file() or not json_path.is_file():
            return False
        if self.config.write_overlays and not overlay_path.is_file():
            return False
        try:
            record = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not (
            record.get("schema_version") == EDGE_HYPOTHESES_SCHEMA_VERSION
            and record.get("config_hash") == self.config_hash
            and record.get("stage1_record_fingerprint")
            == capture.record_fingerprint
            and record.get("model_sha256") == self.model_sha
            and record.get("release_manifest_sha256")
            == self.config.release_manifest_sha256
            and record.get("source_checkpoint_sha256")
            == self.config.source_checkpoint_sha256
            and record.get("arrays_path")
            == npz_path.relative_to(self.config.output_dir).as_posix()
        ):
            return False
        try:
            if record.get("arrays_sha256") != _sha256_file(npz_path):
                return False
            with np.load(npz_path, allow_pickle=False) as arrays:
                point_count = int(record.get("point_count", -1))
                if arrays["points_xy_crop"].shape != (point_count, 2):
                    return False
                if arrays["role_probabilities"].shape != (point_count, 3):
                    return False
                if arrays["transition_probabilities"].shape != (point_count,):
                    return False
        except (OSError, KeyError, ValueError):
            return False
        if self.config.write_overlays:
            expected_overlay = overlay_path.relative_to(
                self.config.output_dir
            ).as_posix()
            if record.get("overlay_path") != expected_overlay:
                return False
            if record.get("overlay_sha256") != _sha256_file(overlay_path):
                return False
        return True

    @staticmethod
    def _stage1_provenance(capture: Stage1Capture) -> dict[str, Any]:
        return {
            key: capture.row[key]
            for key in _STAGE1_PROVENANCE_FIELDS
            if key in capture.row
        }

    def _candidate_records(
        self,
        candidates: list[dict[str, Any]],
        points_crop: np.ndarray,
        points_source: np.ndarray,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for position, candidate in enumerate(candidates, start=1):
            item = dict(candidate)
            start = int(item["start_index"])
            end = int(item["end_index"])
            item.update(
                {
                    "candidate_id": f"candidate_{position:04d}",
                    "start_xy_crop": points_crop[start].astype(float).tolist(),
                    "end_xy_crop": points_crop[end].astype(float).tolist(),
                    "start_xy_source": points_source[start].astype(float).tolist(),
                    "end_xy_source": points_source[end].astype(float).tolist(),
                }
            )
            result.append(item)
        return result

    def run(self) -> dict[str, Any]:
        state = self._prepare_output()
        captures = self.inputs.captures
        records: list[dict[str, Any]] = []
        reused = 0
        with PipelineLock(self.config.output_dir / "inference.lock"):
            started = time.perf_counter()
            for position, capture in enumerate(captures, start=1):
                npz_path, json_path, overlay_path = self._paths(capture)
                if self._is_committed(capture):
                    records.append(json.loads(json_path.read_text(encoding="utf-8")))
                    reused += 1
                    continue

                entry, boundary = self.sampler.prepare_capture(capture)
                try:
                    prediction = self.predictor.predict(entry)
                    points_crop = boundary["points_xy"].astype(np.float32)
                    x, y, _, _ = capture.bbox_xywh
                    points_source = points_crop + np.asarray([x, y], dtype=np.float32)
                    arrays = {
                        "points_xy_crop": points_crop,
                        "points_xy_source": points_source,
                        "arc_length_px": boundary["arc_length_px"].astype(np.float32),
                        "tangent_xy": boundary["tangent_xy"].astype(np.float32),
                        "inward_normal_xy": boundary["inward_normal_xy"].astype(
                            np.float32
                        ),
                        "curvature_per_px": boundary["curvature_per_px"].astype(
                            np.float32
                        ),
                        "corner_strength": boundary["corner_strength"].astype(
                            np.float32
                        ),
                        "inward_thickness_px": boundary[
                            "inward_thickness_px"
                        ].astype(np.float32),
                        "role_logits": prediction["role_logits"].astype(np.float32),
                        "role_probabilities": prediction["role_probabilities"].astype(
                            np.float32
                        ),
                        "transition_logits": prediction["transition_logits"].astype(
                            np.float32
                        ),
                        "transition_probabilities": prediction[
                            "transition_probabilities"
                        ].astype(np.float32),
                    }
                    _write_npz_atomic(npz_path, arrays)
                    arrays_sha = _sha256_file(npz_path)
                    overlay_sha: str | None = None
                    if self.config.write_overlays:
                        render_role_overlay(
                            capture.piece_path,
                            points_crop,
                            prediction["role_probabilities"],
                            overlay_path,
                        )
                        overlay_sha = _sha256_file(overlay_path)
                    candidates = self._candidate_records(
                        prediction["candidate_fragments"],
                        points_crop,
                        points_source,
                    )
                    crop_to_source = [
                        [1.0, 0.0, float(x)],
                        [0.0, 1.0, float(y)],
                        [0.0, 0.0, 1.0],
                    ]
                    record = {
                        "schema_version": EDGE_HYPOTHESES_SCHEMA_VERSION,
                        "created_utc": _utc_now(),
                        "software_version": __version__,
                        "config_hash": self.config_hash,
                        "release_version": self.config.release_version,
                        "release_manifest_sha256": (
                            self.config.release_manifest_sha256
                        ),
                        "model_format": self.config.model_format,
                        "runtime": self.config.runtime,
                        "model_sha256": self.model_sha,
                        "source_checkpoint_sha256": (
                            self.config.source_checkpoint_sha256
                        ),
                        "stage1_input_fingerprint": self.inputs.input_fingerprint,
                        "stage1_record_fingerprint": capture.record_fingerprint,
                        "capture_id": capture.capture_id,
                        "piece_id": capture.piece_id,
                        "stage1_provenance": self._stage1_provenance(capture),
                        "piece_sha256": capture.piece_sha256,
                        "piece_mask_sha256": capture.piece_mask_sha256,
                        "coordinate_frames": {
                            "piece_crop": {
                                "origin": "top_left",
                                "x_axis": "right",
                                "y_axis": "down",
                                "units": "pixels",
                            },
                            "source_capture": {
                                "origin": "top_left",
                                "x_axis": "right",
                                "y_axis": "down",
                                "units": "pixels",
                                "size_hw": list(capture.original_size_hw),
                            },
                            "crop_to_source_homography": crop_to_source,
                        },
                        "point_order": "clockwise_circular",
                        "point_count": len(points_crop),
                        "perimeter_px": float(boundary["perimeter_px"][0]),
                        "actual_spacing_px": float(boundary["actual_spacing_px"][0]),
                        "arrays_path": npz_path.relative_to(
                            self.config.output_dir
                        ).as_posix(),
                        "arrays_sha256": arrays_sha,
                        "overlay_path": overlay_path.relative_to(
                            self.config.output_dir
                        ).as_posix()
                        if self.config.write_overlays
                        else None,
                        "overlay_sha256": overlay_sha,
                        "role_names": list(ROLE_NAMES),
                        "candidate_count": len(candidates),
                        "candidate_fragments": candidates,
                    }
                    # The JSON record is the atomic commit marker and is written last.
                    _write_json_atomic(json_path, record)
                    records.append(record)
                finally:
                    self.sampler.release_capture(entry)

                state.update(
                    {
                        "updated_utc": _utc_now(),
                        "status": "running",
                        "completed_piece_count": position,
                        "current_capture_id": capture.capture_id,
                    }
                )
                _write_json_atomic(self.state_path, state)
                elapsed = time.perf_counter() - started
                processed = max(position - reused, 1)
                rate = processed / max(elapsed, 1e-9)
                remaining = (len(captures) - position) / max(rate, 1e-9)
                print(
                    f"[{position}/{len(captures)}] {capture.capture_id}, "
                    f"candidates={record['candidate_count']}, ETA {remaining:.1f}s",
                    flush=True,
                )

            piece_summaries = [
                {
                    "capture_id": record["capture_id"],
                    "piece_id": record["piece_id"],
                    "record_path": self._paths(
                        next(
                            capture
                            for capture in captures
                            if capture.capture_id == record["capture_id"]
                        )
                    )[1]
                    .relative_to(self.config.output_dir)
                    .as_posix(),
                    "arrays_path": record["arrays_path"],
                    "point_count": record["point_count"],
                    "candidate_count": record["candidate_count"],
                    "stage1_record_fingerprint": record[
                        "stage1_record_fingerprint"
                    ],
                }
                for record in records
            ]
            manifest = {
                "schema_version": EDGE_HYPOTHESES_MANIFEST_SCHEMA_VERSION,
                "created_utc": _utc_now(),
                "status": "complete",
                "software_version": __version__,
                "config_hash": self.config_hash,
                "release_version": self.config.release_version,
                "release_manifest_sha256": self.config.release_manifest_sha256,
                "model_format": self.config.model_format,
                "runtime": self.config.runtime,
                "model_path": str(self.config.model_path.resolve()),
                "model_sha256": self.model_sha,
                "source_checkpoint_sha256": (
                    self.config.source_checkpoint_sha256
                ),
                "stage1_manifest_path": str(self.inputs.manifest_path),
                "stage1_manifest_sha256": self.inputs.manifest_sha256,
                "stage1_input_fingerprint": self.inputs.input_fingerprint,
                "stage1_total_row_count": self.inputs.total_row_count,
                "stage1_skipped_unaccepted_count": (
                    self.inputs.skipped_unaccepted_count
                ),
                "piece_count": len(piece_summaries),
                "reused_piece_count": reused,
                "candidate_count": sum(
                    int(item["candidate_count"]) for item in piece_summaries
                ),
                "coordinate_contract": (
                    "arrays contain crop and original source-capture coordinates"
                ),
                "truth_free": True,
                "configuration": self.configuration,
                "pieces": piece_summaries,
                "elapsed_seconds": round(time.perf_counter() - started, 4),
            }
            _write_json_atomic(self.manifest_path, manifest)
            state.update(
                {
                    "updated_utc": _utc_now(),
                    "completed_utc": _utc_now(),
                    "status": "complete",
                    "completed_piece_count": len(captures),
                    "current_capture_id": None,
                    "manifest_path": "edge_hypotheses_manifest.json",
                }
            )
            _write_json_atomic(self.state_path, state)
            return manifest

